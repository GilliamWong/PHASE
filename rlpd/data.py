"""Transition storage on the training device, and Robomimic demo loading."""

from typing import NamedTuple, Optional

import h5py
import numpy as np
import torch

from rlpd.env import OBS_KEYS, flatten_observation


class Batch(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    # 0 where the episode truly terminated (success), so the target does not
    # bootstrap; time-limit truncations keep 1.
    masks: torch.Tensor
    # 1 where a (synthetic) human chose the action, else 0.
    interventions: torch.Tensor
    next_observations: torch.Tensor


class TransitionBuffer:
    """A fixed-capacity ring buffer of transitions, kept on `device`.

    Each transition is one row [observation | action | reward | mask |
    intervention | next_observation], so sampling a batch is a single gather. The fill level
    is also kept on the device, so sampling never waits on the host and can be
    recorded into a CUDA graph.
    """

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.rows = torch.zeros((capacity, 2 * obs_dim + action_dim + 3), device=device)
        self._num_added = 0
        # float64 so that rand * size stays below size for any realistic capacity.
        self._size = torch.zeros((), dtype=torch.float64, device=device)

    def __len__(self) -> int:
        return min(self._num_added, self.capacity)

    def add(
        self,
        observation,
        action,
        reward: float,
        mask: float,
        next_observation,
        intervention: bool = False,
    ):
        row = np.concatenate(
            [observation, action, [reward, mask, float(intervention)], next_observation]
        )
        self.rows[self._num_added % self.capacity] = torch.from_numpy(row.astype(np.float32))
        self._num_added += 1
        self._size.fill_(len(self))

    def add_many(
        self, observations, actions, rewards, masks, next_observations, interventions: bool = False
    ):
        """Fills an empty buffer with arrays of transitions (one per row),
        such as demonstrations, all flagged as interventions or none."""
        assert self._num_added == 0 and len(rewards) <= self.capacity
        interventions = np.full_like(rewards, float(interventions))
        rows = np.concatenate(
            [
                observations,
                actions,
                rewards[:, None],
                masks[:, None],
                interventions[:, None],
                next_observations,
            ],
            axis=1,
        )
        self.rows[: len(rows)] = torch.from_numpy(rows.astype(np.float32))
        self._num_added = len(rows)
        self._size.fill_(len(self))

    def sample_rows(self, shape) -> torch.Tensor:
        """Uniformly samples filled rows with replacement; returns (*shape, width)."""
        indices = (torch.rand(shape, dtype=torch.float64, device=self.rows.device) * self._size).long()
        return self.rows[indices]

    def split(self, rows: torch.Tensor) -> Batch:
        o, a = self.obs_dim, self.action_dim
        return Batch(
            observations=rows[..., :o],
            actions=rows[..., o : o + a],
            rewards=rows[..., o + a],
            masks=rows[..., o + a + 1],
            interventions=rows[..., o + a + 2],
            next_observations=rows[..., o + a + 3 :],
        )


def sample_mixed(
    demos: Optional[TransitionBuffer],
    online: TransitionBuffer,
    demos_per_minibatch: int,
    online_per_minibatch: int,
    num_minibatches: int,
) -> Batch:
    """RLPD's symmetric sampling: every minibatch has the same demo/online mix.

    Fields have shape (num_minibatches, minibatch_size, ...); the agent runs
    one critic update per minibatch.
    """
    parts = []
    if demos_per_minibatch > 0:
        parts.append(demos.sample_rows((num_minibatches, demos_per_minibatch)))
    if online_per_minibatch > 0:
        parts.append(online.sample_rows((num_minibatches, online_per_minibatch)))
    return online.split(torch.cat(parts, dim=1))


def _demo_names(data_group: h5py.Group) -> list:
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[1]))


def load_demo_buffer(
    dataset_path: str,
    success_reward: float,
    device: torch.device,
    num_demos: Optional[int] = None,
    eps: float = 1e-5,
    as_interventions: bool = False,
    extra_capacity: int = 0,
) -> TransitionBuffer:
    """Robomimic demos as a TransitionBuffer of (s, a, r, s') transitions in
    RobomimicEnv's MDP: the first num_demos in the file's numeric order, or
    all of them when None. as_interventions flags them as interventions;
    extra_capacity leaves room for that many more transitions.

    RobomimicEnv ends an episode at the first success, so each demo is cut
    there: that step becomes terminal with reward success_reward, and the
    steps after it (the demo keeps recording while the task stays solved) are
    dropped. Demos that never succeed are kept whole, without a terminal step.
    """
    columns = {k: [] for k in ("observations", "actions", "rewards", "masks", "next_observations")}
    with h5py.File(dataset_path, "r") as f:
        data = f["data"]
        names = _demo_names(data)
        if num_demos is not None:
            if not 0 < num_demos <= len(names):
                raise ValueError(f"num_demos={num_demos}, but the dataset has {len(names)} demos")
            names = names[:num_demos]
        for name in names:
            demo = data[name]
            success_steps = np.flatnonzero(demo["rewards"][:] > 0)
            succeeded = len(success_steps) > 0
            num_steps = success_steps[0] + 1 if succeeded else len(demo["actions"])

            rewards = np.zeros(num_steps, dtype=np.float32)
            masks = np.ones(num_steps, dtype=np.float32)
            if succeeded:
                rewards[-1] = success_reward
                masks[-1] = 0.0

            columns["observations"].append(
                flatten_observation({k: demo["obs"][k][:num_steps] for k in OBS_KEYS})
            )
            columns["next_observations"].append(
                flatten_observation({k: demo["next_obs"][k][:num_steps] for k in OBS_KEYS})
            )
            columns["actions"].append(demo["actions"][:num_steps].astype(np.float32))
            columns["rewards"].append(rewards)
            columns["masks"].append(masks)

    t = {k: np.concatenate(v) for k, v in columns.items()}
    # Keeps demo actions strictly inside the tanh policy's support; the
    # gripper action is always exactly -1 or 1 in these demos.
    t["actions"] = np.clip(t["actions"], -1 + eps, 1 - eps)
    obs_dim, action_dim = t["observations"].shape[1], t["actions"].shape[1]
    buffer = TransitionBuffer(obs_dim, action_dim, len(t["rewards"]) + extra_capacity, device)
    buffer.add_many(**t, interventions=as_interventions)
    return buffer
