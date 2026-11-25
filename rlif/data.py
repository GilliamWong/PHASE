"""Independent transition storage and demonstration loading for RLIF."""

from typing import NamedTuple, Optional, Sequence

import h5py
import numpy as np
import torch

from rlpd.env import OBS_KEYS, flatten_observation


class Batch(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    # 0 = episode terminated (success), so the target does not
    # bootstrap; time-limit truncations still 1.
    masks: torch.Tensor
    next_observations: torch.Tensor


class TransitionBuffer:
    """A fixed-capacity ring buffer of transitions, kept on `device`.

    Each transition is one row [observation | action | reward | mask |
    next_observation], so sampling a batch is a single gather. The fill level
    is also kept on the device, so sampling never waits on the host and can be
    recorded into a CUDA graph.
    """

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.rows = torch.zeros((capacity, 2 * obs_dim + action_dim + 2), device=device)
        self._num_added = 0
        # float64 so that rand * size stays below size for any realistic capacity.
        self._size = torch.zeros((), dtype=torch.float64, device=device)

    def __len__(self) -> int:
        return min(self._num_added, self.capacity)

    def add(self, observation, action, reward: float, mask: float, next_observation):
        row = np.concatenate([observation, action, [reward, mask], next_observation])
        self.rows[self._num_added % self.capacity] = torch.from_numpy(row.astype(np.float32))
        self._num_added += 1
        self._size.fill_(len(self))

    def add_many(self, observations, actions, rewards, masks, next_observations):
        """Fills an empty buffer with arrays of transitions (one per row)."""
        assert self._num_added == 0 and len(rewards) <= self.capacity
        rows = np.concatenate(
            [observations, actions, rewards[:, None], masks[:, None], next_observations],
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
            next_observations=rows[..., o + a + 2 :],
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


def split_demo_names(
    dataset_path: str,
    train_count: int,
    validation_count: int,
    rlif_count: int,
    seed: int,
) -> dict[str, list[str]]:
    """Returns a deterministic, disjoint split of whole demonstrations."""
    with h5py.File(dataset_path, "r") as f:
        names = _demo_names(f["data"])
    requested = train_count + validation_count + rlif_count
    if min(train_count, validation_count, rlif_count) < 0 or requested > len(names):
        raise ValueError(
            f"requested split ({train_count}, {validation_count}, {rlif_count}) "
            f"from only {len(names)} demonstrations"
        )
    order = np.random.default_rng(seed).permutation(len(names))[:requested]
    shuffled = [names[index] for index in order]
    train_end = train_count
    validation_end = train_end + validation_count
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "rlif": shuffled[validation_end:],
    }


def load_bc_arrays(dataset_path: str, demo_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Loads flattened observations and actions from selected whole demos."""
    observations, actions = [], []
    with h5py.File(dataset_path, "r") as f:
        data = f["data"]
        for name in demo_names:
            demo = data[name]
            observations.append(flatten_observation({key: demo["obs"][key][:] for key in OBS_KEYS}))
            actions.append(demo["actions"][:].astype(np.float32))
    if not observations:
        raise ValueError("at least one demonstration is required")
    return np.concatenate(observations), np.concatenate(actions)


def load_demo_buffer(
    dataset_path: str,
    success_reward: float,
    device: torch.device,
    eps: float = 1e-5,
    demo_names: Optional[Sequence[str]] = None,
) -> TransitionBuffer:
    """Robomimic demos as a full TransitionBuffer of (s, a, r, s') transitions
    in RobomimicEnv's MDP.

    RobomimicEnv ends an episode at the first success, so each demo is cut
    there: that step becomes terminal with reward success_reward, and the
    steps after it (the demo keeps recording while the task stays solved) are
    dropped. Demos that never succeed are kept whole, without a terminal step.
    """
    columns = {k: [] for k in ("observations", "actions", "rewards", "masks", "next_observations")}
    with h5py.File(dataset_path, "r") as f:
        data = f["data"]
        for name in _demo_names(data) if demo_names is None else demo_names:
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
    buffer = TransitionBuffer(obs_dim, action_dim, len(t["rewards"]), device)
    buffer.add_many(**t)
    return buffer


def load_rlif_demo_buffer(
    dataset_path: str,
    device: torch.device,
    demo_names: Optional[Sequence[str]] = None,
) -> TransitionBuffer:
    """Loads initial RLIF data with zero intervention-feedback rewards.

    RLIF does not use the task reward.  The demonstrations only warm-start
    replay, so every stored reward is zero.  True terminal masks are retained.
    """
    return load_demo_buffer(
        dataset_path, success_reward=0.0, device=device, demo_names=demo_names
    )
