import h5py
import numpy as np

from rlpd.data.dataset import Dataset
from rlpd.envs.robomimic import OBS_KEYS, flatten_observation


def _demo_names(data_group: h5py.Group) -> list:
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[1]))


def load_demo_transitions(dataset_path: str, success_reward: float) -> dict:
    """Reads Robomimic demos as (s, a, r, s') transitions in RobomimicEnv's MDP.

    RobomimicEnv ends an episode at the first success, so each demo is cut
    there: that step becomes terminal with reward success_reward, and the
    steps after it (the demo keeps recording while the task stays solved) are
    dropped. Demos that never succeed are kept whole, without a terminal step.
    """
    columns = {
        k: []
        for k in ("observations", "next_observations", "actions", "rewards", "terminals", "dones")
    }
    with h5py.File(dataset_path, "r") as f:
        data = f["data"]
        for name in _demo_names(data):
            demo = data[name]
            success_steps = np.flatnonzero(demo["rewards"][:] > 0)
            succeeded = len(success_steps) > 0
            num_steps = success_steps[0] + 1 if succeeded else len(demo["actions"])

            rewards = np.zeros(num_steps, dtype=np.float32)
            terminals = np.zeros(num_steps, dtype=np.float32)
            dones = np.zeros(num_steps, dtype=bool)
            rewards[-1] = success_reward if succeeded else 0.0
            terminals[-1] = float(succeeded)
            dones[-1] = True

            columns["observations"].append(
                flatten_observation({k: demo["obs"][k][:num_steps] for k in OBS_KEYS})
            )
            columns["next_observations"].append(
                flatten_observation({k: demo["next_obs"][k][:num_steps] for k in OBS_KEYS})
            )
            columns["actions"].append(demo["actions"][:num_steps].astype(np.float32))
            columns["rewards"].append(rewards)
            columns["terminals"].append(terminals)
            columns["dones"].append(dones)

    return {k: np.concatenate(v) for k, v in columns.items()}


class RobomimicDataset(Dataset):
    """Robomimic demonstrations as offline transitions for RLPD."""

    def __init__(
        self,
        dataset_path: str,
        success_reward: float,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ):
        dataset_dict = load_demo_transitions(dataset_path, success_reward)

        if clip_to_eps:
            # Keeps demo actions strictly inside the tanh policy's support.
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dataset_dict["masks"] = 1.0 - dataset_dict.pop("terminals")

        super().__init__(dataset_dict)
