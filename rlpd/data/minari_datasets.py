import copy
from typing import Iterable

import gymnasium as gym
import minari
import numpy as np

from rlpd.data.dataset import Dataset
from rlpd.wrappers.single_precision import _convert_space


def _flatten_batch(space: gym.Space, x) -> np.ndarray:
    """Batched gym.spaces.flatten: maps (N, ...) data to (N, flat_dim)."""
    if isinstance(space, gym.spaces.Box):
        return np.asarray(x).reshape(len(x), -1)
    elif isinstance(space, gym.spaces.Dict):
        return np.concatenate(
            [_flatten_batch(s, x[k]) for k, s in space.spaces.items()], axis=-1
        )
    raise NotImplementedError(f"Unsupported observation space: {space}")


def episodes_to_transitions(
    episodes: Iterable[minari.EpisodeData], observation_space: gym.Space
) -> dict:
    """Converts Minari episodes into flat (s, a, r, s') transition arrays.

    Dict observations are flattened in the same key order that wrap_gym's
    SinglePrecision + FlattenObservation wrappers use, so offline and online
    observations line up.
    """
    flatten_space = _convert_space(copy.deepcopy(observation_space))
    columns = {
        k: []
        for k in (
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "terminals",
            "dones",
        )
    }
    for episode in episodes:
        # Minari stores T + 1 observations per episode, including the final one.
        observations = _flatten_batch(flatten_space, episode.observations)
        num_steps = len(episode.rewards)
        dones = np.zeros(num_steps, dtype=bool)
        dones[-1] = True

        columns["observations"].append(observations[:-1])
        columns["next_observations"].append(observations[1:])
        columns["actions"].append(episode.actions)
        columns["rewards"].append(episode.rewards)
        columns["terminals"].append(episode.terminations)
        columns["dones"].append(dones)

    return {k: np.concatenate(v) for k, v in columns.items()}


class MinariDataset(Dataset):
    """Offline transitions from a Minari dataset (the maintained D4RL successor)."""

    def __init__(
        self,
        dataset: minari.MinariDataset,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ):
        dataset_dict = episodes_to_transitions(
            dataset.iterate_episodes(), dataset.observation_space
        )

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = dataset_dict.pop("dones")
        dataset_dict["masks"] = 1.0 - dataset_dict.pop("terminals")

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones

        super().__init__(dataset_dict)
