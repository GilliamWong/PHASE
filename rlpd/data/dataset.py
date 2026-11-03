from typing import Dict, Optional, Union

import jax
import numpy as np
from flax.core import frozen_dict
from gymnasium.utils import seeding

from rlpd.types import DataType

DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type.")
    return dataset_len


def _sample(
    dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray
) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


class Dataset(object):
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        self._np_random = None
        self._seed = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, self._seed = seeding.np_random(seed)
        return [self._seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(self, batch_size: int) -> frozen_dict.FrozenDict:
        """Samples batch_size transitions uniformly, with replacement."""
        indx = self.np_random.integers(len(self), size=batch_size)
        return frozen_dict.freeze(_sample(self.dataset_dict, indx))


def combine_batches(offline_batch, online_batch, num_minibatches: int):
    """Joins the batches so each UTD minibatch gets the same offline/online mix.

    SACLearner.update splits the batch into num_minibatches consecutive slices,
    so the two batches are merged slice by slice.
    """

    def merge(offline, online):
        offline = offline.reshape(num_minibatches, -1, *offline.shape[1:])
        online = online.reshape(num_minibatches, -1, *online.shape[1:])
        merged = np.concatenate([offline, online], axis=1)
        return merged.reshape(-1, *merged.shape[2:])

    return jax.tree_util.tree_map(merge, dict(offline_batch), dict(online_batch))
