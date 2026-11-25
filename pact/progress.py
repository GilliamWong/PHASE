"""Demonstration supervision and a low-dimensional PACT progress model."""

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rlpd.env import OBS_KEYS, flatten_observation


def load_successful_demo_observations(path, demo_names):
    """Yield the name and observation sequence of each successful demo."""
    with h5py.File(path, "r") as file:
        for name in demo_names:
            demo = file["data"][name]
            success = np.flatnonzero(demo["rewards"][:] > 0)
            if success.size:
                stop = int(success[0]) + 1
                yield name, flatten_observation({key: demo["obs"][key][:stop] for key in OBS_KEYS})


def demonstration_targets(length: int, discount: float = 0.98):
    """Return t/T labels and returns for a unit final reward."""
    if length < 1:
        raise ValueError("a demonstration must contain at least one step")
    steps = torch.arange(length, dtype=torch.float32)
    return steps / length, discount ** (length - 1 - steps)


class ProgressModel(nn.Sequential):
    """Estimate scalar progress from PHASE's flat state vector."""

    def __init__(self, observation_dim: int):
        super().__init__(
            nn.Linear(observation_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, observation):
        return super().forward(observation).squeeze(-1)


def progress_loss(prediction, linear_targets, returns, terminal_mask):
    """Use time supervision everywhere and return supervision near success."""
    loss = F.mse_loss(prediction, linear_targets)
    if terminal_mask.any():
        loss = loss + 0.1 * F.mse_loss(prediction[terminal_mask], returns[terminal_mask])
    return loss
