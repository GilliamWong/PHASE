"""One-step updates, to catch breakage from Flax/JAX API changes."""

import gymnasium as gym
import numpy as np
from flax.core import frozen_dict

from rlpd.agents import SACLearner


def _batch(observations, next_observations, action_dim, batch_size, rng):
    return frozen_dict.freeze(
        dict(
            observations=observations,
            next_observations=next_observations,
            actions=rng.uniform(-1, 1, (batch_size, action_dim)).astype(np.float32),
            rewards=rng.normal(size=batch_size).astype(np.float32),
            masks=np.ones(batch_size, np.float32),
            dones=np.zeros(batch_size, bool),
        )
    )


def test_rlpd_sac_update_runs_and_is_finite():
    # Exercises the RLPD config path: REDQ subsampling and LayerNorm.
    obs_space = gym.spaces.Box(-1, 1, (5,), np.float32)
    act_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    agent = SACLearner.create(
        0, obs_space, act_space, hidden_dims=(32, 32), num_qs=4, num_min_qs=2,
        critic_layer_norm=True,
    )
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(16, 5)).astype(np.float32)
    batch = _batch(obs, obs, 2, 16, rng)

    agent, info = agent.update(batch, utd_ratio=2)

    assert all(np.isfinite(float(v)) for v in info.values())
    action, agent = agent.sample_actions(obs[0])
    assert action.shape == (2,) and np.all(np.abs(action) <= 1)
