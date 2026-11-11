"""Checks the pieces you write in rlpd/learn.py.

A test fails with NotImplementedError until the piece it checks is filled in.
Run one method's tests with e.g. `python -m pytest tests/test_learn.py -k dqn`.
"""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from rlpd.data import TransitionBuffer
from rlpd.learn import (
    AGENTS,
    Config,
    DeterministicPolicy,
    DiscreteQNetwork,
    GaussianPolicy,
    QNetwork,
    sample_batch,
)

OBS_DIM, ACTION_DIM = 5, 2
SMALL = Config(hidden_dims=(64, 64), batch_size=64, device="cpu", target_update_interval=100)
LEARNING_ALGOS = ["bc", "dqn", "ddpg", "td3", "sac"]


def test_deterministic_policy_outputs_actions_in_range():
    # Catches a missing squashing layer: large inputs must still give actions in [-1, 1].
    policy = DeterministicPolicy(OBS_DIM, ACTION_DIM, (16,))
    actions = policy(100 * torch.randn(8, OBS_DIM))
    assert actions.shape == (8, ACTION_DIM)
    assert (actions.abs() <= 1).all()


def test_gaussian_log_prob_matches_change_of_variables_reference():
    # Catches a wrong Gaussian density or a wrong tanh Jacobian term.
    torch.manual_seed(0)
    policy = GaussianPolicy(OBS_DIM, 3, (16,)).double()
    observations = torch.randn(4, OBS_DIM, dtype=torch.float64)

    with torch.no_grad():
        mean, std = policy(observations)
        actions, log_probs = policy.sample(observations)

    assert mean.shape == std.shape == actions.shape == (4, 3) and log_probs.shape == (4,)
    a = actions.numpy()
    expected = norm.logpdf(np.arctanh(a), mean.numpy(), std.numpy()).sum(-1)
    expected -= np.log1p(-(a**2)).sum(-1)
    np.testing.assert_allclose(log_probs.numpy(), expected, rtol=1e-6)


def test_gaussian_log_prob_stays_finite_for_saturated_actions():
    # tanh(10) rounds to 1.0 in float32, so log(1 - tanh(x)^2) computed
    # directly is log(0) = -inf.
    policy = GaussianPolicy(OBS_DIM, ACTION_DIM, (16,))
    policy.forward = lambda obs: (torch.full((len(obs), ACTION_DIM), 10.0), torch.full((len(obs), ACTION_DIM), 0.1))
    actions, log_probs = policy.sample(torch.randn(4, OBS_DIM))
    assert (actions.abs() == 1).all() and torch.isfinite(log_probs).all()


def test_gaussian_sample_is_reparameterized():
    # Catches sampling with .sample() or under no_grad, which leaves the
    # actor without gradients through the action.
    policy = GaussianPolicy(OBS_DIM, ACTION_DIM, (16,))
    actions, _ = policy.sample(torch.randn(4, OBS_DIM))
    actions.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())


def test_q_network_output_shapes():
    # Catches a trailing size-1 dimension, which broadcasts (batch,) targets
    # against (batch, 1) predictions into a (batch, batch) loss.
    q = QNetwork(OBS_DIM, ACTION_DIM, (16,))
    assert q(torch.randn(8, OBS_DIM), torch.rand(8, ACTION_DIM)).shape == (8,)
    discrete_q = DiscreteQNetwork(OBS_DIM, ACTION_DIM, 11, (16,))
    assert discrete_q(torch.randn(8, OBS_DIM)).shape == (8, ACTION_DIM, 11)


def _buffer(observations, actions, rewards, masks, next_observations):
    buffer = TransitionBuffer(OBS_DIM, ACTION_DIM, len(rewards), torch.device("cpu"))
    buffer.add_many(observations, actions, rewards, masks, next_observations)
    return buffer


def _random_buffer(num_rows=256, seed=0):
    rng = np.random.default_rng(seed)
    return _buffer(
        rng.normal(size=(num_rows, OBS_DIM)),
        rng.uniform(-1, 1, (num_rows, ACTION_DIM)),
        rng.normal(size=num_rows),
        rng.choice([0.0, 1.0], num_rows),
        rng.normal(size=(num_rows, OBS_DIM)),
    )


@pytest.mark.parametrize("algo", LEARNING_ALGOS)
def test_agent_acts_and_updates(algo):
    # Catches wrong action shapes or ranges, and updates that crash or
    # return non-finite metrics.
    torch.manual_seed(0)
    agent = AGENTS[algo](OBS_DIM, ACTION_DIM, SMALL, torch.device("cpu"))
    buffer = _random_buffer()

    for _ in range(3):
        metrics = agent.update(sample_batch(None, buffer, 0, SMALL.batch_size))
    assert metrics and all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())

    observation = np.random.default_rng(0).normal(size=OBS_DIM).astype(np.float32)
    for explore in (True, False):
        action = agent.act(observation, explore=explore)
        assert isinstance(action, np.ndarray) and action.shape == (ACTION_DIM,)
        assert np.all(np.abs(action) <= 1)


TARGET_ACTION = 0.6  # on DQN's 11-bin grid, so the best bin is exactly right


# DDPG and TD3 are left out: their deterministic actors often push tanh into
# saturation at +-1 before the critic has learned anything, and then take
# far longer than a test allows to come back, even when implemented
# correctly. Check them with a short Pendulum run instead (SETUP.md).
@pytest.mark.parametrize("algo", ["bc", "dqn", "sac"])
def test_agent_learns_a_one_step_task(algo):
    # Catches algorithms that run but do not learn. Every transition ends its
    # episode (mask 0) with reward -sum_d |a_d - 0.6| from one fixed state,
    # so the TD target is the reward alone and the best action is 0.6 in
    # every dimension. Value-based agents learn it from uniformly random
    # actions; BC from noisy copies of it.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    num_rows = 2048
    observations = np.tile(rng.normal(size=(1, OBS_DIM)), (num_rows, 1))
    if algo == "bc":
        actions = np.clip(TARGET_ACTION + 0.1 * rng.normal(size=(num_rows, ACTION_DIM)), -1, 1)
    else:
        actions = rng.uniform(-1, 1, (num_rows, ACTION_DIM))
    rewards = -np.abs(actions - TARGET_ACTION).sum(-1)
    buffer = _buffer(observations, actions, rewards, np.zeros(num_rows), observations)

    # A lower initial temperature, so SAC's entropy bonus does not dominate
    # this small reward within the test's few updates.
    config = Config(**{**SMALL.__dict__, "init_temperature": 0.05})
    agent = AGENTS[algo](OBS_DIM, ACTION_DIM, config, torch.device("cpu"))
    for _ in range(1500):
        agent.update(sample_batch(None, buffer, 0, config.batch_size))

    action = agent.act(observations[0].astype(np.float32), explore=False)
    np.testing.assert_allclose(action, TARGET_ACTION, atol=0.15)


def test_sample_batch_mixes_demos_and_online():
    # Catches the provided helper returning minibatch-shaped fields.
    demos, online = _random_buffer(seed=1), _random_buffer(seed=2)
    batch = sample_batch(demos, online, 3, 5)
    assert batch.observations.shape == (8, OBS_DIM) and batch.rewards.shape == (8,)
