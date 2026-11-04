"""Checks the RLPD agent's math and its CUDA-graph update."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from rlpd.agent import Actor, AgentConfig, EnsembleCritic, RLPDAgent, capture_cuda_graph
from rlpd.data import TransitionBuffer, sample_mixed

SMALL = AgentConfig(hidden_dims=(32, 32), num_qs=4, num_min_qs=2)


def test_log_prob_matches_change_of_variables_reference():
    # Catches a wrong Jacobian term or Gaussian density in the entropy terms.
    torch.manual_seed(0)
    actor = Actor(obs_dim=5, action_dim=3, hidden_dims=(16,)).double()
    observations = torch.randn(4, 5, dtype=torch.float64)

    with torch.no_grad():
        mean, std = actor(observations)
        actions, log_probs = actor.sample(observations)

    a = actions.numpy()
    expected = norm.logpdf(np.arctanh(a), mean.numpy(), std.numpy()).sum(-1)
    expected -= np.log1p(-(a**2)).sum(-1)
    np.testing.assert_allclose(log_probs.numpy(), expected, rtol=1e-6)


def test_log_prob_stays_finite_for_saturated_actions():
    # tanh(10) rounds to 1.0 in float32; an arctanh-based log_prob would be inf.
    actor = Actor(obs_dim=2, action_dim=2, hidden_dims=(8,))
    with torch.no_grad():
        actor.mean.weight.zero_()
        actor.mean.bias.fill_(10.0)
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(np.log(0.1))
        actions, log_probs = actor.sample(torch.randn(4, 2))
    assert (actions.abs() == 1).all() and torch.isfinite(log_probs).all()


def _buffer(obs_dim, action_dim, num_rows, masks, seed=0):
    rng = np.random.default_rng(seed)
    buffer = TransitionBuffer(obs_dim, action_dim, num_rows, torch.device("cpu"))
    buffer.add_many(
        rng.normal(size=(num_rows, obs_dim)),
        rng.uniform(-1, 1, (num_rows, action_dim)),
        rng.normal(size=num_rows),
        np.full(num_rows, masks),
        rng.normal(size=(num_rows, obs_dim)),
    )
    return buffer


def test_update_runs_and_is_finite():
    buffer = _buffer(5, 2, 64, masks=1.0)
    agent = RLPDAgent(5, 2, SMALL, torch.device("cpu"))

    info = agent.update(sample_mixed(None, buffer, 0, 16, num_minibatches=3))

    assert all(torch.isfinite(v) for v in info.values())
    action = agent.act(np.zeros(5, np.float32))
    assert action.shape == (2,) and np.all(np.abs(action) <= 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_parallel_streams_match_running_critics_one_by_one():
    # Catches missing stream synchronization: Q-values or gradients read
    # before a critic's stream has finished writing them.
    torch.manual_seed(0)
    critic = EnsembleCritic(5, 2, (64, 64), num_qs=4, layer_norm=True).cuda()
    obs, actions = torch.randn(32, 5, device="cuda"), torch.randn(32, 2, device="cuda")

    parallel_q = critic(obs, actions)
    parallel_q.square().sum().backward()
    # Dropping the first autograd graph lets the second backward pass build
    # fresh gradient-accumulation nodes on its own stream.
    parallel_q = parallel_q.detach()
    parallel_grads = [p.grad.clone() for p in critic.parameters()]
    critic.zero_grad()
    x = torch.cat([obs, actions], dim=-1)
    serial_q = torch.stack([member(x) for member in critic.critics]).squeeze(-1)
    serial_q.square().sum().backward()

    torch.testing.assert_close(parallel_q, serial_q, rtol=0, atol=0)
    for parallel_grad, p in zip(parallel_grads, critic.parameters()):
        torch.testing.assert_close(parallel_grad, p.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_graph_replay_matches_eager_updates():
    # Catches replays that do not really update: tensors captured by value,
    # Adam's step count frozen on the host, or the target critic not tracked.
    # With masks 0 and identical rows, critic updates are deterministic, so the
    # graph and the eager loop must reach the same critic.
    device = torch.device("cuda")
    obs_dim, action_dim = 5, 2
    row = _buffer(obs_dim, action_dim, 1, masks=0.0)
    buffer = TransitionBuffer(obs_dim, action_dim, 64, device)
    rows = row.rows.expand(64, -1).numpy()
    o, a = obs_dim, action_dim
    buffer.add_many(rows[:, :o], rows[:, o : o + a], rows[:, o + a], rows[:, o + a + 1], rows[:, o + a + 2 :])

    torch.manual_seed(0)
    eager = RLPDAgent(obs_dim, action_dim, SMALL, device)
    graphed = RLPDAgent(obs_dim, action_dim, SMALL, device)
    graphed.load_state_dict(eager.state_dict())

    def step(agent):
        return lambda: agent.update(sample_mixed(None, buffer, 0, 16, num_minibatches=4))

    initial_actor_weight = graphed.actor.mean.weight.detach().clone()
    replay = capture_cuda_graph(step(graphed), num_warmup=3)
    after_warmup = graphed.actor.mean.weight.detach().clone()
    for _ in range(5):
        replay()
    for _ in range(3 + 5):
        step(eager)()
    torch.cuda.synchronize()

    for name in ("critic", "target_critic"):
        for p_eager, p_graph in zip(getattr(eager, name).parameters(), getattr(graphed, name).parameters()):
            torch.testing.assert_close(p_graph, p_eager, rtol=1e-4, atol=1e-5)
    # The actor's updates are random, so only check that replays move it.
    assert not torch.allclose(after_warmup, initial_actor_weight)
    assert not torch.allclose(graphed.actor.mean.weight, after_warmup)
