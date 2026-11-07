"""Checks the RLPD agent's math, its Q-OIL options and its CUDA-graph update."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from rlpd.agent import Actor, AgentConfig, EnsembleCritic, RLPDAgent, capture_cuda_graph
from rlpd.data import TransitionBuffer, sample_mixed

SMALL = AgentConfig(hidden_dims=(32, 32), num_qs=4, num_min_qs=2)
SMALL_QOIL = AgentConfig(
    hidden_dims=(32, 32), num_qs=4, num_min_qs=2, bonus=0.5, optimistic_critic=True, bc_weight=0.1
)


def test_log_prob_matches_change_of_variables_reference():
    # Catches a wrong Jacobian term or Gaussian density in the entropy terms.
    torch.manual_seed(0)
    actor = Actor(obs_dim=5, action_dim=3, hidden_dims=(16,)).double()
    observations = torch.randn(4, 5, dtype=torch.float64)

    with torch.no_grad():
        mean, std = actor(observations)
        actions, log_probs = actor.sample(observations)

        given_log_probs = actor.log_prob(observations, actions, eps=1e-5)

    a = actions.numpy()
    expected = norm.logpdf(np.arctanh(a), mean.numpy(), std.numpy()).sum(-1)
    expected -= np.log1p(-(a**2)).sum(-1)
    np.testing.assert_allclose(log_probs.numpy(), expected, rtol=1e-6)
    # The BC term's density of given actions, where log_prob does not clamp.
    unclamped = (np.abs(a) < 1 - 1e-5).all(-1)
    assert unclamped.sum() >= 2
    np.testing.assert_allclose(given_log_probs.numpy()[unclamped], expected[unclamped], rtol=1e-6)


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


def _repeated_row_buffer(obs_dim, action_dim, num_rows, device, intervention):
    """num_rows copies of one random transition with mask 0 (terminal)."""
    rng = np.random.default_rng(0)
    obs, next_obs = rng.normal(size=obs_dim), rng.normal(size=obs_dim)
    action = rng.uniform(-1, 1, action_dim)
    buffer = TransitionBuffer(obs_dim, action_dim, num_rows, device)
    for _ in range(num_rows):
        buffer.add(obs, action, 1.0, 0.0, next_obs, intervention=intervention)
    return buffer


@pytest.mark.parametrize("config", [SMALL, SMALL_QOIL], ids=["rlpd", "qoil"])
def test_update_runs_and_is_finite(config):
    buffer = _buffer(5, 2, 64, masks=1.0)
    agent = RLPDAgent(5, 2, config, torch.device("cpu"))
    interventions = _repeated_row_buffer(5, 2, 16, torch.device("cpu"), intervention=True)

    info = agent.update(
        sample_mixed(None, buffer, 0, 16, num_minibatches=3),
        interventions.split(interventions.sample_rows((16,))),
    )

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


def test_decoupling_keeps_the_bonus_out_of_earlier_states():
    # Catches the optimistic critic bootstrapping from itself (or the bonus
    # reaching the task critic), which would let the bonus propagate back
    # from intervention states: Q-OIL's central design choice. Also checks
    # that the no-decoupling ablation does propagate it.
    # Chain s0 -> s1 -> end, reward 0; only the step from s1 is an
    # intervention. Decoupled, Q_opt(s0) stays at Q_TD(s0) = 0; coupled,
    # Q(s0) learns discount * bonus.
    s0, s1, action = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.zeros(1)
    buffer = TransitionBuffer(2, 1, 2, torch.device("cpu"))
    buffer.add(s0, action, 0.0, 1.0, s1, intervention=False)
    buffer.add(s1, action, 0.0, 0.0, s1, intervention=True)
    bonus, discount = 5.0, 0.99
    s0_and_s1 = torch.tensor(np.stack([s0, s1]), dtype=torch.float32)
    actions = torch.zeros(2, 1)

    results = {}
    for decoupled in (True, False):
        torch.manual_seed(0)
        config = AgentConfig(
            hidden_dims=(32, 32),
            num_qs=2,
            num_min_qs=2,
            critic_lr=3e-3,
            discount=discount,
            tau=0.05,
            bonus=bonus,
            optimistic_critic=decoupled,
        )
        agent = RLPDAgent(2, 1, config, torch.device("cpu"))
        for _ in range(300):
            agent.update(sample_mixed(None, buffer, 0, 8, num_minibatches=4))
        with torch.no_grad():
            task_q = agent.critic(s0_and_s1, actions).mean(0)
            policy_critic = agent.optimistic_critic if decoupled else agent.critic
            policy_q = policy_critic(s0_and_s1, actions).mean(0)
        results[decoupled] = (task_q, policy_q)

    task_q, optimistic_q = results[True]
    assert task_q.abs().max() < 0.25  # the task critic never sees the bonus
    assert abs(optimistic_q[1] - bonus) < 0.5  # the bonus at the intervention...
    assert abs(optimistic_q[0]) < 0.25  # ...and none at the state before it
    coupled_q = results[False][1]
    assert abs(coupled_q[0] - discount * bonus) < 0.5


def test_actor_gradient_norms_measure_each_term_without_updating():
    # Catches the diagnostic changing weights, missing a term, or giving the
    # BC term weight before any intervention exists (D_int sampled empty).
    agent = RLPDAgent(5, 2, SMALL_QOIL, torch.device("cpu"))
    observations = torch.randn(16, 5)
    filled = _repeated_row_buffer(5, 2, 16, torch.device("cpu"), intervention=True)
    empty = TransitionBuffer(5, 2, 16, torch.device("cpu"))
    weights = {k: v.clone() for k, v in agent.state_dict().items()}

    norms = agent.actor_gradient_norms(observations, filled.split(filled.sample_rows((16,))))
    empty_norms = agent.actor_gradient_norms(observations, empty.split(empty.sample_rows((16,))))

    assert set(norms) == {"actor_grad_norm_q", "actor_grad_norm_entropy", "actor_grad_norm_bc"}
    assert all(torch.isfinite(v) and v > 0 for v in norms.values())
    assert empty_norms["actor_grad_norm_bc"] == 0
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_mse_bc_fits_the_mean_action_and_leaves_the_std_alone():
    # Catches the MSE form computing a different loss, or still training the
    # std, which is what lets the log_prob form grow without bound.
    config = AgentConfig(hidden_dims=(32, 32), num_qs=4, bc_weight=0.1, bc_loss="mse")
    agent = RLPDAgent(5, 2, config, torch.device("cpu"))
    interventions = _repeated_row_buffer(5, 2, 16, torch.device("cpu"), intervention=True)
    batch = interventions.split(interventions.sample_rows((16,)))

    terms, _ = agent._actor_loss_terms(torch.randn(16, 5), batch)
    std_grads = torch.autograd.grad(terms["bc"], list(agent.actor.log_std.parameters()), allow_unused=True)

    with torch.no_grad():
        mean, _ = agent.actor(batch.observations)
    expected = 0.1 * ((np.tanh(mean.numpy()) - batch.actions.numpy()) ** 2).sum(-1).mean()
    np.testing.assert_allclose(terms["bc"].item(), expected, rtol=1e-5)
    assert all(g is None or (g == 0).all() for g in std_grads)
    # The gradient-norm diagnostic must cope with the unused std weights.
    assert agent.actor_gradient_norms(torch.randn(16, 5), batch)["actor_grad_norm_bc"] > 0


def test_bc_action_eps_below_float32_resolution_is_rejected():
    # Catches an eps whose 1 - eps rounds to 1 in float32, which would make
    # BC's log-density infinite and the actor NaN partway through a run.
    with pytest.raises(ValueError, match="bc_action_eps"):
        RLPDAgent(5, 2, AgentConfig(bc_action_eps=1e-9), torch.device("cpu"))


def test_checkpoint_round_trip_rebuilds_the_agent(tmp_path):
    # Catches checkpoints missing what a corrector needs to be rebuilt
    # (config, dimensions, the optimistic critic's weights).
    agent = RLPDAgent(5, 2, SMALL_QOIL, torch.device("cpu"))
    path = tmp_path / "step_0000000.pt"
    torch.save(agent.checkpoint(), path)

    loaded = RLPDAgent.from_checkpoint(str(path), torch.device("cpu"))

    observation = np.random.default_rng(0).normal(size=5).astype(np.float32)
    np.testing.assert_array_equal(
        loaded.act(observation, deterministic=True), agent.act(observation, deterministic=True)
    )
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("config", [SMALL, SMALL_QOIL], ids=["rlpd", "qoil"])
def test_cuda_graph_replay_matches_eager_updates(config):
    # Catches replays that do not really update: tensors captured by value,
    # Adam's step count frozen on the host, or the target critic not tracked.
    # With masks 0 and identical rows, critic updates are deterministic, so the
    # graph and the eager loop must reach the same critics.
    device = torch.device("cuda")
    obs_dim, action_dim = 5, 2
    # Every row an intervention, so Q-OIL's bonus and BC term are exercised.
    buffer = _repeated_row_buffer(obs_dim, action_dim, 64, device, intervention=True)

    torch.manual_seed(0)
    eager = RLPDAgent(obs_dim, action_dim, config, device)
    graphed = RLPDAgent(obs_dim, action_dim, config, device)
    graphed.load_state_dict(eager.state_dict())

    def step(agent):
        return lambda: agent.update(
            sample_mixed(None, buffer, 0, 16, num_minibatches=4),
            buffer.split(buffer.sample_rows((16,))),
        )

    initial_actor_weight = graphed.actor.mean.weight.detach().clone()
    replay = capture_cuda_graph(step(graphed), num_warmup=3)
    after_warmup = graphed.actor.mean.weight.detach().clone()
    for _ in range(5):
        replay()
    for _ in range(3 + 5):
        step(eager)()
    torch.cuda.synchronize()

    critics = ["critic", "target_critic"]
    if config.optimistic_critic:
        critics.append("optimistic_critic")
    for name in critics:
        for p_eager, p_graph in zip(getattr(eager, name).parameters(), getattr(graphed, name).parameters()):
            torch.testing.assert_close(p_graph, p_eager, rtol=1e-4, atol=1e-5)
    # The actor's updates are random, so only check that replays move it.
    assert not torch.allclose(after_warmup, initial_actor_weight)
    assert not torch.allclose(graphed.actor.mean.weight, after_warmup)


def test_bc_log_prob_cap_stops_the_pull_on_well_fit_actions():
    # Catches the cap clipping the wrong side (blocking BC on poorly fit
    # actions) or not cutting the gradient once log pi exceeds it.
    torch.manual_seed(0)
    agent = RLPDAgent(5, 2, AgentConfig(hidden_dims=(32, 32), num_qs=4, bc_weight=1.0), torch.device("cpu"))
    batch = _repeated_row_buffer(5, 2, 16, torch.device("cpu"), intervention=True)
    batch = batch.split(batch.sample_rows((16,)))
    with torch.no_grad():
        log_prob = agent.actor.log_prob(batch.observations, batch.actions, 1e-5)[0].item()

    def bc_grad(cap):
        agent.config.bc_log_prob_cap = cap
        terms, _ = agent._actor_loss_terms(torch.randn(16, 5), batch)
        return torch.autograd.grad(terms["bc"], agent.actor.mean.weight)[0]

    torch.testing.assert_close(bc_grad(log_prob + 1.0), bc_grad(None))  # below the cap: unchanged
    assert (bc_grad(log_prob - 1.0) == 0).all()  # above the cap: no pull
