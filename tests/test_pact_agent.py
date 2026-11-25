import torch

from pact.agent import PreferenceBatch, SACAgent, SACConfig
from rlif.data import Batch


def test_sac_updates_actor_and_twin_critics_and_respects_terminal_masks():
    torch.manual_seed(0)
    agent = SACAgent(3, 2, SACConfig(hidden_dims=(8,)), torch.device("cpu"))
    batch = Batch(torch.rand(4, 3), torch.rand(4, 2), torch.ones(4),
                  torch.zeros(4), torch.rand(4, 3))
    torch.testing.assert_close(agent.bellman_target(batch), batch.rewards)
    assert agent.critic(batch.observations, batch.actions).shape == (2, 4)
    modules = (agent.actor, *agent.critic.networks, agent.target_critic)
    before = [[p.detach().clone() for p in module.parameters()] for module in modules]
    metrics = agent.update(batch)
    assert all(torch.isfinite(value) for value in metrics.values())
    for module, previous in zip(modules, before):
        assert any(not torch.equal(p, old) for p, old in zip(module.parameters(), previous))
    assert all(p.grad is None for p in agent.target_critic.parameters())


def test_pact_correction_only_lowers_linked_preference_violations():
    agent = SACAgent(1, 1, SACConfig(hidden_dims=()), torch.device("cpu"))
    with torch.no_grad():
        for network in agent.target_critic.networks:
            network[0].weight.copy_(torch.tensor([[0.0, 1.0]]))
            network[0].bias.zero_()
    observations = torch.zeros(2, 1)
    preferences = PreferenceBatch(observations, torch.tensor([[-1.0], [1.0]]),
                                  torch.tensor([[1.0], [-1.0]]))
    corrected = agent.correct_targets(torch.ones(3, requires_grad=True), preferences,
                                      torch.tensor([0, 2]), torch.tensor([0.25, 1.0]))
    torch.testing.assert_close(corrected, torch.tensor([0.5, 1.0, 1.0]))
    assert not corrected.requires_grad
    agent.preference_loss(preferences).backward()
    assert agent.actor.mean.weight.grad is not None
    assert agent.actor.log_std.weight.grad is None
