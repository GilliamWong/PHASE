"""Low-dimensional SAC backbone for PACT."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rlif.agent import Actor
from rlif.data import Batch


@dataclass(frozen=True)
class SACConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    discount: float = 0.98
    tau: float = 0.005
    learning_rate: float = 3e-4
    preference_weight: float = 0.2
    preference_epsilon: float = 1e-3


@dataclass(frozen=True)
class PreferenceBatch:
    observations: torch.Tensor
    human_actions: torch.Tensor
    policy_actions: torch.Tensor


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.networks = nn.ModuleList()
        for _ in range(2):
            layers = []
            width = obs_dim + action_dim
            for hidden in hidden_dims:
                layers.extend((nn.Linear(width, hidden), nn.ReLU()))
                width = hidden
            self.networks.append(nn.Sequential(*layers, nn.Linear(width, 1)))

    def forward(self, observations, actions):
        inputs = torch.cat((observations, actions), dim=-1)
        return torch.stack([network(inputs).squeeze(-1) for network in self.networks])


class SACAgent(nn.Module):
    """Owns PACT's policy, twin critics, target critics, and entropy temperature."""

    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.actor = Actor(obs_dim, action_dim, config.hidden_dims)
        self.critic = TwinCritic(obs_dim, action_dim, config.hidden_dims)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.target_entropy = -action_dim / 2
        self.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=config.learning_rate)

    @torch.no_grad()
    def bellman_target(self, batch: Batch) -> torch.Tensor:
        next_actions, next_log_prob = self.actor.sample(batch.next_observations)
        next_value = self.target_critic(batch.next_observations, next_actions).amin(dim=0)
        next_value = next_value - self.log_temperature.exp() * next_log_prob
        return batch.rewards + self.config.discount * batch.masks * next_value

    @torch.no_grad()
    def correct_targets(self, target, preferences: PreferenceBatch, rows, weights):
        """Subtract a weighted target-critic violation from linked batch rows."""
        policy_value = self.target_critic(preferences.observations, preferences.policy_actions).amin(0)
        human_value = self.target_critic(preferences.observations, preferences.human_actions).amin(0)
        corrected = target.clone()
        corrected[rows] -= weights * (policy_value - human_value).clamp_min(0)
        return corrected

    def preference_loss(self, preferences: PreferenceBatch) -> torch.Tensor:
        action = self.actor.mode(preferences.observations)
        attraction = (action - preferences.human_actions).square().sum(dim=-1)
        repulsion = (action - preferences.policy_actions).square().sum(dim=-1)
        return (attraction + (repulsion + self.config.preference_epsilon).reciprocal()).mean()

    def update(
        self,
        batch: Batch,
        *,
        actor_preferences: PreferenceBatch | None = None,
        critic_preferences: tuple[PreferenceBatch, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Update from one flat batch with scalar rewards and bootstrap masks."""
        target = self.bellman_target(batch)
        if critic_preferences is not None:
            target = self.correct_targets(target, *critic_preferences)
        values = self.critic(batch.observations, batch.actions)
        critic_loss = F.mse_loss(values, target.expand_as(values))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        try:
            actions, log_prob = self.actor.sample(batch.observations)
            temperature = self.log_temperature.detach().exp()
            value = self.critic(batch.observations, actions).amin(dim=0)
            actor_loss = (temperature * log_prob - value).mean()
            preference_loss = actor_loss.new_zeros(())
            if actor_preferences is not None and len(actor_preferences.observations):
                preference_loss = self.preference_loss(actor_preferences)
                actor_loss = actor_loss + self.config.preference_weight * preference_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)

        temperature_loss = -self.log_temperature * (log_prob.detach() + self.target_entropy).mean()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.lerp_(param, self.config.tau)
        return {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "preference_loss": preference_loss.detach(),
            "temperature": self.log_temperature.detach().exp(),
        }
