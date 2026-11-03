"""RLPD (Ball et al., 2023, https://arxiv.org/abs/2302.02948).

RLPD is SAC with three changes: a large critic ensemble with LayerNorm, whose
targets use the min over a random subset of members (REDQ); many critic
updates per environment step; and batches that are half demonstrations, half
online data (data.sample_mixed).
"""

import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rlpd.data import Batch

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass
class AgentConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    discount: float = 0.99
    tau: float = 0.005  # target-critic update rate
    num_qs: int = 10  # critic ensemble size
    num_min_qs: Optional[int] = 2  # members in each target's min; None uses all
    critic_layer_norm: bool = True
    init_temperature: float = 1.0
    target_entropy: Optional[float] = None  # None means -action_dim / 2
    # Off for sparse rewards, as RLPD does for Adroit: with it on, the entropy
    # bonus swamped the 0/10 success reward and Lift Q-values reached ~240
    # when no return can exceed 10.
    backup_entropy: bool = False


def _xavier_uniform_(weight: torch.Tensor, fan_in: int, fan_out: int):
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(weight, -bound, bound)


def tanh_gaussian_sample(
    mean: torch.Tensor, std: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples tanh(x), x ~ N(mean, std), and returns it with its log-density.

    The log-density uses the pre-tanh x; recovering x from the action with
    arctanh is inaccurate once tanh saturates.
    """
    noise = torch.randn_like(mean)
    x = mean + std * noise
    gaussian_log_prob = (-0.5 * noise**2 - torch.log(std) - _HALF_LOG_2PI).sum(-1)
    # log(1 - tanh(x)^2), in a form that stays finite for large |x|.
    log_det_jacobian = 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
    return torch.tanh(x), gaussian_log_prob - log_det_jacobian.sum(-1)


class Actor(nn.Module):
    """A state-dependent diagonal Gaussian squashed into (-1, 1) by tanh."""

    log_std_min = -20.0
    log_std_max = 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        layers, width = [], obs_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(width, action_dim)
        self.log_std = nn.Linear(width, action_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _xavier_uniform_(module.weight, module.in_features, module.out_features)
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the pre-tanh mean and standard deviation."""
        features = self.trunk(observations)
        log_std = self.log_std(features).clamp(self.log_std_min, self.log_std_max)
        return self.mean(features), log_std.exp()

    def sample(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return tanh_gaussian_sample(*self(observations))

    def mode(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self(observations)[0])


class EnsembleCritic(nn.Module):
    """num_qs independent Q(s, a) MLPs, run together as batched matmuls.

    Each layer's parameters are stacked along a leading member axis, so the
    whole ensemble costs one matmul per layer.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        num_qs: int,
        layer_norm: bool,
    ):
        super().__init__()
        dims = [obs_dim + action_dim, *hidden_dims, 1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            weight = torch.empty(num_qs, fan_in, fan_out)
            for member_weight in weight:
                _xavier_uniform_(member_weight, fan_in, fan_out)
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(torch.zeros(num_qs, 1, fan_out)))
        self.layer_norm = layer_norm
        if layer_norm:
            self.norm_scales = nn.ParameterList(
                nn.Parameter(torch.ones(num_qs, 1, h)) for h in hidden_dims
            )
            self.norm_biases = nn.ParameterList(
                nn.Parameter(torch.zeros(num_qs, 1, h)) for h in hidden_dims
            )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        members: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns Q-values of shape (num members, batch).

        members optionally selects a subset of the ensemble by index.
        """

        def pick(param):
            return param if members is None else param[members]

        # (batch, in) @ (members, in, out) broadcasts to (members, batch, out).
        x = torch.cat([observations, actions], dim=-1)
        num_hidden = len(self.weights) - 1
        for layer in range(num_hidden + 1):
            x = torch.matmul(x, pick(self.weights[layer])) + pick(self.biases[layer])
            if layer < num_hidden:
                if self.layer_norm:
                    # eps matches Flax's LayerNorm, which the JAX version used.
                    x = F.layer_norm(x, x.shape[-1:], eps=1e-6)
                    x = x * pick(self.norm_scales[layer]) + pick(self.norm_biases[layer])
                x = F.relu(x)
        return x.squeeze(-1)


class RLPDAgent(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, config: AgentConfig, device: torch.device
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.target_entropy = (
            -action_dim / 2 if config.target_entropy is None else config.target_entropy
        )

        self.actor = Actor(obs_dim, action_dim, config.hidden_dims)
        self.critic = EnsembleCritic(
            obs_dim, action_dim, config.hidden_dims, config.num_qs, config.critic_layer_norm
        )
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_temp = nn.Parameter(torch.tensor(math.log(config.init_temperature)))
        self.to(device)

        # capturable keeps Adam's step count on the GPU, so steps can be
        # recorded into a CUDA graph.
        on_gpu = device.type == "cuda"
        adam = partial(torch.optim.Adam, fused=on_gpu, capturable=on_gpu)
        self.actor_optimizer = adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = adam(self.critic.parameters(), lr=config.critic_lr)
        self.temp_optimizer = adam([self.log_temp], lr=config.temp_lr)

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Action for one observation: sampled for training, the mean for eval."""
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)[None]
        action = self.actor.mode(obs) if deterministic else self.actor.sample(obs)[0]
        return action[0].cpu().numpy()

    def update(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """One RLPD step: a critic update per minibatch, then one actor and one
        temperature update on the last minibatch.

        batch fields have shape (num_minibatches, minibatch_size, ...). Every
        operation stays on the device, so this can be recorded as a CUDA graph.
        """
        for i in range(batch.rewards.shape[0]):
            minibatch = Batch(*(field[i] for field in batch))
            critic_info = self._update_critic(minibatch)
        actor_info = self._update_actor(minibatch.observations)
        temp_info = self._update_temperature(actor_info["entropy"])
        return {**critic_info, **actor_info, **temp_info}

    def _update_critic(self, batch: Batch) -> Dict[str, torch.Tensor]:
        config = self.config
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(batch.next_observations)
            members = None
            if config.num_min_qs is not None:
                # A random subset of members, drawn on the device (REDQ).
                members = torch.rand(config.num_qs, device=self.device).argsort()
                members = members[: config.num_min_qs]
            next_q = self.target_critic(batch.next_observations, next_actions, members)
            next_q = next_q.min(dim=0).values
            target_q = batch.rewards + config.discount * batch.masks * next_q
            if config.backup_entropy:
                target_q -= (
                    config.discount * batch.masks * self.log_temp.exp() * next_log_probs
                )

        qs = self.critic(batch.observations, batch.actions)
        critic_loss = ((qs - target_q) ** 2).mean()
        # Zeroing in place keeps gradient buffers at fixed addresses, which
        # CUDA graph replay relies on.
        self.critic_optimizer.zero_grad(set_to_none=False)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            torch._foreach_lerp_(
                list(self.target_critic.parameters()),
                list(self.critic.parameters()),
                config.tau,
            )
        return {"critic_loss": critic_loss.detach(), "q": qs.detach().mean()}

    def _update_actor(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        # The critic is fixed here; this skips computing its parameter gradients.
        self.critic.requires_grad_(False)
        actions, log_probs = self.actor.sample(observations)
        q = self.critic(observations, actions).mean(dim=0)
        temperature = self.log_temp.exp().detach()
        actor_loss = (temperature * log_probs - q).mean()
        self.actor_optimizer.zero_grad(set_to_none=False)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        return {"actor_loss": actor_loss.detach(), "entropy": -log_probs.detach().mean()}

    def _update_temperature(self, entropy: torch.Tensor) -> Dict[str, torch.Tensor]:
        temperature = self.log_temp.exp()
        temp_loss = temperature * (entropy - self.target_entropy)
        self.temp_optimizer.zero_grad(set_to_none=False)
        temp_loss.backward()
        self.temp_optimizer.step()
        return {"temperature": temperature.detach(), "temperature_loss": temp_loss.detach()}

    def checkpoint(self) -> dict:
        return {
            "model": self.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temp_optimizer": self.temp_optimizer.state_dict(),
        }


def capture_cuda_graph(
    step: Callable[[], Dict[str, torch.Tensor]], num_warmup: int = 3
) -> Callable[[], Dict[str, torch.Tensor]]:
    """Records step's GPU work once; the returned function replays it.

    Replaying skips Python and per-kernel launch costs, which dominate RLPD's
    small networks. step must take no inputs from the host, keep tensor shapes
    fixed and never wait on the GPU (no .item(), no data-dependent shapes).
    Recording first runs step num_warmup times, as CUDA requires, and those
    runs are real updates. Each replay overwrites the returned tensors.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(num_warmup):
            step()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = step()

    def replay():
        graph.replay()
        return outputs

    return replay
