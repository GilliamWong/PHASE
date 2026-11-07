"""RLPD (Ball et al., 2023, https://arxiv.org/abs/2302.02948), with Q-OIL as an option.

RLPD is SAC with three changes: a large critic ensemble with LayerNorm, whose
targets use the min over a random subset of members (REDQ); many critic
updates per environment step; and batches that are half demonstrations, half
online data (data.sample_mixed).

Q-OIL (Teegavarapu et al., CoRL 2026) learns from human interventions. It
adds a bonus reward to intervention transitions, trained into a second,
optimistic critic that bootstraps from the task critic's targets; the actor
maximizes the optimistic critic plus a BC term on intervention actions. The
AgentConfig defaults turn all of it off.
"""

import copy
import dataclasses
import math
from dataclasses import dataclass
from functools import lru_cache, partial
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

    # Q-OIL. The paper's settings are bonus=0.025 (with rewards of 0 or 10),
    # optimistic_critic, bc_weight=0.1 and utd_ratio=10.
    # Reward added to intervention transitions. With optimistic_critic only
    # the optimistic critic sees it; without, it enters the single critic's
    # reward (the paper's no-decoupling ablation).
    bonus: float = 0.0
    # A second critic ensemble, trained on reward + bonus with targets from
    # the task critic's target network so the bonus does not propagate to
    # earlier states. The actor maximizes it instead of the task critic.
    optimistic_critic: bool = False
    # Weight of the BC term over intervention (s, a).
    bc_weight: float = 0.0
    # The BC term's form, which the paper does not give:
    # "log_prob": -E[log pi(a | s)]. On a deterministic corrector it can
    #   grow without bound as the policy's std shrinks (log_std reaches -20),
    #   so bc_weight does not fix its strength;
    # "mse": E[sum_i (tanh(mean_i(s)) - a_i)^2], on the mean action only. It
    #   is bounded (at most 4 per action dimension) and leaves the std to
    #   the entropy term.
    bc_loss: str = "log_prob"
    # For bc_loss="log_prob": the most each sample's log pi(a | s) can count.
    # Above it BC stops pulling, so it cannot keep narrowing the policy once
    # the human's action is likely enough. None: uncapped. A natural value is
    # -target_entropy (action_dim / 2 = 3.5 here), the log-density the
    # entropy term holds the policy's own samples to.
    bc_log_prob_cap: Optional[float] = None
    # BC clamps intervention actions to within this of +-1, where the tanh
    # policy's log-density is infinite. For saturated actions it sets BC's
    # pre-tanh target (atanh(1 - eps), about 6.1 at 1e-5) and so the BC
    # gradient; the paper does not say how it handles them.
    bc_action_eps: float = 1e-5


def _tanh_gaussian_log_prob(
    x: torch.Tensor, standardized: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Log-density of tanh(x), x ~ N(mean, std), summed over action dimensions.

    standardized is (x - mean) / std; sample() passes its noise draw directly.
    """
    gaussian_log_prob = (-0.5 * standardized**2 - torch.log(std) - _HALF_LOG_2PI).sum(-1)
    # log(1 - tanh(x)^2), in a form that stays finite for large |x|.
    log_det_jacobian = 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
    return gaussian_log_prob - log_det_jacobian.sum(-1)


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
        # Xavier-uniform weights and zero biases, as in the JAX version.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the pre-tanh mean and standard deviation."""
        features = self.trunk(observations)
        log_std = self.log_std(features).clamp(self.log_std_min, self.log_std_max)
        return self.mean(features), log_std.exp()

    def sample(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples tanh(x), x ~ N(mean, std), and returns it with its log-density.

        The log-density uses the pre-tanh x; recovering x from the action with
        arctanh is inaccurate once tanh saturates.
        """
        mean, std = self(observations)
        noise = torch.randn_like(mean)
        x = mean + std * noise
        return torch.tanh(x), _tanh_gaussian_log_prob(x, noise, std)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """Log-density of given actions, e.g. a human's for BC.

        Actions are clamped to within eps of +-1, where arctanh is infinite.
        Saturated actions (a corrector's tanh output can round to exactly +-1
        in float32) therefore land at |pre-tanh| = atanh(1 - eps), and their
        log-density depends on eps.
        """
        x = torch.atanh(actions.clamp(-1.0 + eps, 1.0 - eps))
        mean, std = self(observations)
        return _tanh_gaussian_log_prob(x, (x - mean) / std, std)

    def mode(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self(observations)[0])


def _critic_mlp(in_dim: int, hidden_dims: Sequence[int], layer_norm: bool) -> nn.Sequential:
    """One Q(s, a) network: Linear, LayerNorm, ReLU per hidden layer, then Linear to 1."""
    layers = []
    for hidden in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden))
        if layer_norm:
            # eps matches Flax's LayerNorm, which the JAX version used.
            layers.append(nn.LayerNorm(hidden, eps=1e-6))
        layers.append(nn.ReLU())
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))
    # Xavier-uniform weights and zero biases, as in the JAX version.
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers)


@lru_cache(maxsize=None)
def _side_streams(device: torch.device, count: int) -> Tuple[torch.cuda.Stream, ...]:
    # Kept outside the critic so the critic can be deep-copied (streams cannot).
    return tuple(torch.cuda.Stream(device) for _ in range(count))


class EnsembleCritic(nn.Module):
    """num_qs independent Q(s, a) networks.

    On a GPU each critic runs on its own CUDA stream. The networks are small,
    so running them one after another leaves most of the GPU idle; side by
    side, the full update is about 2.3x faster.
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
        self.critics = nn.ModuleList(
            _critic_mlp(obs_dim + action_dim, hidden_dims, layer_norm) for _ in range(num_qs)
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        members: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns Q-values of shape (num_qs, batch).

        members optionally selects a subset of the critics by index; all of
        them are still computed.
        """
        x = torch.cat([observations, actions], dim=-1)
        if x.is_cuda:
            outputs = self._run_on_parallel_streams(x)
        else:
            outputs = [critic(x) for critic in self.critics]
        q = torch.stack(outputs).squeeze(-1)
        return q if members is None else q[members]

    def _run_on_parallel_streams(self, x: torch.Tensor) -> list:
        # Autograd runs each critic's backward pass on the stream of its
        # forward pass, so the backward passes also run side by side.
        main = torch.cuda.current_stream(x.device)
        streams = _side_streams(x.device, len(self.critics))
        outputs = []
        for critic, stream in zip(self.critics, streams):
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                # x was allocated on the main stream; this stops its memory
                # being reused while this stream may still be reading it.
                x.record_stream(stream)
                outputs.append(critic(x))
        for stream in streams:
            main.wait_stream(stream)
        for output in outputs:
            output.record_stream(main)
        return outputs


class RLPDAgent(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, config: AgentConfig, device: torch.device
    ):
        super().__init__()
        if not np.float32(1.0 - config.bc_action_eps) < 1.0:
            # 1 - eps would round to 1 in float32, making BC's log-density
            # infinite and the actor's weights NaN.
            raise ValueError(f"bc_action_eps={config.bc_action_eps} is too small for float32")
        if config.bc_loss not in ("log_prob", "mse"):
            raise ValueError(f"unknown bc_loss {config.bc_loss!r}; use 'log_prob' or 'mse'")
        self.config = config
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.target_entropy = (
            -action_dim / 2 if config.target_entropy is None else config.target_entropy
        )

        self.actor = Actor(obs_dim, action_dim, config.hidden_dims)
        self.critic = EnsembleCritic(
            obs_dim, action_dim, config.hidden_dims, config.num_qs, config.critic_layer_norm
        )
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        # Q-OIL's optimistic critic. Its targets come from the task critic's
        # target network, so it has no target network of its own.
        self.optimistic_critic = (
            EnsembleCritic(
                obs_dim, action_dim, config.hidden_dims, config.num_qs, config.critic_layer_norm
            )
            if config.optimistic_critic
            else None
        )
        self.log_temp = nn.Parameter(torch.tensor(math.log(config.init_temperature)))
        self.to(device)

        # capturable keeps Adam's step count on the GPU, so steps can be
        # recorded into a CUDA graph.
        on_gpu = device.type == "cuda"
        adam = partial(torch.optim.Adam, fused=on_gpu, capturable=on_gpu)
        self.actor_optimizer = adam(self.actor.parameters(), lr=config.actor_lr)
        critic_params = list(self.critic.parameters())
        if self.optimistic_critic is not None:
            # Adam works per parameter, so one optimizer over both ensembles
            # updates each exactly as separate optimizers would.
            critic_params += list(self.optimistic_critic.parameters())
        self.critic_optimizer = adam(critic_params, lr=config.critic_lr)
        self.temp_optimizer = adam([self.log_temp], lr=config.temp_lr)

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> "RLPDAgent":
        """Loads a checkpoint written by train.py, for acting or evaluation
        (the optimizer state is not restored)."""
        checkpoint = torch.load(path, map_location=device)
        config = AgentConfig(**checkpoint["config"])
        agent = cls(checkpoint["obs_dim"], checkpoint["action_dim"], config, device)
        agent.load_state_dict(checkpoint["model"])
        return agent

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Action for one observation: sampled for training, the mean for eval."""
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)[None]
        action = self.actor.mode(obs) if deterministic else self.actor.sample(obs)[0]
        return action[0].cpu().numpy()

    def update(
        self, batch: Batch, intervention_batch: Optional[Batch] = None
    ) -> Dict[str, torch.Tensor]:
        """One RLPD step: a critic update per minibatch, then one actor and one
        temperature update on the last minibatch.

        batch fields have shape (num_minibatches, minibatch_size, ...).
        intervention_batch, a (minibatch_size, ...) sample of intervention
        transitions, is needed when bc_weight > 0. Every operation stays on the
        device, so this can be recorded as a CUDA graph.
        """
        if self.config.bc_weight > 0 and intervention_batch is None:
            raise ValueError("bc_weight > 0 needs an intervention_batch")
        for i in range(batch.rewards.shape[0]):
            minibatch = Batch(*(field[i] for field in batch))
            critic_info = self._update_critic(minibatch)
        actor_info = self._update_actor(minibatch.observations, intervention_batch)
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
            bonus = config.bonus * batch.interventions
            if self.optimistic_critic is None:
                # Without decoupling the bonus goes into the single critic and
                # propagates through its own targets (zero when bonus is 0).
                target_q = target_q + bonus
            else:
                optimistic_target_q = target_q + bonus

        qs = self.critic(batch.observations, batch.actions)
        critic_loss = ((qs - target_q) ** 2).mean()
        info = {
            "critic_loss": critic_loss.detach(),
            "q": qs.detach().mean(),
            # Q-OIL claims bonus transitions are sampled in proportion to the
            # current intervention rate; this is their actual share of batches.
            "batch_intervention_fraction": batch.interventions.mean(),
        }
        loss = critic_loss
        if self.optimistic_critic is not None:
            optimistic_qs = self.optimistic_critic(batch.observations, batch.actions)
            optimistic_loss = ((optimistic_qs - optimistic_target_q) ** 2).mean()
            loss = loss + optimistic_loss
            # Mean optimistic-minus-task gap, separately on intervention and
            # other transitions (each 0 when the minibatch has none).
            gap = (optimistic_qs - qs).detach().mean(dim=0)
            flags = batch.interventions
            num_flagged = flags.sum()
            info.update(
                optimistic_critic_loss=optimistic_loss.detach(),
                optimistic_q=optimistic_qs.detach().mean(),
                gap_intervention=(gap * flags).sum() / num_flagged.clamp(min=1),
                gap_other=(gap * (1 - flags)).sum() / (flags.numel() - num_flagged).clamp(min=1),
            )
        # Zeroing in place keeps gradient buffers at fixed addresses, which
        # CUDA graph replay relies on.
        self.critic_optimizer.zero_grad(set_to_none=False)
        loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            torch._foreach_lerp_(
                list(self.target_critic.parameters()),
                list(self.critic.parameters()),
                config.tau,
            )
        return info

    def _critics(self) -> list:
        """The critics; the last is the one the actor maximizes (Q-OIL's
        optimistic critic, or RLPD's task critic)."""
        if self.optimistic_critic is None:
            return [self.critic]
        return [self.critic, self.optimistic_critic]

    def _actor_loss_terms(
        self, observations: torch.Tensor, intervention_batch: Optional[Batch]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """The actor loss split into its weighted terms, which sum to it, and
        the log-probabilities of the sampled actions."""
        actions, log_probs = self.actor.sample(observations)
        q = self._critics()[-1](observations, actions).mean(dim=0)
        temperature = self.log_temp.exp().detach()
        terms = {"q": -q.mean(), "entropy": (temperature * log_probs).mean()}
        if self.config.bc_weight > 0:
            if self.config.bc_loss == "mse":
                errors = self.actor.mode(intervention_batch.observations) - intervention_batch.actions
                per_sample = (errors**2).sum(-1)
            else:
                log_probs = self.actor.log_prob(
                    intervention_batch.observations,
                    intervention_batch.actions,
                    self.config.bc_action_eps,
                )
                if self.config.bc_log_prob_cap is not None:
                    log_probs = log_probs.clamp(max=self.config.bc_log_prob_cap)
                per_sample = -log_probs
            # Every stored intervention row has interventions = 1. Before the
            # first intervention, sampling the empty buffer returns all-zero
            # rows, which this weight drops.
            bc_loss = (intervention_batch.interventions * per_sample).mean()
            terms["bc"] = self.config.bc_weight * bc_loss
        return terms, log_probs

    def _update_actor(
        self, observations: torch.Tensor, intervention_batch: Optional[Batch]
    ) -> Dict[str, torch.Tensor]:
        critics = self._critics()
        # The critics are fixed here; this skips computing their parameter gradients.
        for critic in critics:
            critic.requires_grad_(False)
        terms, log_probs = self._actor_loss_terms(observations, intervention_batch)
        actor_loss = sum(terms.values())
        self.actor_optimizer.zero_grad(set_to_none=False)
        actor_loss.backward()
        self.actor_optimizer.step()
        for critic in critics:
            critic.requires_grad_(True)
        info = {
            "actor_loss": actor_loss.detach(),
            "actor_q": -terms["q"].detach(),
            "entropy": -log_probs.detach().mean(),
        }
        if "bc" in terms:
            info["bc_loss"] = terms["bc"].detach() / self.config.bc_weight  # unweighted
        return info

    def actor_gradient_norms(
        self, observations: torch.Tensor, intervention_batch: Optional[Batch] = None
    ) -> Dict[str, torch.Tensor]:
        """The L2 norm of each actor-loss term's gradient over the actor's
        weights, on one batch: how hard each term pushes the policy. Unlike
        the loss values, these ignore constants such as BC's tanh correction,
        log(1 - a^2), which depends only on bc_action_eps for saturated
        actions. Draws random numbers but changes no weights."""
        critics = self._critics()
        for critic in critics:
            critic.requires_grad_(False)
        terms, _ = self._actor_loss_terms(observations, intervention_batch)
        params = list(self.actor.parameters())
        norms = {}
        for name, term in terms.items():
            # allow_unused: MSE BC never touches the log-std weights.
            grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            norms[f"actor_grad_norm_{name}"] = torch.linalg.vector_norm(
                torch.cat([g.flatten() for g in grads if g is not None])
            )
        for critic in critics:
            critic.requires_grad_(True)
        return norms

    def _update_temperature(self, entropy: torch.Tensor) -> Dict[str, torch.Tensor]:
        temperature = self.log_temp.exp()
        temp_loss = temperature * (entropy - self.target_entropy)
        self.temp_optimizer.zero_grad(set_to_none=False)
        temp_loss.backward()
        self.temp_optimizer.step()
        return {"temperature": temperature.detach(), "temperature_loss": temp_loss.detach()}

    def checkpoint(self) -> dict:
        return {
            # What from_checkpoint needs to rebuild the agent.
            "config": dataclasses.asdict(self.config),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
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
