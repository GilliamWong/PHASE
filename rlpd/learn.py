"""A template for implementing RL methods yourself.

Everything you write goes in this file: the networks, one agent class per
method, and the training loops. The environments, experts and demos are
provided: fast Gymnasium tasks (Pendulum-v1 by default) with hand-written
experts, and the project's Robomimic tasks (rlpd/env.py, and the replay
buffer and demo loading in rlpd/data.py, unchanged from main).

Search for TODO. A blank is either a method body that raises
NotImplementedError or a `todo("...")` call inside a provided loop; replace
it with your code. tests/test_learn.py checks each piece as you fill it in,
e.g. `python -m pytest tests/test_learn.py -k dqn`.

Methods (--algo), in a suggested order:
  bc      behavior cloning on the demos (offline)
  dagger  DAgger: BC, plus expert labels on the states the learner visits
  dqn     deep Q-learning over discretized actions
  ddpg    deterministic actor-critic
  td3     DDPG with twin critics, target smoothing and delayed actor updates
  sac     soft actor-critic (maximum entropy)

Usage:
  python -m rlpd.learn --algo=dqn                            # on Pendulum-v1
  python -m rlpd.learn --algo=dagger --env=MountainCarContinuous-v0
  python -m rlpd.learn --algo=sac --env=robomimic:lift/ph --offline_ratio=0.5 --max_steps=300000
  python -m rlpd.learn --mode=eval --checkpoint=runs/<run>/checkpoints/step_0020000.pt
Every field of Config is a flag; --help lists them. Each training run writes
TensorBoard logs (and optional checkpoints and videos) to its own folder under
--log_dir; view them with: tensorboard --logdir runs
"""

import argparse
import copy
import dataclasses
import json
import os
import time
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from rlpd.data import Batch, TransitionBuffer, load_demo_buffer, sample_mixed
from rlpd.env import DEFAULT_HORIZON, RobomimicEnv, download_dataset, read_env_args, seed_env


def todo(what: str) -> Any:
    """Marks a blank inside provided code. Replace the call with your code."""
    raise NotImplementedError(f"TODO: {what}")


@dataclass
class Config:
    algo: str = "bc"  # bc, dagger, dqn, ddpg, td3 or sac
    # train, or eval: evaluate --checkpoint (its saved config picks the algo,
    # networks and env; --eval_episodes, --seed, --device and --save_video
    # come from the command line).
    mode: str = "train"
    checkpoint: str = ""

    # Environment: a Gymnasium task with continuous actions (Pendulum-v1,
    # MountainCarContinuous-v0, ...), whose actions are rescaled to [-1, 1],
    # or robomimic:<task>/<quality> (e.g. robomimic:lift/ph), a Robomimic task
    # built from that dataset, with a sparse success reward.
    env: str = "Pendulum-v1"
    seed: int = 42
    # Demos and experts (bc, dagger, and --offline_ratio > 0). The Gymnasium
    # tasks in EXPERTS have hand-written experts, which also record
    # --num_demos demo episodes. Robomimic uses the dataset's human demos
    # and, for DAgger, a trained RLPD agent as the expert (--expert).
    num_demos: int = 10
    expert: str = ""  # Robomimic only: an RLPD checkpoint; see load_expert
    success_reward: float = 10.0  # Robomimic only: reward on the (terminal) success step
    horizon: int = DEFAULT_HORIZON  # Robomimic only: max steps per episode

    # Training length, logging and checkpoints.
    # Environment steps for dqn/ddpg/td3/sac; gradient steps for bc.
    max_steps: int = 30_000
    eval_interval: int = 2_000
    eval_episodes: int = 10
    log_interval: int = 500
    log_dir: str = "runs"
    run_name: str = ""  # optional label at the start of the run folder's name
    save_video: bool = False  # one mp4 per evaluation (Gymnasium tasks need pygame)
    checkpoint_model: bool = False  # save the agent at every evaluation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm: bool = True

    # Shared by every method.
    hidden_dims: Tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005  # soft target-network update rate (ddpg, td3, sac)

    # Off-policy loop (dqn, ddpg, td3, sac).
    # Fraction of each batch drawn from the demos. 0 is pure online RL; the
    # sparse Robomimic tasks need demos (0.5, as in RLPD).
    offline_ratio: float = 0.0
    start_training: int = 1_000  # uniformly random actions before updates begin
    utd_ratio: int = 1  # updates per environment step

    # DAgger.
    dagger_iterations: int = 10
    dagger_episodes: int = 5  # rollouts collected per iteration
    dagger_updates: int = 1_000  # BC updates per iteration
    dagger_beta: float = 0.5  # expert-mixing base: beta_i = dagger_beta ** i

    # DQN.
    num_bins: int = 11  # discrete values per action dimension
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000  # act() calls to go from start to end
    target_update_interval: int = 500  # updates between hard target copies
    double_dqn: bool = True

    # DDPG and TD3.
    exploration_noise: float = 0.1  # std of the Gaussian noise added when acting
    policy_noise: float = 0.2  # TD3 target smoothing noise std
    noise_clip: float = 0.5  # TD3 target smoothing noise clip
    policy_delay: int = 2  # TD3: critic updates per actor update

    # SAC.
    init_temperature: float = 1.0
    target_entropy: Optional[float] = None  # None means -action_dim


# ---------------------------------------------------------------------------
# Networks
#
# Observations are flat float32 vectors of size obs_dim. Actions are vectors in
# [-1, 1]^action_dim (robosuite's controller takes that range; Gymnasium
# tasks are rescaled to it). Build the networks from standard modules
# (nn.Linear, nn.ReLU, nn.Sequential, ...).
# ---------------------------------------------------------------------------


class DeterministicPolicy(nn.Module):
    """a = pi(s). Used by BC, DAgger, DDPG and TD3."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()

        n_layers = len(hidden_dims)
        layer_list = []
        layer_list.append(nn.Linear(obs_dim, hidden_dims[0]))
        layer_list.append(nn.ReLU())

        for layer in range (0, n_layers - 1):
            layer_list.append(nn.Linear(hidden_dims[layer], hidden_dims[layer + 1]))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(hidden_dims[n_layers - 1], action_dim))
        layer_list.append(nn.Tanh())

        self.mlp_layers = nn.Sequential(*layer_list)


    def forward(self, observations: Tensor) -> Tensor:
        """(batch, obs_dim) -> (batch, action_dim), every entry in [-1, 1]."""
        return self.mlp_layers(observations)


class GaussianPolicy(nn.Module):
    """A diagonal Gaussian over pre-squash actions x, squashed by a = tanh(x).

    Used by SAC (and, if you like, BC by maximum likelihood). sample() and
    mode() must get the mean and std from forward(); the tests rely on that.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        # TODO: a shared MLP trunk, then two heads: the mean and the log std.
        # Clamp the log std (e.g. to [-20, 2]) so std neither vanishes nor
        # explodes.
        raise NotImplementedError("GaussianPolicy.__init__")

    def forward(self, observations: Tensor) -> Tuple[Tensor, Tensor]:
        """(batch, obs_dim) -> pre-tanh mean and std, each (batch, action_dim)."""
        raise NotImplementedError("GaussianPolicy.forward")

    def sample(self, observations: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns a = tanh(x) with x ~ N(mean, std), shape (batch, action_dim),
        and log pi(a | s), shape (batch,).

        Gradients must flow through a to the network (the reparameterization
        trick). By the change of variables,
            log pi(a | s) = log N(x; mean, std) - sum_i log(1 - tanh(x_i)^2).
        Compute it from x, not from arctanh(a), and write log(1 - tanh(x)^2)
        in a form that stays finite when tanh(x) rounds to +-1.
        """
        raise NotImplementedError("GaussianPolicy.sample")

    def mode(self, observations: Tensor) -> Tensor:
        """The deterministic action tanh(mean), for evaluation."""
        raise NotImplementedError("GaussianPolicy.mode")


class QNetwork(nn.Module):
    """Q(s, a) for continuous actions. Used by DDPG, TD3 and SAC."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        # TODO: an MLP from obs_dim + action_dim through hidden_dims to 1.
        raise NotImplementedError("QNetwork.__init__")

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        """(batch, obs_dim), (batch, action_dim) -> (batch,)."""
        raise NotImplementedError("QNetwork.forward")


class DiscreteQNetwork(nn.Module):
    """Q-values for every bin of every action dimension. Used by DQN."""

    def __init__(
        self, obs_dim: int, action_dim: int, num_bins: int, hidden_dims: Tuple[int, ...]
    ):
        super().__init__()
        # TODO: an MLP from obs_dim through hidden_dims to action_dim * num_bins.
        raise NotImplementedError("DiscreteQNetwork.__init__")

    def forward(self, observations: Tensor) -> Tensor:
        """(batch, obs_dim) -> (batch, action_dim, num_bins)."""
        raise NotImplementedError("DiscreteQNetwork.forward")


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class Agent(nn.Module):
    """What the training loops need from every method.

    Subclasses take (obs_dim, action_dim, config, device), build their
    networks and optimizers on `device`, and store `self.device = device`.
    Keep every network and tensor the agent needs as an attribute (an
    nn.Module, nn.Parameter or registered buffer), so state_dict() saves it
    for checkpoints. Optimizers live outside the state_dict and are not
    saved.
    """

    device: torch.device

    def act(self, observation: np.ndarray, explore: bool) -> np.ndarray:
        """One action, shape (action_dim,), in [-1, 1], for one observation.

        explore=True while collecting training data (epsilon-greedy, noise or
        sampling, depending on the method); explore=False for evaluation.
        Convert with torch.as_tensor(observation, device=self.device)[None]
        and back with [0].cpu().numpy(), under torch.no_grad().
        """
        raise NotImplementedError

    def update(self, batch: Batch) -> Dict[str, Tensor]:
        """One training step on a batch whose fields have shape (batch_size, ...).

        Returns scalar metrics to log, e.g. {"critic_loss": loss.detach()}.
        batch.masks is 0 where the episode truly ended (Robomimic success,
        MountainCar reaching the goal), so the target must not bootstrap
        there, and 1 otherwise.
        """
        raise NotImplementedError


class BCAgent(Agent):
    """Behavior cloning: regress pi(s) onto the dataset's actions.

    --algo=bc trains it on the demos; --algo=dagger trains it on the demos
    plus expert-labeled learner states. Only batch.observations and
    batch.actions are used.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: Config, device: torch.device):
        super().__init__()
        # TODO: a DeterministicPolicy and an Adam optimizer (config.lr).
        # (Variant to try later: a GaussianPolicy trained by maximizing the
        # log-likelihood of the demo actions.)

        self.device = device
        self.policy = DeterministicPolicy(obs_dim, action_dim, config.hidden_dims).to(device)

        self.optim = torch.optim.Adam(self.policy.parameters(), config.lr)
        

    def act(self, observation: np.ndarray, explore: bool) -> np.ndarray:

        with torch.no_grad():
            obs = torch.as_tensor(observation, device=self.device)

            action = self.policy(obs.unsqueeze(0))

            return action.squeeze(0).cpu().numpy()

    def update(self, batch: Batch) -> Dict[str, Tensor]:
        # TODO: one gradient step on the mean squared error between
        # pi(batch.observations) and batch.actions.
        raise NotImplementedError("BCAgent.update")


class DQNAgent(Agent):
    """DQN (Mnih et al., 2015) with each action dimension discretized.

    Each dimension gets num_bins evenly spaced values in [-1, 1]. Taking every
    combination as one discrete action would give num_bins ** action_dim
    actions (about 19 million for 11 bins and Robomimic's 7 dimensions), too
    many to take a max over. Instead the network outputs one Q-value per
    (dimension, bin), and the value of a whole action is their mean:
        Q(s, a) = mean_d Q_d(s, a_d),
    so the max over actions is a max per dimension (DecQN: Seyde et al., 2023,
    https://arxiv.org/abs/2210.12566). Pendulum and MountainCar have one action
    dimension, where this is plain DQN.

    The pieces:
    - the bin values, and a map from continuous actions to bin indices. The
      buffer stores continuous actions, and demo actions are not on the bin
      grid, so update() maps each stored action to its nearest bin;
    - act(): epsilon-greedy, with epsilon decaying linearly from
      epsilon_start to epsilon_end over epsilon_decay_steps exploring calls
      (keep the count on the agent). With explore=False, act greedily;
    - update(): regress Q(s, a) onto r + discount * mask * max_a' Q_target(s', a').
      With config.double_dqn, the online network picks a' and the target
      network scores it (van Hasselt et al., 2016). A Huber loss
      (F.smooth_l1_loss) is the usual choice;
    - a target network, copied from the online network every
      target_update_interval updates.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: Config, device: torch.device):
        super().__init__()
        raise NotImplementedError("DQNAgent.__init__")

    def act(self, observation: np.ndarray, explore: bool) -> np.ndarray:
        raise NotImplementedError("DQNAgent.act")

    def update(self, batch: Batch) -> Dict[str, Tensor]:
        raise NotImplementedError("DQNAgent.update")


class DDPGAgent(Agent):
    """DDPG (Lillicrap et al., 2016, https://arxiv.org/abs/1509.02971).

    The pieces:
    - a DeterministicPolicy actor and a QNetwork critic, each with a target
      copy (copy.deepcopy) and its own Adam optimizer;
    - act(): pi(s), plus Gaussian noise of std exploration_noise when
      exploring, clipped back into [-1, 1];
    - update(): a critic step toward r + discount * mask * Q_target(s', pi_target(s')),
      then an actor step that maximizes Q(s, pi(s)), then soft target updates
      target <- (1 - tau) * target + tau * online.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: Config, device: torch.device):
        super().__init__()
        raise NotImplementedError("DDPGAgent.__init__")

    def act(self, observation: np.ndarray, explore: bool) -> np.ndarray:
        raise NotImplementedError("DDPGAgent.act")

    def update(self, batch: Batch) -> Dict[str, Tensor]:
        raise NotImplementedError("DDPGAgent.update")


class TD3Agent(Agent):
    """TD3 (Fujimoto et al., 2018, https://arxiv.org/abs/1802.09477): DDPG
    with three fixes for overestimated Q-values. Subclassing DDPGAgent is fine.

    - Clipped double Q: two critics; the target takes the min of the two
      target critics.
    - Target policy smoothing: the target action is
      pi_target(s') + clip(noise, -noise_clip, noise_clip), noise ~ N(0, policy_noise^2),
      clipped into [-1, 1].
    - Delayed updates: the actor and the target networks update once every
      policy_delay critic updates.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: Config, device: torch.device):
        super().__init__()
        raise NotImplementedError("TD3Agent.__init__")

    def act(self, observation: np.ndarray, explore: bool) -> np.ndarray:
        raise NotImplementedError("TD3Agent.act")

    def update(self, batch: Batch) -> Dict[str, Tensor]:
        raise NotImplementedError("TD3Agent.update")


class SACAgent(Agent):
    """Soft actor-critic (Haarnoja et al., 2018, https://arxiv.org/abs/1812.05905).

    Maximizes reward plus temperature * entropy. The pieces:
    - a GaussianPolicy actor; two QNetwork critics with target copies;
    - a learned temperature alpha, kept as log alpha (an nn.Parameter);
    - critic step toward
          r + discount * mask * (min_i Q_target_i(s', a') - alpha * log pi(a' | s')),
      with a' ~ pi(. | s');
    - actor step minimizing alpha * log pi(a | s) - min_i Q_i(s, a), a ~ pi(. | s);
    - temperature step minimizing -alpha * (log pi(a | s) + target_entropy),
      which raises alpha when the entropy is below target_entropy;
    - soft target updates with tau.
    act(): a sample when exploring, the mode otherwise.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: Config, device: torch.device):
        super().__init__()
        raise NotImplementedError("SACAgent.__init__")

    def act(self, observation: np.ndarray, explore: bool) -> np.ndarray:
        raise NotImplementedError("SACAgent.act")

    def update(self, batch: Batch) -> Dict[str, Tensor]:
        raise NotImplementedError("SACAgent.update")


AGENTS = {
    "bc": BCAgent,
    "dagger": BCAgent,
    "dqn": DQNAgent,
    "ddpg": DDPGAgent,
    "td3": TD3Agent,
    "sac": SACAgent,
}
ONLINE_ALGOS = ("dqn", "ddpg", "td3", "sac")


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def train_bc(config: Config, agent: Agent, demos: TransitionBuffer, eval_env, run):
    """Behavior cloning: max_steps updates on uniformly sampled demo batches."""
    for step in tqdm.tqdm(range(config.max_steps + 1), smoothing=0.1, disable=not config.tqdm):
        metrics = agent.update(demos.split(demos.sample_rows((config.batch_size,))))
        if step % config.log_interval == 0:
            run.log("training", metrics, step)
        if step % config.eval_interval == 0:
            run.evaluate_and_checkpoint(agent, eval_env, step)


def train_dagger(config: Config, agent: Agent, demos: TransitionBuffer, env, eval_env, run):
    """DAgger (Ross et al., 2011, https://arxiv.org/abs/1011.0686).

    Behavior cloning keeps failing in states the demos never visit, which
    are exactly where its own mistakes take it. DAgger repeatedly rolls out
    the learner, asks the expert what it would have done in each visited
    state, adds those (state, expert action) pairs to the dataset, and
    retrains on everything collected so far.

    Logged against the iteration number.
    """
    expert = make_expert(config, agent.device)
    seed_env(env, config.seed)
    obs_dim, action_dim = demos.obs_dim, demos.action_dim
    # The aggregated dataset D, starting from the demos (iteration 0 is
    # plain BC). Rows are (state, expert action); DAgger needs nothing else,
    # so add the other fields as zeros (or the state again).
    max_episode_steps = env.spec.max_episode_steps if env.spec is not None else config.horizon
    capacity = len(demos) + config.dagger_iterations * config.dagger_episodes * max_episode_steps
    dataset = TransitionBuffer(obs_dim, action_dim, capacity, agent.device)
    dataset.add_many(*(field.cpu().numpy() for field in demos.split(demos.rows[: len(demos)])))

    for iteration in range(config.dagger_iterations + 1):
        for _ in range(config.dagger_updates):
            metrics = agent.update(dataset.split(dataset.sample_rows((config.batch_size,))))
        run.log("training", {**metrics, "dataset_size": len(dataset)}, iteration)
        run.evaluate_and_checkpoint(agent, eval_env, iteration)
        if iteration == config.dagger_iterations:
            break

        # TODO: roll out config.dagger_episodes episodes in `env`
        # (observation, _ = env.reset(); step until terminated or truncated).
        # At every step:
        # - label the state with the expert's action. The expert maps a batch
        #   of observations to actions, so
        #   expert(torch.as_tensor(observation, device=agent.device)[None])
        #   gives a (1, action_dim) tensor;
        # - add (state, expert action) to `dataset` with dataset.add(...);
        # - execute the expert's action with probability
        #   beta = config.dagger_beta ** (iteration + 1), and otherwise the
        #   learner's, agent.act(observation, explore=False). (Ross et al.
        #   use beta_1 = 1, a first round of pure expert data; the demos
        #   play that role here.)
        # Optionally log the rollouts' mean return (info["episode"]["r"] at
        # each episode's last step) with run.log("training", {...}, iteration).
        todo("DAgger: collect expert-labeled rollouts")


def train_online(config: Config, agent: Agent, demos: Optional[TransitionBuffer], env, eval_env, run):
    """The off-policy loop shared by DQN, DDPG, TD3 and SAC: act, store the
    transition, update from the replay buffer."""
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    # One row per environment step, so the buffer never wraps.
    online = TransitionBuffer(obs_dim, action_dim, config.max_steps + 1, agent.device)
    demos_per_batch = int(config.batch_size * config.offline_ratio) if demos is not None else 0
    online_per_batch = config.batch_size - demos_per_batch

    observation = seed_env(env, config.seed)
    for step in tqdm.tqdm(range(config.max_steps + 1), smoothing=0.1, disable=not config.tqdm):
        # TODO: choose the action: uniformly random (env.action_space.sample())
        # for the first config.start_training steps, to fill the buffer before
        # any update, then agent.act(observation, explore=True).
        action = todo("train_online: choose the action")

        next_observation, reward, terminated, truncated, info = env.step(action)

        # TODO: store the transition with online.add(observation, action,
        # reward, mask, next_observation). The mask multiplies the bootstrap
        # term of the TD target: which of terminated (the task ended, e.g.
        # success) and truncated (the time limit) should stop bootstrapping,
        # and why?
        todo("train_online: store the transition")

        observation = next_observation
        if terminated or truncated:
            episode = {"return": info["episode"]["r"], "length": info["episode"]["l"]}
            if "is_success" in info:  # Robomimic only
                episode["success"] = info["is_success"]
            run.log("training", episode, step)
            observation, _ = env.reset()

        if step >= config.start_training:
            # TODO: run config.utd_ratio updates, each on a fresh batch from
            # sample_batch(demos, online, demos_per_batch, online_per_batch),
            # and keep the last update's metrics.
            metrics = todo("train_online: update the agent")
            if step % config.log_interval == 0:
                run.log("training", metrics, step)

        if step % config.eval_interval == 0:
            run.evaluate_and_checkpoint(agent, eval_env, step)


# ---------------------------------------------------------------------------
# Provided: data, expert, environments, evaluation, logging, entry point
# ---------------------------------------------------------------------------


def sample_batch(
    demos: Optional[TransitionBuffer],
    online: TransitionBuffer,
    demos_per_batch: int,
    online_per_batch: int,
) -> Batch:
    """A batch of demos_per_batch demo rows and online_per_batch online rows,
    each drawn uniformly with replacement. Fields have shape (batch_size, ...)."""
    batch = sample_mixed(demos, online, demos_per_batch, online_per_batch, num_minibatches=1)
    return Batch(*(field[0] for field in batch))


def load_expert(checkpoint_path: str, device: torch.device) -> nn.Sequential:
    """The deterministic policy a = tanh(mean(s)) of a trained RLPD agent, as a
    frozen network: (batch, obs_dim) tensor -> (batch, action_dim) tensor.

    Takes a checkpoint saved by `python -m rlpd.train --checkpoint_model` on
    the main or qoil branch. For Lift, the qoil checkout has
    runs/corrector_lift_ph_s42_20261105-180252/checkpoints/step_0150000.pt,
    which succeeded in 199 of 200 episodes.
    """
    if not checkpoint_path:
        raise ValueError("--algo=dagger needs --expert=<RLPD checkpoint>")
    model = torch.load(checkpoint_path, map_location=device, weights_only=True)["model"]
    # The RLPD actor is trunk = [Linear, ReLU] * depth, then a mean head.
    # Rebuild it with the same layer indices so the trunk's weights load
    # under their own names.
    layers = []
    while f"actor.trunk.{len(layers)}.weight" in model:
        out_dim, in_dim = model[f"actor.trunk.{len(layers)}.weight"].shape
        layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
    mean_index = len(layers)
    out_dim, in_dim = model["actor.mean.weight"].shape
    layers += [nn.Linear(in_dim, out_dim), nn.Tanh()]
    expert = nn.Sequential(*layers)

    weights = {k.removeprefix("actor.trunk."): v for k, v in model.items() if k.startswith("actor.trunk.")}
    for name in ("weight", "bias"):
        weights[f"{mean_index}.{name}"] = model[f"actor.mean.{name}"]
    expert.load_state_dict(weights)
    return expert.to(device).requires_grad_(False).eval()


def pendulum_expert(observations: Tensor) -> Tensor:
    """Pendulum-v1: swing up by pumping energy, then balance with a PD controller.

    Observations are (cos theta, sin theta, theta_dot) with theta = 0
    upright; the dynamics are theta_ddot = 15 sin(theta) + 3 u for a torque
    u in [-2, 2], which is action 2u after rescaling to [-1, 1]. The energy
    E = theta_dot^2 / 2 + 15 cos(theta) is 15 at rest upright and changes at
    rate 3 u theta_dot. Mean return about -156 over 20 episodes.
    """
    cos, sin, velocity = observations[..., 0], observations[..., 1], observations[..., 2]
    energy = 0.5 * velocity**2 + 15.0 * cos
    swing_up = torch.where(velocity >= 0, 1.0, -1.0) * (15.0 - energy)
    balance = -(10.0 * torch.atan2(sin, cos) + 2.0 * velocity)
    torque = torch.where(cos > 0.95, balance, swing_up)
    return (torque / 2.0).clamp(-1.0, 1.0)[..., None]


def mountain_car_expert(observations: Tensor) -> Tensor:
    """MountainCarContinuous-v0: push in the direction of motion, which adds
    energy every step until the car clears the hill. Return about 89."""
    return torch.where(observations[..., 1] >= 0, 1.0, -1.0)[..., None]


# Hand-written experts: (batch, obs_dim) tensor -> (batch, action_dim) actions in [-1, 1].
EXPERTS = {"Pendulum-v1": pendulum_expert, "MountainCarContinuous-v0": mountain_car_expert}
ROBOMIMIC = "robomimic:"


def make_expert(config: Config, device: torch.device) -> Callable[[Tensor], Tensor]:
    """The env's expert: maps a (batch, obs_dim) tensor to (batch, action_dim) actions."""
    if config.env.startswith(ROBOMIMIC):
        return load_expert(config.expert, device)
    if config.env not in EXPERTS:
        raise ValueError(f"No expert for {config.env}; there are experts for {sorted(EXPERTS)} and Robomimic")
    return EXPERTS[config.env]


def make_env(config: Config, render: bool = False) -> gym.Env:
    """A training or evaluation env. RecordEpisodeStatistics adds each
    finished episode's return and length to info["episode"]."""
    render_mode = "rgb_array" if render else None
    if config.env.startswith(ROBOMIMIC):
        dataset_path = download_dataset(config.env.removeprefix(ROBOMIMIC))
        env = RobomimicEnv(
            read_env_args(dataset_path), config.horizon, config.success_reward, render_mode
        )
    else:
        env = gym.make(config.env, render_mode=render_mode)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(f"{config.env} has discrete actions; use a continuous-action env")
        bound = np.ones(env.action_space.shape, dtype=np.float32)
        env = gym.wrappers.RescaleAction(env, -bound, bound)
    return gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1)


def load_demos(config: Config, device: torch.device, eps: float = 1e-5) -> TransitionBuffer:
    """Robomimic's human demos, or --num_demos episodes of the env's expert.

    Like the Robomimic demos, expert actions are kept strictly inside
    (-1, 1), where a tanh policy's log-likelihood is finite.
    """
    if config.env.startswith(ROBOMIMIC):
        dataset_path = download_dataset(config.env.removeprefix(ROBOMIMIC))
        return load_demo_buffer(dataset_path, config.success_reward, device, eps)

    expert = make_expert(config, device)
    env = make_env(config)
    observation = seed_env(env, config.seed + 1000)
    columns = ([], [], [], [], [])  # observations, actions, rewards, masks, next_observations
    for _ in range(config.num_demos):
        done = False
        while not done:
            with torch.no_grad():
                action = expert(torch.as_tensor(observation, device=device)[None])[0].cpu().numpy()
            action = np.clip(action, -1 + eps, 1 - eps)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            row = (observation, action, reward, 0.0 if terminated else 1.0, next_observation)
            for column, value in zip(columns, row):
                column.append(value)
            observation = next_observation
            done = terminated or truncated
        observation, _ = env.reset()

    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    demos = TransitionBuffer(obs_dim, action_dim, len(columns[2]), device)
    demos.add_many(*(np.array(column, dtype=np.float32) for column in columns))
    return demos


def evaluate(
    agent: Agent, env: gym.Env, num_episodes: int, video_path: Optional[str] = None
) -> Dict[str, float]:
    """Mean return, length and (Robomimic) success rate of the agent's
    evaluation policy (explore=False); with video_path, saves the first
    episode as an mp4."""
    returns, lengths, successes, frames = [], [], [], []
    for episode in range(num_episodes):
        record = video_path is not None and episode == 0
        observation, _ = env.reset()
        if record:
            frames.append(env.render())
        done = False
        while not done:
            action = agent.act(observation, explore=False)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if record:
                frames.append(env.render())
        returns.append(info["episode"]["r"])
        lengths.append(info["episode"]["l"])
        if "is_success" in info:
            successes.append(info["is_success"])
    if frames:
        imageio.mimwrite(video_path, frames, fps=20)
    results = {"return": float(np.mean(returns)), "length": float(np.mean(lengths))}
    if successes:
        results["success"] = float(np.mean(successes))
    return results


class Run:
    """A training run's folder: TensorBoard logs, config.json, and optional
    videos/ and checkpoints/ (one file per evaluation, named by step)."""

    def __init__(self, config: Config, obs_dim: int, action_dim: int):
        self.config = config
        self.obs_dim, self.action_dim = obs_dim, action_dim
        env_name = config.env.replace(ROBOMIMIC, "").replace("/", "_")
        name_parts = [config.run_name, config.algo, env_name, f"s{config.seed}", time.strftime("%Y%m%d-%H%M%S")]
        self.dir = os.path.join(config.log_dir, "_".join(part for part in name_parts if part))
        # Raises if the folder exists, so two runs never share one.
        os.makedirs(self.dir)
        self.writer = SummaryWriter(self.dir)
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2)
        for flag, folder in ((config.save_video, "videos"), (config.checkpoint_model, "checkpoints")):
            if flag:
                os.makedirs(os.path.join(self.dir, folder))

    def log(self, prefix: str, metrics: Mapping[str, Any], step: int):
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{name}", float(value), step)

    def evaluate_and_checkpoint(self, agent: Agent, eval_env: gym.Env, step: int):
        video_path = None
        if self.config.save_video:
            video_path = os.path.join(self.dir, "videos", f"step_{step:07d}.mp4")
        self.log("evaluation", evaluate(agent, eval_env, self.config.eval_episodes, video_path), step)
        if self.config.checkpoint_model:
            checkpoint = {
                "config": dataclasses.asdict(self.config),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "step": step,
                "model": agent.state_dict(),
            }
            torch.save(checkpoint, os.path.join(self.dir, "checkpoints", f"step_{step:07d}.pt"))

    def close(self):
        self.writer.close()


def parse_config() -> Config:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    hints = typing.get_type_hints(Config)
    for field in dataclasses.fields(Config):
        kind = hints[field.name]
        # Optional[float] -> float; Tuple[int, ...] -> int.
        inner = [a for a in typing.get_args(kind) if a not in (type(None), Ellipsis)]
        flag = f"--{field.name}"
        if kind is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
        elif typing.get_origin(kind) is tuple:
            parser.add_argument(flag, nargs="+", type=inner[0], default=field.default)
        else:
            parser.add_argument(flag, type=inner[0] if inner else kind, default=field.default)
    args = vars(parser.parse_args())
    # nargs="+" gives lists; the config holds tuples.
    return Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in args.items()})


def run_eval(cli: Config):
    """Evaluates a checkpoint saved by a training run of this file."""
    checkpoint = torch.load(cli.checkpoint, map_location=cli.device, weights_only=True)
    config = dataclasses.replace(
        Config(**checkpoint["config"]),
        eval_episodes=cli.eval_episodes,
        seed=cli.seed,
        device=cli.device,
        save_video=cli.save_video,
    )
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    env = make_env(config, render=config.save_video)
    seed_env(env, config.seed)
    agent = AGENTS[config.algo](checkpoint["obs_dim"], checkpoint["action_dim"], config, device)
    agent.load_state_dict(checkpoint["model"])
    video_path = os.path.splitext(cli.checkpoint)[0] + "_eval.mp4" if config.save_video else None
    results = evaluate(agent, env, config.eval_episodes, video_path)
    print(f"{config.algo} at step {checkpoint['step']}, {config.eval_episodes} episodes: {results}")


def main():
    config = parse_config()
    if config.mode == "eval":
        return run_eval(config)
    if config.mode != "train":
        raise ValueError(f"--mode must be train or eval, not {config.mode}")
    if config.algo not in AGENTS:
        raise ValueError(f"--algo must be one of {sorted(AGENTS)}, not {config.algo}")
    assert 0.0 <= config.offline_ratio < 1.0

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    env = make_env(config)
    eval_env = make_env(config, render=config.save_video)
    seed_env(eval_env, config.seed + 42)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

    use_demos = config.algo not in ONLINE_ALGOS or config.offline_ratio > 0
    demos = load_demos(config, device) if use_demos else None

    agent = AGENTS[config.algo](obs_dim, action_dim, config, device)
    run = Run(config, obs_dim, action_dim)
    if config.algo == "bc":
        train_bc(config, agent, demos, eval_env, run)
    elif config.algo == "dagger":
        train_dagger(config, agent, demos, env, eval_env, run)
    else:
        train_online(config, agent, demos, env, eval_env, run)
    run.close()


if __name__ == "__main__":
    main()
