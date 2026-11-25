"""Paper-faithful RLIF training and a Robomimic Lift/Can entry point.

The paper alternates replay-based learning with rounds of episode collection.
This module implements collection and intervention-feedback labeling.  The
round schedule remains separate so it can be matched to a chosen experiment
without changing the semantics tested here.
"""

import argparse
import dataclasses
import json
import os
import time
import typing
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rlif.agent import AgentConfig, RLPDAgent
from rlif.data import TransitionBuffer, load_rlif_demo_buffer, sample_mixed, split_demo_names
from rlif.human import (
    InterventionController,
    InterventionFeedback,
    LabeledTransition,
    OfficialRandomController,
    OfficialValueController,
    PhaseProgressController,
    PhaseRandomController,
)
from rlpd.env import DEFAULT_HORIZON, RobomimicEnv, download_dataset, read_env_args, seed_env


def add_labeled(buffer: TransitionBuffer, transition: Optional[LabeledTransition]) -> None:
    if transition is None:
        return
    buffer.add(
        transition.observation,
        transition.action,
        transition.reward,
        transition.mask,
        transition.next_observation,
    )


def collect_episode(env, agent, controller: InterventionController, replay: TransitionBuffer):
    """Collects one episode and appends correctly shifted RLIF rewards.

    ``agent.act`` must accept ``deterministic=False`` as the existing RLPD
    learner does.  The policy action is computed even at takeover states so a
    controller may inspect it when deciding whether to intervene.
    """
    observation, _ = env.reset()
    controller.reset()
    feedback = InterventionFeedback()
    feedback.reset()
    intervention_steps = intervention_events = steps = 0

    while True:
        policy_action = agent.act(observation, deterministic=False)
        corrective_action = controller.act(observation, policy_action)
        intervening = corrective_action is not None
        was_intervening = feedback.intervening
        add_labeled(replay, feedback.begin_step(intervening))
        if intervening and not was_intervening:
            intervention_events += 1
        intervention_steps += int(intervening)

        action = np.asarray(corrective_action if intervening else policy_action)
        next_observation, _, terminated, truncated, info = env.step(action)
        feedback.record_step(
            observation,
            action,
            0.0 if terminated else 1.0,
            next_observation,
        )
        steps += 1
        observation = next_observation
        if terminated or truncated:
            add_labeled(replay, feedback.end_episode())
            return {
                "steps": steps,
                "intervention_steps": intervention_steps,
                "intervention_events": intervention_events,
                "info": info,
            }


@dataclass
class RoundConfig:
    """Paper-reported RLIF schedule; task-specific UTD ratios override 1."""

    num_rounds: int = 100
    episodes_per_round: int = 5
    # The paper reports 200 pretraining epochs x 300 train steps.
    pretrain_updates: int = 60_000
    # It reports 25 epochs x 100 train steps after each collection round.
    updates_per_round: int = 2_500
    batch_size: int = 256
    offline_ratio: float = 0.5
    utd_ratio: int = 1


@dataclass
class ExperimentConfig:
    dataset: str = "lift/ph"
    horizon: int = DEFAULT_HORIZON
    seed: int = 42
    gate: str = "official_random"
    corrector_checkpoint: str = ""
    reference_checkpoint: str = ""
    corrector_deterministic: bool = False

    # Official RLIF gates.
    official_random_probability: float = 0.5
    official_value_threshold: float = 0.97
    official_takeover_steps: int = 4
    official_bad_probability: float = 0.95
    official_good_probability: float = 0.05
    official_reference_uses_corrector_action: bool = True

    # PHASE/Q-OIL synthetic-human adapter.
    phase_intervention_probability: float = 0.005
    phase_full_takeover_probability: float = 0.5
    phase_min_takeover_steps: int = 10
    phase_max_takeover_steps: int = 40
    phase_progress_window: int = 12
    phase_progress_threshold: float = -0.158
    phase_stall_probability: float = 0.599

    eval_every_rounds: int = 1
    eval_episodes: int = 10
    eval_deterministic: bool = False
    log_dir: str = "runs"
    run_name: str = ""
    checkpoint_model: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Disjoint BC-corrector / initial-RLIF demonstration split.
    demo_split_seed: int = 42
    bc_train_demos: int = 140
    bc_validation_demos: int = 10
    rlif_initial_demos: int = 50


def _updates(agent, demos, online, count: int, config: RoundConfig, demo_only: bool = False):
    if count < 0 or config.utd_ratio < 1:
        raise ValueError("update counts must be nonnegative and utd_ratio positive")
    if not 0 <= config.offline_ratio < 1:
        raise ValueError("offline_ratio must be in [0, 1)")
    last_info = None
    for _ in range(count):
        if demo_only:
            num_demos, num_online = config.batch_size, 0
        else:
            num_demos = int(config.batch_size * config.offline_ratio)
            num_online = config.batch_size - num_demos
        batch = sample_mixed(demos, online, num_demos, num_online, config.utd_ratio)
        last_info = agent.update(batch)
    return last_info


def train_rounds(
    env,
    agent,
    controller: InterventionController,
    demos: TransitionBuffer,
    online: TransitionBuffer,
    config: RoundConfig,
    on_round: Optional[Callable[[int, list, Optional[dict]], None]] = None,
):
    """Runs the paper's pretrain/collect/update abstraction.

    Collection precedes each round's updates, matching the executable
    dataset-aggregation loop used by the authors. This differs from their
    separate step-interleaved RLIF scripts and is intentionally exposed as a
    distinct driver rather than silently mixing the two protocols.
    """
    _updates(agent, demos, online, config.pretrain_updates, config, demo_only=True)
    history = []
    for round_index in range(config.num_rounds):
        collection = [
            collect_episode(env, agent, controller, online)
            for _ in range(config.episodes_per_round)
        ]
        update_info = _updates(agent, demos, online, config.updates_per_round, config)
        history.append({"collection": collection, "update": update_info})
        if on_round is not None:
            on_round(round_index, collection, update_info)
    return history


def _config_from_run(checkpoint_path: str) -> AgentConfig:
    """Reads the learner fields saved beside an RLPD/RLIF checkpoint."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        return AgentConfig()
    with open(config_path, encoding="utf-8") as f:
        saved = json.load(f)
    fields = {field.name for field in dataclasses.fields(AgentConfig)}
    values = {key: value for key, value in saved.items() if key in fields}
    if "hidden_dims" in values:
        values["hidden_dims"] = tuple(values["hidden_dims"])
    return AgentConfig(**values)


def load_agent(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
) -> RLPDAgent:
    if not checkpoint_path:
        raise ValueError("a checkpoint path is required for this gate")
    agent = RLPDAgent(obs_dim, action_dim, _config_from_run(checkpoint_path), device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent.load_state_dict(payload["model"])
    agent.eval()
    return agent


def make_controller(
    config: ExperimentConfig,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
) -> InterventionController:
    """Builds one explicitly named paper-native or PHASE-adapted gate."""
    corrector_agent = load_agent(config.corrector_checkpoint, obs_dim, action_dim, device)

    def corrector(observation):
        return corrector_agent.act(observation, deterministic=config.corrector_deterministic)

    rng = np.random.default_rng(config.seed)
    if config.gate == "official_random":
        return OfficialRandomController(
            corrector, rng, config.official_random_probability, config.official_takeover_steps
        )
    if config.gate == "phase_random":
        return PhaseRandomController(
            corrector,
            rng,
            config.phase_intervention_probability,
            config.phase_full_takeover_probability,
            config.phase_min_takeover_steps,
            config.phase_max_takeover_steps,
        )

    reference_agent = load_agent(config.reference_checkpoint, obs_dim, action_dim, device)

    @torch.no_grad()
    def reference_q(observation, action):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=device)[None]
        act = torch.as_tensor(action, dtype=torch.float32, device=device)[None]
        return float(reference_agent.critic(obs, act).mean())

    if config.gate == "official_value":
        reference_policy = corrector if config.official_reference_uses_corrector_action else (
            lambda observation: reference_agent.act(
                observation, deterministic=config.corrector_deterministic
            )
        )
        return OfficialValueController(
            corrector,
            reference_q,
            rng,
            config.official_value_threshold,
            config.official_takeover_steps,
            reference_policy,
            config.official_bad_probability,
            config.official_good_probability,
        )
    if config.gate == "phase_progress":
        @torch.no_grad()
        def state_value(observation):
            obs = torch.as_tensor(observation, dtype=torch.float32, device=device)[None]
            action = reference_agent.actor.mode(obs)
            return float(reference_agent.critic(obs, action).mean())

        return PhaseProgressController(
            corrector,
            state_value,
            rng,
            config.phase_intervention_probability,
            config.phase_progress_window,
            config.phase_progress_threshold,
            config.phase_stall_probability,
            full_takeover_probability=config.phase_full_takeover_probability,
            min_takeover_steps=config.phase_min_takeover_steps,
            max_takeover_steps=config.phase_max_takeover_steps,
        )
    raise ValueError(
        f"unknown gate {config.gate!r}; use official_random, official_value, "
        "phase_random or phase_progress"
    )


def evaluate(agent, env, num_episodes: int, deterministic: bool) -> Mapping[str, float]:
    returns, lengths, successes = [], [], []
    for _ in range(num_episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action = agent.act(observation, deterministic=deterministic)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        returns.append(float(info["episode"]["r"]))
        lengths.append(float(info["episode"]["l"]))
        successes.append(float(info["is_success"]))
    return {
        "return": float(np.mean(returns)),
        "length": float(np.mean(lengths)),
        "success": float(np.mean(successes)),
    }


def _add_scalars(writer: SummaryWriter, prefix: str, values: Mapping, step: int):
    for name, value in values.items():
        writer.add_scalar(f"{prefix}/{name}", float(value), step)


def parse_config() -> tuple[ExperimentConfig, RoundConfig, AgentConfig]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate",
        choices=("official_random", "official_value", "phase_random", "phase_progress"),
        default=ExperimentConfig.gate,
    )
    for cls in (ExperimentConfig, RoundConfig, AgentConfig):
        hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            if field.name == "gate":
                continue
            kind = hints[field.name]
            inner = [arg for arg in typing.get_args(kind) if arg not in (type(None), Ellipsis)]
            flag = f"--{field.name}"
            if kind is bool:
                parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
            elif typing.get_origin(kind) is tuple:
                parser.add_argument(flag, nargs="+", type=inner[0], default=field.default)
            else:
                parser.add_argument(flag, type=inner[0] if inner else kind, default=field.default)
    args = vars(parser.parse_args())

    def build(cls):
        values = {field.name: args[field.name] for field in dataclasses.fields(cls)}
        return cls(**{key: tuple(value) if isinstance(value, list) else value for key, value in values.items()})

    return build(ExperimentConfig), build(RoundConfig), build(AgentConfig)


def main():
    experiment, rounds, agent_config = parse_config()
    torch.manual_seed(experiment.seed)
    device = torch.device(experiment.device)
    dataset_path = download_dataset(experiment.dataset)
    env_args = read_env_args(dataset_path)

    def wrapped_env(seed):
        env = gym.wrappers.RecordEpisodeStatistics(
            RobomimicEnv(env_args, experiment.horizon), buffer_length=1
        )
        seed_env(env, seed)
        return env

    env, eval_env = wrapped_env(experiment.seed), wrapped_env(experiment.seed + 42)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = RLPDAgent(obs_dim, action_dim, agent_config, device)
    split = split_demo_names(
        dataset_path,
        experiment.bc_train_demos,
        experiment.bc_validation_demos,
        experiment.rlif_initial_demos,
        experiment.demo_split_seed,
    )
    demos = load_rlif_demo_buffer(dataset_path, device, split["rlif"])
    capacity = rounds.num_rounds * rounds.episodes_per_round * experiment.horizon
    online = TransitionBuffer(obs_dim, action_dim, capacity, device)
    controller = make_controller(experiment, obs_dim, action_dim, device)

    name = "_".join(
        part for part in (
            experiment.run_name,
            "rlif",
            experiment.gate,
            experiment.dataset.replace("/", "_"),
            f"s{experiment.seed}",
            time.strftime("%Y%m%d-%H%M%S"),
        ) if part
    )
    run_dir = os.path.join(experiment.log_dir, name)
    os.makedirs(run_dir)
    writer = SummaryWriter(run_dir)
    full_config = {
        **dataclasses.asdict(experiment),
        **dataclasses.asdict(rounds),
        **dataclasses.asdict(agent_config),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2)

    def on_round(index, collection, update_info):
        step = index + 1
        assistance = {
            "intervention_steps": sum(item["intervention_steps"] for item in collection),
            "intervention_events": sum(item["intervention_events"] for item in collection),
            "environment_steps": sum(item["steps"] for item in collection),
        }
        assistance["intervention_fraction"] = (
            assistance["intervention_steps"] / assistance["environment_steps"]
        )
        _add_scalars(writer, "training", assistance, step)
        if update_info:
            _add_scalars(writer, "training", update_info, step)
        if step % experiment.eval_every_rounds == 0:
            _add_scalars(
                writer,
                "evaluation",
                evaluate(agent, eval_env, experiment.eval_episodes, experiment.eval_deterministic),
                step,
            )
            if experiment.checkpoint_model:
                checkpoint_dir = os.path.join(run_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    {"round": step, **agent.checkpoint()},
                    os.path.join(checkpoint_dir, f"round_{step:03d}.pt"),
                )

    try:
        train_rounds(env, agent, controller, demos, online, rounds, on_round)
    finally:
        writer.close()
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
