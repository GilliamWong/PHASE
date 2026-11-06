"""Trains RLPD on a Robomimic task, optionally with a synthetic human
intervening and Q-OIL learning from the interventions.

Usage: python -m rlpd.train --dataset=lift/ph [--flag=value ...]
Every field of TrainConfig, AgentConfig and HumanConfig is a flag; --help
lists them.
Each run writes TensorBoard logs, videos and checkpoints to its own folder
under --log_dir; view them with: tensorboard --logdir runs
"""

import argparse
import collections
import dataclasses
import json
import os
import time
import typing
from dataclasses import dataclass
from typing import Mapping, Optional

import gymnasium as gym
import imageio
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from rlpd.agent import AgentConfig, RLPDAgent, capture_cuda_graph
from rlpd.data import TransitionBuffer, load_demo_buffer, sample_mixed
from rlpd.env import DEFAULT_HORIZON, RobomimicEnv, download_dataset, read_env_args, seed_env
from rlpd.human import HumanConfig, SyntheticHuman, critic_value


@dataclass
class TrainConfig:
    # Robomimic task/quality, e.g. lift/ph or can/mh. Downloaded on first use;
    # the env is built from the dataset's recorded env config.
    dataset: str = "lift/ph"
    offline_ratio: float = 0.5  # fraction of each batch from the demos; 0 is purely online
    num_demos: Optional[int] = None  # use the dataset's first num_demos demos; None uses all
    success_reward: float = 10.0  # reward on the (terminal) success step
    horizon: int = DEFAULT_HORIZON  # max steps per episode
    seed: int = 42
    batch_size: int = 256  # per critic update
    # Critic updates per environment step: Q-OIL's setting for every method
    # (RLPD's own default is 20, about twice as slow).
    utd_ratio: int = 10
    max_steps: int = 300_000  # environment steps
    start_training: int = 5_000  # random-action steps before updates begin
    pretrain_steps: int = 0  # demo-only updates before online training
    eval_interval: int = 5_000
    eval_episodes: int = 10
    log_interval: int = 1_000
    log_dir: str = "runs"
    run_name: str = ""  # optional label at the start of the run folder's name, e.g. qoil_b0.25
    save_video: bool = False  # save one evaluation episode per evaluation as an mp4
    # Save the agent every checkpoint_interval steps, to checkpoints/step_<step>.pt.
    checkpoint_model: bool = False
    # Save the online transitions added since the previous save every
    # checkpoint_interval steps, to buffer/step_<step>.pt.
    checkpoint_buffer: bool = False
    # Independent of eval_interval: saving is cheap, while an evaluation of
    # failing episodes takes about as long as 1,500 training steps.
    checkpoint_interval: int = 5_000
    cuda_graph: bool = True  # replay updates as a CUDA graph (GPU only)
    # TF32 matrix multiplies on the GPU: updates about 25% faster, with
    # matmul inputs rounded to 10 mantissa bits instead of float32's 23. This
    # changes a run's exact numbers, not the algorithm; --no-tf32 for full
    # float32 (e.g. to compare against the JAX version).
    tf32: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm: bool = True


def parse_config() -> typing.Tuple[TrainConfig, AgentConfig, HumanConfig]:
    parser = argparse.ArgumentParser(description=__doc__)
    for cls in (TrainConfig, AgentConfig, HumanConfig):
        hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            kind = hints[field.name]
            # Optional[int] -> int; Tuple[int, ...] -> int.
            inner = [a for a in typing.get_args(kind) if a not in (type(None), Ellipsis)]
            flag = f"--{field.name}"
            if kind is bool:
                parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
            elif typing.get_origin(kind) is tuple:
                parser.add_argument(flag, nargs="+", type=inner[0], default=field.default)
            else:
                parser.add_argument(flag, type=inner[0] if inner else kind, default=field.default)
    args = vars(parser.parse_args())

    def build(cls):
        values = {f.name: args[f.name] for f in dataclasses.fields(cls)}
        # nargs="+" gives lists; the configs hold tuples.
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})

    return build(TrainConfig), build(AgentConfig), build(HumanConfig)


def log_scalars(writer: SummaryWriter, prefix: str, metrics: Mapping, step: int):
    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", float(value), step)


def make_env(
    env_args: dict, config: TrainConfig, render_mode: typing.Optional[str] = None
) -> gym.Env:
    """The task env used for training and evaluation."""
    # RecordEpisodeStatistics adds each finished episode's return, length and
    # duration to info["episode"].
    return gym.wrappers.RecordEpisodeStatistics(
        RobomimicEnv(env_args, config.horizon, config.success_reward, render_mode=render_mode),
        buffer_length=1,
    )


def evaluate(
    agent: RLPDAgent,
    env: gym.Env,
    num_episodes: int,
    video_path: typing.Optional[str] = None,
) -> dict:
    """Runs the deterministic policy and returns its mean return, length and
    success rate; with video_path, also saves the first episode as an mp4."""
    returns, lengths, successes, frames = [], [], [], []
    for episode in range(num_episodes):
        record = video_path is not None and episode == 0
        observation, _ = env.reset()
        if record:
            frames.append(env.render())
        done = False
        while not done:
            action = agent.act(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if record:
                frames.append(env.render())
        returns.append(info["episode"]["r"])
        lengths.append(info["episode"]["l"])
        successes.append(info["is_success"])

    if frames:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=20)
    return {"return": np.mean(returns), "length": np.mean(lengths), "success": np.mean(successes)}


def main():
    config, agent_config, human_config = parse_config()
    assert 0.0 <= config.offline_ratio < 1.0
    learns_from_interventions = (
        agent_config.bonus != 0 or agent_config.bc_weight != 0 or agent_config.optimistic_critic
    )
    if learns_from_interventions and not human_config.corrector:
        # Without interventions these would silently train plain RLPD.
        raise ValueError("--bonus, --bc_weight and --optimistic_critic need --corrector")
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high" if config.tf32 else "highest")
    device = torch.device(config.device)
    use_cuda_graph = config.cuda_graph and device.type == "cuda"

    name_parts = [
        config.run_name,
        config.dataset.replace("/", "_"),
        f"s{config.seed}",
        time.strftime("%Y%m%d-%H%M%S"),
    ]
    run_dir = os.path.join(config.log_dir, "_".join(part for part in name_parts if part))
    # Raises if the folder exists (the same settings started in the same
    # second), so two runs never write into one folder.
    os.makedirs(run_dir)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    buffer_dir = os.path.join(run_dir, "buffer")
    if config.checkpoint_model:
        os.makedirs(checkpoint_dir)
    if config.checkpoint_buffer:
        os.makedirs(buffer_dir)
    writer = SummaryWriter(run_dir)
    full_config = {
        **dataclasses.asdict(config),
        **dataclasses.asdict(agent_config),
        **dataclasses.asdict(human_config),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=2)
    # Indented so TensorBoard's text tab shows it as a code block.
    writer.add_text("config", "    " + json.dumps(full_config, indent=2).replace("\n", "\n    "))
    video_dir = os.path.join(run_dir, "videos") if config.save_video else None

    dataset_path = download_dataset(config.dataset)
    env_args = read_env_args(dataset_path)
    env = make_env(env_args, config)
    observation = seed_env(env, config.seed)
    eval_env = make_env(env_args, config, render_mode="rgb_array" if config.save_video else None)
    seed_env(eval_env, config.seed + 42)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = RLPDAgent(obs_dim, action_dim, agent_config, device)
    # The loop below runs max_steps + 1 steps, so every transition fits.
    online = TransitionBuffer(obs_dim, action_dim, config.max_steps + 1, device)

    human = None
    if human_config.corrector:
        corrector = RLPDAgent.from_checkpoint(human_config.corrector, device)
        if (corrector.obs_dim, corrector.action_dim) != (obs_dim, action_dim):
            raise ValueError(f"{human_config.corrector} was trained on a different task")
        value = None
        if human_config.gate == "progress":
            reference = RLPDAgent.from_checkpoint(human_config.progress_reference, device)
            value = critic_value(reference)
        human = SyntheticHuman(human_config, corrector, np.random.default_rng(config.seed), value)
        human.reset()
    # Q-OIL's D_int: the intervention transitions, sampled for the BC term.
    interventions = None
    if agent_config.bc_weight > 0:
        interventions = TransitionBuffer(obs_dim, action_dim, config.max_steps + 1, device)

    demos_per_minibatch = int(config.batch_size * config.offline_ratio)
    online_per_minibatch = config.batch_size - demos_per_minibatch
    use_demos = demos_per_minibatch > 0 or config.pretrain_steps > 0
    demos = None
    if use_demos:
        demos = load_demo_buffer(dataset_path, config.success_reward, device, config.num_demos)

    def sample_batches(num_demos: int, num_online: int, num_minibatches: int):
        """Minibatches for the critic and, for the BC term, one batch of
        intervention transitions (None without BC)."""
        batch = sample_mixed(demos, online, num_demos, num_online, num_minibatches)
        intervention_batch = None
        if interventions is not None:
            intervention_batch = interventions.split(interventions.sample_rows((config.batch_size,)))
        return batch, intervention_batch

    def make_update(num_demos: int, num_online: int):
        def step():
            return agent.update(*sample_batches(num_demos, num_online, config.utd_ratio))

        return capture_cuda_graph(step) if use_cuda_graph else step

    def actor_gradient_norms():
        # Run outside the CUDA graph, on a fresh batch like the actor's.
        batch, intervention_batch = sample_batches(demos_per_minibatch, online_per_minibatch, 1)
        return agent.actor_gradient_norms(batch.observations[0], intervention_batch)

    if config.pretrain_steps > 0:
        pretrain_update = make_update(config.batch_size, 0)
        for i in tqdm.tqdm(range(config.pretrain_steps), smoothing=0.1, disable=not config.tqdm):
            update_info = pretrain_update()
            if i % config.log_interval == 0:
                log_scalars(writer, "offline-training", update_info, i)
            if i % config.eval_interval == 0:
                eval_metrics = evaluate(agent, eval_env, config.eval_episodes)
                log_scalars(writer, "offline-evaluation", eval_metrics, i)

    # Built at the first update, once the buffer has data: recording a CUDA
    # graph runs real updates.
    online_update = None
    buffer_rows_saved = 0
    episode_intervention_steps = total_intervention_steps = 0
    # Per action dimension, how many intervention actions so far are within
    # bc_action_eps of +-1, where BC clamps them (see Actor.log_prob).
    clamped_intervention_actions = np.zeros(action_dim)
    clamp_limit = np.float32(1.0 - agent_config.bc_action_eps)
    # Shown at the end of the progress bar: the latest evaluation's success
    # rate and the success rate of the last 20 training episodes.
    progress_stats = {}
    recent_successes = collections.deque(maxlen=20)
    progress = tqdm.tqdm(range(config.max_steps + 1), smoothing=0.1, disable=not config.tqdm)
    for i in progress:
        step = i + config.pretrain_steps
        human_action = human.act(observation) if human is not None else None
        intervened = human_action is not None
        if intervened:
            action = human_action
        elif i < config.start_training:
            action = env.action_space.sample()
        else:
            action = agent.act(observation)

        # Launched before env.step, which runs on the CPU while the GPU
        # updates (the update is asynchronous; the next agent.act waits for
        # it). It therefore trains on transitions up to the previous step.
        # Running them in turn instead made each step about 27% slower.
        if i >= config.start_training:
            if online_update is None:
                online_update = make_update(demos_per_minibatch, online_per_minibatch)
            update_info = online_update()

        next_observation, reward, terminated, truncated, info = env.step(action)
        # Only true terminations stop bootstrapping; time-limit truncations do not.
        transition = (observation, action, reward, 0.0 if terminated else 1.0, next_observation)
        online.add(*transition, intervention=intervened)
        if intervened and interventions is not None:
            interventions.add(*transition, intervention=True)
        if intervened:
            episode_intervention_steps += 1
            clamped_intervention_actions += np.abs(action) >= clamp_limit
        observation = next_observation

        if terminated or truncated:
            observation, _ = env.reset()
            episode = info["episode"]
            episode_metrics = {
                "return": episode["r"],
                "length": episode["l"],
                "time": episode["t"],
                "success": info["is_success"],
            }
            if human is not None:
                human.reset()
                total_intervention_steps += episode_intervention_steps
                episode_metrics.update(
                    intervention_steps=episode_intervention_steps,
                    intervention_fraction=episode_intervention_steps / episode["l"],
                    total_intervention_steps=total_intervention_steps,
                )
                if total_intervention_steps > 0:
                    clamped = clamped_intervention_actions / total_intervention_steps
                    episode_metrics["intervention_clamped_fraction"] = clamped.mean()
                    for dim, fraction in enumerate(clamped):
                        episode_metrics[f"intervention_clamped_fraction_dim{dim}"] = fraction
            episode_intervention_steps = 0
            log_scalars(writer, "training", episode_metrics, step)
            recent_successes.append(info["is_success"])
            progress_stats["train_success"] = f"{np.mean(recent_successes):.2f}"
            progress.set_postfix(progress_stats, refresh=False)

        # Logged after env.step: reading the update's values waits for it.
        if i >= config.start_training and i % config.log_interval == 0:
            log_scalars(writer, "training", update_info, step)
            log_scalars(writer, "training", actor_gradient_norms(), step)

        if i % config.eval_interval == 0:
            video_path = None
            if video_dir is not None:
                video_path = os.path.join(video_dir, f"evaluation_{step:07d}.mp4")
            eval_metrics = evaluate(agent, eval_env, config.eval_episodes, video_path)
            log_scalars(writer, "evaluation", eval_metrics, step)
            progress_stats["eval_success"] = f"{eval_metrics['success']:.2f}"
            progress.set_postfix(progress_stats, refresh=False)
        # Files are named by the step on TensorBoard's x-axis.
        if i % config.checkpoint_interval == 0:
            if config.checkpoint_model:
                torch.save(
                    {"step": step, **agent.checkpoint()},
                    os.path.join(checkpoint_dir, f"step_{step:07d}.pt"),
                )
            if config.checkpoint_buffer:
                # Only the rows added since the previous save: full copies at
                # every save would grow quadratically on disk. The online
                # buffer holds one row per step and never wraps, so
                # concatenating the files in step order rebuilds it.
                torch.save(
                    online.rows[buffer_rows_saved : len(online)].cpu(),
                    os.path.join(buffer_dir, f"step_{step:07d}.pt"),
                )
                buffer_rows_saved = len(online)

    writer.close()


if __name__ == "__main__":
    main()
