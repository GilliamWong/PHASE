"""Trains RLPD on a Robomimic task.

Usage: python -m rlpd.train --dataset=lift/ph [--flag=value ...]
Every field of TrainConfig and AgentConfig is a flag; --help lists them.
Each run writes TensorBoard logs, videos and checkpoints to its own folder
under --log_dir; view them with: tensorboard --logdir runs
"""

import argparse
import dataclasses
import json
import os
import time
import typing
from dataclasses import dataclass
from typing import Mapping

import gymnasium as gym
import imageio
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from rlpd.agent import AgentConfig, RLPDAgent, capture_cuda_graph
from rlpd.data import TransitionBuffer, load_demo_buffer, sample_mixed
from rlpd.env import DEFAULT_HORIZON, RobomimicEnv, download_dataset, read_env_args, seed_env


@dataclass
class TrainConfig:
    # Robomimic task/quality, e.g. lift/ph or can/mh. Downloaded on first use;
    # the env is built from the dataset's recorded env config.
    dataset: str = "lift/ph"
    offline_ratio: float = 0.5  # fraction of each batch from the demos; 0 is purely online
    success_reward: float = 10.0  # reward on the (terminal) success step
    horizon: int = DEFAULT_HORIZON  # max steps per episode
    seed: int = 42
    batch_size: int = 256  # per critic update
    utd_ratio: int = 20  # critic updates per environment step
    max_steps: int = 300_000  # environment steps
    start_training: int = 5_000  # random-action steps before updates begin
    pretrain_steps: int = 0  # demo-only updates before online training
    eval_interval: int = 5_000
    eval_episodes: int = 10
    log_interval: int = 1_000
    log_dir: str = "runs"
    run_name: str = ""  # optional label at the start of the run folder's name, e.g. qoil_b0.25
    save_video: bool = False  # save one evaluation episode per evaluation as an mp4
    # Save the agent at each evaluation, to checkpoints/step_<step>.pt.
    checkpoint_model: bool = False
    # Save the online transitions added since the previous save at each
    # evaluation, to buffer/step_<step>.pt.
    checkpoint_buffer: bool = False
    cuda_graph: bool = True  # replay updates as a CUDA graph (GPU only)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm: bool = True


def parse_config() -> typing.Tuple[TrainConfig, AgentConfig]:
    parser = argparse.ArgumentParser(description=__doc__)
    for cls in (TrainConfig, AgentConfig):
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

    return build(TrainConfig), build(AgentConfig)


def log_scalars(writer: SummaryWriter, prefix: str, metrics: Mapping, step: int):
    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", float(value), step)


def evaluate(
    agent: RLPDAgent,
    env: gym.Env,
    num_episodes: int,
    writer: SummaryWriter,
    prefix: str,
    step: int,
    video_dir: typing.Optional[str] = None,
):
    """Runs the deterministic policy and logs its mean return, length and
    success rate; with video_dir, also saves the first episode as an mp4."""
    returns, lengths, successes, frames = [], [], [], []
    for episode in range(num_episodes):
        record = video_dir is not None and episode == 0
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

    log_scalars(
        writer,
        prefix,
        {"return": np.mean(returns), "length": np.mean(lengths), "success": np.mean(successes)},
        step,
    )
    if frames:
        os.makedirs(video_dir, exist_ok=True)
        imageio.mimwrite(os.path.join(video_dir, f"{prefix}_{step:07d}.mp4"), frames, fps=20)


def main():
    config, agent_config = parse_config()
    assert 0.0 <= config.offline_ratio < 1.0
    torch.manual_seed(config.seed)
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
    full_config = {**dataclasses.asdict(config), **dataclasses.asdict(agent_config)}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=2)
    # Indented so TensorBoard's text tab shows it as a code block.
    writer.add_text("config", "    " + json.dumps(full_config, indent=2).replace("\n", "\n    "))
    video_dir = os.path.join(run_dir, "videos") if config.save_video else None

    dataset_path = download_dataset(config.dataset)
    env_args = read_env_args(dataset_path)
    # RecordEpisodeStatistics adds each finished episode's return, length and
    # duration to info["episode"].
    env = gym.wrappers.RecordEpisodeStatistics(
        RobomimicEnv(env_args, config.horizon, config.success_reward), buffer_length=1
    )
    observation = seed_env(env, config.seed)
    eval_env = gym.wrappers.RecordEpisodeStatistics(
        RobomimicEnv(
            env_args,
            config.horizon,
            config.success_reward,
            render_mode="rgb_array" if config.save_video else None,
        ),
        buffer_length=1,
    )
    seed_env(eval_env, config.seed + 42)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = RLPDAgent(obs_dim, action_dim, agent_config, device)
    # The loop below runs max_steps + 1 steps, so every transition fits.
    online = TransitionBuffer(obs_dim, action_dim, config.max_steps + 1, device)

    demos_per_minibatch = int(config.batch_size * config.offline_ratio)
    online_per_minibatch = config.batch_size - demos_per_minibatch
    use_demos = demos_per_minibatch > 0 or config.pretrain_steps > 0
    demos = load_demo_buffer(dataset_path, config.success_reward, device) if use_demos else None

    def make_update(num_demos: int, num_online: int):
        def step():
            batch = sample_mixed(demos, online, num_demos, num_online, config.utd_ratio)
            return agent.update(batch)

        return capture_cuda_graph(step) if use_cuda_graph else step

    if config.pretrain_steps > 0:
        pretrain_update = make_update(config.batch_size, 0)
        for i in tqdm.tqdm(range(config.pretrain_steps), smoothing=0.1, disable=not config.tqdm):
            update_info = pretrain_update()
            if i % config.log_interval == 0:
                log_scalars(writer, "offline-training", update_info, i)
            if i % config.eval_interval == 0:
                evaluate(agent, eval_env, config.eval_episodes, writer, "offline-evaluation", i)

    # Built at the first update, once the buffer has data: recording a CUDA
    # graph runs real updates.
    online_update = None
    buffer_rows_saved = 0
    for i in tqdm.tqdm(range(config.max_steps + 1), smoothing=0.1, disable=not config.tqdm):
        step = i + config.pretrain_steps
        action = env.action_space.sample() if i < config.start_training else agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        # Only true terminations stop bootstrapping; time-limit truncations do not.
        online.add(observation, action, reward, 0.0 if terminated else 1.0, next_observation)
        observation = next_observation

        if terminated or truncated:
            observation, _ = env.reset()
            episode = info["episode"]
            log_scalars(
                writer,
                "training",
                {
                    "return": episode["r"],
                    "length": episode["l"],
                    "time": episode["t"],
                    "success": info["is_success"],
                },
                step,
            )

        if i >= config.start_training:
            if online_update is None:
                online_update = make_update(demos_per_minibatch, online_per_minibatch)
            update_info = online_update()
            if i % config.log_interval == 0:
                log_scalars(writer, "training", update_info, step)

        if i % config.eval_interval == 0:
            evaluate(agent, eval_env, config.eval_episodes, writer, "evaluation", step, video_dir)
            # Files are named by the step on TensorBoard's x-axis, so each
            # checkpoint lines up with the evaluation logged just above.
            if config.checkpoint_model:
                torch.save(
                    {"step": step, **agent.checkpoint()},
                    os.path.join(checkpoint_dir, f"step_{step:07d}.pt"),
                )
            if config.checkpoint_buffer:
                # Only the rows added since the previous save: full copies at
                # every evaluation would grow quadratically on disk. The online
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
