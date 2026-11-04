"""Trains RLPD on a Robomimic task.

Usage: python -m rlpd.train --dataset=lift/ph [--flag=value ...]
Every field of TrainConfig and AgentConfig is a flag; --help lists them.
"""

import argparse
import dataclasses
import os
import typing
from dataclasses import dataclass
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import tqdm
import wandb

from rlpd.agent import AgentConfig, RLPDAgent, capture_cuda_graph
from rlpd.data import TransitionBuffer, load_demo_buffer, sample_mixed
from rlpd.env import (
    DEFAULT_HORIZON,
    RobomimicEnv,
    download_dataset,
    read_env_args,
    seed_env,
)


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
    project_name: str = "rlpd"  # wandb project
    log_dir: str = "runs"
    save_video: bool = False  # log one evaluation episode per evaluation
    checkpoint_model: bool = False  # save the agent at each evaluation
    checkpoint_buffer: bool = False  # save the online buffer at each evaluation
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
        if cls is AgentConfig:
            values["hidden_dims"] = tuple(values["hidden_dims"])
        return cls(**values)

    return build(TrainConfig), build(AgentConfig)


def evaluate(
    agent: RLPDAgent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    """Runs the deterministic policy; optionally logs the first episode's video."""
    returns, lengths, successes, frames = [], [], [], []
    for episode in range(num_episodes):
        record = save_video and episode == 0
        observation, _ = env.reset()
        if record:
            frames.append(env.render())
        episode_return, length, done = 0.0, 0, False
        while not done:
            action = agent.act(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            length += 1
            done = terminated or truncated
            if record:
                frames.append(env.render())
        returns.append(episode_return)
        lengths.append(length)
        successes.append(info["is_success"])

    if save_video:
        # wandb expects (time, channel, height, width).
        video = np.moveaxis(np.stack(frames), -1, 1)
        wandb.log({"eval_video": wandb.Video(video, fps=20, format="mp4")}, commit=False)
    return {
        "return": float(np.mean(returns)),
        "length": float(np.mean(lengths)),
        "success": float(np.mean(successes)),
    }


def main():
    config, agent_config = parse_config()
    assert 0.0 <= config.offline_ratio < 1.0
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    use_cuda_graph = config.cuda_graph and device.type == "cuda"

    wandb.init(
        project=config.project_name,
        config={**dataclasses.asdict(config), **dataclasses.asdict(agent_config)},
    )
    log_dir = os.path.join(config.log_dir, f"{config.dataset.replace('/', '_')}_s{config.seed}")
    if config.checkpoint_model or config.checkpoint_buffer:
        os.makedirs(log_dir, exist_ok=True)

    dataset_path = download_dataset(config.dataset)
    env_args = read_env_args(dataset_path)
    env = RobomimicEnv(env_args, config.horizon, config.success_reward)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1)
    observation = seed_env(env, config.seed)
    eval_env = RobomimicEnv(
        env_args,
        config.horizon,
        config.success_reward,
        render_mode="rgb_array" if config.save_video else None,
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
            wandb.log({f"offline-training/{k}": v.item() for k, v in update_info.items()}, step=i)
        if i % config.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, config.eval_episodes)
            wandb.log({f"offline-evaluation/{k}": v for k, v in eval_info.items()}, step=i)

    online_update = None  # built at the first update, once the buffer has data
    for i in tqdm.tqdm(range(config.max_steps + 1), smoothing=0.1, disable=not config.tqdm):
        step = i + config.pretrain_steps
        if i < config.start_training:
            action = env.action_space.sample()
        else:
            action = agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        # Only true terminations stop bootstrapping; time-limit truncations do not.
        online.add(observation, action, reward, 0.0 if terminated else 1.0, next_observation)
        observation = next_observation

        if terminated or truncated:
            observation, _ = env.reset()
            episode = info["episode"]
            wandb.log(
                {
                    "training/return": episode["r"],
                    "training/length": episode["l"],
                    "training/time": episode["t"],
                    "training/success": float(info["is_success"]),
                },
                step=step,
            )

        if i >= config.start_training:
            if online_update is None:
                online_update = make_update(demos_per_minibatch, online_per_minibatch)
            update_info = online_update()
            if i % config.log_interval == 0:
                wandb.log({f"training/{k}": v.item() for k, v in update_info.items()}, step=step)

        if i % config.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, config.eval_episodes, config.save_video)
            wandb.log({f"evaluation/{k}": v for k, v in eval_info.items()}, step=step)
            if config.checkpoint_model:
                torch.save(
                    {"step": i, **agent.checkpoint()}, os.path.join(log_dir, "checkpoint.pt")
                )
            if config.checkpoint_buffer:
                torch.save(online.rows[: len(online)].cpu(), os.path.join(log_dir, "buffer.pt"))


if __name__ == "__main__":
    main()
