"""Train the low-dimensional PACT baseline on Robomimic Lift or Can."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pact.agent import SACAgent, SACConfig
from pact.collect import collect_episode, random_gate
from pact.progress import ProgressModel
from pact.replay import OnlineReplay
from rlif.data import Batch, load_demo_buffer
from rlif.train import evaluate
from rlpd.env import DEFAULT_HORIZON, RobomimicEnv, download_dataset, read_env_args


def update_from_replay(agent, demos, online, batch_size, rng):
    """Mix equal demo and online halves and sample 16 actor preferences."""
    half = batch_size // 2
    offline = demos.split(demos.sample_rows((half,)))
    live, correction = online.sample(half, rng, row_offset=half)
    batch = Batch(*(torch.cat((demo, row)) for demo, row in zip(offline, live)))
    return agent.update(
        batch,
        actor_preferences=online.sample_actor_preferences(16, rng),
        critic_preferences=correction,
    )


class EvaluationPolicy:
    """Expose the bounded mean action to the shared autonomous evaluator."""

    def __init__(self, actor):
        self.actor = actor

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        device = next(self.actor.parameters()).device
        state = torch.as_tensor(observation, dtype=torch.float32, device=device)[None]
        return self.actor.mode(state)[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("lift/ph", "can/ph"))
    parser.add_argument("--progress-checkpoint", required=True)
    parser.add_argument("--corrector-checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=int)
    parser.add_argument("--updates-per-episode", required=True, type=int)
    parser.add_argument("--eval-every", required=True, type=int)
    parser.add_argument("--eval-episodes", default=20, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--replay-capacity", default=100_000, type=int)
    parser.add_argument("--preference-capacity", default=10_000, type=int)
    parser.add_argument("--horizon", default=DEFAULT_HORIZON, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    counts = (args.episodes, args.updates_per_episode, args.eval_every, args.eval_episodes,
              args.replay_capacity, args.preference_capacity, args.horizon, args.batch_size)
    if min(counts) < 1 or args.batch_size % 2:
        parser.error("counts must be positive and batch size must be even")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    torch.manual_seed(args.seed)
    gate_rng, replay_rng = (np.random.default_rng(seed) for seed in
                            np.random.SeedSequence(args.seed).spawn(2))
    device = torch.device(args.device)
    checkpoint = torch.load(args.progress_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["settings"]["dataset"] != args.dataset:
        parser.error("progress checkpoint belongs to a different dataset")
    names = checkpoint["demo_names"]
    if len(names) != 20 or len(set(names)) != 20:
        parser.error("progress checkpoint must identify 20 distinct demonstrations")

    path = download_dataset(args.dataset)
    env_args = read_env_args(path)
    with ExitStack() as stack:
        env = RobomimicEnv(env_args, horizon=args.horizon, success_reward=1.0)
        stack.callback(env.close)
        eval_env = gym.wrappers.RecordEpisodeStatistics(
            RobomimicEnv(env_args, horizon=args.horizon, success_reward=1.0)
        )
        stack.callback(eval_env.close)
        env.reset(seed=args.seed)
        eval_env.reset(seed=args.seed + 10_000)
        obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
        if checkpoint["observation_dim"] != obs_dim:
            parser.error("progress checkpoint observation dimensions do not match")
        progress = ProgressModel(obs_dim).to(device).eval()
        progress.load_state_dict(checkpoint["model"])
        demos = load_demo_buffer(path, success_reward=1.0, device=device, demo_names=names)
        agent = SACAgent(obs_dim, action_dim, SACConfig(), device)
        gate = random_gate(args.corrector_checkpoint, obs_dim, action_dim, device, gate_rng)
        online = OnlineReplay(args.replay_capacity, args.preference_capacity, device)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(
            {**vars(args), "output": str(args.output), "demo_names": names,
             "sac": asdict(agent.config), "offline_ratio": 0.5, "actor_preference_batch_size": 16},
            indent=2,
        ), encoding="utf-8")
        writer = stack.enter_context(SummaryWriter(str(args.output)))
        for episode in range(1, args.episodes + 1):
            collection = collect_episode(env, agent, gate, progress, online)
            for key, value in collection.items():
                writer.add_scalar(f"collection/{key}", value, episode)
            for _ in range(args.updates_per_episode):
                metrics = update_from_replay(agent, demos, online, args.batch_size, replay_rng)
            for key, value in metrics.items():
                writer.add_scalar(f"training/{key}", value.item(), episode)
            if episode % args.eval_every == 0 or episode == args.episodes:
                result = evaluate(EvaluationPolicy(agent.actor), eval_env, args.eval_episodes, True)
                for key, value in result.items():
                    writer.add_scalar(f"eval/{key}", value, episode)
                print(f"episode {episode}: {result}", flush=True)
                torch.save({"model": agent.state_dict(), "episode": episode},
                           args.output / f"agent_{episode:06d}.pt")


if __name__ == "__main__":
    main()
