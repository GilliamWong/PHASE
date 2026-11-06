"""Re-evaluates a run's saved checkpoints, e.g. to pick correctors by success rate.

Each checkpoint runs its deterministic policy (as in training's evaluation and
the corrector's takeovers) for --episodes episodes, all from the same sequence
of initial states (set by --seed), so differences between checkpoints are not
due to easier or harder starts. Checkpoints are evaluated on the CPU, one per
worker process at a time.

Results go to <run>/checkpoint_eval_<episodes>ep_seed<seed>.csv. Steps
already in that file are skipped, so rerunning with more --steps only adds
the new ones.

Usage: python -m scripts.evaluate_checkpoints runs/<run> [--steps 40000 42000 ...]
"""

import argparse
import csv
import dataclasses
import glob
import json
import math
import multiprocessing
import os
import re

import torch

from rlpd.agent import RLPDAgent
from rlpd.env import download_dataset, read_env_args, seed_env
from rlpd.train import TrainConfig, evaluate, make_env

FIELDS = ("step", "success", "success_se", "length", "return", "episodes", "seed")


def load_train_config(run_dir: str) -> TrainConfig:
    with open(os.path.join(run_dir, "config.json")) as f:
        saved = json.load(f)
    # Fields added after the run was made keep their defaults; only the task
    # (dataset, horizon, success_reward) matters here.
    names = {field.name for field in dataclasses.fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in saved.items() if k in names})


def evaluate_checkpoint(job) -> dict:
    run_dir, path, step, episodes, seed = job
    torch.set_num_threads(1)  # parallelism comes from the worker processes
    config = load_train_config(run_dir)
    env = make_env(read_env_args(download_dataset(config.dataset)), config)
    seed_env(env, seed)
    agent = RLPDAgent.from_checkpoint(path, torch.device("cpu"))
    metrics = evaluate(agent, env, episodes)
    success = float(metrics["success"])
    env.close()
    return {
        "step": step,
        "success": success,
        "success_se": math.sqrt(success * (1 - success) / episodes),
        "length": float(metrics["length"]),
        "return": float(metrics["return"]),
        "episodes": episodes,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, nargs="+", help="checkpoint steps; default all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    checkpoints = {}
    for path in glob.glob(os.path.join(args.run_dir, "checkpoints", "step_*.pt")):
        checkpoints[int(re.search(r"step_(\d+)\.pt$", path).group(1))] = path
    steps = sorted(checkpoints) if args.steps is None else sorted(args.steps)
    missing = [s for s in steps if s not in checkpoints]
    if missing:
        raise SystemExit(f"no checkpoint for steps {missing}")

    out_path = os.path.join(
        args.run_dir, f"checkpoint_eval_{args.episodes}ep_seed{args.seed}.csv"
    )
    rows = {}
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            rows = {int(r["step"]): r for r in csv.DictReader(f)}
    jobs = [
        (args.run_dir, checkpoints[s], s, args.episodes, args.seed)
        for s in steps
        if s not in rows
    ]
    print(f"evaluating {len(jobs)} checkpoints ({len(steps) - len(jobs)} already in {out_path})")

    with multiprocessing.Pool(args.workers) as pool:
        for result in pool.imap_unordered(evaluate_checkpoint, jobs):
            rows[result["step"]] = result
            print(f"  step {result['step']:>7}: success {result['success']:.2f}", flush=True)
            # Rewritten after each result, so an interrupted run keeps its progress.
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows[s] for s in sorted(rows))

    print(f"\n{'step':>7}  success  ± s.e.  length")
    for s in steps:
        r = rows[s]
        print(
            f"{s:>7}  {float(r['success']):7.2f}  {float(r['success_se']):6.2f}"
            f"  {float(r['length']):6.0f}"
        )


if __name__ == "__main__":
    main()
