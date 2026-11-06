"""Measures how far a run's policy moves toward the human's actions.

At a fixed sample of the states where the synthetic human acted (read from
the run's saved replay buffer), compares each checkpoint's mean action with
the corrector's action and with a converged expert's action:

- d_corrector: mean L2 distance between the policy's and the corrector's actions;
- d_expert: the same against the expert (a converged run's checkpoint).

A policy that imitates the corrector gets closer to it than to the expert
(d_expert - d_corrector > 0); one that only learns the task moves toward the
expert. Needs --checkpoint_model and --checkpoint_buffer runs with
interventions. Writes <run>/imitation.csv and prints one line per run.

Usage: python -m scripts.imitation_analysis runs/f10_full_hil_* --expert <ckpt>
"""

import argparse
import csv
import glob
import multiprocessing
import os

import numpy as np
import torch

from rlpd.agent import RLPDAgent


def intervention_states(run_dir, obs_dim, action_dim, num_states, seed):
    rows = torch.cat([torch.load(p) for p in sorted(glob.glob(os.path.join(run_dir, "buffer", "step_*.pt")))])
    flag = rows[:, obs_dim + action_dim + 2]
    rows = rows[flag == 1]
    if len(rows) == 0:
        return None, None
    pick = np.random.default_rng(seed).choice(len(rows), min(num_states, len(rows)), replace=False)
    rows = rows[torch.as_tensor(pick)]
    return rows[:, :obs_dim], rows[:, obs_dim:obs_dim + action_dim]


@torch.no_grad()
def analyze(args):
    run_dir, expert_path, num_states, every, seed = args
    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    checkpoints = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "step_*.pt")))
    if not checkpoints:
        return run_dir, "no checkpoints"
    first = RLPDAgent.from_checkpoint(checkpoints[0], cpu)
    states, human_actions = intervention_states(run_dir, first.obs_dim, first.action_dim, num_states, seed)
    if states is None or human_actions is None:
        return run_dir, "no intervention transitions"
    expert_actions = RLPDAgent.from_checkpoint(expert_path, cpu).actor.mode(states)
    out = []
    for path in checkpoints:
        step = int(os.path.basename(path)[5:12])
        if step % every:
            continue
        policy = RLPDAgent.from_checkpoint(path, cpu).actor.mode(states)
        out.append((step,
                    (policy - human_actions).norm(dim=1).mean().item(),
                    (policy - expert_actions).norm(dim=1).mean().item()))
    with open(os.path.join(run_dir, "imitation.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "d_corrector", "d_expert"])
        writer.writerows(out)
    half = [r for r in out if r[0] >= out[-1][0] / 2]
    return run_dir, (f"{len(states)} states; second half: d_corrector {np.mean([r[1] for r in half]):.3f}, "
                     f"d_expert {np.mean([r[2] for r in half]):.3f}, "
                     f"expert - corrector {np.mean([r[2] - r[1] for r in half]):+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--expert", required=True, help="checkpoint of a converged policy")
    parser.add_argument("--states", type=int, default=2000)
    parser.add_argument("--every", type=int, default=10_000, help="checkpoint spacing to evaluate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [(d, args.expert, args.states, args.every, args.seed) for d in args.run_dirs]
    with multiprocessing.Pool(args.workers) as pool:
        for run_dir, line in pool.imap(analyze, jobs):
            print(f"{os.path.basename(os.path.normpath(run_dir))}: {line}", flush=True)


if __name__ == "__main__":
    main()
