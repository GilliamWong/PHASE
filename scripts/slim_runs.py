"""Shrinks finished runs' checkpoints about 10x, in place, for storage.

Every checkpoint keeps the actor and temperature (all that evaluation,
scripts.evaluate_checkpoints and scripts.imitation_analysis need) and is
marked "actor_only". The checkpoints at --full_steps and the last one also
keep the critics (but not the optimizer state, which only resuming training
needs). Buffers and logs are untouched. Each file is replaced atomically, so
a reader never sees a partial checkpoint, and rerunning skips files already
slimmed. What was kept is written to <run>/slim.json.

This deletes data: use it on runs whose full checkpoints are not needed (or
are stored elsewhere). Do not use it on corrector runs that jobs still load
as --corrector or --progress_reference: those need the critics.

Usage: python -m scripts.slim_runs runs/<run> [runs/<run> ...] [--full_steps 50000 100000]
"""

import argparse
import glob
import json
import os
import re

import torch

from rlpd.agent import ACTOR_ONLY_PREFIXES

OPTIMIZER_KEYS = ("actor_optimizer", "critic_optimizer", "temp_optimizer")


def slim_checkpoint(path: str, keep_critics: bool) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("actor_only") or (keep_critics and not any(k in checkpoint for k in OPTIMIZER_KEYS)):
        return  # already slimmed
    slim = {k: v for k, v in checkpoint.items() if k not in OPTIMIZER_KEYS}
    if not keep_critics:
        slim["model"] = {k: v for k, v in checkpoint["model"].items() if k.startswith(ACTOR_ONLY_PREFIXES)}
        slim["actor_only"] = True
    torch.save(slim, path + ".tmp")
    os.replace(path + ".tmp", path)


def slim_run(run_dir: str, full_steps: set) -> None:
    paths = {int(re.search(r"step_(\d+)\.pt$", p).group(1)): p
             for p in glob.glob(os.path.join(run_dir, "checkpoints", "step_*.pt"))}
    if not paths:
        return
    keep_full = sorted((full_steps & set(paths)) | {max(paths)})
    for step, path in sorted(paths.items()):
        slim_checkpoint(path, keep_critics=step in keep_full)
    with open(os.path.join(run_dir, "slim.json"), "w") as f:
        json.dump({"actor_only_steps": sorted(set(paths) - set(keep_full)), "full_steps": keep_full,
                   "optimizer_state": "removed"}, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--full_steps", type=int, nargs="*", default=[50000, 100000])
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        slim_run(run_dir, set(args.full_steps))
        print("slimmed", run_dir, flush=True)


if __name__ == "__main__":
    main()
