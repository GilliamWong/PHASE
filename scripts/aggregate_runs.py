"""Average success over training per experiment, across seeds, with 95% CIs.

Groups run folders by run name (the part before _<task>_<quality>_s<seed>_),
so all seeds of one experiment form one row. Each run's score is its success
averaged over training (Q-OIL's Fig. 10 metric): from the 100-episode
checkpoint re-evaluation (checkpoint_eval_100ep_seed1000.csv) when present,
else from training's own evaluations (--eval_episodes, 10 by default).
The CI is a t-interval over seeds.

Usage: python -m scripts.aggregate_runs [glob ...] [--csv out.csv]
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

NAME = re.compile(r"^(?P<name>.+)_(?P<task>lift|can|square)_(?P<quality>ph|mh)_s(?P<seed>\d+)_\d{8}-\d{6}$")


def run_score(run_dir):
    """(score, source) for a finished run, or None."""
    eval_csv = os.path.join(run_dir, "checkpoint_eval_100ep_seed1000.csv")
    if os.path.exists(eval_csv):
        rows = list(csv.DictReader(open(eval_csv)))
        if rows:
            return float(np.mean([float(r["success"]) for r in rows])), "100ep"
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    if "evaluation/success" not in acc.Tags()["scalars"]:
        return None
    return float(np.mean([e.value for e in acc.Scalars("evaluation/success")])), "10ep"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("globs", nargs="*", default=["runs/*"])
    parser.add_argument("--csv", help="also write the table here")
    args = parser.parse_args()

    groups = defaultdict(list)
    for pattern in args.globs:
        for run_dir in sorted(glob.glob(pattern)):
            m = NAME.match(os.path.basename(os.path.normpath(run_dir)))
            if not m or not os.path.isdir(run_dir):
                continue
            scored = run_score(run_dir)
            if scored:
                groups[(m["task"], m["name"])].append((int(m["seed"]), *scored))

    rows = []
    for (task, name), runs in sorted(groups.items()):
        scores = np.array([s for _, s, _ in runs])
        half = (stats.t.ppf(0.975, len(scores) - 1) * scores.std(ddof=1) / np.sqrt(len(scores))
                if len(scores) > 1 else float("nan"))
        rows.append({"task": task, "experiment": name, "seeds": len(scores),
                     "n_100ep": sum(src == "100ep" for *_, src in runs),
                     "mean": scores.mean(), "ci95": half, "min": scores.min(), "max": scores.max()})
    width = max(len(r["experiment"]) for r in rows)
    print(f"{'task':<6} {'experiment':<{width}} seeds 100ep   mean   ±95%    min    max")
    for r in rows:
        print(f"{r['task']:<6} {r['experiment']:<{width}} {r['seeds']:>5} {r['n_100ep']:>5} "
              f"{r['mean']:6.3f} {r['ci95']:6.3f} {r['min']:6.3f} {r['max']:6.3f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
