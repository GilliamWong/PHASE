"""Tables and plots for the RaC-human experiment (Baselines/rac_results).

Compares our runs (runs_cap100/<task>/<name>_<task>_ph_s<seed>_*) with
William's matching cap100 runs (will_runs/<prefix><name>_*), read from their
TensorBoard logs.

Per group (mean +- standard error over seeds):
- avg: evaluation success averaged over every evaluation (Q-OIL's Fig. 10
  metric);
- final: success averaged over the last 3 evaluations;
- human: total steps the synthetic human controlled;
Also prints success at matched human-effort budgets (--budgets).

Usage:
  python scripts/rac_results.py --ours runs_cap100 runs_cap100_long --william will_runs \
      --out Baselines/rac_results
"""

import argparse
import collections
import glob
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WILLIAM_PREFIX = {"lift": "f10_cap100_", "can": "can_cap100_"}
ORDER = ["W:hil", "W:bcrl", "repro_hil", "long_hil", "rac_hil", "rac_bcrl", "rule2_hil"]
LABEL = {
    "W:hil": "William: corrector human, HIL",
    "W:bcrl": "William: corrector human, BC+RL",
    "repro_hil": "Ours: corrector human, HIL (repro)",
    "rac_hil": "RaC human, HIL",
    "rac_bcrl": "RaC human, BC+RL",
    "rule2_hil": "Rule 2 only, HIL",
    "long_hil": "Ours: corrector human, HIL, 150k steps",
}


def load(run_dir):
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags()["scalars"])

    def series(tag):
        if tag not in tags:
            return {}
        return {e.step: e.value for e in acc.Scalars(tag)}

    return {
        "success": series("evaluation/success"),
        "human": series("training/total_intervention_steps"),
    }


def groups_for(task, ours, william):
    groups = collections.defaultdict(list)
    for d in sorted(d for o in ours for d in glob.glob(os.path.join(o, task, "*"))):
        name = os.path.basename(d).split(f"_{task}_ph_s")[0]
        groups[name].append(load(d))
    prefix = WILLIAM_PREFIX[task]
    for d in sorted(glob.glob(os.path.join(william, prefix + "*"))):
        name = os.path.basename(d)[len(prefix):].split("_")[0]
        groups["W:" + name].append(load(d))
    return groups


def common_steps(runs):
    steps = set(runs[0]["success"])
    for r in runs[1:]:
        steps &= set(r["success"])
    return sorted(steps)


def mean_se(x):
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return x.mean(), se


def summarize(groups, max_step):
    rows = []
    for name in ORDER:
        runs = groups.get(name)
        if not runs:
            continue
        steps = [s for s in common_steps(runs) if s <= max_step]
        avg = [np.mean([r["success"][s] for s in steps]) for r in runs]
        final = [np.mean([r["success"][s] for s in steps[-3:]]) for r in runs]
        human = [max([v for s, v in r["human"].items() if s <= max_step], default=0) for r in runs]
        rows.append((name, len(runs), steps[-1], mean_se(avg), mean_se(final), mean_se(human)))
    return rows


def fmt(m, se, digits=2):
    if np.isnan(se):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {se:.{digits}f}"


def table(task, rows):
    lines = [
        f"### {task.capitalize()}",
        "",
        "| Run | Seeds | Up to step | Avg success | Final (last 3 evals) | Human steps |",
        "|---|---|---|---|---|---|",
    ]
    for name, n, last, avg, final, human in rows:
        lines.append(
            f"| {LABEL[name]} | {n} | {last // 1000}k | {fmt(*avg)} | {fmt(*final)} "
            f"| {fmt(human[0], human[1], 0)} |"
        )
    return "\n".join(lines)


def matched_effort(task, groups, budget):
    """Success over the last 3 evaluations reached within `budget` human steps."""
    lines = [f"{task}, human steps <= {budget}:"]
    for name in ORDER:
        runs = groups.get(name)
        if not runs:
            continue
        vals = []
        for r in runs:
            within = [s for s in sorted(r["success"])
                      if max([v for s2, v in r["human"].items() if s2 <= s], default=0) <= budget]
            vals.append(np.mean([r["success"][s] for s in within[-3:]]))
        lines.append(f"  {LABEL[name]:38s} {fmt(*mean_se(vals))}")
    return "\n".join(lines)


def plot(task, groups, max_step, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name in ORDER:
        runs = groups.get(name)
        if not runs:
            continue
        steps = [s for s in common_steps(runs) if s <= max_step]
        y = np.array([[r["success"][s] for s in steps] for r in runs])
        m = y.mean(0)
        se = y.std(0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(m)
        x = np.array(steps) / 1000
        style = "--" if name.startswith("W:") else "-"
        ax1.plot(x, m, style, label=f"{LABEL[name]} (n={len(runs)})")
        ax1.fill_between(x, m - se, m + se, alpha=0.15)
        h = np.array([[max([v for s2, v in r["human"].items() if s2 <= s], default=0)
                       for s in steps] for r in runs]).mean(0)
        ax2.plot(h / 1000, m, style, label=LABEL[name])
    ax1.set(xlabel="environment steps (k)", ylabel="eval success", title=f"{task}: success vs. steps", ylim=(0, 1.02))
    ax2.set(xlabel="cumulative human steps (k)", ylabel="eval success", title=f"{task}: success vs. human effort", ylim=(0, 1.02))
    ax1.legend(fontsize=7, loc="lower right")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", nargs="+", default=["runs_cap100", "runs_cap100_long"])
    parser.add_argument("--william", default="will_runs")
    parser.add_argument("--out", default="Baselines/rac_results")
    parser.add_argument("--max_step", type=int, default=100000)
    parser.add_argument("--budgets", type=int, nargs="*", default=[10000, 18000, 20000])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tables = []
    for task in ["lift", "can"]:
        groups = groups_for(task, args.ours, args.william)
        if not groups:
            continue
        rows = summarize(groups, args.max_step)
        tables.append(table(task, rows))
        for budget in args.budgets:
            print(matched_effort(task, groups, budget))
        plot(task, groups, 10**9, os.path.join(args.out, f"cap100_{task}.png"))
    text = "\n\n".join(tables)
    with open(os.path.join(args.out, "cap100_tables.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
