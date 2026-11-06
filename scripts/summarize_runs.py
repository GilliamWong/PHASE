"""Compares training runs from their TensorBoard logs, one row per run.

Columns:
- avg_success: evaluation success averaged over every evaluation in the run
  (Q-OIL's Fig. 10 metric, "success averaged over training");
- steps_to_80: first evaluation step with success >= 0.8 (none: "-");
- final: success at the last evaluation;
- int_frac: share of training steps under intervention;
- gap_int / gap_other: mean optimistic-minus-task critic gap on intervention
  and other transitions, over the second half of training (Q-OIL only);
- grad_q / grad_bc: mean gradient norms of the actor's Q and BC terms over
  the second half of training;
- clamped: share of intervention actions BC clamps at +-1.

Each evaluation uses --eval_episodes episodes (10 by default), so with one
seed only large differences mean anything.

Usage: python -m scripts.summarize_runs runs/pilot_*
"""

import argparse
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(run_dir: str) -> dict:
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags()["scalars"])

    def series(tag):
        if tag not in tags:
            return np.array([]), np.array([])
        events = acc.Scalars(tag)
        return np.array([e.step for e in events]), np.array([e.value for e in events])

    def second_half_mean(tag):
        steps, values = series(tag)
        if not len(values):
            return float("nan")
        return values[steps >= steps.max() / 2].mean()

    eval_steps, eval_success = series("evaluation/success")
    reached = eval_steps[eval_success >= 0.8]
    _, lengths = series("training/length")
    _, int_steps = series("training/intervention_steps")
    _, clamped = series("training/intervention_clamped_fraction")
    return {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "last_step": int(eval_steps.max()) if len(eval_steps) else 0,
        "avg_success": eval_success.mean() if len(eval_success) else float("nan"),
        "steps_to_80": f"{reached[0] // 1000}k" if len(reached) else "-",
        "final": eval_success[-1] if len(eval_success) else float("nan"),
        "int_frac": int_steps.sum() / lengths.sum() if len(int_steps) else float("nan"),
        "gap_int": second_half_mean("training/gap_intervention"),
        "gap_other": second_half_mean("training/gap_other"),
        "grad_q": second_half_mean("training/actor_grad_norm_q"),
        "grad_bc": second_half_mean("training/actor_grad_norm_bc"),
        "clamped": clamped[-1] if len(clamped) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    rows = [summarize(d) for d in args.run_dirs]
    width = max(len(r["run"]) for r in rows)
    columns = ["last_step", "avg_success", "steps_to_80", "final", "int_frac",
               "gap_int", "gap_other", "grad_q", "grad_bc", "clamped"]
    print(f"{'run':<{width}}  " + "  ".join(f"{c:>11}" for c in columns))
    for r in rows:
        cells = []
        for c in columns:
            v = r[c]
            if isinstance(v, str):
                cells.append(f"{v:>11}")
            elif isinstance(v, int):
                cells.append(f"{v:>11}")
            elif np.isnan(v):
                cells.append(f"{'-':>11}")
            else:
                cells.append(f"{v:>11.3g}")
        print(f"{r['run']:<{width}}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
