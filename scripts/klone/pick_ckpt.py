"""Prints the checkpoint whose evaluation success is closest to a target."""
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run, target = sys.argv[1], float(sys.argv[2])
acc = EventAccumulator(run); acc.Reload()
evals = [(e.step, e.value) for e in acc.Scalars("evaluation/success") if e.step >= 10000]
ckpts = {int(p.split("_")[-1][:-3]): p for p in glob.glob(f"{run}/checkpoints/step_*.pt")}
evals = [(s, v) for s, v in evals if s in ckpts]
print("evals:", evals, file=sys.stderr)
step, v = min(evals, key=lambda sv: (abs(sv[1] - target), -sv[0]))
print(ckpts[step]); print(f"picked step {step} success {v}", file=sys.stderr)
