# RaC-style human: results

Code: `rlpd/human.py` (`--style=rac`, `--end_after_takeover`); see `Baselines/RaC notes.md`.
Scripts: `scripts/klone/` (Hyak, `ckpt-all`).

## Pilot: Lift, 2026-11-20 (`pilot_lift_20261120.csv`)

Script: `scripts/klone/rac_lift.sbatch`.
- **Stand-in human.** We trained our own, because William's checkpoints weren't
  accessible yet. It is plain RLPD on 20 demos, and `pick_ckpt.py` chose the
  25k-step checkpoint, which scored 55% on one 20-episode eval.
- **When the human steps in.** A random gate: a 0.5% chance per step, no full
  takeovers, takeovers last 10–40 steps.
- **Run settings.** 30k steps, UTD 10, temperature 0.01 (fixed), an
  evaluation every 2.5k steps with 20 episodes each, seeds 42 and 43.

Success is averaged over the last 3 evaluations (25k, 27.5k, 30k):

| Human | Learner | s42 | s43 | Human steps by 30k (s42 / s43) |
|---|---|---|---|---|
| Corrector only (Q-OIL style) | HIL | 0.28 | 0.57 | 3591 / 3294 |
| Corrector only | BC+RL (λ=0.1) | 0.10 | 0.08 | 3580 / 3241 |
| RaC (recover → correct, Rule 2) | HIL | 0.73 | 0.60 | 5011 / 4720 |
| RaC | BC+RL (λ=0.1) | 0.57 | 0.30 | 4314 / 4346 |

Take this as a smoke test only:
- **The comparison isn't effort-matched.** Rule 2 ends episodes early, so the
  RaC human runs more episodes and takes over about 1.3–1.4x as many steps.
- **The stand-in human is noisy.** Its neighboring checkpoints scored 25% and
  10%.
- **It's small.** Two seeds, 20-episode evals (SE about 0.11), and Lift is
  close to a single subtask.

As in William's runs, BC+RL does worse than HIL on Lift.

## Cap100 replication: Lift + Can (in progress)

Script: `scripts/klone/rac_cap100.sbatch <lift|can>`. It matches William's
`f10_cap100_*` / `can_cap100_*` configs (HF `gilliamwong/qoil-runs`):
- **Stand-in human:** his corrector (Lift at 50k, Can at 75k).
- **Gate:** his progress gate, using the converged checkpoint as V (Lift 150k,
  Can 500k), with k=12, δ=-1.58 and α=0.599.
- **Takeovers:** 10–100 steps.
- **Run settings:** 100k steps, with an evaluation every 5k steps (10 episodes
  each).

The Q-OIL-style HIL and BC+RL baselines are William's existing 6-seed runs.
New runs:

| Run | Seeds | What it isolates |
|---|---|---|
| `repro_hil` | 42 | Sanity check that our branch reproduces William's HIL |
| `rac_hil` | 42–44 | Full RaC protocol, HIL learner |
| `rac_bcrl` | 42–44 | Full RaC protocol, BC+RL learner |
| `rule2_hil` | 42–44 | Rule 2 alone (no scripted recovery), so we can tell whether recovery or early termination drives any gain |
