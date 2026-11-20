# RaC: notes and how we use it as a baseline

Hu et al., "RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and
Correction", Sep 2025. arXiv 2509.07953 (https://arxiv.org/abs/2509.07953);
project page https://rac-scaling-robot.github.io. No public code as of
2026-11-20.

## Summary

**Problem.** Imitation policies for long, multi-step tasks fail partway through
and cannot get unstuck. Collecting more full demos scales poorly.

**Method.** RaC is a human-in-the-loop data-collection protocol that runs after
ordinary imitation pre-training, over several rounds:

1. Roll out the current policy. A human takes over when failure looks imminent.
2. **Rule 1: recover, then correct.** Each takeover has two parts:
   - *Recovery*: the human moves the robot back to a familiar,
     in-distribution state. A gripper-visitation heatmap built from the Round 0
     demos shows them where "familiar" is.
   - *Correction*: the human then finishes the current subtask.
3. **Rule 2.** The episode ends as soon as the intervention is over.
4. Retrain with **plain BC** on the demos, all intervention segments, and the
   policy's own clean successes. Recovery and correction are weighted equally.
   There is no RL and no reweighting.

**How the split is made.** The operator makes it while following the protocol.
Afterwards, for analysis only (Sec. 5.4), each intervention frame is labeled by
where it lands: *recovery* if its state falls within the demo states before the
takeover point, *correction* if it falls within the demo states after it. The
training loss does not use these labels.
TODO: check Sec. 5.4 for how "within the distribution" is computed.

**Setup.**
- Two-arm robot (xArm-7) with a ~368M-parameter vision flow-matching policy
  (MM-DiT).
- Real tasks: shirt-hanging, airtight lid sealing, takeout-box packing. Plus one
  custom simulated assembly task (not robomimic).
- 6-10 rounds. 60 eval trials on the real robot, 100 in sim.
- Human effort is charged as the full length of every intervention.

**Results (as reported).**
- Beats HG-DAgger-style interventions and batched full demos, with a steeper
  scaling curve.
- At least 2x data efficiency over batched demos on shirt-hanging.
- 78% shirt-hanging success from about 5 hours of data, versus about 75% from
  about 89 hours for ALOHA Unleashed. That comparison crosses papers and
  setups, so it is approximate.
- Among successful rollouts, the number of recoveries correlates linearly with
  success.

**Why they say it works.**
- Recovering is easier to learn than doing the task: many past states are
  valid targets (a verification-generation gap).
- A policy that can recover gets to retry.
- RaC data has a recovery:correction ratio of about 1:1 to 1:2, versus 1:3 to
  1:10 for ordinary intervention data.

**Limitations.**
- Batched demos were compared on only one task.
- Small set of tasks.
- Effort is counted in frames only.
- The policy class, vision inputs and bimanual setup do not carry over to ours.

## Relevance to PHASE

- **Same split, different use.** RaC clones both halves of an intervention.
  PHASE clones only the recovery half and uses the states the correction
  reaches as shaping targets. RaC is essentially PHASE's "BC-all" baseline,
  applied to well-structured intervention data.
- **Opposite intuitions.** RaC says recovery is *easy to imitate*. PHASE says
  recovery is the part worth cloning because RL cannot *find* it by
  exploration. Both can be true.
- **Labeling.** RaC's after-the-fact labeling is close to PHASE's
  proximity-to-D_good rule.

## Implementation here (branch `rac`)

There is no public code, and RaC's contribution is the intervention protocol,
so we reimplement the protocol inside our RLPD setup, not RaC's learner.

- `--style=rac` (`rlpd/human.py`). At each takeover, the synthetic human first
  runs a scripted recovery. It moves the gripper toward the nearest demo state,
  measured as the eef-to-object offset, and uses that demo's gripper command.
  Recovery ends within `--recovery_tolerance` (2 cm) or after
  `--max_recovery_steps` (30). Then the corrector checkpoint acts for the
  sampled takeover length.
- `--end_after_takeover` is Rule 2. The episode ends when the takeover hands
  back, as a time-limit truncation, so the critic still bootstraps. It needs
  `--full_takeover_prob=0`.
- Each online step's phase (0 learner, 1 recovery, 2 correction) is saved to
  `phases.npy` in the run folder; row i matches online buffer row i. These are
  ground-truth labels for checking PHASE's labeler.
- `recovery_steps` per episode is logged to TensorBoard.
- Tests: `tests/test_rac.py`.

Example (HIL learner, Can). Add `--bc_weight=0.1` for BC+RL, the closest RL
analog of RaC's clone-everything objective:

```bash
python -m rlpd.train --dataset=can/ph --num_demos=20 --utd_ratio=10 \
  --init_temperature=0.01 --temp_lr=0 \
  --corrector=runs/corrector200_can_ph_s42_20261106-074707/checkpoints/step_0075000.pt \
  --gate=progress --intervention_rate=0 --progress_threshold=-1.58 \
  --progress_reference=runs/corrector200_can_ph_s42_20261106-074707/checkpoints/step_0500000.pt \
  --style=rac --full_takeover_prob=0 --end_after_takeover --run_name=rac_hil
```

**Caveats.**
- Not yet run in the simulator; only unit-tested.
- Recovery goes to the *nearest* demo state, not specifically an earlier point
  in the demo.
- The observation layout is assumed: object position in the first 3 entries,
  eef position 9 from the end. Verified for Lift; assumed for Can and Square.
- Scripting a recovery phase builds PHASE's premise into the synthetic human.
  Use this human to test PHASE's labeler and as one of several human models,
  not as the headline comparison.
- Use Can and Square: Lift is essentially a single subtask.
