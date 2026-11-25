# PACT for PHASE

This directory implements progress estimation, segment localization, and
episode replay annotation from [PACT](https://arxiv.org/html/2606.03949v2) for PHASE's
Robomimic benchmark. See also the
[authors' implementation](https://github.com/Liuzeyi25/HILRL-A1X).

## Changes from the original baseline

- **Observations and model.** The original progress estimator uses two camera
  streams with frozen ResNet-18 encoders plus a learned robot-state difference
  embedding. This port feeds PHASE's flattened object and robot observations
  directly into the 128/64-unit regression head. It retains LayerNorm, ReLU,
  and the sigmoid output.
- **Demonstration data.** The original uses real-robot demonstrations. The
  loader here accepts selected Robomimic trajectories and cuts each successful
  trajectory at its first positive reward, consistent with PHASE's success
  termination. The target helper uses the paper's elapsed-step fraction and
  discounted unit terminal reward. The loss retains the paper's 0.1 weight for
  terminal return supervision. The caller supplies the terminal mask.
- **Segment boundaries.** The detector uses the appendix's nearest-anomaly
  convention. It groups consecutive regression flags as one anomaly and
  searches only the caller-specified learner-controlled prefix. It uses the
  published drop margin, lookahead, recovery length, and exponential weighting.
  A single-step segment gets weight 1.
- **Interventions.** The authors' experiments used a human operator. Our
  simulated collector uses the same frozen teammate correctors and random
  gate as RLIF, with deterministic corrective actions, 0.5 takeover
  probability when eligible, and four-step takeovers. PACT stores the
  environment task rewards during collection.
- **Replay integration.** The original stores segment IDs in its online
  replay. This port first builds episode-local links, with one preference
  tuple per takeover onset. Critic links are added only in successful
  episodes and only within the uninterrupted learner-controlled prefix
  before each takeover. Preference tuples from failed episodes are kept
  available for actor guidance. Online replay resolves episode-local links
  to shared tuple references in a bounded, in-memory transition ring. A
  separate bounded actor sampling window can evict a tuple without dropping
  its critic links. This port uses CPU-side storage and transfers sampled
  batches to the learner device.
- **Preference losses.** The learner implements the directional target
  correction and the bounded-mean attraction and inverse-distance repulsion
  from equations 7–11. The target comparison uses the smaller of the two
  target critic values for each action. The actor preference weight is 0.2;
  the configurable denominator epsilon currently defaults to 0.001.
- **Learner backbone.** The original uses HIL-SERL with visual encoders.
  This port uses a standalone PyTorch SAC learner with low-dimensional MLPs,
  twin critics, and the paper's discount of 0.98. It reuses the repository's
  squashed Gaussian actor module and replay batch type. The base update uses
  an entropy-adjusted target and the minimum of the two critic estimates.

## Files and verification

- `progress.py` loads the observation sequences used for progress supervision,
  builds both target signals, and defines the estimator and its training loss.
- `train_progress.py` deterministically selects 20 successful trajectories per
  task, fits the estimator, and saves its weights, settings, and demo IDs.
  The SAC driver will use those same IDs to populate its initial demo replay.
- `train.py` loads the fitted progress model and exact demonstration list,
  alternates episode collection and updates with a 50/50 demo-to-online mix,
  and samples 16 preferences per actor update when available. It writes
  TensorBoard metrics and learner weights at evaluation intervals.
- `segments.py` maps a progress trace and takeover index to an inclusive
  learner segment and provides the weights for its future critic correction.
- `agent.py` defines the SAC training step, from bootstrapping masked
  next-state values through updating both critics, policy, entropy temperature,
  and target critics. It takes a single replay minibatch.
- `replay.py` turns aligned episode observations, executed actions, control
  flags and resampled policy actions into preference tuples and weighted
  transition links. This lets multiple replay rows refer to a single
  counterfactual comparison at takeover.
- `collect.py` loads the frozen corrector, rolls out the learner and gate,
  resamples the learner action at takeover onset, and sends the finished
  episode and its annotations to replay.
- `__init__.py` marks the standalone PACT package.
- `tests/test_pact_progress.py` verifies the target values, gradient flow,
  regression detection, recovery boundaries, and correction weights.

From the repository root, run
`python -m pytest tests/test_pact_progress.py tests/test_pact_agent.py tests/test_pact_replay.py`.
To fit the progress model, run the command below from the repository root.
Set `EPOCHS` and `TERMINAL_STEPS` to the values for the experiment. The
terminal window determines how many final transitions receive auxiliary
return supervision. The command defaults to CPU and accepts `--device cuda`.
Use a separate output file for each task.

```bash
python -m pact.train_progress --dataset lift/ph --output runs/pact/lift_progress.pt --epochs "$EPOCHS" --terminal-steps "$TERMINAL_STEPS"
python -m pact.train_progress --dataset can/ph --output runs/pact/can_progress.pt --epochs "$EPOCHS" --terminal-steps "$TERMINAL_STEPS"
```

## Train SAC with PACT

Follow the [RLIF checkpoint download instructions](../rlif/README.md) to set
`$lift` and `$can` and download the frozen correctors. Set `EPISODES`,
`UPDATES_PER_EPISODE` and `EVAL_EVERY` to the run budget. The PHASE port
updates after each completed episode, whereas the original uses concurrent
actor and learner processes. The driver uses a unit sparse task reward and
deterministic autonomous evaluation. Saved agent files contain model weights
for evaluation.

```bash
python -m pact.train --dataset lift/ph --progress-checkpoint runs/pact/lift_progress.pt --corrector-checkpoint "$lift/checkpoints/step_0064000.pt" --output runs/pact/lift --episodes "$EPISODES" --updates-per-episode "$UPDATES_PER_EPISODE" --eval-every "$EVAL_EVERY"
python -m pact.train --dataset can/ph --progress-checkpoint runs/pact/can_progress.pt --corrector-checkpoint "$can/checkpoints/step_0075000.pt" --output runs/pact/can --episodes "$EPISODES" --updates-per-episode "$UPDATES_PER_EPISODE" --eval-every "$EVAL_EVERY"
```
