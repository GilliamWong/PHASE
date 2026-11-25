# RLIF baseline

Reproduces RLIF: Interactive Imitation Learning as Reinforcement Learning

## Learning signal

RLIF ignores the environment's task reward during training. When a takeover
starts at state `s_t`, the policy transition `(s_{t-1}, a_{t-1}, s_t)` receives
reward `-1`; every other transition receives `0`. Corrective actions are kept
in ordinary replay but are not behavior-cloned. Autonomous evaluation still
uses the environment's task reward and success condition.

## Intervention modes

- `official_random`: random gate from the RLIF experiments, with the intended
  fixed-duration takeover.
- `official_value`: reference-Q comparison with the 0.95/0.05 noisy
  intervention probabilities.
- `phase_random`: random synthetic human from `origin/qoil`.
- `phase_progress`: progress-stall synthetic human from `origin/qoil`.

## Robomimic entry point

All gates require a corrector checkpoint. The value and progress gates also
require a reference checkpoint. Both may point to checkpoints produced by the
existing `rlpd` package because the learner architecture is copied exactly.

### Run Lift and Can on a GPU

Follow [the repository setup](../SETUP.md) for Python 3.12, CUDA-enabled
PyTorch, and the robosuite setup script. Run the commands below from the
repository root with the environment activated. The frozen RLPD correctors
use the base requirements. They do not require the optional `robomimic`
Python package.

The corrector training configs are included under
`rlif/correctors/`. Keep each weight file in that run's `checkpoints/`
directory so the loader finds the adjacent run config. The configs describe
how the correctors were trained. RLIF's training settings come from CLI flags.

```bash
lift=rlif/correctors/corrector_lift_ph_s42_20261105-180252
can=rlif/correctors/corrector200_can_ph_s42_20261106-074707
base=https://huggingface.co/datasets/gilliamwong/qoil-runs/resolve/main
mkdir -p "$lift/checkpoints" "$can/checkpoints"
curl -fL "$base/corrector_lift_ph_s42_20261105-180252/checkpoints/step_0064000.pt" -o "$lift/checkpoints/step_0064000.pt"
curl -fL "$base/corrector200_can_ph_s42_20261106-074707/checkpoints/step_0075000.pt" -o "$can/checkpoints/step_0075000.pt"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

```bash
python -m rlif.train --dataset=lift/ph --gate=official_random --corrector_deterministic --corrector_checkpoint="$lift/checkpoints/step_0064000.pt" --device=cuda --checkpoint_model --run_name=lift_random_s42
python -m rlif.train --dataset=can/ph --gate=official_random --corrector_deterministic --corrector_checkpoint="$can/checkpoints/step_0075000.pt" --device=cuda --checkpoint_model --run_name=can_random_s42
```

The default driver uses the paper-reported round abstraction: demo-only
pretraining, then five collected episodes and 2,500 updates per round for 100
rounds. These settings are flags and should be recorded separately from the
step-interleaved loop in the authors' released simulation scripts.

### File guide

| File | Purpose |
| --- | --- |
| `rlif/__init__.py` | Marks the standalone RLIF package. |
| `rlif/train.py` | Start here to run an experiment. It builds the simulator and learner, restores the frozen corrector, and alternates rollout collection with replay updates. It also runs autonomous evaluation and writes the run config, metrics, and optional learner checkpoints. |
| `rlif/agent.py` | Defines the trainable policy and critic ensemble. It turns replay batches into policy and value updates and provides the model layout needed to load the teammate's RLPD weights. |
| `rlif/data.py` | Turns selected Robomimic trajectories into the initial zero-reward replay data. It stores new transitions from the simulator and samples the demonstration and online portions of each training batch. |
| `rlif/human.py` | Decides which policy controls each step and how long takeovers last. Its feedback logic holds the previous transition until the next control decision, so the intervention penalty lands on the learner action that preceded takeover. |
| `rlif/correctors/*/config.json` | These are the two original Lift and Can run configs. The loader uses their model settings to rebuild the network before loading weights. Each config stays one directory above the corresponding checkpoint folder. |
| `rlif/correctors/.gitignore` | Keeps downloaded weight files out of Git while allowing the small run configs to ship with the branch. |
| `tests/test_rlif_feedback.py` | Checks that replay receives the executed action and the correctly timed penalty. It also exercises the round schedule and verifies that an RLPD checkpoint can be loaded into the standalone learner. |
| `tests/test_rlif_controllers.py` | Exercises the random, value-based, and progress-based gates with controlled inputs so takeover timing and duration can be checked without a simulator. |
| `rlpd/env.py` | Shared simulation wrapper, observation layout, and dataset download. |
| `requirements.txt` and `scripts/setup_robosuite.py` | Base Python dependencies and simulator setup. |

Run focused tests with
`python -m pytest tests/test_rlif_feedback.py tests/test_rlif_controllers.py`.
