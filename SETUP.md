# Setup

This branch (`learn-rl`) is a template for implementing RL methods yourself:
BC, DAgger, DQN, DDPG, TD3 and SAC. The environments, experts, demos, replay
buffer, logging and evaluation are provided; the networks, the agents and the
key steps of the training loops are blanks to fill in, all in
`rlpd/learn.py`. The PHASE/RLPD code is on `main`. It runs on the GPU or CPU,
on native Windows and on Linux.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/). Torch comes from PyTorch's
CUDA index; pick the CUDA version your driver supports (`nvidia-smi` shows it;
RTX 50-series cards need cu128 or newer).

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv -r requirements.txt
.venv\Scripts\activate
python scripts/setup_robosuite.py
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

The same commands work on Linux or WSL2 (`source .venv/bin/activate`).
`scripts/setup_robosuite.py` writes robosuite's settings file and, on Windows,
works around two robosuite bugs (a DLL it expects in its own folder, and a
Linux-only rendering backend it forces). Rerun it after reinstalling robosuite.

### Tests

```bash
python -m pytest tests
python -m pytest tests/test_learn.py -k dqn   # one method's checks
```

`tests/test_learn.py` checks the pieces of `rlpd/learn.py`; each test fails
with `NotImplementedError` until the piece it checks is filled in. Besides
shape and range checks, BC, DQN and SAC must learn a one-step task. DDPG and
TD3 are checked with a Pendulum run instead (below): their deterministic
actors can saturate tanh early on that task even when they are correct. The
env tests need the Lift and Can datasets in the Hugging Face cache (a
Robomimic training run downloads them) and skip otherwise.

## Working through the methods

Search `rlpd/learn.py` for `TODO`. A suggested order:

1. The networks: `DeterministicPolicy`, `QNetwork`, `DiscreteQNetwork`,
   `GaussianPolicy`.
2. `BCAgent`, then `--algo=bc`.
3. The DAgger rollout loop in `train_dagger`, then `--algo=dagger`.
4. The three blanks in `train_online` (shared by every RL method), then
   `DQNAgent`.
5. `DDPGAgent`, `TD3Agent`, `SACAgent`.

```bash
python -m rlpd.learn --algo=bc                         # Pendulum-v1 by default
python -m rlpd.learn --algo=dagger --num_demos=1
python -m rlpd.learn --algo=dqn
python -m rlpd.learn --algo=sac --env=MountainCarContinuous-v0
python -m rlpd.learn --algo=sac --env=robomimic:lift/ph --offline_ratio=0.5 --max_steps=300000
python -m rlpd.learn --mode=eval --checkpoint=runs/<run>/checkpoints/step_0020000.pt
tensorboard --logdir runs
```

Every field of `Config` is a flag (`python -m rlpd.learn --help`).
`--max_steps` counts environment steps for the RL methods and gradient steps
for BC; DAgger runs `--dagger_iterations` rounds instead.

### Environments, experts and demos

| `--env` | Actions | Reward | Expert |
| --- | --- | --- | --- |
| `Pendulum-v1` (default) | 1 (torque) | dense, every step | energy swing-up + PD balance, return about -156 |
| `MountainCarContinuous-v0` | 1 (force) | +100 at the goal, minus action cost | push along the velocity, return about 89 |
| `robomimic:lift/ph`, `robomimic:can/ph`, ... | 7 (arm + gripper) | 10 on success, else 0 | a trained RLPD checkpoint, `--expert=...` |

Actions are always in [-1, 1] (Gymnasium tasks are rescaled). On the
Gymnasium tasks, the hand-written expert records `--num_demos` episodes as
the demos for BC, DAgger and `--offline_ratio`; Robomimic uses its human
demos. `--offline_ratio` is 0 by default (pure online RL), but sparse-reward
Robomimic tasks need demos, e.g. `--offline_ratio=0.5` as in RLPD. For
DAgger on Lift, `--expert` can be the qoil checkout's
`runs/corrector_lift_ph_s42_20261105-180252/checkpoints/step_0150000.pt`
(199/200 successes).

### What a correct implementation reaches

From a reference solution, seed 42, default flags, 1 to 4 minutes per run
on an RTX 5070 Ti with three runs in parallel. Evaluation return on
Pendulum-v1 (10 episodes; -150 is about as good as the expert):

| `--algo` | Reaches about -150 by | Notes |
| --- | --- | --- |
| `bc` | 4k gradient steps | 10 demos. With `--num_demos=1`, stays near -1300 |
| `dagger` | iteration 2 | with `--num_demos=1`, iteration 3 |
| `dqn` | 12k env steps | |
| `ddpg` | 6k env steps | |
| `td3` | 12k env steps | |
| `sac` | 6k env steps | |

Evaluation returns swing between about -100 and -230 after that, from the
start states drawn. BC vs DAgger with one demo is the clearest view of
compounding errors: the BC policy drifts off the single demo trajectory and
never recovers, while DAgger labels those drifted states.

Each run gets its own folder,
`runs/[<run_name>_]<algo>_<env>_s<seed>_<date>-<time>/`, with TensorBoard
logs and `config.json`, plus `checkpoints/step_<step>.pt` with
`--checkpoint_model` and `videos/` with `--save_video` (Gymnasium tasks need
`pip install pygame` for videos).

## Code layout

| File | Contents |
| --- | --- |
| `rlpd/learn.py` | Everything you write (networks, agents, training loops), plus the provided experts, demos, env construction, evaluation, logging and checkpoints |
| `rlpd/data.py` | `TransitionBuffer` (replay buffer on the GPU); Robomimic demo loading; `sample_mixed` (demo/online batches). Unchanged from main |
| `rlpd/env.py` | `RobomimicEnv`, a robosuite task as a Gymnasium env with the sparse success reward; dataset download. Unchanged from main |
