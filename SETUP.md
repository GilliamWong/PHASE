# Setup

This code base started from [ikostrikov/rlpd](https://github.com/ikostrikov/rlpd)
and was ported to current libraries so it runs on Windows and on recent NVIDIA
GPUs (tested on an RTX 5070 Ti).

## Where to run it

| Where | Device | Use it for |
| --- | --- | --- |
| WSL2 (Ubuntu 24.04) | GPU | Training |
| Native Windows | CPU only | Debugging, quick checks |

JAX does not publish GPU builds for native Windows, so GPU training runs inside
WSL2. The Windows NVIDIA driver is all WSL2 needs; do not install a Linux
driver or the CUDA toolkit inside WSL, because the `jax[cuda12]` wheels bring
their own CUDA libraries.

Both setups use Python 3.12 and [uv](https://docs.astral.sh/uv/).

### WSL2 (GPU)

Keep the virtual environment on the Linux filesystem; one under `/mnt/c` is
much slower.

```bash
cd "/mnt/c/Users/<you>/Desktop/599J Project"
uv venv --python 3.12 ~/.venvs/phase
uv pip install --python ~/.venvs/phase -r requirements.txt
source ~/.venvs/phase/bin/activate
python scripts/setup_robosuite.py
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```

### Native Windows (CPU)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\activate
python scripts/setup_robosuite.py
```

`scripts/setup_robosuite.py` writes robosuite's settings file and, on Windows,
works around two robosuite bugs (a DLL it expects in its own folder, and a
Linux-only rendering backend it forces). Rerun it after reinstalling robosuite.

### Tests

From the repository root:

```bash
python -m pytest tests
```

## Running experiments

Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX does not reserve most of the
GPU's memory up front. Add `WANDB_MODE=offline` (or `disabled`) to run without
a wandb login.

Tasks and demonstrations come from [Robomimic](https://robomimic.github.io/)
(robosuite 1.5 versions of its low-dim datasets). `--dataset=<task>/<quality>`
picks one, e.g. `lift/ph` (200 proficient-human demos) or `can/mh` (300
multi-human demos). It downloads on first use to the Hugging Face cache, and
the environment is built from the env config stored in the file.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
    --dataset=lift/ph \
    --utd_ratio=20 \
    --start_training 5000 \
    --max_steps 300000 \
    --config=configs/rlpd_config.py \
    --project_name=phase_lift
```

The MDP, shared by the env and the demos:

- Observations: Robomimic's standard low-dim keys (`object`,
  `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`), concatenated.
- Reward: `--success_reward` (default 10) on the success step, 0 otherwise.
  Success ends the episode, so demos are cut at their first success.
- Episodes are truncated after `--horizon` steps (default 400, as in Robomimic).

`--offline_ratio` sets the fraction of each batch drawn from the demos
(default 0.5, as in RLPD); `--offline_ratio=0` trains purely online and does
not load the demos.

`--save_video=True` logs one evaluation episode per evaluation to wandb.

## Changes from upstream RLPD

- `gym` 0.23 → `gymnasium` 1.x. Environments use the five-value `step` API;
  only true terminations (not time-limit truncations) zero the bootstrap mask.
- D4RL → Robomimic. `train_finetuning.py` takes `--dataset` (a Robomimic
  task/quality) instead of `--env_name`, builds the robosuite environment from
  the dataset, and logs success rate. It no longer runs the D4RL tasks.
- `tensorflow-probability` → a small tanh-Gaussian in
  `rlpd/distributions/tanh_normal.py`; TFP's JAX backend no longer imports with
  current JAX. The math is the same, but random draws differ from upstream's.
- Removed what this project does not use: the pixel pipeline (DrQ,
  `train_finetuning_pixels.py`, V-D4RL loading, frame stacking), RLPD's
  ablation options (critic dropout, weight decay, ResNet critic, p-norm), and
  unused dataset helpers. They are in the upstream repository if needed.
