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
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```

### Native Windows (CPU)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\activate
```

### Tests

From the repository root:

```bash
python -m pytest tests
```

## Running experiments

Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX does not reserve most of the
GPU's memory up front. Add `WANDB_MODE=offline` (or `disabled`) to run without
a wandb login.

Offline data comes from [Minari](https://minari.farama.org/), the maintained
successor of D4RL. Datasets download to `~/.minari/datasets` on first use.

### MuJoCo locomotion

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
    --dataset_id=mujoco/halfcheetah/expert-v0 \
    --utd_ratio=20 \
    --start_training 5000 \
    --max_steps 250000 \
    --config=configs/rlpd_config.py \
    --project_name=rlpd_locomotion
```

### AntMaze

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
    --dataset_id=D4RL/antmaze/umaze-v1 \
    --utd_ratio=20 \
    --start_training 5000 \
    --max_steps 300000 \
    --config=configs/rlpd_config.py \
    --config.backup_entropy=False \
    --config.hidden_dims="(256, 256, 256)" \
    --config.num_min_qs=1 \
    --project_name=rlpd_antmaze
```

### Evaluation videos

`--save_video=True` logs one evaluation episode per evaluation to wandb. In
WSL, also set `MUJOCO_GL=egl` for headless rendering; on Windows the default
works.

## Changes from upstream RLPD

- `gym` 0.23 → `gymnasium` 1.x. Environments use the five-value `step` API;
  only true terminations (not time-limit truncations) zero the bootstrap mask.
- D4RL → Minari. `train_finetuning.py` takes `--dataset_id` (a Minari id)
  instead of `--env_name`, and builds the environment from the dataset.
  Minari's `mujoco/*` datasets were re-collected with the current MuJoCo
  environments, so scores are not directly comparable with the paper's D4RL
  numbers.
- `tensorflow-probability` → a small tanh-Gaussian in
  `rlpd/distributions/tanh_normal.py`; TFP's JAX backend no longer imports with
  current JAX. The math is the same, but random draws differ from upstream's.
- Removed the AWAC Adroit "binary" datasets (`--env_name=pen-binary-v0`, etc.).
  They need `mj_envs`, which is built on `mujoco-py` and does not install on
  current Python. Minari's `D4RL/pen/*`, `D4RL/door/*`, ... Adroit datasets
  work with `train_finetuning.py`.
- Removed what this project does not use: the pixel pipeline (DrQ,
  `train_finetuning_pixels.py`, V-D4RL loading, frame stacking), RLPD's
  ablation options (critic dropout, weight decay, ResNet critic, p-norm), and
  unused dataset helpers. They are in the upstream repository if needed.
