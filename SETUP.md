# Setup

This code base started from [ikostrikov/rlpd](https://github.com/ikostrikov/rlpd)
(JAX) and is now a PyTorch port of it, trimmed to what this project uses. It
runs on the GPU on native Windows and on Linux (tested on an RTX 5070 Ti).

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
```

The env tests need the Lift and Can datasets in the Hugging Face cache (a
training run downloads them) and skip otherwise; the CUDA graph test needs a GPU.

## Running experiments

```bash
python -m rlpd.train --dataset=lift/ph
tensorboard --logdir runs
```

The defaults are RLPD's settings: 10 critics with LayerNorm, 2 of them in each
target's min, 20 critic updates per environment step, and batches that are half
demonstrations, half online data. Every field of `TrainConfig`
(`rlpd/train.py`) and `AgentConfig` (`rlpd/agent.py`) is a flag, e.g.
`--utd_ratio=10`, `--hidden_dims 256 256`, `--save_video`, `--no-cuda_graph`;
`python -m rlpd.train --help` lists them.

Each run gets its own folder, `runs/<task>_<quality>_s<seed>_<date>-<time>/`,
holding the TensorBoard logs, `config.json` (every setting), evaluation videos
(`videos/`, with `--save_video`) and checkpoints (`checkpoint.pt` and
`buffer.pt`, with `--checkpoint_model` / `--checkpoint_buffer`, overwritten at
each evaluation). TensorBoard shows `training/*` (losses, Q, entropy,
temperature every `--log_interval` steps; return, length and success per
episode) and `evaluation/*` (mean return, length and success over
`--eval_episodes` deterministic episodes every `--eval_interval` steps).

Tasks and demonstrations come from [Robomimic](https://robomimic.github.io/)
(robosuite 1.5 versions of its low-dim datasets). `--dataset=<task>/<quality>`
picks one, e.g. `lift/ph` (200 proficient-human demos) or `can/mh` (300
multi-human demos). It downloads on first use to the Hugging Face cache, and
the environment is built from the env config stored in the file.

The MDP, shared by the env and the demos:

- Observations: Robomimic's standard low-dim keys (`object`,
  `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`), concatenated.
- Reward: `--success_reward` (default 10) on the success step, 0 otherwise.
  Success ends the episode, so demos are cut at their first success.
- Episodes are truncated after `--horizon` steps (default 400, as in Robomimic).

`--offline_ratio` sets the fraction of each batch drawn from the demos
(default 0.5); `--offline_ratio=0` trains purely online and does not load the
demos.

## Code layout

| File | Contents |
| --- | --- |
| `rlpd/agent.py` | `AgentConfig`; the tanh-Gaussian actor; the critic ensemble; `RLPDAgent.update` (critic, actor and temperature losses); `capture_cuda_graph` |
| `rlpd/data.py` | `TransitionBuffer` (replay buffer on the GPU); Robomimic demo loading; `sample_mixed` (the 50/50 demo/online batches) |
| `rlpd/env.py` | `RobomimicEnv`, a robosuite task as a Gymnasium env with the sparse success reward; dataset download |
| `rlpd/train.py` | `TrainConfig` and flags, the training loop, evaluation, logging |

### The CUDA graph, and what it means for new losses

On a GPU, the whole update (sampling the batch, 20 critic steps, the actor and
temperature steps) is recorded once as a CUDA graph and replayed every step.
That removes Python and kernel-launch overhead, which dominate with networks
this small: about 12 ms per update instead of about 50 ms run normally.

A recording replays the same GPU work every time, so code inside
`RLPDAgent.update` and `sample_mixed` must:

- keep every tensor's shape fixed. Data that grows (for example a set of
  states for reward shaping) goes in a preallocated, fixed-capacity tensor
  with its fill level stored on the GPU, as `TransitionBuffer` does;
- never read values back to the host (`.item()`, `.cpu()`, `if tensor:`,
  data-dependent shapes such as boolean-mask indexing);
- keep host-side numbers that change between steps (such as a decaying loss
  weight) in GPU tensors updated in place, not Python floats, which are frozen
  into the recording.

Recording fails loudly on host reads. Debug new code with `--no-cuda_graph`,
then turn the graph back on; `tests/test_agent.py` checks that replays match
updates run normally.

## Changes from upstream RLPD

- JAX/Flax → PyTorch. One update from the same weights, batch and random draws
  matches the JAX version to float32 precision; the random streams differ, so
  runs are not bit-identical.
- `gym` 0.23 → `gymnasium` 1.x. Environments use the five-value `step` API;
  only true terminations (not time-limit truncations) zero the bootstrap mask.
- D4RL → Robomimic. Training takes `--dataset` (a Robomimic task/quality),
  builds the robosuite environment from the dataset, and logs success rate.
- wandb → TensorBoard, with the same metric names; videos are mp4 files in
  the run folder.
- Removed what this project does not use: the pixel pipeline (DrQ, V-D4RL),
  RLPD's ablation options (critic dropout, weight decay, ResNet critic,
  p-norm), and the separate SAC and RLPD configs. The defaults are RLPD; the
  old SAC config is `--num_qs=2 --num_min_qs=2 --hidden_dims 256 256
  --no-critic_layer_norm --backup_entropy --utd_ratio=1 --offline_ratio=0`.
  The removed parts are in the upstream repository if needed.
