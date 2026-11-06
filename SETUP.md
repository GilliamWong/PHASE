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

Each run gets its own new folder,
`runs/[<run_name>_]<task>_<quality>_s<seed>_<date>-<time>/` (`--run_name` is an
optional label, e.g. `--run_name=qoil_b0.25`), and nothing in it is
overwritten. It holds:

- the TensorBoard logs and `config.json` (every setting);
- `videos/`, one evaluation episode per evaluation (with `--save_video`);
- `checkpoints/step_<step>.pt`, the agent and optimizer state at every
  evaluation (with `--checkpoint_model`);
- `buffer/step_<step>.pt`, the online transitions added since the previous
  evaluation (with `--checkpoint_buffer`). Concatenating the files in step
  order rebuilds the buffer up to that step.

`<step>` matches the x-axis in TensorBoard. TensorBoard shows `training/*` (losses, Q, entropy,
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
demos. `--num_demos=N` uses only the dataset's first N demos.

## Interventions and Q-OIL

With no intervention flags set, training is plain RLPD. Two independent
groups of flags add interventions and learning from them.

**A synthetic human** (`HumanConfig`, `rlpd/human.py`) takes over from the
learner during training, following Q-OIL's simulated human. `--corrector`
names the checkpoint of the agent that acts during takeovers (deterministic
mean actions), for example a partly trained RLPD run's
`checkpoints/step_<step>.pt`. The gate decides when to take over:

- `--gate=random` (default): each step with probability `--intervention_rate`.
- `--gate=progress` (Q-OIL's PAM): also with probability
  `--stall_intervention_prob` (0.599) when `V(s_t) - V(s_{t-k})` is below
  `--progress_threshold` (-0.158), with `k = --progress_window` (12) and `V`
  from the critic in `--progress_reference`. The paper's values assume
  rewards of 0 or 10.

A takeover lasts until the episode ends with probability
`--full_takeover_prob` (0.5), otherwise for a uniform number of steps in
[`--min_takeover_steps`, `--max_takeover_steps`] (10 to 40). Intervention
transitions are flagged in the replay buffer; evaluation is always autonomous.

**Q-OIL** (`AgentConfig`) learns from the flagged transitions:

- `--bonus=b` adds `b` to the reward of intervention transitions;
- `--optimistic_critic` trains a second critic ensemble on reward + bonus,
  bootstrapping from the task critic's targets, and has the actor maximize it.
  Without it, the bonus enters the single critic (the paper's no-decoupling
  ablation);
- `--bc_weight=λ` adds `-λ E[log π(a | s)]` over intervention transitions,
  sampled from their own buffer (Q-OIL's D_int).

The tanh policy's log-density is infinite at actions of exactly ±1, which a
corrector's gripper (and other saturated dimensions) can produce, so BC first
clamps intervention actions to within `--bc_action_eps` (1e-5) of ±1. For
those actions, eps sets BC's pre-tanh target (`atanh(1 - eps)`, about 6.1)
and so the size of its gradient; it also adds a constant, `log(2 eps)`, to
the BC loss. The paper does not say how it handles ±1, so before drawing
conclusions from BC's strength, check `intervention_clamped_fraction` and, if
it is not small, repeat the comparison at a few eps (e.g. 1e-3, 1e-5, 1e-7).

These need `--corrector`. The paper's Q-OIL is

```bash
python -m rlpd.train --dataset=lift/ph --num_demos=20 --utd_ratio=10 \
    --corrector=runs/<corrector run>/checkpoints/step_<step>.pt \
    --bonus=0.025 --optimistic_critic --bc_weight=0.1
```

with `--bonus=0 --no-optimistic_critic` giving BC+RL and also
`--bc_weight=0` giving HIL (interventions in the buffer only). The paper fixes
the temperature but gives no value for these tasks; this code keeps RLPD's
learned temperature.

Extra TensorBoard metrics:

- per episode: `intervention_steps`, `intervention_fraction`,
  `total_intervention_steps`, and `intervention_clamped_fraction` (overall
  and `_dim<j>` per action dimension): the share of all intervention actions
  so far that BC clamps;
- every `--log_interval` steps: `batch_intervention_fraction` (the share of
  intervention transitions in critic batches), `actor_q` (the Q the actor
  maximizes), `bc_loss` (the unweighted BC term) and, with the optimistic
  critic, `optimistic_q` and the mean optimistic-minus-task gap on
  intervention (`gap_intervention`) and other (`gap_other`) transitions;
- every `--log_interval` steps, `actor_grad_norm_q`, `actor_grad_norm_entropy`
  and `actor_grad_norm_bc`: the norm of each weighted actor-loss term's
  gradient over the actor's weights, on a fresh batch. These compare how hard
  each term pushes the policy; unlike loss values, they ignore constants such
  as BC's `log(2 eps)`.

## Code layout

| File | Contents |
| --- | --- |
| `rlpd/agent.py` | `AgentConfig`; the tanh-Gaussian actor; the critic ensemble; `RLPDAgent.update` (critic, actor and temperature losses, with Q-OIL's optional optimistic critic, bonus and BC term); loading checkpoints; `capture_cuda_graph` |
| `rlpd/data.py` | `TransitionBuffer` (replay buffer on the GPU, with an intervention flag per transition); Robomimic demo loading; `sample_mixed` (the 50/50 demo/online batches) |
| `rlpd/env.py` | `RobomimicEnv`, a robosuite task as a Gymnasium env with the sparse success reward; dataset download |
| `rlpd/human.py` | `HumanConfig` and `SyntheticHuman`: the gate and the corrector that take over during training |
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
- Added: a synthetic human that intervenes during training, and Q-OIL's
  bonus, optimistic critic and BC term, all off by default (see
  [Interventions and Q-OIL](#interventions-and-q-oil)).
