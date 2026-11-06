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

The defaults are RLPD's settings, except the update-to-data ratio: 10 critics
with LayerNorm, 2 of them in each target's min, batches that are half
demonstrations, half online data, and 10 critic updates per environment step
(Q-OIL's setting for every method; RLPD's is `--utd_ratio=20`, about twice as
slow). Every field of `TrainConfig` (`rlpd/train.py`), `AgentConfig`
(`rlpd/agent.py`) and `HumanConfig` (`rlpd/human.py`) is a flag, e.g.
`--utd_ratio=20`, `--hidden_dims 256 256`, `--save_video`, `--no-cuda_graph`;
`python -m rlpd.train --help` lists them.

Speed: on an RTX 5070 Ti, a Lift step takes about 4.5 ms in robosuite (CPU)
and 6.6 ms for the update at UTD 10 (12.7 ms at 20). The loop launches each
update before stepping the env, so the two overlap; each update therefore
trains on transitions up to the previous step. Matrix multiplies use TF32 by
default, which makes updates about 25% faster at reduced precision and
changes a run's exact numbers, not the algorithm; `--no-tf32` gives full
float32. Evaluations of
failing episodes are slow (up to `--eval_episodes` × 400 env steps), so keep
`--eval_interval` coarse and save checkpoints more often with
`--checkpoint_interval` if needed.

Each run gets its own new folder,
`runs/[<run_name>_]<task>_<quality>_s<seed>_<date>-<time>/` (`--run_name` is an
optional label, e.g. `--run_name=qoil_b0.25`), and nothing in it is
overwritten. It holds:

- the TensorBoard logs and `config.json` (every setting);
- `videos/`, one evaluation episode per evaluation (with `--save_video`);
- `checkpoints/step_<step>.pt`, the agent and optimizer state every
  `--checkpoint_interval` steps (with `--checkpoint_model`);
- `buffer/step_<step>.pt`, the online transitions added since the previous
  save (with `--checkpoint_buffer`). Concatenating the files in step order
  rebuilds the buffer up to that step.

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
names the checkpoint of the agent that acts during takeovers (its mean
action; `--no-corrector_deterministic` samples from its policy instead), for example a partly trained RLPD run's
`checkpoints/step_<step>.pt`. The gate decides when to take over:

- `--gate=random` (default): each step with probability `--intervention_rate`.
- `--gate=progress` (Q-OIL's PAM): also with probability
  `--stall_intervention_prob` (0.599) when `V(s_t) - V(s_{t-k})` is below
  `--progress_threshold` (-0.158), with `k = --progress_window` (12) and `V`
  from the critic in `--progress_reference`. On our reward-10 critics the
  paper's threshold (for a 0-1 value scale) is -1.58.

A takeover lasts until the episode ends with probability
`--full_takeover_prob` (0.5), otherwise for a uniform number of steps in
[`--min_takeover_steps`, `--max_takeover_steps`] (10 to 40). Intervention
transitions are flagged in the replay buffer; evaluation is always autonomous.
The human is active from the first step, as in Q-OIL's Algorithm 1. Updates
start after `--start_training` (200) random-action steps, the only "training
starts" value the paper gives; this repo used 5,000 before.
`--demos_as_interventions` counts the warm-start demos as interventions:
they get the bonus and start the BC buffer (the paper does not say whether
its D_int includes them).

**Q-OIL** (`AgentConfig`) learns from the flagged transitions:

- `--bonus=b` adds `b` to the reward of intervention transitions;
- `--optimistic_critic` trains a second critic ensemble on reward + bonus,
  bootstrapping from the task critic's targets, and has the actor maximize it.
  Without it, the bonus enters the single critic (the paper's no-decoupling
  ablation);
- `--bc_weight=λ` adds a BC term, weighted by λ, over intervention
  transitions sampled from their own buffer (Q-OIL's D_int). `--bc_loss`
  picks its form, which the paper does not give: `log_prob` (default),
  `-E[log π(a | s)]`, or `mse`, `E[Σ_i (tanh(μ_i(s)) - a_i)²]` on the mean
  action.

On a deterministic corrector, `log_prob` keeps rising as the policy's std
shrinks, so its gradient grows during training and λ does not fix BC's
strength: in our fixed-temperature Lift BC+RL run, BC's gradient was about
20 times the critic's and the policy's entropy fell to about -10. `mse` is
bounded and leaves the std to the entropy term.
The same λ means a different strength under each form (MSE BC can become
nearly negligible once the mean is close), so compare forms at matched
`actor_grad_norm_bc / actor_grad_norm_q`, or over a few λ, not at equal λ.

The tanh policy's log-density is infinite at actions of exactly ±1, which a
corrector's gripper (and other saturated dimensions) can produce, so BC first
clamps intervention actions to within `--bc_action_eps` (1e-5) of ±1 (for
`--bc_loss=log_prob`; `mse` needs no clamp). For
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
the temperature but gives no value for these tasks; `--init_temperature=0.01
--temp_lr=0` fixes it at 0.01, the value its VLA table gives. Use that for
BC runs: with a learned temperature, BC narrows the policy, the temperature
rises about 20 times to widen it again, and BC+RL on Lift fell from 0.49 to
0.32 average success.

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

## Evaluating checkpoints

Training's evaluations use few episodes (10 gives a standard error of up to
0.16 on a success rate). To pick checkpoints, for example correctors of a
given success rate, re-evaluate them:

```bash
python -m scripts.evaluate_checkpoints runs/<run> --steps 40000 42000 44000
```

Each checkpoint runs its deterministic policy for `--episodes` (100)
episodes from the same initial states (`--seed`, 1000), on the CPU in
`--workers` (4) processes. Results, with standard errors, go to
`runs/<run>/checkpoint_eval_<episodes>ep_seed<seed>.csv`; steps already in
that file are skipped, so a rerun with more `--steps` only adds the new ones.
Without `--steps` it evaluates every checkpoint.

## Saved runs

Git holds each run's metrics under `runs/<run>/`: `config.json`, the
TensorBoard log, `train.log` (the run's console output, for runs trained by
a job queue) and any checkpoint-evaluation CSVs or evaluation videos, so
`tensorboard --logdir runs` and `python -m scripts.summarize_runs runs/<glob>`
work on a fresh clone. Checkpoints and replay buffers (about 0.5 GB per run)
are in the private Hugging Face dataset
[gilliamwong/qoil-runs](https://huggingface.co/datasets/gilliamwong/qoil-runs),
laid out the same way. After `hf auth login` with an account that can read
it, download one run's files, or everything, into `runs/`:

```bash
hf download gilliamwong/qoil-runs --repo-type dataset --local-dir runs --include "<run>/*"
hf download gilliamwong/qoil-runs --repo-type dataset --local-dir runs
```

Run folders are named `<run_name>_<task>_<quality>_s<seed>_<start time>`.
The figure 10 sweep's runs start with `f10_<takeover rule>_<method>`, where
the rule is `full` (half of takeovers last until the episode ends, otherwise
10-40 steps) or `cap100` (every takeover lasts 10-100 steps); all use the
progress gate with `--progress_threshold=-1.58`, `--intervention_rate=0`
and a temperature fixed at 0.01. Each run's `config.json` has its full
configuration.

## Code layout

| File | Contents |
| --- | --- |
| `rlpd/agent.py` | `AgentConfig`; the tanh-Gaussian actor; the critic ensemble; `RLPDAgent.update` (critic, actor and temperature losses, with Q-OIL's optional optimistic critic, bonus and BC term); loading checkpoints; `capture_cuda_graph` |
| `rlpd/data.py` | `TransitionBuffer` (replay buffer on the GPU, with an intervention flag per transition); Robomimic demo loading; `sample_mixed` (the 50/50 demo/online batches) |
| `rlpd/env.py` | `RobomimicEnv`, a robosuite task as a Gymnasium env with the sparse success reward; dataset download |
| `rlpd/human.py` | `HumanConfig` and `SyntheticHuman`: the gate and the corrector that take over during training |
| `rlpd/train.py` | `TrainConfig` and flags, the training loop, evaluation, logging |
| `scripts/evaluate_checkpoints.py` | Re-evaluates a run's saved checkpoints on many episodes (see [Evaluating checkpoints](#evaluating-checkpoints)) |
| `scripts/summarize_runs.py` | One row per run from its TensorBoard logs: success averaged over training (Q-OIL's Fig. 10 metric), steps to 80% success, intervention share, critic gap and actor gradient norms. Compare runs with the same `--max_steps` and `--eval_interval`. `python -m scripts.summarize_runs runs/pilot_*` |

### The CUDA graph, and what it means for new losses

On a GPU, the whole update (sampling the batch, the `--utd_ratio` critic
steps, the actor and temperature steps) is recorded once as a CUDA graph and
replayed every step. That removes Python and kernel-launch overhead, which
dominate with networks this small: at 20 critic steps, about 12 ms per update
instead of about 50 ms run normally.

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
  matches the JAX version to float32 precision (with `--no-tf32`); the random
  streams differ, so runs are not bit-identical.
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
