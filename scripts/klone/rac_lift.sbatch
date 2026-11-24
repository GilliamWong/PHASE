#!/bin/bash
#SBATCH --job-name=rac-lift
#SBATCH --partition=ckpt-all
#SBATCH --account=cse
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/cse/bbioren/PHASE/slurm-%x-%j.out
set -euo pipefail
cd /gscratch/cse/bbioren/PHASE
export HF_HOME=/gscratch/cse/bbioren/.hf MUJOCO_GL=egl PYTHONUNBUFFERED=1
PY=.venv/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
COMMON="--dataset=lift/ph --num_demos=20 --no-tqdm"
# 1. Stand-in human: plain RLPD on 20 demos, checkpoint every 2.5k steps.
CORR_DIR=$(ls -d runs/corrector_lift_ph_s42_* 2>/dev/null | head -1 || true)
if [ -z "$CORR_DIR" ] || [ ! -f "$CORR_DIR/done" ]; then
  $PY -m rlpd.train $COMMON --run_name=corrector --max_steps=40000 \
    --checkpoint_model --checkpoint_interval=2500 --eval_interval=2500 --eval_episodes=20
  CORR_DIR=$(ls -d runs/corrector_lift_ph_s42_* | head -1); touch $CORR_DIR/done
fi
CORR=$($PY pick_ckpt.py $CORR_DIR 0.6)
echo "corrector: $CORR"
# 2. HIL and BC+RL under William's human (corrector only) and the RaC human.
HUMAN="--corrector=$CORR --gate=random --intervention_rate=0.005 --full_takeover_prob=0 \
  --min_takeover_steps=10 --max_takeover_steps=40"
RUN="--utd_ratio=10 --init_temperature=0.01 --temp_lr=0 --max_steps=30000 \
  --eval_interval=2500 --eval_episodes=20"
for seed in 42 43; do
  $PY -m rlpd.train $COMMON $RUN $HUMAN --seed=$seed --run_name=racexp_qoilhuman_hil &
  $PY -m rlpd.train $COMMON $RUN $HUMAN --seed=$seed --run_name=racexp_qoilhuman_bcrl --bc_weight=0.1 &
  $PY -m rlpd.train $COMMON $RUN $HUMAN --seed=$seed --run_name=racexp_rachuman_hil --style=rac --end_after_takeover &
  $PY -m rlpd.train $COMMON $RUN $HUMAN --seed=$seed --run_name=racexp_rachuman_bcrl --style=rac --end_after_takeover --bc_weight=0.1 &
done
wait
echo ALL_DONE
