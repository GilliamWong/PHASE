#!/bin/bash
# William's human (HIL) run to 150k steps, so its human effort reaches the RaC/Rule 2 runs' (~28-34k).
# Usage: sbatch --job-name=long-<task> long_hil_cap100.sbatch <lift|can>
#SBATCH --partition=ckpt-all
#SBATCH --account=cse
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=09:00:00
#SBATCH --requeue
#SBATCH --output=/gscratch/cse/bbioren/PHASE/slurm-%x-%j.out
set -euo pipefail
TASK=$1
cd /gscratch/cse/bbioren/PHASE
export HF_HOME=/gscratch/cse/bbioren/.hf MUJOCO_GL=egl PYTHONUNBUFFERED=1
PY=.venv/bin/python
HF=https://huggingface.co/datasets/gilliamwong/qoil-runs/resolve/main
if [ "$TASK" = lift ]; then RUNDIR=corrector_lift_ph_s42_20261105-180252; CORR=0050000; REF=0150000
else RUNDIR=corrector200_can_ph_s42_20261106-074707; CORR=0075000; REF=0500000; fi
D=ckpts/$TASK; mkdir -p $D
for s in $CORR $REF; do [ -s $D/step_$s.pt ] || curl -sfL -o $D/step_$s.pt $HF/$RUNDIR/checkpoints/step_$s.pt; done
OUT=runs_cap100_long/$TASK; mkdir -p $OUT
COMMON="--dataset=$TASK/ph --num_demos=20 --no-tqdm --log_dir=$OUT --utd_ratio=10 \
  --init_temperature=0.01 --temp_lr=0 --max_steps=150000 --eval_interval=5000 --eval_episodes=10"
HUMAN="--corrector=$D/step_$CORR.pt --gate=progress --intervention_rate=0 \
  --progress_reference=$D/step_$REF.pt --progress_window=12 --progress_threshold=-1.58 \
  --stall_intervention_prob=0.599 --full_takeover_prob=0 --min_takeover_steps=10 --max_takeover_steps=100"
run() { name=$1; seed=$2; shift 2
  ls -d $OUT/${name}_${TASK}_ph_s${seed}_*/done >/dev/null 2>&1 && return 0
  rm -rf $OUT/${name}_${TASK}_ph_s${seed}_*
  $PY -m rlpd.train $COMMON $HUMAN --seed=$seed --run_name=$name "$@" \
    && touch $(ls -d $OUT/${name}_${TASK}_ph_s${seed}_* | head -1)/done; }
for seed in 42 43 44; do run long_hil $seed & done   # William human, run to ~30k human steps
wait
echo ALL_DONE
