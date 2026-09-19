#!/bin/bash -l
#SBATCH --partition=gpu,roxanad
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/groups/roxanad/sonnet/med-merge/logs/%x_%j.out
#SBATCH --error=/home/groups/roxanad/sonnet/med-merge/logs/%x_%j.err
# Training-budget ablation cell: one (backbone, seed, cap) on the binary trio.
# Mirrors bin2k_cell.sh but takes CAP from the environment and writes to
# outputs/$BB/seed_${SEED}_cap${CAP}. Only the leading tuned methods plus the
# two contrasts (simple averaging, Iso-C) are merged.
set -uo pipefail
ml gcc/12.4.0 python/3.12.1 cuda/12.4.0
[ -f "$HOME/.secrets" ] && source "$HOME/.secrets"
PD=/home/groups/roxanad/sonnet/med-merge
source "$PD/venv/bin/activate"
export PYTHONPATH="$PD/src"
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export TMPDIR=/scratch/users/$USER/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_ETAG_TIMEOUT=10 HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p "$TMPDIR"
cd "$PD"
: "${BB:?}"; : "${BID:?}"; : "${SEED:?}"; : "${CAP:?}"
OUT="$PD/outputs/$BB/seed_${SEED}_cap${CAP}"
METHODS="${METHODS:-gram_ls lines task_arithmetic dare simple_avg iso_c}"
echo "=== CAP-SWEEP CELL $BB seed_$SEED cap=$CAP ($BID) === $(date)"

# 1. Train the three binary specialists at this cap (skip if task vector exists)
for D in isic_mel chexpert_pe patchcamelyon; do
  if [ -f "$OUT/task_vectors/$D/task_vector.pt" ]; then echo "SKIP train $D"; continue; fi
  echo "########## $BB s$SEED cap$CAP TRAIN $D ########## $(date)"
  python3 -m med_merge.cli train --dataset "$D" --backbone "$BID" --seed "$SEED" \
    --output-dir "$OUT" --max-train-samples "$CAP" --device cuda || echo "TRAIN $D FAILED"
done

# 2 & 3. Merge with validation hyperopt, then evaluate
DS="--datasets isic_mel --datasets chexpert_pe --datasets patchcamelyon"
for M in $METHODS; do
  R="$OUT/results_binary/$M/results.json"
  if [ -f "$R" ]; then echo "SKIP $M (done)"; continue; fi
  echo "########## $BB s$SEED cap$CAP MERGE $M ########## $(date)"
  python3 -m med_merge.cli merge --method "$M" --backbone "$BID" $DS --hyperopt \
    --checkpoint-dir "$OUT/checkpoints" --task-vector-dir "$OUT/task_vectors" \
    --output-dir "$OUT/merged_binary" --device cuda || { echo "MERGE $M FAILED"; continue; }
  echo "########## $BB s$SEED cap$CAP EVAL $M ########## $(date)"
  python3 -m med_merge.cli evaluate --model-path "$OUT/merged_binary/$M/merged_encoder.pt" \
    --backbone "$BID" $DS --head-dir "$OUT/checkpoints" \
    --output-dir "$OUT/results_binary/$M" --device cuda || echo "EVAL $M FAILED"
done
echo "CAP_SWEEP_CELL_DONE $BB s$SEED cap$CAP $(date)"
