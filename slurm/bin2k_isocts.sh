#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -uo pipefail
ml gcc/12.4.0 python/3.12.1 cuda/12.4.0
PD=.
source "$PD/venv/bin/activate"
export PYTHONPATH="$PD/src"
export HF_HOME=/scratch/users/$USER/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ETAG_TIMEOUT=10
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export TMPDIR=/scratch/users/$USER/tmp
mkdir -p "$TMPDIR"
cd "$PD"
: "${BB:?}"; : "${BID:?}"; : "${SEED:?}"
OUT="$PD/outputs/$BB/seed_${SEED}_cap2k"
CAP=2000
echo "=== BINARY-2k CELL (fast finish) $BB seed_$SEED ($BID) ==="
for D in isic_mel chexpert_pe patchcamelyon; do
  if [ -f "$OUT/task_vectors/$D/task_vector.pt" ]; then echo "SKIP train $D"; continue; fi
  echo "########## $BB s$SEED TRAIN $D ##########"
  python3 -m med_merge.cli train --dataset "$D" --backbone "$BID" --seed "$SEED" \
    --output-dir "$OUT" --max-train-samples "$CAP" --device cuda || echo "TRAIN $D FAILED"
done
DS="--datasets isic_mel --datasets chexpert_pe --datasets patchcamelyon"
for M in iso_cts; do
  R="$OUT/results_binary/$M/results.json"
  if [ -f "$R" ]; then echo "SKIP $M (done)"; continue; fi
  echo "########## $BB s$SEED MERGE $M ##########"
  python3 -m med_merge.cli merge --method "$M" --backbone "$BID" $DS --hyperopt \
    --checkpoint-dir "$OUT/checkpoints" --task-vector-dir "$OUT/task_vectors" \
    --output-dir "$OUT/merged_binary" --device cuda || { echo "MERGE $M FAILED"; continue; }
  echo "########## $BB s$SEED EVAL $M ##########"
  python3 -m med_merge.cli evaluate --model-path "$OUT/merged_binary/$M/merged_encoder.pt" \
    --backbone "$BID" $DS --head-dir "$OUT/checkpoints" \
    --output-dir "$OUT/results_binary/$M" --device cuda || echo "EVAL $M FAILED"
done
echo "BIN2K_CELL_DONE $BB s$SEED"
