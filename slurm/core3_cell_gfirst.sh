#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --time=02:30:00
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
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export TMPDIR=/scratch/users/$USER/tmp
mkdir -p "$TMPDIR"
cd "$PD"
: "${BB:?set BB}"; : "${BID:?set BID}"; : "${SEED:?set SEED}"
OUT="$PD/outputs/$BB/seed_$SEED"
DS="--datasets isic2017 --datasets chexpert --datasets pathmnist"
echo "=== CELL $BB seed_$SEED ($BID) ==="
for M in gram_ls tsv_merge task_arithmetic simple_avg iso_c; do
  R="$OUT/results_core3/$M/results.json"
  if [ -f "$R" ]; then echo "SKIP $M (already done)"; continue; fi
  echo "########## $BB s$SEED MERGE $M ##########"
  python3 -m med_merge.cli merge --method "$M" --backbone "$BID" $DS --hyperopt \
    --checkpoint-dir "$OUT/checkpoints" --task-vector-dir "$OUT/task_vectors" \
    --output-dir "$OUT/merged_core3" --device cuda || { echo "MERGE $M FAILED"; continue; }
  echo "########## $BB s$SEED EVAL $M ##########"
  python3 -m med_merge.cli evaluate --model-path "$OUT/merged_core3/$M/merged_encoder.pt" \
    --backbone "$BID" $DS --head-dir "$OUT/checkpoints" \
    --output-dir "$OUT/results_core3/$M" --device cuda || echo "EVAL $M FAILED"
done
echo "CELL_DONE $BB s$SEED"
