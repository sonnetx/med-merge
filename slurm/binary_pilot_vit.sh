#!/bin/bash
#SBATCH --job-name=bin_pilot_vit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --time=05:00:00
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
BID=google/vit-base-patch16-224
OUT="$PD/outputs/vit/seed_42"
CAP=8000

# 1. Train binary specialists (skip if task vector already present)
for D in isic_mel chexpert_pe patchcamelyon; do
  if [ -f "$OUT/task_vectors/$D/task_vector.pt" ]; then echo "SKIP train $D (task vector exists)"; continue; fi
  echo "########## TRAIN $D (cap=$CAP) ##########"
  python3 -m med_merge.cli train --dataset "$D" --backbone "$BID" --seed 42 \
    --output-dir "$OUT" --max-train-samples "$CAP" --device cuda || echo "TRAIN $D FAILED"
done

# 2 & 3. Merge + eval the binary trio
DS="--datasets isic_mel --datasets chexpert_pe --datasets patchcamelyon"
for M in simple_avg task_arithmetic iso_c tsv_merge gram_ls gram_ls_na; do
  R="$OUT/results_binary/$M/results.json"
  if [ -f "$R" ]; then echo "SKIP $M (done)"; continue; fi
  echo "########## MERGE $M ##########"
  python3 -m med_merge.cli merge --method "$M" --backbone "$BID" $DS --hyperopt \
    --checkpoint-dir "$OUT/checkpoints" --task-vector-dir "$OUT/task_vectors" \
    --output-dir "$OUT/merged_binary" --device cuda || { echo "MERGE $M FAILED"; continue; }
  echo "########## EVAL $M ##########"
  python3 -m med_merge.cli evaluate --model-path "$OUT/merged_binary/$M/merged_encoder.pt" \
    --backbone "$BID" $DS --head-dir "$OUT/checkpoints" \
    --output-dir "$OUT/results_binary/$M" --device cuda || echo "EVAL $M FAILED"
done
echo BINARY_PILOT_DONE
