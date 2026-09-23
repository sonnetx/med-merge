#!/bin/bash
#SBATCH --job-name=core3_vit42
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
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export TMPDIR=/scratch/users/$USER/tmp
mkdir -p "$TMPDIR"
cd "$PD"
BID=google/vit-base-patch16-224
OUT="$PD/outputs/vit/seed_42"
DS="--datasets isic2017 --datasets chexpert --datasets pathmnist"
for M in simple_avg task_arithmetic ties dare dare_ties pcb_merging lines fisher iso_c tsv_merge gram_ls; do
  echo "########## MERGE $M ##########"
  python3 -m med_merge.cli merge --method "$M" --backbone "$BID" $DS --hyperopt \
    --checkpoint-dir "$OUT/checkpoints" --task-vector-dir "$OUT/task_vectors" \
    --output-dir "$OUT/merged_core3" --device cuda || echo "MERGE $M FAILED"
  echo "########## EVAL $M ##########"
  python3 -m med_merge.cli evaluate --model-path "$OUT/merged_core3/$M/merged_encoder.pt" \
    --backbone "$BID" $DS --head-dir "$OUT/checkpoints" \
    --output-dir "$OUT/results_core3/$M" --device cuda || echo "EVAL $M FAILED"
done
echo CORE3_VIT42_DONE
