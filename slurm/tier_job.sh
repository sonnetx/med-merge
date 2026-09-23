#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C "GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
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
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export TMPDIR=/scratch/users/$USER/tmp
mkdir -p "$TMPDIR"
cd "$PD"
: "${BB:?}"; : "${BID:?}"
DS="--datasets isic_mel --datasets chexpert_pe --datasets patchcamelyon"
STEP_TIMEOUT=3600
for SEED in 42 123 456; do
  OUT="$PD/outputs/$BB/seed_${SEED}_cap2k"
  echo "=== TIER $BB seed_$SEED ==="
  timeout $STEP_TIMEOUT python scripts/tier_merge.py --backbone-id "$BID" --cell "$OUT" \
    || { echo "FAILED tier_merge s$SEED rc=$?"; continue; }
  for M in ta_sum_notune gram_ls_notune metagpt_norm nan_invnorm; do
    R="$OUT/results_binary_tier/$M/results.json"
    [ -f "$R" ] && { echo "SKIP $M s$SEED (done)"; continue; }
    echo "########## $BB s$SEED EVAL $M ##########"
    timeout $STEP_TIMEOUT python3 -m med_merge.cli evaluate \
      --model-path "$OUT/merged_binary_tier/$M/merged_encoder.pt" \
      --backbone "$BID" $DS --head-dir "$OUT/checkpoints" \
      --output-dir "$OUT/results_binary_tier/$M" --device cuda
    rc=$?
    [ $rc -eq 124 ] && echo "TIMEOUT eval $M s$SEED"
    [ $rc -ne 0 ] && [ $rc -ne 124 ] && echo "FAILED eval $M s$SEED rc=$rc"
  done
done
echo "TIER_DONE $BB"
