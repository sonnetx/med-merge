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

# Native-task ablation, one variable.
#
# Both arms use the SAME three datasets (ISIC-2017, CheXpert, PathMNIST) at the SAME
# 2k training cap, on the same backbone and seed. Only the label structure differs:
#   ARM=binary  -> melanoma vs rest, effusion vs rest, TUM vs rest      (all binary, AUROC)
#   ARM=native  -> 3-class lesion, 5-label finding, 9-class tissue      (native task forms)
# This is what makes the comparison attributable to task formulation rather than to a
# dataset swap or a size difference.
: "${BB:?}"; : "${BID:?}"; : "${SEED:?}"; : "${ARM:?}"
CAP=2000

case "$ARM" in
  binary) TRIO="isic_mel chexpert_pe pathmnist_bin" ;;
  native) TRIO="isic2017 chexpert pathmnist" ;;
  *) echo "ARM must be binary|native"; exit 2 ;;
esac

OUT="$PD/outputs/$BB/seed_${SEED}_abl_${ARM}"
echo "=== ABLATION CELL arm=$ARM $BB seed_$SEED ($BID) cap=$CAP ==="
echo "=== trio: $TRIO ==="

for D in $TRIO; do
  if [ -f "$OUT/task_vectors/$D/task_vector.pt" ]; then echo "SKIP train $D"; continue; fi
  echo "########## $ARM $BB s$SEED TRAIN $D ##########"
  python3 -m med_merge.cli train --dataset "$D" --backbone "$BID" --seed "$SEED" \
    --output-dir "$OUT" --max-train-samples "$CAP" --device cuda || echo "TRAIN $D FAILED"
done

DS=""
for D in $TRIO; do DS="$DS --datasets $D"; done

# Gram-LS is the method under test; TSV-Merge is the low-rank comparator that stayed
# robust in the original observation; task arithmetic anchors the tuned-sum baseline.
for M in gram_ls tsv_merge task_arithmetic; do
  R="$OUT/results/$M/results.json"
  if [ -f "$R" ]; then echo "SKIP $M (done)"; continue; fi
  echo "########## $ARM $BB s$SEED MERGE $M ##########"
  python3 -m med_merge.cli merge --method "$M" --backbone "$BID" $DS --hyperopt \
    --checkpoint-dir "$OUT/checkpoints" --task-vector-dir "$OUT/task_vectors" \
    --output-dir "$OUT/merged" --device cuda || { echo "MERGE $M FAILED"; continue; }
  echo "########## $ARM $BB s$SEED EVAL $M ##########"
  python3 -m med_merge.cli evaluate --model-path "$OUT/merged/$M/merged_encoder.pt" \
    --backbone "$BID" $DS --head-dir "$OUT/checkpoints" \
    --output-dir "$OUT/results/$M" --device cuda || echo "EVAL $M FAILED"
done

echo "ABL_CELL_DONE arm=$ARM $BB s$SEED"
