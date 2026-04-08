#!/bin/bash
#SBATCH --job-name=medmerge_smoke
#SBATCH --partition=roxanad
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -C GPU_MEM:80GB
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# med-merge Smoke Test
#
# Quick end-to-end pipeline test: train 1 epoch on 2 datasets
# (ISIC 2017 + CheXpert), merge with simple averaging, evaluate.
#
# Usage:
#   sbatch slurm/smoke_test.sh
#   sbatch --export=ALL,EPOCHS=3 slurm/smoke_test.sh

set -euo pipefail

# --- Modules ---
ml gcc/12.4.0
ml python/3.12.1
ml cuda/12.4.0

# --- Secrets (HF_TOKEN for DINOv3, etc.) ---
[ -f "$HOME/.secrets" ] && source "$HOME/.secrets"

echo "Python: $(which python3) — $(python3 --version)"

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/med-merge}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
OUTPUT_DIR="${PROJECT_DIR}/outputs"

# --- Config ---
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="cuda"

# --- Activate environment ---
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: No venv found at $VENV_DIR. Run setup.sh first:"
    echo "  sbatch slurm/setup.sh"
    exit 1
fi
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# Cache dirs — keep on scratch
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME"

cd "$PROJECT_DIR"
mkdir -p logs

echo "============================================================"
echo "med-merge Smoke Test"
echo "Epochs: $EPOCHS | Batch size: $BATCH_SIZE"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Dataset configs have the real Sherlock paths baked in,
# so we don't need --data-dir at all — just --dataset.

# =====================================================================
# Step 1: Train on ISIC 2017 (dermatology, 3-class, local)
# =====================================================================
echo "[Step 1/5] Training on ISIC 2017 ($EPOCHS epoch(s))..."
python3 -m med_merge.cli train \
    --dataset isic2017 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --wandb-mode disabled

echo "ISIC 2017 training done."
echo ""

# =====================================================================
# Step 2: Train on CheXpert (radiology, 5-label multilabel, local)
# =====================================================================
echo "[Step 2/5] Training on CheXpert ($EPOCHS epoch(s))..."
python3 -m med_merge.cli train \
    --dataset chexpert \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --wandb-mode disabled

echo "CheXpert training done."
echo ""

# =====================================================================
# Step 3: Merge with Simple Averaging
# =====================================================================
echo "[Step 3/5] Merging with simple_avg..."
python3 -m med_merge.cli merge \
    --method simple_avg \
    --datasets isic2017 \
    --datasets chexpert \
    --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
    --task-vector-dir "$OUTPUT_DIR/task_vectors" \
    --output-dir "$OUTPUT_DIR/merged" \
    --device "$DEVICE"

echo "Merging done."
echo ""

# =====================================================================
# Step 4: Evaluate merged model
# =====================================================================
MERGED_PATH="$OUTPUT_DIR/merged/simple_avg/merged_encoder.pt"
echo "[Step 4/5] Evaluating merged model..."
python3 -m med_merge.cli evaluate \
    --model-path "$MERGED_PATH" \
    --datasets isic2017 \
    --datasets chexpert \
    --head-dir "$OUTPUT_DIR/checkpoints" \
    --output-dir "$OUTPUT_DIR/results/simple_avg" \
    --device "$DEVICE"

echo "Evaluation done."
echo ""

# =====================================================================
# Step 5: Summary
# =====================================================================
echo "[Step 5/5] Results summary:"
echo "------------------------------------------------------------"
if [ -f "$OUTPUT_DIR/results/simple_avg/results.json" ]; then
    python3 -c "
import json
with open('$OUTPUT_DIR/results/simple_avg/results.json') as f:
    results = json.load(f)
for ds, metrics in results.items():
    print(f'  {ds}:')
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f'    {k}: {v:.4f}')
"
else
    echo "  No results file found."
fi
echo "------------------------------------------------------------"

echo ""
echo "============================================================"
echo "Smoke test complete!"
echo "Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "Task vectors: $OUTPUT_DIR/task_vectors/"
echo "Merged model: $MERGED_PATH"
echo "Results: $OUTPUT_DIR/results/simple_avg/"
echo "============================================================"
