#!/bin/bash
#SBATCH --job-name=medmerge_full
#SBATCH --partition=roxanad
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH -C GPU_MEM:80GB
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# med-merge Full Benchmark
#
# Train all 3 datasets, merge with all 9 methods, evaluate everything.
#
# Usage:
#   sbatch slurm/full_benchmark.sh
#   sbatch --export=ALL,SKIP_TRAIN=1 slurm/full_benchmark.sh  # skip training

set -euo pipefail

# --- Modules ---
ml gcc/12.4.0
ml python/3.12.1
ml cuda/12.4.0

# --- Secrets ---
[ -f "$HOME/.secrets" ] && source "$HOME/.secrets"

echo "Python: $(which python3) — $(python3 --version)"

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/med-merge}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
OUTPUT_DIR="${PROJECT_DIR}/outputs"

# --- Config ---
SKIP_TRAIN="${SKIP_TRAIN:-0}"
DEVICE="cuda"

# Matches ALL_DATASETS and ALL_METHODS in config/constants.py
DATASETS="isic2017 chexpert tcga"
METHODS="simple_avg task_arithmetic ties dare dare_ties pcb_merging lines slerp fisher"

# --- Activate environment ---
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: No venv found at $VENV_DIR. Run setup.sh first."
    exit 1
fi

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# Cache dirs
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME"

cd "$PROJECT_DIR"
mkdir -p logs

echo "============================================================"
echo "med-merge Full Benchmark"
echo "Datasets: $DATASETS"
echo "Methods: $METHODS"
echo "============================================================"

# =====================================================================
# Phase 1: Train all datasets
# =====================================================================
if [ "$SKIP_TRAIN" -eq 0 ]; then
    for DS in $DATASETS; do
        echo ""
        echo "============================================================"
        echo "[Train] $DS"
        echo "============================================================"

        if [ -f "$OUTPUT_DIR/task_vectors/$DS/task_vector.pt" ]; then
            echo "  Task vector already exists, skipping training."
            continue
        fi

        python3 -m med_merge.cli train \
            --dataset "$DS" \
            --output-dir "$OUTPUT_DIR" \
            --device "$DEVICE" \
            --wandb-mode disabled

        if [ $? -ne 0 ]; then
            echo "  WARNING: Training failed for $DS, continuing..."
        fi
    done
else
    echo "Skipping training (SKIP_TRAIN=1)"
fi

# =====================================================================
# Phase 2: Merge with all methods
# =====================================================================
echo ""
echo "============================================================"
echo "[Merge] Running all merging methods"
echo "============================================================"

# Build list of datasets that have task vectors
TRAINED_DS=""
for DS in $DATASETS; do
    if [ -f "$OUTPUT_DIR/task_vectors/$DS/task_vector.pt" ]; then
        TRAINED_DS="$TRAINED_DS $DS"
    else
        echo "  WARNING: No task vector for $DS, excluding from merge."
    fi
done

if [ -z "$TRAINED_DS" ]; then
    echo "ERROR: No trained datasets found. Run training first."
    exit 1
fi

echo "  Merging datasets:$TRAINED_DS"

# Count trained datasets (SLERP needs exactly 2)
N_TRAINED=$(echo $TRAINED_DS | wc -w)

for METHOD in $METHODS; do
    echo ""
    echo "--- Merging: $METHOD ---"

    # SLERP only works with exactly 2 task vectors
    if [ "$METHOD" = "slerp" ] && [ "$N_TRAINED" -ne 2 ]; then
        echo "  Skipping SLERP (needs exactly 2 task vectors, have $N_TRAINED)"
        continue
    fi

    if [ -f "$OUTPUT_DIR/merged/$METHOD/merged_encoder.pt" ]; then
        echo "  Already merged, skipping."
        continue
    fi

    python3 -m med_merge.cli merge \
        --method "$METHOD" \
        --datasets $TRAINED_DS \
        --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
        --task-vector-dir "$OUTPUT_DIR/task_vectors" \
        --output-dir "$OUTPUT_DIR/merged" \
        --device "$DEVICE"

    if [ $? -ne 0 ]; then
        echo "  WARNING: Merge failed for $METHOD, continuing..."
    fi
done

# =====================================================================
# Phase 3: Evaluate all merged models
# =====================================================================
echo ""
echo "============================================================"
echo "[Evaluate] All merged models"
echo "============================================================"

for METHOD in $METHODS; do
    MERGED_PATH="$OUTPUT_DIR/merged/$METHOD/merged_encoder.pt"
    if [ ! -f "$MERGED_PATH" ]; then
        echo "  Skipping $METHOD (no merged model)"
        continue
    fi

    echo ""
    echo "--- Evaluating: $METHOD ---"

    python3 -m med_merge.cli evaluate \
        --model-path "$MERGED_PATH" \
        --datasets $TRAINED_DS \
        --head-dir "$OUTPUT_DIR/checkpoints" \
        --output-dir "$OUTPUT_DIR/results/$METHOD" \
        --device "$DEVICE"

    if [ $? -ne 0 ]; then
        echo "  WARNING: Evaluation failed for $METHOD"
    fi
done

# =====================================================================
# Phase 4: Generate report
# =====================================================================
echo ""
echo "============================================================"
echo "[Report] Generating tables and plots"
echo "============================================================"

python3 -m med_merge.cli report \
    --results-dir "$OUTPUT_DIR/results" \
    --output-dir "$OUTPUT_DIR/reports" \
    --format both

# =====================================================================
# Summary
# =====================================================================
echo ""
echo "============================================================"
echo "Full benchmark complete!"
echo ""
echo "Results directory: $OUTPUT_DIR/results/"
echo "Reports: $OUTPUT_DIR/reports/"
echo ""
echo "Per-method results:"
for METHOD in $METHODS; do
    RESULTS_FILE="$OUTPUT_DIR/results/$METHOD/results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "  $METHOD: $RESULTS_FILE"
    fi
done
echo "============================================================"
