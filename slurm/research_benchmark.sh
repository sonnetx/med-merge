#!/bin/bash
#SBATCH --job-name=medmerge_research
#SBATCH --partition=roxanad
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -C GPU_MEM:80GB
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==========================================================================
# med-merge Research Benchmark
#
# Full experimental grid for the paper:
#   3 backbones x 3 datasets x 3 seeds x 9 merge methods
#
# Answers all three research questions:
#   RQ1: Can merging preserve performance across medical domains?
#   RQ2: How does merge quality vary with lambda/sparsity? (hyperopt)
#   RQ3: Does the base model matter? (multi-backbone)
#
# Usage:
#   sbatch slurm/research_benchmark.sh                         # full run
#   sbatch --export=ALL,BACKBONES=clip slurm/research_benchmark.sh  # single backbone
#   sbatch --export=ALL,SEEDS="42" slurm/research_benchmark.sh     # single seed
#   sbatch --export=ALL,SKIP_TRAIN=1 slurm/research_benchmark.sh   # merge+eval only
# ==========================================================================

set -euo pipefail

# --- Modules ---
ml gcc/12.4.0
ml python/3.12.1
ml cuda/12.4.0

# --- Secrets (HF_TOKEN for DINOv3 gated model, WANDB_API_KEY, etc.) ---
[ -f "$HOME/.secrets" ] && source "$HOME/.secrets"

# Auto-enable wandb if API key is available
if [ -z "${WANDB_MODE:-}" ] && [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_MODE="online"
fi

echo "Python: $(which python3) — $(python3 --version)"

# --- Paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/med-merge}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
BASE_OUTPUT_DIR="${PROJECT_DIR}/outputs"

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

# --- Experimental grid ---
DATASETS="${DATASETS:-isic2017 chexpert tcga}"
SEEDS="${SEEDS:-42 123 456}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
DEVICE="cuda"

# Backbone short names -> HuggingFace model IDs
# Override with e.g. BACKBONES="clip vit"
BACKBONES="${BACKBONES:-clip vit dinov3}"

declare -A BACKBONE_IDS
BACKBONE_IDS[clip]="openai/clip-vit-base-patch16"
BACKBONE_IDS[vit]="google/vit-base-patch16-224"
BACKBONE_IDS[dinov3]="facebook/dinov3-vits16-pretrain-lvd1689m"

# Methods to run (skip slerp by default — needs exactly 2 tasks)
METHODS="simple_avg task_arithmetic ties dare dare_ties pcb_merging lines fisher"

echo "============================================================"
echo "med-merge Research Benchmark"
echo "Backbones: $BACKBONES"
echo "Datasets:  $DATASETS"
echo "Seeds:     $SEEDS"
echo "Methods:   $METHODS"
echo "============================================================"

# ==========================================================================
# Phase 1: Train all (backbone x dataset x seed)
# ==========================================================================
if [ "$SKIP_TRAIN" -eq 0 ]; then
    for BACKBONE in $BACKBONES; do
        BACKBONE_ID="${BACKBONE_IDS[$BACKBONE]}"
        for SEED in $SEEDS; do
            OUTPUT_DIR="$BASE_OUTPUT_DIR/$BACKBONE/seed_$SEED"

            for DS in $DATASETS; do
                echo ""
                echo "============================================================"
                echo "[Train] backbone=$BACKBONE seed=$SEED dataset=$DS"
                echo "============================================================"

                # Skip if task vector already exists
                if [ -f "$OUTPUT_DIR/task_vectors/$DS/task_vector.pt" ]; then
                    echo "  Already trained, skipping."
                    continue
                fi

                python3 -m med_merge.cli train \
                    --dataset "$DS" \
                    --backbone "$BACKBONE_ID" \
                    --seed "$SEED" \
                    --output-dir "$OUTPUT_DIR" \
                    --device "$DEVICE" \
                    --wandb-mode "${WANDB_MODE:-disabled}" \
                    || echo "  WARNING: Training failed for $DS, continuing..."
            done
        done
    done
else
    echo "Skipping training (SKIP_TRAIN=1)"
fi

# ==========================================================================
# Phase 2: Merge (backbone x seed x method)
# ==========================================================================
echo ""
echo "============================================================"
echo "[Merge] All backbones x seeds x methods"
echo "============================================================"

# Helper: convert space-separated list to repeated --datasets flags
ds_flags() {
    for ds in $1; do echo -n "--datasets $ds "; done
}

for BACKBONE in $BACKBONES; do
    BACKBONE_ID="${BACKBONE_IDS[$BACKBONE]}"
    for SEED in $SEEDS; do
        OUTPUT_DIR="$BASE_OUTPUT_DIR/$BACKBONE/seed_$SEED"

        # Find which datasets have task vectors for this backbone/seed
        TRAINED_DS=""
        for DS in $DATASETS; do
            if [ -f "$OUTPUT_DIR/task_vectors/$DS/task_vector.pt" ]; then
                TRAINED_DS="$TRAINED_DS $DS"
            fi
        done

        if [ -z "$TRAINED_DS" ]; then
            echo "  No task vectors for backbone=$BACKBONE seed=$SEED, skipping merge."
            continue
        fi

        DS_FLAGS=$(ds_flags "$TRAINED_DS")

        for METHOD in $METHODS; do
            echo "--- merge: backbone=$BACKBONE seed=$SEED method=$METHOD ---"

            if [ -f "$OUTPUT_DIR/merged/$METHOD/merged_encoder.pt" ]; then
                echo "  Already merged, skipping."
                continue
            fi

            python3 -m med_merge.cli merge \
                --method "$METHOD" \
                --backbone "$BACKBONE_ID" \
                $DS_FLAGS \
                --hyperopt \
                --checkpoint-dir "$OUTPUT_DIR/checkpoints" \
                --task-vector-dir "$OUTPUT_DIR/task_vectors" \
                --output-dir "$OUTPUT_DIR/merged" \
                --device "$DEVICE" \
                || echo "  WARNING: Merge failed, continuing..."
        done
    done
done

# ==========================================================================
# Phase 3: Evaluate (backbone x seed x method)
# ==========================================================================
echo ""
echo "============================================================"
echo "[Evaluate] All merged models"
echo "============================================================"

for BACKBONE in $BACKBONES; do
    BACKBONE_ID="${BACKBONE_IDS[$BACKBONE]}"
    for SEED in $SEEDS; do
        OUTPUT_DIR="$BASE_OUTPUT_DIR/$BACKBONE/seed_$SEED"

        TRAINED_DS=""
        for DS in $DATASETS; do
            if [ -f "$OUTPUT_DIR/task_vectors/$DS/task_vector.pt" ]; then
                TRAINED_DS="$TRAINED_DS $DS"
            fi
        done

        DS_FLAGS=$(ds_flags "$TRAINED_DS")

        for METHOD in $METHODS; do
            MERGED_PATH="$OUTPUT_DIR/merged/$METHOD/merged_encoder.pt"
            if [ ! -f "$MERGED_PATH" ]; then
                continue
            fi

            echo "--- eval: backbone=$BACKBONE seed=$SEED method=$METHOD ---"

            python3 -m med_merge.cli evaluate \
                --model-path "$MERGED_PATH" \
                --backbone "$BACKBONE_ID" \
                $DS_FLAGS \
                --head-dir "$OUTPUT_DIR/checkpoints" \
                --output-dir "$OUTPUT_DIR/results/$METHOD" \
                --device "$DEVICE" \
                || echo "  WARNING: Eval failed, continuing..."
        done
    done
done

# ==========================================================================
# Phase 4: Summary
# ==========================================================================
echo ""
echo "============================================================"
echo "Research benchmark complete!"
echo ""
echo "Output structure:"
echo "  outputs/"
echo "    clip/seed_42/     — CLIP ViT-B/16 results"
echo "    clip/seed_123/"
echo "    clip/seed_456/"
echo "    vit/seed_42/      — ViT-B/16 results"
echo "    ..."
echo "    dinov3/seed_42/   — DINOv3 results"
echo "    ..."
echo ""
echo "Each contains: checkpoints/ task_vectors/ merged/ results/"
echo ""

# Count completed experiments
TOTAL=0
for BACKBONE in $BACKBONES; do
    for SEED in $SEEDS; do
        for METHOD in $METHODS; do
            RESULTS="$BASE_OUTPUT_DIR/$BACKBONE/seed_$SEED/results/$METHOD/results.json"
            if [ -f "$RESULTS" ]; then
                TOTAL=$((TOTAL + 1))
            fi
        done
    done
done
EXPECTED=$(( $(echo $BACKBONES | wc -w) * $(echo $SEEDS | wc -w) * $(echo $METHODS | wc -w) ))
echo "Completed: $TOTAL / $EXPECTED experiments"
echo "============================================================"
