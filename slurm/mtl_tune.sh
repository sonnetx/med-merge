#!/bin/bash
#SBATCH --job-name=medmerge_mtl_tune
#SBATCH --partition=roxanad,gpu
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Tuned-MTL sweep

set -euo pipefail

ml gcc/12.4.0
ml python/3.12.1
ml cuda/12.4.0

[ -f "$HOME/.secrets" ] && source "$HOME/.secrets"
if [ -z "${WANDB_MODE:-}" ] && [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_MODE="online"
fi

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/med-merge}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs_tuned}"

BACKBONES="${BACKBONES:-clip vit dinov3 rad_dino}"
SEEDS="${SEEDS:-42 123 456}"
DATASETS="${DATASETS:-isic2017 chexpert tcga nih_cxr}"
EPOCHS="${EPOCHS:-16}"
SAMPLING="${SAMPLING:-inverse_frequency}"
DEVICE="cuda"

declare -A BACKBONE_IDS
BACKBONE_IDS[clip]="openai/clip-vit-base-patch16"
BACKBONE_IDS[vit]="google/vit-base-patch16-224"
BACKBONE_IDS[dinov3]="facebook/dinov3-vits16-pretrain-lvd1689m"
BACKBONE_IDS[rad_dino]="microsoft/rad-dino"
BACKBONE_IDS[dinov2]="facebook/dinov2-base"
BACKBONE_IDS[mae]="facebook/vit-mae-base"
BACKBONE_IDS[beit]="microsoft/beit-base-patch16-224-pt22k-ft22k"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"

ds_flags() { for ds in $1; do echo -n "--datasets $ds "; done; }
DS_FLAGS=$(ds_flags "$DATASETS")

echo "Tuned MTL sweep"
echo "  Backbones: $BACKBONES"
echo "  Seeds:     $SEEDS"
echo "  Datasets:  $DATASETS"
echo "  Epochs:    $EPOCHS"
echo "  Sampling:  $SAMPLING"
echo "  Output:    $OUTPUT_DIR"

for BACKBONE in $BACKBONES; do
    BACKBONE_ID="${BACKBONE_IDS[$BACKBONE]}"
    for SEED in $SEEDS; do
        echo ""
        echo "[MTL-tune] backbone=$BACKBONE seed=$SEED epochs=$EPOCHS sampling=$SAMPLING"

        if [ -f "$OUTPUT_DIR/$BACKBONE/seed_$SEED/mtl/best_metrics.json" ]; then
            echo "  Already trained, skipping."
            continue
        fi

        python3 -m med_merge.cli train-mtl \
            $DS_FLAGS \
            --backbone "$BACKBONE_ID" \
            --seed "$SEED" \
            --output-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS" \
            --sampling "$SAMPLING" \
            --device "$DEVICE" \
            --wandb-mode "${WANDB_MODE:-disabled}" \
            || echo "  WARNING: tuned MTL failed for $BACKBONE/$SEED"
    done
done

echo ""
echo "Tuned MTL sweep complete."
