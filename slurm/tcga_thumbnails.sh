#!/bin/bash
#SBATCH --job-name=tcga_thumbnails
#SBATCH --partition=roxanad,normal
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SCRATCH_USER="/scratch/users/$USER"
SIF_STORE="$SCRATCH_USER/simg"
SIF_IMAGE="${SIF_IMAGE:-tcga_thumb.sif}"
DEF_FILE="$PROJECT_DIR/slurm/tcga_thumb.def"

OUT_DIR="${OUT_DIR:-/oak/stanford/groups/roxanad/sonnet/tcga}"
LUAD_SVS="${LUAD_SVS:-/oak/stanford/groups/roxanad/wsi-datasets/tcga/luad/svs}"
LUSC_SVS="${LUSC_SVS:-/oak/stanford/groups/roxanad/wsi-datasets/tcga/lusc/svs}"
SIZE="${SIZE:-512}"

TOOL=$(command -v apptainer || command -v singularity)
if [ -z "$TOOL" ]; then
    echo "ERROR: apptainer/singularity not found on PATH" >&2
    exit 1
fi

mkdir -p "$SIF_STORE" "$PROJECT_DIR/logs" "$OUT_DIR/thumbnails" "$OUT_DIR/tables"

if [ ! -f "$SIF_STORE/$SIF_IMAGE" ]; then
    echo "Building $SIF_STORE/$SIF_IMAGE from $DEF_FILE"
    cd "$SIF_STORE"
    $TOOL build --fakeroot "$SIF_IMAGE" "$DEF_FILE" \
        || $TOOL build "$SIF_IMAGE" "$DEF_FILE"
    cd "$PROJECT_DIR"
fi

echo "TCGA thumbnail build"
echo "  Container:  $SIF_STORE/$SIF_IMAGE"
echo "  LUAD SVS:   $LUAD_SVS"
echo "  LUSC SVS:   $LUSC_SVS"
echo "  Output:     $OUT_DIR"
echo "  Size:       ${SIZE}x${SIZE}"

"$TOOL" exec \
    -B "$PROJECT_DIR:/workspace" \
    -B "$OUT_DIR:/out" \
    -B "$LUAD_SVS:/svs/luad" \
    -B "$LUSC_SVS:/svs/lusc" \
    --pwd /workspace \
    "$SIF_STORE/$SIF_IMAGE" \
    python scripts/tcga_build_thumbnails.py \
        --luad-svs-dir /svs/luad \
        --lusc-svs-dir /svs/lusc \
        --out-dir /out \
        --size "$SIZE" \
        --workers 16

echo ""
echo "Thumbnail count:"
ls "$OUT_DIR/thumbnails" 2>/dev/null | wc -l
echo "CSV head:"
head -3 "$OUT_DIR/tables/dataset.csv" 2>/dev/null || echo "no csv"
