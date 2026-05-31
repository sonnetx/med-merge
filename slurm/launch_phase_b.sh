#!/bin/bash
# Phase B launch: MedMNIST download, then new-backbone and new-dataset training
# jobs that depend on it via afterok. Override SEEDS / NEW_BACKBONES /
# NEW_DATASETS / DRY_RUN via env.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/med-merge}"
cd "$PROJECT_DIR"
mkdir -p logs

SEEDS="${SEEDS:-42 123 456}"
NEW_BACKBONES="${NEW_BACKBONES:-dinov2 mae beit medclip}"
NEW_DATASETS="${NEW_DATASETS:-pathmnist retinamnist}"
ALL_BACKBONES="${ALL_BACKBONES:-clip vit dinov3 rad_dino dinov2 mae beit medclip}"
DRY_RUN="${DRY_RUN:-0}"

submit() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY: sbatch $*"
        echo "12345678"
    else
        sbatch "$@" | awk '{print $NF}'
    fi
}

echo "==================================================="
echo "Phase B launch"
echo "  seeds          = $SEEDS"
echo "  new backbones  = $NEW_BACKBONES   (on existing 4 datasets)"
echo "  new datasets   = $NEW_DATASETS   (on all 8 backbones)"
echo "  all backbones  = $ALL_BACKBONES"
echo "  dry run        = $DRY_RUN"
echo "==================================================="

echo ""
echo "[1] Submitting MedMNIST download ..."
DL_JOB=$(submit slurm/medmnist_download.sh)
echo "    Download job id: $DL_JOB"

DEP="--dependency=afterok:${DL_JOB}"

echo ""
echo "[2] Submitting new-backbone jobs (depend on $DL_JOB)..."
NB_IDS=()
for bb in $NEW_BACKBONES; do
    for s in $SEEDS; do
        # All training jobs gate on the download so we keep one dependency list;
        # new-backbone jobs don't read medmnist but the wait is ~5 min and worth
        # the simpler graph.
        ID=$(submit $DEP --export=ALL,BACKBONES=$bb,SEEDS=$s slurm/research_benchmark.sh)
        echo "    bb=$bb seed=$s  -> $ID"
        NB_IDS+=("$ID")
    done
done

echo ""
echo "[3] Submitting new-dataset jobs (depend on $DL_JOB)..."
ND_IDS=()
for ds in $NEW_DATASETS; do
    for bb in $ALL_BACKBONES; do
        for s in $SEEDS; do
            ID=$(submit $DEP --export=ALL,BACKBONES=$bb,SEEDS=$s,DATASETS=$ds slurm/research_benchmark.sh)
            echo "    ds=$ds bb=$bb seed=$s  -> $ID"
            ND_IDS+=("$ID")
        done
    done
done

echo ""
echo "==================================================="
echo "Phase B summary"
echo "  Download job:        $DL_JOB"
echo "  New-backbone jobs:   ${#NB_IDS[@]} submitted"
echo "  New-dataset jobs:    ${#ND_IDS[@]} submitted"
echo "  Total queued:        $((${#NB_IDS[@]} + ${#ND_IDS[@]} + 1))"
echo ""
echo "  Watch progress:"
echo "    squeue -u \$USER --format='%.10i %.20j %.8T %.10M %.20R'"
echo "    sacct -u \$USER --starttime=now-1day --format=JobID,JobName,State,Elapsed -P | head -30"
echo "==================================================="
