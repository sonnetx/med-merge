#!/bin/bash
#SBATCH --job-name=medmerge_analysis
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Full analysis pipeline on a compute node (the login node OOM-kills these).
# Reads existing task_vector.pt + results.json files; writes geometry / LOBO /
# permutation / figure outputs into outputs/_figures/.

set -euo pipefail

ml gcc/12.4.0
ml python/3.12.1
ml cuda/12.4.0

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/med-merge}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/scripts:${PYTHONPATH:-}"

cd "$PROJECT_DIR"
mkdir -p logs outputs/_figures

echo "Python: $(which python3) -- $(python3 --version)"
echo "PWD: $PWD"
echo ""

# Geometry features computed per-(backbone, seed) to keep peak RAM bounded.
echo "===== Step 1/5: compute_geometry.py ====="
for bb in clip vit dinov3 rad_dino dinov2 mae beit; do
    for s in 42 123 456; do
        python scripts/compute_geometry.py --backbones "$bb" --seeds "$s" \
            || echo "  WARNING: geometry failed for $bb/$s"
    done
done

echo ""
echo "===== Step 2/5: predict_forgetting_lobo.py ====="
python scripts/predict_forgetting_lobo.py --seeds 42 123 456

echo ""
echo "===== Step 3/5: lobo_stats.py ====="
python scripts/lobo_stats.py --seeds 42 123 456 --n-perm 1000

echo ""
echo "===== Step 4/5: isic_vs_tcga_geometry.py ====="
python scripts/isic_vs_tcga_geometry.py --seed 42

echo ""
echo "===== Step 5/5: make_lobo_figure.py ====="
python scripts/make_lobo_figure.py

echo ""
echo "===== Done. Output files: ====="
ls -la outputs/_figures/ | tail -20
