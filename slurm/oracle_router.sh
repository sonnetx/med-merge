#!/bin/bash
#SBATCH --job-name=medmerge_oracle
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Oracle router baseline: evaluate each specialist on its own test set.
# Produces the upper bound for merged-model performance.

set -euo pipefail

ml gcc/12.4.0
ml python/3.12.1
ml cuda/12.4.0

[ -f "$HOME/.secrets" ] && source "$HOME/.secrets"

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/med-merge}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

cd "$PROJECT_DIR"
mkdir -p logs outputs/_figures

echo "Python: $(which python3) — $(python3 --version)"
echo "Running oracle router..."
python3 scripts/compute_oracle_router.py
echo ""
echo "Done. Results: outputs/_figures/oracle_router_*.{csv,md}"
