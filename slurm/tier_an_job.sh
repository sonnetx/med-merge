#!/bin/bash -l
#SBATCH --job-name=tier_analysis
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
ml python/3.12.1
PD=.
source "$PD/venv/bin/activate"
export PYTHONPATH="$PD/src"
cd "$PD"
python scripts/tier_analysis.py
echo KAPPA_PREVIEW_DONE
