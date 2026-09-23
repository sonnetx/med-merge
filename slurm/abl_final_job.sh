#!/bin/bash -l
#SBATCH --job-name=abl_final
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -uo pipefail
ml python/3.12.1
PD=.
source "$PD/venv/bin/activate"
export PYTHONPATH="$PD/src"
cd "$PD"
python scripts/abl_final.py
echo ABL_FINAL_DONE
