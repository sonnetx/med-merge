#!/bin/bash
#SBATCH --job-name=gram_cond
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
python scripts/gram_cond.py --out gram_conditioning_full.json
echo GRAM_COND_DONE
