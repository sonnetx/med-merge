#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C "GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -uo pipefail
ml gcc/12.4.0 python/3.12.1 cuda/12.4.0
PD=.
source "$PD/venv/bin/activate"
export PYTHONPATH="$PD/src"
export HF_HOME=/scratch/users/$USER/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export TMPDIR=/scratch/users/$USER/tmp
cd "$PD"
: "${HELDOUT:?}"; : "${SEED:?}"; : "${DDIR:?}"
timeout 18000 python scripts/heldout_probe.py --heldout "$HELDOUT" --data-dir "$DDIR" \
  --seed "$SEED" --out "heldout_${HELDOUT}_s${SEED}.json"
echo "PROBE_DONE $HELDOUT s$SEED"
