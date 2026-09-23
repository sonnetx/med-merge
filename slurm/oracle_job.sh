#!/bin/bash -l
#SBATCH --job-name=oracle_bin
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C "GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --time=08:00:00
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
export HF_HUB_ETAG_TIMEOUT=10
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export TMPDIR=/scratch/users/$USER/tmp
cd "$PD"
python scripts/compute_oracle_router.py \
  --backbones clip vit dinov3 rad_dino dinov2 mae beit \
  --datasets isic_mel chexpert_pe patchcamelyon \
  --seeds 42 123 456 --cell-suffix _cap2k --out-prefix oracle_binary
echo ORACLE_DONE
