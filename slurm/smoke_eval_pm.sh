#!/bin/bash
#SBATCH --job-name=smoke_pm
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -uo pipefail
ml gcc/12.4.0 python/3.12.1 cuda/12.4.0
PD=.
source $PD/venv/bin/activate
export PYTHONPATH="$PD/src" HF_HOME=/scratch/users/$USER/huggingface TORCH_HOME=/scratch/users/$USER/torch TMPDIR=/scratch/users/$USER/tmp
mkdir -p $TMPDIR; cd $PD
BID=google/vit-base-patch16-224; OUT=$PD/outputs/vit/seed_42
for M in simple_avg iso_c tsv_merge gram_ls; do
  echo "##### EVAL $M (pathmnist) #####"
  python3 -m med_merge.cli evaluate --model-path $OUT/merged_core3_cpu/$M/merged_encoder.pt     --backbone $BID --datasets pathmnist --head-dir $OUT/checkpoints     --output-dir $OUT/results_pm_smoke/$M --device cuda 2>&1 | grep -E "pathmnist|Error|error" || echo "$M FAILED"
done
echo ALL_DONE
