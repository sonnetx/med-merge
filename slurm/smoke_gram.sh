#!/bin/bash
#SBATCH --job-name=smoke_gram
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C "GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
ml gcc/12.4.0 python/3.12.1 cuda/12.4.0
PD=.
source $PD/venv/bin/activate
export PYTHONPATH="$PD/src:${PYTHONPATH:-}"
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets TORCH_HOME=/scratch/users/$USER/torch TMPDIR=/scratch/users/$USER/tmp
mkdir -p $TMPDIR $HF_HOME $TORCH_HOME
cd $PD
BID=google/vit-base-patch16-224
OUT=$PD/outputs/vit/seed_42
DS="--datasets isic2017 --datasets chexpert --datasets pathmnist"
for M in iso_c tsv_merge gram_ls; do
  echo "########## MERGE $M ##########"
  python3 -m med_merge.cli merge --method $M --backbone $BID $DS --hyperopt     --checkpoint-dir $OUT/checkpoints --task-vector-dir $OUT/task_vectors     --output-dir $OUT/merged_core3 --device cuda || echo "MERGE $M FAILED"
  echo "########## EVAL $M ##########"
  python3 -m med_merge.cli evaluate --model-path $OUT/merged_core3/$M/merged_encoder.pt     --backbone $BID $DS --head-dir $OUT/checkpoints     --output-dir $OUT/results_core3/$M --device cuda || echo "EVAL $M FAILED"
done
echo "ALL DONE"
