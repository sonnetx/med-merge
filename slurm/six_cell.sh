#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C "GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --time=20:00:00
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
# Weights and datasets are cached; going offline avoids the hub calls that previously hung
# jobs indefinitely while SLURM still reported them RUNNING.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_ETAG_TIMEOUT=10
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export TMPDIR=/scratch/users/$USER/tmp
mkdir -p "$TMPDIR"
cd "$PD"

# Six binary tasks, two per domain. Doubling the task count is secondary here; the point is
# that the within-domain pairs are correlated, which raises the off-diagonal terms of the
# task-vector Gram matrix. Every setting measured so far sits at kappa ~= 1.4, where Gram-LS
# provably reduces to a scaled sum and therefore cannot differ from task arithmetic. This is
# the regime the method was designed for.
: "${BB:?}"; : "${BID:?}"; : "${SEED:?}"
CAP=2000
OUT="$PD/outputs/$BB/seed_${SEED}_cap2k"
SIX="isic_mel ham10000_mel chexpert_pe chexpert_cm patchcamelyon pathmnist_bin"
STEP_TIMEOUT=${STEP_TIMEOUT:-16000}

echo "=== SIX-TASK CELL $BB seed_$SEED ($BID) ==="

# Fail before touching anything if the backbone cannot load from cache.
python3 - "$BID" <<'PYCHK'
import sys
from med_merge.models.factory import BACKBONE_REGISTRY, create_model
from med_merge.config.schema import ModelConfig, DatasetConfig
bid = sys.argv[1]
create_model(DatasetConfig(name="isic_mel", num_classes=1, task_type="binary"),
             model_config=ModelConfig(backbone=bid, hidden_size=BACKBONE_REGISTRY[bid][1], num_layers=12))
print("backbone loads offline OK")
PYCHK
[ $? -ne 0 ] && { echo "ABORT: $BID cannot load offline"; exit 3; }

# The three original specialists already exist in this cell and are reused.
for D in $SIX; do
  if [ -f "$OUT/task_vectors/$D/task_vector.pt" ]; then echo "SKIP train $D (exists)"; continue; fi
  echo "########## $BB s$SEED TRAIN $D ##########"
  timeout $STEP_TIMEOUT python3 -m med_merge.cli train --dataset "$D" --backbone "$BID" --seed "$SEED" \
    --output-dir "$OUT" --max-train-samples "$CAP" --device cuda || echo "TRAIN $D FAILED rc=$?"
done

DS=""
for D in $SIX; do DS="$DS --datasets $D"; done

for M in gram_ls task_arithmetic tsv_merge iso_c iso_cts simple_avg; do
  echo "########## $BB s$SEED MERGE6 $M ##########"
  # Results are overwritten on success, never deleted first.
  timeout $STEP_TIMEOUT python3 -m med_merge.cli merge --method "$M" --backbone "$BID" $DS --hyperopt \
    --checkpoint-dir "$OUT/checkpoints" --task-vector-dir "$OUT/task_vectors" \
    --output-dir "$OUT/merged_binary6" --device cuda
  rc=$?
  [ $rc -eq 124 ] && { echo "TIMEOUT merge $M"; continue; }
  [ $rc -ne 0 ] && { echo "FAILED merge $M rc=$rc"; continue; }

  timeout $STEP_TIMEOUT python3 -m med_merge.cli evaluate \
    --model-path "$OUT/merged_binary6/$M/merged_encoder.pt" \
    --backbone "$BID" $DS --head-dir "$OUT/checkpoints" \
    --output-dir "$OUT/results_binary6/$M" --device cuda
  rc=$?
  [ $rc -eq 124 ] && echo "TIMEOUT eval $M"
  [ $rc -ne 0 ] && [ $rc -ne 124 ] && echo "FAILED eval $M rc=$rc"
done

echo "SIX_CELL_DONE $BB s$SEED"
