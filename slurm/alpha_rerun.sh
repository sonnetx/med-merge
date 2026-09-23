#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH -C "GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --time=16:00:00
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
# Root cause of the previous stalls: transformers resolves refs over the network on every
# from_pretrained, and those requests can hang forever on a compute node while SLURM reports
# RUNNING. Weights are already cached, so force offline and cap any residual hub call.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ETAG_TIMEOUT=10
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch
export TMPDIR=/scratch/users/$USER/tmp
mkdir -p "$TMPDIR"
cd "$PD"

# Re-tune every alpha-searching method over the widened grid (max 2.5 rather than 0.9).
# The subspace methods selected 0.9, the old maximum, in 19 of 21 cells, so their optimum
# was outside the search and their reported numbers are lower bounds rather than tuned
# results. Hyperopt resumes from its cached trials, so only the new alphas actually run.
: "${BB:?}"; : "${BID:?}"; : "${SEED:?}"
OUT="$PD/outputs/$BB/seed_${SEED}_cap2k"
DS="--datasets isic_mel --datasets chexpert_pe --datasets patchcamelyon"

# Hard per-step ceiling. A step that exceeds it is killed and reported rather than
# silently holding the GPU for the rest of the walltime.
# Sized for the largest grid: dare_ties is 11 alphas x 3 drop rates x 3 trim fractions = 99
# trials at roughly 95s each, so 3600s cut it off mid-search. Overridable per job.
STEP_TIMEOUT=${STEP_TIMEOUT:-16000}

echo "=== ALPHA RERUN $BB seed_$SEED ($BID) ==="

# Fail fast if this backbone cannot load offline, rather than failing once per method.
python3 - "$BID" <<'PYCHK'
import sys
from med_merge.models.factory import BACKBONE_REGISTRY
from med_merge.config.schema import ModelConfig, DatasetConfig
from med_merge.models.factory import create_model
bid = sys.argv[1]
create_model(DatasetConfig(name="isic_mel", num_classes=1, task_type="binary"),
             model_config=ModelConfig(backbone=bid, hidden_size=BACKBONE_REGISTRY[bid][1], num_layers=12))
print("backbone loads offline OK")
PYCHK
if [ $? -ne 0 ]; then echo "ABORT: $BID cannot load offline, not touching any results"; exit 3; fi
for M in ${METHODS:-iso_c iso_cts tsv_merge task_arithmetic gram_ls dare ties dare_ties lines}; do
  echo "########## $BB s$SEED RETUNE $M ##########"
  # Never delete the existing result. The evaluate step overwrites results.json on success,
  # so a failed or timed-out step leaves the previous value intact rather than destroying it.
  # An earlier version of this script deleted first and lost 100 results when the merge
  # step failed. hyperopt_state.json is kept either way so cached alphas are reused.
  timeout $STEP_TIMEOUT python3 -m med_merge.cli merge --method "$M" --backbone "$BID" $DS --hyperopt \
    --checkpoint-dir "$OUT/checkpoints" --task-vector-dir "$OUT/task_vectors" \
    --output-dir "$OUT/merged_binary" --device cuda
  rc=$?
  if [ $rc -eq 124 ]; then echo "TIMEOUT merge $M after ${STEP_TIMEOUT}s"; continue; fi
  if [ $rc -ne 0 ]; then echo "FAILED merge $M rc=$rc"; continue; fi

  timeout $STEP_TIMEOUT python3 -m med_merge.cli evaluate \
    --model-path "$OUT/merged_binary/$M/merged_encoder.pt" \
    --backbone "$BID" $DS --head-dir "$OUT/checkpoints" \
    --output-dir "$OUT/results_binary/$M" --device cuda
  rc=$?
  [ $rc -eq 124 ] && echo "TIMEOUT eval $M after ${STEP_TIMEOUT}s"
  [ $rc -ne 0 ] && [ $rc -ne 124 ] && echo "FAILED eval $M rc=$rc"
done

echo "ALPHA_RERUN_DONE $BB s$SEED"
