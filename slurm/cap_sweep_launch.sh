#!/bin/bash
# Launch the training-budget ablation: matched caps below the 2,000-image
# benchmark on three backbones. ISIC 2017 has 2,200 training images, so 2,000
# is the largest matched cap and the curve can only extend downward.
# Usage: ./cap_sweep_launch.sh [cap ...]   (default: 500 1000)
set -uo pipefail
PD=/home/groups/roxanad/sonnet/med-merge
CAPS="${*:-500 1000}"
SEEDS="${SEEDS:-42 123 456}"

declare -a PAIRS=(
  "clip openai/clip-vit-base-patch16"
  "vit google/vit-base-patch16-224"
  "dinov2 facebook/dinov2-base"
)
NMETHODS=6

n=0
for CAP in $CAPS; do
  for SEED in $SEEDS; do
    for p in "${PAIRS[@]}"; do
      set -- $p; BB=$1; BID=$2
      NAME="cap${CAP}_${BB}_s${SEED}"
      done_ct=$(ls "$PD/outputs/$BB/seed_${SEED}_cap${CAP}/results_binary"/*/results.json 2>/dev/null | wc -l)
      if [ "$done_ct" -ge "$NMETHODS" ]; then echo "SKIP $NAME (complete)"; continue; fi
      # A cell that is already pending or running must not be submitted twice,
      # since both copies would train into the same output directory.
      if squeue -u "$USER" -h -n "$NAME" -o "%i" | grep -q .; then echo "SKIP $NAME (already queued)"; continue; fi
      jid=$(sbatch --parsable --job-name="$NAME" \
        --export=ALL,BB="$BB",BID="$BID",SEED="$SEED",CAP="$CAP" \
        "$PD/slurm/cap_sweep_cell.sh")
      echo "submitted $jid  cap=$CAP $BB seed=$SEED"
      n=$((n+1))
    done
  done
done
echo "== submitted $n cells =="
