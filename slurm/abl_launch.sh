#!/bin/bash
# Launch the native-task ablation: both arms, same datasets, same 2k cap.
# Usage: ./abl_launch.sh [seed ...]   (default: 42)
set -uo pipefail
PD=.
SEEDS="${*:-42}"

declare -a PAIRS=(
  "clip openai/clip-vit-base-patch16"
  "vit google/vit-base-patch16-224"
  "dinov3 facebook/dinov3-vits16-pretrain-lvd1689m"
  "rad_dino microsoft/rad-dino"
  "dinov2 facebook/dinov2-base"
  "mae facebook/vit-mae-base"
  "beit microsoft/beit-base-patch16-224-pt22k-ft22k"
)

n=0
for SEED in $SEEDS; do
  for ARM in binary native; do
    for p in "${PAIRS[@]}"; do
      set -- $p; BB=$1; BID=$2
      # Skip if this cell already produced all three method results.
      done_ct=$(ls "$PD/outputs/$BB/seed_${SEED}_abl_${ARM}/results"/*/results.json 2>/dev/null | wc -l)
      if [ "$done_ct" -ge 3 ]; then echo "SKIP $ARM $BB s$SEED (complete)"; continue; fi
      jid=$(sbatch --parsable --job-name="abl_${ARM}_${BB}_s${SEED}" \
        --export=ALL,BB="$BB",BID="$BID",SEED="$SEED",ARM="$ARM" \
        "$PD/slurm/abl_cell.sh")
      echo "submitted $jid  arm=$ARM $BB seed=$SEED"
      n=$((n+1))
    done
  done
done
echo "== submitted $n cells =="
