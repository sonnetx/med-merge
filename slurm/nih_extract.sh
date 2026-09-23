#!/bin/bash
#SBATCH --job-name=nih_extract
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=nih_extract_%j.out
#SBATCH --error=nih_extract_%j.err

set -uo pipefail   # NOTE: no -e, we want to continue past benign tar warnings
cd $GROUP_SCRATCH/datasets/nih-cxr14

mkdir -p images

for f in images_*.tar.gz; do
  echo "=== Extracting $f ==="
  # Pipe through gunzip and discard its stderr so trailing-garbage warnings
  # don't propagate as a nonzero exit and abort tar.
  gunzip -c "$f" 2>/dev/null | tar -x -C images --strip-components=1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  WARN: $f exited $rc, continuing"
  fi
done

n_images=$(find images -type f -name "*.png" | wc -l)
echo "Extracted $n_images images (expected 112120)"

if [ "$n_images" -ge 112000 ]; then
  echo "Cleaning up archives..."
  rm -f images_*.tar.gz
else
  echo "WARNING: image count low, keeping archives for inspection"
  exit 1
fi

echo "Finished: $(date)"

