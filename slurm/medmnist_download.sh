#!/bin/bash
#SBATCH --job-name=medmnist_dl
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Idempotent install + download of MedMNIST PathMNIST and RetinaMNIST at 224x224.
# Designed to be chained from launch_phase_b.sh via afterok.

set -euo pipefail

ml gcc/12.4.0
ml python/3.12.1

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/med-merge}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
# Exported so the single-quoted heredoc below picks it up via os.environ
# (otherwise Python sees the literal "$USER").
export MEDMNIST_ROOT="${MEDMNIST_ROOT:-/scratch/users/$USER/medmnist}"

source "$VENV_DIR/bin/activate"
mkdir -p "$MEDMNIST_ROOT" "$PROJECT_DIR/logs"

echo "Python: $(which python3) -- $(python3 --version)"
echo "MedMNIST cache: $MEDMNIST_ROOT"

pip install --quiet --upgrade medmnist

python3 - <<'PY'
import os
from pathlib import Path
from medmnist import PathMNIST, RetinaMNIST

root = os.environ["MEDMNIST_ROOT"]
Path(root).mkdir(parents=True, exist_ok=True)
print(f"Python sees MEDMNIST_ROOT = {root}")

for cls in (PathMNIST, RetinaMNIST):
    for split in ("train", "val", "test"):
        ds = cls(split=split, download=True, size=224, root=root)
        print(f"  {cls.__name__:14s} {split:6s} n={len(ds)}")
print("MedMNIST download complete.")
PY
