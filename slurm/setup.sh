#!/bin/bash
#SBATCH --job-name=medmerge_setup
#SBATCH --partition=normal
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# med-merge Environment Setup
#
# Creates venv and installs all dependencies with pinned versions.
# Run this once before smoke_test.sh or full_benchmark.sh.
#
# Usage:
#   sbatch slurm/setup.sh

ml gcc/12.4.0
ml python/3.12.1
ml cuda/12.4.0

PROJECT_DIR="${PROJECT_DIR:-/home/groups/roxanad/sonnet/med-merge}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"

# Remove existing environment if it exists (in case of partial/failed install)
rm -rf "$VENV_DIR"

# Create virtual environment
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

export TMPDIR="/scratch/users/$USER/tmp"
export HF_HOME="/scratch/users/$USER/huggingface"
export HF_DATASETS_CACHE="/scratch/users/$USER/huggingface/datasets"
export TORCH_HOME="/scratch/users/$USER/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME"

which python3
python3 --version

# Upgrade pip/setuptools
pip3 install --no-cache-dir --upgrade pip setuptools wheel

# ---- Heavy binary packages (install first, all with --only-binary) ----

# numpy first (needed by pandas/scipy build metadata)
pip3 install --no-cache-dir --only-binary :all: numpy==1.26.4

# PyTorch with CUDA 12.4 support
pip3 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Packages that must NOT build from source on Sherlock
pip3 install --no-cache-dir --only-binary :all: pyarrow==15.0.2
pip3 install --no-cache-dir --only-binary :all: pandas==2.2.3
pip3 install --no-cache-dir --only-binary :all: scipy==1.12.0
pip3 install --no-cache-dir --only-binary :all: scikit-learn==1.4.2

# ---- HuggingFace stack (pyarrow already satisfied above) ----
# Use --only-binary for pyarrow to prevent datasets from pulling a different version
pip3 install --no-cache-dir "transformers>=4.36.0"
pip3 install --no-cache-dir --only-binary pyarrow "datasets>=2.16.0"

# ---- Pure Python dependencies ----
pip3 install --no-cache-dir Pillow click omegaconf pydantic tqdm requests
# contourpy >= 1.3.2 dropped glibc 2.17 wheels (RHEL 7)
pip3 install --no-cache-dir --only-binary :all: "contourpy>=1.3,<1.3.2"
pip3 install --no-cache-dir matplotlib seaborn
pip3 install --no-cache-dir kaggle

# wandb — pin version with prebuilt wheel, --no-deps to avoid pulling different torch
pip3 install --no-cache-dir --only-binary :all: wandb==0.19.1 --no-deps
pip3 install --no-cache-dir psutil docker-pycreds sentry-sdk setproctitle gitpython "protobuf>=3.19,<6" platformdirs

# Dev dependencies
pip3 install --no-cache-dir pytest pytest-cov ruff

# ---- Install med-merge itself last (editable, deps already satisfied) ----
pip3 install --no-cache-dir --no-deps -e "$PROJECT_DIR"

# Verify
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers {transformers.__version__}')"
python3 -c "import datasets; print(f'Datasets {datasets.__version__}')"
python3 -c "import pandas; print(f'Pandas {pandas.__version__}')"
python3 -c "import pyarrow; print(f'PyArrow {pyarrow.__version__}')"
python3 -c "import med_merge; print('med_merge importable')"
python3 -c "from med_merge.cli import cli; print('CLI importable')"

echo ""
echo "============================================================"
echo "med-merge environment setup complete!"
echo "Venv: $VENV_DIR"
echo ""
echo "Next: sbatch slurm/smoke_test.sh"
echo "============================================================"
