#!/usr/bin/env bash
set -euo pipefail

echo "==> Creating conda environment: deepmedalign"
conda env create -f environment.yml

echo "==> Activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepmedalign

echo "==> Verifying imports"
python -c "import torch, SimpleITK, nibabel, voxelmorph, antspyx, wandb; print('  torch:', torch.__version__); print('  SimpleITK:', SimpleITK.Version()); print('  nibabel:', nibabel.__version__); print('  voxelmorph: ok'); print('  wandb:', wandb.__version__)"

echo "==> Running tests"
pytest tests/ -v --tb=short

echo ""
echo "✓ Environment ready. Activate with: conda activate deepmedalign"
