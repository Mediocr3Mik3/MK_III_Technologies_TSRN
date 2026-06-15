#!/usr/bin/env bash
set -euo pipefail

# Azure VM environment setup for TropFormer training.
# Run this on a fresh Azure VM after cloning the repo.
# This script installs all Python dependencies and verifies GPU availability.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "=== TropFormer Azure VM Setup ==="
echo "This will:"
echo "  1. Verify GPU availability and CUDA"
echo "  2. Install Python dependencies"
echo "  3. Verify HF token (if set)"
echo "  4. Verify WandB token (if set)"
echo ""

# Step 0: Verify we're in the repo
if [ ! -f "$REPO_ROOT/research/tsrn_gist.py" ]; then
    echo "ERROR: Not in TropFormer repo root. Run this from inside the repo."
    exit 1
fi
echo "[OK] Repo root: $REPO_ROOT"

# Step 1: Verify GPU
echo ""
echo "=== Step 1: Verifying GPU availability ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU drivers not installed?"
    exit 1
fi
nvidia-smi
echo "[OK] GPU detected"

# Step 2: Install Python dependencies
echo ""
echo "=== Step 2: Installing Python dependencies ==="
cd "$REPO_ROOT"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install CUDA requirements
echo "Installing CUDA requirements..."
pip install -r research/cloud/requirements_cuda.txt

# Install additional data dependencies
echo "Installing data dependencies..."
pip install datasets huggingface_hub zstandard pyyaml

# Verify PyTorch CUDA availability
echo ""
echo "Verifying PyTorch CUDA..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('ERROR: CUDA not available')
    exit(1)
"

# Step 3: Verify HF token (optional but recommended)
echo ""
echo "=== Step 3: Verifying HuggingFace token ==="
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Gated datasets will fail."
    echo "  Set it with: export HF_TOKEN=your_token_here"
else
    echo "[OK] HF_TOKEN is set"
    python -c "
from huggingface_hub import whoami
try:
    user = whoami()
    print(f'  Authenticated as: {user}')
except Exception as e:
    print(f'  ERROR: Invalid HF_TOKEN: {e}')
    exit(1)
"
fi

# Step 4: Verify WandB token (optional)
echo ""
echo "=== Step 4: Verifying WandB token ==="
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set. Logging will be disabled."
    echo "  Set it with: export WANDB_API_KEY=your_key_here"
else
    echo "[OK] WANDB_API_KEY is set"
    python -c "
import wandb
try:
    wandb.login(key='${WANDB_API_KEY}', verify=True)
    print('  WandB authentication successful')
except Exception as e:
    print(f'  ERROR: Invalid WANDB_API_KEY: {e}')
    exit(1)
"
fi

# Step 5: Verify blob mount (optional)
echo ""
echo "=== Step 5: Verifying blob mount ==="
if [ ! -d /mnt/blob ]; then
    echo "WARNING: /mnt/blob not found. Run blob_mount.sh before training."
    echo "  Or skip if using AzureML (paths will be mounted automatically)."
else
    echo "[OK] Blob mount found at /mnt/blob"
fi

echo ""
echo "=== Setup Complete ==="
echo "You can now:"
echo "  1. Run smoke test: bash research/cloud/azure/scripts/smoke_test.sh"
echo "  2. Start pretraining: bash research/cloud/azure/scripts/run_pretrain.sh"
echo "  3. Or use AzureML: az ml job create -f research/cloud/azure/jobs/aml_pretrain.yaml"
