#!/usr/bin/env bash
# RunPod / Lambda Labs / vast.ai setup script.
# =============================================
# Run this once on a fresh GPU pod (Ubuntu 22.04 + CUDA 12.x base image).
#
#   wget -O - https://raw.githubusercontent.com/Mediocr3Mik3/MK_III_Technologies_TSRN/nvidia-cloud/research/cloud/launch/runpod_setup.sh | bash
#
# Or (recommended — clone first so you can edit configs)::
#
#   git clone -b nvidia-cloud https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer
#   cd TropFormer
#   bash research/cloud/launch/runpod_setup.sh
#
# Then::
#
#   bash research/cloud/launch/ssh_launch.sh medium_40gb my_run_tag

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"

echo "==> TropFormer cloud setup"
echo "    Repo dir : ${REPO_DIR}"
echo "    Python   : ${PYTHON_BIN}"
echo "    Venv     : ${VENV_DIR}"

# ---------------------------------------------------------------------------
# 1. Sanity check — do we see a GPU?
# ---------------------------------------------------------------------------
echo
echo "==> Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "!! nvidia-smi not found.  Are you on a CPU-only pod?"
    echo "   Continuing anyway (CPU smoke-test mode)."
else
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
fi

# ---------------------------------------------------------------------------
# 2. System packages (RunPod images usually have these; Lambda may not)
# ---------------------------------------------------------------------------
echo
echo "==> Installing system packages..."
if command -v apt-get &>/dev/null; then
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        git wget curl build-essential \
        python3-venv python3-pip
fi

# ---------------------------------------------------------------------------
# 3. Virtualenv
# ---------------------------------------------------------------------------
echo
echo "==> Creating virtualenv at ${VENV_DIR}..."
${PYTHON_BIN} -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 4. PyTorch with the right CUDA version
# ---------------------------------------------------------------------------
CUDA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | awk -F. '{print $1}' || true)"
echo
echo "==> Driver major: ${CUDA_VERSION:-unknown}"

# Default to cu121 (works with driver >= 525, includes most cloud images).
# Use cu124 only on H100/H200 with driver >= 550.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
echo "==> Installing torch from ${TORCH_INDEX}"
pip install --index-url "${TORCH_INDEX}" \
    "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1" \
    || pip install torch torchvision torchaudio  # fallback to default index

# ---------------------------------------------------------------------------
# 5. TropFormer base + cloud requirements
# ---------------------------------------------------------------------------
echo
echo "==> Installing TropFormer requirements..."
pip install -r "${REPO_DIR}/requirements.txt"
pip install -r "${REPO_DIR}/research/cloud/requirements_cuda.txt"

# ---------------------------------------------------------------------------
# 6. flash-attn (optional, big speedup on Ampere+)
# ---------------------------------------------------------------------------
echo
echo "==> Attempting flash-attn install..."
pip install --no-build-isolation "flash-attn>=2.6" \
    || echo "!! flash-attn build failed (need nvcc + CUDA dev tools); continuing without it"

# ---------------------------------------------------------------------------
# 7. enwik8 download (~36 MB)
# ---------------------------------------------------------------------------
echo
echo "==> Pre-downloading enwik8..."
mkdir -p "${REPO_DIR}/data" "${REPO_DIR}/checkpoints" "${REPO_DIR}/results" "${REPO_DIR}/logs"
if [ ! -f "${REPO_DIR}/data/enwik8" ] && [ ! -f "${REPO_DIR}/enwik8" ]; then
    wget -nc -O "${REPO_DIR}/data/enwik8.zip" \
        http://mattmahoney.net/dc/enwik8.zip
    (cd "${REPO_DIR}/data" && unzip -o enwik8.zip)
fi

# ---------------------------------------------------------------------------
# 8. Smoke test — make sure the trainer imports
# ---------------------------------------------------------------------------
echo
echo "==> Smoke-test trainer import..."
cd "${REPO_DIR}"
python -m research.cloud.train_cloud --help | head -n 20

echo
echo "============================================================"
echo "  Setup complete."
echo "  To start training:"
echo "    source ${VENV_DIR}/bin/activate"
echo "    bash research/cloud/launch/ssh_launch.sh medium_40gb my_run"
echo "============================================================"
