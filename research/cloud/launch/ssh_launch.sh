#!/usr/bin/env bash
# Provider-agnostic launcher.
# ===========================
# Usage::
#
#   bash research/cloud/launch/ssh_launch.sh <preset> <tag> [extra args...]
#
# Examples::
#
#   # Single GPU, A100-40
#   bash research/cloud/launch/ssh_launch.sh medium_40gb a100_run0
#
#   # Single GPU, RTX 4090
#   bash research/cloud/launch/ssh_launch.sh small_24gb rtx_run0
#
#   # 8x A100 DDP (auto-uses torchrun if NUM_GPUS>1)
#   NUM_GPUS=8 bash research/cloud/launch/ssh_launch.sh multi_8xa100 multi8 \
#       --steps 100000
#
#   # Resume from a checkpoint
#   bash research/cloud/launch/ssh_launch.sh medium_40gb resumed \
#       --resume checkpoints/20260426_train_cloud_a100_run0_step050000.pt
#
# Output:
#   - logs/<DATE>_train_cloud_<tag>.log
#   - checkpoints/<DATE>_train_cloud_<tag>_step*.pt (every 5K steps by default)
#   - results/<DATE>_train_cloud_<tag>_progress_step*.json

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <preset> <tag> [extra args...]"
    echo "Presets: small_24gb, medium_40gb, large_80gb, multi_8xa100"
    exit 2
fi

PRESET="$1"
TAG="$2"
shift 2
EXTRA_ARGS=("$@")

REPO_DIR="${REPO_DIR:-$(pwd)}"
LOG_DIR="${REPO_DIR}/logs"
DATE="$(date +%Y%m%d)"
LOG_FILE="${LOG_DIR}/${DATE}_train_cloud_${TAG}.log"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)}"

mkdir -p "${LOG_DIR}" "${REPO_DIR}/checkpoints" "${REPO_DIR}/results"

echo "==> Preset    : ${PRESET}"
echo "==> Tag       : ${TAG}"
echo "==> GPUs      : ${NUM_GPUS}"
echo "==> Log       : ${LOG_FILE}"
echo "==> Extra args: ${EXTRA_ARGS[*]:-(none)}"

cd "${REPO_DIR}"

if [ "${NUM_GPUS}" -gt 1 ]; then
    # Distributed (DDP) launch
    echo "==> Launching with torchrun (DDP, ${NUM_GPUS} GPUs)"
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        -m research.cloud.train_cloud \
        --preset "${PRESET}" --tag "${TAG}" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
else
    # Single GPU
    echo "==> Launching single-GPU"
    python -m research.cloud.train_cloud \
        --preset "${PRESET}" --tag "${TAG}" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
fi

echo
echo "==> Done.  Tail logs: tail -n 100 ${LOG_FILE}"
