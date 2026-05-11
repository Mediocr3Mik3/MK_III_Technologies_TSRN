#!/usr/bin/env bash
# VM-based DPO launcher. SFT must already be complete.
set -euo pipefail

cd "$(dirname "$0")/../../../.."

BLOB_ROOT="${BLOB_ROOT:-/mnt/blob}"
RAW_DIR="${BLOB_ROOT}/raw/dpo"
TOK_DIR="${BLOB_ROOT}/tokenizers"
CKPT_DIR="${BLOB_ROOT}/checkpoints/dpo"
SFT_BEST="${SFT_BEST:-${BLOB_ROOT}/checkpoints/sft/best.pt}"
TMT_PATH="${TOK_DIR}/tmt_32k.json"

NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"
TAG="${TAG:-azure_run}"

mkdir -p "${RAW_DIR}" "${CKPT_DIR}"

if [ ! -f "${SFT_BEST}" ]; then
    echo "ERROR: SFT checkpoint not found at ${SFT_BEST}"
    exit 1
fi

echo "=================================================="
echo "  Step 0/2: verify DPO dataset URLs (no download)"
echo "=================================================="
python -m research.cloud.azure.data.verify_manifests \
    --manifest dpo_mix

echo "=================================================="
echo "  Step 1/2: download DPO preference pairs"
echo "=================================================="
python -m research.cloud.azure.data.download \
    --manifest research/cloud/azure/data/manifests/dpo_mix.yaml \
    --output   "${RAW_DIR}" \
    --workers  2

echo "=================================================="
echo "  Step 2/2: DPO training on ${NPROC} GPUs"
echo "=================================================="
exec torchrun --standalone --nproc_per_node="${NPROC}" \
    -m research.cloud.azure.train_dpo_cloud \
    --config    dpo_h100x8 \
    --tag       "${TAG}" \
    --init-from "${SFT_BEST}" \
    "$@"
