#!/usr/bin/env bash
# VM-based SFT launcher. Pretrain must already be complete.
set -euo pipefail

cd "$(dirname "$0")/../../../.."

BLOB_ROOT="${BLOB_ROOT:-/mnt/blob}"
RAW_DIR="${BLOB_ROOT}/raw/sft"
TOK_DIR="${BLOB_ROOT}/tokenizers"
CKPT_DIR="${BLOB_ROOT}/checkpoints/sft"
PRETRAIN_BEST="${PRETRAIN_BEST:-${BLOB_ROOT}/checkpoints/pretrain/best.pt}"
TMT_PATH="${TOK_DIR}/tmt_32k.json"

NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"
TAG="${TAG:-azure_run}"

mkdir -p "${RAW_DIR}" "${CKPT_DIR}"

if [ ! -f "${TMT_PATH}" ]; then
    echo "ERROR: TMT tokenizer not found at ${TMT_PATH}"
    echo "Run pretrain pipeline first (it trains TMT)."
    exit 1
fi

echo "=================================================="
echo "  Step 0/2: verify SFT dataset URLs (no download)"
echo "=================================================="
python -m research.cloud.azure.data.verify_manifests \
    --manifest sft_mix

echo "=================================================="
echo "  Step 1/2: download SFT raw shards"
echo "=================================================="
python -m research.cloud.azure.data.download \
    --manifest research/cloud/azure/data/manifests/sft_mix.yaml \
    --output   "${RAW_DIR}" \
    --workers  2

echo "=================================================="
echo "  Step 2/2: SFT training (curriculum) on ${NPROC} GPUs"
echo "=================================================="
exec torchrun --standalone --nproc_per_node="${NPROC}" \
    -m research.cloud.azure.train_sft_cloud \
    --config    sft_h100x8 \
    --tag       "${TAG}" \
    --init-from "${PRETRAIN_BEST}" \
    "$@"
