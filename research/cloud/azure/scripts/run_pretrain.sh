#!/usr/bin/env bash
# VM-based pretrain launcher. Assumes:
#   - blob mounted at /mnt/blob (run blob_mount.sh first)
#   - python env active with requirements installed
#
# Steps:
#   1. Download raw shards (resumable; skips if already done)
#   2. Train TMT tokenizer if not present
#   3. Tokenize raw shards to packed .bin
#   4. Launch torchrun pretrain on all local GPUs
set -euo pipefail

cd "$(dirname "$0")/../../../.."   # repo root

BLOB_ROOT="${BLOB_ROOT:-/mnt/blob}"
RAW_DIR="${BLOB_ROOT}/raw/pretrain"
TOK_DIR="${BLOB_ROOT}/tokenizers"
TOKENS_DIR="${BLOB_ROOT}/tokens/pretrain"
CKPT_DIR="${BLOB_ROOT}/checkpoints/pretrain"
TMT_PATH="${TOK_DIR}/tmt_32k.json"

NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"
TAG="${TAG:-azure_run}"

mkdir -p "${RAW_DIR}" "${TOK_DIR}" "${TOKENS_DIR}" "${CKPT_DIR}"

echo "=================================================="
echo "  Step 1/4: download raw pretrain shards"
echo "=================================================="
python -m research.cloud.azure.data.download \
    --manifest research/cloud/azure/data/manifests/pretrain_mix.yaml \
    --output   "${RAW_DIR}" \
    --workers  4

if [ ! -f "${TMT_PATH}" ]; then
    echo "=================================================="
    echo "  Step 2/4: train TMT tokenizer"
    echo "=================================================="
    python -m research.cloud.azure.data.train_tmt \
        --raw-dir   "${RAW_DIR}" \
        --output    "${TMT_PATH}" \
        --vocab-size 32000 \
        --sample-bytes 5e9 \
        --rounds 14
else
    echo "[skip] TMT tokenizer already at ${TMT_PATH}"
fi

if [ ! -f "${TOKENS_DIR}/_summary.json" ]; then
    echo "=================================================="
    echo "  Step 3/4: tokenize + shard"
    echo "=================================================="
    python -m research.cloud.azure.data.tokenize_shard \
        --raw-dir   "${RAW_DIR}" \
        --tokenizer "${TMT_PATH}" \
        --output    "${TOKENS_DIR}" \
        --workers   16 \
        --shard-tokens 200000000
else
    echo "[skip] tokens already prepared at ${TOKENS_DIR}"
fi

echo "=================================================="
echo "  Step 4/4: launch pretrain on ${NPROC} GPUs"
echo "=================================================="
exec torchrun --standalone --nproc_per_node="${NPROC}" \
    -m research.cloud.azure.train_pretrain_cloud \
    --config pretrain_h100x8 \
    --tag    "${TAG}" \
    "$@"
