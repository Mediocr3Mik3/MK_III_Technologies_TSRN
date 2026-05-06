#!/usr/bin/env bash
# Full pipeline: pretrain -> SFT -> DPO. Each stage resumes safely.
set -euo pipefail

cd "$(dirname "$0")/../../../.."

bash research/cloud/azure/scripts/run_pretrain.sh "$@"
bash research/cloud/azure/scripts/run_sft.sh
bash research/cloud/azure/scripts/run_dpo.sh
