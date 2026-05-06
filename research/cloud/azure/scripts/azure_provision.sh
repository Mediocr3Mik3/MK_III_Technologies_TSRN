#!/usr/bin/env bash
# Provision a single Azure VM for Tropformer training (8x H100).
#
# Prereqs:
#   - az login
#   - GPU quota approved in target region (request via Azure portal).
#   - SSH key pair generated (default ~/.ssh/id_rsa.pub).
#
# Usage:
#   bash research/cloud/azure/scripts/azure_provision.sh
#
# Edit the env block below to switch SKU/region/spot.
set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-tropformer-rg}"
LOCATION="${LOCATION:-eastus2}"
VM_NAME="${VM_NAME:-tropformer-h100}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa.pub}"

# 8x H100 80GB SXM5 — Standard_ND96isr_H100_v5
# Cheaper alternatives:
#   8x A100 80GB  -> Standard_ND96amsr_A100_v4
#   1x H100 80GB  -> Standard_NC40ads_H100_v5
VM_SIZE="${VM_SIZE:-Standard_ND96isr_H100_v5}"
SPOT="${SPOT:-true}"

# Premium SSD for fast checkpointing of multi-GB blobs
OS_DISK_GB="${OS_DISK_GB:-256}"
DATA_DISK_GB="${DATA_DISK_GB:-2048}"

# Storage account for blob (used by blobfuse)
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-tropformerblob}"
STORAGE_CONTAINER="${STORAGE_CONTAINER:-tropformer}"

IMAGE="microsoft-dsvm:ubuntu-hpc:2204:latest"

echo "=================================================================="
echo "  Provisioning Tropformer Azure VM"
echo "  RG          : ${RESOURCE_GROUP}"
echo "  Location    : ${LOCATION}"
echo "  VM Name     : ${VM_NAME}"
echo "  VM Size     : ${VM_SIZE}"
echo "  Spot        : ${SPOT}"
echo "  Storage Acct: ${STORAGE_ACCOUNT}/${STORAGE_CONTAINER}"
echo "=================================================================="

# 1. Resource group
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" --output none

# 2. Storage account + container (idempotent)
az storage account create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${STORAGE_ACCOUNT}" \
    --location "${LOCATION}" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --output none || true

ACCOUNT_KEY=$(az storage account keys list \
    --resource-group "${RESOURCE_GROUP}" \
    --account-name "${STORAGE_ACCOUNT}" \
    --query "[0].value" -o tsv)

az storage container create \
    --name "${STORAGE_CONTAINER}" \
    --account-name "${STORAGE_ACCOUNT}" \
    --account-key "${ACCOUNT_KEY}" \
    --output none || true

# 3. VM
SPOT_ARGS=""
if [ "${SPOT}" = "true" ]; then
    SPOT_ARGS="--priority Spot --eviction-policy Deallocate --max-price -1"
fi

az vm create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    --image "${IMAGE}" \
    --size "${VM_SIZE}" \
    --admin-username azureuser \
    --ssh-key-values "${SSH_KEY}" \
    --os-disk-size-gb "${OS_DISK_GB}" \
    --data-disk-sizes-gb "${DATA_DISK_GB}" \
    --accelerated-networking true \
    ${SPOT_ARGS} \
    --output json | tee /tmp/azure_vm_info.json

PUBLIC_IP=$(az vm show -d -g "${RESOURCE_GROUP}" -n "${VM_NAME}" --query publicIps -o tsv)

az vm open-port -g "${RESOURCE_GROUP}" -n "${VM_NAME}" --port 22 --priority 100 --output none
az vm open-port -g "${RESOURCE_GROUP}" -n "${VM_NAME}" --port 8888 --priority 200 --output none

# 4. Print credentials helper for blobfuse
cat <<EOF

============================================================
  VM ready: azureuser@${PUBLIC_IP}
============================================================
Next steps:
  ssh azureuser@${PUBLIC_IP}

On the VM:
  git clone -b kleene-star https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer
  cd TropFormer

  # Mount blob storage at /mnt/blob (uses fuse)
  STORAGE_ACCOUNT=${STORAGE_ACCOUNT} \\
  STORAGE_CONTAINER=${STORAGE_CONTAINER} \\
  STORAGE_KEY=${ACCOUNT_KEY} \\
    bash research/cloud/azure/scripts/blob_mount.sh

  # Stage data + tokenizer (only needed once)
  bash research/cloud/azure/scripts/run_pretrain.sh

============================================================
  Stop VM when idle:  az vm deallocate -g ${RESOURCE_GROUP} -n ${VM_NAME}
============================================================
EOF
