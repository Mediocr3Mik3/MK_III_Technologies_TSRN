#!/usr/bin/env bash
# Azure VM creation + training setup script.
# ==========================================
# Run this from YOUR LOCAL machine after installing Azure CLI (az).
#
# Prerequisites (do these FIRST — see ROADMAP.md):
#   1. az login
#   2. GPU quota approved in your target region (file request in portal first!)
#   3. Set RESOURCE_GROUP, LOCATION, VM_NAME, SSH_KEY below.
#
# Usage:
#   bash research/cloud/launch/azure_vm_create.sh
#
# After creation it prints the SSH command and the tmux launch command.

set -euo pipefail

# ==========================================================================
# USER CONFIGURATION — edit these before running
# ==========================================================================
RESOURCE_GROUP="${RESOURCE_GROUP:-tropformer-rg}"
LOCATION="${LOCATION:-eastus2}"           # regions with good A100 availability:
                                          # eastus2, westus2, centralus, eastus
VM_NAME="${VM_NAME:-tropformer-gpu}"
SSH_KEY="${SSH_KEY:-~/.ssh/id_rsa.pub}"   # your local SSH public key path

# VM size — choose ONE based on your needs (spot vs on-demand, see costs below)
# Standard_NC4as_T4_v3  : 1x T4  16GB,  $0.526/hr on-demand, ~$0.15/hr spot  — cheapest
# Standard_NC6s_v3      : 1x V100 16GB,  $3.06/hr on-demand,  ~$0.70/hr spot  — best value
# Standard_NV36ads_A10_v5: 1x A10 24GB, ~$2.50/hr on-demand,  ~$0.80/hr spot  — good
# Standard_NC24ads_A100_v4: 1x A100 80GB,$14.07/hr on-demand, ~$4.00/hr spot  — Pro training
# Standard_NC48ads_A100_v4: 2x A100 80GB,$28.14/hr on-demand, ~$8.00/hr spot  — Pro multi-GPU
VM_SIZE="${VM_SIZE:-Standard_NC6s_v3}"

# Use spot instances to save ~75%.  Set SPOT=false only for critical final runs.
SPOT="${SPOT:-true}"

# ==========================================================================
# IMAGE: NVIDIA CUDA 12.1 + Ubuntu 20.04
# ==========================================================================
IMAGE="microsoft-dsvm:ubuntu-2004:2004:latest"
# Alternatively use the free NGC image:
# IMAGE="nvidia:pytorch-and-tools:23.12-py3"

echo "=================================================================="
echo "  TropFormer Azure VM Setup"
echo "  Resource Group : ${RESOURCE_GROUP}"
echo "  Location       : ${LOCATION}"
echo "  VM Name        : ${VM_NAME}"
echo "  VM Size        : ${VM_SIZE}"
echo "  Spot           : ${SPOT}"
echo "=================================================================="

# ---------------------------------------------------------------------------
# 1. Create resource group (idempotent)
# ---------------------------------------------------------------------------
echo
echo "==> Creating resource group ${RESOURCE_GROUP} in ${LOCATION}..."
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" \
    --output none

# ---------------------------------------------------------------------------
# 2. Create the VM
# ---------------------------------------------------------------------------
echo
echo "==> Creating VM ${VM_NAME} (${VM_SIZE})..."

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
    --os-disk-size-gb 128 \
    --data-disk-sizes-gb 512 \
    --output json \
    ${SPOT_ARGS} \
    | tee /tmp/azure_vm_info.json

PUBLIC_IP=$(python3 -c "import json,sys; d=json.load(open('/tmp/azure_vm_info.json')); print(d['publicIpAddress'])" 2>/dev/null || \
            az vm show -d -g "${RESOURCE_GROUP}" -n "${VM_NAME}" --query publicIps -o tsv)

echo
echo "=================================================================="
echo "  VM CREATED!"
echo "  Public IP: ${PUBLIC_IP}"
echo "=================================================================="

# ---------------------------------------------------------------------------
# 3. Open port 22 (SSH) — usually open by default; add 6006 for TensorBoard
# ---------------------------------------------------------------------------
az vm open-port --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}" \
    --port 22 --priority 100 --output none
az vm open-port --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}" \
    --port 6006 --priority 200 --output none   # TensorBoard / WandB local

# ---------------------------------------------------------------------------
# 4. Print what to do next
# ---------------------------------------------------------------------------
cat <<EOF

================================================================
  NEXT STEPS
================================================================

1. SSH into the VM:
   ssh azureuser@${PUBLIC_IP}

2. Once inside, run the one-shot setup:
   git clone -b kleene-star https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer
   cd TropFormer
   bash research/cloud/launch/runpod_setup.sh

3. Start a tmux session (so training survives SSH drops):
   tmux new -s train

4. Run CUDA validation first (takes ~5 minutes):
   cd TropFormer
   source .venv/bin/activate
   python research/test_kleene.py

5. Start Nano training (~7 hours on V100 spot):
   bash research/cloud/launch/ssh_launch.sh small_24gb nano_v1 \\
       --steps 100000

6. Monitor:
   tail -f logs/*nano_v1*.log

================================================================
  COST REMINDER
================================================================
   V100 spot: ~\$0.70/hr. 100k-step Nano run = ~7h = ~\$5.
   Stop the VM after training: az vm deallocate -g ${RESOURCE_GROUP} -n ${VM_NAME}
   Restart later:              az vm start -g ${RESOURCE_GROUP} -n ${VM_NAME}
================================================================

EOF
