#!/usr/bin/env bash
set -euo pipefail

# Azure CLI provisioning script for TropFormer training
# Run this from your local PC to provision all Azure resources

# ============================================
# CONFIGURATION
# ============================================
SUBSCRIPTION_ID="5bca8e1c-5fed-46a8-918d-91bb5afec472"
RESOURCE_GROUP="m.freeman-rg"
LOCATION="eastus2"

# Storage account (must be globally unique - script will append random if needed)
STORAGE_ACCOUNT_BASE="tropformer"
CONTAINER_NAME="tropformer"

# VM configuration
VM_NAME="tropformer-vm"
ADMIN_USERNAME="azureuser"
SSH_KEY_PATH="$HOME/.ssh/tropformer_azure_key"
VM_SIZE="Standard_ND96isr_H100_v5"  # 8x H100 80GB

# Spot instance (set to true for cheaper but evictable instances)
USE_SPOT_INSTANCE="false"

# Disk configuration
OS_DISK_SIZE=1024  # GB
DATA_DISK_SIZE=1024  # GB

# Network configuration
VNET_NAME="tropformer-vnet"
SUBNET_NAME="tropformer-subnet"
NSG_NAME="tropformer-nsg"
PUBLIC_IP_NAME="tropformer-pip"

# ============================================
# VALIDATION
# ============================================
echo "=== TropFormer Azure Provisioning ==="
echo "Subscription: $SUBSCRIPTION_ID"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "VM Size: $VM_SIZE"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "ERROR: Azure CLI not found. Install from https://docs.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi

# Login to Azure
echo "=== Step 1: Azure Login ==="
az login --subscription "$SUBSCRIPTION_ID" || {
    echo "ERROR: Failed to login to Azure"
    exit 1
}

# Set subscription
az account set --subscription "$SUBSCRIPTION_ID"

# ============================================
# RESOURCE GROUP
# ============================================
echo ""
echo "=== Step 2: Resource Group ==="
if az group show --name "$RESOURCE_GROUP" &> /dev/null; then
    echo "[OK] Resource group '$RESOURCE_GROUP' already exists"
else
    echo "Creating resource group '$RESOURCE_GROUP' in $LOCATION..."
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
    echo "[OK] Resource group created"
fi

# ============================================
# STORAGE ACCOUNT
# ============================================
echo ""
echo "=== Step 3: Storage Account ==="

# Function to check storage account availability
check_storage_name() {
    local name=$1
    az storage account check-name --name "$name" --query "nameAvailable" -o tsv 2>/dev/null || echo "false"
}

# Try to find an available storage account name
STORAGE_ACCOUNT="${STORAGE_ACCOUNT_BASE}"
if check_storage_name "$STORAGE_ACCOUNT" | grep -q "true"; then
    echo "[OK] Storage account name '$STORAGE_ACCOUNT' is available"
else
    echo "Storage account name '$STORAGE_ACCOUNT' is not available, appending random suffix..."
    RANDOM_SUFFIX=$(openssl rand -hex 4 | head -c 8)
    STORAGE_ACCOUNT="${STORAGE_ACCOUNT_BASE}${RANDOM_SUFFIX}"
    if check_storage_name "$STORAGE_ACCOUNT" | grep -q "true"; then
        echo "[OK] Storage account name '$STORAGE_ACCOUNT' is available"
    else
        echo "ERROR: Could not find available storage account name"
        exit 1
    fi
fi

# Create storage account
if az storage account show --name "$STORAGE_ACCOUNT" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    echo "[OK] Storage account '$STORAGE_ACCOUNT' already exists"
else
    echo "Creating storage account '$STORAGE_ACCOUNT'..."
    az storage account create \
        --name "$STORAGE_ACCOUNT" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --sku Standard_LRS \
        --kind StorageV2 \
        --access-tier Hot
    echo "[OK] Storage account created"
fi

# Get storage account key
STORAGE_KEY=$(az storage account keys list \
    --resource-group "$RESOURCE_GROUP" \
    --account-name "$STORAGE_ACCOUNT" \
    --query "[0].value" -o tsv)

# Create container
echo ""
echo "=== Step 4: Storage Container ==="
if az storage container show \
    --name "$CONTAINER_NAME" \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" &> /dev/null; then
    echo "[OK] Container '$CONTAINER_NAME' already exists"
else
    echo "Creating container '$CONTAINER_NAME'..."
    az storage container create \
        --name "$CONTAINER_NAME" \
        --account-name "$STORAGE_ACCOUNT" \
        --account-key "$STORAGE_KEY"
    echo "[OK] Container created"
fi

# ============================================
# NETWORKING
# ============================================
echo ""
echo "=== Step 5: Networking ==="

# Create VNet
if az network vnet show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VNET_NAME" &> /dev/null; then
    echo "[OK] VNet '$VNET_NAME' already exists"
else
    echo "Creating VNet '$VNET_NAME'..."
    az network vnet create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VNET_NAME" \
        --address-prefix 10.0.0.0/16
    echo "[OK] VNet created"
fi

# Get VNet ID
VNET_ID=$(az network vnet show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VNET_NAME" \
    --query id -o tsv)

# Create Subnet
if az network vnet subnet show \
    --resource-group "$RESOURCE_GROUP" \
    --vnet-name "$VNET_NAME" \
    --name "$SUBNET_NAME" &> /dev/null; then
    echo "[OK] Subnet '$SUBNET_NAME' already exists"
else
    echo "Creating subnet '$SUBNET_NAME'..."
    az network vnet subnet create \
        --resource-group "$RESOURCE_GROUP" \
        --vnet-name "$VNET_NAME" \
        --name "$SUBNET_NAME" \
        --address-prefix 10.0.1.0/24
    echo "[OK] Subnet created"
fi

# Get Subnet ID
SUBNET_ID=$(az network vnet subnet show \
    --resource-group "$RESOURCE_GROUP" \
    --vnet-name "$VNET_NAME" \
    --name "$SUBNET_NAME" \
    --query id -o tsv)

# Create NSG
if az network nsg show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$NSG_NAME" &> /dev/null; then
    echo "[OK] NSG '$NSG_NAME' already exists"
else
    echo "Creating NSG '$NSG_NAME'..."
    az network nsg create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$NSG_NAME" \
        --location "$LOCATION"

    # Allow SSH
    az network nsg rule create \
        --resource-group "$RESOURCE_GROUP" \
        --nsg-name "$NSG_NAME" \
        --name AllowSSH \
        --access Allow \
        --protocol Tcp \
        --direction Inbound \
        --priority 100 \
        --source-address-prefix '*' \
        --source-port-range '*' \
        --destination-address-prefix '*' \
        --destination-port-range 22

    echo "[OK] NSG created with SSH rule"
fi

# Get NSG ID
NSG_ID=$(az network nsg show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$NSG_NAME" \
    --query id -o tsv)

# Create Public IP
if az network public-ip show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$PUBLIC_IP_NAME" &> /dev/null; then
    echo "[OK] Public IP '$PUBLIC_IP_NAME' already exists"
else
    echo "Creating public IP '$PUBLIC_IP_NAME'..."
    az network public-ip create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$PUBLIC_IP_NAME" \
        --location "$LOCATION" \
        --sku Standard \
        --allocation-method Static
    echo "[OK] Public IP created"
fi

# Get Public IP
PUBLIC_IP=$(az network public-ip show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$PUBLIC_IP_NAME" \
    --query ipAddress -o tsv)

# ============================================
# SSH KEY
# ============================================
echo ""
echo "=== Step 6: SSH Key ==="
if [ -f "$SSH_KEY_PATH" ]; then
    echo "[OK] SSH key already exists at $SSH_KEY_PATH"
else
    echo "Generating SSH key at $SSH_KEY_PATH..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N ""
    echo "[OK] SSH key generated"
fi

SSH_PUBLIC_KEY=$(cat "${SSH_KEY_PATH}.pub")

# ============================================
# VM CREATION
# ============================================
echo ""
echo "=== Step 7: VM Creation ==="
echo "This will take 10-20 minutes..."

if az vm show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" &> /dev/null; then
    echo "[OK] VM '$VM_NAME' already exists"
    VM_EXISTS=true
else
    # Build VM creation command
    VM_CREATE_CMD="az vm create \
        --resource-group $RESOURCE_GROUP \
        --name $VM_NAME \
        --location $LOCATION \
        --size $VM_SIZE \
        --admin-username $ADMIN_USERNAME \
        --ssh-key-values '$SSH_PUBLIC_KEY' \
        --subnet $SUBNET_ID \
        --public-ip-address $PUBLIC_IP_NAME \
        --nsg $NSG_NAME \
        --os-disk-size-gb $OS_DISK_SIZE \
        --data-disk-sizes-gb $DATA_DISK_SIZE \
        --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"

    if [ "$USE_SPOT_INSTANCE" = "true" ]; then
        VM_CREATE_CMD="$VM_CREATE_CMD \
            --priority Spot \
            --eviction-policy Deallocate \
            --max-price -1"
    fi

    echo "Creating VM with command:"
    echo "$VM_CREATE_CMD"
    echo ""

    eval $VM_CREATE_CMD
    echo "[OK] VM created"
    VM_EXISTS=false
fi

# ============================================
# OUTPUT CONNECTION DETAILS
# ============================================
echo ""
echo "=== Provisioning Complete ==="
echo ""
echo "CONNECTION DETAILS:"
echo "==================="
echo "Subscription ID:    $SUBSCRIPTION_ID"
echo "Resource Group:     $RESOURCE_GROUP"
echo "Location:           $LOCATION"
echo "VM Name:            $VM_NAME"
echo "Public IP:          $PUBLIC_IP"
echo "Admin Username:     $ADMIN_USERNAME"
echo "SSH Key:            $SSH_KEY_PATH"
echo ""
echo "STORAGE DETAILS:"
echo "================"
echo "Storage Account:    $STORAGE_ACCOUNT"
echo "Container:          $CONTAINER_NAME"
echo "Storage Key:        $STORAGE_KEY"
echo ""
echo "NEXT STEPS:"
echo "==========="
echo "1. SSH into the VM:"
echo "   ssh -i $SSH_KEY_PATH ${ADMIN_USERNAME}@${PUBLIC_IP}"
echo ""
echo "2. Clone the repo:"
echo "   git clone -b kleene-star https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer"
echo "   cd TropFormer"
echo ""
echo "3. Mount blob storage:"
echo "   export STORAGE_ACCOUNT=$STORAGE_ACCOUNT"
echo "   export STORAGE_CONTAINER=$CONTAINER_NAME"
echo "   export STORAGE_KEY='$STORAGE_KEY'"
echo "   bash research/cloud/azure/scripts/blob_mount.sh"
echo ""
echo "4. Setup Python environment:"
echo "   bash research/cloud/azure/scripts/setup_vm.sh"
echo ""
echo "5. Run smoke test:"
echo "   bash research/cloud/azure/scripts/smoke_test.sh"
echo ""
echo "6. Start pretraining:"
echo "   bash research/cloud/azure/scripts/run_pretrain.sh"
echo ""
echo "SAVE THIS OUTPUT - you'll need the storage key and connection details!"
