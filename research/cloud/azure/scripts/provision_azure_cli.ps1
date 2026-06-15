# Azure CLI provisioning script for TropFormer training (PowerShell version)
# Run this from your local PC to provision all Azure resources

# ============================================
# CONFIGURATION
# ============================================
$SubscriptionId = "5bca8e1c-5fed-46a8-918d-91bb5afec472"
$ResourceGroup = "m.freeman-rg"
$Location = "eastus2"

# Storage account (must be globally unique - script will append random if needed)
$StorageAccountBase = "tropformer"
$ContainerName = "tropformer"

# VM configuration
$VMName = "tropformer-vm"
$AdminUsername = "azureuser"
$SSHKeyPath = "$env:USERPROFILE\.ssh\tropformer_azure_key"
$VMSize = "Standard_ND96isr_H100_v5"  # 8x H100 80GB

# Spot instance (set to true for cheaper but evictable instances)
$UseSpotInstance = $false

# Disk configuration
$OSDiskSize = 1024  # GB
$DataDiskSize = 1024  # GB

# Network configuration
$VNetName = "tropformer-vnet"
$SubnetName = "tropformer-subnet"
$NSGName = "tropformer-nsg"
$PublicIPName = "tropformer-pip"

# ============================================
# VALIDATION
# ============================================
Write-Host "=== TropFormer Azure Provisioning ===" -ForegroundColor Cyan
Write-Host "Subscription: $SubscriptionId"
Write-Host "Resource Group: $ResourceGroup"
Write-Host "Location: $Location"
Write-Host "VM Size: $VMSize"
Write-Host ""

# Check if Azure CLI is installed
$azCmd = Get-Command az -ErrorAction SilentlyContinue
if (-not $azCmd) {
    Write-Host "ERROR: Azure CLI not found. Install from https://docs.microsoft.com/cli/azure/install-azure-cli" -ForegroundColor Red
    exit 1
}

# Login to Azure
Write-Host "=== Step 1: Azure Login ===" -ForegroundColor Cyan
az login --subscription $SubscriptionId
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to login to Azure" -ForegroundColor Red
    exit 1
}

# Set subscription
az account set --subscription $SubscriptionId

# ============================================
# RESOURCE GROUP
# ============================================
Write-Host ""
Write-Host "=== Step 2: Resource Group ===" -ForegroundColor Cyan
$rgExists = az group show --name $ResourceGroup --query "id" -o tsv 2>$null
if ($rgExists) {
    Write-Host "[OK] Resource group '$ResourceGroup' already exists" -ForegroundColor Green
} else {
    Write-Host "Creating resource group '$ResourceGroup' in $Location..."
    az group create --name $ResourceGroup --location $Location
    Write-Host "[OK] Resource group created" -ForegroundColor Green
}

# ============================================
# STORAGE ACCOUNT
# ============================================
Write-Host ""
Write-Host "=== Step 3: Storage Account ===" -ForegroundColor Cyan

# Function to check storage account availability
function Check-StorageName {
    param($name)
    $result = az storage account check-name --name $name --query "nameAvailable" -o tsv 2>$null
    return ($result -eq "true")
}

# Try to find an available storage account name
$StorageAccount = $StorageAccountBase
if (Check-StorageName -name $StorageAccount) {
    Write-Host "[OK] Storage account name '$StorageAccount' is available" -ForegroundColor Green
} else {
    Write-Host "Storage account name '$StorageAccount' is not available, appending random suffix..."
    $randomSuffix = -join ((48..57) + (97..102) | Get-Random -Count 8 | ForEach-Object { [char]$_ })
    $StorageAccount = "${StorageAccountBase}${randomSuffix}"
    if (Check-StorageName -name $StorageAccount) {
        Write-Host "[OK] Storage account name '$StorageAccount' is available" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Could not find available storage account name" -ForegroundColor Red
        exit 1
    }
}

# Create storage account
$saExists = az storage account show --name $StorageAccount --resource-group $ResourceGroup --query "id" -o tsv 2>$null
if ($saExists) {
    Write-Host "[OK] Storage account '$StorageAccount' already exists" -ForegroundColor Green
} else {
    Write-Host "Creating storage account '$StorageAccount'..."
    az storage account create `
        --name $StorageAccount `
        --resource-group $ResourceGroup `
        --location $Location `
        --sku Standard_LRS `
        --kind StorageV2 `
        --access-tier Hot
    Write-Host "[OK] Storage account created" -ForegroundColor Green
}

# Get storage account key
$StorageKey = az storage account keys list `
    --resource-group $ResourceGroup `
    --account-name $StorageAccount `
    --query "[0].value" -o tsv

# Create container
Write-Host ""
Write-Host "=== Step 4: Storage Container ===" -ForegroundColor Cyan
$containerExists = az storage container show `
    --name $ContainerName `
    --account-name $StorageAccount `
    --account-key $StorageKey --query "name" -o tsv 2>$null
if ($containerExists) {
    Write-Host "[OK] Container '$ContainerName' already exists" -ForegroundColor Green
} else {
    Write-Host "Creating container '$ContainerName'..."
    az storage container create `
        --name $ContainerName `
        --account-name $StorageAccount `
        --account-key $StorageKey
    Write-Host "[OK] Container created" -ForegroundColor Green
}

# ============================================
# NETWORKING
# ============================================
Write-Host ""
Write-Host "=== Step 5: Networking ===" -ForegroundColor Cyan

# Create VNet
$vnetExists = az network vnet show `
    --resource-group $ResourceGroup `
    --name $VNetName --query "id" -o tsv 2>$null
if ($vnetExists) {
    Write-Host "[OK] VNet '$VNetName' already exists" -ForegroundColor Green
} else {
    Write-Host "Creating VNet '$VNetName'..."
    az network vnet create `
        --resource-group $ResourceGroup `
        --name $VNetName `
        --address-prefix 10.0.0.0/16
    Write-Host "[OK] VNet created" -ForegroundColor Green
}

# Get VNet ID
$VNetId = az network vnet show `
    --resource-group $ResourceGroup `
    --name $VNetName `
    --query id -o tsv

# Create Subnet
$subnetExists = az network vnet subnet show `
    --resource-group $ResourceGroup `
    --vnet-name $VNetName `
    --name $SubnetName --query "id" -o tsv 2>$null
if ($subnetExists) {
    Write-Host "[OK] Subnet '$SubnetName' already exists" -ForegroundColor Green
} else {
    Write-Host "Creating subnet '$SubnetName'..."
    az network vnet subnet create `
        --resource-group $ResourceGroup `
        --vnet-name $VNetName `
        --name $SubnetName `
        --address-prefix 10.0.1.0/24
    Write-Host "[OK] Subnet created" -ForegroundColor Green
}

# Get Subnet ID
$SubnetId = az network vnet subnet show `
    --resource-group $ResourceGroup `
    --vnet-name $VNetName `
    --name $SubnetName `
    --query id -o tsv

# Create NSG
$nsgExists = az network nsg show `
    --resource-group $ResourceGroup `
    --name $NSGName --query "id" -o tsv 2>$null
if ($nsgExists) {
    Write-Host "[OK] NSG '$NSGName' already exists" -ForegroundColor Green
} else {
    Write-Host "Creating NSG '$NSGName'..."
    az network nsg create `
        --resource-group $ResourceGroup `
        --name $NSGName `
        --location $Location

    # Allow SSH
    az network nsg rule create `
        --resource-group $ResourceGroup `
        --nsg-name $NSGName `
        --name AllowSSH `
        --access Allow `
        --protocol Tcp `
        --direction Inbound `
        --priority 100 `
        --source-address-prefix '*' `
        --source-port-range '*' `
        --destination-address-prefix '*' `
        --destination-port-range 22

    Write-Host "[OK] NSG created with SSH rule" -ForegroundColor Green
}

# Get NSG ID
$NSGId = az network nsg show `
    --resource-group $ResourceGroup `
    --name $NSGName `
    --query id -o tsv

# Create Public IP
$pipExists = az network public-ip show `
    --resource-group $ResourceGroup `
    --name $PublicIPName --query "ipAddress" -o tsv 2>$null
if ($pipExists) {
    Write-Host "[OK] Public IP '$PublicIPName' already exists" -ForegroundColor Green
} else {
    Write-Host "Creating public IP '$PublicIPName'..."
    az network public-ip create `
        --resource-group $ResourceGroup `
        --name $PublicIPName `
        --location $Location `
        --sku Standard `
        --allocation-method Static
    Write-Host "[OK] Public IP created" -ForegroundColor Green
}

# Get Public IP
$PublicIP = az network public-ip show `
    --resource-group $ResourceGroup `
    --name $PublicIPName `
    --query ipAddress -o tsv

# ============================================
# SSH KEY
# ============================================
Write-Host ""
Write-Host "=== Step 6: SSH Key ===" -ForegroundColor Cyan
if (Test-Path $SSHKeyPath) {
    Write-Host "[OK] SSH key already exists at $SSHKeyPath" -ForegroundColor Green
} else {
    Write-Host "Generating SSH key at $SSHKeyPath..."
    # Create .ssh directory if it doesn't exist
    $sshDir = Split-Path $SSHKeyPath -Parent
    if (-not (Test-Path $sshDir)) {
        New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
    }
    # Generate SSH key using ssh-keygen (requires Git Bash or WSL)
    # If ssh-keygen is not available, use PowerShell alternative
    $sshKeygen = Get-Command ssh-keygen -ErrorAction SilentlyContinue
    if ($sshKeygen) {
        ssh-keygen -t rsa -b 4096 -f $SSHKeyPath -N ""
    } else {
        Write-Host "WARNING: ssh-keygen not found. You'll need to generate SSH key manually." -ForegroundColor Yellow
        Write-Host "Install Git for Windows or use WSL to get ssh-keygen." -ForegroundColor Yellow
    }
    Write-Host "[OK] SSH key generated" -ForegroundColor Green
}

if (Test-Path "${SSHKeyPath}.pub") {
    $SSHPublicKey = Get-Content "${SSHKeyPath}.pub"
} else {
    Write-Host "WARNING: SSH public key not found at ${SSHKeyPath}.pub" -ForegroundColor Yellow
    $SSHPublicKey = ""
}

# ============================================
# VM CREATION
# ============================================
Write-Host ""
Write-Host "=== Step 7: VM Creation ===" -ForegroundColor Cyan
Write-Host "This will take 10-20 minutes..."

$vmExists = az vm show `
    --resource-group $ResourceGroup `
    --name $VMName --query "id" -o tsv 2>$null
if ($vmExists) {
    Write-Host "[OK] VM '$VMName' already exists" -ForegroundColor Green
    $VMExists = $true
} else {
    # Build VM creation command
    $VMCreateCmd = "az vm create --resource-group $ResourceGroup --name $VMName --location $Location --size $VMSize --admin-username $AdminUsername --ssh-key-values '$SSHPublicKey' --subnet $SubnetId --public-ip-address $PublicIPName --nsg $NSGName --os-disk-size-gb $OSDiskSize --data-disk-sizes-gb $DataDiskSize --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"

    if ($UseSpotInstance) {
        $VMCreateCmd += " --priority Spot --eviction-policy Deallocate --max-price -1"
    }

    Write-Host "Creating VM..."
    Invoke-Expression $VMCreateCmd
    Write-Host "[OK] VM created" -ForegroundColor Green
    $VMExists = $false
}

# ============================================
# OUTPUT CONNECTION DETAILS
# ============================================
Write-Host ""
Write-Host "=== Provisioning Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "CONNECTION DETAILS:" -ForegroundColor Yellow
Write-Host "==================="
Write-Host "Subscription ID:    $SubscriptionId"
Write-Host "Resource Group:     $ResourceGroup"
Write-Host "Location:           $Location"
Write-Host "VM Name:            $VMName"
Write-Host "Public IP:          $PublicIP"
Write-Host "Admin Username:     $AdminUsername"
Write-Host "SSH Key:            $SSHKeyPath"
Write-Host ""
Write-Host "STORAGE DETAILS:" -ForegroundColor Yellow
Write-Host "================"
Write-Host "Storage Account:    $StorageAccount"
Write-Host "Container:          $ContainerName"
Write-Host "Storage Key:        $StorageKey"
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "==========="
Write-Host "1. SSH into the VM:"
Write-Host "   ssh -i $SSHKeyPath ${AdminUsername}@${PublicIP}"
Write-Host ""
Write-Host "2. Clone the repo:"
Write-Host "   git clone -b kleene-star https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer"
Write-Host "   cd TropFormer"
Write-Host ""
Write-Host "3. Mount blob storage:"
Write-Host "   export STORAGE_ACCOUNT=$StorageAccount"
Write-Host "   export STORAGE_CONTAINER=$ContainerName"
Write-Host "   export STORAGE_KEY='$StorageKey'"
Write-Host "   bash research/cloud/azure/scripts/blob_mount.sh"
Write-Host ""
Write-Host "4. Setup Python environment:"
Write-Host "   bash research/cloud/azure/scripts/setup_vm.sh"
Write-Host ""
Write-Host "5. Run smoke test:"
Write-Host "   bash research/cloud/azure/scripts/smoke_test.sh"
Write-Host ""
Write-Host "6. Start pretraining:"
Write-Host "   bash research/cloud/azure/scripts/run_pretrain.sh"
Write-Host ""
Write-Host "SAVE THIS OUTPUT - you'll need the storage key and connection details!" -ForegroundColor Red
