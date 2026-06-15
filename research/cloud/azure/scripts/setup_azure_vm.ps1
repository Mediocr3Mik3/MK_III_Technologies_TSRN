# PowerShell automation script for Azure VM setup
# This script automates SSH connection, repo clone, blob mount, Python setup, and smoke test

# Configuration
$VM_IP = "20.96.36.106"
$AdminUser = "azureuser"
$SSHKeyPath = "C:\Users\freem\.ssh\tropformer_azure_key"

# Storage credentials (read from environment — NEVER hardcode secrets in tracked files)
$STORAGE_ACCOUNT = if ($env:STORAGE_ACCOUNT) { $env:STORAGE_ACCOUNT } else { "tropformer" }
$STORAGE_CONTAINER = if ($env:STORAGE_CONTAINER) { $env:STORAGE_CONTAINER } else { "tropformer" }
$STORAGE_KEY = $env:TROPFORMER_STORAGE_KEY

# API tokens (read from environment)
$HF_TOKEN = $env:HF_TOKEN
$WANDB_API_KEY = $env:WANDB_API_KEY

# Fail fast if any required secret is missing from the environment
$missing = @()
if (-not $STORAGE_KEY)   { $missing += "TROPFORMER_STORAGE_KEY" }
if (-not $HF_TOKEN)      { $missing += "HF_TOKEN" }
if (-not $WANDB_API_KEY) { $missing += "WANDB_API_KEY" }
if ($missing.Count -gt 0) {
    Write-Host "ERROR: missing required environment variable(s): $($missing -join ', ')" -ForegroundColor Red
    Write-Host "Set them in your shell before running this script, e.g.:" -ForegroundColor Yellow
    Write-Host '  $env:TROPFORMER_STORAGE_KEY = "<azure-storage-key>"' -ForegroundColor Yellow
    Write-Host '  $env:HF_TOKEN = "<hugging-face-token>"' -ForegroundColor Yellow
    Write-Host '  $env:WANDB_API_KEY = "<wandb-api-key>"' -ForegroundColor Yellow
    exit 1
}

Write-Host "=== TropFormer Azure VM Setup Automation ===" -ForegroundColor Cyan
Write-Host "VM IP: $VM_IP"
Write-Host "Admin: $AdminUser"
Write-Host ""

# Check for SSH client
$sshCmd = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $sshCmd) {
    Write-Host "ERROR: ssh command not found. Install OpenSSH for Windows or use PuTTY." -ForegroundColor Red
    exit 1
}

# Function to run SSH command
function Invoke-SSHCommand {
    param(
        [string]$Command
    )
    $sshCommand = "ssh -i `"$SSHKeyPath`" -o StrictHostKeyChecking=no ${AdminUser}@${VM_IP} `"$Command`""
    Write-Host "Executing: $Command" -ForegroundColor Gray
    Invoke-Expression $sshCommand
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Command failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
}

# Function to run SSH command and return output
function Get-SSHOutput {
    param(
        [string]$Command
    )
    $sshCommand = "ssh -i `"$SSHKeyPath`" -o StrictHostKeyChecking=no ${AdminUser}@${VM_IP} `"$Command`""
    $output = Invoke-Expression $sshCommand
    return $output
}

# Step 1: Test SSH connection
Write-Host "=== Step 1: Testing SSH Connection ===" -ForegroundColor Cyan
try {
    Invoke-SSHCommand "echo 'SSH connection successful'"
    Write-Host "[OK] SSH connection verified" -ForegroundColor Green
} catch {
    Write-Host "ERROR: SSH connection failed. Check SSH key and permissions." -ForegroundColor Red
    Write-Host "TIP: If SSH key doesn't exist, generate it manually:" -ForegroundColor Yellow
    Write-Host "  ssh-keygen -t rsa -b 4096 -f `"$SSHKeyPath`"" -ForegroundColor Yellow
    exit 1
}

# Step 2: Clone repo
Write-Host ""
Write-Host "=== Step 2: Cloning TropFormer Repository ===" -ForegroundColor Cyan
$repoExists = Get-SSHOutput "if [ -d ~/TropFormer ]; then echo 'exists'; else echo 'not_exists'; fi"
if ($repoExists -eq "exists") {
    Write-Host "[OK] Repository already exists, pulling latest..." -ForegroundColor Green
    Invoke-SSHCommand "cd ~/TropFormer && git pull"
} else {
    Write-Host "Cloning repository from kleene-star branch..." -ForegroundColor Gray
    Invoke-SSHCommand "git clone -b kleene-star https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer"
    Write-Host "[OK] Repository cloned" -ForegroundColor Green
}

# Step 3: Set environment variables
Write-Host ""
Write-Host "=== Step 3: Setting Environment Variables ===" -ForegroundColor Cyan
$envScript = @"
export STORAGE_ACCOUNT=$STORAGE_ACCOUNT
export STORAGE_CONTAINER=$STORAGE_CONTAINER
export STORAGE_KEY='$STORAGE_KEY'
export HF_TOKEN='$HF_TOKEN'
export WANDB_API_KEY='$WANDB_API_KEY'
echo "Environment variables set"
"@
$envScript | Out-File -FilePath "$env:TEMP\tropformer_env.sh" -Encoding ASCII
Write-Host "Environment variables script created locally" -ForegroundColor Gray

# Copy env script to VM and source it
Write-Host "Copying environment variables to VM..." -ForegroundColor Gray
scp -i "$SSHKeyPath" -o StrictHostKeyChecking=no "$env:TEMP\tropformer_env.sh" "${AdminUser}@${VM_IP}:~/tropformer_env.sh"
Invoke-SSHCommand "source ~/tropformer_env.sh"
Write-Host "[OK] Environment variables set" -ForegroundColor Green

# Step 4: Mount blob storage
Write-Host ""
Write-Host "=== Step 4: Mounting Blob Storage ===" -ForegroundColor Cyan
$mountCheck = Get-SSHOutput "mountpoint -q /mnt/blob && echo 'mounted' || echo 'not_mounted'"
if ($mountCheck -eq "mounted") {
    Write-Host "[OK] Blob storage already mounted" -ForegroundColor Green
} else {
    Write-Host "Mounting blob storage..." -ForegroundColor Gray
    Invoke-SSHCommand "cd ~/TropFormer && export STORAGE_ACCOUNT=$STORAGE_ACCOUNT && export STORAGE_CONTAINER=$STORAGE_CONTAINER && export STORAGE_KEY='$STORAGE_KEY' && bash research/cloud/azure/scripts/blob_mount.sh"
    Write-Host "[OK] Blob storage mounted" -ForegroundColor Green
}

# Step 5: Setup Python environment
Write-Host ""
Write-Host "=== Step 5: Setting Up Python Environment ===" -ForegroundColor Cyan
Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow
Invoke-SSHCommand "cd ~/TropFormer && export HF_TOKEN='$HF_TOKEN' && export WANDB_API_KEY='$WANDB_API_KEY' && bash research/cloud/azure/scripts/setup_vm.sh"
Write-Host "[OK] Python environment setup complete" -ForegroundColor Green

# Step 6: Run smoke test
Write-Host ""
Write-Host "=== Step 6: Running Smoke Test ===" -ForegroundColor Cyan
Write-Host "This will verify compile+DDP and checkpoint resume..." -ForegroundColor Yellow
Write-Host "Expected runtime: ~2 minutes on 8x H100" -ForegroundColor Gray
Invoke-SSHCommand "cd ~/TropFormer && bash research/cloud/azure/scripts/smoke_test.sh"
Write-Host "[OK] Smoke test completed" -ForegroundColor Green

# Step 7: Verify checkpoint resume capability
Write-Host ""
Write-Host "=== Step 7: Verifying Checkpoint Resume ===" -ForegroundColor Cyan
$ckptList = Get-SSHOutput "ls -t /mnt/blob/checkpoints/smoke/*.pt 2>/dev/null | head -1"
if ($ckptList) {
    Write-Host "Found checkpoint: $ckptList" -ForegroundColor Gray
    Write-Host "Testing resume from arbitrary checkpoint..." -ForegroundColor Gray
    Invoke-SSHCommand "cd ~/TropFormer && torchrun --standalone --nproc_per_node=8 -m research.cloud.azure.train_pretrain_cloud --config pretrain_smoke --tag resume_test --resume '$ckptList' --steps 25"
    Write-Host "[OK] Checkpoint resume verified" -ForegroundColor Green
} else {
    Write-Host "WARNING: No checkpoint found for resume test" -ForegroundColor Yellow
}

# Complete
Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "All steps completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now start full pretraining:" -ForegroundColor Yellow
Write-Host "  ssh -i `"$SSHKeyPath`" ${AdminUser}@${VM_IP}" -ForegroundColor Gray
Write-Host "  cd ~/TropFormer" -ForegroundColor Gray
Write-Host "  bash research/cloud/azure/scripts/run_pretrain.sh" -ForegroundColor Gray
