# TropFormer Azure Cloud Training Guide

Complete guide for training TropFormer on Azure Cloud, including quick test runs and checkpoint/resume verification.

---

## Overview

This guide covers two deployment paths:

- **Path A: Azure ML (recommended)** - Managed training with automatic scaling
- **Path B: Azure VM (fallback/debugging)** - Direct VM control with manual setup

Both paths support:
- `torch.compile` with DDP (fixed to work on multi-GPU)
- Checkpoint save/resume
- Pre-flight dataset verification
- WandB logging

---

## Prerequisites

### Azure Resources

- **Storage Account**: `tropformerblob` (or your custom name)
- **Container**: `tropformer` (or your custom name)
- **VM SKU (Path B)**: `Standard_ND96isr_H100_v5` (8x H100 80GB SXM5)
- **AML Compute Cluster (Path A)**: `h100-cluster` (8x H100 80GB)

### Local Setup

```bash
# Clone the repo (on your local machine)
git clone -b kleene-star https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer
cd TropFormer
```

### Required Environment Variables

Set these before running any training:

```bash
# HuggingFace token (for gated datasets)
export HF_TOKEN="your_hf_token_here"

# Weights & Biases (for experiment tracking)
export WANDB_API_KEY="your_wandb_key_here"
```

---

## Path A: Azure ML (Recommended)

### Step 1: Create AML Workspace (one-time)

```bash
# Login to Azure
az login

# Create workspace (if not exists)
az ml workspace create -w tropformer-aml -g your-resource-group
```

### Step 2: Create AML Environment

```bash
# Upload the curated environment
az ml environment create -f research/cloud/azure/jobs/environment.yaml
```

This creates `tropformer-cuda124@latest` with:
- PyTorch 2.6 + CUDA 12.4
- All required dependencies (datasets, huggingface_hub, zstandard, pyyaml, wandb, etc.)

### Step 3: Configure Compute Targets

Edit the YAML files to match your compute target names:

```bash
# Edit each job YAML to set your compute target
# research/cloud/azure/jobs/aml_pretrain.yaml
# research/cloud/azure/jobs/aml_sft.yaml
# research/cloud/azure/jobs/aml_dpo.yaml

# Change line 7 from:
#   compute: azureml:h100-cluster
# To your actual compute target name:
#   compute: azureml:your-cluster-name
```

### Step 4: Configure Datastore Paths

Edit the YAML files to match your blob datastore:

```bash
# Change path references from:
#   path: azureml://datastores/blobdata/paths/...
# To your actual datastore name:
#   path: azureml://datastores/your-datastore-name/paths/...
```

### Step 5: Set WandB Secret

```bash
# Set as workspace secret so jobs can access it
az ml workspace secrets set -w tropformer-aml -g your-resource-group \
  --name WANDB_API_KEY --value your_wandb_key_here
```

### Step 6: Run Training Pipeline

```bash
# 1. Download raw data → blob (CPU job)
az ml job create -f research/cloud/azure/jobs/aml_data_download.yaml

# 2. Train TMT tokenizer (1x GPU job, ~1h)
az ml job create -f research/cloud/azure/jobs/aml_train_tmt.yaml

# 3. Tokenize + shard (CPU cluster job, ~12h)
az ml job create -f research/cloud/azure/jobs/aml_tokenize_shard.yaml

# 4. Pretrain (8x H100 job, ~5 days)
az ml job create -f research/cloud/azure/jobs/aml_pretrain.yaml

# 5. SFT (8x H100 job, ~36h)
az ml job create -f research/cloud/azure/jobs/aml_sft.yaml

# 6. DPO (8x H100 job, ~6h)
az ml job create -f research/cloud/azure/jobs/aml_dpo.yaml
```

### AML Notes

- **Env var overrides**: The trainers (`train_pretrain_cloud.py`, `train_sft_cloud.py`, `train_dpo_cloud.py`) now read `TROPFORMER_TOKENS_DIR`, `TROPFORMER_TMT_PATH`, `TROPFORMER_OUTPUT_DIR`, and `TROPFORMER_INIT_FROM` from environment variables. This allows AML to pass dynamically-mounted input/output paths without editing config files.
- **Checkpoint resume**: To resume from a checkpoint in AML, set the `TROPFORMER_RESUME` environment variable in the job YAML or pass `--resume` as a command argument.

---

## Path B: Azure VM (Fallback/Debugging)

### Step 1: Provision VM

```bash
# Run from your local machine
bash research/cloud/azure/scripts/azure_provision.sh
```

This script:
- Creates resource group and storage account
- Creates VM with 8x H100
- Generates SSH keys
- Prints connection details

**Save the output** — you'll need:
- Public IP address
- Storage account name
- Storage key
- Container name

### Step 2: SSH into VM

```bash
ssh -i ~/.ssh/tropformer_azure_key azureuser@<public-ip>
```

### Step 3: Clone Repo

```bash
git clone -b kleene-star https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer
cd TropFormer
```

### Step 4: Mount Blob Storage

```bash
# Set environment variables (from azure_provision.sh output)
export STORAGE_ACCOUNT=tropformerblob
export STORAGE_CONTAINER=tropformer
export STORAGE_KEY=<your-storage-key>

# Mount blob at /mnt/blob
bash research/cloud/azure/scripts/blob_mount.sh
```

Verify mount:
```bash
ls /mnt/blob  # Should show empty or existing data
```

### Step 5: Setup Python Environment

```bash
# Install all dependencies
bash research/cloud/azure/scripts/setup_vm.sh
```

This script:
- Verifies GPU availability
- Installs PyTorch and CUDA dependencies
- Installs data processing libraries
- Verifies HF and WandB tokens
- Checks blob mount

### Step 6: Set Environment Variables

```bash
# HuggingFace token (for gated datasets)
export HF_TOKEN="your_hf_token_here"

# WandB API key
export WANDB_API_KEY="your_wandb_key_here"
```

### Step 7: Quick Smoke Test (Recommended)

Before committing to a 5-day training run, verify everything works:

```bash
bash research/cloud/azure/scripts/smoke_test.sh
```

This test:
1. Generates synthetic token shards (no real data needed)
2. Trains a tiny model (d_model=128, 2 blocks) for 20 steps
3. Saves checkpoint at step 10
4. Resumes from step 10 to step 20
5. Verifies checkpoint structure is valid

**Expected runtime**: ~2 minutes on 8x H100

**If smoke test fails**: Check GPU drivers, blob mount, and Python dependencies before proceeding.

### Step 8: Run Full Pretraining

```bash
bash research/cloud/azure/scripts/run_pretrain.sh
```

This script:
1. Runs pre-flight dataset verification (no downloads)
2. Downloads raw shards from HuggingFace (~92B tokens)
3. Trains TMT tokenizer
4. Tokenizes and shards data
5. Launches distributed training with `torchrun`

**Expected runtime**: ~5 days on 8x H100

### Step 9: Run SFT (After Pretraining)

```bash
bash research/cloud/azure/scripts/run_sft.sh
```

This script:
1. Verifies pretrain checkpoint exists
2. Downloads SFT raw shards
3. Runs curriculum SFT training

**Expected runtime**: ~36 hours on 8x H100

### Step 10: Run DPO (After SFT)

```bash
bash research/cloud/azure/scripts/run_dpo.sh
```

This script:
1. Verifies SFT checkpoint exists
2. Downloads DPO preference pairs
3. Runs DPO training

**Expected runtime**: ~6 hours on 8x H100

---

## Checkpointing and Resume

### How Checkpoints Work

All trainers save checkpoints at regular intervals:

- **Pretrain**: Every 1000 steps (configurable via `ckpt_every`)
- **SFT**: Every 2000 steps
- **DPO**: Every 500 steps

Checkpoints are saved as:
```
{output_dir}/{run_tag}_step{step:07d}.pt
```

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state (Adam moments, etc.)
- `scaler_state_dict`: GradScaler state (for mixed precision)
- `step`: Current training step
- `log`: Training metrics history
- `config`: Training configuration
- `run_tag`: Run identifier

### Best Checkpoint

Pretrain additionally saves:
```
{output_dir}/{run_tag}_best.pt
```

This is the checkpoint with the lowest validation BPC (bits per character).

### Resuming from Checkpoint

#### On Azure VM

```bash
# Resume from a specific checkpoint
torchrun --standalone --nproc_per_node=8 \
    -m research.cloud.azure.train_pretrain_cloud \
    --config pretrain_h100x8 \
    --tag resume_run \
    --resume /mnt/blob/checkpoints/pretrain/20250101_pretrain_step050000.pt
```

The trainer will:
1. Load model weights
2. Load optimizer state (critical for Adam moments)
3. Load grad scaler state
4. Continue training from the saved step

#### On Azure ML

Add to the job YAML:

```yaml
environment_variables:
  TROPFORMER_RESUME: azureml://datastores/blobdata/paths/checkpoints/pretrain/20250101_pretrain_step050000.pt
```

Or pass as command argument:

```yaml
command: >-
  python -m research.cloud.azure.train_pretrain_cloud
  --config pretrain_h100x8
  --tag azure_run
  --resume ${{inputs.resume_ckpt}}
```

### Resume Verification

After resuming, verify in the logs:
- Loaded step matches expected checkpoint step
- Loss continues smoothly (no spike)
- Optimizer state loaded successfully

---

## torch.compile and DDP

### The Fix

Previously, `torch.compile` was disabled on multi-GPU due to a `world == 1` guard. This has been fixed across all trainers:

- `train_pretrain_cloud.py`
- `train_sft_cloud.py`
- `train_dpo_cloud.py`
- `train_cloud.py` (legacy)

### Current Behavior

```python
# DDP first, then compile (PyTorch >=2.0 recommended order for max-autotune)
if world > 1:
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False, gradient_as_bucket_view=True,
    )
if cfg.get("compile"):
    model = tsrn_cuda.maybe_compile(
        model, mode=cfg.get("compile_mode", "max-autotune"))
```

### Compile Modes

- **max-autotune**: Aggressive kernel fusion, longer compilation time, best performance (default for pretrain)
- **reduce-overhead**: Moderate fusion, faster compilation (default for SFT, though SFT has compile=False due to ragged shapes)
- **default**: Basic fusion only (used in small_24gb config)

### Verification

The smoke test includes `compile: True` with `compile_mode: max-autotune` to verify compile+DDP interaction works correctly.

---

## Pre-flight Dataset Verification

Before downloading 92B tokens, verify all datasets are reachable:

```bash
# On VM or locally (with HF token set)
python -m research.cloud.azure.data.verify_manifests
```

This checks:
- HuggingFace dataset IDs are valid
- Datasets are accessible (not deleted or private)
- Local paths exist (for proprietary Kyro data)
- Gated datasets have proper auth (via HF_TOKEN)

**Output**: ASCII-safe symbols for Windows compatibility
- `[OK]` ✓ — Dataset reachable
- `[FAIL]` ✗ — Dataset not found or auth required
- `[WARN]` ⚠ — Potential issue
- `[GATE]` ⊘ — Gated dataset (requires HF_TOKEN)

This is automatically run as Step 0 in `run_pretrain.sh`, `run_sft.sh`, and `run_dpo.sh`.

---

## Proprietary Kyro Data

Place Kyro-specific JSONL files in blob storage:

```
/mnt/blob/raw/sft/kyro_synthetic_tools/kyro_synthetic_tools.jsonl
/mnt/blob/raw/sft/kyro_temporal_reasoning/kyro_temporal_reasoning.jsonl
/mnt/blob/raw/sft/kyro_memory_reasoning/kyro_memory_reasoning.jsonl
/mnt/blob/raw/sft/kyro_uncertainty/kyro_uncertainty.jsonl
/mnt/blob/raw/sft/kyro_consequence/kyro_consequence.jsonl
/mnt/blob/raw/sft/kyro_voice_patterns/kyro_voice_patterns.jsonl
/mnt/blob/raw/dpo/kyro_voice_brevity/kyro_voice_brevity.jsonl
/mnt/blob/raw/dpo/human_reviewed_ambiguity/human_reviewed_ambiguity.jsonl
```

The download manifests (`sft_mix.yaml`, `dpo_mix.yaml`) reference these paths. The `download.py` script will skip these entries if the files don't exist (with a warning).

---

## Cost Estimates

| Stage    | SKU                     | Time      | $/h spot | Total |
|----------|-------------------------|-----------|----------|-------|
| Download | Standard_F16s_v2 (CPU)  | ~24 h     | ~0.20    | ~$5   |
| TMT      | Standard_NC4as_T4_v3    | ~2 h      | ~0.15    | ~$0.5 |
| Tokenize | Standard_F32s_v2 (CPU)  | ~12 h     | ~0.40    | ~$5   |
| Pretrain | ND96isr_H100_v5 (8xH100)| ~5 days   | ~10      | ~$1.2K|
| SFT      | ND96isr_H100_v5         | ~36 h     | ~10      | ~$360 |
| DPO      | ND96isr_H100_v5         | ~6 h      | ~10      | ~$60  |

**Total**: ~$1.6K for full pipeline (pretrain → SFT → DPO)

*Prices vary by region. Spot instances can be evicted, extending wall-clock time. All stages are resumable.*

---

## Troubleshooting

### GPU not detected

```bash
nvidia-smi  # Should show GPU info
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### Blob mount failed

```bash
# Check blobfuse2 installation
which blobfuse2

# Check mount
ls /mnt/blob

# Remount
bash research/cloud/azure/scripts/blob_mount.sh
```

### Dataset download fails

```bash
# Verify HF token
export HF_TOKEN="your_token"
python -c "from huggingface_hub import whoami; print(whoami())"

# Run pre-flight check
python -m research.cloud.azure.data.verify_manifests
```

### torch.compile fails

```bash
# Disable compile for debugging
torchrun --standalone --nproc_per_node=8 \
    -m research.cloud.azure.train_pretrain_cloud \
    --config pretrain_h100x8 \
    --no-compile
```

### Resume from checkpoint fails

```bash
# Verify checkpoint exists
ls /mnt/blob/checkpoints/pretrain/

# Verify checkpoint structure
python -c "
import torch
ckpt = torch.load('/mnt/blob/checkpoints/pretrain/20250101_pretrain_step050000.pt', map_location='cpu')
print(ckpt.keys())
"
```

### Out of memory

Reduce batch size or gradient accumulation in the config:

```python
# In pretrain_h100x8.py
"batch_size": 4,  # Reduce from default
"grad_accum_steps": 8,  # Increase to maintain effective batch
```

---

## Additional Considerations

### Branch Consistency

Always use the `kleene-star` branch for Azure training:

```bash
git checkout kleene-star
git pull origin kleene-star
```

### Monitoring

- **WandB**: View training metrics in real-time at wandb.ai
- **Azure Monitor**: For AML jobs, use the Azure Portal to view logs and metrics
- **VM logs**: Check stdout/stderr for training progress

### Spot Instance Eviction

If using spot instances:
- All stages are resumable via checkpoints
- Monitor eviction notices in Azure Portal
- Consider using `--resume` to automatically resume after eviction

### Data Freshness

The pretrain manifests reference specific HuggingFace dataset versions. If datasets are updated, you may need to pin versions in the manifests:

```yaml
# In pretrain_mix.yaml
hf_dataset: "allenai/fineweb_edu"
hf_config: "default"
revision: "v1.0.0"  # Pin to specific version
```

---

## Quick Reference Commands

### VM Setup
```bash
bash research/cloud/azure/scripts/setup_vm.sh
```

### Smoke Test
```bash
bash research/cloud/azure/scripts/smoke_test.sh
```

### Pretrain
```bash
bash research/cloud/azure/scripts/run_pretrain.sh
```

### SFT
```bash
bash research/cloud/azure/scripts/run_sft.sh
```

### DPO
```bash
bash research/cloud/azure/scripts/run_dpo.sh
```

### Verify Datasets
```bash
python -m research.cloud.azure.data.verify_manifests
```

### Resume from Checkpoint
```bash
torchrun --standalone --nproc_per_node=8 \
    -m research.cloud.azure.train_pretrain_cloud \
    --config pretrain_h100x8 \
    --resume /mnt/blob/checkpoints/pretrain/20250101_pretrain_step050000.pt
```

---

## Summary

This guide provides a complete end-to-end process for training TropFormer on Azure Cloud:

1. **Choose path**: Azure ML (managed) or Azure VM (manual)
2. **Set up resources**: Storage, compute, environment
3. **Configure tokens**: HF_TOKEN, WANDB_API_KEY
4. **Verify setup**: Run smoke test
5. **Train pipeline**: Pretrain → SFT → DPO
6. **Monitor**: WandB + Azure logs
7. **Resume**: Checkpoint-based resumption for robustness

All critical issues have been addressed:
- ✅ `torch.compile` works with DDP on multi-GPU
- ✅ AML path overrides via environment variables
- ✅ Pre-flight dataset verification
- ✅ Checkpoint save/resume functionality
- ✅ Smoke test for quick validation

You are now ready to train TropFormer on Azure Cloud.
