# Cloud Training Roadmap — Lightning.ai + Microsoft Azure

End-to-end guide for validating CUDA kernels on Lightning.ai (15 credits) and
running full convergence training on Microsoft Azure ($5000 startup credits).

This document is the single source of truth for the cloud workflow.
Open it on the VM, on your laptop, or wherever — every step is copy-pasteable.

---

## What's Already Built

| Asset | Path | Purpose |
|---|---|---|
| Lightning quickstart | `research/cloud/launch/lightning_quickstart.sh` | Paste-and-run CUDA validation |
| Azure VM creator | `research/cloud/launch/azure_vm_create.sh` | One-shot Azure VM provisioning |
| Generic SSH launcher | `research/cloud/launch/ssh_launch.sh` | Single + multi-GPU training |
| Setup script | `research/cloud/launch/runpod_setup.sh` | Works on any Ubuntu+CUDA VM |
| Trainer | `research/cloud/train_cloud.py` | Auto-detects DDP, AMP, GPU mem |
| Presets | `research/cloud/configs/*.py` | small_24gb, medium_40gb, large_80gb, multi_8xa100 |
| Kleene/Tier flags | `--use-kleene-ssm`, `--tier {nano,pro,kyro}` | Plumbed end-to-end |
| WandB logging | `--wandb-project` / `--wandb-run-name` flags | Optional cloud monitoring |

---

## ⚠️ Do This RIGHT NOW (before everything else)

Azure GPU quota defaults to **0** for new accounts. Quota approval takes
**1–3 business days** — submit it before you do anything else.

1. Go to <https://portal.azure.com> → search **"Quotas"**
2. Filter: **Compute** → region **East US 2**
3. Request increases:
   - `Standard NCSv3 Family vCPUs` → **12** (1× V100 = 6 vCPU; this gives you 2)
   - `Standard NCADSv4 Family vCPUs` → **24** (1× A100 80GB = 24 vCPU)
   - `Total Regional Spot vCPUs` → **48** (so you can use Spot pricing, ~75% off)
4. Submit. Azure will email you when approved.

**While you wait, do the Lightning.ai validation (Phase 1).**

---

## Phase 1 — Lightning.ai CUDA Validation

**Goal:** Verify the entire stack runs correctly on real CUDA hardware before
spending Azure credits. Total cost ~$3–5 of your 15 Lightning credits.

### What it validates

- All 16 `test_kleene.py` tests pass on CUDA
- KleeneSSM faster than TropicalSSM baseline on real GPU
- Loss curves go down (no NaN, no divergence) over a 2k-step smoke run
- Tokens/sec measurement → extrapolates exact Azure budget

### You do (5 minutes, manual)

1. Go to <https://lightning.ai/studios>
2. Sign in (free account works)
3. Click **New Studio**
4. GPU: pick **L4 24GB** (~$0.80/hr)
5. Click **Start**, wait ~30 seconds
6. Open the **Terminal** tab in the Studio web UI

### Then paste:

```bash
git clone -b kleene-star https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer
cd TropFormer
bash research/cloud/launch/lightning_quickstart.sh
```

**That's it.** The script runs ~25–30 minutes and prints:

- All 16 test results
- Baseline (TropicalSSM) vs KleeneSSM step-time comparison
- Estimated time + cost for full Nano/Pro runs on V100/A100

### What to look for (success criteria)

- ✅ All 16 tests pass
- ✅ KleeneSSM step time **lower** than TropicalSSM
- ✅ Both runs produce decreasing val_bpc (no NaN)
- ✅ Estimated Nano 100k cost on V100 < $10

If any of these fail, **stop**. Fix locally, push, repeat. Don't move to Azure.

### Stop the Lightning Studio when done

In the Lightning UI, click the Studio's **stop** button. Otherwise you keep
paying for idle GPU time.

---

## Phase 2 — Azure Full Convergence Training

### One-time local setup (you do, on your Windows machine)

```powershell
winget install -e --id Microsoft.AzureCLI
az login
```

A browser window opens → log in with your startup account → close it when prompted.

Test it works:

```powershell
az account show
```

### Edit the VM script (3 lines)

Open `research/cloud/launch/azure_vm_create.sh`. Edit only the top section:

```bash
RESOURCE_GROUP="tropformer-rg"
LOCATION="eastus2"          # match where your quota was approved
VM_SIZE="Standard_NC6s_v3"  # 1x V100 16GB. See VM Size table below.
SPOT="true"                 # true = ~75% cheaper but can be evicted
```

### VM size cheat-sheet

| VM Size | GPU | VRAM | On-demand $/hr | Spot $/hr | Use for |
|---|---|---|---|---|---|
| `Standard_NC4as_T4_v3` | 1× T4 | 16 GB | $0.53 | ~$0.15 | Cheapest validation |
| `Standard_NC6s_v3` | 1× V100 | 16 GB | $3.06 | ~$0.70 | **Nano training** |
| `Standard_NV36ads_A10_v5` | 1× A10 | 24 GB | ~$2.50 | ~$0.80 | Small Pro experiments |
| `Standard_NC24ads_A100_v4` | 1× A100 | 80 GB | $14.07 | ~$4.00 | **Pro training** |
| `Standard_NC48ads_A100_v4` | 2× A100 | 80 GB ea | $28.14 | ~$8.00 | DDP Pro training |
| `Standard_ND96asr_A100_v4` | 8× A100 | 40 GB ea | $27.20 | ~$8.00 | Kyro full convergence |

Spot prices fluctuate; check current pricing at:
<https://azure.microsoft.com/pricing/details/virtual-machines/linux/>

### Create the VM

```bash
# In Git Bash or WSL:
bash research/cloud/launch/azure_vm_create.sh
```

The script prints the VM's public IP and the exact `ssh` command at the end.

### Set up the VM (run on the VM)

```bash
ssh azureuser@<YOUR_VM_IP>
git clone -b kleene-star https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer
cd TropFormer
bash research/cloud/launch/runpod_setup.sh
```

Setup takes ~5 minutes (PyTorch + deps + enwik8 download + smoke import test).

### Start tmux (so training survives SSH drops)

```bash
tmux new -s train
```

Re-attach later with `tmux attach -t train`. Detach with `Ctrl+B` then `D`.

### Training runs (in order)

#### Run A: Nano baseline + Kleene (100k steps, ~7 hrs, ~$5 V100 spot)

```bash
source .venv/bin/activate
bash research/cloud/launch/ssh_launch.sh small_24gb nano_v1_kleene \
    --use-kleene-ssm --tier nano \
    --wandb-project tropformer-cloud --wandb-run-name nano_v1_kleene
```

#### Run B: Resume + extend to 200k steps

```bash
bash research/cloud/launch/ssh_launch.sh small_24gb nano_v2_200k \
    --use-kleene-ssm --tier nano --steps 200000 \
    --resume checkpoints/*nano_v1_kleene_best.pt \
    --wandb-project tropformer-cloud --wandb-run-name nano_v2_200k
```

#### Run C: Pro tier (A100 80GB, ~18 hrs, ~$72 spot)

First, **deallocate** the V100 VM and re-create with A100 size:

```bash
# Local machine:
az vm deallocate -g tropformer-rg -n tropformer-gpu
# Edit azure_vm_create.sh: VM_SIZE="Standard_NC24ads_A100_v4"
# Then either delete + recreate, or resize:
az vm resize -g tropformer-rg -n tropformer-gpu \
    --size Standard_NC24ads_A100_v4
az vm start -g tropformer-rg -n tropformer-gpu
```

Then on the VM:

```bash
bash research/cloud/launch/ssh_launch.sh large_80gb pro_v1_kleene \
    --use-kleene-ssm --tier pro --steps 500000 \
    --wandb-project tropformer-cloud --wandb-run-name pro_v1_kleene
```

#### Run D (optional): Kyro tier — 8× A100 DDP

Requires `Standard_ND96asr_A100_v4`. Multi-node not needed at this scale.

```bash
NUM_GPUS=8 bash research/cloud/launch/ssh_launch.sh multi_8xa100 kyro_v1 \
    --use-kleene-ssm --tier kyro --steps 1000000 \
    --wandb-project tropformer-cloud --wandb-run-name kyro_v1
```

### Monitoring (WandB)

If you set `--wandb-project`, the trainer logs every eval step:
`train_loss, train_bpc, val_bpc, val_ppl, lr, grad_norm, step_time_ms`,
plus the Maslov-h schedule and PaCS effective context.

View live training at <https://wandb.ai> → your project → run name.

WandB API key setup (one-time, on the VM):

```bash
wandb login   # paste API key from wandb.ai/authorize
```

### Checkpoint locations

All checkpoints save to `checkpoints/` on the VM:

- Every 5000 steps: `checkpoints/<DATE>_train_cloud_<tag>_step050000.pt`
- Best val BPC: `checkpoints/<DATE>_train_cloud_<tag>_best.pt`
- Final: `checkpoints/<DATE>_train_cloud_<tag>_final_step*.pt`
- Interrupt (Ctrl+C): `checkpoints/<DATE>_train_cloud_<tag>_interrupt_step*.pt`

### Pull checkpoints to your local machine

```powershell
# From your Windows machine (Git Bash or WSL):
mkdir -p azure_checkpoints
scp -r azureuser@<VM_IP>:~/TropFormer/checkpoints/* azure_checkpoints/
scp -r azureuser@<VM_IP>:~/TropFormer/results/* azure_checkpoints/
scp -r azureuser@<VM_IP>:~/TropFormer/logs/* azure_checkpoints/
```

### **CRITICAL — stop the VM between runs**

Spot pricing is cheap, but Azure still charges while the VM is running.

```powershell
# Stops + releases the VM (no compute charges, only disk):
az vm deallocate -g tropformer-rg -n tropformer-gpu

# Restart when you're ready for the next run:
az vm start -g tropformer-rg -n tropformer-gpu
```

`stop` (without deallocate) **still charges**. Always use `deallocate`.

### Tear down completely (when project is done)

```powershell
az group delete -n tropformer-rg --yes --no-wait
```

This deletes the VM, disks, network, public IP — everything.

---

## Budget Breakdown

| Run | VM | Hours | Spot Cost | Cumulative |
|---|---|---|---|---|
| Lightning validation | L4 | 0.5 | ~$0.40 | $0.40 |
| Run A: Nano 100k | V100 | ~7 | ~$5 | $5.40 |
| Run B: Nano 200k | V100 | ~7 | ~$5 | $10.40 |
| Run C: Pro 500k | A100 | ~18 | ~$72 | $82.40 |
| Run D: Kyro 1M | 8× A100 | ~30 | ~$240 | $322.40 |
| 5× ablations on Nano | V100 | 35 | ~$25 | $347.40 |

**Total: ~$350 of your $5000 budget.** You can run dozens of experiments.

---

## Division of Labor

| Task | Who |
|---|---|
| Write all scripts + configs | Cascade |
| Update model code (flags, wiring) | Cascade |
| Write WandB / monitoring integration | Cascade |
| Click "New Studio" on Lightning.ai | **You** |
| Request Azure quota in portal | **You** |
| `az login` on your machine | **You** |
| Run `azure_vm_create.sh` | **You** (copy-paste) |
| SSH into VM + run `runpod_setup.sh` | **You** (copy-paste) |
| Launch tmux + training commands | **You** (copy-paste) |
| Download checkpoints with `scp` | **You** (copy-paste) |
| Deallocate VM between runs | **You** (1 command) |
| Read training logs / decide next steps | **Both** (you watch, Cascade interprets) |

---

## Troubleshooting

### "OutOfMemory" on Azure VM

- Drop `--batch` (e.g. `--batch 4`) and bump `--grad-accum` to keep effective
  batch the same.
- Or switch preset: `large_80gb` → `medium_40gb` → `small_24gb`.

### Spot VM evicted mid-training

- All progress is preserved — checkpoints saved every 5000 steps.
- Re-create VM (or wait for spot capacity) and resume:

  ```bash
  bash research/cloud/launch/ssh_launch.sh small_24gb resumed \
      --resume checkpoints/<latest>.pt
  ```

### `nvidia-smi` shows no GPU on VM

- Check VM size — confirm it's an NC/ND/NV series.
- Microsoft DSVM image already has drivers. If using a different image,
  install: `sudo apt-get install -y nvidia-driver-535`, then reboot.

### Training is slow, GPU util < 50%

- Disable gradient checkpointing: `--no-grad-ckpt` (uses more VRAM).
- Increase batch size if VRAM allows.
- Enable `torch.compile`: don't pass `--no-compile`.
- Check for CPU bottleneck on dataloading (rare with enwik8, common on C4).

### WandB not logging

- `wandb login` on the VM with your API key from <https://wandb.ai/authorize>.
- Check `WANDB_MODE` env var isn't set to `disabled`.
- If air-gapped, set `WANDB_MODE=offline` and sync later with `wandb sync`.

---

## Quick Reference Commands

```bash
# Lightning.ai validation
bash research/cloud/launch/lightning_quickstart.sh

# Azure VM (run locally)
bash research/cloud/launch/azure_vm_create.sh
ssh azureuser@<IP>

# On the VM
bash research/cloud/launch/runpod_setup.sh
tmux new -s train
source .venv/bin/activate
bash research/cloud/launch/ssh_launch.sh small_24gb my_run \
    --use-kleene-ssm --tier nano \
    --wandb-project tropformer-cloud

# Local machine — VM management
az vm deallocate -g tropformer-rg -n tropformer-gpu     # stop billing
az vm start      -g tropformer-rg -n tropformer-gpu     # resume
az vm show       -g tropformer-rg -n tropformer-gpu -d  # show IP / status
az group delete  -n tropformer-rg --yes --no-wait       # nuke everything

# Pull checkpoints
scp -r azureuser@<IP>:~/TropFormer/checkpoints/* ./azure_checkpoints/
```
