# TropFormer — Cloud Training (NVIDIA / CUDA)

Auto-detecting CUDA port of the TSRNGist research stack.  Designed to run on
**any** NVIDIA-cloud GPU — from a free-tier T4 to an 8× H100 box — with a single
trainer script.

> **Branch**: `nvidia-cloud`. The DirectML AMD branch (`nexus-maslov-sheaf-rgfp`)
> is preserved and continues to work for local validation.

---

## Quick start (TL;DR)

```bash
# Clone the cloud branch on whatever pod you rented
git clone -b nvidia-cloud https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer
cd TropFormer

# Install everything (creates .venv, installs torch+cu121, flash-attn, deps, downloads enwik8)
bash research/cloud/launch/runpod_setup.sh

# Activate venv
source .venv/bin/activate

# Train (auto-detects # of GPUs, picks bf16 on Ampere+, fp16 on older)
bash research/cloud/launch/ssh_launch.sh medium_40gb my_first_run
```

---

## What's in this directory

```
research/cloud/
├── __init__.py
├── tsrn_cuda.py              CUDA fast-path helpers (device, optimizer, prefix_max, DDP init)
├── train_cloud.py            Main trainer — auto-detects single GPU / DDP / DML / CPU
├── configs/                  Four preset configurations
│   ├── small_24gb.py            RTX 4090, A10, T4, L4
│   ├── medium_40gb.py           A100-40, A6000
│   ├── large_80gb.py            A100-80, H100, H200
│   └── multi_8xa100.py          8×A100/H100 DDP
├── launch/
│   ├── Dockerfile               CUDA 12.1 + PyTorch 2.4 + flash-attn
│   ├── modal_app.py             Modal Labs serverless wrapper
│   ├── runpod_setup.sh          One-shot setup for RunPod / Lambda / vast.ai / your server
│   └── ssh_launch.sh            Provider-agnostic train launcher (single-GPU + DDP)
├── requirements_cuda.txt     CUDA-only deps (flash-attn, deepspeed, bitsandbytes, wandb)
└── README.md                 (this file)
```

---

## Running on each provider

### 1. Modal Labs (recommended for "no funding" — $30/mo free credits, no card)

[modal.com](https://modal.com) gives **$30/month** in free credits with just an email.
That buys roughly:
- 30 hours on a T4
- 10 hours on an A10G
- 4 hours on an A100-40
- 2 hours on an H100

```bash
# One-time setup
pip install modal
modal token new

# Smoke test (CPU container, free)
modal run research/cloud/launch/modal_app.py::smoke

# Real training
modal run research/cloud/launch/modal_app.py::train_a100_40 --tag a100_run0 --steps 100000
```

Checkpoints persist in a Modal **Volume** named `tropformer-vol` and survive
container shutdown.  Pull them locally:

```bash
modal volume get tropformer-vol checkpoints/<run_tag>_best.pt ./checkpoints/
```

### 2. RunPod / Lambda Labs / vast.ai (cheapest $/hour for sustained runs)

These let you rent a GPU pod by the hour or second.  RunPod also has:
- ~$10 of free credit on signup (varies)
- Spot pricing as low as $0.13/hr for an RTX 3090
- 24-hour max session on community cloud (use Secure Cloud for longer)

```bash
# In the pod's web terminal:
git clone -b nvidia-cloud https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git TropFormer
cd TropFormer
bash research/cloud/launch/runpod_setup.sh
source .venv/bin/activate

# Single GPU
bash research/cloud/launch/ssh_launch.sh medium_40gb run0

# DDP across all visible GPUs (auto)
bash research/cloud/launch/ssh_launch.sh multi_8xa100 multi8
```

### 3. Raw SSH on any CUDA box (max flexibility)

Same `runpod_setup.sh` works on any Ubuntu 22.04 / Debian / Amazon Linux 2 box
with an NVIDIA driver ≥ 525.  Tested on:
- AWS `p4d.24xlarge` (8× A100-40)
- GCP `a3-highgpu-8g`  (8× H100)
- Lambda Labs 1× / 8× A100
- Personal RTX 4090 desktops

For **Docker-based** deployment (better isolation, faster spin-up):

```bash
docker build -t tropformer:cuda12 -f research/cloud/launch/Dockerfile .

# Single GPU
docker run --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints \
    tropformer:cuda12 \
    bash research/cloud/launch/ssh_launch.sh medium_40gb docker_run

# DDP — pass NVIDIA_VISIBLE_DEVICES if you want to limit GPUs
docker run --gpus all --ipc=host \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    tropformer:cuda12 \
    bash research/cloud/launch/ssh_launch.sh multi_8xa100 docker_ddp
```

---

## Free-tier credit map (as of late 2026)

| Provider              | Free credit          | Best GPU available           | Notes                                       |
|-----------------------|----------------------|------------------------------|---------------------------------------------|
| **Modal Labs**        | $30/month no card    | H100, A100-80, L40S          | Best free tier overall.  Serverless.        |
| **Google Colab**      | Free notebook (T4)   | T4 (16 GB)                   | 12-hour cap.  Use `small_24gb` preset.      |
| **Kaggle**            | 30 GPU-hr/week       | P100 (16 GB) or T4×2         | No DDP between the two T4s; use single.     |
| **Lightning AI Studio** | $15/month free     | T4, A10G, A100               | Web IDE; good for iterating.                |
| **HF Spaces**         | Free A10G (with PR)  | A10G                         | Apply for academic/OSS credits.             |
| **RunPod**            | ~$10 signup          | H100, A100, RTX 4090         | Lowest $/hr on community cloud.             |
| **vast.ai**           | varies               | Any (peer-to-peer)           | Cheap RTX 3090/4090.  Reliability varies.   |
| **Lambda Labs**       | Promo credits        | H100, A100-80                | Sometimes runs $200 free credit promos.     |
| **Hyperbolic**        | $10 signup           | H100                         | New player; worth checking.                 |
| **Paperspace**        | Free M4000 / sometimes A4000 | A100 on paid          | Free-tier GPU tier exists but queues long.  |
| **AWS / GCP / Azure** | $300 signup credits  | A100-80, H100                | Painful to set up.  Use as last resort.     |

**Strategy for zero-budget pretraining**: chain Modal $30/mo + Kaggle 30 GPU-hr/wk
+ Colab free as your three rotating sources, all writing to the same HuggingFace
Hub repo for checkpoints (HF storage is free up to 1 TB for public repos).
Resume from the last HF-Hub checkpoint each session.

---

## Choosing a preset

| Preset           | Params  | Context | Per-GPU batch | Eff. batch | GPU mem peak | Steps  | Wall-clock¹     |
|------------------|---------|---------|---------------|------------|--------------|--------|-----------------|
| `small_24gb`     | ~22 M   | 512     | 8 × 4         |  32        | ~16 GB       | 100K   | 3.5h on RTX4090 |
| `medium_40gb`    | ~50 M   | 1024    | 16 × 2        |  32        | ~30 GB       | 100K   | 6h on A100-40   |
| `large_80gb`     | ~150 M  | 2048    | 32 × 2        |  64        | ~60 GB       | 150K   | 10h on H100     |
| `multi_8xa100`   | ~100 M  | 1024    | 16 × 4 × 8    | 512        | ~25 GB/GPU   | 100K   | 1.5h on 8×A100  |

¹ Wall-clock is approximate, with `bf16 + grad-ckpt + torch.compile` (where supported).

---

## CLI reference

```text
python -m research.cloud.train_cloud --help

  --preset {small_24gb,medium_40gb,large_80gb,multi_8xa100}
  --steps INT           override total step count
  --batch INT           override per-GPU micro-batch
  --grad-accum INT      override gradient-accumulation steps
  --lr FLOAT            override peak learning rate
  --context INT         override context length
  --ckpt-every INT      override checkpoint cadence (default: 5000)
  --eval-every INT      override eval cadence (default: ~ steps/50)
  --resume PATH         resume from a checkpoint (.pt)
  --tag TAG             string suffix for filenames
  --no-compile          disable torch.compile even if preset enables it
  --no-grad-ckpt        disable gradient checkpointing (uses more VRAM)
  --use-8bit-optim      use bitsandbytes AdamW8bit (saves optimizer VRAM)
```

---

## Architectural fidelity vs the DML branch

The cloud trainer instantiates the **same** `TSRNGist` class as the DML branch
(`research/tsrn_gist.py`).  The model is unmodified.  Only the *training-loop
infrastructure* is different:

| Aspect              | DML (`research/tsrn_convergence_gist.py`) | CUDA (`research/cloud/train_cloud.py`)         |
|---------------------|--------------------------------------------|------------------------------------------------|
| Optimizer           | `AdamWDML` (custom, avoids `aten::lerp`)   | `torch.optim.AdamW(fused=True)`                |
| Prefix-max          | Hillis-Steele (fp32, eager)                | `torch.cummax` (one CUDA kernel) via fastpath  |
| Mixed precision     | fp32 only                                  | bf16 (Ampere+) / fp16 (older) via autocast     |
| Distributed         | single GPU only                            | auto-detect → single / DDP via torchrun        |
| `torch.compile`     | n/a (DML)                                  | enabled where supported                        |
| Memory mitigation   | `gc.collect()` every 200/500 steps         | `gc.collect()` + `torch.cuda.empty_cache()`    |
| Best-model save     | write to disk + free immediately           | same                                           |
| NEXUS Maslov-h cycle | yes                                        | yes (identical schedule)                       |
| NEXUS Sheaf PE      | yes (in model)                             | yes (model unchanged)                          |
| NEXUS RG fixed-pt   | yes (in model)                             | yes (model unchanged)                          |

All checkpoints are **forward and backward compatible** between branches —
DML-trained checkpoints can resume on CUDA and vice versa.

---

## Verifying the run

After a successful first iteration you should see, on rank 0:

```
  Device  : NVIDIA CUDA  (1 GPU)
  GPU 0   : NVIDIA A100-SXM4-40GB  (39.4 GiB, sm_80)
  Backend : torch 2.4.1 + cuda 12.1
  AMP     : torch.bfloat16 (scaler=off)
  World   : 1 rank, local rank 0
  GPU mem : 39.4 GiB

  TSRNGist: 50,123,008 parameters
  Vocab   : 256  |  Context: 1024  |  d_model: 512
  Steps   : 100000  |  Batch: 16*2*1=32
  LR      : 0.00025  |  Eval/2000 steps  |  Ckpt/5000

========================================================================================
  TSRNGist Cloud Trainer  —  enwik8 byte-level  —  100000 steps
========================================================================================
  Step      TrLoss    TrBPC   ValLoss     ValPPL   ValBPC    GNorm    ms/step
----------------------------------------------------------------------------------------
     1      6.2438   9.0091    6.2123     499.34   8.9637    2.341    427.2ms
  2000      1.6892   2.4376    1.6543     5.231    2.3873    0.987    245.8ms
  4000      1.4123   2.0381    1.3987     4.049    2.0177    0.834    242.1ms
   ...
```

Successful indicators:
- `TrBPC` and `ValBPC` both decreasing
- `GNorm` < 5 (otherwise the LR is too high or warmup too short)
- `ms/step` constant (memory should NOT grow — that's why we ported)

---

## Troubleshooting

### "CUDA out of memory" on first step

- Drop one preset tier (e.g. `medium_40gb` → `small_24gb`)
- Or pass `--batch 4 --grad-accum 8` (same effective batch, half the peak memory)
- Or pass `--use-8bit-optim` (saves ~3× param-bytes of optimizer memory)

### `torch.compile` errors

Pass `--no-compile`.  Some operations in the channel-chunked tropical attention
are still graph-break sources in PyTorch 2.4; we'll fix this incrementally.

### flash-attn not loading

Not fatal — the trainer uses PyTorch SDPA internally as a fallback.  flash-attn
gives ~25% speedup on Ampere+ but is optional.

### Resuming on a different number of GPUs

Optimizer state ordering uses `named_parameters()` (deterministic), so resumes
across world sizes work.  Just pass `--resume` and the right preset.
