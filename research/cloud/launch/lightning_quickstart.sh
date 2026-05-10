#!/usr/bin/env bash
# Lightning.ai Studio quickstart — CUDA kernel validation.
# =========================================================
# Run this in the Lightning.ai Studio TERMINAL (not locally).
#
# 1. Go to lightning.ai/studios
# 2. Click "New Studio"
# 3. Choose GPU: L4-24GB (cheapest that validates CUDA, ~$0.80/hr)
# 4. Open the Terminal tab
# 5. Paste and run this script
#
# What this validates (~20-30 min total, ~$0.40 cost):
#   - All 16 test_kleene.py tests pass on real CUDA
#   - KleeneSSM vs TropicalSSM GPU speedup measured
#   - 2000-step smoke train: loss goes down, no NaN, step time measured
#   - Cost per full training run estimated from step time

set -euo pipefail

REPO="${REPO:-/teamspace/studios/this_studio/TropFormer}"
BRANCH="${BRANCH:-kleene-star}"
LOG_FILE="${LOG_FILE:-logs/lightning_validation_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p logs

# Logging function - use this for all echo statements
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

log "=================================================================="
log "  TropFormer Lightning.ai CUDA Validation"
log "=================================================================="
log "  Full terminal output being logged to: ${LOG_FILE}"

# ---------------------------------------------------------------------------
# 1. Clone / update repo
# ---------------------------------------------------------------------------
if [ ! -d "${REPO}" ]; then
    log "==> Cloning TropFormer (${BRANCH} branch)..."
    git clone -b "${BRANCH}" \
        https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git \
        "${REPO}" | tee -a "${LOG_FILE}"
else
    log "==> Updating existing repo..."
    cd "${REPO}" && git fetch origin && git checkout "${BRANCH}" && git pull | tee -a "${LOG_FILE}"
fi
cd "${REPO}"

# ---------------------------------------------------------------------------
# 2. Install / upgrade PyTorch for CUDA
# ---------------------------------------------------------------------------
# Lightning.ai Studios forbid creating new venvs — every Studio has exactly
# one default conda env.  Detect that case and install into the active env.
log ""
log "==> Setting up Python environment..."
IS_LIGHTNING=0
if [ -d "/teamspace" ] || [ -n "${LIGHTNING_CLOUD_URL:-}" ]; then
    IS_LIGHTNING=1
    log "    Detected Lightning.ai Studio — using default conda env (no venv)."
fi

if [ "${IS_LIGHTNING}" = "0" ]; then
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Lightning.ai images already ship a CUDA-capable torch; only upgrade if
# torch isn't installed or doesn't have CUDA available.
NEEDS_TORCH=$(python -c 'import sys
try:
    import torch
    sys.exit(0 if torch.cuda.is_available() else 1)
except Exception:
    sys.exit(1)') ; rc=$?
if [ "${rc}" != "0" ]; then
    log "    Installing PyTorch (CUDA 12.1)..."
    pip install -q --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.4.1" "torchvision==0.19.1" 2>&1 | tee -a "${LOG_FILE}"
else
    log "    PyTorch with CUDA already present — skipping torch install."
fi

pip install -q -r requirements.txt 2>&1 | tee -a "${LOG_FILE}"
pip install -q -r research/cloud/requirements_cuda.txt 2>&1 | tee -a "${LOG_FILE}"
log "==> PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
log "==> CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
log "==> GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

# ---------------------------------------------------------------------------
# 3. Download enwik8 (validation dataset, 36MB, fast)
# ---------------------------------------------------------------------------
log ""
log "==> Downloading enwik8..."
mkdir -p data
if [ ! -f data/enwik8 ]; then
    wget -q -O data/enwik8.zip http://mattmahoney.net/dc/enwik8.zip 2>&1 | tee -a "${LOG_FILE}"
    (cd data && unzip -q -o enwik8.zip) 2>&1 | tee -a "${LOG_FILE}"
    log "   enwik8 downloaded."
fi

# ---------------------------------------------------------------------------
# 4. Run the full test suite on GPU
# ---------------------------------------------------------------------------
log ""
log "=================================================================="
log "  STEP 1/5 — Correctness tests: kleene + tropical kernels (GPU)"
log "=================================================================="
python research/test_kleene.py 2>&1 | tee -a "${LOG_FILE}"
log ""
log "  -- tropical_kernels correctness tests --"
python -m research.test_tropical_kernels 2>&1 | tee -a "${LOG_FILE}"
log ""
log ">>> TEST SUITE COMPLETE"

# ---------------------------------------------------------------------------
# 5. Forward-pass profiler — finds the eager bottleneck for each backend
# ---------------------------------------------------------------------------
log ""
log "=================================================================="
log "  STEP 2/5 — Forward-pass profiler (TropicalSSM baseline)"
log "=================================================================="
python -m research.profile_forward --preset small_24gb \
    2>&1 | tee logs/profile_tropical_ssm.log | tee -a "${LOG_FILE}"

log ""
log "=================================================================="
log "  STEP 3/5 — Forward-pass profiler (KleeneSSM, soft Tensor-Core path)"
log "=================================================================="
python -m research.profile_forward --preset small_24gb \
    --use-kleene-ssm --tier nano \
    --tropical-mode soft --tropical-h 1.0 \
    2>&1 | tee logs/profile_kleene_soft.log | tee -a "${LOG_FILE}"

log ""
log "=================================================================="
log "  STEP 3b/5 — Forward-pass profiler (KleeneSSM, hard Triton)"
log "=================================================================="
python -m research.profile_forward --preset small_24gb \
    --use-kleene-ssm --tier nano \
    --tropical-mode triton --tropical-h 0 \
    2>&1 | tee logs/profile_kleene_triton.log | tee -a "${LOG_FILE}" || \
    log "  [warn] triton profile failed (kernel issue?) — continuing"

# ---------------------------------------------------------------------------
# 6. 2000-step smoke train: baseline (TropicalSSM) vs Kleene
# ---------------------------------------------------------------------------
log ""
log "=================================================================="
log "  STEP 4/5 — 2000-step baseline smoke run (TropicalSSM)"
log "=================================================================="
mkdir -p checkpoints results logs

# Baseline: TropicalSSM (no Kleene flag)
python -m research.cloud.train_cloud \
    --preset small_24gb \
    --steps 2000 \
    --tag smoke_baseline \
    2>&1 | tee logs/smoke_baseline.log | tee -a "${LOG_FILE}"

log ""
log "=================================================================="
log "  STEP 5/5 — 2000-step KleeneSSM smoke run (soft Tensor-Core path)"
log "=================================================================="
# KleeneSSM run (tier=nano, Kleene SSM enabled, soft tropical matmul)
python -m research.cloud.train_cloud \
    --preset small_24gb \
    --steps 2000 \
    --tag smoke_kleene \
    --use-kleene-ssm \
    --tier nano \
    2>&1 | tee logs/smoke_kleene.log | tee -a "${LOG_FILE}"

# ---------------------------------------------------------------------------
# 6. Extract and compare results
# ---------------------------------------------------------------------------
log ""
log "=================================================================="
log "  RESULTS SUMMARY"
log "=================================================================="

python3 - <<'PYEOF' 2>&1 | tee -a "${LOG_FILE}"
import glob, json, os

def extract_stats(tag):
    files = sorted(glob.glob(f"results/*{tag}*.json"))
    if not files:
        return None
    d = json.load(open(files[-1]))
    return d

baseline = extract_stats("smoke_baseline")
kleene   = extract_stats("smoke_kleene")

print("  Metric           Baseline (TropSSM)   KleeneSSM")
print("  " + "-"*52)
for k in ["val_bpc", "step_time_ms", "tokens_per_sec"]:
    b = baseline.get(k, "n/a") if baseline else "n/a"
    kl = kleene.get(k, "n/a") if kleene else "n/a"
    print(f"  {k:<20} {str(b):<20} {str(kl):<20}")

# Extrapolate full-run cost
if kleene and "step_time_ms" in kleene:
    step_ms = kleene["step_time_ms"]
    for steps, label in [(100_000, "Nano 100k"), (500_000, "Pro 500k")]:
        hrs = (step_ms * steps) / (1000 * 3600)
        cost_l4  = hrs * 0.80
        cost_v100 = hrs * 0.70
        cost_a100 = hrs * 4.00
        print(f"\n  {label} ({steps:,} steps):")
        print(f"    Estimated time : {hrs:.1f} hours")
        print(f"    L4 cost        : ${cost_l4:.0f}")
        print(f"    V100 spot cost : ${cost_v100:.0f}")
        print(f"    A100 spot cost : ${cost_a100:.0f}")
PYEOF

log ""
log "=================================================================="
log "  Lightning.ai validation complete."
log "  Review logs: smoke_baseline.log, smoke_kleene.log, ${LOG_FILE}"
log "=================================================================="
