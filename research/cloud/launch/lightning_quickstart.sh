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

echo "=================================================================="
echo "  TropFormer Lightning.ai CUDA Validation"
echo "=================================================================="

# ---------------------------------------------------------------------------
# 1. Clone / update repo
# ---------------------------------------------------------------------------
if [ ! -d "${REPO}" ]; then
    echo "==> Cloning TropFormer (${BRANCH} branch)..."
    git clone -b "${BRANCH}" \
        https://github.com/Mediocr3Mik3/MK_III_Technologies_TSRN.git \
        "${REPO}"
else
    echo "==> Updating existing repo..."
    cd "${REPO}" && git fetch origin && git checkout "${BRANCH}" && git pull
fi
cd "${REPO}"

# ---------------------------------------------------------------------------
# 2. Install / upgrade PyTorch for CUDA (Lightning.ai has CUDA 12.x)
# ---------------------------------------------------------------------------
echo
echo "==> Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.1" "torchvision==0.19.1"
pip install -q -r requirements.txt
pip install -q -r research/cloud/requirements_cuda.txt
echo "==> PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "==> CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "==> GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

# ---------------------------------------------------------------------------
# 3. Download enwik8 (validation dataset, 36MB, fast)
# ---------------------------------------------------------------------------
echo
echo "==> Downloading enwik8..."
mkdir -p data
if [ ! -f data/enwik8 ]; then
    wget -q -O data/enwik8.zip http://mattmahoney.net/dc/enwik8.zip
    (cd data && unzip -q -o enwik8.zip)
    echo "   enwik8 downloaded."
fi

# ---------------------------------------------------------------------------
# 4. Run the full test suite on GPU
# ---------------------------------------------------------------------------
echo
echo "=================================================================="
echo "  STEP 1/3 — Correctness tests (GPU)"
echo "=================================================================="
python research/test_kleene.py
echo
echo ">>> TEST SUITE COMPLETE"

# ---------------------------------------------------------------------------
# 5. 2000-step smoke train: baseline (TropicalSSM) vs Kleene
# ---------------------------------------------------------------------------
echo
echo "=================================================================="
echo "  STEP 2/3 — 2000-step baseline smoke run (TropicalSSM)"
echo "=================================================================="
mkdir -p checkpoints results logs

# Baseline: TropicalSSM (no Kleene flag)
python -m research.cloud.train_cloud \
    --preset small_24gb \
    --steps 2000 \
    --tag smoke_baseline \
    2>&1 | tee logs/smoke_baseline.log

echo
echo "=================================================================="
echo "  STEP 3/3 — 2000-step KleeneSSM smoke run"
echo "=================================================================="
# KleeneSSM run (tier=nano, Kleene SSM enabled)
python -m research.cloud.train_cloud \
    --preset small_24gb \
    --steps 2000 \
    --tag smoke_kleene \
    --use-kleene-ssm \
    2>&1 | tee logs/smoke_kleene.log

# ---------------------------------------------------------------------------
# 6. Extract and compare results
# ---------------------------------------------------------------------------
echo
echo "=================================================================="
echo "  RESULTS SUMMARY"
echo "=================================================================="

python3 - <<'PYEOF'
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

echo
echo "=================================================================="
echo "  Lightning.ai validation complete."
echo "  Review logs: smoke_baseline.log, smoke_kleene.log"
echo "=================================================================="
