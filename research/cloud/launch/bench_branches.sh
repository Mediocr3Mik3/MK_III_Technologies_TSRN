#!/usr/bin/env bash
# ============================================================================
# bench_branches.sh — A/B eager-mode benchmark: kleene-star vs nvidia-cloud
# ============================================================================
# Times one fwd+bwd micro-step of TSRNGist on the same L4 GPU, on both
# branches, using the same self-contained `research.bench_eager` script.
# Uses git worktrees so the current checkout is undisturbed.
#
# Usage (from repo root):
#   bash research/cloud/launch/bench_branches.sh
#
# Output: ms/step for each branch, side by side.
# ============================================================================

set -e
cd "$(git rev-parse --show-toplevel)"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo ""
echo "=================================================================="
echo "  Eager fwd+bwd A/B benchmark"
echo "=================================================================="
echo "  current : ${CURRENT_BRANCH}"
echo "  comparing against: nvidia-cloud"
echo ""

# ----------------------------------------------------------------------------
# 1.  Run on the current branch (already checked out).
# ----------------------------------------------------------------------------
echo "------------------------------------------------------------------"
echo "  [A] ${CURRENT_BRANCH} (HEAD)"
echo "------------------------------------------------------------------"
python -m research.bench_eager --steps 10 --warmup 3 \
    --batch 8 --ctx 512 --d-model 512 --n-blocks 3 --n-heads 8 \
    --label "${CURRENT_BRANCH}"

# ----------------------------------------------------------------------------
# 2.  Spin up a worktree at /tmp/tropformer_nvidia_cloud, copy bench_eager.py
#     in (it doesn't exist on nvidia-cloud), and run.
# ----------------------------------------------------------------------------
WT_DIR="${TMPDIR:-/tmp}/tropformer_nvidia_cloud_$$"
echo ""
echo "------------------------------------------------------------------"
echo "  [B] nvidia-cloud (worktree at ${WT_DIR})"
echo "------------------------------------------------------------------"

# Make sure we have the latest nvidia-cloud ref locally.
git fetch origin nvidia-cloud:nvidia-cloud 2>/dev/null || true
git worktree add --detach "${WT_DIR}" nvidia-cloud
cleanup() { git worktree remove --force "${WT_DIR}" 2>/dev/null || rm -rf "${WT_DIR}"; }
trap cleanup EXIT

# bench_eager.py only exists on the current branch — copy it in.
cp research/bench_eager.py "${WT_DIR}/research/bench_eager.py"

# Run from the worktree using its tsrn_dml / tsrn_gist.
(
    cd "${WT_DIR}"
    python -m research.bench_eager --steps 10 --warmup 3 \
        --batch 8 --ctx 512 --d-model 512 --n-blocks 3 --n-heads 8 \
        --label "nvidia-cloud"
)

echo ""
echo "=================================================================="
echo "  done"
echo "=================================================================="
