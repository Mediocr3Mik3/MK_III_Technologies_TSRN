#!/usr/bin/env bash
# ============================================================================
# bench_branches.sh — N-way eager-mode benchmark across branches
# ============================================================================
# Times one fwd+bwd micro-step of TSRNGist on the same GPU, across the
# current branch and every branch listed in $COMPARE_BRANCHES (default:
# "main nvidia-cloud").  Uses git worktrees so the current checkout is
# undisturbed.  bench_eager.py from the current branch is copied into
# each worktree because older branches don't have it.
#
# Usage (from repo root):
#   bash research/cloud/launch/bench_branches.sh
#   COMPARE_BRANCHES="main nvidia-cloud" bash research/cloud/launch/bench_branches.sh
#
# Output: ms/step for each branch.
# ============================================================================

set -e
cd "$(git rev-parse --show-toplevel)"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMPARE_BRANCHES="${COMPARE_BRANCHES:-main nvidia-cloud}"

BENCH_ARGS=(--steps 10 --warmup 3 --batch 8 --ctx 512
            --d-model 512 --n-blocks 3 --n-heads 8)

echo ""
echo "=================================================================="
echo "  Eager fwd+bwd N-way benchmark"
echo "=================================================================="
echo "  current  : ${CURRENT_BRANCH}"
echo "  vs       : ${COMPARE_BRANCHES}"
echo ""

# ----------------------------------------------------------------------------
# 1. Current branch (already checked out).
# ----------------------------------------------------------------------------
echo "------------------------------------------------------------------"
echo "  [HEAD] ${CURRENT_BRANCH}"
echo "------------------------------------------------------------------"
python -m research.bench_eager "${BENCH_ARGS[@]}" --label "${CURRENT_BRANCH}"

# ----------------------------------------------------------------------------
# 2. Each comparison branch via its own worktree.
# ----------------------------------------------------------------------------
WORKTREES=()
cleanup() {
    for wt in "${WORKTREES[@]}"; do
        git worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
    done
}
trap cleanup EXIT

for branch in ${COMPARE_BRANCHES}; do
    # Skip if the branch is the same as current.
    if [ "${branch}" = "${CURRENT_BRANCH}" ]; then
        echo "  (skipping ${branch}: same as current)"
        continue
    fi

    WT_DIR="${TMPDIR:-/tmp}/tropformer_${branch//\//_}_$$"
    WORKTREES+=("${WT_DIR}")

    echo ""
    echo "------------------------------------------------------------------"
    echo "  [${branch}] (worktree ${WT_DIR})"
    echo "------------------------------------------------------------------"

    # Fetch latest ref (non-fatal if offline).
    git fetch origin "${branch}:${branch}" 2>/dev/null || true
    if ! git worktree add --detach "${WT_DIR}" "${branch}" 2>/dev/null; then
        # Branch may only exist on origin.
        git worktree add --detach "${WT_DIR}" "origin/${branch}"
    fi

    # Copy current bench_eager.py + research/__init__.py if missing.
    cp research/bench_eager.py "${WT_DIR}/research/bench_eager.py"
    [ -f "${WT_DIR}/research/__init__.py" ] || \
        touch "${WT_DIR}/research/__init__.py"

    (
        cd "${WT_DIR}"
        python -m research.bench_eager "${BENCH_ARGS[@]}" --label "${branch}"
    )
done

echo ""
echo "=================================================================="
echo "  done"
echo "=================================================================="
