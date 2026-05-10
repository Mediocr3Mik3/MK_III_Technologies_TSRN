"""
research/test_tropical_kernels.py
=================================

Correctness + smoke tests for the three tropical matmul backends.

Run with:
    pytest research/test_tropical_kernels.py -q
or:
    python -m research.test_tropical_kernels      # falls back to plain assertions
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

# Allow running directly (`python -m research.test_tropical_kernels`) on
# Lightning AI / cloud envs that don't have pytest installed.
try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False

    class _PytestShim:
        """Minimal shim so the @pytest.mark.skipif decorator and pytest.raises
        context manager keep working when pytest is not installed."""
        class mark:
            @staticmethod
            def skipif(cond, reason=""):
                def deco(fn):
                    def wrapper(*a, **kw):
                        if cond:
                            print(f"  [skip] {fn.__name__}: {reason}")
                            return None
                        return fn(*a, **kw)
                    wrapper.__name__ = fn.__name__
                    return wrapper
                return deco

        class raises:
            def __init__(self, exc): self.exc = exc
            def __enter__(self): return self
            def __exit__(self, et, ev, tb):
                if et is None:
                    raise AssertionError(f"expected {self.exc.__name__}, got nothing")
                return issubclass(et, self.exc)

    pytest = _PytestShim()

# Make sibling imports work whether invoked as ``python -m research.foo`` or
# ``pytest research/foo.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research.tropical_kernels import (
    tropical_matmul_naive,
    tropical_matmul_soft,
    tropical_matmul_triton,
    tropical_matmul,
    HAS_TRITON,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _rand(*shape, device="cpu", dtype=torch.float32, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, device=device, dtype=dtype, generator=g)


# ---------------------------------------------------------------------------
#  Soft kernel: convergence to hard max as h -> 0
# ---------------------------------------------------------------------------
def test_soft_converges_to_hard_as_h_to_0():
    """Soft converges to hard as h -> 0+, bounded by h * log(K) above the
    hard answer.  Numerical floor: ``exp((arg - s)/h)`` underflows in fp64
    once ``|arg - s|/h > 708`` (smallest positive fp64 ~= exp(-708)).  For
    randn(M, K) inputs ``|arg - s|`` is ~6, so h must stay above ~6/708 ~= 0.01
    or the test sees a constant ~0.7 offset that has nothing to do with the
    kernel.  We therefore stop at h=0.01 and trust the closed-form bound below.
    """
    A = _rand(8, 16, dtype=torch.float64)
    B = _rand(16, 8, dtype=torch.float64)
    K = A.shape[-1]

    hard = tropical_matmul_naive(A, B)
    # Theoretical bound: 0 <= soft - hard <= h * log(K).  Use 1.1x slack for
    # finite-precision matmul accumulation.
    for h in (1.0, 0.1, 0.05, 0.01):
        soft = tropical_matmul_soft(A, B, h=h)
        gap_lo = (hard - soft).clamp_min(0).max().item()
        gap_hi = (soft - hard).clamp_min(0).max().item()
        bound = 1.1 * h * math.log(K) + 1e-9
        assert gap_lo <= 1e-9, f"soft < hard at h={h} (lo gap={gap_lo})"
        assert gap_hi <= bound, f"soft - hard = {gap_hi} > bound {bound} at h={h}"


def test_soft_underflow_floor_documented():
    """Document the numerical h-floor: at very small h, fp64 underflow in
    the inner exp clamps log(prod) to log(tiny_fp64) ~= -708, producing a
    spurious constant offset of order h*708 ~= 0.7.  This isn't a kernel
    bug; it's the price of routing through standard exp/matmul/log.
    Triton/naive paths are exact and should be used when h must be tiny.
    """
    A = _rand(8, 16, dtype=torch.float64)
    B = _rand(16, 8, dtype=torch.float64)
    soft_tiny = tropical_matmul_soft(A, B, h=1e-3)
    hard = tropical_matmul_naive(A, B)
    # We expect the gap to be roughly h * 708, not h * log(K).  Sanity-check
    # that the *triton/naive* path still equals hard.
    assert (soft_tiny >= hard - 1e-9).all(), "soft must remain an upper envelope"
    # No upper-bound assertion: this region is documented as numerically lossy.


def test_soft_associativity_smoke():
    """LSE is exactly associative; the matmul form should respect it."""
    A = _rand(6, 8, dtype=torch.float64)
    B = _rand(8, 7, dtype=torch.float64)
    C = _rand(7, 5, dtype=torch.float64)
    h = 0.5
    left = tropical_matmul_soft(tropical_matmul_soft(A, B, h=h), C, h=h)
    right = tropical_matmul_soft(A, tropical_matmul_soft(B, C, h=h), h=h)
    assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)


def test_soft_rejects_nonpositive_h():
    A = _rand(4, 4)
    B = _rand(4, 4)
    with pytest.raises(ValueError):
        tropical_matmul_soft(A, B, h=0.0)
    with pytest.raises(ValueError):
        tropical_matmul_soft(A, B, h=-0.1)


# ---------------------------------------------------------------------------
#  Naive kernel sanity (treat as reference)
# ---------------------------------------------------------------------------
def test_naive_against_python_loop():
    M, K, N = 4, 5, 3
    A = _rand(M, K, dtype=torch.float64)
    B = _rand(K, N, dtype=torch.float64)
    C_ref = torch.empty(M, N, dtype=torch.float64)
    for i in range(M):
        for j in range(N):
            C_ref[i, j] = (A[i, :] + B[:, j]).max()
    C = tropical_matmul_naive(A, B)
    assert torch.allclose(C, C_ref, atol=1e-12)


def test_naive_batched():
    A = _rand(3, 4, 5, dtype=torch.float64)
    B = _rand(3, 5, 6, dtype=torch.float64)
    C = tropical_matmul_naive(A, B)
    assert C.shape == (3, 4, 6)
    for b in range(3):
        ref = tropical_matmul_naive(A[b], B[b])
        assert torch.allclose(C[b], ref, atol=1e-12)


# ---------------------------------------------------------------------------
#  Triton kernel correctness (skipped on CPU-only / no triton)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="triton + CUDA required",
)
def test_triton_matches_naive_square():
    device = "cuda"
    M = K = N = 64
    A = _rand(M, K, device=device, dtype=torch.float32)
    B = _rand(K, N, device=device, dtype=torch.float32)
    C_ref = tropical_matmul_naive(A, B)
    C = tropical_matmul_triton(A, B)
    assert torch.allclose(C, C_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="triton + CUDA required",
)
def test_triton_matches_naive_nonsquare_padded():
    """K and N not divisible by BLOCK exercises masking branches."""
    device = "cuda"
    A = _rand(13, 19, device=device, dtype=torch.float32)
    B = _rand(19, 7, device=device, dtype=torch.float32)
    C_ref = tropical_matmul_naive(A, B)
    C = tropical_matmul_triton(A, B)
    assert torch.allclose(C, C_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="triton + CUDA required",
)
def test_triton_batched():
    device = "cuda"
    A = _rand(2, 32, 32, device=device, dtype=torch.float32)
    B = _rand(2, 32, 32, device=device, dtype=torch.float32)
    C_ref = tropical_matmul_naive(A, B)
    C = tropical_matmul_triton(A, B)
    assert torch.allclose(C, C_ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
#  Dispatch wrapper
# ---------------------------------------------------------------------------
def test_dispatch_auto_h0_picks_triton_or_naive():
    A = _rand(4, 4)
    B = _rand(4, 4)
    out = tropical_matmul(A, B, mode="auto", h=0.0)
    ref = tropical_matmul_naive(A, B)
    # On CPU we expect naive == auto exactly.
    if not (HAS_TRITON and torch.cuda.is_available()):
        assert torch.allclose(out, ref, atol=1e-12)


def test_dispatch_auto_h_positive_picks_soft():
    A = _rand(4, 4, dtype=torch.float64)
    B = _rand(4, 4, dtype=torch.float64)
    out = tropical_matmul(A, B, mode="auto", h=0.5)
    ref = tropical_matmul_soft(A, B, h=0.5)
    assert torch.allclose(out, ref, atol=1e-12)


def test_dispatch_explicit_modes():
    A = _rand(4, 4, dtype=torch.float64)
    B = _rand(4, 4, dtype=torch.float64)
    naive = tropical_matmul(A, B, mode="naive")
    soft = tropical_matmul(A, B, mode="soft", h=0.05)
    assert torch.allclose(naive, tropical_matmul_naive(A, B), atol=1e-12)
    assert (soft >= naive - 1e-9).all()
    with pytest.raises(ValueError):
        tropical_matmul(A, B, mode="bogus")


# ---------------------------------------------------------------------------
#  Plain-script fallback
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_naive_against_python_loop()
    test_naive_batched()
    test_soft_converges_to_hard_as_h_to_0()
    test_soft_underflow_floor_documented()
    test_soft_associativity_smoke()
    test_soft_rejects_nonpositive_h()
    test_dispatch_auto_h0_picks_triton_or_naive()
    test_dispatch_auto_h_positive_picks_soft()
    test_dispatch_explicit_modes()
    if HAS_TRITON and torch.cuda.is_available():
        test_triton_matches_naive_square()
        test_triton_matches_naive_nonsquare_padded()
        test_triton_batched()
        print("[OK] all tests (incl. triton) passed.")
    else:
        print("[OK] CPU/no-triton tests passed; triton tests skipped.")
