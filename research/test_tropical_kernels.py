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

import pytest
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
    A = _rand(8, 16, dtype=torch.float64)
    B = _rand(16, 8, dtype=torch.float64)

    hard = tropical_matmul_naive(A, B)
    for h, tol in [(1.0, 2.0), (0.1, 0.2), (0.01, 0.03), (0.001, 0.005)]:
        soft = tropical_matmul_soft(A, B, h=h)
        # Soft is an upper envelope of hard (LSE >= max), and the gap is
        # bounded by h * log(K).  Verify both sides.
        assert (soft >= hard - 1e-9).all(), f"soft < hard at h={h}"
        gap = (soft - hard).abs().max().item()
        assert gap <= tol, f"gap={gap} too big at h={h} (tol={tol})"


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
    test_soft_associativity_smoke()
    test_dispatch_auto_h_positive_picks_soft()
    test_dispatch_explicit_modes()
    if HAS_TRITON and torch.cuda.is_available():
        test_triton_matches_naive_square()
        test_triton_matches_naive_nonsquare_padded()
        test_triton_batched()
        print("[OK] all tests (incl. triton) passed.")
    else:
        print("[OK] CPU/no-triton tests passed; triton tests skipped.")
