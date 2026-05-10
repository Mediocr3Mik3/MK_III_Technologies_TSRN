"""
research/tropical_kernels.py
============================

Tropical (max-plus) matrix multiplication kernels.

Three implementations sharing the signature ``C = f(A, B)`` where
    C_ij = max_k (A_ik + B_kj)        (tropical / max-plus semiring)

  1. ``tropical_matmul_naive``   -- broadcasting + max-reduce.  Reference.
                                   Materialises a (..., M, K, N) tensor.
  2. ``tropical_matmul_soft``    -- Maslov-h log-sum-exp form, factored as
                                   ``h * log(exp(A/h) @ exp(B/h))`` so the
                                   inner '@' lands on Tensor Cores.
                                   As ``h -> 0+`` recovers the hard max.
  3. ``tropical_matmul_triton``  -- Custom Triton kernel that performs the
                                   tile-parallel max-plus reduction without
                                   materialising the (M, K, N) intermediate.
                                   Cannot use Tensor Cores (no max in mma)
                                   but saturates the L1/SM compute path.

Why this exists
---------------
Standard transformer matmul rides Tensor Cores (242 TFLOPS BF16 on L4).
Pure max-plus reductions are stuck on regular CUDA cores (~30 TFLOPS).
The soft path recovers the Tensor-Core path during training (Maslov h>0)
while preserving the tropical limit at inference (h -> 0).

References
----------
* Kolokoltsov & Maslov, *Idempotent Analysis and its Applications* (1997).
* Litvinov, *The Maslov dequantization, idempotent and tropical
  mathematics: a brief introduction*, J. Math. Sci. 2007.
* Pachter & Sturmfels, *Tropical Geometry of Statistical Models*, PNAS 2004.
"""
from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "tropical_matmul_naive",
    "tropical_matmul_soft",
    "tropical_matmul_triton",
    "tropical_matmul",
    "HAS_TRITON",
]


# ---------------------------------------------------------------------------
#  Reference: broadcasting + max-reduce.  Exact, slow, no Tensor Cores.
# ---------------------------------------------------------------------------
def tropical_matmul_naive(A: Tensor, B: Tensor) -> Tensor:
    """``C_ij = max_k (A_ik + B_kj)``.

    Materialises a ``(..., M, K, N)`` intermediate so it scales as O(M*K*N)
    memory.  Use only as ground truth in tests.
    """
    return (A.unsqueeze(-1) + B.unsqueeze(-3)).max(dim=-2).values


# ---------------------------------------------------------------------------
#  Soft tropical matmul: log-sum-exp via Tensor-Core GEMM.
# ---------------------------------------------------------------------------
def tropical_matmul_soft(A: Tensor, B: Tensor, h: float = 1.0) -> Tensor:
    """Soft tropical matmul: ``C_ij = h * logsumexp((A_ik + B_kj)/h, axis=k)``.

    The contraction is rewritten as

        logsumexp_k(A_ik + B_kj)
            = log(sum_k exp(A_ik) * exp(B_kj))
            = log( (exp(A) @ exp(B))_ij )

    so the heavy lifting is a *standard* matmul that uses Tensor Cores under
    bf16/fp16 autocast.  Numerical stability is achieved by row/col-max
    shifting before the exponential, exactly as in stable softmax.

    Limit behaviour:
        h -> 0+ : converges to ``max_k (A_ik + B_kj)`` (hard tropical).
        h -> inf: smooth, dominated by the average entry.

    Args:
        A: ``(..., M, K)`` real tensor.
        B: ``(..., K, N)`` real tensor.
        h: Maslov temperature; must be > 0 (use ``tropical_matmul_naive`` or
           ``tropical_matmul_triton`` for h == 0).
    Returns:
        ``(..., M, N)`` tensor.
    """
    if h <= 0:
        raise ValueError(
            f"h must be > 0 (got {h}); use tropical_matmul_triton/naive for h=0.")

    # Stability: shift along the contraction axis only.
    A_max = A.amax(dim=-1, keepdim=True)              # (..., M, 1)
    B_max = B.amax(dim=-2, keepdim=True)              # (..., 1, N)

    # exp((A - A_max) / h) and exp((B - B_max) / h) are in [0, 1].
    expA = ((A - A_max) / h).exp()                    # (..., M, K)
    expB = ((B - B_max) / h).exp()                    # (..., K, N)

    prod = expA @ expB                                # (..., M, N)  Tensor Core
    # Underflow can give exact 0 in lower precision; clamp to avoid -inf.
    prod = prod.clamp_min(torch.finfo(prod.dtype).tiny)

    return h * prod.log() + A_max + B_max


# ---------------------------------------------------------------------------
#  Hard tropical matmul via Triton.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:                                     # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _tropical_gemm_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One program -> one (BLOCK_M, BLOCK_N) tile of C.

        Tropical accumulator is initialised to -inf.  At each K-tile we
        materialise the (BLOCK_M, BLOCK_K, BLOCK_N) sum, max-reduce along K,
        and fold into the running tropical accumulator.  This avoids the
        full (M, K, N) materialisation that the eager broadcasting path
        suffers from.

        Block sizes default to 16x16x16 -> the (BM, BK, BN) inner tensor is
        16^3 = 4096 fp32 = 16 KiB, fits comfortably in registers/L1.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        NEG_INF = -1.0e30
        acc = tl.full((BLOCK_M, BLOCK_N), NEG_INF, dtype=tl.float32)

        n_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(0, n_tiles):
            k_base = k_tile * BLOCK_K
            mask_k = (k_base + offs_k) < K
            mask_a = (offs_m[:, None] < M) & mask_k[None, :]
            mask_b = mask_k[:, None] & (offs_n[None, :] < N)

            a_tile = tl.load(a_ptrs, mask=mask_a, other=NEG_INF).to(tl.float32)
            b_tile = tl.load(b_ptrs, mask=mask_b, other=NEG_INF).to(tl.float32)

            # Broadcast add then max-reduce over the K-tile.
            sum3 = a_tile[:, :, None] + b_tile[None, :, :]    # (BM, BK, BN)
            tile_max = tl.max(sum3, axis=1)                   # (BM, BN)
            acc = tl.maximum(acc, tile_max)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask_c)


def tropical_matmul_triton(
    A: Tensor,
    B: Tensor,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 16,
) -> Tensor:
    """Hard ``max_k (A_ik + B_kj)`` via Triton kernel.

    Falls back to ``tropical_matmul_naive`` if Triton is unavailable or the
    inputs are on CPU.  Supports arbitrary leading batch dims (broadcast or
    matched).  Inner block sizes are exposed for benchmarking; the default
    16/16/16 is sound for d_state up to ~256 on Ada/Ampere.
    """
    if (not HAS_TRITON) or (not A.is_cuda) or (not B.is_cuda):
        return tropical_matmul_naive(A, B)

    # Flatten leading batch dims; keep the last two as (M, K) / (K, N).
    *batch_a, M, K = A.shape
    *batch_b, K2, N = B.shape
    if K != K2:
        raise ValueError(f"inner dims mismatch: A {tuple(A.shape)}, B {tuple(B.shape)}")

    # Broadcast batches to a common shape (cheap, no copy when sizes match).
    A_b = A.expand(*torch.broadcast_shapes(tuple(batch_a), tuple(batch_b)), M, K).contiguous()
    B_b = B.expand(*torch.broadcast_shapes(tuple(batch_a), tuple(batch_b)), K, N).contiguous()
    Bsz = max(1, A_b.numel() // (M * K))
    A_flat = A_b.reshape(Bsz, M, K)
    B_flat = B_b.reshape(Bsz, K, N)

    out_dtype = A.dtype
    C = torch.empty((Bsz, M, N), device=A.device, dtype=out_dtype)

    # Triton kernel runs in fp32 internally (load.to(fp32)) and we store back
    # in the input dtype — a single launch per batch element keeps the kernel
    # simple; for typical Bsz=1 this is exactly one launch.
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    for b in range(Bsz):
        _tropical_gemm_kernel[grid](
            A_flat[b], B_flat[b], C[b],
            M, N, K,
            A_flat[b].stride(0), A_flat[b].stride(1),
            B_flat[b].stride(0), B_flat[b].stride(1),
            C[b].stride(0), C[b].stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )

    return C.reshape(*A_b.shape[:-2], M, N)


# ---------------------------------------------------------------------------
#  Dispatch wrapper.
# ---------------------------------------------------------------------------
def tropical_matmul(
    A: Tensor,
    B: Tensor,
    *,
    mode: str = "auto",
    h: float = 0.0,
) -> Tensor:
    """Tropical (max-plus) matrix multiply with selectable backend.

    Args:
        A: ``(..., M, K)``.
        B: ``(..., K, N)``.
        mode: one of
            - ``"auto"``   : ``"soft"`` if ``h > 0`` else ``"triton"``
                            (or ``"naive"`` if Triton missing / CPU).
            - ``"soft"``   : Maslov-h Tensor-Core path (uses ``h``).
            - ``"triton"`` : Hard max-plus Triton kernel.
            - ``"naive"``  : Reference broadcasting kernel.
        h: Maslov temperature for the soft path; must be > 0 when used.
    """
    if mode == "auto":
        if h > 0:
            mode = "soft"
        elif HAS_TRITON and A.is_cuda and B.is_cuda:
            mode = "triton"
        else:
            mode = "naive"

    if mode == "soft":
        return tropical_matmul_soft(A, B, h=h if h > 0 else 1.0)
    if mode == "triton":
        return tropical_matmul_triton(A, B)
    if mode == "naive":
        return tropical_matmul_naive(A, B)
    raise ValueError(f"unknown mode: {mode!r}")
