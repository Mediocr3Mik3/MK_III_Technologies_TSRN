"""
tsrn_cuda — CUDA-native helpers that drop into the TSRNGist stack.
====================================================================

The research-branch model code (`tsrn_dml.py`, `tsrn_gist.py`) was authored
for AMD GPUs via DirectML.  Several of those helpers carry workarounds that
are **correct but suboptimal** on NVIDIA CUDA:

  - ``AdamWDML``      — replaces ``aten::lerp`` (DML CPU fallback) with
                         per-step in-place ops.  On CUDA, the stock
                         ``torch.optim.AdamW(fused=True)`` is faster.
  - ``prefix_max``    — Hillis-Steele scan to dodge ``cummax`` CPU fallback.
                         On CUDA, ``torch.cummax`` is a native single kernel.
  - manual masked attn — DML can't take ``F.pad(value=-inf)`` cleanly.
                          On CUDA we use ``scaled_dot_product_attention`` with
                          ``is_causal=True`` (flash-attn 2 backend on Ampere+).

This module provides:

  - :func:`detect_cuda_device`          — pick CUDA, fall back gracefully
  - :func:`make_optimizer`              — fused AdamW with sane defaults
  - :func:`autocast_dtype_for_gpu`      — bf16 on Ampere/Hopper, fp16 on older
  - :func:`maybe_compile`               — torch.compile() when supported
  - :func:`prefix_max_cuda`             — wraps ``torch.cummax`` for the
                                          tropical SSM head
  - :func:`get_gpu_memory_gib`          — for batch-size auto-tuning

It does **not** redefine the model.  TSRNGist runs unchanged on CUDA; we
only swap optimizer / scan / attention paths via these helpers.
"""

from __future__ import annotations

import os
import math
import warnings
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  Device detection
# ---------------------------------------------------------------------------

def detect_cuda_device(verbose: bool = True) -> torch.device:
    """Return ``cuda:0`` if available, else fall back to DML, else CPU.

    On a multi-GPU node the local-rank assignment is left to the trainer
    (which sets ``CUDA_VISIBLE_DEVICES`` / calls ``torch.cuda.set_device``).
    """
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if verbose:
            print(f"  Device  : NVIDIA CUDA  ({n} GPU{'s' if n > 1 else ''})")
            print(f"  GPU 0   : {name}  ({mem:.1f} GiB, sm_{cap[0]}{cap[1]})")
            print(f"  Backend : torch {torch.__version__} + cuda {torch.version.cuda}")
        return torch.device("cuda:0")

    try:
        import torch_directml  # type: ignore
        if verbose:
            print("  Device  : DirectML (no CUDA detected)")
        return torch_directml.device()
    except ImportError:
        pass

    if verbose:
        print("  Device  : CPU (no GPU available)")
    return torch.device("cpu")


def get_gpu_memory_gib(device: Optional[torch.device] = None) -> float:
    """Total memory of the active CUDA device in GiB; 0.0 if not CUDA."""
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        return 0.0
    idx = device.index if device.index is not None else 0
    return torch.cuda.get_device_properties(idx).total_memory / (1024**3)


def device_sync(device: torch.device) -> None:
    """Synchronize whichever backend is active."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
#  Mixed-precision policy
# ---------------------------------------------------------------------------

def autocast_dtype_for_gpu(device: Optional[torch.device] = None) -> torch.dtype:
    """Return the right ``autocast`` dtype for the active GPU.

      - bf16 on Ampere (sm_80) and newer  (A100, H100, L40S, RTX 4090, RTX 3090...)
      - fp16 on older  (V100, T4, P100, RTX 2080, RTX 1080...)
      - fp32 on CPU / DML (autocast is a no-op there in our trainer)

    bf16 has the same dynamic range as fp32 so it never overflows; fp16 needs
    a ``GradScaler``.
    """
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def amp_components(device: torch.device) -> Tuple[torch.dtype, bool]:
    """Return ``(dtype, needs_grad_scaler)``."""
    dt = autocast_dtype_for_gpu(device)
    needs_scaler = (dt == torch.float16)
    return dt, needs_scaler


# ---------------------------------------------------------------------------
#  Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: Optional[bool] = None,
    use_8bit: bool = False,
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer with weight-decay applied only to 2D+ params.

    Param ordering uses ``named_parameters()`` (module-registration order),
    which is deterministic across runs — required for safe resume of the
    optimizer state.

    Parameters
    ----------
    fused
        ``None`` -> auto: True on CUDA, False elsewhere.
    use_8bit
        If True, use ``bitsandbytes.optim.AdamW8bit`` (saves ~75% optimizer
        memory; ~5% slower).  Only on CUDA, requires ``bitsandbytes``.
    """
    decay_params, no_decay_params = [], []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay_params if p.dim() >= 2 else no_decay_params).append(p)

    groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if use_8bit:
        try:
            import bitsandbytes as bnb  # type: ignore
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=eps)
        except ImportError:
            warnings.warn("bitsandbytes not installed; falling back to fp32 AdamW.")

    if fused is None:
        fused = torch.cuda.is_available()

    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)


# ---------------------------------------------------------------------------
#  Prefix-max — CUDA fast path
# ---------------------------------------------------------------------------

def prefix_max_cuda(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Drop-in for ``tsrn_dml.prefix_max`` that uses ``torch.cummax`` on CUDA
    (one kernel) and falls back to the Hillis-Steele scan elsewhere.
    """
    if x.is_cuda:
        return torch.cummax(x, dim=dim).values
    # Lazy import; assumes research/ is on sys.path (the cloud trainer
    # arranges this on startup).
    import tsrn_dml as _dml  # type: ignore
    return _dml._original_prefix_max(x, dim=dim) if getattr(_dml, "_cuda_fastpaths_installed", False) \
        else _dml.prefix_max(x, dim=dim)


def install_cuda_fastpaths() -> None:
    """Monkey-patch ``tsrn_dml.prefix_max`` to the CUDA fast path.

    Call exactly once, *before* the model is instantiated.  Idempotent.
    The TSRNGist model imports ``prefix_max`` from ``tsrn_dml`` at module
    load, so we patch the binding inside that module so all downstream
    callers (TropicalSSM in particular) benefit.

    Requires ``research/`` to already be on ``sys.path`` so the legacy
    top-level ``import tsrn_dml`` resolves to ``research/tsrn_dml.py``.
    """
    import tsrn_dml as _dml  # type: ignore
    if getattr(_dml, "_cuda_fastpaths_installed", False):
        return
    _dml._original_prefix_max = _dml.prefix_max  # type: ignore[attr-defined]
    _dml.prefix_max = prefix_max_cuda            # type: ignore[assignment]
    _dml._cuda_fastpaths_installed = True        # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
#  torch.compile wrapper (PyTorch 2.x speedup, optional)
# ---------------------------------------------------------------------------

def maybe_compile(
    model: nn.Module,
    mode: str = "reduce-overhead",
    enabled: bool = True,
) -> nn.Module:
    """Best-effort ``torch.compile``.  Returns the original model on failure
    or when disabled / unavailable.  ``mode`` choices:

      - ``"reduce-overhead"`` — best for training, low compile time
      - ``"max-autotune"``    — slow first iter, faster steady-state
      - ``"default"``         — balanced
    """
    if not enabled:
        return model
    if not torch.cuda.is_available():
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode=mode, dynamic=False)
    except Exception as e:
        warnings.warn(f"torch.compile failed ({type(e).__name__}): {e}; using eager model.")
        return model


# ---------------------------------------------------------------------------
#  Distributed helpers
# ---------------------------------------------------------------------------

def is_distributed() -> bool:
    return (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_global_rank() == 0


def init_distributed(backend: str = "nccl") -> Optional[torch.device]:
    """Initialise ``torch.distributed`` if launched under torchrun.

    Returns the per-rank CUDA device, or ``None`` if not distributed.
    """
    if not is_distributed():
        return None
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def cleanup_distributed() -> None:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
#  Misc
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (incl. CUDA)."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_batch_size(
    gpu_mem_gib: float,
    d_model: int,
    context_len: int,
    n_blocks: int,
    grad_ckpt: bool = True,
    safety_margin: float = 0.85,
) -> int:
    """Heuristic per-GPU micro-batch size for TSRNGist.

    Empirical scaling: peak activation ~= B * T * d * n_blocks * (4 if grad_ckpt else 12) bytes.
    Plus param + optimizer state (~12 bytes/param for AdamW fp32 master).
    """
    if gpu_mem_gib <= 0:
        return 1
    bytes_per_token = d_model * n_blocks * (4 if grad_ckpt else 12)
    usable = safety_margin * gpu_mem_gib * 1024**3
    # Reserve ~30% for optimizer + framework overhead
    activation_budget = 0.7 * usable
    micro = max(1, int(activation_budget // (context_len * bytes_per_token)))
    # Round to common batch sizes
    for cand in (64, 32, 16, 8, 4, 2, 1):
        if micro >= cand:
            return cand
    return 1
