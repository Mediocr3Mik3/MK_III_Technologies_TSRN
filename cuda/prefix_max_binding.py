"""
PyTorch binding for the prefix-max scan CUDA kernel.

This kernel is correct-by-construction for **forward only**.  Computing the
true gradient of prefix_max requires also tracking argmax positions, which
the kernel does not do today.  Therefore:

  * In ``torch.no_grad()`` (eval / inference):  use the CUDA kernel.
  * In training (autograd active):              fall back to the pure-PyTorch
    implementation, whose ``torch.maximum`` chain is differentiated correctly
    by autograd.

This split gives us the inference speedup we need for phone deployment
while keeping training mathematically sound.
"""

import os
import torch
import ctypes

_CUDA_AVAILABLE = False
_prefix_max_lib = None

try:
    kernel_path = os.path.join(os.path.dirname(__file__), "prefix_max_scan.so")
    if not os.path.exists(kernel_path):
        kernel_path = os.path.join(os.path.dirname(__file__), "prefix_max_scan.dll")
    if os.path.exists(kernel_path):
        _prefix_max_lib = ctypes.CDLL(kernel_path)
        _CUDA_AVAILABLE = True
except Exception:
    pass


def is_cuda_kernel_available() -> bool:
    return _CUDA_AVAILABLE and torch.cuda.is_available()


def _prefix_max_forward_only(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Run the CUDA forward kernel.  Caller must guarantee no autograd is needed."""
    if dim != 1:
        perm = list(range(x.ndim))
        perm[1], perm[dim] = perm[dim], perm[1]
        x_perm = x.permute(*perm).contiguous()
    else:
        x_perm = x.contiguous()

    B = x_perm.shape[0]
    T = x_perm.shape[1]
    D = x_perm.numel() // (B * T)
    out_perm = torch.empty_like(x_perm)

    _prefix_max_lib.prefix_max_scan_cuda(
        ctypes.c_void_p(x_perm.data_ptr()),
        ctypes.c_void_p(out_perm.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_int(T),
        ctypes.c_int(D),
    )

    if dim != 1:
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return out_perm.permute(*inv_perm).contiguous()
    return out_perm


def prefix_max_cuda(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Prefix-maximum along ``dim``.

      out[..., t, ...] = max(x[..., 0, ...], ..., x[..., t, ...])

    Uses the CUDA kernel only when (a) the kernel is loaded, (b) the tensor
    lives on CUDA in float32, and (c) gradient tracking is OFF.  In all other
    cases falls back to the PyTorch Hillis-Steele scan, whose autograd is
    correct because ``torch.maximum`` is differentiable.
    """
    use_cuda = (
        is_cuda_kernel_available()
        and x.is_cuda
        and x.dtype == torch.float32
        and not torch.is_grad_enabled()
        and not x.requires_grad
    )
    if use_cuda:
        return _prefix_max_forward_only(x, dim)

    # Fallback to pure-PyTorch (autograd-correct) Hillis-Steele scan
    from research.tsrn_dml import prefix_max
    return prefix_max(x, dim=dim)
