"""
PyTorch binding for the max-plus GEMM CUDA kernel.

This module provides a drop-in replacement for TropicalLinear that uses
the custom CUDA kernel for forward and backward passes.

Usage:
    from cuda.maxplus_binding import CUDATropicalLinear
    layer = CUDATropicalLinear(128, 256)

Falls back to the pure-PyTorch TropicalLinear if CUDA is not available
or if the kernel is not compiled.
"""

import os
import torch
import torch.nn as nn
from torch.autograd import Function

# Try to load the compiled CUDA kernel
_CUDA_AVAILABLE = False
_maxplus_lib = None

try:
    import ctypes
    kernel_path = os.path.join(os.path.dirname(__file__), "maxplus_gemm.so")
    if not os.path.exists(kernel_path):
        # Try Windows DLL
        kernel_path = os.path.join(os.path.dirname(__file__), "maxplus_gemm.dll")
    if os.path.exists(kernel_path):
        _maxplus_lib = ctypes.CDLL(kernel_path)
        _CUDA_AVAILABLE = True
except Exception:
    pass


def is_cuda_kernel_available() -> bool:
    """Check if the compiled CUDA kernel is available."""
    return _CUDA_AVAILABLE and torch.cuda.is_available()


class MaxPlusForwardBackward(Function):
    """Custom autograd function using the CUDA max-plus kernel."""

    @staticmethod
    def forward(ctx, x, weight, bias, ste_temp):
        """
        Forward: y_i = max_j(W_ij + x_j) + b_i
        Uses the CUDA kernel for computation.
        """
        batch_size = x.shape[0]
        in_features = x.shape[1]
        out_features = weight.shape[0]

        y = torch.empty(batch_size, out_features, device=x.device, dtype=x.dtype)

        # Call CUDA kernel
        _maxplus_lib.maxplus_forward_cuda(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(bias.data_ptr()) if bias is not None else None,
            ctypes.c_int(batch_size),
            ctypes.c_int(in_features),
            ctypes.c_int(out_features),
            ctypes.c_bool(in_features > 128),
        )

        ctx.save_for_backward(x, weight, bias)
        ctx.ste_temp = ste_temp
        ctx.in_features = in_features
        ctx.out_features = out_features
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, bias = ctx.saved_tensors
        batch_size = x.shape[0]

        grad_W = torch.zeros_like(weight)
        grad_x = torch.zeros_like(x)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        _maxplus_lib.maxplus_backward_cuda(
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(grad_y.contiguous().data_ptr()),
            ctypes.c_void_p(grad_W.data_ptr()),
            ctypes.c_void_p(grad_x.data_ptr()),
            ctypes.c_void_p(grad_bias.data_ptr()) if grad_bias is not None else None,
            ctypes.c_float(ctx.ste_temp),
            ctypes.c_int(batch_size),
            ctypes.c_int(ctx.in_features),
            ctypes.c_int(ctx.out_features),
        )

        return grad_x, grad_W, grad_bias, None


class CUDATropicalLinear(nn.Module):
    """
    Max-plus linear layer using the custom CUDA kernel.
    Falls back to pure-PyTorch implementation if CUDA kernel unavailable.

        y_i = max_j(W_ij + x_j) + b_i

    Drop-in replacement for tropformer.TropicalLinear.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 ste_temp: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ste_temp = ste_temp
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.uniform_(self.weight, -0.5, 0.5)

        self._use_cuda = is_cuda_kernel_available()
        if not self._use_cuda:
            import warnings
            warnings.warn(
                "CUDA max-plus kernel not available. "
                "Falling back to pure-PyTorch TropicalLinear. "
                "Compile with: nvcc -shared -o cuda/maxplus_gemm.so cuda/maxplus_gemm.cu"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        if self._use_cuda and x_flat.is_cuda:
            out = MaxPlusForwardBackward.apply(
                x_flat, self.weight, self.bias, self.ste_temp
            )
        else:
            # Fallback: pure PyTorch (same as TropicalLinear)
            scores = self.weight.unsqueeze(0) + x_flat.unsqueeze(1)
            if self.training:
                hard = scores.detach().max(dim=-1).values
                soft_w = torch.nn.functional.softmax(scores / self.ste_temp, dim=-1)
                soft = (soft_w * scores).sum(dim=-1)
                out = hard + (soft - soft.detach())
            else:
                out = scores.max(dim=-1).values
            if self.bias is not None:
                out = out + self.bias

        return out.reshape(*leading, self.out_features)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, cuda={self._use_cuda}"
