"""
Device Utilities for TropFormer
================================
Auto-detects AMD GPU (DirectML) and provides a unified device interface.

DirectML compatibility notes:
  - Classical matmul: works, ~2x faster than CPU at scale (d>=256, batch>=64)
  - Tropical max backward: FAILS with native scatter. Fixed by:
    1. Using scores.detach().max() in TropicalLinear (no scatter in graph)
    2. Enabling _USE_SMOOTH_MAX for _tropical_max (logsumexp STE backward)
  - AdamW: minor CPU fallback for aten::lerp (negligible overhead)
"""

import torch

_DEVICE = None
_IS_DML = False


def get_device() -> torch.device:
    """Return the best available device. Caches result."""
    global _DEVICE, _IS_DML
    if _DEVICE is not None:
        return _DEVICE

    # Try DirectML (AMD GPU)
    try:
        import torch_directml
        _DEVICE = torch_directml.device()
        _IS_DML = True
        _enable_dml_compat()
        print(f"[device] Using AMD GPU via DirectML: {_DEVICE}")
        return _DEVICE
    except ImportError:
        pass

    # Try CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
        print(f"[device] Using CUDA: {torch.cuda.get_device_name(0)}")
        return _DEVICE

    _DEVICE = torch.device("cpu")
    print(f"[device] Using CPU")
    return _DEVICE


def is_directml() -> bool:
    """Check if we're using DirectML."""
    if _DEVICE is None:
        get_device()
    return _IS_DML


def _enable_dml_compat():
    """Enable DirectML-compatible modes in tropical modules."""
    import tropformer
    tropformer._USE_SMOOTH_MAX = True
    print("[device] Enabled smooth max mode for DirectML compatibility")
