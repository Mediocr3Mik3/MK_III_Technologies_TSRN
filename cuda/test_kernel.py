"""
Test script for the max-plus GEMM CUDA kernel.
Tests the PyTorch fallback path (which works on CPU/DirectML) and
validates correctness against the reference TropicalLinear implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def test_fallback_correctness():
    """Test that CUDATropicalLinear fallback matches TropicalLinear."""
    from tropformer import TropicalLinear
    from cuda.maxplus_binding import CUDATropicalLinear

    torch.manual_seed(42)
    in_f, out_f = 64, 32
    batch = 8

    # Create both layers with same weights
    ref = TropicalLinear(in_f, out_f, bias=True)
    cuda_layer = CUDATropicalLinear(in_f, out_f, bias=True, ste_temp=1.0)

    # Copy weights
    cuda_layer.weight.data.copy_(ref.weight.data)
    cuda_layer.bias.data.copy_(ref.bias.data)

    x = torch.randn(batch, in_f)

    # Eval mode (hard max, no STE)
    ref.eval()
    cuda_layer.eval()
    y_ref = ref(x)
    y_cuda = cuda_layer(x)

    diff_eval = (y_ref - y_cuda).abs().max().item()
    print(f"Eval mode max diff: {diff_eval:.2e}")
    assert diff_eval < 1e-5, f"Eval mode mismatch: {diff_eval}"

    # Train mode (STE backward)
    ref.train()
    cuda_layer.train()
    y_ref = ref(x)
    y_cuda = cuda_layer(x)

    diff_train = (y_ref - y_cuda).abs().max().item()
    print(f"Train mode max diff: {diff_train:.2e}")
    assert diff_train < 1e-5, f"Train mode mismatch: {diff_train}"

    # Backward pass
    y_ref.sum().backward()
    y_cuda.sum().backward()

    grad_w_diff = (ref.weight.grad - cuda_layer.weight.grad).abs().max().item()
    grad_b_diff = (ref.bias.grad - cuda_layer.bias.grad).abs().max().item()
    print(f"Weight grad max diff: {grad_w_diff:.2e}")
    print(f"Bias grad max diff: {grad_b_diff:.2e}")
    assert grad_w_diff < 1e-4, f"Weight grad mismatch: {grad_w_diff}"
    assert grad_b_diff < 1e-4, f"Bias grad mismatch: {grad_b_diff}"

    print("PASSED: Fallback matches reference TropicalLinear")


def test_leading_dims():
    """Test that CUDATropicalLinear handles arbitrary leading dimensions."""
    from cuda.maxplus_binding import CUDATropicalLinear

    torch.manual_seed(42)
    layer = CUDATropicalLinear(32, 16, ste_temp=1.0)

    # 2D input
    x2 = torch.randn(4, 32)
    y2 = layer(x2)
    assert y2.shape == (4, 16), f"2D shape mismatch: {y2.shape}"

    # 3D input (batch, seq, features)
    x3 = torch.randn(4, 10, 32)
    y3 = layer(x3)
    assert y3.shape == (4, 10, 16), f"3D shape mismatch: {y3.shape}"

    # 4D input
    x4 = torch.randn(2, 3, 5, 32)
    y4 = layer(x4)
    assert y4.shape == (2, 3, 5, 16), f"4D shape mismatch: {y4.shape}"

    print("PASSED: Leading dimension handling")


def test_gradient_flow():
    """Test that gradients flow through CUDATropicalLinear."""
    from cuda.maxplus_binding import CUDATropicalLinear

    torch.manual_seed(42)
    layer = CUDATropicalLinear(16, 8, ste_temp=1.0)
    layer.train()

    x = torch.randn(4, 16, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "No gradient for input"
    assert layer.weight.grad is not None, "No gradient for weight"
    assert layer.bias.grad is not None, "No gradient for bias"

    # Check that gradients are non-trivial
    assert x.grad.abs().sum() > 0, "Zero input gradient"
    assert layer.weight.grad.abs().sum() > 0, "Zero weight gradient"

    print("PASSED: Gradient flow")


def test_maxplus_matmul_reference():
    """Test tropical matrix multiply (reference implementation)."""
    # C_ij = max_k(A_ik + B_kj)
    A = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    B = torch.tensor([[0.0, 1.0], [2.0, 0.0]])

    # Manual computation:
    # C[0,0] = max(1+0, 2+2) = max(1, 4) = 4
    # C[0,1] = max(1+1, 2+0) = max(2, 2) = 2
    # C[1,0] = max(3+0, 0+2) = max(3, 2) = 3
    # C[1,1] = max(3+1, 0+0) = max(4, 0) = 4
    expected = torch.tensor([[4.0, 2.0], [3.0, 4.0]])

    # Compute using broadcast (reference)
    C = (A.unsqueeze(2) + B.unsqueeze(0)).max(dim=1).values
    diff = (C - expected).abs().max().item()
    print(f"Tropical matmul max diff: {diff:.2e}")
    assert diff < 1e-6, f"Tropical matmul mismatch"

    print("PASSED: Tropical matrix multiply reference")


if __name__ == "__main__":
    print("=" * 50)
    print("  Max-Plus GEMM Kernel Tests")
    print("=" * 50)

    test_maxplus_matmul_reference()
    test_fallback_correctness()
    test_leading_dims()
    test_gradient_flow()

    print("\n" + "=" * 50)
    print("  ALL TESTS PASSED")
    print("=" * 50)
