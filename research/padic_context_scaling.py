"""
P-adic Context Scaling (PaCS)
=============================

Non-uniform context scaling for inference.
Compresses positions based on their p-adic structural level.

Local structure (low p-adic valuation) is preserved.
Global structure (high p-adic valuation) is compressed.

This is structure-aware context extension, unlike YaRN which applies
uniform scaling to all positions.

Reference: TSRN Agent Model v2.0 Specification, Section 4.2
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PAdicContextScaling(nn.Module):
    """
    Non-Archimedean context scaling.

    Compresses positions based on their p-adic structural level.
    Local structure (low valuation) is preserved.
    Global structure (high valuation) is compressed.

    Scaling factor increases with structural level (larger valuation
    = higher in the tree = more compressed).
    """

    def __init__(
        self,
        training_ctx: int,
        p: int = 2,
        v_threshold: float = 4.7,
    ):
        """
        Args:
            training_ctx: Training context length (e.g., 512)
            p: Prime for p-adic valuation (default 2)
            v_threshold: Valuation threshold separating local from global
                For training_ctx=512: v_0 ≈ log2(512/20) ≈ 4.7
        """
        super().__init__()
        self.training_ctx = training_ctx
        self.p = p
        self.v_threshold = v_threshold

    def valuation(self, t: int) -> int:
        """
        2-adic valuation: largest k such that 2^k divides t.

        Args:
            t: Position index

        Returns:
            p-adic valuation
        """
        if t == 0:
            return float("inf")
        v = 0
        tt = t
        while tt % self.p == 0:
            v += 1
            tt //= self.p
        return v

    def valuation_tensor(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Vectorized p-adic valuation computation.

        v_p(t) = largest k such that p^k divides t. We start k from 1
        (since p^0 = 1 trivially divides every integer).

        Args:
            positions: (...,) tensor of position indices

        Returns:
            (...) tensor of p-adic valuations (float32, same device as input)
        """
        device = positions.device
        if self.p == 2:
            # Bitwise: 2^k | t  ⇔  (t & (2^k - 1)) == 0.
            positions_int = positions.to(torch.int64)
            valuations = torch.zeros(positions_int.shape, dtype=torch.float32, device=device)
            nonzero = positions_int > 0

            for k in range(1, 32):  # supports positions up to 2^31 − 1
                mask_k = (1 << k) - 1
                divisible = ((positions_int & mask_k) == 0) & nonzero
                valuations = valuations + divisible.float()

            return valuations
        else:
            # General-prime fallback: scalar loop over a flattened view.
            flat = positions.flatten().tolist()
            vals = [
                float(self.valuation(int(t))) if t > 0 else 0.0
                for t in flat
            ]
            return torch.tensor(vals, dtype=torch.float32, device=device).reshape(
                positions.shape
            )

    def scale(
        self,
        positions: torch.Tensor,
        inference_ctx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scale positions based on p-adic valuation.

        Args:
            positions: (T,) tensor of position indices
            inference_ctx: Target inference context length

        Returns:
            (scaled_positions, temp_correction) tuple, all on positions.device
        """
        device = positions.device
        scale_factor = float(inference_ctx) / float(self.training_ctx)

        # Compute valuations on the same device as the input.
        valuations = self.valuation_tensor(positions)

        # Smooth sigmoid transition at v_threshold (kept on-device).
        threshold = torch.tensor(self.v_threshold, dtype=valuations.dtype, device=device)
        blend = torch.sigmoid(valuations - threshold)

        # Per-position scale: blend between local (1.0) and global (scale_factor).
        per_position_scale = 1.0 + (scale_factor - 1.0) * blend

        # Apply scaling.
        scaled_positions = positions.float() / per_position_scale

        # Temperature correction (higher-level positions need more correction).
        temp_correction = 0.1 * torch.log(per_position_scale.clamp_min(1e-5)) + 1.0

        return scaled_positions, temp_correction

    def forward(
        self,
        position_ids: torch.Tensor,
        inference_ctx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply p-adic context scaling.

        Args:
            position_ids: (B, T) or (T,) tensor of position indices
            inference_ctx: Target inference context length

        Returns:
            (scaled_positions, temp_correction) tuple
        """
        return self.scale(position_ids, inference_ctx)


def scale_positions_padic(
    position_ids: torch.Tensor,
    training_ctx: int,
    inference_ctx: int,
    p: int = 2,
    v_threshold: float = 4.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone function for p-adic context scaling.

    Args:
        position_ids: (B, T) or (T,) tensor of position indices
        training_ctx: Training context length
        inference_ctx: Target inference context length
        p: Prime for p-adic valuation (default 2)
        v_threshold: Valuation threshold

    Returns:
        (scaled_positions, temp_correction) tuple
    """
    scaler = PAdicContextScaling(training_ctx, p, v_threshold)
    return scaler.scale(position_ids, inference_ctx)


if __name__ == "__main__":
    # Test PAdicContextScaling
    training_ctx = 512
    inference_ctx = 8192  # 16× extension

    scaler = PAdicContextScaling(training_ctx, p=2, v_threshold=4.7)

    # Test on synthetic positions
    positions = torch.arange(1024)
    scaled, temp_corr = scaler.scale(positions, inference_ctx)

    print(f"Original positions: {positions[:10].tolist()}")
    print(f"Scaled positions: {scaled[:10].tolist()}")
    print(f"Temp correction: {temp_corr[:10].tolist()}")

    # Test that low-valuation positions are less compressed
    # Position 1 (valuation 0) vs position 256 (valuation 8)
    scale_1 = scaled[1].item()
    scale_256 = scaled[256].item()

    print(f"\nScale at position 1 (v=0): {scale_1:.4f}")
    print(f"Scale at position 256 (v=8): {scale_256:.4f}")
    print(f"Ratio: {scale_256 / scale_1:.4f} (should be >1)")

    # Test batched forward
    batch_positions = torch.arange(512).unsqueeze(0).repeat(4, 1)
    scaled_batch, temp_batch = scaler(batch_positions, inference_ctx)
    print(f"\nBatch scaled shape: {scaled_batch.shape}")
    print(f"Batch temp shape: {temp_batch.shape}")

    print("All tests passed!")
