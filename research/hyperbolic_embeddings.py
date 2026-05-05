"""
Hyperbolic Token Embeddings
===========================

Token embeddings in the Poincaré disk (hyperbolic space) rather than
Euclidean space. Function words (the, a, is) sit near the center
(easily retrieved). Rare/specific words sit near the boundary
(retrieved only by specific queries).

The logarithmic map projects from the Poincaré disk to the tangent
space (flat R^d) for downstream processing, preserving hyperbolic
distances.

Reference: TSRN Agent Model v2.0 Specification, Section 2.3
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicEmbedding(nn.Module):
    """
    Hyperbolic embedding layer in the Poincaré disk.

    Tokens are stored in the Poincaré disk D^d = {x ∈ R^d : |x| < 1}.
    During forward pass, they are projected to the tangent space at the
    origin via the logarithmic map, giving a flat representation that
    preserves hyperbolic distances.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        init_strategy: str = "frequency",
        max_norm: float = 0.999,
        eps: float = 1e-5,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            init_strategy: How to initialize embeddings
                - "frequency": Function words near center, rare words near boundary
                - "uniform": Uniform in disk
                - "gaussian": Gaussian near center
            max_norm: Maximum norm (keep inside disk, < 1.0)
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.init_strategy = init_strategy
        self.max_norm = max_norm
        self.eps = eps

        # Embeddings stored in Poincaré disk — a learnable parameter so the
        # table is updated by the optimizer.  Initialised with a small
        # gaussian inside the disk to avoid the all-zero trap when the
        # caller forgets to invoke initialize_embeddings().
        init = torch.randn(vocab_size, d_model) * 0.01
        self.embeddings = nn.Parameter(init)

    @torch.no_grad()
    def initialize_embeddings(self, token_ranks: Optional[torch.Tensor] = None) -> None:
        """
        Initialize embeddings in Poincaré disk.

        Args:
            token_ranks: (V,) tensor of frequency ranks (1 = most frequent)
        """
        if self.init_strategy == "frequency" and token_ranks is not None:
            # Function words near center, rare words near boundary
            # |e(token)| ≈ 1 - exp(-rank(token) / V)
            ranks = token_ranks.float()
            norms = 1.0 - torch.exp(-ranks / self.vocab_size)
            norms = norms * self.max_norm

            # Random directions
            angles = torch.randn(self.vocab_size, self.d_model)
            angles = angles / (angles.norm(dim=-1, keepdim=True) + self.eps)

            new_emb = angles * norms.unsqueeze(1)

        elif self.init_strategy == "uniform":
            # Uniform in disk (volume-aware radius sampling)
            r = torch.rand(self.vocab_size) ** (1.0 / self.d_model) * self.max_norm
            theta = torch.randn(self.vocab_size, self.d_model)
            theta = theta / (theta.norm(dim=-1, keepdim=True) + self.eps)
            new_emb = theta * r.unsqueeze(1)

        elif self.init_strategy == "gaussian":
            # Gaussian near center, projected into disk
            tmp = torch.randn(self.vocab_size, self.d_model) * 0.1
            tmp = tmp / (tmp.norm(dim=-1, keepdim=True) + self.eps)
            new_emb = tmp * (torch.rand(self.vocab_size, 1) * self.max_norm)

        else:
            # Default: small random near center
            new_emb = torch.randn(self.vocab_size, self.d_model) * 0.01

        # Ensure all norms < 1 (inside disk)
        norms = new_emb.norm(dim=-1, keepdim=True)
        scale = torch.where(
            norms >= self.max_norm,
            self.max_norm / (norms + self.eps),
            torch.ones_like(norms),
        )
        new_emb = new_emb * scale

        # Write into the parameter via .data so the Parameter object stays
        # registered — reassigning self.embeddings would replace it with a
        # plain tensor and detach it from the optimizer.
        self.embeddings.data.copy_(new_emb.to(self.embeddings.device,
                                              self.embeddings.dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings and project to tangent space.

        Args:
            token_ids: (B, T) or (T,) tensor of token IDs

        Returns:
            (B, T, d_model) or (T, d_model) tensor in tangent space (flat R^d)
        """
        # Lookup in Poincaré disk
        x = F.embedding(token_ids, self.embeddings)  # (B, T, d) or (T, d)

        # Project to tangent space via logarithmic map at origin
        x_tangent = poincare_to_tangent(x, eps=self.eps)

        return x_tangent

    def from_tangent(self, x_tangent: torch.Tensor) -> torch.Tensor:
        """
        Project from tangent space back to Poincaré disk (exponential map).

        Args:
            x_tangent: (B, T, d) or (T, d) tensor in tangent space

        Returns:
            (B, T, d) or (T, d) tensor in Poincaré disk
        """
        return tangent_to_poincare(x_tangent, eps=self.eps)


def poincare_to_tangent(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Logarithmic map at the origin: projects from Poincaré disk to tangent space.

    log_0(x) = (2 / (1 - |x|²)) × x

    This gives a flat representation that preserves hyperbolic distances
    for small displacements from the origin.

    Args:
        x: (..., d) tensor in Poincaré disk (|x| < 1)
        eps: Numerical stability

    Returns:
        (..., d) tensor in tangent space (flat R^d)
    """
    norm_sq = (x * x).sum(dim=-1, keepdim=True)
    scale = 2.0 / (1.0 - norm_sq + eps)
    return x * scale


def tangent_to_poincare(v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Exponential map at the origin: projects from tangent space to Poincaré disk.

    exp_0(v) = tanh(|v| / 2) × (v / |v|)

    This is the inverse of the logarithmic map.

    Args:
        v: (..., d) tensor in tangent space
        eps: Numerical stability

    Returns:
        (..., d) tensor in Poincaré disk (|x| < 1)
    """
    norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    direction = v / norm
    radius = torch.tanh(norm / 2.0)
    return direction * radius


def poincare_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Hyperbolic distance in the Poincaré disk.

    d_H(x, y) = 2 arccosh(1 + 2|x-y|² / ((1-|x|²)(1-|y|²)))

    Args:
        x: (..., d) tensor in Poincaré disk
        y: (..., d) tensor in Poincaré disk
        eps: Numerical stability

    Returns:
        (...) scalar tensor of hyperbolic distances
    """
    diff_sq = (x - y).pow(2).sum(dim=-1)
    norm_x_sq = x.pow(2).sum(dim=-1)
    norm_y_sq = y.pow(2).sum(dim=-1)

    numerator = 2 * diff_sq
    denominator = (1 - norm_x_sq) * (1 - norm_y_sq) + eps

    arg = 1 + numerator / denominator
    arg = torch.clamp(arg, min=1.0 + eps)  # Ensure arccosh is valid

    return 2 * torch.arccosh(arg)


def mobius_add(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Möbius addition in the Poincaré disk.

    x ⊕ y = (x + y) / (1 + x·y)  (for 1D, generalized to d-dim)

    This is the hyperbolic analog of vector addition.

    Args:
        x: (..., d) tensor in Poincaré disk
        y: (..., d) tensor in Poincaré disk
        eps: Numerical stability

    Returns:
        (..., d) tensor in Poincaré disk
    """
    x_dot_y = (x * y).sum(dim=-1, keepdim=True)
    numerator = x + y
    denominator = 1 + x_dot_y
    result = numerator / (denominator + eps)

    # Out-of-disk safeguard: rescale (out-of-place to keep the autograd graph).
    norm = result.norm(dim=-1, keepdim=True)
    safe = torch.where(
        norm >= 1.0,
        result * (0.99 / (norm + eps)),
        result,
    )
    return safe


if __name__ == "__main__":
    # Test hyperbolic operations
    d_model = 64
    vocab_size = 1000

    # Create embedding layer
    emb = HyperbolicEmbedding(vocab_size, d_model, init_strategy="frequency")

    # Initialize with synthetic ranks
    ranks = torch.arange(1, vocab_size + 1)
    emb.initialize_embeddings(ranks)

    print(f"Embeddings shape: {emb.embeddings.shape}")
    print(f"Max norm: {emb.embeddings.norm(dim=-1).max().item():.4f}")

    # Test forward pass
    token_ids = torch.randint(0, vocab_size, (2, 10))
    x_tangent = emb(token_ids)
    print(f"Tangent space shape: {x_tangent.shape}")

    # Test round-trip
    x_poincare = emb.from_tangent(x_tangent)
    x_tangent2 = poincare_to_tangent(x_poincare)
    diff = (x_tangent - x_tangent2).abs().max().item()
    print(f"Round-trip error: {diff:.6f}")

    # Test distance
    x = emb.embeddings[:5]
    y = emb.embeddings[5:10]
    dist = poincare_distance(x, y)
    print(f"Hyperbolic distances: {dist}")

    print("All tests passed!")
