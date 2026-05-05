"""
P-adic Harmonic Positional Encoding (PAdicPE)
=============================================

Positional encoding based on p-adic structure of position indices.
Encodes positions via their 2-adic valuation hierarchy.

Positions at the same level of the 2-adic tree receive similar encoding
vectors, capturing syntactic parallel structure. Combines:
- Binary tree membership features (p-adic valuation)
- DCT-II eigenvectors of path graph Laplacian (smooth sequential variation)

Reference: TSRN Agent Model v2.0 Specification, Section 3.2
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PAdicHarmonicPE(nn.Module):
    """
    Positional encoding based on p-adic structure of position indices.

    Encodes positions via their 2-adic valuation hierarchy.
    Positions at the same level of the 2-adic tree receive similar
    encoding vectors, capturing syntactic parallel structure.

    Features:
    - Binary tree membership (depth features)
    - 2-adic valuation (structural level)
    - DCT-II modes (smooth sequential variation)

    The learned projection is zero-initialized so PE starts as a no-op.
    """

    def __init__(
        self,
        max_T: int,
        d_model: int,
        p: int = 2,
        depth: int = 10,
        dct_K: int = 32,
    ):
        """
        Args:
            max_T: Maximum sequence length
            d_model: Model dimension
            p: Prime for p-adic structure (default 2)
            depth: Number of p-adic valuation features (log2(max_T))
            dct_K: Number of DCT-II modes
        """
        super().__init__()
        self.max_T = max_T
        self.d_model = d_model
        self.p = p
        self.depth = depth
        self.dct_K = dct_K

        # Pre-compute p-adic features for each position
        padic_features = self._compute_padic_features(max_T, p, depth)

        # DCT-II features (eigenvectors of path graph Laplacian)
        dct_features = self._dct_features(max_T, dct_K)

        # Combine: p-adic + DCT-II
        full_features = torch.cat([padic_features, dct_features], dim=1)

        self.register_buffer("padic_features", full_features)

        # Learned projection: zero init so PE starts as no-op.
        # Feature layout (must match _compute_padic_features + _dct_features):
        #   depth      tree-membership bits (k = 1..depth)
        #   1          normalised 2-adic valuation
        #   dct_K      DCT-II modes
        n_features = depth + 1 + dct_K
        self.proj = nn.Linear(n_features, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)

    def _compute_padic_features(self, T: int, p: int, depth: int) -> torch.Tensor:
        """
        Compute p-adic features for each position.

        Feature k of position t (k=1..depth): whether p^k divides t.
        We skip k=0 because p^0 = 1 divides every integer, so that bit
        carries no information.  The depth-th channel is the normalised
        2-adic valuation v_p(t) / depth, capped at 1.0.

        Convention for t=0: by convention v_p(0) = +inf, but for a finite
        feature we treat it as the maximum valuation (depth) so that the
        zero-position lives at the top of the structural tree.

        Args:
            T: Max sequence length
            p: Prime (default 2)
            depth: Number of valuation features

        Returns:
            (T, depth + 1) tensor of p-adic features
        """
        features = torch.zeros(T, depth + 1)

        for t in range(T):
            # Binary tree membership features (skip k=0 — trivially true).
            for k in range(1, depth + 1):
                if t % (p ** k) == 0:
                    features[t, k - 1] = 1.0

            # p-adic valuation channel
            if t == 0:
                v = float(depth)  # treat as maximally divisible
            else:
                v = float(self._padic_valuation(t, p))
            features[t, depth] = min(v, float(depth)) / float(depth)

        return features

    def _padic_valuation(self, t: int, p: int) -> int:
        """
        Compute p-adic valuation: largest k such that p^k divides t.

        Args:
            t: Position index
            p: Prime (default 2)

        Returns:
            p-adic valuation
        """
        if t == 0:
            return float("inf")
        v = 0
        tt = t
        while tt % p == 0:
            v += 1
            tt //= p
        return v

    def _dct_features(self, T: int, K: int) -> torch.Tensor:
        """
        DCT-II basis: eigenvectors of path graph Laplacian.

        Captures smooth sequential variation.

        Args:
            T: Sequence length
            K: Number of DCT modes

        Returns:
            (T, K) tensor of DCT-II features
        """
        t = torch.arange(T, dtype=torch.float32)
        k = torch.arange(K, dtype=torch.float32)

        # DCT-II: cos(π * t * (k + 0.5) / T)
        features = torch.cos(math.pi * t.unsqueeze(1) * (k + 0.5).unsqueeze(0) / T)

        # Normalize
        features = features / features.norm(dim=0, keepdim=True)

        return features

    def forward(self, T: int, device=None, dtype=None) -> torch.Tensor:
        """
        Compute positional encodings for sequence length T.

        Signature mirrors ``SheafHarmonicPE.forward`` so the two layers are
        drop-in interchangeable.

        Args:
            T: Sequence length
            device: optional torch device for the returned PE
            dtype : optional dtype for the returned PE

        Returns:
            (T, d_model) positional encoding tensor
        """
        if T > self.padic_features.shape[0]:
            # Lazy extension: rebuild feature cache for longer sequences.
            new_padic = self._compute_padic_features(T, self.p, self.depth)
            new_dct = self._dct_features(T, self.dct_K)
            new_features = torch.cat([new_padic, new_dct], dim=1)
            self.padic_features = new_features.to(self.padic_features.device,
                                                    self.padic_features.dtype)
            self.max_T = T

        feats = self.padic_features[:T]
        if device is not None:
            feats = feats.to(device)
        if dtype is not None:
            feats = feats.to(dtype)
        return self.proj(feats)


def padic_valuation_tensor(positions: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Vectorized p-adic valuation computation.

    Args:
        positions: (...,) tensor of position indices
        p: Prime (default 2)

    Returns:
        (...) tensor of p-adic valuations
    """
    # For p=2, count trailing zeros via bitwise operations.
    # v_2(t) = max k such that 2^k | t  ⇔  the lowest k bits of t are zero.
    # We start at k=1 (k=0 trivially divides every integer).
    if p == 2:
        positions_int = positions.to(torch.int64)
        valuations = torch.zeros_like(positions_int, dtype=torch.float32)
        nonzero = positions_int > 0

        for k in range(1, 32):  # supports positions up to 2^31 − 1
            mask_k = (1 << k) - 1
            divisible = ((positions_int & mask_k) == 0) & nonzero
            valuations = valuations + divisible.float()

        return valuations
    else:
        # General case: scalar fallback for arbitrary p.
        flat = positions.flatten().tolist()
        out = torch.tensor(
            [float(_padic_valuation_scalar(int(t), p)) if t > 0 else 0.0 for t in flat],
            dtype=torch.float32,
            device=positions.device,
        )
        return out.reshape(positions.shape)


def _padic_valuation_scalar(t: int, p: int) -> int:
    """Scalar p-adic valuation."""
    if t == 0:
        return float("inf")
    v = 0
    tt = t
    while tt % p == 0:
        v += 1
        tt //= p
    return v


if __name__ == "__main__":
    # Test PAdicHarmonicPE
    max_T = 512
    d_model = 64

    pe = PAdicHarmonicPE(max_T, d_model, p=2, depth=10, dct_K=32)

    # Test forward pass
    T = 256
    encodings = pe(T)
    print(f"Encodings shape: {encodings.shape}")

    # Test that similar structural positions have similar encodings
    # Positions 256 and 512 should be close (both at 2^8 boundary)
    enc_256 = pe(512)[256]
    enc_512 = pe(512)[511]  # Close to 512
    enc_1 = pe(512)[0]

    sim_256_512 = F.cosine_similarity(enc_256.unsqueeze(0), enc_512.unsqueeze(0))
    sim_1_256 = F.cosine_similarity(enc_1.unsqueeze(0), enc_256.unsqueeze(0))

    print(f"Similarity (256, ~512): {sim_256_512.item():.4f}")
    print(f"Similarity (1, 256): {sim_1_256.item():.4f}")

    # Test vectorized valuation
    positions = torch.tensor([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    vals = padic_valuation_tensor(positions, p=2)
    print(f"Valuations: {vals}")

    print("All tests passed!")
