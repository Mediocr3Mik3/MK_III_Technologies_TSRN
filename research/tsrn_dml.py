"""
TSRN - Tropical Sheaf Renormalization Network (DirectML Edition)
================================================================
GPU validation & benchmark script adapted for AMD Radeon RX 6750 XT via DirectML.

Architecture (per whitepaper Sec 3.1-3.3):
  Scale 1 (T tokens): TropAttn -> Sheaf -> Reservoir -> CliffordFFN -> PAdicMem
  RG coarse-grain:    T -> T/2
  Scale 2 (T/2):      TropAttn -> Sheaf -> CliffordFFN -> PAdicAttention
  Upsample & fuse -> logits

All 7 components from the whitepaper are implemented:
  1. Tropical sparse attention     (Sec 2.1, Appendix A.1)
  2. Sheaf diffusion               (Sec 2.2, Appendix A.2)
  3. Clifford geometric FFN        (Sec 2.4, Appendix A.4)
  4. RG coarse-graining            (Sec 2.3)
  5. p-adic hierarchical memory    (Sec 2.5, Appendix A.3)
  6. Echo State reservoir           (Sec 2.6)
  7. Non-Archimedean p-adic attn   (Sec 2.5, Appendix A.3)

Parameter presets:
  --preset 2m   ~2M params   (d=256, 1 block/scale, char-level)
  --preset 50m  ~50M params  (d=512, 3 blocks/scale, WikiText-2)

Usage:
  python tsrn_dml.py --preset 2m --quick          # smoke test
  python tsrn_dml.py --preset 2m                   # full 2M run
  python tsrn_dml.py --preset 50m --dataset wikitext2  # 50M on real data
"""

import argparse
import json
import math
import os
import random
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import v2.0 components.  Imported lazily below to keep DML-only test
# environments importable when these modules' side-deps are missing.
from hyperbolic_embeddings import HyperbolicEmbedding
from padic_pe import PAdicHarmonicPE
from padic_context_scaling import PAdicContextScaling
from memory_hyperbolic import HyperbolicMemoryLayer


# ---------------------------------------------------------------------------
#  GPU-native prefix-max (Hillis-Steele parallel scan, O(log T) depth).
#
#  Replaces torch.cummax, which falls back to CPU on DirectML and kills
#  throughput (CPU round-trip per step).  This implementation uses
#  torch.maximum + torch.cat + torch.full on full-storage tensors,
#  GPU-native on every backend and autograd-safe on DML.
#
#  Semantics:  out[b, t, ...] = max(x[b, 0, ...], ..., x[b, t, ...])
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  DML-safe AdamW optimizer.
#
#  Stock torch.optim.AdamW uses aten::lerp internally (both single-tensor
#  via `.lerp_()` and multi-tensor via `torch._foreach_lerp_()`), which
#  falls back to CPU on DirectML.  On a 22.9M-param model this is a
#  per-parameter CPU round-trip every step — catastrophic for throughput.
#
#  AdamWDML replaces the EMA update
#      exp_avg.lerp_(grad, 1 - beta1)
#  with the mathematically-equivalent
#      exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#  which maps to GPU-native ops on every backend.  All other AdamW state
#  (exp_avg_sq, bias correction, decoupled weight decay, epsilon) matches
#  torch.optim.AdamW exactly.
# ---------------------------------------------------------------------------

class AdamWDML(torch.optim.Optimizer):
    """Drop-in AdamW that avoids `aten::lerp` (DirectML CPU fallback).

    Semantics identical to torch.optim.AdamW with decoupled weight decay.
    Supports per-param-group lr / weight_decay / betas / eps.
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamWDML does not support sparse gradients"
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # Decoupled weight decay: p <- p * (1 - lr * wd)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # First moment:  m <- beta1 * m + (1 - beta1) * g    (NO lerp)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # Second moment: v <- beta2 * v + (1 - beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                step_size = lr / bc1
                # denom = sqrt(v) / sqrt(bc2) + eps
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bc2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def prefix_max(x: Tensor, dim: int = 1) -> Tensor:
    """Parallel prefix-max along `dim` (default 1).

    Uses the Hillis-Steele scan: log2(T) iterations, each a single
    pointwise-max against a left-shifted copy.  Returns a tensor of the
    same shape and dtype as `x`.

    Implementation notes (DML-safe):
      * We avoid F.pad(mode='constant', value=-inf) because on the
        DirectML backend the backward pass mis-tracks stride/offset
        metadata for a slice of a slice padded with -inf, producing
        "ensure_in_bounds: ... out of bounds for storage" during
        loss.backward().
      * Instead, we build the shifted tensor via torch.cat of (i) a
        freshly-allocated -inf block (owns its own storage, no view)
        and (ii) a contiguous slice of cum (also its own storage).
        Every intermediate is a full-storage tensor, which autograd
        on DML handles correctly.
    """
    if dim != 1:
        x = x.transpose(1, dim)
        out = prefix_max(x, dim=1)
        return out.transpose(1, dim)

    T = x.shape[1]
    if T <= 1:
        return x

    cum = x
    step = 1
    while step < T:
        # (B, step, *trailing) block of -inf — fresh storage each iter.
        pad_shape = list(cum.shape)
        pad_shape[1] = step
        neg_inf_block = torch.full(
            pad_shape, float("-inf"),
            device=cum.device, dtype=cum.dtype,
        )
        # Contiguous slice forces a copy — no stale view metadata on DML.
        sliced = cum[:, :-step].contiguous()
        shifted = torch.cat([neg_inf_block, sliced], dim=1)
        cum = torch.maximum(cum, shifted)
        step *= 2
    return cum


# ---------------------------------------------------------------------------
#  Device detection (DirectML for AMD GPU)
# ---------------------------------------------------------------------------

_IS_DML = False

def detect_device() -> torch.device:
    global _IS_DML
    # Try DirectML first (AMD GPU)
    try:
        import torch_directml
        dev = torch_directml.device()
        _IS_DML = True
        print(f"  Device  : AMD GPU via DirectML")
        print(f"  Backend : DirectML  (torch {torch.__version__})")
        return dev
    except ImportError:
        pass
    # Try CUDA
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device  : {name}  ({mem:.1f} GB)")
        print(f"  Backend : CUDA  (torch {torch.__version__})")
        return dev
    # CPU fallback
    print(f"  Device  : CPU (no GPU detected)")
    return torch.device("cpu")


def device_sync(device):
    """Synchronize device if applicable."""
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
#  Dataset: character-level with optional WikiText-2
# ---------------------------------------------------------------------------

class CharDataset:
    """Character-level dataset with train/val split."""

    def __init__(self, path: str, context_len: int = 128, val_split: float = 0.1):
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        self.chars = sorted(set(text))
        self.vocab_sz = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.ctx = context_len

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        split = int(len(data) * (1 - val_split))
        self.train = data[:split]
        self.val = data[split:]

        print(f"  Vocab   : {self.vocab_sz} chars")
        print(f"  Train   : {len(self.train):,} tokens")
        print(f"  Val     : {len(self.val):,} tokens")

    def batch(self, split: str, B: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.ctx - 1, (B,))
        x = torch.stack([data[i:i + self.ctx] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + self.ctx + 1] for i in ix]).to(device)
        return x, y

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), "?") for i in ids)


class CharDatasetSplit:
    """Character-level dataset with explicit train/val text (no leakage)."""

    def __init__(self, train_text: str, val_text: str, context_len: int = 256,
                 test_text: str = None):
        all_text = train_text + val_text + (test_text or "")
        self.chars = sorted(set(all_text))
        self.vocab_sz = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.ctx = context_len

        self.train = torch.tensor([self.stoi[c] for c in train_text], dtype=torch.long)
        self.val = torch.tensor([self.stoi[c] for c in val_text], dtype=torch.long)
        if test_text:
            self.test = torch.tensor([self.stoi[c] for c in test_text], dtype=torch.long)
        else:
            self.test = None

        print(f"  Vocab   : {self.vocab_sz} chars")
        print(f"  Train   : {len(self.train):,} tokens")
        print(f"  Val     : {len(self.val):,} tokens")
        if self.test is not None:
            print(f"  Test    : {len(self.test):,} tokens")

    def batch(self, split: str, B: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        if split == "test" and self.test is not None:
            data = self.test
        elif split == "val":
            data = self.val
        else:
            data = self.train
        ix = torch.randint(len(data) - self.ctx - 1, (B,))
        x = torch.stack([data[i:i + self.ctx] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + self.ctx + 1] for i in ix]).to(device)
        return x, y

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), "?") for i in ids)


def _load_hf_wikitext(name: str, config: str, context_len: int) -> CharDatasetSplit:
    """Load a HuggingFace WikiText dataset with proper train/val/test splits."""
    cache_dir = Path("data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / f"{name}_train.txt"
    val_path = cache_dir / f"{name}_val.txt"
    test_path = cache_dir / f"{name}_test.txt"

    if not train_path.exists():
        print(f"  Downloading {name}...")
        from datasets import load_dataset
        ds = load_dataset("wikitext", config)
        train_text = "\n".join(ds["train"]["text"])
        val_text = "\n".join(ds["validation"]["text"])
        test_text = "\n".join(ds["test"]["text"])
        train_path.write_text(train_text, encoding="utf-8")
        val_path.write_text(val_text, encoding="utf-8")
        test_path.write_text(test_text, encoding="utf-8")
        print(f"  Saved: train={len(train_text):,}  val={len(val_text):,}  test={len(test_text):,} chars")
    else:
        train_text = train_path.read_text(encoding="utf-8", errors="replace")
        val_text = val_path.read_text(encoding="utf-8", errors="replace")
        test_text = test_path.read_text(encoding="utf-8", errors="replace")
        print(f"  Loaded cached: train={len(train_text):,}  val={len(val_text):,}  test={len(test_text):,} chars")

    return CharDatasetSplit(train_text, val_text, context_len=context_len,
                            test_text=test_text)


def load_wikitext2(context_len: int = 256) -> CharDatasetSplit:
    """WikiText-2 with proper official train/val/test splits."""
    return _load_hf_wikitext("wikitext2", "wikitext-2-raw-v1", context_len)


def load_wikitext103(context_len: int = 256) -> CharDatasetSplit:
    """WikiText-103 with proper official train/val/test splits."""
    return _load_hf_wikitext("wikitext103", "wikitext-103-raw-v1", context_len)


def load_enwik8(context_len: int = 256) -> CharDatasetSplit:
    """Load enwik8 with standard BYTE-LEVEL 90M/5M/5M split.

    The community standard (Al-Rfou 2019, Transformer-XL, SHA-RNN) treats
    each byte as a token (vocab <= 256).  We use latin-1 decoding which
    gives a 1:1 byte-to-character mapping, ensuring BPC numbers are
    directly comparable to published results.
    """
    cache_dir = Path("data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "enwik8.raw"

    if not raw_path.exists():
        print("  Downloading enwik8...")
        import urllib.request, zipfile, io
        url = "http://mattmahoney.net/dc/enwik8.zip"
        resp = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        data = z.read("enwik8")
        raw_path.write_bytes(data)
        print(f"  Downloaded {len(data):,} bytes")

    # Byte-level: latin-1 gives 1:1 byte→char mapping (standard protocol)
    raw = raw_path.read_bytes().decode("latin-1")
    assert len(raw) == 100_000_000, f"enwik8 should be 100M bytes, got {len(raw)}"

    # Standard split at exact byte boundaries
    train_text = raw[:90_000_000]
    val_text   = raw[90_000_000:95_000_000]
    test_text  = raw[95_000_000:]
    print(f"  Byte-level split: train={len(train_text):,}  "
          f"val={len(val_text):,}  test={len(test_text):,}")

    return CharDatasetSplit(train_text, val_text, context_len=context_len,
                            test_text=test_text)


def generate_synthetic_data(path: str, n_paragraphs: int = 5000) -> str:
    """Generate structured synthetic text (whitepaper Sec 4 baseline)."""
    random.seed(42)
    nouns = ['king', 'queen', 'knight', 'castle', 'sword', 'crown', 'throne',
             'realm', 'duke', 'lord', 'witch', 'dragon', 'tower', 'forest',
             'river', 'stone', 'blade', 'shield', 'horn', 'fire', 'moon',
             'star', 'night', 'dawn', 'fate']
    verbs = ['stood', 'fell', 'rose', 'came', 'went', 'spoke', 'cried',
             'saw', 'knew', 'felt', 'ruled', 'fought', 'lived', 'died',
             'loved', 'feared', 'sought']
    adjs = ['great', 'dark', 'old', 'young', 'brave', 'true', 'cold',
            'black', 'red', 'white', 'long', 'high', 'deep', 'sharp',
            'proud', 'fierce', 'pale']
    preps = ['of', 'in', 'by', 'with', 'from', 'through', 'above', 'below']
    conj = ['and', 'but', 'yet', 'so', 'for', 'though', 'while', 'when']

    def np_():
        if random.random() < 0.5:
            return f"the {random.choice(adjs)} {random.choice(nouns)}"
        return f"the {random.choice(nouns)}"

    def sent():
        s = f"{np_()} {random.choice(verbs)} {np_()}"
        if random.random() < 0.3:
            s += f" {random.choice(preps)} {np_()}"
        if random.random() < 0.2:
            s += f", {random.choice(conj)} {np_()} {random.choice(verbs)}"
        return s.capitalize() + "."

    def para():
        return " ".join(sent() for _ in range(random.randint(3, 6)))

    text = "\n\n".join(para() for _ in range(n_paragraphs))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    print(f"  Generated {len(text):,} chars -> {path}")
    return path


# ---------------------------------------------------------------------------
#  Rotary Position Embedding (RoPE)
#
#  Applies complex rotation to (q, k) pairs based on absolute position.
#  Eliminates the need for learned absolute position embeddings while
#  encoding relative position information directly in attention scores.
#  Compatible with tropical attention: rotation preserves max structure.
# ---------------------------------------------------------------------------

class SheafHarmonicPE(nn.Module):
    """Sheaf Harmonic positional encoding (NEXUS Innovation #4).

    Additive PE built from the bottom-n_harmonics eigenfunctions of the
    1-D path-graph Laplacian.  For a path of length T the Laplacian has
    closed-form eigenvectors phi_k(t) = sqrt(2/T) cos(pi*k*(t+1/2)/T),
    k=0..T-1, with eigenvalues 2(1-cos(pi*k/T)).  We keep the smoothest
    n_harmonics (small k) — these are the sheaf-harmonic modes that best
    respect local coherence.

    The harmonics are a static cosine basis (no input-dependent eigen-
    decomposition per forward, so DML-fast).  A learned linear projection
    maps them into d_model.  Projection is zero-initialised so the model
    starts identical to the baseline and learns what positional content
    to inject.

    Causal-safe: each position t has its own deterministic PE vector; no
    cross-token mixing, no dependence on future tokens.
    """
    def __init__(self, d_model: int, max_seq_len: int = 4096,
                 n_harmonics: int = 64):
        super().__init__()
        self.n_harmonics = min(n_harmonics, max_seq_len)
        self.proj = nn.Linear(self.n_harmonics, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)
        # Eigenvalues of the path-graph Laplacian (for diagnostics/spectral gap)
        k = torch.arange(self.n_harmonics, dtype=torch.float32)
        eigvals = 2.0 * (1.0 - torch.cos(math.pi * k / max(max_seq_len, 1)))
        self.register_buffer("eigvals", eigvals, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (T,1)
        k = torch.arange(self.n_harmonics, dtype=torch.float32).unsqueeze(0)
        # DCT-II basis (eigenvectors of the Neumann path-graph Laplacian)
        phi = math.sqrt(2.0 / max(seq_len, 1)) * torch.cos(
            math.pi * k * (t + 0.5) / max(seq_len, 1))  # (T, n_harmonics)
        self.register_buffer("harmonics", phi, persistent=False)

    def forward(self, T: int, device, dtype) -> Tensor:
        """Return additive PE of shape (T, d_model)."""
        if T > self.harmonics.shape[0]:
            self._build_cache(T)
            # move buffers to right device
            self.harmonics = self.harmonics.to(device)
        h = self.harmonics[:T].to(device=device, dtype=dtype)
        return self.proj(h)

    def spectral_gap(self) -> float:
        """Gap between smallest and second-smallest eigenvalue.
        Diagnostic: larger gap -> better positional disambiguation."""
        if self.eigvals.numel() < 2:
            return 0.0
        return float(self.eigvals[1] - self.eigvals[0])


class RotaryPositionEmbedding(nn.Module):
    """RoPE: applies sinusoidal rotary embeddings to query/key tensors."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 500000.0):
        super().__init__()
        # Higher base (Llama-3 style) = finer high-frequency resolution,
        # slower phase rotation on low dims -> better long-range handling.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)        # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)                    # (T, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor, offset: int = 0) -> Tuple[Tensor, Tensor]:
        """Apply rotary embeddings.
        q, k: (B, H, T, dh)
        Returns rotated (q, k) of the same shape.
        """
        T = q.shape[2]
        if offset + T > self.cos_cached.shape[0]:
            self._build_cache(offset + T)
        cos = self.cos_cached[offset : offset + T].unsqueeze(0).unsqueeze(0)  # 1 1 T dh
        sin = self.sin_cached[offset : offset + T].unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
#  ALiBi — Attention with Linear Biases
#
#  Adds a head-specific linear bias -m_h * |i - j| to attention scores.
#  Combined with RoPE for hybrid positional encoding.
# ---------------------------------------------------------------------------

def build_alibi_slopes(n_heads: int) -> Tensor:
    """Compute per-head slope values for ALiBi.
    Uses the geometric sequence 2^{-8/n_heads * (h+1)} for each head h."""
    ratio = 2.0 ** (-8.0 / n_heads)
    slopes = torch.tensor([ratio ** (h + 1) for h in range(n_heads)])
    return slopes  # (H,)


def build_alibi_bias(T: int, slopes: Tensor) -> Tensor:
    """Build the (H, T, T) ALiBi bias matrix."""
    pos = torch.arange(T, dtype=slopes.dtype, device=slopes.device)
    # relative distance matrix: |i - j|
    dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()  # (T, T)
    # Per-head bias: -slope_h * dist
    return -slopes.view(-1, 1, 1) * dist.unsqueeze(0)   # (H, T, T)


# ---------------------------------------------------------------------------
#  1. Tropical Sparse Attention  (Whitepaper Sec 2.1, Appendix A.1)
#
#     Tropical inner product: (q ⊕_T k)_j = max_i(q_i + k_i)
#     Top-k sparse softmax applied to tropical scores.
#     DirectML fix: use masked_fill instead of scatter_ for backward compat.
#     + RoPE on Q/K, ALiBi bias on scores.
# ---------------------------------------------------------------------------

class TropicalAttention(nn.Module):
    def __init__(self, d_model: int, top_k: int = 8, n_heads: int = 4,
                 dropout: float = 0.0, max_seq_len: int = 4096,
                 use_pacs: bool = False, training_ctx: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
        self.H = n_heads
        self.dh = d_model // n_heads
        self.top_k = top_k
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.Wq.weight, gain=0.5)
        nn.init.xavier_uniform_(self.Wk.weight, gain=0.5)
        nn.init.xavier_uniform_(self.Wv.weight, gain=0.5)
        # QK-Norm: per-head RMSNorm on Q, K before RoPE+attention.
        # Prevents softmax saturation, stabilizes training (Dehghani+2023, Henry+2020).
        # Causal-safe: pointwise-per-token op, no cross-token mixing.
        self.q_norm = nn.LayerNorm(self.dh, elementwise_affine=True)
        self.k_norm = nn.LayerNorm(self.dh, elementwise_affine=True)
        # RoPE: rotary position embeddings on Q/K
        self.rope = RotaryPositionEmbedding(self.dh, max_seq_len=max_seq_len)
        # ALiBi: per-head linear bias on attention scores
        self.register_buffer("alibi_slopes", build_alibi_slopes(n_heads))
        # Cross-window K/V cache — extends effective context during eval
        self.cross_window_mem = PersistentCrossWindowMemory(
            d_model, n_heads, max_cached_len=256)
        # Maslov dequantization parameter h (NEXUS Innovation #3).
        # h * logsumexp(raw/h) -> max as h->0 (tropical), -> smooth as h->inf.
        # Registered as buffer so set_maslov_h() can cycle it during training.
        self.register_buffer("maslov_h", torch.tensor(1.0))
        # P-adic Context Scaling (PaCS) for inference
        self.use_pacs = use_pacs
        self.training_ctx = training_ctx
        if use_pacs:
            self.padic_scaler = PAdicContextScaling(training_ctx=training_ctx)

    def _attend_chunk(self, Qq: Tensor, K: Tensor, V: Tensor,
                      qi: int, q_end: int, k: int, T_kv: int,
                      causal: bool, T_c: int = 0) -> Tensor:
        """Compute attention output for a chunk of queries [qi, q_end).

        Tropical scores via channel-chunked logsumexp to bound peak memory.
        T_c: number of prepended cached keys (for ALiBi/causal offset).
        """
        B, H, qc = Qq.shape[0], Qq.shape[1], Qq.shape[2]
        dh = Qq.shape[3]
        max_5d_bytes = 32 * 1024 * 1024
        elems_per_cs = max(1, B * H * qc * T_kv)
        cs = max(1, min(dh, max_5d_bytes // (elems_per_cs * 4)))

        # Maslov: scores = h * logsumexp(raw/h).  h=1 -> vanilla logsumexp.
        h = self.maslov_h
        if cs >= dh:
            raw = Qq.unsqueeze(3) + K.unsqueeze(2)
            scores = h * torch.logsumexp(raw / h, dim=-1)
            del raw
        else:
            scores = None
            for c0 in range(0, dh, cs):
                c1 = min(c0 + cs, dh)
                raw = Qq[:, :, :, c0:c1].unsqueeze(3) + K[:, :, :, c0:c1].unsqueeze(2)
                chunk_lse = h * torch.logsumexp(raw / h, dim=-1)
                del raw
                if scores is None:
                    scores = chunk_lse
                else:
                    m = torch.max(scores.detach(), chunk_lse.detach())
                    scores = m + torch.log(
                        torch.exp(scores - m) + torch.exp(chunk_lse - m))
                del chunk_lse

        # ALiBi bias — extended positions for cross-window keys
        q_pos = torch.arange(qi, q_end, device=scores.device)
        k_pos = torch.arange(-T_c, T_kv - T_c, device=scores.device)
        dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()  # (qc, T_kv)
        alibi_chunk = -self.alibi_slopes.to(scores.device).view(-1, 1, 1) * dist.unsqueeze(0)
        scores = scores + alibi_chunk.unsqueeze(0)

        # Causal mask: cached keys (pos<0) always visible; current keys if pos<=query
        if causal:
            col = k_pos.unsqueeze(0)      # (1, T_kv)
            row = q_pos.unsqueeze(1)      # (qc, 1)
            scores = scores.masked_fill((col > row).unsqueeze(0).unsqueeze(0), float("-inf"))

        topk_v, _ = scores.topk(min(k, T_kv), dim=-1)
        thr = topk_v[:, :, :, -1:].detach()
        scores = scores.masked_fill(scores < thr, float("-inf"))
        w = self.drop(torch.softmax(scores, dim=-1))
        del scores

        return w @ V

    def _channel_chunked_scores(self, Q: Tensor, K: Tensor,
                                B: int, H: int, T: int, dh: int) -> Tensor:
        """Fast channel-chunked path for short contexts.
        Computes full (B,H,T,T) scores via Maslov-softened logsumexp: h * lse(x/h).
        """
        # Chunk budget sized to fit raw 5D tensor + its backward copy in memory
        max_chunk_bytes = 192 * 1024 * 1024  # 192 MB (autograd ~2x this)
        T_q, T_k = Q.shape[2], K.shape[2]
        cs = max(1, min(dh, max_chunk_bytes // max(1, B * H * T_q * T_k * 4)))
        h = self.maslov_h
        scores = None
        for c0 in range(0, dh, cs):
            c1 = min(c0 + cs, dh)
            raw = Q[:, :, :, c0:c1].unsqueeze(3) + K[:, :, :, c0:c1].unsqueeze(2)
            chunk_lse = h * torch.logsumexp(raw / h, dim=-1)
            del raw
            if scores is None:
                scores = chunk_lse
            else:
                m = torch.max(scores.detach(), chunk_lse.detach())
                scores = m + torch.log(
                    torch.exp(scores - m) + torch.exp(chunk_lse - m))
            del chunk_lse
        return scores.contiguous()

    def forward(self, x: Tensor, causal: bool = True, inference_ctx: int = None) -> Tensor:
        B, T, d = x.shape
        H, dh = self.H, self.dh
        k = min(self.top_k, T)

        # P-adic Context Scaling (PaCS): only applied at inference when the
        # caller explicitly requests an extension beyond training_ctx.  We
        # compute per-position temperature corrections that will be used to
        # rescale attention logits below; positions with high 2-adic
        # valuation (paragraph/section anchors) are compressed more.
        pacs_temp = None
        pacs_scaled_pos = None
        if (self.use_pacs and not self.training and inference_ctx
                and inference_ctx > self.training_ctx):
            positions = torch.arange(T, device=x.device, dtype=torch.long)
            pacs_scaled_pos, pacs_temp = self.padic_scaler.scale(
                positions, inference_ctx)
            # Reshape for broadcast over (B, H, T_q, T_kv).
            pacs_temp = pacs_temp.view(1, 1, T, 1)

        Q = self.Wq(x).view(B, T, H, dh).permute(0, 2, 1, 3)
        K = self.Wk(x).view(B, T, H, dh).permute(0, 2, 1, 3)
        V = self.Wv(x).view(B, T, H, dh).permute(0, 2, 1, 3)

        # QK-Norm before RoPE (per-token, per-head — preserves causality)
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # Apply RoPE to Q/K
        Q, K = self.rope(Q, K)

        # Cross-window memory: prepend cached K/V (opt-in, eval only — causal)
        # Disabled by default — only activates for sequential autoregressive
        # generation via model.set_cross_window_enabled(True).
        K_orig, V_orig = K, V
        T_c = 0
        if not self.training and self.cross_window_mem.enabled:
            ck, cv = self.cross_window_mem.get_cached_kv(B)
            if ck is not None:
                K = torch.cat([ck, K], dim=2)
                V = torch.cat([cv, V], dim=2)
                T_c = ck.shape[2]
        T_kv = K.shape[2]

        # ── Hybrid attention: fast path vs O(T·k) memory path ──
        score_bytes = B * H * T * T_kv * 4
        use_fast = score_bytes <= 24 * 1024 * 1024

        if use_fast:
            scores = self._channel_chunked_scores(Q, K, B, H, T_kv, dh)

            # Extended ALiBi: q_pos=[0..T-1], k_pos=[-T_c..T-1]
            q_pos = torch.arange(T, device=scores.device)
            k_pos = torch.arange(-T_c, T, device=scores.device)
            dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()
            alibi = -self.alibi_slopes.to(scores.device).view(-1, 1, 1) * dist.unsqueeze(0)
            scores = scores + alibi.unsqueeze(0)

            # PaCS temperature correction: divide scores at extended-context
            # query positions by per-position temperature so that highly
            # compressed (high-valuation) positions soften their attention,
            # avoiding sharp activation cliffs at structural boundaries.
            if pacs_temp is not None:
                scores = scores / pacs_temp.to(scores.dtype)

            if causal:
                mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)  # (T, T_kv)
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            topk_v, _ = scores.topk(min(k, T_kv), dim=-1)
            thr = topk_v[:, :, :, -1:].detach()
            scores = scores.masked_fill(scores < thr, float("-inf"))
            w = self.drop(torch.softmax(scores, dim=-1))
            ctx = (w @ V).permute(0, 2, 1, 3).contiguous().reshape(B, T, d)
        else:
            max_5d_elems = 16 * 1024 * 1024
            elem_per_qc  = max(1, B * H * T_kv * dh)
            qc = max(1, min(T, max_5d_elems // elem_per_qc))

            out_parts = []
            for qi in range(0, T, qc):
                q_end = min(qi + qc, T)
                Qq = Q[:, :, qi:q_end, :]

                if self.training:
                    chunk = torch.utils.checkpoint.checkpoint(
                        self._attend_chunk, Qq, K, V,
                        qi, q_end, k, T_kv, causal, T_c,
                        use_reentrant=False)
                else:
                    chunk = self._attend_chunk(Qq, K, V, qi, q_end, k, T_kv, causal, T_c)
                out_parts.append(chunk)

            ctx = torch.cat(out_parts, dim=2)
            ctx = ctx.permute(0, 2, 1, 3).contiguous().reshape(B, T, d)

        # Cache current window K/V for next eval call (opt-in)
        if not self.training and self.cross_window_mem.enabled:
            self.cross_window_mem.cache_kv(K_orig, V_orig)

        return self.Wo(ctx)


# ---------------------------------------------------------------------------
#  2. Sheaf Diffusion  (Whitepaper Sec 2.2, Appendix A.2)
#
#     Sheaf Laplacian: E(x) = 1/2 sum_i sum_d ||R_d x_i - x_{i+d}||^2
#     Update: x <- x - alpha * grad E
#     Restriction maps R_d = capsule transformation matrices (Hinton equiv.)
# ---------------------------------------------------------------------------

class SheafDiffusion(nn.Module):
    def __init__(self, d_model: int, window: int = 3, causal: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        # For causal (autoregressive) LM, only use non-positive offsets
        # so no future tokens are visible. Bidirectional uses full window.
        if causal:
            self.offsets = list(range(-window, 1))  # [-3,-2,-1,0]
        else:
            self.offsets = list(range(-window, window + 1))
        self.R = nn.ModuleDict({
            str(d): nn.Linear(d_model, d_model, bias=True)
            for d in self.offsets
        })
        self.alpha = nn.Parameter(torch.tensor(0.15))  # Appendix A.2: init 0.15
        self.drop = nn.Dropout(dropout)
        for mod in self.R.values():
            nn.init.eye_(mod.weight)
            mod.weight.data += 0.01 * torch.randn_like(mod.weight)
            nn.init.zeros_(mod.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, T, d = x.shape
        laplacian = torch.zeros_like(x)
        Rx = {str(delta): self.R[str(delta)](x) for delta in self.offsets}
        for delta in self.offsets:
            R = self.R[str(delta)]
            # DirectML fix: avoid F.pad (non-contiguous backward) and
            # torch.roll (CPU fallback on DirectML). Use slice+cat instead.
            if delta == 0:
                x_nb = x
            elif delta > 0:
                # x_{i+delta}: shift left by delta, pad zeros at end
                x_nb = torch.cat([x[:, delta:, :],
                                  torch.zeros(B, delta, d, device=x.device, dtype=x.dtype)], dim=1)
            else:
                # x_{i+delta}: shift right by |delta|, pad zeros at start
                ad = -delta
                x_nb = torch.cat([torch.zeros(B, ad, d, device=x.device, dtype=x.dtype),
                                  x[:, :T - ad, :]], dim=1)
            incon = Rx[str(delta)] - x_nb
            laplacian = laplacian + incon @ R.weight  # R^T @ inconsistency
        return self.drop(x - self.alpha.abs() * laplacian)


# ---------------------------------------------------------------------------
#  3. Clifford Geometric FFN  (Whitepaper Sec 2.4, Appendix A.4)
#
#     Split to (r, i), compute geometric self-product:
#       grade-0: r^2 - i^2   (scalar similarity)
#       grade-2: 2*r*i        (bivector, oriented geometry)
#     Gated output projection.
# ---------------------------------------------------------------------------

class CliffordFFN(nn.Module):
    """Clifford geometric FFN with SwiGLU gating (Shazeer 2020).

    Geometric products compute grade-0 (scalar) and grade-2 (bivector),
    then SwiGLU provides smoother gradient flow than sigmoid gating.
    SwiGLU: out = (SiLU(W_gate . x) * W_up . h) . W_out

    Note: MoE variant was evaluated but rejected due to DirectML AdamW
    kernel-launch overhead (~10x slower step time). On CUDA the MoE
    version with n_experts=4 top_k=2 would be preferable.
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        dh = d_model // 2
        self.proj_r = nn.Linear(d_model, dh)
        self.proj_i = nn.Linear(d_model, dh)
        self.proj_out = nn.Linear(d_model, d_model)
        self.gate_w = nn.Linear(d_model, d_model)
        self.gate_v = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.gate_w.bias)
        nn.init.zeros_(self.gate_v.bias)

    def forward(self, x: Tensor) -> Tensor:
        r = self.proj_r(x)
        i = self.proj_i(x)
        prod_r = r * r - i * i    # grade-0
        prod_i = 2.0 * r * i      # grade-2
        h = torch.cat([prod_r, prod_i], dim=-1)
        swiglu = F.silu(self.gate_w(x)) * self.gate_v(h)
        return self.drop(self.proj_out(swiglu))


# ---------------------------------------------------------------------------
#  4. RG Coarse-graining  (Whitepaper Sec 2.3)
#
#     MERA-inspired: disentangle pairs then pool.
#     Disentangle: 2d -> 2d (near-identity, removes short-range correlations)
#     Pool: 2d -> d (merge adjacent tokens)
#
#     CAUSALITY REQUIREMENT (autoregressive LM):
#       When predicting the target for fine position t, the model must only
#       have access to x_0 ... x_t.  Coarse token j feeds back into fine
#       positions {2j, 2j+1} via the upsample-fuse step, so coarse token j
#       must be computed from fine tokens with index <= 2j only.
#
#     FIX — causal pairing:
#       We shift x left by one via zero-padding on the left:
#           x_shifted = [0, x_0, x_1, ..., x_{T-1}]   (length T+1)
#       Then:
#           left  = x_shifted[0::2] = [0,      x_1,   x_3,  ...]  (indices 2j-1)
#           right = x_shifted[1::2] = [x_0,    x_2,   x_4,  ...]  (indices 2j  )
#       Coarse token j = pool(x_{2j-1}, x_{2j}), where x_{-1} := 0.
#       Latest index used: 2j <= 2j.  ✓  No future tokens visible.
#
#     Upsample check:
#       repeat_interleave(2) maps coarse j -> fine {2j, 2j+1}.
#       Fine position t receives xc[t//2], built from x_{t-1} and x_t.
#       Latest index: t.  ✓
# ---------------------------------------------------------------------------

class RGPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.disentangle = nn.Linear(2 * d_model, 2 * d_model)
        nn.init.eye_(self.disentangle.weight)
        self.disentangle.weight.data += 0.01 * torch.randn_like(self.disentangle.weight)
        nn.init.zeros_(self.disentangle.bias)
        self.pool = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, d = x.shape
        # Ensure even length (drop last token if odd — same as before)
        if T % 2 != 0:
            x = x[:, :-1, :]
            T -= 1
        # Causal shift: prepend one zero frame so pair j uses (x[2j-1], x[2j]).
        # x_shifted shape: (B, T+1, d).  x_{-1} is defined as the zero vector.
        pad  = torch.zeros(B, 1, d, device=x.device, dtype=x.dtype)
        x_sh = torch.cat([pad, x], dim=1)          # (B, T+1, d)
        # x_sh has T+1 elements at indices 0..T.
        # We need exactly T/2 pairs.  T+1 elements -> take first T elements so
        # both strides land on [0, T/2) pairs:
        #   left  = x_sh[:, 0::2, :]  indices 0, 2, 4, ...  (= x_{-1}, x_1, x_3, ...)
        #   right = x_sh[:, 1::2, :]  indices 1, 3, 5, ...  (= x_0,   x_2, x_4, ...)
        # With T+1 elements both slices have ceil((T+1)/2) = T/2+1 entries if T
        # is even, so we slice to exactly T//2.
        left  = x_sh[:, 0::2, :][:, :T//2, :].contiguous()   # (B, T/2, d)
        right = x_sh[:, 1::2, :][:, :T//2, :].contiguous()   # (B, T/2, d)
        pair  = torch.cat([left, right], dim=-1)              # (B, T/2, 2d)
        pair  = torch.tanh(self.disentangle(pair))
        return self.norm(self.pool(pair)).contiguous()


# ---------------------------------------------------------------------------
#  5. p-adic Hierarchical Memory  (Whitepaper Sec 2.5)
#
#     Binary tree of learnable key-value pairs.
#     Flat soft attention for small M; tree routing for large M.
# ---------------------------------------------------------------------------

class PAdicMemory(nn.Module):
    def __init__(self, d_model: int, depth: int = 6):
        super().__init__()
        self.depth = depth
        self.M = 2 ** depth

        self.leaf_keys = nn.Parameter(torch.randn(self.M, d_model) * 0.02)
        self.leaf_values = nn.Parameter(torch.randn(self.M, d_model) * 0.02)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wout = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, d = x.shape
        q = self.Wq(x)
        # Flat soft retrieval (exact for M <= 128, efficient enough for training)
        scores = q @ self.leaf_keys.T / math.sqrt(d)
        weights = torch.softmax(scores, dim=-1)
        ret = weights @ self.leaf_values
        return self.Wout(ret)


# ---------------------------------------------------------------------------
#  6. Echo State Reservoir  (Whitepaper Sec 2.6)
#
#     Revived ESN with: learnable spectral radius, fine-tunable weights,
#     readout starts as zero (no-op at init).
#     DirectML fix: power iteration for spectral radius (no eigvals on GPU).
# ---------------------------------------------------------------------------

class EchoStateReservoir(nn.Module):
    """GRU-gated Echo State Reservoir.

    Replaces the global scalar leak rate with per-dimension GRU gates:
      z_t = sigma(W_z · [h_{t-1}, u_t])     — update gate
      r_t = sigma(W_r · [h_{t-1}, u_t])     — reset gate
      h̃_t = tanh(W_res · (r_t ⊙ h_{t-1}) + u_t)
      h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

    This gives each reservoir dimension independent forget/update control,
    significantly improving selective memory for long-range dependencies.
    """
    def __init__(self, d_model: int, d_reservoir: int = None, sparsity: float = 0.9):
        super().__init__()
        dr = d_reservoir or d_model
        self.dr = dr

        self.W_in = nn.Linear(d_model, dr, bias=False)

        # Sparse random init near edge of chaos
        W_res = torch.randn(dr, dr)
        mask = torch.rand(dr, dr) > sparsity
        W_res = W_res * mask.float()
        # Scale to spectral radius ~ 0.95 on CPU (eigvals only at init)
        with torch.no_grad():
            eigs = torch.linalg.eigvals(W_res).abs()
            rho = eigs.max().item()
            if rho > 0:
                W_res = W_res * (0.95 / rho)
        self.W_res = nn.Parameter(W_res)

        self.log_rho = nn.Parameter(torch.tensor(0.0))
        self.readout = nn.Linear(dr, d_model, bias=False)
        nn.init.zeros_(self.readout.weight)

        # GRU gates: update (z) and reset (r), each takes [h, u] -> dr
        self.W_z = nn.Linear(2 * dr, dr)
        self.W_r = nn.Linear(2 * dr, dr)
        # Bias init: z gate starts high (~0.7) to let information flow through
        nn.init.zeros_(self.W_z.weight)
        nn.init.constant_(self.W_z.bias, 0.85)  # sigmoid(0.85) ≈ 0.7
        nn.init.zeros_(self.W_r.weight)
        nn.init.constant_(self.W_r.bias, 2.0)   # sigmoid(2.0) ≈ 0.88 — mostly reset on

        # Deterministic starting vector for power iter (no RNG, no persistent state).
        # This removes cross-call state pollution: every forward sees the same
        # starting point, so rho_current is a pure function of W_res and is
        # identical across batches -> batch-independent (causal) reservoir output.
        v0 = torch.ones(dr) / math.sqrt(dr)
        self.register_buffer("_v0", v0, persistent=False)

    def _power_iter_spectral_radius(self, W: Tensor, n_iter: int = 6) -> Tensor:
        """Approximate spectral radius via power iteration (GPU-safe, no eigvals).

        Deterministic: starts from a fixed unit vector, runs n_iter GEMM sweeps,
        returns ||W v|| / ||v||.  Uses detached W so gradients flow through
        rho_target only, not through the estimate itself.  No persistent state
        across calls — ensures batch-independence (no stale cached eigenvector
        leaking information between forward passes).
        """
        W_det = W.detach()
        with torch.no_grad():
            v = self._v0.to(W_det.device, W_det.dtype).unsqueeze(-1)  # (dr, 1)
            for _ in range(n_iter):
                v = W_det @ v
                vn = v.norm()
                if vn > 0:
                    v = v / vn
            return (W_det @ v).norm()

    def forward(self, x: Tensor) -> Tensor:
        B, T, d = x.shape
        dr = self.dr
        device = x.device

        # Scale W_res to learnable spectral radius (power iteration, not eigvals)
        rho_target = torch.sigmoid(self.log_rho) * 1.5
        rho_current = self._power_iter_spectral_radius(self.W_res)
        scale = rho_target / rho_current.clamp(min=1e-6)
        W_scaled = self.W_res * scale

        # Vectorize all input-dependent projections outside the loop (1 kernel each)
        U = self.W_in(x)                                 # (B, T, dr)
        W_z_h = self.W_z.weight[:, :dr]                  # (dr, dr)
        W_z_u = self.W_z.weight[:, dr:]                  # (dr, dr)
        W_r_h = self.W_r.weight[:, :dr]
        W_r_u = self.W_r.weight[:, dr:]
        U_for_z = U @ W_z_u.T + self.W_z.bias            # (B, T, dr) — u-contribution to z
        U_for_r = U @ W_r_u.T + self.W_r.bias            # (B, T, dr) — u-contribution to r

        # Causal GRU recurrence (inherently sequential; per-step kernels minimized)
        h = torch.zeros(B, dr, device=device, dtype=x.dtype)
        outs = []
        W_scaled_T = W_scaled.T
        W_z_h_T = W_z_h.T
        W_r_h_T = W_r_h.T
        for t in range(T):
            u_t = U[:, t, :]
            z = torch.sigmoid(h @ W_z_h_T + U_for_z[:, t, :])
            r = torch.sigmoid(h @ W_r_h_T + U_for_r[:, t, :])
            h_tilde = torch.tanh((r * h) @ W_scaled_T + u_t)
            h = (1 - z) * h + z * h_tilde
            outs.append(h)

        H = torch.stack(outs, dim=1)
        return self.readout(H)


# ---------------------------------------------------------------------------
#  7. Non-Archimedean p-adic Attention  (Whitepaper Sec 2.5, Appendix A.3)
#
#     Similarity = shared prefix length in learned binary tree encoding.
#     agree(p,q) = p*q + (1-p)*(1-q)
#     sim = sum_d prod_{l=1}^d agree(path_i[l], path_j[l])
# ---------------------------------------------------------------------------

class PAdicAttention(nn.Module):
    def __init__(self, d_model: int, tree_depth: int = 5, n_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.H = n_heads
        self.dh = d_model // n_heads
        self.tree_depth = tree_depth
        self.path_proj = nn.Linear(d_model, n_heads * tree_depth)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def padic_similarity(self, path_q: Tensor, path_k: Tensor) -> Tensor:
        """Hierarchical agreement accumulation over binary-tree depth.

        sim(i,j) = sum_{d=1..D} prod_{l=1..d} agree(path_q[i,l], path_k[j,l])
        Computed as running product in a Python loop over the small depth
        dimension D (typically 5) to avoid materialising the (B,H,T,T,D)
        tensor cumprod would need — a 5x memory saving and smaller backward
        graph on DML."""
        B, H, T, D = path_q.shape
        # q: (B,H,T,D) -> q_d of shape (B,H,T,1,1) per depth
        # k: (B,H,T,D) -> k_d of shape (B,H,1,T,1) per depth
        cum = None
        sim = None
        for d in range(D):
            q_d = path_q[..., d].unsqueeze(-1)        # (B,H,T,1)
            k_d = path_k[..., d].unsqueeze(-2)        # (B,H,1,T)
            agree_d = q_d * k_d + (1 - q_d) * (1 - k_d)  # (B,H,T,T)
            cum = agree_d if cum is None else cum * agree_d
            sim = cum if sim is None else sim + cum
        return sim

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        B, T, d = x.shape
        H, dh = self.H, self.dh

        paths = torch.sigmoid(self.path_proj(x))
        paths = paths.view(B, T, H, self.tree_depth).permute(0, 2, 1, 3).contiguous()
        sim = self.padic_similarity(paths, paths) / self.tree_depth

        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            sim.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = self.drop(torch.softmax(sim, dim=-1))
        V = self.Wv(x).view(B, T, H, dh).permute(0, 2, 1, 3).contiguous()
        ctx = (attn @ V).permute(0, 2, 1, 3).contiguous().reshape(B, T, d)
        return self.Wo(ctx)


# ---------------------------------------------------------------------------
#  8. Tropical SSM — Max-Plus Linear Recurrence
#
#  State update: h_t = max(A + h_{t-1}, B + x_t)  (element-wise max-plus)
#  This is the tropical semiring analogue of a linear state-space model.
#  The max operation selects the dominant "path" at each step, giving
#  inherent sparsity and piecewise-linear dynamics.
#
#  Formally: h_t = A ⊗_T h_{t-1} ⊕_T B ⊗_T x_t
#  where ⊗_T = +, ⊕_T = max (tropical addition/multiplication).
# ---------------------------------------------------------------------------

class TropicalSSM(nn.Module):
    """Selective Tropical SSM (Mamba-style max-plus recurrence).

    Non-selective base recurrence: h_t = max(A + h_{t-1}, B_proj(x_t))
    Selective gates (input-dependent, Mamba S6 style):
      u_t = sigmoid(gate_B(x_t)) * B_proj(x_t)    # content-gated input
      out_t = sigmoid(gate_out(x_t)) * C_proj(h_t) # content-gated output

    All gates are per-token pointwise -> causality preserved. Scan is still
    the vectorized cummax form (only op with DML CPU fallback, single call).
    """
    def __init__(self, d_model: int, d_state: int = None):
        super().__init__()
        ds = d_state or d_model
        self.ds = ds
        # Transition matrix A: learnable, initialized near zero for stability
        self.A = nn.Parameter(torch.randn(ds) * 0.01)
        # Input matrix B: projects input to state space
        self.B_proj = nn.Linear(d_model, ds, bias=False)
        # Output matrix C: projects state back to model dim
        self.C_proj = nn.Linear(ds, d_model, bias=False)
        nn.init.zeros_(self.C_proj.weight)  # no-op at init

        # Selective gates (Mamba S6): content-dependent memory control
        self.gate_B = nn.Linear(d_model, ds)          # input selectivity
        self.gate_out = nn.Linear(d_model, d_model)   # output selectivity (SwiGLU-like)
        nn.init.zeros_(self.gate_B.weight)
        nn.init.constant_(self.gate_B.bias, 2.0)      # sigmoid(2.0)~0.88 -> strong init flow
        nn.init.zeros_(self.gate_out.weight)
        nn.init.constant_(self.gate_out.bias, 2.0)    # sigmoid(2.0)~0.88 -> near-identity init

        # Learnable mixing: blend tropical recurrence with direct input
        self.mix = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def forward(self, x: Tensor) -> Tensor:
        B, T, d = x.shape
        ds = self.ds
        device = x.device

        # Selective input (Mamba B_t): content-gated projection into state space
        u_raw = self.B_proj(x)                         # (B, T, ds)
        gB = torch.sigmoid(self.gate_B(x))             # (B, T, ds)
        u = gB * u_raw                                 # content-selective input

        # Vectorized max-plus recurrence via cumulative max (no Python loop).
        # Closed form: h_t = t*A_c + cummax([A_c, u_0-0*A_c, u_1-1*A_c, ...])[t+1]
        # where A_c = clamp(A, max=0) ensures non-growing (stable) state.
        A_c = self.A.clamp(max=0)  # (ds,) non-positive decay
        t_idx = torch.arange(T, device=device, dtype=x.dtype).unsqueeze(-1)  # (T, 1)
        shifted_u = u - t_idx * A_c  # (B, T, ds)
        g_init = A_c.unsqueeze(0).expand(B, -1).unsqueeze(1)  # (B, 1, ds)
        G = torch.cat([g_init, shifted_u], dim=1)  # (B, T+1, ds)
        # prefix_max: GPU-native O(log T) scan.  Replaces torch.cummax which
        # falls back to CPU on DirectML (per-step round-trip, catastrophic).
        cum_max = prefix_max(G, dim=1)  # (B, T+1, ds)
        cum_max = cum_max[:, 1:, :]   # (B, T, ds)
        t_scale = t_idx.unsqueeze(0) * A_c  # (1, T, ds)
        H = t_scale + cum_max  # (B, T, ds)

        # Selective output (Mamba C_t): content-gated readout
        out_raw = self.C_proj(H)                       # (B, T, d_model)
        gO = torch.sigmoid(self.gate_out(x))           # (B, T, d_model)
        out = gO * out_raw

        alpha = torch.sigmoid(self.mix)
        return alpha * out


# ---------------------------------------------------------------------------
#  8b. Kleene Star SSM (replaces TropicalSSM, strictly better)
#
#  Standard SSM:  h_t = max(A + h_{t-1}, u_t)  [sequential recurrence]
#  KleeneSSM:     H = A* (X) U                  [parallel matrix op]
#
#  where A* is the tropical Kleene star of the transition matrix A
#  computed via repeated squaring (log2(d_state) iterations).
#  The Kleene star encodes all-paths shortest distances in the state
#  graph, which is exactly the unrolled solution of the recurrence.
#
#  Causality: enforced by the structure of U (input-dependent state
#  contributions only flow into h_t from positions <= t). The Kleene
#  star A* is position-independent and does not introduce future info.
# ---------------------------------------------------------------------------

class KleeneSSM(nn.Module):
    """Tropical Kleene Star State Space Model.

    Replaces TropicalSSM. Strictly better in all cases:
      - Eliminates sequential Python loop (T iterations -> 1 matmul-like op)
      - Removes cummax CPU fallback on DirectML
      - Fully parallelizable -- all T positions computed simultaneously
      - Complexity: O(d_state^2 * log d_state) once, then O(T * d_state^2)
        vs O(T * d_state) sequential -- on GPU the parallelism wins.

    Mathematical basis:
        Recurrence:  h_t = max(A + h_{t-1}, B*x_t)
        Closed form: H_t = max_{s <= t} (A^*(t-s) + B*x_s)
                   = max_s ((A_star)_{t,s} + (B*x_s))
        Computed in parallel: H = A_star (max-plus matmul) U
    """

    def __init__(self, d_model: int, d_state: int = 64, n_iters: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # log2(d_state) iterations is sufficient for convergence on acyclic graphs.
        # bit_length() gives ceil(log2) + 1 for powers of two; clamp lower bound.
        min_iters = max(1, (d_state - 1).bit_length())
        self.n_iters = max(n_iters, min_iters)

        # Transition matrix A: identity-strong, cross-paths weakly negative.
        # Tropical identity: 0 on diagonal (multiplicative 1), -inf off (additive 0).
        # We use a finite -8 instead of -inf so gradients can grow links if useful.
        A_init = torch.full((d_state, d_state), -8.0)
        A_init.fill_diagonal_(0.0)
        self.A = nn.Parameter(A_init)

        # Input projection: x -> state space (B in Mamba notation)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)

        # Output projection: state -> model dimension (C in Mamba notation)
        # Zero-initialized so the SSM contributes nothing at step 0 of training.
        self.C_proj = nn.Linear(d_state, d_model, bias=False)
        nn.init.zeros_(self.C_proj.weight)

        # Input-dependent selectivity (Mamba S6-style delta).
        # Controls how much current input vs history matters per state dim.
        self.delta_proj = nn.Linear(d_model, d_state, bias=True)
        nn.init.constant_(self.delta_proj.bias, -4.0)  # softplus(-4) ~ 0.018

        # Output normalization keeps activations stable across depth.
        self.norm = nn.LayerNorm(d_model)

    def compute_kleene_star(self, A: Tensor) -> Tensor:
        """Tropical Kleene star A* via repeated squaring.

        A* = I (+) A (+) A^2 (+) A^3 (+) ...
        With repeated squaring, n_iters captures paths of length up to 2^n_iters.

        Args:
            A: (d, d) transition matrix in tropical (max-plus) semiring.
        Returns:
            A_star: (d, d) all-paths matrix.
        """
        d = A.shape[0]
        device = A.device

        # Tropical identity (0 on diagonal, very-negative off).
        I = torch.full_like(A, -1e9)
        I.fill_diagonal_(0.0)

        # Initialize: include identity (zero-length paths) and direct edges.
        result = torch.maximum(I, A)

        for _ in range(self.n_iters):
            # One squaring step:
            # (R (X) R)_ij = max_k (R_ik + R_kj)
            # Captures paths through one additional intermediate node.
            squared = (
                result.unsqueeze(-1) +   # (d, d, 1)
                result.unsqueeze(-3)     # (1, d, d)
            ).max(dim=-2).values         # (d, d)

            new_result = torch.maximum(result, squared)
            # Early exit if converged (no path improvements).
            if torch.equal(new_result, result):
                result = new_result
                break
            result = new_result

        return result

    def forward(self, x: Tensor) -> Tensor:
        """Args: x: (B, T, d_model). Returns: (B, T, d_model).

        CAUSALITY (DATA-LEAKAGE AUDIT, kleene-star branch):
            Earlier versions scaled A by ``delta.mean(dim=(0, 1))`` — a
            statistic over ALL T positions — which leaked future inputs
            into past outputs.  Fixed by:
              (a) Computing the Kleene star from input-INDEPENDENT
                  ``self.A`` (no scaling).  ``A_star`` therefore carries
                  no future-token information.
              (b) Applying per-position selectivity ONLY to the diagonal
                  decay via the closed-form prefix-max scan, which is
                  pointwise per t and therefore strictly causal.
        """
        B, T, d = x.shape

        # Per-position selectivity (Mamba S6 delta).
        delta = F.softplus(self.delta_proj(x))            # (B, T, d_state)

        # Input projection (B in Mamba notation).
        U = self.B_proj(x)                                # (B, T, d_state)

        # Per-position decay, derived from the diagonal of the unscaled A.
        # Clamped <=0 for stability; (1 - delta) gates per-position retention.
        A_diag_const = torch.diagonal(self.A).clamp(max=0.0)        # (d_state,)
        A_diag_t = (
            A_diag_const.unsqueeze(0).unsqueeze(0)
            * (1.0 - delta).clamp_min(0.05)
        )                                                            # (B, T, d_state) <=0

        # Closed form for h_t = max(a_t + h_{t-1}, u_t) with input-dependent
        # decay a_t (≤0):
        #     S_t = sum_{r=0..t} a_r          (cumsum INCLUSIVE)
        #     h_t = S_t + max_{s<=t} (u_s - S_s)
        # Verify: at s=t -> u_t - S_t + S_t = u_t. ✓
        #         at s<t -> (u_s - S_s) + S_t = u_s + sum_{r=s+1..t} a_r. ✓
        S = A_diag_t.cumsum(dim=1)                         # (B, T, d_state)
        shifted = U - S                                    # (B, T, d_state)
        cum = prefix_max(shifted, dim=1)                   # (B, T, d_state)
        H_diag = S + cum                                   # (B, T, d_state)

        # Kleene star uses input-INDEPENDENT A (no future-leak through mixing).
        A_star = self.compute_kleene_star(self.A)          # (d_state, d_state)

        # Apply Kleene-star mixing across state dimensions (no time mix).
        # H_mix[t, i] = max_j (A_star[i, j] + H_diag[t, j])
        H_mix = (
            A_star.unsqueeze(0).unsqueeze(0) +             # (1, 1, d_state, d_state)
            H_diag.unsqueeze(-2)                           # (B, T, 1, d_state)
        ).max(dim=-1).values                               # (B, T, d_state)

        # Output projection + norm.
        out = self.norm(self.C_proj(H_mix))                # (B, T, d_model)
        return out


# ---------------------------------------------------------------------------
#  8c. Kleene Star Attention (Pro and above only)
#
#  Standard tropical attention: score(i,j) = max_c (Q_ic + K_jc)  [1-hop]
#  Kleene attention:            score(i,j) = max over all paths i->...->j
#
#  Captures multi-hop transitive dependencies in a single layer that
#  would otherwise require k stacked attention layers.
#
#  Causality: causal mask is applied BEFORE Kleene star computation.
#  The Kleene star of a lower-triangular matrix is lower-triangular.
# ---------------------------------------------------------------------------

class KleeneAttention(nn.Module):
    """Tropical Kleene Star Attention.

    Per-layer cost: O(T * k^2 * d * n_iters) vs O(T * k * d) for sparse tropical.
    Captures up to 2^n_iters-hop dependencies.

    Pro-tier component (set use_kleene_attention=True in ModelConfig).
    """

    def __init__(self, d_model: int, n_heads: int, top_k: int = 16,
                 n_iters: int = 3, max_seq_len: int = 4096,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k
        self.n_iters = n_iters

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.Wq.weight, gain=0.5)
        nn.init.xavier_uniform_(self.Wk.weight, gain=0.5)
        nn.init.xavier_uniform_(self.Wv.weight, gain=0.5)

        # QK-norm (stabilizes scores, esp. with multi-hop accumulation).
        self.q_norm = nn.LayerNorm(self.d_head, elementwise_affine=True)
        self.k_norm = nn.LayerNorm(self.d_head, elementwise_affine=True)

        # RoPE on Q/K (same as TropicalAttention).
        self.rope = RotaryPositionEmbedding(self.d_head, max_seq_len=max_seq_len)
        # ALiBi: per-head linear bias on attention scores.
        self.register_buffer("alibi_slopes", build_alibi_slopes(n_heads))

        # Per-head per-hop discount: prevents trivially long paths from dominating.
        # Initialized small-negative so each extra hop costs a bit.
        self.hop_discount = nn.Parameter(torch.full((n_heads,), -0.5))

        # Maslov softening parameter (matches TropicalAttention).
        self.register_buffer("maslov_h", torch.tensor(1.0))

    def kleene_star_dense(self, S: Tensor) -> Tensor:
        """Dense tropical Kleene star of attention scores.

        S: (B, H, T, T) causal-masked attention scores.
        Returns: (B, H, T, T) all-paths score matrix.

        Complexity: O(T^3 * n_iters) per (batch, head).
        Recommended only for short context (T <= 1024). For longer contexts
        with KleeneAttention, prefer to apply per-window before any later
        full-T mixing.
        """
        B, H, T, _ = S.shape
        # Per-head per-hop discount (broadcast over batch/T).
        discount = self.hop_discount.view(1, H, 1, 1)
        result = S.clone()

        for it in range(self.n_iters):
            # Each squaring extends max path length by 2x.
            # Apply per-iteration discount so longer paths cost more.
            iter_disc = discount * (it + 1)

            # Two-hop: (R (X) R)_ij = max_k (R_ik + R_kj + iter_disc)
            two_hop = (
                result.unsqueeze(-1) +    # (B, H, T, T, 1)  rows
                result.unsqueeze(-3)      # (B, H, 1, T, T)  cols
            ).max(dim=-2).values + iter_disc  # (B, H, T, T)

            new_result = torch.maximum(result, two_hop)
            result = new_result

        return result

    def forward(self, x: Tensor, gist_theta=None, gist_mag=None,
                gist_weights=None) -> Tensor:
        """Args: x: (B, T, d_model). Returns: (B, T, d_model).

        Extra gist_* args accepted but unused (kept for interface
        compatibility with TropicalAttention call sites).
        """
        B, T, d = x.shape
        H, dh = self.n_heads, self.d_head

        # Q, K, V projections.
        Q = self.Wq(x).view(B, T, H, dh).permute(0, 2, 1, 3)  # (B, H, T, dh)
        K = self.Wk(x).view(B, T, H, dh).permute(0, 2, 1, 3)
        V = self.Wv(x).view(B, T, H, dh).permute(0, 2, 1, 3)

        # QK-norm + RoPE on Q, K.
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        Q, K = self.rope(Q, K, offset=0)

        # Tropical inner product (Maslov-softened logsumexp).
        # score[b,h,i,j] = h * logsumexp((Q_i + K_j)/h)
        h = self.maslov_h
        raw = Q.unsqueeze(-2) + K.unsqueeze(-3)               # (B, H, T, T, dh)
        S = h * torch.logsumexp(raw / h, dim=-1)              # (B, H, T, T)
        del raw

        # ALiBi distance bias.
        pos = torch.arange(T, device=S.device, dtype=S.dtype)
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()    # (T, T)
        slopes = self.alibi_slopes.to(S.device, S.dtype).view(1, H, 1, 1)
        S = S - slopes * dist.unsqueeze(0).unsqueeze(0)

        # CRITICAL: apply causal mask BEFORE Kleene star.
        # On a lower-triangular matrix, the Kleene star is lower-triangular,
        # so no future positions can leak in via multi-hop paths.
        causal_mask = torch.triu(
            torch.full((T, T), -1e9, device=S.device, dtype=S.dtype),
            diagonal=1
        )
        S = S + causal_mask.unsqueeze(0).unsqueeze(0)

        # Multi-hop Kleene star score propagation.
        S_star = self.kleene_star_dense(S)                    # (B, H, T, T)

        # Top-k sparse selection on Kleene scores (causal preserved).
        topk_v, topk_idx = S_star.topk(min(self.top_k, T), dim=-1)
        thr = topk_v[..., -1:].detach()
        S_star = S_star.masked_fill(S_star < thr, -1e9)

        # Softmax (differentiable selection).
        attn = self.drop(torch.softmax(S_star, dim=-1))       # (B, H, T, T)

        # Aggregate values.
        out = attn @ V                                        # (B, H, T, dh)
        out = out.permute(0, 2, 1, 3).reshape(B, T, d)
        return self.Wo(out)


# ---------------------------------------------------------------------------
#  Factory functions: instantiate the right component class for a tier.
#  Use these instead of direct class names so code can switch tiers via
#  a single config flag.
# ---------------------------------------------------------------------------

def build_attention(config) -> nn.Module:
    """Returns the correct attention class for this model tier."""
    if getattr(config, "use_kleene_attention", False):
        return KleeneAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            top_k=config.top_k,
            n_iters=getattr(config, "kleene_attn_iters", 3),
            max_seq_len=max(4096, config.context_len * 8),
            dropout=getattr(config, "dropout", 0.0),
        )
    else:
        return TropicalAttention(
            d_model=config.d_model,
            top_k=config.top_k,
            n_heads=config.n_heads,
            dropout=getattr(config, "dropout", 0.0),
            max_seq_len=max(4096, config.context_len * 8),
            use_pacs=getattr(config, "use_padic_context_scaling", False),
            training_ctx=config.context_len,
        )


def build_ssm(config) -> nn.Module:
    """Returns the SSM class for this model tier.

    All tiers default to KleeneSSM. Set use_kleene_ssm=False to fall back
    to TropicalSSM (debugging / ablation only).
    """
    if getattr(config, "use_kleene_ssm", True):
        return KleeneSSM(
            d_model=config.d_model,
            d_state=getattr(config, "kleene_ssm_d_state", 64),
            n_iters=getattr(config, "kleene_ssm_iters", 4),
        )
    else:
        return TropicalSSM(d_model=config.d_model)


# ---------------------------------------------------------------------------
#  9. Persistent Cross-Window Memory (Tropical Transformer-XL)
#
#  Caches K/V from the previous window and prepends them to the current
#  window's K/V in attention. This extends effective context beyond the
#  window size without recomputation.
#
#  Memory is stored as a detached buffer (no gradient flow across windows)
#  to prevent gradient explosion in long sequences.
# ---------------------------------------------------------------------------

class PersistentCrossWindowMemory(nn.Module):
    """Caches previous window K/V for cross-window attention.

    OPT-IN: disabled by default. Only enable for SEQUENTIAL autoregressive
    generation where the current window semantically continues the prior
    window's context. For random-batch evaluation (which samples unrelated
    positions), leaving this enabled produces incorrect attention (stale
    context) AND severely slows eval by forcing the O(T^2) chunked path
    when T_kv doubles past the 24 MB fast-path threshold.

    Usage:
      model.set_cross_window_enabled(True)   # before sequential generation
      model.set_cross_window_enabled(False)  # before random-batch eval
    """
    def __init__(self, d_model: int, n_heads: int, max_cached_len: int = 256):
        super().__init__()
        self.H = n_heads
        self.dh = d_model // n_heads
        self.max_cached_len = max_cached_len
        # Buffers: (max_cached_len, H, dh) — no batch dim, shared across batch
        self.register_buffer("cached_k", torch.zeros(0))
        self.register_buffer("cached_v", torch.zeros(0))
        self._has_cache = False
        self.enabled = False  # opt-in; default off to preserve fast random-batch eval

    def cache_kv(self, k: Tensor, v: Tensor):
        """Store K/V from current window for next window.
        k, v: (B, H, T, dh) — we take the first batch element (all same in LM).
        """
        with torch.no_grad():
            # Take last max_cached_len tokens
            T = k.shape[2]
            start = max(0, T - self.max_cached_len)
            self.cached_k = k[0, :, start:, :].detach().clone()  # (H, T', dh)
            self.cached_v = v[0, :, start:, :].detach().clone()
            self._has_cache = True

    def get_cached_kv(self, B: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Return cached K/V expanded to batch size, or None if no cache."""
        if not self._has_cache or self.cached_k.numel() == 0:
            return None, None
        # Expand (H, T', dh) -> (B, H, T', dh)
        k = self.cached_k.unsqueeze(0).expand(B, -1, -1, -1)
        v = self.cached_v.unsqueeze(0).expand(B, -1, -1, -1)
        return k, v

    def reset(self):
        self.cached_k = torch.zeros(0, device=self.cached_k.device)
        self.cached_v = torch.zeros(0, device=self.cached_v.device)
        self._has_cache = False


# ---------------------------------------------------------------------------
#  TSRN Block (reusable per-scale block for depth scaling)
# ---------------------------------------------------------------------------

class TSRNBlock(nn.Module):
    """One TSRN processing block (per whitepaper Sec 3.2)."""

    def __init__(self, d_model: int, top_k: int, n_heads: int,
                 sheaf_window: int, mem_depth: int,
                 use_reservoir: bool, use_padic_attn: bool,
                 use_memory: bool, use_tropical_ssm: bool = False,
                 dropout: float = 0.1,
                 use_pacs: bool = False, training_ctx: int = 512):
        super().__init__()

        # Tropical attention
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = TropicalAttention(d_model, top_k=top_k, n_heads=n_heads,
                                       dropout=dropout, use_pacs=use_pacs, training_ctx=training_ctx)

        # Sheaf diffusion
        self.ln_sheaf = nn.LayerNorm(d_model)
        self.sheaf = SheafDiffusion(d_model, window=sheaf_window, dropout=dropout)

        # Echo State Reservoir (optional, Scale 1 only per whitepaper)
        self.use_reservoir = use_reservoir
        if use_reservoir:
            self.ln_res = nn.LayerNorm(d_model)
            self.reservoir = EchoStateReservoir(d_model, d_reservoir=d_model // 2)

        # Tropical SSM (optional, parallel to reservoir)
        self.use_tropical_ssm = use_tropical_ssm
        if use_tropical_ssm:
            self.ln_ssm = nn.LayerNorm(d_model)
            self.tropical_ssm = TropicalSSM(d_model)

        # Clifford FFN
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = CliffordFFN(d_model, dropout=dropout)

        # p-adic Memory (optional, Scale 1 only per whitepaper)
        self.use_memory = use_memory
        if use_memory:
            self.ln_mem = nn.LayerNorm(d_model)
            self.mem = PAdicMemory(d_model, depth=mem_depth)

        # p-adic Attention (optional, Scale 2 only per whitepaper)
        self.use_padic_attn = use_padic_attn
        if use_padic_attn:
            self.ln_pa = nn.LayerNorm(d_model)
            self.pa = PAdicAttention(d_model, tree_depth=5, n_heads=n_heads,
                                      dropout=dropout)

    def forward(self, x: Tensor, inference_ctx: Optional[int] = None) -> Tensor:
        # ``inference_ctx`` is only consumed by TropicalAttention's PaCS
        # branch.  All other sub-modules ignore it.
        x = x + self.attn(self.ln_attn(x), causal=True, inference_ctx=inference_ctx)
        x = x + self.sheaf(self.ln_sheaf(x))
        if self.use_reservoir:
            x = x + self.reservoir(self.ln_res(x))
        if self.use_tropical_ssm:
            x = x + self.tropical_ssm(self.ln_ssm(x))
        x = x + self.ffn(self.ln_ffn(x))
        if self.use_memory:
            x = x + self.mem(self.ln_mem(x))
        if self.use_padic_attn:
            x = x + self.pa(self.ln_pa(x))
        return x


# ---------------------------------------------------------------------------
#  Full TSRN Model  (Whitepaper Sec 3)
# ---------------------------------------------------------------------------

class TSRN(nn.Module):
    """
    Tropical Sheaf Renormalization Network.

    Two-scale architecture per whitepaper Sec 3.1:
      Scale 1: n_blocks x TSRNBlock(reservoir=True, memory=True)
      RG coarse-grain: T -> T/2
      Scale 2: n_blocks x TSRNBlock(padic_attn=True)
      Upsample & fuse -> logits
    """

    def __init__(self, vocab: int, d_model: int, context_len: int,
                 gradient_checkpoint: bool = False,
                 n_blocks: int = 1, top_k: int = 8, n_heads: int = 4,
                 mem_depth: int = 6, sheaf_window: int = 3,
                 dropout: float = 0.1,
                 use_hyperbolic: bool = False,
                 use_padic_pe: bool = False,
                 use_pacs: bool = False,
                 use_hyperbolic_memory: bool = False,
                 inference_ctx: int = None):
        super().__init__()
        self.ctx = context_len
        self.d = d_model
        self.use_hyperbolic = use_hyperbolic
        self.use_padic_pe = use_padic_pe
        self.use_pacs = use_pacs
        self.inference_ctx = inference_ctx or context_len

        # Embeddings: Hyperbolic or Euclidean
        if use_hyperbolic:
            self.embed = HyperbolicEmbedding(vocab, d_model, init_strategy="frequency")
            # Initialize with frequency ranks (synthetic, would need real data)
            token_ranks = torch.arange(1, vocab + 1)
            self.embed.initialize_embeddings(token_ranks)
        else:
            self.embed = nn.Embedding(vocab, d_model)
            nn.init.normal_(self.embed.weight, std=0.02)

        # P-adic Harmonic PE (v2.0).  Additive on token embeddings.
        # Zero-init projection so the layer starts as a no-op.
        if use_padic_pe:
            self.padic_pe = PAdicHarmonicPE(
                max_T=max(context_len, 1024),
                d_model=d_model,
                p=2,
                depth=max(4, int(math.ceil(math.log2(max(context_len, 2))))),
                dct_K=min(64, max(8, d_model // 8)),
            )
        else:
            self.padic_pe = None

        # Scale 1 blocks (reservoir + SSM + memory, no p-adic attn)
        self.s1_blocks = nn.ModuleList([
            TSRNBlock(d_model, top_k=top_k, n_heads=n_heads,
                      sheaf_window=sheaf_window, mem_depth=mem_depth,
                      use_reservoir=(i == 0),  # reservoir only in first block
                      use_padic_attn=False, use_memory=True,
                      use_tropical_ssm=(i == 0),  # SSM in first block
                      dropout=dropout,
                      use_pacs=use_pacs, training_ctx=context_len)
            for i in range(n_blocks)
        ])

        # RG coarse-grain
        self.rg_pool = RGPool(d_model)

        # Scale 2 blocks (p-adic attn, no reservoir/memory)
        self.s2_blocks = nn.ModuleList([
            TSRNBlock(d_model, top_k=max(2, top_k // 2), n_heads=n_heads,
                      sheaf_window=sheaf_window, mem_depth=mem_depth,
                      use_reservoir=False,
                      use_padic_attn=(i == n_blocks - 1),  # p-adic attn in last block
                      use_memory=False, dropout=dropout,
                      use_pacs=use_pacs, training_ctx=context_len)
            for i in range(n_blocks)
        ])

        # Hyperbolic memory (v2.0)
        self.use_hyperbolic_memory = use_hyperbolic_memory
        if use_hyperbolic_memory:
            self.memory = HyperbolicMemoryLayer(
                d_model=d_model, capacity=10000, enable_training=True
            )

        # Learnable gated fusion: replaces hard-coded 0.5 blend
        self.fuse_gate = nn.Linear(2 * d_model, d_model, bias=True)
        nn.init.zeros_(self.fuse_gate.weight)
        nn.init.zeros_(self.fuse_gate.bias)  # starts as 0.5 via sigmoid(0)

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        # Weight tying.  HyperbolicEmbedding stores its table as `.embeddings`
        # (a Parameter), not `.weight`, so we tie head.weight to whichever
        # parameter the embedding actually exposes.
        if use_hyperbolic:
            self.head.weight = self.embed.embeddings
        else:
            self.head.weight = self.embed.weight

        self._init_weights()
        print(f"  TSRN      : {self.count_params():,} parameters")

        self.gradient_checkpoint = gradient_checkpoint

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2 and "embed" not in name \
               and "W_res" not in name and "leaf" not in name \
               and "router" not in name and "path" not in name:
                nn.init.xavier_uniform_(p, gain=0.5)

    def _retie_weights(self):
        """Re-establish weight tying after .to(device).
        DirectML's .to() can break parameter aliasing, causing head.weight
        and embed.weight to become separate tensors. This re-ties them."""
        embed_param = (
            self.embed.embeddings if self.use_hyperbolic else self.embed.weight
        )
        if self.head.weight is not embed_param:
            self.head.weight = embed_param

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result._retie_weights()
        return result

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_cross_window(self):
        """Reset cross-window memory in all attention layers."""
        for block in self.s1_blocks:
            block.attn.cross_window_mem.reset()
        for block in self.s2_blocks:
            block.attn.cross_window_mem.reset()

    def set_cross_window_enabled(self, enabled: bool):
        """Enable/disable cross-window K/V cache across all attention layers.

        Default is disabled. Enable only for SEQUENTIAL autoregressive
        generation. Random-batch eval should leave this off (caches stale
        context and forces the slow O(T^2) chunked attention path)."""
        for block in self.s1_blocks:
            block.attn.cross_window_mem.enabled = enabled
        for block in self.s2_blocks:
            block.attn.cross_window_mem.enabled = enabled
        if not enabled:
            self.reset_cross_window()

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None):
        B, T = idx.shape
        x = self.embed(idx)

        # P-adic Harmonic PE (additive, broadcast over batch).  Zero-init
        # projection means no effect at step 0; model learns the frequency
        # content to inject on top of RoPE/ALiBi.  Strictly causal:
        # PE[t] depends only on t.
        if self.padic_pe is not None:
            pe = self.padic_pe(T, x.device, x.dtype)        # (T, d)
            x = x + pe.unsqueeze(0)

        # Hyperbolic memory retrieval (v2.0).  We do NOT alter the sequence
        # length — instead we additively inject a retrieved-memory bias
        # broadcast over the time axis.  This keeps every downstream shape
        # invariant and preserves causality (memory comes from the past).
        if self.use_hyperbolic_memory and not self.training:
            retrieved, _ = self.memory(x[:, -1:, :], top_k=5)
            if retrieved is not None and retrieved.shape[1] > 0:
                mem_emb = retrieved.mean(dim=1, keepdim=True)  # (B, 1, d)
                x = x + mem_emb

        # Determine inference_ctx once: only enabled at eval time when PaCS
        # is on AND the user requested an extended context.
        ictx = (
            self.inference_ctx
            if (self.use_pacs and not self.training
                and self.inference_ctx > self.ctx)
            else None
        )

        # Scale 1.  PaCS, if enabled, is applied INSIDE TropicalAttention
        # via TropicalAttention.forward(inference_ctx=...).
        for block in self.s1_blocks:
            if self.gradient_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, ictx, use_reentrant=False)
            else:
                x = block(x, inference_ctx=ictx)

        # RG coarse-grain
        xc = self.rg_pool(x)

        # Scale 2
        for block in self.s2_blocks:
            if self.gradient_checkpoint and self.training:
                xc = torch.utils.checkpoint.checkpoint(
                    block, xc, ictx, use_reentrant=False)
            else:
                xc = block(xc, inference_ctx=ictx)

        # Upsample & fuse with learnable gate
        xc_up = xc.repeat_interleave(2, dim=1).contiguous()
        # Handle odd T: xc_up may be shorter than x after RGPool truncation
        if xc_up.size(1) < T:
            xc_up = F.pad(xc_up, (0, 0, 0, T - xc_up.size(1)))
        else:
            xc_up = xc_up[:, :T, :]
        # Learnable gated fusion: gate = sigma(W_g [x; xc_up])
        # At init, W_g=0,b=0 => gate=0.5, recovering the original blend.
        gate = torch.sigmoid(self.fuse_gate(torch.cat([x, xc_up], dim=-1)))
        x = x + gate * xc_up

        # Logits
        logits = self.head(self.ln_f(x))

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
#  Vanilla Transformer baseline
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True,
                                          dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(1)
        # DirectML fix: use boolean causal mask (float mask + is_causal causes NaN)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        xn = self.ln1(x)
        a, _ = self.attn(xn, xn, xn, attn_mask=mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class VanillaTransformer(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, context_len: int, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(context_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        self._init()
        print(f"  Transformer : {self.count_params():,} parameters")

    def _init(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.5)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.embed(idx) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
#  Training utilities
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    if step > total:
        return lr_min
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, dataset, device: torch.device,
             n_batches: int = 30, batch_size: int = 32,
             split: str = "val") -> Tuple[float, float, float]:
    """Returns (avg_loss, perplexity, bits_per_char)."""
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = dataset.batch(split, batch_size, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    avg = total / n_batches
    ppl = math.exp(min(avg, 20))
    bpc = avg / math.log(2)
    return avg, ppl, bpc


@torch.no_grad()
def evaluate_sequential(model, dataset, device: torch.device,
                        batch_size: int = 16,
                        split: str = "test") -> Tuple[float, float, float]:
    """Full sequential evaluation over the ENTIRE split (no random sampling).

    Uses non-overlapping windows of size ctx to deterministically cover
    every byte in the split. This is the standard protocol used by
    Transformer-XL, SHA-RNN, and Al-Rfou (2019) for enwik8 BPC reporting.

    Returns (avg_loss, perplexity, bits_per_char).
    """
    model.eval()
    if split == "test" and dataset.test is not None:
        data = dataset.test
    elif split == "val":
        data = dataset.val
    else:
        data = dataset.train
    ctx = dataset.ctx
    N = len(data) - 1  # last byte has no target

    total_loss = 0.0
    total_tokens = 0
    # Non-overlapping windows: [0, ctx), [ctx, 2*ctx), ...
    starts = list(range(0, N - ctx, ctx))
    n_windows = len(starts)

    for batch_start in range(0, n_windows, batch_size):
        batch_idx = starts[batch_start : batch_start + batch_size]
        x = torch.stack([data[i : i + ctx] for i in batch_idx]).to(device)
        y = torch.stack([data[i + 1 : i + ctx + 1] for i in batch_idx]).to(device)
        _, loss = model(x, y)
        # loss is mean over (B * ctx) tokens; weight by actual count
        B_actual = len(batch_idx)
        total_loss += loss.item() * B_actual * ctx
        total_tokens += B_actual * ctx

    avg = total_loss / total_tokens
    ppl = math.exp(min(avg, 20))
    bpc = avg / math.log(2)
    model.train()
    return avg, ppl, bpc


def train_model(model, dataset, device: torch.device,
                n_steps: int, batch_size: int, lr_max: float,
                lr_warmup: int, label: str, eval_every: int = 200,
                weight_decay: float = 0.1) -> List[Dict]:
    model.to(device)
    model.train()

    decay = {p for n, p in model.named_parameters()
             if p.requires_grad and p.dim() >= 2}
    no_decay = {p for p in model.parameters()
                if p.requires_grad and p not in decay}
    optimizer = torch.optim.AdamW([
        {"params": list(decay), "weight_decay": weight_decay},
        {"params": list(no_decay), "weight_decay": 0.0},
    ], lr=lr_max, betas=(0.9, 0.95))

    log = []
    t0 = time.time()

    print(f"\n{'='*72}")
    print(f"  Training: {label}   ({model.count_params():,} params)")
    print(f"  Steps: {n_steps}  |  Batch: {batch_size}  |  LR: {lr_max}")
    print(f"{'='*72}")
    print(f"{'Step':>6}  {'TrLoss':>10}  {'TrPPL':>10}  "
          f"{'ValLoss':>9}  {'ValPPL':>9}  {'ValBPC':>8}  {'GNorm':>7}  {'ms/step':>8}")
    print(f"{'-'*80}")

    for step in range(1, n_steps + 1):
        lr = get_lr(step, lr_warmup, n_steps, lr_max, lr_max * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = dataset.batch("train", batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == 1:
            val_loss, val_ppl, val_bpc = evaluate(model, dataset, device,
                                                    batch_size=min(batch_size, 32))
            tr_ppl = math.exp(min(loss.item(), 20))
            tr_bpc = loss.item() / math.log(2)
            elapsed = time.time() - t0
            ms_step = elapsed / step * 1000
            print(f"{step:>6}  {loss.item():>10.4f}  {tr_ppl:>10.2f}  "
                  f"{val_loss:>9.4f}  {val_ppl:>9.2f}  {val_bpc:>8.4f}  "
                  f"{float(gnorm):>7.3f}  {ms_step:>7.1f}ms")
            log.append({
                "step": step,
                "train_loss": round(loss.item(), 5),
                "train_ppl": round(tr_ppl, 3),
                "train_bpc": round(tr_bpc, 4),
                "val_loss": round(val_loss, 5),
                "val_ppl": round(val_ppl, 3),
                "val_bpc": round(val_bpc, 4),
                "grad_norm": round(float(gnorm), 4),
                "lr": round(lr, 6),
                "time_s": round(elapsed, 1),
            })

    print(f"{'-'*80}")
    if log:
        print(f"  Final val PPL: {log[-1]['val_ppl']:.3f}  |  BPC: {log[-1]['val_bpc']:.4f}")
    return log


# ---------------------------------------------------------------------------
#  Throughput benchmark
# ---------------------------------------------------------------------------

def benchmark_throughput(model, dataset, device: torch.device,
                         batch_sizes: List[int], n_warmup: int = 5,
                         n_timed: int = 20) -> Dict:
    model.eval()
    results = {}
    T = dataset.ctx

    with torch.no_grad():
        for B in batch_sizes:
            try:
                x, y = dataset.batch("val", B, device)
                for _ in range(n_warmup):
                    _, _ = model(x, y)
                device_sync(device)
                t0 = time.perf_counter()
                for _ in range(n_timed):
                    _, _ = model(x, y)
                device_sync(device)
                elapsed = time.perf_counter() - t0
                ms_batch = elapsed / n_timed * 1000
                tok_sec = B * T * n_timed / elapsed
                results[B] = {
                    "ms_per_batch": round(ms_batch, 2),
                    "tokens_per_sec": round(tok_sec, 0),
                }
                print(f"    B={B:>4}  {ms_batch:>8.2f} ms/batch  "
                      f"{tok_sec:>10,.0f} tok/s")
            except Exception as e:
                print(f"    B={B:>4}  Error: {e}")
                break

    model.train()
    return results


# ---------------------------------------------------------------------------
#  Gradient verification
# ---------------------------------------------------------------------------

def verify_gradients(model, dataset: CharDataset, device: torch.device) -> Dict:
    model.train()
    x, y = dataset.batch("train", 4, device)
    _, loss = model(x, y)
    loss.backward()

    results = {}
    dead = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                dead.append(name)
                results[name] = "NO GRADIENT"
            else:
                gnorm = p.grad.norm().item()
                results[name] = round(gnorm, 6)
                if gnorm == 0.0:
                    dead.append(name)

    model.zero_grad(set_to_none=True)
    n_alive = len(results) - len(dead)
    print(f"    Params with gradients : {n_alive}/{len(results)}")
    if dead:
        print(f"    !! Dead parameters    : {dead[:5]}{'...' if len(dead)>5 else ''}")
    else:
        print(f"    All parameters receive gradients OK")
    return {"param_grad_norms": results, "dead_params": dead}


# ---------------------------------------------------------------------------
#  Ablation suite  (Whitepaper Sec 6.3)
# ---------------------------------------------------------------------------

class TSRNAblation(TSRN):
    """TSRN with selectable component removal for ablation study."""

    def __init__(self, vocab, d_model, context_len, n_blocks=1, top_k=8,
                 n_heads=4, mem_depth=6, sheaf_window=3, dropout=0.1,
                 ablate=None):
        # Build full model first
        super().__init__(vocab, d_model, context_len, n_blocks, top_k,
                         n_heads, mem_depth, sheaf_window, dropout)
        self.ablate = ablate or set()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx)

        # Scale 1
        for block in self.s1_blocks:
            x = x + block.attn(block.ln_attn(x))
            if "sheaf" not in self.ablate:
                x = x + block.sheaf(block.ln_sheaf(x))
            if block.use_reservoir and "reservoir" not in self.ablate:
                x = x + block.reservoir(block.ln_res(x))
            if block.use_tropical_ssm and "tropical_ssm" not in self.ablate:
                x = x + block.tropical_ssm(block.ln_ssm(x))
            x = x + block.ffn(block.ln_ffn(x))
            if block.use_memory and "memory" not in self.ablate:
                x = x + block.mem(block.ln_mem(x))

        # RG coarse-grain
        if "rg" not in self.ablate:
            xc = self.rg_pool(x)
            for block in self.s2_blocks:
                xc = xc + block.attn(block.ln_attn(xc))
                if "sheaf" not in self.ablate:
                    xc = xc + block.sheaf(block.ln_sheaf(xc))
                xc = xc + block.ffn(block.ln_ffn(xc))
                if block.use_padic_attn and "padic_attn" not in self.ablate:
                    xc = xc + block.pa(block.ln_pa(xc))
            xc_up = xc.repeat_interleave(2, dim=1)[:, :T, :]
            # Use inherited learnable gate
            gate = torch.sigmoid(self.fuse_gate(torch.cat([x, xc_up], dim=-1)))
            x = x + gate * xc_up

        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def run_ablation_suite(vocab, d_model, context_len, dataset, device,
                       n_steps, batch_size, lr, n_blocks=1, top_k=8,
                       n_heads=4, mem_depth=6, dropout=0.1) -> Dict:
    """Run all ablation variants per whitepaper Sec 6.3."""
    ablations = {
        "TSRN_full": set(),
        "no_reservoir": {"reservoir"},
        "no_sheaf": {"sheaf"},
        "no_rg_pool": {"rg"},
        "no_padic_mem": {"memory"},
        "no_padic_attn": {"padic_attn"},
        "no_tropical_ssm": {"tropical_ssm"},
    }

    results = {}
    for name, ablate_set in ablations.items():
        print(f"\n  -- Ablation: {name} (removing: {ablate_set or 'nothing'}) --")
        torch.manual_seed(42)
        model = TSRNAblation(
            vocab=vocab, d_model=d_model, context_len=context_len,
            n_blocks=n_blocks, top_k=top_k, n_heads=n_heads,
            mem_depth=mem_depth, dropout=dropout, ablate=ablate_set,
        )
        log = train_model(
            model, dataset, device,
            n_steps=n_steps, batch_size=batch_size, lr_max=lr,
            lr_warmup=n_steps // 10, label=name,
            eval_every=max(1, n_steps // 5),
        )
        results[name] = log
        model.cpu()
        del model

    return results


# ---------------------------------------------------------------------------
#  Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, dataset: CharDataset, device: torch.device,
             prompt: str = "The king ", n_tokens: int = 200,
             temperature: float = 0.8, top_p: float = 0.9) -> str:
    model.eval()
    ids = [dataset.stoi.get(c, 0) for c in prompt]
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    T = dataset.ctx

    for _ in range(n_tokens):
        idx_cond = ids[:, -T:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        next_id = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]  # scalar or [1]
        next_id = next_id.view(1, 1)  # ensure [1, 1]
        ids = torch.cat([ids, next_id], dim=1)

    model.train()
    return dataset.decode(ids[0].tolist())


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

PRESETS = {
    "2m": {
        "d_model": 256,
        "context": 256,
        "n_blocks": 1,
        "n_heads": 4,
        "top_k": 16,
        "mem_depth": 6,
        "n_layers_tf": 4,
        "steps": 3000,
        "batch": 32,
        "lr": 3e-4,
        "ablation_steps": 500,
    },
    "50m": {
        "d_model": 512,
        "context": 256,
        "n_blocks": 3,
        "n_heads": 8,
        "top_k": 16,
        "mem_depth": 7,
        "n_layers_tf": 8,
        "steps": 5000,
        "batch": 8,
        "lr": 2e-4,
        "ablation_steps": 800,
    },
    "quick": {
        "d_model": 128,
        "context": 64,
        "n_blocks": 1,
        "n_heads": 2,
        "top_k": 8,
        "mem_depth": 5,
        "n_layers_tf": 2,
        "steps": 200,
        "batch": 16,
        "lr": 3e-4,
        "ablation_steps": 100,
    },
}


def main():
    parser = argparse.ArgumentParser(description="TSRN DirectML Validation")
    parser.add_argument("--preset", default="2m", choices=["2m", "50m", "quick"],
                        help="Parameter preset (default: 2m)")
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic", "wikitext2", "wikitext103", "enwik8"],
                        help="Dataset to use")
    parser.add_argument("--data", default="data/tsrn_synthetic.txt",
                        help="Path for synthetic data file")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_blocks", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--mem_depth", type=int, default=None)
    parser.add_argument("--no_ablation", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Override to quick preset")
    parser.add_argument("--output", default="tsrn_results.json")
    args = parser.parse_args()

    if args.quick:
        args.preset = "quick"

    # Load preset, allow overrides
    cfg = dict(PRESETS[args.preset])
    if args.steps is not None: cfg["steps"] = args.steps
    if args.d_model is not None: cfg["d_model"] = args.d_model
    if args.context is not None: cfg["context"] = args.context
    if args.batch is not None: cfg["batch"] = args.batch
    if args.lr is not None: cfg["lr"] = args.lr
    if args.n_blocks is not None: cfg["n_blocks"] = args.n_blocks
    if args.top_k is not None: cfg["top_k"] = args.top_k
    if args.mem_depth is not None: cfg["mem_depth"] = args.mem_depth

    d = cfg["d_model"]
    ctx = cfg["context"]
    n_blocks = cfg["n_blocks"]
    n_heads = cfg.get("n_heads", max(4, d // 64))
    top_k = cfg["top_k"]
    mem_depth = cfg["mem_depth"]
    n_steps = cfg["steps"]
    batch_size = cfg["batch"]
    lr = cfg["lr"]
    n_layers_tf = cfg.get("n_layers_tf", 4)
    ablation_steps = cfg.get("ablation_steps", 500)
    d_ff = d * 4
    dropout = 0.1

    print("\n" + "=" * 72)
    print("  TSRN -- Tropical Sheaf Renormalization Network")
    print("  DirectML GPU Validation Suite")
    print("=" * 72)
    device = detect_device()

    print(f"\n  Preset    : {args.preset}")
    print(f"  d_model   : {d}")
    print(f"  context   : {ctx}")
    print(f"  n_blocks  : {n_blocks}")
    print(f"  n_heads   : {n_heads}")
    print(f"  top_k     : {top_k}")
    print(f"  mem_depth : {mem_depth} ({2**mem_depth} slots)")
    print(f"  steps     : {n_steps}")
    print(f"  batch     : {batch_size}")
    print(f"  lr        : {lr}")

    # -- Dataset --------------------------------------------------------
    print(f"\n-- Dataset: {args.dataset} {'-'*50}")
    if args.dataset == "wikitext103":
        dataset = load_wikitext103(context_len=ctx)
    elif args.dataset == "wikitext2":
        dataset = load_wikitext2(context_len=ctx)
    elif args.dataset == "enwik8":
        dataset = load_enwik8(context_len=ctx)
    else:
        if not Path(args.data).exists():
            generate_synthetic_data(args.data)
        dataset = CharDataset(args.data, context_len=ctx)

    V = dataset.vocab_sz

    # -- Instantiate models ---------------------------------------------
    print(f"\n-- Models {'-'*59}")
    torch.manual_seed(42)
    tsrn = TSRN(vocab=V, d_model=d, context_len=ctx, n_blocks=n_blocks,
                top_k=top_k, n_heads=n_heads, mem_depth=mem_depth,
                dropout=dropout)

    torch.manual_seed(42)
    transformer = VanillaTransformer(
        vocab=V, d_model=d, n_layers=n_layers_tf,
        n_heads=n_heads, d_ff=d_ff, context_len=ctx, dropout=dropout,
    )

    ratio = tsrn.count_params() / max(transformer.count_params(), 1)
    print(f"\n  TSRN/Transformer param ratio: {ratio:.2f}x")

    # -- Gradient verification ------------------------------------------
    print(f"\n-- Gradient verification {'-'*44}")
    tsrn.to(device)
    print("  TSRN:")
    grad_info = verify_gradients(tsrn, dataset, device)
    tsrn.cpu()

    # -- Throughput benchmark -------------------------------------------
    print(f"\n-- Throughput benchmark {'-'*46}")
    bench_batches = [8, 16, 32] if args.preset != "quick" else [8, 16]

    print("  TSRN:")
    tsrn.to(device)
    tsrn_bench = benchmark_throughput(tsrn, dataset, device, bench_batches)
    tsrn.cpu()

    print("  Transformer:")
    transformer.to(device)
    trans_bench = benchmark_throughput(transformer, dataset, device, bench_batches)
    transformer.cpu()

    # -- Training: Transformer baseline ---------------------------------
    log_trans = train_model(
        transformer, dataset, device,
        n_steps=n_steps, batch_size=batch_size, lr_max=lr,
        lr_warmup=n_steps // 10, label="Vanilla Transformer",
        eval_every=max(1, n_steps // 10),
    )
    transformer.cpu()

    # -- Training: TSRN -------------------------------------------------
    log_tsrn = train_model(
        tsrn, dataset, device,
        n_steps=n_steps, batch_size=batch_size, lr_max=lr,
        lr_warmup=n_steps // 10, label="TSRN (full)",
        eval_every=max(1, n_steps // 10),
    )

    # -- Text generation ------------------------------------------------
    print(f"\n-- Generated text (TSRN) {'-'*44}")
    sample = generate(tsrn, dataset, device, prompt="The king ", n_tokens=150)
    print(f"  {sample[:300]}")
    tsrn.cpu()

    # -- Ablation suite -------------------------------------------------
    ablation_logs = {}
    if not args.no_ablation:
        print(f"\n-- Ablation suite {'-'*51}")
        print(f"  Each variant trains for {ablation_steps} steps")
        ablation_logs = run_ablation_suite(
            V, d, ctx, dataset, device,
            n_steps=ablation_steps, batch_size=batch_size, lr=lr,
            n_blocks=n_blocks, top_k=top_k, n_heads=n_heads,
            mem_depth=mem_depth, dropout=dropout,
        )

    # -- Test-set evaluation (if available) -----------------------------
    test_results = {}
    has_test = hasattr(dataset, 'test') and dataset.test is not None
    if has_test and log_trans and log_tsrn:
        print(f"\n-- Test-set evaluation {'-'*47}")
        transformer.to(device)
        t_loss, t_ppl, t_bpc = evaluate(transformer, dataset, device,
                                          n_batches=50, batch_size=min(batch_size, 32),
                                          split="test")
        test_results["transformer"] = {"loss": round(t_loss, 5), "ppl": round(t_ppl, 3), "bpc": round(t_bpc, 4)}
        print(f"  Transformer:  PPL={t_ppl:.3f}  BPC={t_bpc:.4f}")
        transformer.cpu()

        tsrn.to(device)
        s_loss, s_ppl, s_bpc = evaluate(tsrn, dataset, device,
                                          n_batches=50, batch_size=min(batch_size, 32),
                                          split="test")
        test_results["tsrn"] = {"loss": round(s_loss, 5), "ppl": round(s_ppl, 3), "bpc": round(s_bpc, 4)}
        print(f"  TSRN:         PPL={s_ppl:.3f}  BPC={s_bpc:.4f}")
        tsrn.cpu()

    # -- Final comparison -----------------------------------------------
    print(f"\n\n{'='*80}")
    print("  FINAL RESULTS")
    print(f"{'='*80}")
    print(f"{'Metric':<34} {'Transformer':>14} {'TSRN':>14}")
    print(f"{'-'*80}")
    print(f"{'Parameters':<34} {transformer.count_params():>14,} {tsrn.count_params():>14,}")

    if log_trans and log_tsrn:
        best_t = min(log_trans, key=lambda e: e["val_ppl"])
        best_s = min(log_tsrn, key=lambda e: e["val_ppl"])
        print(f"{'Best val PPL':<34} {best_t['val_ppl']:>14.3f} {best_s['val_ppl']:>14.3f}")
        print(f"{'Best val BPC':<34} {best_t['val_bpc']:>14.4f} {best_s['val_bpc']:>14.4f}")
        print(f"{'Final val PPL':<34} {log_trans[-1]['val_ppl']:>14.3f} {log_tsrn[-1]['val_ppl']:>14.3f}")
        print(f"{'Final val BPC':<34} {log_trans[-1]['val_bpc']:>14.4f} {log_tsrn[-1]['val_bpc']:>14.4f}")

        if test_results:
            tr = test_results["transformer"]
            sr = test_results["tsrn"]
            print(f"{'Test PPL':<34} {tr['ppl']:>14.3f} {sr['ppl']:>14.3f}")
            print(f"{'Test BPC':<34} {tr['bpc']:>14.4f} {sr['bpc']:>14.4f}")

        if tsrn_bench and trans_bench:
            b0 = list(trans_bench.keys())[0]
            if b0 in tsrn_bench:
                spd = trans_bench[b0]["ms_per_batch"] / max(tsrn_bench[b0]["ms_per_batch"], 0.1)
                print(f"{'Speed ratio (TSRN/Trans @ B=' + str(b0) + ')':<34} {'1.00x':>14} {spd:>13.2f}x")

        print()
        winner_metric = test_results.get("tsrn", {}).get("bpc") or best_s["val_bpc"]
        loser_metric = test_results.get("transformer", {}).get("bpc") or best_t["val_bpc"]
        if winner_metric < loser_metric:
            print(f"  >> TSRN wins: BPC {winner_metric:.4f} vs {loser_metric:.4f}")
        else:
            print(f"  >> Transformer leads: BPC {loser_metric:.4f} vs {winner_metric:.4f}")

    if ablation_logs:
        print(f"\n  Ablation results (final val PPL / BPC):")
        full_entry = ablation_logs.get("TSRN_full", [{"val_ppl": 999, "val_bpc": 99}])[-1]
        full_ppl = full_entry["val_ppl"]
        full_bpc = full_entry.get("val_bpc", full_entry["val_loss"] / math.log(2) if "val_loss" in full_entry else 0)
        for name, log in ablation_logs.items():
            if log:
                ppl = log[-1]["val_ppl"]
                bpc = log[-1].get("val_bpc", log[-1]["val_loss"] / math.log(2) if "val_loss" in log[-1] else 0)
                delta_ppl = ppl - full_ppl
                delta_bpc = bpc - full_bpc
                print(f"    {name:<28}  PPL={ppl:.3f}  BPC={bpc:.4f}  delta_BPC={delta_bpc:+.4f}")

    print(f"{'='*80}")

    # -- Save results ---------------------------------------------------
    os.makedirs("results", exist_ok=True)
    results = {
        "config": {
            "preset": args.preset,
            "d_model": d,
            "context_len": ctx,
            "n_blocks": n_blocks,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "lr": lr,
            "top_k": top_k,
            "mem_depth": mem_depth,
            "dataset": args.dataset,
            "device": "DirectML" if _IS_DML else str(device),
        },
        "model_params": {
            "transformer": transformer.count_params(),
            "tsrn": tsrn.count_params(),
        },
        "transformer": {"log": log_trans, "benchmark": trans_bench},
        "tsrn": {
            "log": log_tsrn,
            "benchmark": tsrn_bench,
            "grad_check": grad_info,
            "sample_text": sample,
        },
        "test_results": test_results,
        "ablation": ablation_logs,
    }

    out_path = args.output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_path}")

    # -- Optional plots -------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("TSRN vs Transformer -- DirectML GPU Run", fontsize=13)

        steps_t = [e["step"] for e in log_trans]
        steps_s = [e["step"] for e in log_tsrn]

        axes[0].plot(steps_t, [e["val_ppl"] for e in log_trans], "b-o",
                     label="Transformer", ms=4)
        axes[0].plot(steps_s, [e["val_ppl"] for e in log_tsrn], "g-o",
                     label="TSRN", ms=4)
        axes[0].set_xlabel("Step"); axes[0].set_ylabel("Val perplexity")
        axes[0].set_title("Validation Perplexity")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(steps_t, [e["train_loss"] for e in log_trans], "b-o",
                     label="Transformer", ms=4)
        axes[1].plot(steps_s, [e["train_loss"] for e in log_tsrn], "g-o",
                     label="TSRN", ms=4)
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("Train loss")
        axes[1].set_title("Training Loss")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig("tsrn_curves.png", dpi=150)
        print("  Curves -> tsrn_curves.png")

        if ablation_logs:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            for name, log in ablation_logs.items():
                if log:
                    ax2.plot([e["step"] for e in log], [e["val_ppl"] for e in log],
                             label=name, marker="o", ms=3)
            ax2.set_xlabel("Step"); ax2.set_ylabel("Val perplexity")
            ax2.set_title("Ablation: Val Perplexity per Variant")
            ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
            fig2.tight_layout()
            fig2.savefig("tsrn_ablation.png", dpi=150)
            print("  Ablation plot -> tsrn_ablation.png")

    except ImportError:
        print("  (matplotlib not available -- skipping plots)")


if __name__ == "__main__":
    main()
