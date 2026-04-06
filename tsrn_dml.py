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
#  1. Tropical Sparse Attention  (Whitepaper Sec 2.1, Appendix A.1)
#
#     score(q,k) = max_c(q_c + k_c)  -- tropical inner product
#     Select top-k keys per query; softmax over top-k only.
#     DirectML fix: use masked_fill instead of scatter_ for backward compat.
# ---------------------------------------------------------------------------

class TropicalAttention(nn.Module):
    def __init__(self, d_model: int, top_k: int = 8, n_heads: int = 4,
                 dropout: float = 0.0):
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

    def _attend_chunk(self, Qq: Tensor, K: Tensor, V: Tensor,
                      qi: int, q_end: int, k: int, T: int,
                      causal: bool) -> Tensor:
        """Compute attention output for a chunk of queries [qi, q_end).

        Tropical scores via channel-chunked logsumexp to bound peak memory.
        """
        B, H, qc = Qq.shape[0], Qq.shape[1], Qq.shape[2]
        dh = Qq.shape[3]
        # Budget channel chunk size so 5D tensor stays under 32 MB
        max_5d_bytes = 32 * 1024 * 1024
        elems_per_cs = max(1, B * H * qc * T)
        cs = max(1, min(dh, max_5d_bytes // (elems_per_cs * 4)))

        if cs >= dh:
            raw = Qq.unsqueeze(3) + K.unsqueeze(2)       # B H qc T dh
            scores = torch.logsumexp(raw, dim=-1)         # B H qc T
            del raw
        else:
            scores = None
            for c0 in range(0, dh, cs):
                c1 = min(c0 + cs, dh)
                raw = Qq[:, :, :, c0:c1].unsqueeze(3) + K[:, :, :, c0:c1].unsqueeze(2)
                chunk_lse = torch.logsumexp(raw, dim=-1)
                del raw
                if scores is None:
                    scores = chunk_lse
                else:
                    m = torch.max(scores.detach(), chunk_lse.detach())
                    scores = m + torch.log(
                        torch.exp(scores - m) + torch.exp(chunk_lse - m))
                del chunk_lse

        # Causal mask
        if causal:
            col = torch.arange(T, device=scores.device).unsqueeze(0)     # 1 T
            row = torch.arange(qi, q_end, device=scores.device).unsqueeze(1)  # qc 1
            scores.masked_fill_((col > row).unsqueeze(0).unsqueeze(0), float("-inf"))

        # Top-k sparse softmax -- DirectML-safe (threshold masking)
        topk_v, _ = scores.topk(k, dim=-1)           # B H qc k
        thr = topk_v[:, :, :, -1:].detach()
        scores.masked_fill_(scores < thr, float("-inf"))
        w = self.drop(torch.softmax(scores, dim=-1))  # B H qc T (sparse, ~k nonzero)
        del scores

        return w @ V  # (B H qc T) @ (B H T dh) -> B H qc dh

    def _channel_chunked_scores(self, Q: Tensor, K: Tensor,
                                B: int, H: int, T: int, dh: int) -> Tensor:
        """Fast channel-chunked path for short contexts.
        Computes full (B,H,T,T) scores via logsumexp over dh chunks.
        """
        max_chunk_bytes = 128 * 1024 * 1024  # 128 MB
        cs = max(1, min(dh, max_chunk_bytes // max(1, B * H * T * T * 4)))
        scores = None
        for c0 in range(0, dh, cs):
            c1 = min(c0 + cs, dh)
            raw = Q[:, :, :, c0:c1].unsqueeze(3) + K[:, :, :, c0:c1].unsqueeze(2)
            chunk_lse = torch.logsumexp(raw, dim=-1)
            del raw
            if scores is None:
                scores = chunk_lse
            else:
                m = torch.max(scores.detach(), chunk_lse.detach())
                scores = m + torch.log(
                    torch.exp(scores - m) + torch.exp(chunk_lse - m))
            del chunk_lse
        return scores.contiguous()

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        B, T, d = x.shape
        H, dh = self.H, self.dh
        k = min(self.top_k, T)

        Q = self.Wq(x).view(B, T, H, dh).permute(0, 2, 1, 3)
        K = self.Wk(x).view(B, T, H, dh).permute(0, 2, 1, 3)
        V = self.Wv(x).view(B, T, H, dh).permute(0, 2, 1, 3)

        # ── Hybrid attention: fast path vs O(T·k) memory path ──
        # score_matrix = B*H*T*T*4 bytes. Autograd stores ~128× this
        # for the channel-chunk chain.  If total < ~4GB → fast path.
        score_bytes = B * H * T * T * 4
        use_fast = score_bytes <= 24 * 1024 * 1024  # ≤24 MB score matrix

        if use_fast:
            # --- Fast channel-chunked path (no query chunking) ---
            scores = self._channel_chunked_scores(Q, K, B, H, T, dh)

            if causal:
                mask = torch.triu(torch.ones(T, T, device=x.device,
                                             dtype=torch.bool), diagonal=1)
                scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            topk_v, _ = scores.topk(k, dim=-1)
            thr = topk_v[:, :, :, -1:].detach()
            scores.masked_fill_(scores < thr, float("-inf"))
            w = self.drop(torch.softmax(scores, dim=-1))
            ctx = (w @ V).permute(0, 2, 1, 3).contiguous().reshape(B, T, d)
        else:
            # --- O(T·k) query-chunked path with per-chunk checkpoint ---
            max_5d_elems = 16 * 1024 * 1024       # 64 MB / 4 bytes
            elem_per_qc  = max(1, B * H * T * dh)
            qc = max(1, min(T, max_5d_elems // elem_per_qc))

            out_parts = []
            for qi in range(0, T, qc):
                q_end = min(qi + qc, T)
                Qq = Q[:, :, qi:q_end, :]

                if self.training:
                    chunk = torch.utils.checkpoint.checkpoint(
                        self._attend_chunk, Qq, K, V,
                        qi, q_end, k, T, causal,
                        use_reentrant=False)
                else:
                    chunk = self._attend_chunk(Qq, K, V, qi, q_end, k, T, causal)
                out_parts.append(chunk)

            ctx = torch.cat(out_parts, dim=2)
            ctx = ctx.permute(0, 2, 1, 3).contiguous().reshape(B, T, d)

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
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        dh = d_model // 2
        self.proj_r = nn.Linear(d_model, dh)
        self.proj_i = nn.Linear(d_model, dh)
        self.proj_out = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: Tensor) -> Tensor:
        r = self.proj_r(x)
        i = self.proj_i(x)
        prod_r = r * r - i * i    # grade-0
        prod_i = 2.0 * r * i      # grade-2
        h = torch.cat([prod_r, prod_i], dim=-1)
        gate = torch.sigmoid(self.gate(x))
        return self.drop(self.proj_out(h * gate))


# ---------------------------------------------------------------------------
#  4. RG Coarse-graining  (Whitepaper Sec 2.3)
#
#     MERA-inspired: disentangle pairs then pool.
#     Disentangle: 2d -> 2d (near-identity, removes short-range correlations)
#     Pool: 2d -> d (merge adjacent tokens)
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
        if T % 2 != 0:
            x = x[:, :-1, :]
            T -= 1
        pair = torch.cat([x[:, 0::2, :].contiguous(), x[:, 1::2, :].contiguous()], dim=-1)
        pair = torch.tanh(self.disentangle(pair))
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
        self.leak = nn.Parameter(torch.tensor(0.3))

        # Register cache as buffer so it's saved/loaded with checkpoints
        self.register_buffer("_cached_v", torch.zeros(dr))

    def _power_iter_spectral_radius(self, W: Tensor, n_iter: int = 3) -> Tensor:
        """Approximate spectral radius via power iteration (GPU-safe, no eigvals).
        
        Uses detached W — gradients flow through rho_target, not through the
        spectral radius estimate itself.
        """
        W_det = W.detach()  # spectral radius is a constant for gradient purposes
        
        with torch.no_grad():
            if torch.all(self._cached_v == 0):
                self._cached_v.copy_(
                    torch.randn(W_det.shape[0], 
                            device=W_det.device, 
                            dtype=W_det.dtype)
                )
            v = self._cached_v.clone()  # clone so iteration doesn't touch the buffer
            for _ in range(n_iter):
                v = W_det @ v
                v_norm = v.norm()
                if v_norm > 0:
                    v = v / v_norm
            self._cached_v.copy_(v)
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

        # Run reservoir through time
        h = torch.zeros(B, dr, device=device, dtype=x.dtype)
        outs = []
        lk = torch.sigmoid(self.leak)
        for t in range(T):
            u = self.W_in(x[:, t, :])
            h = (1 - lk) * h + lk * torch.tanh(h @ W_scaled.T + u)
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
        B, H, T, D = path_q.shape
        agree = (path_q.unsqueeze(-2) * path_k.unsqueeze(-3) +
                 (1 - path_q.unsqueeze(-2)) * (1 - path_k.unsqueeze(-3)))
        cum_agree = torch.cumprod(agree, dim=-1)
        return cum_agree.sum(dim=-1)

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
#  TSRN Block (reusable per-scale block for depth scaling)
# ---------------------------------------------------------------------------

class TSRNBlock(nn.Module):
    """One TSRN processing block (per whitepaper Sec 3.2)."""

    def __init__(self, d_model: int, top_k: int, n_heads: int,
                 sheaf_window: int, mem_depth: int,
                 use_reservoir: bool, use_padic_attn: bool,
                 use_memory: bool, dropout: float = 0.1):
        super().__init__()

        # Tropical attention
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = TropicalAttention(d_model, top_k=top_k, n_heads=n_heads,
                                       dropout=dropout)

        # Sheaf diffusion
        self.ln_sheaf = nn.LayerNorm(d_model)
        self.sheaf = SheafDiffusion(d_model, window=sheaf_window, dropout=dropout)

        # Echo State Reservoir (optional, Scale 1 only per whitepaper)
        self.use_reservoir = use_reservoir
        if use_reservoir:
            self.ln_res = nn.LayerNorm(d_model)
            self.reservoir = EchoStateReservoir(d_model)

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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_attn(x))
        x = x + self.sheaf(self.ln_sheaf(x))
        if self.use_reservoir:
            x = x + self.reservoir(self.ln_res(x))
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
                 dropout: float = 0.1):
        super().__init__()
        self.ctx = context_len
        self.d = d_model

        # Embeddings
        self.embed = nn.Embedding(vocab, d_model)
        self.pos_s1 = nn.Embedding(context_len, d_model)
        self.pos_s2 = nn.Embedding(context_len // 2, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_s1.weight, std=0.01)
        nn.init.normal_(self.pos_s2.weight, std=0.01)

        # Scale 1 blocks (reservoir + memory, no p-adic attn)
        self.s1_blocks = nn.ModuleList([
            TSRNBlock(d_model, top_k=top_k, n_heads=n_heads,
                      sheaf_window=sheaf_window, mem_depth=mem_depth,
                      use_reservoir=(i == 0),  # reservoir only in first block
                      use_padic_attn=False, use_memory=True,
                      dropout=dropout)
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
                      use_memory=False, dropout=dropout)
            for i in range(n_blocks)
        ])

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

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
        if self.head.weight is not self.embed.weight:
            self.head.weight = self.embed.weight

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result._retie_weights()
        return result

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.embed(idx) + self.pos_s1(pos)

        # Scale 1
        for block in self.s1_blocks:
            if self.gradient_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # RG coarse-grain
        T2 = T // 2
        pos2 = torch.arange(T2, device=idx.device)
        xc = self.rg_pool(x) + self.pos_s2(pos2)

        # Scale 2
        for block in self.s2_blocks:
            if self.gradient_checkpoint and self.training:
                xc = torch.utils.checkpoint.checkpoint(block, xc, use_reentrant=False)
            else:
                xc = block(xc)

        # Upsample & fuse (whitepaper Sec 3.1: 0.5 weight blend)
        xc_up = xc.repeat_interleave(2, dim=1).contiguous()
        # Handle odd T: xc_up may be shorter than x after RGPool truncation
        if xc_up.size(1) < T:
            xc_up = F.pad(xc_up, (0, 0, 0, T - xc_up.size(1)))
        else:
            xc_up = xc_up[:, :T, :]
        x = x + 0.5 * xc_up

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
        pos = torch.arange(T, device=idx.device)
        x = self.embed(idx) + self.pos_s1(pos)

        # Scale 1
        for block in self.s1_blocks:
            x = x + block.attn(block.ln_attn(x))
            if "sheaf" not in self.ablate:
                x = x + block.sheaf(block.ln_sheaf(x))
            if block.use_reservoir and "reservoir" not in self.ablate:
                x = x + block.reservoir(block.ln_res(x))
            x = x + block.ffn(block.ln_ffn(x))
            if block.use_memory and "memory" not in self.ablate:
                x = x + block.mem(block.ln_mem(x))

        # RG coarse-grain
        if "rg" not in self.ablate:
            T2 = T // 2
            pos2 = torch.arange(T2, device=idx.device)
            xc = self.rg_pool(x) + self.pos_s2(pos2)
            for block in self.s2_blocks:
                xc = xc + block.attn(block.ln_attn(xc))
                if "sheaf" not in self.ablate:
                    xc = xc + block.sheaf(block.ln_sheaf(xc))
                xc = xc + block.ffn(block.ln_ffn(xc))
                if block.use_padic_attn and "padic_attn" not in self.ablate:
                    xc = xc + block.pa(block.ln_pa(xc))
            xc_up = xc.repeat_interleave(2, dim=1)[:, :T, :]
            x = x + 0.5 * xc_up

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
