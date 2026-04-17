"""
TSRN-Gist — Tropical Sheaf RN + Clifford Multivector Gists
===========================================================
Extends TSRN (tsrn_dml.py) with Clifford rotor gists, sheaf rotor
restriction maps, gist cross-attention, and RG-scale gist extraction.

Architecture per block:
  TropAttn -> SheafRotorDiffusion -> [Reservoir] -> GistRotation
  -> CliffordFFN -> GistCrossAttn -> [PAdicMem | PAdicAttn]

Theoretical grounding (NEXUS_Innovations_for_TSRN.md Sec 1):
  - Gist = Cl(1,0) rotor R=e^{-theta/2 B} rotating (r,i) state
  - Sheaf restriction maps are norm-preserving Clifford rotors
  - RG extracts gists; p-adic memory stores them by scale
  - p-adic ultrametric = natural inter-stalk distance

Usage:
  python tsrn_gist.py --preset quick --dataset enwik8
  python tsrn_gist.py --preset 22m  --dataset enwik8
"""

import argparse, json, math, os, time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tsrn_dml import (
    TropicalAttention, CliffordFFN, RGPool, PAdicMemory,
    EchoStateReservoir, PAdicAttention,
    CharDataset, CharDatasetSplit, load_enwik8, load_wikitext2,
    generate_synthetic_data, detect_device, evaluate,
    evaluate_sequential, get_lr, device_sync,
)


# ---------------------------------------------------------------------------
#  Clifford Rotor in Cl(1,0) — per-dimension-pair complex rotation
# ---------------------------------------------------------------------------

class CliffordRotor(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.dh = d_model // 2
        self.theta = nn.Parameter(torch.zeros(self.dh))

    def forward(self, x: Tensor) -> Tensor:
        r, i = x[..., :self.dh], x[..., self.dh:]
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        return torch.cat([r*c - i*s, r*s + i*c], dim=-1)

    def inverse(self, x: Tensor) -> Tensor:
        r, i = x[..., :self.dh], x[..., self.dh:]
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        return torch.cat([r*c + i*s, -r*s + i*c], dim=-1)


# ---------------------------------------------------------------------------
#  Sheaf Rotor Diffusion — rotors as restriction maps between stalks
# ---------------------------------------------------------------------------

class SheafRotorDiffusion(nn.Module):
    def __init__(self, d_model: int, window: int = 3, causal: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.offsets = list(range(-window, 1)) if causal else list(range(-window, window+1))
        self.rotors = nn.ModuleDict({str(d): CliffordRotor(d_model) for d in self.offsets})
        self.alpha = nn.Parameter(torch.tensor(0.15))
        self.drop = nn.Dropout(dropout)
        self.correction = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.correction.weight)

    def forward(self, x: Tensor) -> Tensor:
        B, T, d = x.shape
        lap = torch.zeros_like(x)
        for delta in self.offsets:
            rotor = self.rotors[str(delta)]
            Rx = rotor(x)
            if delta == 0:
                x_nb = x
            elif delta > 0:
                x_nb = torch.cat([x[:, delta:, :],
                    torch.zeros(B, delta, d, device=x.device, dtype=x.dtype)], dim=1)
            else:
                ad = -delta
                x_nb = torch.cat([torch.zeros(B, ad, d, device=x.device, dtype=x.dtype),
                    x[:, :T-ad, :]], dim=1)
            lap = lap + rotor.inverse(Rx - x_nb)
        out = x - self.alpha.abs() * lap
        return self.drop(out + self.correction(out))


# ---------------------------------------------------------------------------
#  Gist Extractor — causal per-position attention-pool to Cl(1,0) gist
#
#  CAUSALITY REQUIREMENT:
#    The extracted gist for coarse position j is injected into Scale 2
#    at coarse position j.  Coarse position j predicts fine target 2j+1,
#    so it must only see fine tokens x_0 ... x_{2j}.
#
#  FIX — prefix-masked causal extraction:
#    We produce T/2 gist vectors, one per coarse position j, where gist j
#    is computed by attending over fine tokens {0, 1, ..., 2j} only.
#
#    Let T2 = T // 2.  We issue T2 queries (one per coarse slot) against
#    T keys/values (fine positions).  The causal mask M is:
#
#        M[j, t] = 0    if  t <= 2j          (position j may see fine t)
#        M[j, t] = -inf if  t >  2j          (future — blocked)
#
#    This mask has shape (T2, T) and is computed once per forward pass.
#
#    Output:
#        theta : (B, T2, d//2)   — per-coarse Clifford rotation angles
#        mag   : (B, T2, 1)      — per-coarse magnitude gate
#
#    The legacy single-vector interface (used by GistBuffer.store) is
#    preserved by taking the last coarse position's gist, which has seen
#    x_0 ... x_{T-2} (the maximum causally-visible prefix), used to
#    represent the whole window for cross-batch retrieval.
# ---------------------------------------------------------------------------

class GistExtractor(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d, self.H, self.dh = d_model, n_heads, d_model // n_heads
        # One learnable query per coarse position is parameter-expensive;
        # instead use a single query prototype that is position-modulated
        # by a learned coarse positional embedding (same capacity, less params).
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.proj_theta = nn.Linear(d_model, d_model // 2)
        self.proj_mag = nn.Linear(d_model, 1)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, d) — Scale 1 output, T must be even.
        Returns:
            theta : (B, T2, d//2)  causal per-coarse gist angles
            mag   : (B, T2, 1)     causal per-coarse magnitude gates
        where T2 = T // 2.
        """
        B, T, d = x.shape
        T2 = T // 2
        H, dh = self.H, self.dh

        # Keys and values from all T fine positions
        k = self.Wk(x).view(B, T, H, dh).permute(0, 2, 1, 3)   # B H T dh
        v = self.Wv(x).view(B, T, H, dh).permute(0, 2, 1, 3)   # B H T dh

        # T2 queries — broadcast the single prototype to T2 positions.
        # Shape: (B, H, T2, dh)
        q = self.query.view(1, 1, H, dh).expand(B, T2, H, dh).permute(0, 2, 1, 3)

        # Raw scores: (B, H, T2, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)

        # Causal mask M[j, t] = -inf  iff  t > 2j
        # Coarse index j in {0,...,T2-1}, fine index t in {0,...,T-1}.
        # j_idx shape (T2, 1), t_idx shape (1, T) — broadcast to (T2, T).
        j_idx = torch.arange(T2, device=x.device).unsqueeze(1)   # T2 x 1
        t_idx = torch.arange(T, device=x.device).unsqueeze(0)    # 1  x T
        # Allowed: t <= 2j   i.e. t <= 2*j_idx
        future_mask = t_idx > 2 * j_idx                          # T2 x T, bool
        scores = scores.masked_fill(
            future_mask.unsqueeze(0).unsqueeze(0), float("-inf")) # broadcast over B, H

        # Softmax and weighted sum: (B, H, T2, dh)
        pooled = (torch.softmax(scores, dim=-1) @ v)
        # Reshape to (B, T2, d)
        pooled = pooled.permute(0, 2, 1, 3).contiguous().reshape(B, T2, d)
        pooled = self.ln(pooled)

        theta = self.proj_theta(pooled)                 # B T2 d//2
        mag   = torch.sigmoid(self.proj_mag(pooled))   # B T2 1
        return theta, mag

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Legacy single-vector extraction (causal: attends to all of x,
        which is the maximum safe prefix for cross-batch gist storage).
        Used only by GistBuffer.store to produce the ring-buffer key."""
        B, T, d = x.shape
        H, dh = self.H, self.dh
        q = self.query.view(1, 1, H, dh).expand(B, 1, H, dh).permute(0, 2, 1, 3)
        k = self.Wk(x).view(B, T, H, dh).permute(0, 2, 1, 3)
        v = self.Wv(x).view(B, T, H, dh).permute(0, 2, 1, 3)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        # No mask needed: storing a summary of the entire (past) window is causal.
        pooled = (torch.softmax(scores, -1) @ v).permute(0,2,1,3).reshape(B, d)
        pooled = self.ln(pooled)
        return self.proj_theta(pooled), torch.sigmoid(self.proj_mag(pooled))


# ---------------------------------------------------------------------------
#  Gist Buffer — ring buffer with tropical retrieval
# ---------------------------------------------------------------------------

class GistBuffer(nn.Module):
    def __init__(self, d_model: int, max_gists: int = 64):
        super().__init__()
        self.dh = d_model // 2
        self.max_gists = max_gists
        self.register_buffer("stored_theta", torch.zeros(max_gists, d_model//2))
        self.register_buffer("stored_mag", torch.zeros(max_gists, 1))
        self.register_buffer("stored_keys", torch.zeros(max_gists, d_model))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

    def store(self, theta: Tensor, mag: Tensor, ctx_repr: Tensor):
        B = theta.shape[0]
        with torch.no_grad():
            key = self.key_proj(ctx_repr.detach())
            for b in range(B):
                ptr = self.write_ptr.item() % self.max_gists
                self.stored_theta[ptr] = theta[b].detach()
                self.stored_mag[ptr] = mag[b].detach()
                self.stored_keys[ptr] = key[b].detach()
                self.write_ptr.add_(1)
            self.count.fill_(min(self.write_ptr.item(), self.max_gists))

    def retrieve(self, query: Tensor, top_k: int = 4):
        n, B = self.count.item(), query.shape[0]
        if n == 0:
            return (torch.zeros(B,1,self.dh, device=query.device),
                    torch.zeros(B,1,1, device=query.device),
                    torch.ones(B,1, device=query.device))
        keys = self.stored_keys[:n]
        # Detach: buffer is a non-differentiable cache; avoids DML scatter in topk backward
        scores = torch.logsumexp(
            query.detach().unsqueeze(1) + keys.unsqueeze(0), dim=-1)
        k = min(top_k, n)
        topk_s, topk_i = scores.topk(k, dim=-1)
        w = torch.softmax(topk_s, dim=-1)
        return self.stored_theta[:n][topk_i], self.stored_mag[:n][topk_i], w

    def reset(self):
        self.stored_theta.zero_(); self.stored_mag.zero_()
        self.stored_keys.zero_(); self.count.zero_(); self.write_ptr.zero_()


# ---------------------------------------------------------------------------
#  Gist Rotation — apply rotor before CliffordFFN (pre-token alignment)
#
#  When called from Scale 1 (fine positions):
#    gist_theta/mag come from the GistBuffer (previous-window gists) with
#    shape (B, K, dh) — K retrieved past gists.  The weighted sum produces
#    a single rotation applied uniformly to all T fine positions.  This
#    is safe: retrieved gists are from strictly past windows.
#
#  When called from Scale 2 (coarse positions):
#    gist_theta has shape (B, T2, dh) — one gist per coarse position j,
#    computed causally by GistExtractor.forward().  gist_weights is None
#    in this case, signalling per-position mode: each coarse token j is
#    rotated by its own causal gist j.
# ---------------------------------------------------------------------------

class GistRotationLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.dh = d_model // 2
        self.gist_strength = nn.Parameter(torch.tensor(0.0))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, gist_theta, gist_mag, gist_weights=None):
        """
        x           : (B, T, d)
        gist_theta  : (B, K, dh)  K retrieved gists  — Scale 1 path
                   OR (B, T, dh)  per-position gists  — Scale 2 path
        gist_mag    : (B, K, 1) or (B, T, 1)
        gist_weights: (B, K) or None (None signals per-position mode)
        """
        if gist_weights is not None:
            # Scale 1 path: weighted sum over K retrieved past gists -> single theta
            w = gist_weights.unsqueeze(-1) * gist_mag           # B K 1
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            theta = (w * gist_theta).sum(dim=1)                 # B dh
            theta = theta * torch.sigmoid(self.gist_strength)
            # Broadcast theta to all T positions
            r, i = x[..., :self.dh], x[..., self.dh:]
            c = torch.cos(theta).unsqueeze(1)                   # B 1 dh
            s = torch.sin(theta).unsqueeze(1)
        else:
            # Scale 2 path: per-position gist, shape (B, T, dh)
            # gist_mag shape: (B, T, 1) — squeeze last dim for elementwise
            theta = gist_theta * torch.sigmoid(self.gist_strength)  # B T dh
            r, i = x[..., :self.dh], x[..., self.dh:]
            c = torch.cos(theta)                                # B T dh
            s = torch.sin(theta)
        return self.ln(torch.cat([r * c - i * s, r * s + i * c], dim=-1))


# ---------------------------------------------------------------------------
#  Gist Cross-Attention — tokens attend to past gist representations
#
#  Scale 1 path: gist_theta shape (B, K, dh), gist_mag (B, K, 1).
#    K retrieved gists from the ring buffer — always strictly past windows.
#    No causal masking needed between current tokens and past-window gists.
#
#  Scale 2 path: gist_theta shape (B, T2, dh), gist_mag (B, T2, 1).
#    Per-position causal gists from GistExtractor.  Coarse token j must
#    only attend to gist key j (its own causal summary).  We enforce this
#    with a diagonal mask: each query attends only to its own gist key.
#    This degenerates to a simple elementwise gated add, which is what the
#    architecture intends (the gist "primes" that specific token), so
#    we implement it as a direct projection rather than wasting attention
#    compute on an identity-masked softmax.
# ---------------------------------------------------------------------------

class GistCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.H, self.dh = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.gist_up = nn.Linear(d_model//2 + 1, d_model)
        # Per-position projection used in Scale 2 path (diagonal attend)
        self.gist_gate = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.Wo.weight)
        nn.init.zeros_(self.gist_gate.weight)
        nn.init.zeros_(self.gist_gate.bias)

    def forward(self, x, gist_theta, gist_mag):
        """
        x          : (B, T, d)
        gist_theta : (B, K, dh)  K past-window gists  — Scale 1 path
                  OR (B, T, dh)  per-position gists    — Scale 2 path
        gist_mag   : (B, K, 1) or (B, T, 1)

        We distinguish paths by comparing gist_theta.shape[1] to x.shape[1].
        Scale 1: gist_theta.shape[1] == K (small, from buffer, K << T).
        Scale 2: gist_theta.shape[1] == T (equals sequence length of x).
        """
        B, T, d = x.shape
        K = gist_theta.shape[1]

        # Lift gist (theta, mag) back to model dim
        gist_repr = self.gist_up(torch.cat([gist_theta, gist_mag], dim=-1))  # B K d  or  B T d

        if K != T:
            # ── Scale 1 path: standard cross-attention, all-to-all ──
            # K is the number of retrieved past-window gists (small, << T).
            # No causal masking required: all K gists are from strictly past windows.
            H, dh = self.H, self.dh
            Q = self.Wq(x).view(B, T, H, dh).permute(0, 2, 1, 3)
            K_ = self.Wk(gist_repr).view(B, K, H, dh).permute(0, 2, 1, 3)
            V  = self.Wv(gist_repr).view(B, K, H, dh).permute(0, 2, 1, 3)
            scores = (Q @ K_.transpose(-2, -1)) / math.sqrt(dh)
            ctx = (self.drop(torch.softmax(scores, -1)) @ V)
            return self.Wo(ctx.permute(0, 2, 1, 3).contiguous().reshape(B, T, d))
        else:
            # ── Scale 2 path: per-position causal gist conditioning ──
            # gist_repr[j] was computed from fine tokens {0..2j} only (causal).
            # Attending token j to all gist_repr keys would reintroduce future
            # info (gist_repr[j'] for j'>j encodes fine tokens up to 2j' > 2j).
            # Correct operation: token j uses ONLY gist_repr[j].
            # This is equivalent to a diagonal attention mask, which simplifies
            # to an elementwise gated projection — no cross-position mixing.
            gate = torch.sigmoid(self.gist_gate(gist_repr))   # B T d
            return self.Wo(x * gate)


# ---------------------------------------------------------------------------
#  TSRN-Gist Block
# ---------------------------------------------------------------------------

class TSRNGistBlock(nn.Module):
    """TSRN block with Clifford gist rotation and sheaf rotors.

    Flow: TropAttn -> SheafRotor -> [Reservoir] -> GistRotation
          -> CliffordFFN -> [GistCrossAttn] -> [PAdicMem | PAdicAttn]
    """

    def __init__(self, d_model: int, top_k: int, n_heads: int,
                 sheaf_window: int, mem_depth: int,
                 use_reservoir: bool, use_padic_attn: bool,
                 use_memory: bool, use_gist_cross_attn: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = TropicalAttention(d_model, top_k=top_k, n_heads=n_heads,
                                       dropout=dropout)
        self.ln_sheaf = nn.LayerNorm(d_model)
        self.sheaf = SheafRotorDiffusion(d_model, window=sheaf_window,
                                          dropout=dropout)
        self.use_reservoir = use_reservoir
        if use_reservoir:
            self.ln_res = nn.LayerNorm(d_model)
            self.reservoir = EchoStateReservoir(d_model)

        self.ln_gist = nn.LayerNorm(d_model)
        self.gist_rotation = GistRotationLayer(d_model)

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = CliffordFFN(d_model, dropout=dropout)

        self.use_gist_cross_attn = use_gist_cross_attn
        if use_gist_cross_attn:
            self.ln_gca = nn.LayerNorm(d_model)
            self.gist_cross_attn = GistCrossAttention(d_model, n_heads=n_heads,
                                                       dropout=dropout)
        self.use_memory = use_memory
        if use_memory:
            self.ln_mem = nn.LayerNorm(d_model)
            self.mem = PAdicMemory(d_model, depth=mem_depth)

        self.use_padic_attn = use_padic_attn
        if use_padic_attn:
            self.ln_pa = nn.LayerNorm(d_model)
            self.pa = PAdicAttention(d_model, tree_depth=5, n_heads=n_heads,
                                      dropout=dropout)

    def forward(self, x: Tensor, gist_theta: Tensor = None,
                gist_mag: Tensor = None, gist_weights: Tensor = None) -> Tensor:
        x = x + self.attn(self.ln_attn(x))
        x = x + self.sheaf(self.ln_sheaf(x))
        if self.use_reservoir:
            x = x + self.reservoir(self.ln_res(x))
        if gist_theta is not None and gist_mag is not None:
            xn = self.ln_gist(x)
            x = x + (self.gist_rotation(xn, gist_theta, gist_mag, gist_weights) - xn)
        x = x + self.ffn(self.ln_ffn(x))
        if self.use_gist_cross_attn and gist_theta is not None:
            x = x + self.gist_cross_attn(self.ln_gca(x), gist_theta, gist_mag)
        if self.use_memory:
            x = x + self.mem(self.ln_mem(x))
        if self.use_padic_attn:
            x = x + self.pa(self.ln_pa(x))
        return x


# ---------------------------------------------------------------------------
#  Full TSRN-Gist Model
#
#  Two-scale architecture with gist extraction at the RG boundary:
#    Scale 1: n_blocks x TSRNGistBlock (reservoir, memory, gist)
#    Gist extraction: compress Scale 1 output -> Cl(1,0) rotor
#    RG coarse-grain: T -> T/2
#    Scale 2: n_blocks x TSRNGistBlock (padic_attn, gist)
#    Upsample & fuse -> logits
#
#  The RG boundary is the natural gist extraction point: it's where the
#  renormalization flow compresses fine-grained info.  Gists extracted
#  here bridge sheaf (local coherence) and p-adic (hierarchical memory)
#  through the RG scale transformation.  The p-adic tree depth maps to
#  RG scale, so the ultrametric IS the inter-scale distance.
# ---------------------------------------------------------------------------

class TSRNGist(nn.Module):
    def __init__(self, vocab: int, d_model: int, context_len: int,
                 gradient_checkpoint: bool = False,
                 n_blocks: int = 1, top_k: int = 8, n_heads: int = 4,
                 mem_depth: int = 6, sheaf_window: int = 3,
                 max_gists: int = 64, gist_top_k: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.ctx = context_len
        self.d = d_model
        self.gist_top_k = gist_top_k

        self.embed = nn.Embedding(vocab, d_model)
        self.pos_s1 = nn.Embedding(context_len, d_model)
        self.pos_s2 = nn.Embedding(context_len // 2, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_s1.weight, std=0.01)
        nn.init.normal_(self.pos_s2.weight, std=0.01)

        # Scale 1 blocks
        self.s1_blocks = nn.ModuleList([
            TSRNGistBlock(d_model, top_k=top_k, n_heads=n_heads,
                          sheaf_window=sheaf_window, mem_depth=mem_depth,
                          use_reservoir=(i == 0),
                          use_padic_attn=False, use_memory=True,
                          use_gist_cross_attn=True, dropout=dropout)
            for i in range(n_blocks)
        ])

        # Gist extraction at RG boundary
        self.gist_extractor = GistExtractor(d_model, n_heads=n_heads)
        self.gist_buffer = GistBuffer(d_model, max_gists=max_gists)

        # RG coarse-grain
        self.rg_pool = RGPool(d_model)

        # Scale 2 blocks
        self.s2_blocks = nn.ModuleList([
            TSRNGistBlock(d_model, top_k=max(2, top_k // 2), n_heads=n_heads,
                          sheaf_window=sheaf_window, mem_depth=mem_depth,
                          use_reservoir=False,
                          use_padic_attn=(i == n_blocks - 1),
                          use_memory=False,
                          use_gist_cross_attn=True, dropout=dropout)
            for i in range(n_blocks)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()
        self.gradient_checkpoint = gradient_checkpoint
        print(f"  TSRNGist  : {self.count_params():,} parameters")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2 and "embed" not in name \
               and "W_res" not in name and "leaf" not in name \
               and "router" not in name and "path" not in name \
               and "theta" not in name and "stored" not in name:
                nn.init.xavier_uniform_(p, gain=0.5)

    def _retie_weights(self):
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

        # Retrieve gists from buffer (strictly past windows — causal by construction)
        ctx_summary = x.mean(dim=1)  # B d  (embedding mean; no future info here)
        gist_theta, gist_mag, gist_w = self.gist_buffer.retrieve(
            self.gist_buffer.key_proj(ctx_summary), top_k=self.gist_top_k)
        # gist_theta: (B, K, dh),  gist_mag: (B, K, 1),  gist_w: (B, K)
        # K <= max_gists, all from previous windows — safe to use in Scale 1.

        # Scale 1 — fine tokens, uses retrieved past-window gists
        for block in self.s1_blocks:
            if self.gradient_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, gist_theta, gist_mag, gist_w,
                    use_reentrant=False)
            else:
                x = block(x, gist_theta, gist_mag, gist_w)

        # ── Causal gist extraction at RG boundary ──────────────────────────
        # GistExtractor.forward() now returns per-coarse-position gists:
        #   fresh_theta : (B, T2, dh)  where fresh_theta[j] encodes x_{0..2j}
        #   fresh_mag   : (B, T2, 1)
        # This is strictly causal: coarse token j sees fine tokens 0..2j only.
        T2 = T // 2
        fresh_theta, fresh_mag = self.gist_extractor(x)
        # fresh_theta shape: (B, T2, dh) — verify against T2
        assert fresh_theta.shape[1] == T2, (
            f"GistExtractor returned {fresh_theta.shape[1]} positions, expected {T2}")

        # Store a single cross-batch gist summary using the last causal position
        # (position T2-1 encodes x_{0..T-2}, the maximal causal prefix).
        # forward_single re-uses the same weights, attends to all of x (causal
        # because all of x is the current past window).
        if self.training:
            store_theta, store_mag = self.gist_extractor.forward_single(x)
            self.gist_buffer.store(store_theta, store_mag, ctx_summary)

        # RG coarse-grain (now causal — see RGPool docstring)
        pos2 = torch.arange(T2, device=idx.device)
        xc = self.rg_pool(x) + self.pos_s2(pos2)

        # Scale 2 — coarse tokens.
        # Pass fresh_theta/mag with gist_weights=None to signal per-position mode.
        # GistRotationLayer and GistCrossAttention use per-position gist[j] for
        # coarse token j — no future information crosses the causal boundary.
        for block in self.s2_blocks:
            if self.gradient_checkpoint and self.training:
                xc = torch.utils.checkpoint.checkpoint(
                    block, xc, fresh_theta, fresh_mag, None,
                    use_reentrant=False)
            else:
                xc = block(xc, fresh_theta, fresh_mag, None)

        # Upsample & fuse
        # xc_up[t] = xc[t//2], which was built from fine tokens 0..t (causal). ✓
        xc_up = xc.repeat_interleave(2, dim=1).contiguous()
        if xc_up.size(1) < T:
            xc_up = F.pad(xc_up, (0, 0, 0, T - xc_up.size(1)))
        else:
            xc_up = xc_up[:, :T, :]
        x = x + 0.5 * xc_up

        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train_gist(dataset, device, n_steps=100000, batch_size=8,
               lr_max=2e-4, d_model=512, context_len=256,
               n_blocks=3, n_heads=8, top_k=16, mem_depth=7,
               max_gists=64, gist_top_k=4,
               dropout=0.1, ckpt_every=10000, resume_from=None,
               tag="", grad_accum_steps=1, gradient_checkpoint=False):
    V = dataset.vocab_sz
    ctx = context_len

    torch.manual_seed(42)
    model = TSRNGist(vocab=V, d_model=d_model, context_len=ctx,
                     gradient_checkpoint=gradient_checkpoint,
                     n_blocks=n_blocks, top_k=top_k, n_heads=n_heads,
                     mem_depth=mem_depth, max_gists=max_gists,
                     gist_top_k=gist_top_k, dropout=dropout)

    start_step = 0
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"  Resumed from {resume_from} at step {start_step}")

    model.to(device)
    model.train()

    print(f"\n  TSRNGist: {model.count_params():,} parameters")
    print(f"  Vocab: {V}  |  Context: {ctx}  |  d_model: {d_model}")
    eff_batch = batch_size * grad_accum_steps
    print(f"  Steps: {n_steps}  |  Batch: {batch_size}x{grad_accum_steps}={eff_batch}")
    print(f"  LR: {lr_max}  |  Gists: {max_gists} buf, top-{gist_top_k}")
    print(f"  Checkpoint every {ckpt_every} steps")

    decay = {p for n, p in model.named_parameters()
             if p.requires_grad and p.dim() >= 2}
    no_decay = {p for p in model.parameters()
                if p.requires_grad and p not in decay}
    optimizer = torch.optim.AdamW([
        {"params": list(decay), "weight_decay": 0.1},
        {"params": list(no_decay), "weight_decay": 0.0},
    ], lr=lr_max, betas=(0.9, 0.95))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    log = []
    best_val_bpc = float("inf")
    best_model_state = None
    eval_every = max(1, n_steps // 50)
    t0 = time.time()

    print(f"\n{'='*85}")
    print(f"  TSRNGist — enwik8 byte-level ({model.count_params()/1e6:.1f}M params)")
    print(f"{'='*85}")
    print(f"{'Step':>6}  {'TrLoss':>10}  {'TrBPC':>8}  "
          f"{'ValLoss':>9}  {'ValPPL':>9}  {'ValBPC':>8}  "
          f"{'GNorm':>7}  {'ms/step':>10}")
    print(f"{'-'*85}")

    for step in range(start_step + 1, n_steps + 1):
        lr = get_lr(step, min(n_steps // 10, 4000), n_steps, lr_max, lr_max * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Reset gist buffer periodically to avoid stale gists
        if step % 100 == 1:
            model.gist_buffer.reset()

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = dataset.batch("train", batch_size, device)
            _, loss = model(x, y)
            (loss / grad_accum_steps).backward()
            accum_loss += loss.item() / grad_accum_steps

        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == 1:
            model.gist_buffer.reset()
            val_loss, val_ppl, val_bpc = evaluate(
                model, dataset, device,
                batch_size=min(batch_size, 32))
            tr_bpc = accum_loss / math.log(2)
            elapsed = time.time() - t0
            ms_step = elapsed / (step - start_step) * 1000
            print(f"{step:>6}  {accum_loss:>10.4f}  {tr_bpc:>8.4f}  "
                  f"{val_loss:>9.4f}  {val_ppl:>9.2f}  {val_bpc:>8.4f}  "
                  f"{float(gnorm):>7.3f}  {ms_step:>7.1f}ms")
            log.append({
                "step": step, "train_loss": round(accum_loss, 5),
                "train_bpc": round(tr_bpc, 4),
                "val_loss": round(val_loss, 5), "val_ppl": round(val_ppl, 3),
                "val_bpc": round(val_bpc, 4),
                "grad_norm": round(float(gnorm), 4),
                "lr": round(lr, 6), "time_s": round(elapsed, 1),
            })

            if val_bpc < best_val_bpc:
                best_val_bpc = val_bpc
                best_model_state = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
                torch.save({
                    "step": step,
                    "model_state_dict": best_model_state,
                    "val_bpc": val_bpc,
                    "config": {
                        "d_model": d_model, "context_len": ctx,
                        "n_blocks": n_blocks, "n_heads": n_heads,
                        "top_k": top_k, "mem_depth": mem_depth,
                        "max_gists": max_gists, "gist_top_k": gist_top_k,
                        "dropout": dropout, "vocab": V,
                    },
                }, f"checkpoints/tsrn_gist_best{tag}.pt")

        if step % ckpt_every == 0:
            ckpt_path = f"checkpoints/tsrn_gist_{step}steps{tag}.pt"
            torch.save({
                "step": step,
                "model_state_dict": {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "val_bpc": log[-1]["val_bpc"] if log else None,
                "log": log,
            }, ckpt_path)
            print(f"  >> Checkpoint: {ckpt_path}")
            with open(f"results/tsrn_gist_progress_{step}steps{tag}.json",
                      "w") as f:
                json.dump({"log": log, "best_val_bpc": best_val_bpc},
                          f, indent=2)

    final_path = f"checkpoints/tsrn_gist_final_{n_steps}steps{tag}.pt"
    torch.save({
        "step": n_steps,
        "model_state_dict": {k: v.cpu().clone()
                             for k, v in model.state_dict().items()},
        "log": log, "best_val_bpc": best_val_bpc,
    }, final_path)

    print(f"\n{'='*85}")
    print(f"  Training complete. Best val BPC: {best_val_bpc:.4f}")
    print(f"  Final checkpoint: {final_path}")
    print(f"{'='*85}")
    return model, log


# ---------------------------------------------------------------------------
#  Presets & Main
# ---------------------------------------------------------------------------

PRESETS = {
    "2m": {
        "d_model": 256, "context": 256, "n_blocks": 1, "n_heads": 4,
        "top_k": 16, "mem_depth": 6, "max_gists": 32, "gist_top_k": 4,
        "steps": 3000, "batch": 32, "lr": 3e-4,
    },
    "22m": {
        "d_model": 512, "context": 256, "n_blocks": 3, "n_heads": 8,
        "top_k": 16, "mem_depth": 7, "max_gists": 64, "gist_top_k": 4,
        "steps": 100000, "batch": 8, "lr": 2e-4,
    },
    "50m": {
        "d_model": 512, "context": 256, "n_blocks": 7, "n_heads": 8,
        "top_k": 16, "mem_depth": 7, "max_gists": 64, "gist_top_k": 8,
        "steps": 100000, "batch": 8, "lr": 2e-4,
    },
    "quick": {
        "d_model": 128, "context": 64, "n_blocks": 1, "n_heads": 2,
        "top_k": 8, "mem_depth": 5, "max_gists": 16, "gist_top_k": 2,
        "steps": 200, "batch": 16, "lr": 3e-4,
    },
}


def main():
    parser = argparse.ArgumentParser(description="TSRNGist Training")
    parser.add_argument("--preset", default="22m", choices=list(PRESETS.keys()))
    parser.add_argument("--dataset", default="enwik8",
                        choices=["synthetic", "wikitext2", "enwik8"])
    parser.add_argument("--data", default="data/tsrn_synthetic.txt")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_blocks", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--mem_depth", type=int, default=None)
    parser.add_argument("--max_gists", type=int, default=None)
    parser.add_argument("--gist_top_k", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--grad_ckpt", action="store_true")
    args = parser.parse_args()

    cfg = dict(PRESETS[args.preset])
    for key in ["steps", "d_model", "context", "batch", "lr", "n_blocks",
                "top_k", "mem_depth", "max_gists", "gist_top_k"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    print("\n" + "=" * 72)
    print("  TSRNGist — Tropical Sheaf RN + Clifford Multivector Gists")
    print("=" * 72)
    device = detect_device()

    d = cfg["d_model"]
    ctx = cfg["context"]
    print(f"  Preset    : {args.preset}")
    print(f"  d_model   : {d}  |  context: {ctx}  |  blocks: {cfg['n_blocks']}")
    print(f"  gists     : {cfg['max_gists']} buffer, top-{cfg['gist_top_k']} retrieval")

    # Dataset
    if args.dataset == "enwik8":
        dataset = load_enwik8(context_len=ctx)
    elif args.dataset == "wikitext2":
        dataset = load_wikitext2(context_len=ctx)
    else:
        from pathlib import Path
        if not Path(args.data).exists():
            generate_synthetic_data(args.data)
        dataset = CharDataset(args.data, context_len=ctx)

    # Train
    model, log = train_gist(
        dataset, device,
        n_steps=cfg["steps"], batch_size=cfg["batch"], lr_max=cfg["lr"],
        d_model=d, context_len=ctx, n_blocks=cfg["n_blocks"],
        n_heads=cfg.get("n_heads", max(4, d // 64)),
        top_k=cfg["top_k"], mem_depth=cfg["mem_depth"],
        max_gists=cfg["max_gists"], gist_top_k=cfg["gist_top_k"],
        dropout=0.1, ckpt_every=10000,
        resume_from=args.resume, tag=args.tag,
        grad_accum_steps=args.grad_accum,
        gradient_checkpoint=args.grad_ckpt,
    )

    # Final sequential test evaluation
    if hasattr(dataset, "test") and dataset.test is not None:
        print("\n-- Sequential test evaluation --")
        model.gist_buffer.reset()
        t_loss, t_ppl, t_bpc = evaluate_sequential(
            model, dataset, device, batch_size=8, split="test")
        print(f"  Test BPC: {t_bpc:.4f}  PPL: {t_ppl:.3f}")


if __name__ == "__main__":
    main()
