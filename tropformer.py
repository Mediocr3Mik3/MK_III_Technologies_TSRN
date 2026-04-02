"""
TropFormer: Tropical Transformer
=================================
A transformer architecture grounded in tropical geometry, Maslov
dequantization, and Legendre-Fenchel duality.

Mathematical foundations
------------------------
1. Tropical semiring T = (ℝ ∪ {-∞}, ⊕, ⊗)
       a ⊕ b = max(a, b)         tropical addition
       a ⊗ b = a + b             tropical multiplication
   Tropical linear map:
       (W ⊗ x)_i = max_j(W_ij + x_j)

2. Maslov dequantization (Litvinov 2005)
   Classical math uses (ℝ, +, ×). Tropical math uses (ℝ, max, +).
   The bridge: ℏ · log(Σ exp(xᵢ/ℏ))  →  max(xᵢ)  as ℏ → 0
   In attention: temperature τ (= ℏ) is a *learnable parameter per head*,
   letting each head find its own point on the tropical↔classical spectrum.

3. Legendre-Fenchel duality
   f*(y) = sup_x { ⟨x, y⟩ − f(x) }      (convex conjugate)

   Key connections:
   (a) softmax = ∇LSE(x) where LSE is the LF conjugate of negative entropy.
       As τ→0, LSE_τ → max, ∇max = argmax.  Maslov temperature interpolates
       between these two conjugate structures.
   (b) A tropical polynomial f(x) = max_k(s_k·x + b_k) is a piecewise-linear
       convex function.  Its LF conjugate f*(y) = max_j(x_j·y − f(x_j)) is
       ANOTHER tropical polynomial — the dual lives in the same algebra.
   (c) The breakpoints of f define the Newton polytope of the tropical variety;
       f* corresponds to the dual polytope under polyhedral duality.

Architecture
------------
   Image  →  Patch Embed  →  CLS + pos_enc
          →  [TropicalTransformerBlock × L]
          →  LayerNorm → head(CLS token) → logits

   TropicalTransformerBlock:
     Pre-LN → TropicalMultiHeadAttention → residual
     Pre-LN → TropicalHybridFFN          → residual

   TropicalMultiHeadAttention (per head):
     Q, K  ←  TropicalLinear           (max-plus projection)
     V     ←  Classical Linear          (smooth, gradient-rich values)
     scores = score_gate · trop_scores + (1−gate) · classical_scores
              where trop_score(q,k)  = max_i(q_i + k_i) / √d    [tropical ⟨·,·⟩]
              and   class_score(q,k) = (Q·Kᵀ)  / √d             [Euclidean ⟨·,·⟩]
     attn  =  MaslovSoftmax(scores, τ_head)
     out   =  attn @ V  →  out_proj

   TropicalHybridFFN:
     x → TropLinear → LFDualActivation  ]  gated blend
     x → ClassLinear → GELU             ]
     → GatedFusion → down_proj → residual

   LFDualActivation:
     primal f(x)  = max_k(s_k·x + b_k)        [tropical polynomial]
     dual   f*(y) = max_j(x_j·y − f(x_j))     [LF conjugate — also tropical!]
     Learnable mode gate chooses primal / dual / blend per invocation.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import torch_directml
    DML_AVAILABLE = True
except ImportError:
    DML_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# §0  DirectML-compatible tropical max (STE: hard forward, smooth backward)
# ═══════════════════════════════════════════════════════════════════════════════

_SMOOTH_MAX_TEMP = 50.0   # high temp → close to hard max; avoids scatter in backward
_USE_SMOOTH_MAX  = False  # set True only for DirectML (scatter workaround)

def _tropical_max(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Tropical max with optional DirectML compatibility.
    On CPU/CUDA: uses fast native max (default).
    On DirectML: uses logsumexp STE to avoid scatter in backward.
    """
    if not _USE_SMOOTH_MAX:
        return x.max(dim=dim).values
    hard = x.max(dim=dim).values.detach()
    soft = _SMOOTH_MAX_TEMP * torch.logsumexp(x / _SMOOTH_MAX_TEMP, dim=dim)
    return hard + (soft - soft.detach())          # STE: value=hard, grad=soft


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Tropical primitives
# ═══════════════════════════════════════════════════════════════════════════════

class TropicalLinear(nn.Module):
    """
    Max-plus linear layer with STE gradient stabilization.
        y_i = max_j(W_ij + x_j) + b_i

    Forward: exact tropical max (hard routing preserved).
    Backward: softmax-weighted smooth approximation (STE) so ALL weights
    receive gradient signal proportional to how close they were to winning.

    Input shape:  (..., in_features)   [any leading dims]
    Output shape: (..., out_features)
    """

    _ste_temp = 1.0   # STE temperature: lower = closer to hard max gradient

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.uniform_(self.weight, -0.5, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]
        x_flat  = x.reshape(-1, self.in_features)
        # scores[b, out, in] = W[out, in] + x[b, in]
        scores = self.weight.unsqueeze(0) + x_flat.unsqueeze(1)

        if self.training:
            # Hard forward value (detach BEFORE max to avoid scatter in autograd graph)
            hard = scores.detach().max(dim=-1).values
            # STE backward: softmax-weighted smooth approximation
            soft_w = F.softmax(scores / self._ste_temp, dim=-1)
            soft = (soft_w * scores).sum(dim=-1)
            out = hard + (soft - soft.detach())
        else:
            out = scores.max(dim=-1).values

        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*leading, self.out_features)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}"


class TropicalDropout(nn.Module):
    """
    Tropical dropout: sets activations to −∞ (tropical zero = identity under max).
    Classical dropout sets to 0 (additive identity), which is NOT tropical zero.
    Using −∞ correctly propagates "this neuron did not participate" in the
    downstream max operations.
    """
    TROP_ZERO = -1e9

    def __init__(self, p: float = 0.05):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = torch.bernoulli(torch.full_like(x, 1.0 - self.p)).bool()
        return torch.where(mask, x, torch.full_like(x, self.TROP_ZERO))


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Maslov dequantization
# ═══════════════════════════════════════════════════════════════════════════════

class MaslovTemperature(nn.Module):
    """
    Learnable Maslov dequantization temperature, one per attention head.

    The Maslov bridge:
        LSE_τ(x) = τ · log Σ exp(xᵢ/τ)     →   max(xᵢ)   as  τ → 0
        ∇LSE_τ  = softmax(x/τ)              →   argmax(x) as  τ → 0

    τ is parameterised as log(τ) for unconstrained optimisation.
    After training, inspecting τ per head reveals:
        τ ≈ 0.1  →  nearly argmax / tropical (hard, routing-like)
        τ ≈ 1.0  →  standard softmax (dense, entropic)
        τ >> 1   →  near-uniform attention (maximum entropy)

    LF connection: the two extremes correspond to two different convex
    regularisers on the attention simplex:
        τ=1  →  entropic reg  →  f*(x) = LSE(x)  (log-partition)
        τ→0  →  no reg        →  f*(x) = max(x)  (tropical conjugate)
    The learnable τ interpolates along the entire family of conjugate pairs.
    """

    def __init__(self, num_heads: int, init_temp: float = 1.0):
        super().__init__()
        self.log_temps = nn.Parameter(
            torch.full((num_heads,), math.log(init_temp))
        )

    @property
    def temperatures(self) -> torch.Tensor:
        return self.log_temps.exp().clamp(min=0.02, max=10.0)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        scores: (B, H, L_q, L_k)
        Returns attention weights with per-head temperature applied.
        """
        τ = self.temperatures.view(1, -1, 1, 1)
        return F.softmax(scores / τ, dim=-1)

    def extra_repr(self) -> str:
        temps = self.temperatures.detach()
        return (f"num_heads={len(temps)}, "
                f"τ=[{temps.min():.3f}..{temps.max():.3f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Legendre-Fenchel dual activation
# ═══════════════════════════════════════════════════════════════════════════════

class LFDualActivation(nn.Module):
    """
    Tropical polynomial activation with its Legendre-Fenchel dual.

    Primal activation (tropical polynomial in one variable, applied elementwise):
        f(x) = max_k ( s_k · x + b_k )
    This is a piecewise-linear convex function.  The breakpoints x_k where
    consecutive pieces intersect define the Newton polytope of the tropical variety.

    Legendre-Fenchel conjugate (also a tropical polynomial!):
        f*(y) = sup_x { x·y − f(x) }
              = max_j { x_j · y − f(x_j) }   [evaluated at grid points x_j]
    The conjugate partitions y-space by which evaluation point x_j wins —
    a dual set of tropical Voronoi cells.

    mode  'primal'  → f(x)  : partition by piece index  (primal cells)
    mode  'dual'    → f*(x) : partition by conjugate index (dual cells)
    mode  'blend'   → learnable σ(g)·f(x) + (1−σ(g))·f*(x) per channel
    """

    def __init__(self, dim: int, num_pieces: int = 8, mode: str = "primal"):
        super().__init__()
        self.mode       = mode
        self.num_pieces = num_pieces

        # Primal: s_k and b_k
        # Initialise slopes spread around zero so no piece dominates initially
        self.slopes = nn.Parameter(torch.linspace(-1.5, 1.5, num_pieces))
        self.biases = nn.Parameter(torch.zeros(num_pieces))

        # Dual grid: x_j values at which to evaluate f
        self.x_grid = nn.Parameter(torch.linspace(-3.0, 3.0, num_pieces))

        # Blend mode gate (per channel)
        if mode == "blend":
            self.blend_gate = nn.Parameter(torch.zeros(dim))

    # ── primal ───────────────────────────────────────────────────────────────
    def f_primal(self, x: torch.Tensor) -> torch.Tensor:
        """f(x) = max_k(s_k·x + b_k).  Shape: (..., D) → (..., D)."""
        pieces = x.unsqueeze(-1) * self.slopes + self.biases   # (..., D, K)
        return _tropical_max(pieces, dim=-1)                      # (..., D)

    # ── dual ─────────────────────────────────────────────────────────────────
    def f_dual(self, y: torch.Tensor) -> torch.Tensor:
        """
        f*(y) = max_j { x_j·y − f(x_j) }
        f(x_j) is the primal evaluated at each grid point x_j.
        """
        # f at each grid point: (K,)
        f_at_grid = _tropical_max(
            self.x_grid.unsqueeze(-1) * self.slopes + self.biases,
            dim=-1)                                               # (K,)

        # dual pieces: (..., D, K)
        dual_pieces = y.unsqueeze(-1) * self.x_grid - f_at_grid
        return _tropical_max(dual_pieces, dim=-1)                 # (..., D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "primal":
            return self.f_primal(x)
        elif self.mode == "dual":
            return self.f_dual(x)
        else:   # blend
            g = torch.sigmoid(self.blend_gate)
            return g * self.f_primal(x) + (1.0 - g) * self.f_dual(x)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Tropical multi-head attention
# ═══════════════════════════════════════════════════════════════════════════════

class TropicalMultiHeadAttention(nn.Module):
    """
    Multi-head attention fusing tropical and classical score computation,
    normalised via Maslov temperature.

    Per head:
      trop_score(q, k)  = max_i(q_i + k_i) / √d_k     [tropical inner product]
      class_score(q, k) = (q · k)         / √d_k     [Euclidean inner product]
      blended_score     = g(q) · trop + (1−g(q)) · classic
                          g : ℝ^d_k → (0,1) per query position (learned)
      attn_weights      = MaslovSoftmax(blended_score, τ_head)
      output            = attn_weights @ V  →  out_proj

    Q, K: TropicalLinear — encodes queries/keys as tropical Voronoi routing
    V:    Classical Linear — preserves smooth gradient flow in values
    """

    def __init__(
        self,
        d_model:      int,
        num_heads:    int,
        dropout:      float = 0.1,
        trop_dropout: float = 0.05,
        init_temp:    float = 1.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self._scale    = self.d_k ** -0.5

        # Q, K via classical linear (for stable learning) + tropical score path
        # The novelty is in the SCORE FUNCTION (max-plus inner product), not projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Score blend gate: conditioned on pre-split input (d_model → num_heads)
        # so each head gets its own gate value per query position
        self.score_gate = nn.Linear(d_model, num_heads)
        # Initialize gate bias negative so sigmoid → ~0.12, favoring classical
        # path early. Tropical routing activates as the gate learns.
        nn.init.zeros_(self.score_gate.weight)
        nn.init.constant_(self.score_gate.bias, -2.0)

        # Maslov temperature (one per head)
        self.maslov = MaslovTemperature(num_heads, init_temp)

        self.attn_dropout = nn.Dropout(dropout)
        self.trop_dropout = TropicalDropout(trop_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) → (B, H, L, d_k)"""
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x    : (B, L, d_model)
        mask : (B, 1, L, L) optional causal / padding mask
        Returns: (output, attention_weights)
        """
        B, L, D = x.shape

        # ── Projections ───────────────────────────────────────────────────────
        Q = self._split_heads(self.q_proj(x))                       # (B, H, L, d_k)
        K = self._split_heads(self.k_proj(x))                       # (B, H, L, d_k)
        V = self._split_heads(self.v_proj(x))                       # (B, H, L, d_k)

        # ── Classical scores: Euclidean dot product (clean Q, K) ─────────────
        class_scores = torch.matmul(Q, K.transpose(-2, -1)) * self._scale

        # ── Tropical scores: max-plus inner product ───────────────────────────
        # Tropical scores: max-plus inner product on clean Q, K
        # (TropicalDropout removed — its -1e9 values corrupt gated blending)
        # trop_score[b,h,i,j] = max_f( Q[b,h,i,f] + K[b,h,j,f] )
        trop_scores = (
            Q.unsqueeze(3) + K.unsqueeze(2)
        )
        trop_scores = _tropical_max(trop_scores, dim=-1) * self._scale  # (B, H, L, L)

        # ── Per-head, per-query score gate ────────────────────────────────────
        # g: (B, L, H) → (B, H, L, 1) to broadcast over keys
        g = torch.sigmoid(self.score_gate(x))                     # (B, L, H)
        g = g.permute(0, 2, 1).unsqueeze(-1)                      # (B, H, L, 1)
        scores = g * trop_scores + (1.0 - g) * class_scores       # (B, H, L, L)

        # ── Masking ───────────────────────────────────────────────────────────
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # ── Maslov temperature attention weights ──────────────────────────────
        attn = self.maslov(scores)                                 # (B, H, L, L)
        attn = self.attn_dropout(attn)

        # ── Aggregate values ──────────────────────────────────────────────────
        out = torch.matmul(attn, V)                                # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), attn


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Tropical hybrid FFN
# ═══════════════════════════════════════════════════════════════════════════════

class TropicalHybridFFN(nn.Module):
    """
    Feed-forward block with parallel tropical and classical branches,
    using Legendre-Fenchel dual activations on the tropical side.

    Tropical branch:  x → TropLinear → LFDualActivation → TropDropout
    Classical branch: x → Linear     → GELU
    Fusion:           σ(G·x) · trop  +  (1−σ(G·x)) · classical
    Output:           down_proj → dropout → LayerNorm + residual
    """

    def __init__(
        self,
        d_model:      int,
        ffn_dim:      int,
        dropout:      float = 0.1,
        trop_dropout: float = 0.05,
        lf_pieces:    int   = 8,
        lf_mode:      str   = "blend",
    ):
        super().__init__()
        self.trop_up    = TropicalLinear(d_model, ffn_dim)
        self.lf_act     = LFDualActivation(ffn_dim, num_pieces=lf_pieces, mode=lf_mode)
        # Use standard dropout here — TropicalDropout (-1e9) would corrupt
        # the gated fusion (classical multiplication), not a max operation.
        self.trop_drop  = nn.Dropout(trop_dropout)

        self.class_up   = nn.Linear(d_model, ffn_dim)
        self.gelu       = nn.GELU()

        # Gate conditioned on input for the fusion
        # Initialize to favor classical branch early (sigmoid(-2) ≈ 0.12)
        self.gate_proj  = nn.Linear(d_model, ffn_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

        self.down_proj  = nn.Linear(ffn_dim, d_model)
        self.dropout    = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trop      = self.trop_drop(self.lf_act(self.trop_up(x)))
        classical = self.gelu(self.class_up(x))

        g     = torch.sigmoid(self.gate_proj(x))
        fused = g * trop + (1.0 - g) * classical

        out = self.dropout(self.down_proj(fused))
        return self.norm(out + x)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Tropical transformer block
# ═══════════════════════════════════════════════════════════════════════════════

class TropicalTransformerBlock(nn.Module):
    """
    One TropFormer block.

      x → PreNorm → TropicalMHA → + x  (attention sub-layer)
        → PreNorm → TropicalFFN          (FFN sub-layer has internal norm+residual)

    Pre-LayerNorm (as in GPT-2 / Llama) improves gradient flow with tropical
    layers, whose subgradients can be sparse.
    """

    def __init__(
        self,
        d_model:      int,
        num_heads:    int,
        ffn_dim:      int,
        dropout:      float = 0.1,
        trop_dropout: float = 0.05,
        lf_pieces:    int   = 8,
        lf_mode:      str   = "blend",
        init_temp:    float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = TropicalMultiHeadAttention(
            d_model, num_heads, dropout, trop_dropout, init_temp
        )
        self.drop1 = nn.Dropout(dropout)
        self.ffn   = TropicalHybridFFN(
            d_model, ffn_dim, dropout, trop_dropout, lf_pieces, lf_mode
        )

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(self.norm1(x), mask)
        x = x + self.drop1(attn_out)
        x = self.ffn(x)
        return x, attn_w


# ═══════════════════════════════════════════════════════════════════════════════
# §7  TropFormer (full model)
# ═══════════════════════════════════════════════════════════════════════════════

class TropFormer(nn.Module):
    """
    Vision TropFormer for image classification.

    Image → patchify → patch_embed → [CLS ++ tokens] + pos_enc
          → [TropicalTransformerBlock × num_layers]
          → LayerNorm → head(CLS token) → logits

    Default config targets MNIST (28×28 grayscale, 10 classes):
      patch_size=7 → 16 patches of dim 49 → embed to d_model=128
      4 blocks, 4 heads, ffn_dim=256

    Can trivially adapt to CIFAR-10 (32×32, patch=8) or others.
    """

    def __init__(
        self,
        img_size:     int   = 28,
        patch_size:   int   = 7,
        in_channels:  int   = 1,
        num_classes:  int   = 10,
        d_model:      int   = 128,
        num_heads:    int   = 4,
        num_layers:   int   = 4,
        ffn_dim:      int   = 256,
        dropout:      float = 0.1,
        trop_dropout: float = 0.05,
        lf_pieces:    int   = 8,
        lf_mode:      str   = "blend",
        init_temp:    float = 1.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size  = patch_size
        num_patches      = (img_size // patch_size) ** 2
        patch_dim        = in_channels * patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, d_model)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)

        # CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.pos_embed,  std=0.02)

        self.embed_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TropicalTransformerBlock(
                d_model, num_heads, ffn_dim,
                dropout, trop_dropout,
                lf_pieces, lf_mode, init_temp,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.zeros_(self.head.bias)

    # ── helpers ───────────────────────────────────────────────────────────────

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, num_patches, C·p·p)"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B, -1, C * p * p)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def maslov_summary(self) -> dict[str, torch.Tensor]:
        """Per-block, per-head Maslov temperature after training."""
        return {
            f"block_{i}": block.attn.maslov.temperatures.detach().cpu()
            for i, block in enumerate(self.blocks)
        }

    def lf_mode_summary(self) -> dict[str, torch.Tensor]:
        """
        For 'blend' mode: per-block mean LF gate value.
        1.0 = fully primal (tropical polynomial).
        0.0 = fully dual (LF conjugate — Newton polytope dual).
        """
        out = {}
        for i, block in enumerate(self.blocks):
            lf = block.ffn.lf_act
            if hasattr(lf, "blend_gate"):
                g = torch.sigmoid(lf.blend_gate).detach().cpu()
                out[f"block_{i}_lf_gate_mean"] = g.mean().item()
        return out

    def score_gate_summary(self) -> dict[str, torch.Tensor]:
        """
        Mean absolute weight of the score-blend gate per block.
        High → gate is decisive (strongly routing tropical vs classical).
        Low  → gate is passive (outputs ≈ 0.5, equal blend).
        """
        return {
            f"block_{i}_score_gate_|W|": (
                block.attn.score_gate.weight.abs().mean().item()
            )
            for i, block in enumerate(self.blocks)
        }

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patchify and embed
        x   = self.patchify(x)                         # (B, N, patch_dim)
        x   = self.patch_embed(x)                      # (B, N, d_model)

        # Prepend CLS token, add positional encoding
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)               # (B, N+1, d_model)
        x   = self.embed_drop(x + self.pos_embed)

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])                      # classify on CLS token


# ═══════════════════════════════════════════════════════════════════════════════
# §8  Data
# ═══════════════════════════════════════════════════════════════════════════════

def get_mnist_loaders(
    batch_size:  int = 128,
    data_dir:    str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """MNIST with standard normalisation, returns (train, test) DataLoaders."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    test  = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    use_pin = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=use_pin)
    return (
        DataLoader(train, shuffle=True,  **kw),
        DataLoader(test,  shuffle=False, **kw),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §9  Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = correct = total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(data)
                loss   = F.cross_entropy(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(data)
            loss   = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        bs          = data.size(0)
        total_loss += loss.item() * bs
        correct    += logits.argmax(1).eq(target).sum().item()
        total      += bs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits      = model(data)
        loss        = F.cross_entropy(logits, target)
        bs          = data.size(0)
        total_loss += loss.item() * bs
        correct    += logits.argmax(1).eq(target).sum().item()
        total      += bs

    return total_loss / total, correct / total


# ═══════════════════════════════════════════════════════════════════════════════
# §10  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Hyperparameters ───────────────────────────────────────────────────────
    EPOCHS       = 25
    BATCH_SIZE   = 128
    LR           = 3e-3
    D_MODEL      = 128
    NUM_HEADS    = 4
    NUM_LAYERS   = 4
    FFN_DIM      = 256
    DROPOUT      = 0.1
    TROP_DROPOUT = 0.05
    LF_PIECES    = 8
    LF_MODE      = "blend"       # 'primal' | 'dual' | 'blend'
    INIT_TEMP    = 1.0           # Maslov temperature init
    DATA_DIR     = "./data"
    SAVE_PATH    = "tropformer_best.pt"

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif DML_AVAILABLE:
        device = torch_directml.device()
    else:
        device = "cpu"
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("=" * 68)
    print("  TropFormer: Tropical Transformer")
    print("  Maslov dequantization - LF dual activations - Tropical MHA")
    print("=" * 68)
    print(f"  Device       : {device}")
    print(f"  d_model      : {D_MODEL}   num_heads : {NUM_HEADS}")
    print(f"  num_layers   : {NUM_LAYERS}   ffn_dim   : {FFN_DIM}")
    print(f"  LF mode      : {LF_MODE} ({LF_PIECES} pieces)")
    print(f"  Maslov tau0  : {INIT_TEMP}  (learned per head)")
    print(f"  Epochs       : {EPOCHS}    LR        : {LR}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE, DATA_DIR)
    print(f"\n  Train : {len(train_loader.dataset):,}   "
          f"Test  : {len(test_loader.dataset):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TropFormer(
        img_size=28, patch_size=7, in_channels=1, num_classes=10,
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM, dropout=DROPOUT, trop_dropout=TROP_DROPOUT,
        lf_pieces=LF_PIECES, lf_mode=LF_MODE, init_temp=INIT_TEMP,
    ).to(device)

    print(f"\n  Parameters   : {model.count_params():,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = EPOCHS * len(train_loader)
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.cuda.amp.GradScaler() if (device == "cuda") else None

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_acc = 0.0
    history  = {k: [] for k in ("train_loss", "train_acc", "test_loss", "test_acc")}

    header = (f"\n{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}"
              f"  {'TeLoss':>8}  {'TeAcc':>7}  {'Time':>6}")
    print(header)
    print("-" * len(header.strip()))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler
        )
        te_loss, te_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        flag = " *" if te_acc > best_acc else ""
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}"
              f"  {te_loss:>8.4f}  {te_acc:>7.4f}  {elapsed:>5.1f}s{flag}")

    # ── Post-training diagnostics ─────────────────────────────────────────────
    print(f"\n  -- Best test accuracy: {best_acc:.4f}  ({best_acc*100:.2f}%) --")
    print(f"  Model saved -> {SAVE_PATH}")

    # Maslov temperatures per block/head
    print("\n  Maslov temperatures (tau per block, per head):")
    print("  (tau ~ 0 = tropical/argmax,  tau = 1 = softmax,  tau > 1 = diffuse)")
    for blk, temps in model.maslov_summary().items():
        bar_parts = []
        for h, t in enumerate(temps.tolist()):
            regime = ("trop" if t < 0.4 else "soft" if t < 1.5 else "diff")
            bar_parts.append(f"h{h}={t:.2f}({regime})")
        print(f"    {blk}: " + "  ".join(bar_parts))

    # LF blend gate (primal vs dual activation usage)
    print("\n  LF dual activation blend (1=primal f, 0=dual f*):")
    for blk, val in model.lf_mode_summary().items():
        bar = "#" * int(val * 30) + "." * (30 - int(val * 30))
        print(f"    {blk}: {val:.3f}  [{bar}]")

    # Score gate norms (how decisive the tropical/classical blend is)
    print("\n  Score gate |W| mean (higher = more decisive trop/classic routing):")
    for blk, val in model.score_gate_summary().items():
        bar = "#" * min(30, int(val * 60))
        print(f"    {blk}: {val:.4f}  {bar}")

    return model, history


if __name__ == "__main__":
    model, history = main()
