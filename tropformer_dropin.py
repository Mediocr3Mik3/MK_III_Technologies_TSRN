"""
TropFormer Drop-in Replacement
===============================
API-compatible with torch.nn.TransformerEncoderLayer / TransformerDecoderLayer.

A downstream user can swap:
    nn.TransformerEncoderLayer → TropicalEncoderLayer
with no other code changes. The tropical machinery is an internal detail.

Module hierarchy:
    TropicalEncoderLayer          — single encoder block (attn + FFN)
    TropicalDecoderLayer          — single decoder block (self-attn + cross-attn + FFN)
    TropicalEncoder               — stack of N TropicalEncoderLayers
    TropicalDecoder               — stack of N TropicalDecoderLayers
    TropicalTransformer           — encoder + decoder (seq2seq)
    TropicalEncoderModel          — encoder-only (BERT-style)
    TropicalDecoderModel          — decoder-only (GPT-style)

Vision front-ends:
    TropicalViT                   — image patch embedding + TropicalEncoder
    TropicalViTForClassification  — TropicalViT + classification head
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# §0  Tropical Primitives (from tropformer.py, proven working)
# =============================================================================

def _tropical_max(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Tropical max: hard forward, native backward (CPU/CUDA)."""
    return x.max(dim=dim).values


class TropicalLinear(nn.Module):
    """
    Max-plus linear layer with STE gradient stabilization.
        y_i = max_j(W_ij + x_j) + b_i

    Forward: exact tropical max (hard routing preserved).
    Backward: softmax-weighted smooth approximation (STE).
    """
    _ste_temp = 1.0

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.uniform_(self.weight, -0.5, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        scores = self.weight.unsqueeze(0) + x_flat.unsqueeze(1)
        hard = scores.max(dim=-1).values
        if self.training:
            soft_w = F.softmax(scores / self._ste_temp, dim=-1)
            soft = (soft_w * scores).sum(dim=-1)
            out = hard.detach() + (soft - soft.detach())
        else:
            out = hard
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*leading, self.out_features)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}"


class MaslovTemperature(nn.Module):
    """Learnable Maslov dequantization temperature, one per attention head."""

    def __init__(self, num_heads: int, init_temp: float = 1.0):
        super().__init__()
        self.log_temps = nn.Parameter(
            torch.full((num_heads,), math.log(init_temp))
        )

    @property
    def temperatures(self) -> torch.Tensor:
        return self.log_temps.exp().clamp(min=0.02, max=10.0)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        τ = self.temperatures.view(1, -1, 1, 1)
        return F.softmax(scores / τ, dim=-1)


class LFDualActivation(nn.Module):
    """
    Tropical polynomial activation with its Legendre-Fenchel dual.
    Primal: f(x) = max_k(s_k·x + b_k)
    Dual:   f*(y) = max_j(x_j·y - f(x_j))
    """

    def __init__(self, dim: int, num_pieces: int = 8, mode: str = "blend"):
        super().__init__()
        self.mode = mode
        self.num_pieces = num_pieces
        self.slopes = nn.Parameter(torch.linspace(-1.5, 1.5, num_pieces))
        self.biases = nn.Parameter(torch.zeros(num_pieces))
        self.x_grid = nn.Parameter(torch.linspace(-3.0, 3.0, num_pieces))
        if mode == "blend":
            self.blend_gate = nn.Parameter(torch.zeros(dim))

    def f_primal(self, x: torch.Tensor) -> torch.Tensor:
        pieces = x.unsqueeze(-1) * self.slopes + self.biases
        return _tropical_max(pieces, dim=-1)

    def f_dual(self, y: torch.Tensor) -> torch.Tensor:
        f_at_grid = _tropical_max(
            self.x_grid.unsqueeze(-1) * self.slopes + self.biases, dim=-1
        )
        dual_pieces = y.unsqueeze(-1) * self.x_grid - f_at_grid
        return _tropical_max(dual_pieces, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "primal":
            return self.f_primal(x)
        elif self.mode == "dual":
            return self.f_dual(x)
        else:
            g = torch.sigmoid(self.blend_gate)
            return g * self.f_primal(x) + (1.0 - g) * self.f_dual(x)


# =============================================================================
# §1  KV Cache for Autoregressive Inference
# =============================================================================

@dataclass
class TropicalKVCache:
    """
    Caches K and V tensors from previous steps for autoregressive decoding.
    The tropical score max_i(Q_i + K_i) is correct across the full cache.
    """
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = self.v = None


# =============================================================================
# §2  Tropical Multi-Head Attention (supports self-attn and cross-attn)
# =============================================================================

class TropicalMultiHeadAttention(nn.Module):
    """
    Multi-head attention with tropical + classical score blending.

    Supports both self-attention and cross-attention:
    - Self-attention: Q, K, V all from the same input
    - Cross-attention: Q from tgt, K/V from memory

    The score gate conditions on:
    - Self-attention: gate_proj(x) → (B, L, H)
    - Cross-attention: gate_proj(cat(tgt_ctx, mem_ctx)) → (B, H)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        init_temp: float = 1.0,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self._scale = self.d_k ** -0.5
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Score blend gate
        if is_cross_attention:
            self.score_gate = nn.Linear(2 * d_model, num_heads)
        else:
            self.score_gate = nn.Linear(d_model, num_heads)
        nn.init.zeros_(self.score_gate.weight)
        nn.init.constant_(self.score_gate.bias, -2.0)

        self.maslov = MaslovTemperature(num_heads, init_temp)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[TropicalKVCache] = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L_q, D = query.shape

        Q = self._split_heads(self.q_proj(query))
        K = self._split_heads(self.k_proj(key))
        V = self._split_heads(self.v_proj(value))

        # Update KV cache if provided
        if cache is not None:
            K, V = cache.update(K, V)

        L_k = K.shape[2]

        # Classical scores
        class_scores = torch.matmul(Q, K.transpose(-2, -1)) * self._scale

        # Tropical scores: max-plus inner product
        trop_scores = (
            Q.unsqueeze(3) + K.unsqueeze(2)
        ).max(dim=-1).values * self._scale

        # Score gate
        if self.is_cross_attention:
            q_ctx = query.mean(dim=1)
            k_ctx = key.mean(dim=1)
            gate_input = torch.cat([q_ctx, k_ctx], dim=-1)
            g = torch.sigmoid(self.score_gate(gate_input))  # (B, H)
            g = g.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        else:
            g = torch.sigmoid(self.score_gate(query))  # (B, L_q, H)
            g = g.permute(0, 2, 1).unsqueeze(-1)  # (B, H, L_q, 1)

        scores = g * trop_scores + (1.0 - g) * class_scores

        # Apply masks
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                scores = scores + attn_mask.unsqueeze(1)
            else:
                scores = scores + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (B, L_k), True = ignore
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_k)
            scores = scores.masked_fill(mask, -1e9)

        # Maslov softmax
        attn = self.maslov(scores)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)
        out = self.out_proj(out)

        return out, attn if need_weights else None


# =============================================================================
# §3  Tropical Hybrid FFN
# =============================================================================

class TropicalHybridFFN(nn.Module):
    """
    Feed-forward block with parallel tropical and classical branches.
    Tropical: TropLinear → LFDualActivation → Dropout
    Classical: Linear → GELU
    Fusion: gate * trop + (1-gate) * classical → down_proj → dropout
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.1,
        lf_pieces: int = 8,
        lf_mode: str = "blend",
    ):
        super().__init__()
        self.trop_up = TropicalLinear(d_model, ffn_dim)
        self.lf_act = LFDualActivation(ffn_dim, num_pieces=lf_pieces, mode=lf_mode)
        self.trop_drop = nn.Dropout(dropout)

        self.class_up = nn.Linear(d_model, ffn_dim)
        self.gelu = nn.GELU()

        self.gate_proj = nn.Linear(d_model, ffn_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

        self.down_proj = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trop = self.trop_drop(self.lf_act(self.trop_up(x)))
        classical = self.gelu(self.class_up(x))
        g = torch.sigmoid(self.gate_proj(x))
        fused = g * trop + (1.0 - g) * classical
        return self.dropout(self.down_proj(fused))


# =============================================================================
# §4  TropicalEncoderLayer — drop-in for nn.TransformerEncoderLayer
# =============================================================================

class TropicalEncoderLayer(nn.Module):
    """
    Drop-in replacement for torch.nn.TransformerEncoderLayer.

    API matches PyTorch conventions:
        forward(src, src_mask=None, src_key_padding_mask=None)

    Internally uses tropical-classical hybrid attention and FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        lf_pieces: int = 8,
        lf_mode: str = "blend",
        init_temp: float = 1.0,
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = TropicalMultiHeadAttention(
            d_model, nhead, dropout=dropout, init_temp=init_temp,
        )
        self.ffn = TropicalHybridFFN(
            d_model, dim_feedforward, dropout=dropout,
            lf_pieces=lf_pieces, lf_mode=lf_mode,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.batch_first:
            src = src.transpose(0, 1)

        if self.norm_first:
            x = src
            attn_out, _ = self.self_attn(
                self.norm1(x), self.norm1(x), self.norm1(x),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            x = x + self.dropout1(attn_out)
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        else:
            x = src
            attn_out, _ = self.self_attn(
                x, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            x = self.norm1(x + self.dropout1(attn_out))
            x = self.norm2(x + self.dropout2(self.ffn(x)))

        if not self.batch_first:
            x = x.transpose(0, 1)
        return x


# =============================================================================
# §5  TropicalDecoderLayer — drop-in for nn.TransformerDecoderLayer
# =============================================================================

class TropicalDecoderLayer(nn.Module):
    """
    Drop-in replacement for torch.nn.TransformerDecoderLayer.

    API matches PyTorch conventions:
        forward(tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        lf_pieces: int = 8,
        lf_mode: str = "blend",
        init_temp: float = 1.0,
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.self_attn = TropicalMultiHeadAttention(
            d_model, nhead, dropout=dropout, init_temp=init_temp,
        )
        self.cross_attn = TropicalMultiHeadAttention(
            d_model, nhead, dropout=dropout, init_temp=init_temp,
            is_cross_attention=True,
        )
        self.ffn = TropicalHybridFFN(
            d_model, dim_feedforward, dropout=dropout,
            lf_pieces=lf_pieces, lf_mode=lf_mode,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.batch_first:
            tgt = tgt.transpose(0, 1)
            memory = memory.transpose(0, 1)

        if self.norm_first:
            x = tgt
            # Self-attention
            h = self.norm1(x)
            sa_out, _ = self.self_attn(h, h, h, attn_mask=tgt_mask,
                                       key_padding_mask=tgt_key_padding_mask)
            x = x + self.dropout1(sa_out)
            # Cross-attention
            h = self.norm2(x)
            ca_out, _ = self.cross_attn(h, memory, memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)
            x = x + self.dropout2(ca_out)
            # FFN
            x = x + self.dropout3(self.ffn(self.norm3(x)))
        else:
            x = tgt
            sa_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask,
                                       key_padding_mask=tgt_key_padding_mask)
            x = self.norm1(x + self.dropout1(sa_out))
            ca_out, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)
            x = self.norm2(x + self.dropout2(ca_out))
            x = self.norm3(x + self.dropout3(self.ffn(x)))

        if not self.batch_first:
            x = x.transpose(0, 1)
        return x


# =============================================================================
# §6  TropicalEncoder / TropicalDecoder stacks
# =============================================================================

class TropicalEncoder(nn.Module):
    """Stack of N TropicalEncoderLayers with optional final LayerNorm."""

    def __init__(self, encoder_layer: TropicalEncoderLayer, num_layers: int,
                 norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(self, src: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                          src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TropicalDecoder(nn.Module):
    """Stack of N TropicalDecoderLayers with optional final LayerNorm."""

    def __init__(self, decoder_layer: TropicalDecoderLayer, num_layers: int,
                 norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                          memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


# =============================================================================
# §7  TropicalTransformer (encoder + decoder, seq2seq)
# =============================================================================

class TropicalTransformer(nn.Module):
    """
    Full encoder-decoder transformer with tropical-classical hybrid layers.
    Drop-in replacement for torch.nn.Transformer.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        lf_pieces: int = 8,
        lf_mode: str = "blend",
        init_temp: float = 1.0,
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        enc_layer = TropicalEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            lf_pieces, lf_mode, init_temp, batch_first, norm_first,
        )
        self.encoder = TropicalEncoder(
            enc_layer, num_encoder_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
        )

        dec_layer = TropicalDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            lf_pieces, lf_mode, init_temp, batch_first, norm_first,
        )
        self.decoder = TropicalDecoder(
            dec_layer, num_decoder_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encoder(src, mask=src_mask,
                             src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
        return output


# =============================================================================
# §8  Encoder-only and Decoder-only models
# =============================================================================

class TropicalEncoderModel(nn.Module):
    """Encoder-only model (BERT-style). Wraps TropicalEncoder with embedding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        lf_pieces: int = 8,
        lf_mode: str = "blend",
        init_temp: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)
        self.embed_scale = math.sqrt(d_model)

        enc_layer = TropicalEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            lf_pieces, lf_mode, init_temp,
        )
        self.encoder = TropicalEncoder(
            enc_layer, num_layers, norm=nn.LayerNorm(d_model),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) * self.embed_scale + self.pos_embed(positions)
        x = self.embed_drop(x)

        # Convert attention_mask (1=attend, 0=ignore) to key_padding_mask (True=ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class TropicalDecoderModel(nn.Module):
    """Decoder-only model (GPT-style) with causal masking."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        lf_pieces: int = 8,
        lf_mode: str = "blend",
        init_temp: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)
        self.embed_scale = math.sqrt(d_model)

        # Use encoder layers with causal mask (decoder-only = encoder with causal)
        enc_layer = TropicalEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            lf_pieces, lf_mode, init_temp,
        )
        self.layers = TropicalEncoder(
            enc_layer, num_layers, norm=nn.LayerNorm(d_model),
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) * self.embed_scale + self.pos_embed(positions)
        x = self.embed_drop(x)

        causal_mask = make_causal_mask(L, input_ids.device)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        x = self.layers(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.lm_head(x)


# =============================================================================
# §9  Causal mask utility
# =============================================================================

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Upper-triangular additive mask for autoregressive decoding.
    Uses -1e9 (tropical zero) not -inf, to avoid NaN in tropical score path.
    Shape: (seq_len, seq_len), broadcast over (B, H, L, L).
    """
    return torch.triu(
        torch.full((seq_len, seq_len), -1e9, device=device),
        diagonal=1,
    )


# =============================================================================
# §10  Vision front-ends
# =============================================================================

class TropicalViT(nn.Module):
    """
    Vision Transformer with TropicalEncoder backbone.
    Image → patchify → embed → CLS + pos → TropicalEncoder → norm
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        lf_pieces: int = 8,
        lf_mode: str = "blend",
        init_temp: float = 1.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, d_model)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.embed_drop = nn.Dropout(dropout)

        enc_layer = TropicalEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            lf_pieces, lf_mode, init_temp,
        )
        self.encoder = TropicalEncoder(
            enc_layer, num_layers, norm=nn.LayerNorm(d_model),
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B, -1, C * p * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patchify(x)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.embed_drop(x + self.pos_embed)
        return self.encoder(x)


class TropicalViTForClassification(nn.Module):
    """TropicalViT + classification head on CLS token."""

    def __init__(self, num_classes: int = 10, **vit_kwargs):
        super().__init__()
        self.vit = TropicalViT(**vit_kwargs)
        d_model = vit_kwargs.get("d_model", 768)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit(x)
        return self.head(features[:, 0])

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# §11  Diagnostics
# =============================================================================

def maslov_summary(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract per-head Maslov temperatures from any model containing TropicalMultiHeadAttention."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, MaslovTemperature):
            out[name] = module.temperatures.detach().cpu()
    return out


def lf_blend_summary(model: nn.Module) -> dict[str, float]:
    """Extract LF blend gate values from any model containing LFDualActivation."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, LFDualActivation) and hasattr(module, "blend_gate"):
            g = torch.sigmoid(module.blend_gate).detach().cpu().mean().item()
            out[name] = g
    return out
