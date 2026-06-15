"""
InferenceContextManager — orchestrates TSRN long-context inference.
MK III Technologies / TropFormer.

Combines four mechanisms that together extend a model trained at ``context_len``
to a much longer effective inference context, at bounded cost:

  1. Attention sinks       — first N tokens stay permanently visible
                             (SinkTokenCache, opt-in eval-only).
  2. Cross-window KV cache  — previous window's K/V is prepended to the current
                             window (PersistentCrossWindowMemory, opt-in eval).
  3. PaCS                    — p-adic context scaling softens attention at
                             compressed positions (TropicalAttention, eval-only,
                             active when model.inference_ctx > context_len).
  4. Gist chain             — each evicted window is compressed to one gist
                             summary retrievable by tropical similarity
                             (GistChainContext, gist models only).

All four are causal: every cached/retrieved item comes from a strictly-past
window, and the per-window forward pass remains causally masked internally.

Usage:
    mgr = InferenceContextManager(model, context_len=1024, inference_ctx=8192)
    mgr.reset_conversation()
    logits_last = mgr.process_long_context(token_ids)   # (B, vocab)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class InferenceContextManager:
    def __init__(self, model: nn.Module, context_len: int,
                 inference_ctx: Optional[int] = None,
                 use_sinks: bool = True,
                 use_cross_window: bool = True,
                 use_gist_chain: bool = True,
                 gist_top_k: int = 8):
        self.model = model
        self.context_len = int(context_len)
        self.inference_ctx = int(inference_ctx or context_len)
        self.use_sinks = use_sinks
        self.use_cross_window = use_cross_window
        self.use_gist_chain = use_gist_chain
        self.gist_top_k = gist_top_k

        self.device = next(model.parameters()).device

        # Discover capabilities without assuming a specific model class.
        self._has_sinks = hasattr(model, "set_sink_enabled")
        self._has_cross_window = hasattr(model, "set_cross_window_enabled")
        self._has_inference_ctx = hasattr(model, "inference_ctx")
        self._gist_extractor = getattr(model, "gist_extractor", None)
        self._gist_buffer = getattr(model, "gist_buffer", None)

        self.gist_chain = None
        if self.use_gist_chain and self._gist_extractor is not None \
                and self._gist_buffer is not None:
            from tsrn_gist import GistChainContext
            d_model = getattr(model, "d", None) or getattr(model, "d_model")
            # Gist theta is the Clifford-rotor angle vector of size d_model//2,
            # matching GistBuffer.dh (NOT the attention head dim).
            dh = getattr(self._gist_buffer, "dh", d_model // 2)
            self.gist_chain = GistChainContext(d_model, dh, device=self.device)

    # -- lifecycle ----------------------------------------------------------

    def reset_conversation(self) -> None:
        """Reset all caches and enable the long-context mechanisms."""
        self.model.eval()
        if self._has_sinks:
            self.model.set_sink_enabled(self.use_sinks)
            self.model.reset_sinks()
        if self._has_cross_window:
            self.model.set_cross_window_enabled(self.use_cross_window)
            self.model.reset_cross_window()
        if self._has_inference_ctx:
            self.model.inference_ctx = self.inference_ctx
        if self.gist_chain is not None:
            self.gist_chain.reset()

    def teardown(self) -> None:
        """Disable the opt-in caches (return model to plain eval behaviour)."""
        if self._has_sinks:
            self.model.set_sink_enabled(False)
        if self._has_cross_window:
            self.model.set_cross_window_enabled(False)
        if self._has_inference_ctx:
            self.model.inference_ctx = self.context_len

    # -- core ---------------------------------------------------------------

    @torch.no_grad()
    def process_long_context(self, token_ids: Tensor) -> Tensor:
        """Feed a long sequence through the model window-by-window.

        token_ids: (B, L) or (L,) token ids, L may greatly exceed context_len.
        Returns the logits for the final position, (B, vocab).

        Windows are processed sequentially with sinks + cross-window enabled, so
        each window attends to the previous window's K/V and the permanent
        sinks. After each window we compress it into the gist chain.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids.to(self.device)
        B, L = token_ids.shape
        W = self.context_len

        last_logits = None
        for start in range(0, L, W):
            window = token_ids[:, start:start + W]
            out = self.model(window)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            last_logits = logits[:, -1, :]
            if self.gist_chain is not None:
                self._chain_append(window)
        return last_logits

    @torch.no_grad()
    def retrieve_gist_context(self, query_window: Tensor):
        """Retrieve top-k past-window gists by tropical similarity (or None)."""
        if self.gist_chain is None or self._gist_buffer is None:
            return None
        x = self.model.embed(query_window.to(self.device))
        key = self._gist_buffer.key_proj(x[:, 0, :])           # (B, d)
        return self.gist_chain.retrieve(key[0], top_k=self.gist_top_k)

    # -- helpers ------------------------------------------------------------

    @torch.no_grad()
    def _chain_append(self, window: Tensor) -> None:
        """Compress a processed window to one gist link and append it."""
        if window.shape[1] < 2:
            return
        x = self.model.embed(window)
        pe = getattr(self.model, "sheaf_pe", None) or getattr(self.model, "padic_pe", None)
        if pe is not None:
            try:
                x = x + pe(window.shape[1], x.device, x.dtype).unsqueeze(0)
            except Exception:
                pass
        theta, mag, _ = self._gist_extractor.forward_single(x)
        key = self._gist_buffer.key_proj(x[:, 0, :])
        self.gist_chain.append(key[0], theta[0], mag[0])


__all__ = ["InferenceContextManager"]
