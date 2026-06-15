"""
verify_context_memory.py — long-context memory budget verification.
MK III Technologies / TropFormer.

For each tier and each inference context level (1x..8x the training context),
reports the on-device memory required and whether it fits the phone budget:

    memory = quantized_params + KV_cache(active_window + sinks + cross_window)
             + gist_chain

The active attention window is always ``context_len`` (cost is bounded by the
window, NOT the full inference context) — extension comes from the detached
sink / cross-window K/V caches and the O(links) gist chain. This is the whole
point of the architecture: 8x effective context at ~1x window cost.

Also runs a functional check: the InferenceContextManager actually processes a
sequence of length ``inference_ctx`` window-by-window and returns finite logits.

Run:  python verify_context_memory.py
"""

from __future__ import annotations

import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Phone memory budgets (bytes) — generous on-device ceilings.
PHONE_BUDGET = {
    "nano": 2 * 1024**3,   # 2 GB  — always-on phone model
    "pro":  4 * 1024**3,   # 4 GB  — flagship on-device
    "kyro": 64 * 1024**3,  # 64 GB — cloud-only ceiling
}


def estimate_params(cfg) -> int:
    """Rough parameter count from config dims (embeddings + blocks)."""
    d = cfg.d_model
    vocab = 50_000
    embed = vocab * d                      # tied input/output
    per_block = (
        4 * d * d +                        # attention Wq,Wk,Wv,Wo
        2 * d * d +                        # sheaf-ish / ffn-ish projections
        8 * d * d                          # clifford ffn + misc
    )
    return embed + cfg.n_blocks * per_block


def kv_cache_bytes(cfg, bytes_per_elem: int = 1) -> int:
    """K/V cache for active window + sinks + cross-window (int8 by default).
    2 (K and V) x n_blocks x (window + sinks + cross_window) x d_model."""
    window = cfg.context_len
    sinks = getattr(cfg, "sink_tokens", 4)
    xwin = getattr(cfg, "cross_window_size", 0) if cfg.use_cross_window_memory else 0
    tokens = window + sinks + xwin
    return 2 * cfg.n_blocks * tokens * cfg.d_model * bytes_per_elem


def gist_chain_bytes(cfg, inference_ctx: int) -> int:
    """One gist link per evicted window: (key d_model + theta d_model/2 + mag)."""
    n_links = max(0, inference_ctx // cfg.context_len)
    per_link = (cfg.d_model + cfg.d_model // 2 + 1) * 4   # float32
    return n_links * per_link


def verify_tier(tier_fn, tier_name: str) -> bool:
    cfg = tier_fn()
    quant_bytes = max(1, cfg.quantization_bits // 8)
    params = estimate_params(cfg)
    param_mem = params * quant_bytes
    budget = PHONE_BUDGET[tier_name]

    print(f"\n{'='*72}")
    print(f"  Tier: {tier_name.upper()}  "
          f"(d_model={cfg.d_model}, n_blocks={cfg.n_blocks}, "
          f"train_ctx={cfg.context_len}, params~{params/1e6:.0f}M @ "
          f"{cfg.quantization_bits}-bit)")
    print(f"{'='*72}")
    print(f"  {'InfCtx':>8} {'KV cache':>12} {'GistChain':>12} "
          f"{'Total':>12} {'Budget':>10} {'Fit':>5}")

    all_fit = True
    for mult in (1, 2, 4, 8):
        inf_ctx = cfg.context_len * mult
        kv = kv_cache_bytes(cfg, quant_bytes)
        gc = gist_chain_bytes(cfg, inf_ctx)
        total = param_mem + kv + gc
        fit = total <= budget
        all_fit = all_fit and fit
        print(f"  {inf_ctx:>8} {kv/1e6:>10.1f}MB {gc/1e6:>10.2f}MB "
              f"{total/1e9:>10.3f}GB {budget/1e9:>8.0f}GB "
              f"{'OK' if fit else 'XX':>5}")
    return all_fit


def functional_check() -> bool:
    """Confirm the inference machinery actually runs a >1-window sequence."""
    from tsrn_gist import TSRNGist
    from inference_context_manager import InferenceContextManager

    print(f"\n{'='*72}\n  Functional check (tiny TSRNGist proxy)\n{'='*72}")
    torch.manual_seed(0)
    m = TSRNGist(vocab=64, d_model=64, context_len=32,
                 n_blocks=1, n_heads=4, max_gists=8, gist_top_k=2).eval()
    mgr = InferenceContextManager(m, context_len=32, inference_ctx=256)
    mgr.reset_conversation()
    ids = torch.randint(0, 64, (1, 256))
    last = mgr.process_long_context(ids)
    ok = torch.isfinite(last).all().item() and mgr.gist_chain.n_links == 8
    print(f"  processed 256 tokens in 8 windows -> logits {tuple(last.shape)}, "
          f"gist links={mgr.gist_chain.n_links}, finite={bool(torch.isfinite(last).all())}")
    mgr.teardown()
    print(f"  {'OK' if ok else 'XX'}  inference context manager functional")
    return ok


def main() -> int:
    from model_config import nano_config, pro_config, kyro_config

    results = {
        "nano": verify_tier(nano_config, "nano"),
        "pro":  verify_tier(pro_config, "pro"),
        "kyro": verify_tier(kyro_config, "kyro"),
    }
    func_ok = functional_check()

    print(f"\n{'='*72}")
    all_ok = all(results.values()) and func_ok
    for tier, ok in results.items():
        print(f"  {tier:>5}: {'all context levels fit' if ok else 'OVER BUDGET'}")
    print(f"  functional: {'OK' if func_ok else 'FAILED'}")
    print(f"{'='*72}")
    print("VERIFICATION PASSED" if all_ok else "VERIFICATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
