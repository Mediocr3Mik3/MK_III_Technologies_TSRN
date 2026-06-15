"""DirectML hot-path profiler for TSRNGist.

Answers three questions for one training step (real nano config, B=8, T=256):
  1. Which ops fall back to the CPU? (each = a GPU<->CPU round trip)
  2. How is time split between forward and backward?
  3. Which block component dominates? (attn / sheaf / reservoir /
     tropical_ssm / ffn / mem / pa ...)

Run:
    .venv312/Scripts/python.exe research/_dml_profile.py
"""
from __future__ import annotations

import gc
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

_R = Path(__file__).resolve().parent
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))

import torch  # noqa: E402

from model_config import nano_directml_config  # noqa: E402
from tsrn_gist import TSRNGist  # noqa: E402
from tsrn_dml import AdamWDML  # noqa: E402


def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        print("  (torch_directml not found - CPU)")
        return torch.device("cpu")


def main():
    dev = get_device()
    is_dml = dev.type not in ("cpu", "cuda")
    print(f"device: {dev}")
    torch.manual_seed(0)

    def sync():
        # DirectML executes in-order on one queue, so blocking on a freshly
        # queued scalar guarantees all previously queued work has finished.
        if is_dml:
            torch.ones(1, device=dev).add_(1.0).item()
        elif dev.type == "cuda":
            torch.cuda.synchronize()

    # --- dedup + print every unique CPU-fallback op ---
    seen = set()
    _orig_show = warnings.showwarning

    def _show(message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        if "fall back to run on the CPU" in msg or "not currently supported on the DML" in msg:
            op = msg.split("'")[1] if "'" in msg else msg[:80]
            if op not in seen:
                seen.add(op)
            return
        _orig_show(message, category, filename, lineno, file, line)

    warnings.showwarning = _show
    warnings.simplefilter("always")

    cfg = nano_directml_config(d_model=256, n_heads=4, n_blocks=3,
                               context_len=256, max_gists=32)
    model = TSRNGist(
        vocab=cfg.vocab_size, d_model=cfg.d_model, context_len=cfg.context_len,
        n_blocks=cfg.n_blocks, top_k=cfg.top_k, n_heads=cfg.n_heads,
        mem_depth=cfg.padic_depth, max_gists=cfg.max_gists,
        gist_top_k=cfg.gist_top_k, dropout=0.0, config=cfg,
    ).to(dev)
    model.train()
    print(f"params: {model.count_params():,}")

    # B=4 (not 8) for profiler headroom: running many heterogeneous phases in
    # one process fragments DirectML's cached pool. Relative breakdown holds.
    B, T = 4, cfg.context_len
    opt = AdamWDML(model.parameters(), lr=3e-4)

    # ---- 1. capture fallbacks over 2 full train steps ----
    for _ in range(2):
        x = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
        y = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
    sync()
    print("\n=== CPU-fallback ops in one train step ===")
    for op in sorted(seen):
        print(f"  [FALLBACK] {op}")
    if not seen:
        print("  (none)")

    # ---- 2. forward vs backward timing (backward each iter frees the graph) ----
    del loss
    gc.collect()
    x = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    _, loss = model(x, y); loss.backward()  # warmup
    model.zero_grad(set_to_none=True); del loss; gc.collect(); sync()

    n = 3
    t_fwd = t_bwd = 0.0
    for _ in range(n):
        model.zero_grad(set_to_none=True)
        sync(); t0 = time.perf_counter()
        _, loss = model(x, y)
        loss.item()
        t_fwd += time.perf_counter() - t0
        sync(); t0 = time.perf_counter()
        loss.backward(); sync()
        t_bwd += time.perf_counter() - t0
        del loss
    t_fwd /= n
    t_bwd /= n
    model.zero_grad(set_to_none=True); gc.collect()

    print(f"\n=== timing (B={B}, T={T}) ===")
    print(f"  forward : {t_fwd*1000:8.1f} ms")
    print(f"  backward: {t_bwd*1000:8.1f} ms")
    print(f"  fwd+bwd : {(t_fwd+t_bwd)*1000:8.1f} ms")

    # ---- 2b. B=8 single-phase timing (real training shape; validates that
    #          the raised attn fast-path threshold doesn't OOM at B=8) ----
    gc.collect()
    try:
        x8 = torch.randint(0, cfg.vocab_size, (8, T), device=dev)
        y8 = torch.randint(0, cfg.vocab_size, (8, T), device=dev)
        _, loss = model(x8, y8); loss.backward()
        model.zero_grad(set_to_none=True); del loss; gc.collect(); sync()
        m = 3
        tf = tb = 0.0
        for _ in range(m):
            model.zero_grad(set_to_none=True)
            sync(); t0 = time.perf_counter()
            _, loss = model(x8, y8); loss.item()
            tf += time.perf_counter() - t0
            sync(); t0 = time.perf_counter()
            loss.backward(); sync()
            tb += time.perf_counter() - t0
            del loss
        tf /= m; tb /= m
        model.zero_grad(set_to_none=True); gc.collect()
        print(f"\n=== B=8 timing (real training shape) ===")
        print(f"  forward : {tf*1000:8.1f} ms")
        print(f"  backward: {tb*1000:8.1f} ms")
        print(f"  fwd+bwd : {(tf+tb)*1000:8.1f} ms"
              f"  (x4 grad-accum ~ {4*(tf+tb)*1000:.0f} ms/step)")
    except RuntimeError as e:
        print(f"\n=== B=8 timing: FAILED ({str(e)[:90]}) ===")

    # ---- 3. per-component forward breakdown ----
    comp = defaultdict(float)

    def wrap(mod, label):
        orig = mod.forward

        def timed(*a, **k):
            sync(); t0 = time.perf_counter()
            out = orig(*a, **k)
            sync(); comp[label] += time.perf_counter() - t0
            return out
        mod.forward = timed

    for _mn, mod in model.named_modules():
        if "Block" in type(mod).__name__:
            for cname, child in mod.named_children():
                wrap(child, cname)
    # Also time the big non-block costs (embed, head over 50k vocab, final LN).
    for tl in ("embed", "head", "ln_f"):
        if hasattr(model, tl):
            wrap(getattr(model, tl), tl)

    # no_grad keeps this memory-safe (forward-only relative cost).
    with torch.no_grad():
        for _ in range(2):
            _, loss = model(x, y)
            loss.item()
            del loss

    print(f"\n=== forward per-component (sum over all blocks, 2 fwd, no_grad) ===")
    total = sum(comp.values()) or 1.0
    for label, t in sorted(comp.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<16} {t*1000:8.1f} ms  ({100*t/total:4.1f}%)")


if __name__ == "__main__":
    main()
