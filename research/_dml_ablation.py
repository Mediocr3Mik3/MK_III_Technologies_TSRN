"""Controlled component ablation on the full nano_directml model (DirectML).

Measures each component's TRUE marginal cost by toggling it off on ONE model
and timing the full fwd+bwd step delta -- the same reliable method as the
attention A/B (unlike the sync-profiler, which over-attributes to pipelined
parallel modules). Per-block flags are saved/restored so the original
per-block pattern (e.g. reservoir only on s1 block 0) is preserved.

Run: .venv312\\Scripts\\python.exe research\\_dml_ablation.py
"""
import time
import torch
from model_config import nano_directml_config
from tsrn_gist import TSRNGist

try:
    import torch_directml
    DEV = torch_directml.device()
    NAME = "DirectML"
except ImportError:
    DEV = torch.device("cpu")
    NAME = "CPU"


def blocks(model):
    return list(model.s1_blocks) + [model.s2_block]


def get_flags(model, attr):
    return [getattr(b, attr, None) for b in blocks(model)]


def set_all(model, attr, val):
    for b in blocks(model):
        if hasattr(b, attr):
            setattr(b, attr, val)


def restore(model, attr, vals):
    for b, v in zip(blocks(model), vals):
        if v is not None:
            setattr(b, attr, v)


def step_time(model, idx, tgt, iters=8, warmup=3):
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _, loss = model(idx, tgt)
        loss.backward()
        _ = loss.item()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        _, loss = model(idx, tgt)
        loss.backward()
        _ = loss.item()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    B, T, vocab = 8, 256, 50349
    cfg = nano_directml_config(vocab_size=vocab, context_len=T)
    model = TSRNGist(
        vocab=vocab, d_model=cfg.d_model, context_len=cfg.context_len,
        gradient_checkpoint=False, n_blocks=cfg.n_blocks, top_k=cfg.top_k,
        n_heads=cfg.n_heads, mem_depth=cfg.padic_depth, max_gists=cfg.max_gists,
        gist_top_k=cfg.gist_top_k, dropout=cfg.dropout, use_hyperbolic=False,
        gist_chaining=False, config=cfg,
    ).to(DEV).train()
    model.set_maslov_h(1.0)
    idx = torch.randint(0, vocab, (B, T), device=DEV)
    tgt = torch.randint(0, vocab, (B, T), device=DEV)

    base = step_time(model, idx, tgt)
    print(f"Device: {NAME}  nano_directml  B={B} T={T}  (full fwd+bwd)")
    print(f"Baseline step: {base:.1f} ms")
    print(f"{'ablation':<26}{'step ms':>10}{'delta':>10}{'% of step':>11}")

    for attr in ("use_reservoir", "use_tropical_ssm", "use_memory", "use_padic_attn"):
        saved = get_flags(model, attr)
        set_all(model, attr, False)
        t = step_time(model, idx, tgt)
        restore(model, attr, saved)
        print(f"{'OFF: ' + attr:<26}{t:>9.1f} {base - t:>9.1f} {100 * (base - t) / base:>9.1f}%")

    # attention as a swap (linear replaces quadratic), not a removal
    model.set_linear_attn(True)
    t = step_time(model, idx, tgt)
    model.set_linear_attn(False)
    print(f"{'SWAP: linear attention':<26}{t:>9.1f} {base - t:>9.1f} {100 * (base - t) / base:>9.1f}%")


if __name__ == "__main__":
    main()
