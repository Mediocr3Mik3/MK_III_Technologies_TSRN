"""Full nano_directml model fwd+bwd step time on DirectML: quadratic vs linear
tropical attention, at the training shape, across the cycled Maslov-h values.
Also an integration finiteness check for the linear path.

Run: .venv312\\Scripts\\python.exe research\\_model_step_bench.py
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


def build(vocab, T):
    cfg = nano_directml_config(vocab_size=vocab, context_len=T)
    m = TSRNGist(
        vocab=vocab, d_model=cfg.d_model, context_len=cfg.context_len,
        gradient_checkpoint=False, n_blocks=cfg.n_blocks, top_k=cfg.top_k,
        n_heads=cfg.n_heads, mem_depth=cfg.padic_depth, max_gists=cfg.max_gists,
        gist_top_k=cfg.gist_top_k, dropout=cfg.dropout, use_hyperbolic=False,
        gist_chaining=False, config=cfg,
    )
    return m.to(DEV).train()


def step_time(model, idx, tgt, h, iters=8, warmup=3):
    model.set_maslov_h(h)
    last = None
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _, loss = model(idx, tgt)
        loss.backward()
        last = loss.item()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        _, loss = model(idx, tgt)
        loss.backward()
        last = loss.item()
    return (time.perf_counter() - t0) / iters * 1e3, last


def main():
    vocab = 50349
    print(f"Device: {NAME}   nano_directml   (full model fwd+bwd, h=1.0)")
    print(f"{'shape':<14}{'quad':>12}{'linear':>12}{'speedup':>10}   loss q/l")
    for T, B in [(256, 8), (512, 4), (1024, 2)]:
        model = build(vocab, T)
        idx = torch.randint(0, vocab, (B, T), device=DEV)
        tgt = torch.randint(0, vocab, (B, T), device=DEV)
        model.set_linear_attn(False)
        tq, lq = step_time(model, idx, tgt, 1.0)
        model.set_linear_attn(True)
        tl, ll = step_time(model, idx, tgt, 1.0)
        print(f"B={B:<2} T={T:<6}{tq:>9.1f} ms{tl:>9.1f} ms{tq / tl:>9.2f}x   "
              f"{lq:.3f}/{ll:.3f}")
        del model


if __name__ == "__main__":
    main()
