"""Forward+backward wall-time of TropicalAttention on DirectML:
quadratic (current) vs linear (new) at the training shape and long contexts.

Run: .venv312\\Scripts\\python.exe research\\_linattn_bench.py
"""
import time
import torch
from tsrn_dml import TropicalAttention

try:
    import torch_directml
    DEV = torch_directml.device()
    NAME = "DirectML"
except ImportError:
    DEV = torch.device("cpu")
    NAME = "CPU"

D, H = 256, 4


def bench(T, B, linear, iters=10, warmup=3):
    att = TropicalAttention(d_model=D, n_heads=H, top_k=16).to(DEV).train()
    att.linear_attn = linear
    x = torch.randn(B, T, D, device=DEV, requires_grad=True)
    for _ in range(warmup):
        att.zero_grad(set_to_none=True)
        loss = att(x, causal=True).float().pow(2).mean()
        loss.backward()
        _ = loss.item()
    t0 = time.perf_counter()
    for _ in range(iters):
        att.zero_grad(set_to_none=True)
        loss = att(x, causal=True).float().pow(2).mean()
        loss.backward()
        _ = loss.item()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    print(f"Device: {NAME}   d_model={D} heads={H}")
    print(f"{'shape':<18}{'quadratic':>14}{'linear':>14}{'speedup':>10}")
    for T, B in [(256, 8), (512, 8), (1024, 2), (2048, 2)]:
        try:
            q = bench(T, B, linear=False)
            ql = f"{q:8.1f} ms"
        except RuntimeError as e:
            q, ql = None, f"OOM/{str(e)[:8]}"
        try:
            l = bench(T, B, linear=True)
            ll = f"{l:8.1f} ms"
        except RuntimeError as e:
            l, ll = None, f"OOM/{str(e)[:8]}"
        sp = f"{q / l:6.2f}x" if (q and l) else "  --"
        print(f"B={B:<2} T={T:<5}    {ql:>14}{ll:>14}{sp:>10}")


if __name__ == "__main__":
    main()
