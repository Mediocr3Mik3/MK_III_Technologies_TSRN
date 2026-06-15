"""Full-model step benchmark: EchoStateReservoir vs OscillatoryMemory.

Measures a single forward+backward step on nano_directml config on the
AMD RX 6750 XT (DirectML). Also records param counts.

Run: .venv312\\Scripts\\python.exe research\\_dml_reservoir_variant_bench.py
"""
import time, sys, os, math, torch, torch_directml

_RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from tsrn_gist import TSRNGist
from model_config import ModelConfig, nano_config

V = 256
B, T = 4, 256
N_RUNS = 10
WARMUP = 3


def make_cfg(kind: str):
    cfg = nano_config(
        vocab_size=V,
        use_reservoir=True,
        reservoir_kind=kind,
        use_linear_attention=False,
    )
    return cfg


def step_once(model, x, y):
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    return loss.item()


def bench(kind: str):
    cfg = make_cfg(kind)
    model = TSRNGist(vocab=V, d_model=cfg.d_model, context_len=cfg.context_len,
                     n_blocks=cfg.n_blocks, top_k=8, n_heads=cfg.n_heads,
                     mem_depth=cfg.padic_depth, max_gists=cfg.max_gists,
                     dropout=cfg.dropout, config=cfg)
    dev = torch_directml.device()
    model = model.to(dev)
    torch.manual_seed(42)
    x = torch.randint(0, V, (B, T), device=dev)
    y = torch.randint(0, V, (B * T,), device=dev)

    # warmup
    for _ in range(WARMUP):
        model.zero_grad(set_to_none=True)
        _ = step_once(model, x, y)

    # timed
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        model.zero_grad(set_to_none=True)
        loss = step_once(model, x, y)
    # force GPU completion by evaluating scalar on CPU
    loss_val = float(loss)
    elapsed = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "kind": kind,
        "loss": loss,
        "ms_per_step": (elapsed / N_RUNS) * 1000,
        "params": n_params,
    }


def main():
    print("=" * 60)
    print("Reservoir variant benchmark (nano_directml, B=4, T=256)")
    print("=" * 60)
    results = [bench("echo"), bench("oscillatory")]
    for r in results:
        print(f"\n  {r['kind']:12s}: loss={r['loss']:.4f} | "
              f"step={r['ms_per_step']:.1f} ms | params={r['params']:,}")
    r0, r1 = results
    speedup = r0["ms_per_step"] / r1["ms_per_step"]
    print(f"\n  Speedup oscillatory vs echo: {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
