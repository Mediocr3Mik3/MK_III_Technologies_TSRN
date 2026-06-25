"""Per-component forward-time breakdown of the full nano_directml model on
DirectML, to find the TRUE dominant cost (the attention module bench
over-represented attention; this measures in-model shares directly).

Times are wall-clock per module-class, with an explicit DML sync (a 1-elem
.cpu() transfer) at each module boundary so async dispatch doesn't smear the
attribution. Forward-only (backward cost roughly mirrors forward).

Run: .venv312\\Scripts\\python.exe research\\_dml_component_profile.py
"""
import time
import collections
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

TARGETS = {
    "TropicalAttention", "KleeneAttention", "SheafRotorDiffusion", "SheafDiffusion",
    "EchoStateReservoir", "KleeneSSM", "TropicalSSM", "CliffordFFN",
    "PAdicMemory", "PAdicAttention", "RGPool", "SheafHarmonicPE", "Embedding",
}

times = collections.defaultdict(float)
counts = collections.defaultdict(int)
_t = {}


def _sync(x):
    if isinstance(x, torch.Tensor):
        x.detach().reshape(-1)[:1].cpu()
    elif isinstance(x, (tuple, list)):
        for e in x:
            _sync(e)


def pre_hook(mod, inp):
    _sync(inp)
    _t[id(mod)] = time.perf_counter()


def post_hook(mod, inp, out):
    _sync(out)
    dt = time.perf_counter() - _t[id(mod)]
    name = type(mod).__name__
    times[name] += dt * 1e3
    counts[name] += 1


def main():
    B, T, vocab = 8, 256, 50349
    cfg = nano_directml_config(vocab_size=vocab, context_len=T)
    model = TSRNGist(
        vocab=vocab, d_model=cfg.d_model, context_len=cfg.context_len,
        gradient_checkpoint=False, n_blocks=cfg.n_blocks, top_k=cfg.top_k,
        n_heads=cfg.n_heads, mem_depth=cfg.padic_depth, max_gists=cfg.max_gists,
        gist_top_k=cfg.gist_top_k, dropout=cfg.dropout, use_hyperbolic=False,
        gist_chaining=False, config=cfg,
    ).to(DEV).eval()

    for m in model.modules():
        if type(m).__name__ in TARGETS:
            m.register_forward_pre_hook(pre_hook)
            m.register_forward_hook(post_hook)

    idx = torch.randint(0, vocab, (B, T), device=DEV)
    with torch.no_grad():
        for _ in range(3):  # warmup
            model(idx)
        times.clear(); counts.clear()
        iters = 5
        t0 = time.perf_counter()
        for _ in range(iters):
            model(idx)
        _sync(torch.zeros(1, device=DEV))
        full_fwd = (time.perf_counter() - t0) / iters * 1e3

    print(f"Device: {NAME}  nano_directml  B={B} T={T}  (forward-only, {iters} iters)")
    print(f"Full forward: {full_fwd:.1f} ms/iter")
    print(f"{'component':<22}{'ms/iter':>10}{'% of step':>11}{'calls':>7}")
    total_attrib = 0.0
    for name, tot in sorted(times.items(), key=lambda kv: -kv[1]):
        ms = tot / 5
        total_attrib += ms
        print(f"{name:<22}{ms:>9.1f} {100 * ms / full_fwd:>9.1f}% {counts[name] // 5:>7}")
    print(f"{'[attributed]':<22}{total_attrib:>9.1f} {100 * total_attrib / full_fwd:>9.1f}%")
    print(f"{'[unattributed]':<22}{full_fwd - total_attrib:>9.1f} "
          f"{100 * (full_fwd - total_attrib) / full_fwd:>9.1f}%  (embed proj, head, gist, glue)")


if __name__ == "__main__":
    main()
