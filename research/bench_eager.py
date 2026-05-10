"""
research/bench_eager.py
=======================

Self-contained eager-mode forward+backward timing.  Constructs the
``TSRNGist`` (or ``TSRN`` if Gist isn't present) at the small_24gb
config and times N micro-steps with CUDA events.  No torch.compile,
no gradient-accumulation, no profiler — just raw eager wall-clock.

Designed to work on BOTH the current branch AND historical branches
(e.g. ``nvidia-cloud``) by tolerating constructor-signature drift via
inspect.signature filtering of kwargs.

Usage:
    python -m research.bench_eager --steps 10 --batch 8 --ctx 512
"""
from __future__ import annotations
import argparse
import inspect
import time
import torch


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Return only the kwargs that ``cls.__init__`` actually accepts."""
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--vocab", type=int, default=205)  # enwik8 byte-level
    p.add_argument("--label", default="")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    device = torch.device("cuda")

    # Try the new TSRNGist first; fall back to TSRN if unavailable.
    Model = None
    try:
        from research.tsrn_gist import TSRNGist as Model
        print(f"  using TSRNGist from research.tsrn_gist")
    except Exception:
        try:
            from research.tsrn_dml import TSRN as Model
            print(f"  using TSRN from research.tsrn_dml")
        except Exception as e:
            print(f"  ERROR: could not import a model class: {e}")
            return

    base_kwargs = dict(
        vocab=args.vocab,
        d_model=args.d_model,
        context_len=args.ctx,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        top_k=16, mem_depth=7,
        max_gists=64, gist_top_k=4,
        dropout=0.1,
        use_hyperbolic=False,
        gist_chaining=False,
    )
    kwargs = _filter_kwargs(Model, base_kwargs)
    print(f"  branch label   : {args.label or '(unset)'}")
    print(f"  ctor kwargs    : {sorted(kwargs.keys())}")

    model = Model(**kwargs).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params         : {n_params:,}")

    x = torch.randint(0, args.vocab, (args.batch, args.ctx), device=device)
    y = torch.randint(0, args.vocab, (args.batch, args.ctx), device=device)
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(args.warmup):
        with amp:
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Time
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(args.steps):
        with amp:
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    ender.record()
    torch.cuda.synchronize()

    ms = starter.elapsed_time(ender) / args.steps
    print(f"\n  >>> {args.label:<24s} fwd+bwd: {ms:8.2f} ms / micro-step  "
          f"(B={args.batch}, T={args.ctx}, eager)")


if __name__ == "__main__":
    main()
