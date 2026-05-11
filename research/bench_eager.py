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
import sys
import types
from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Compatibility shims — injected BEFORE any research.* imports
# ---------------------------------------------------------------------------

# 1. nvidia-cloud's tsrn_dml.py calls torch.utils.checkpoint.checkpoint()
#    without importing the submodule.  Pre-import it here.
import torch.utils.checkpoint  # noqa: F401

# 2. nvidia-cloud's tsrn_gist.py uses bare `from tsrn_dml import ...` and
#    bare `from hyperbolic_embeddings import ...`.  Add the research/ dir to
#    sys.path so those imports resolve to whichever real files exist in the
#    current worktree.  This must happen BEFORE any stubs are registered.
_research_dir = Path(__file__).parent
if str(_research_dir) not in sys.path:
    sys.path.insert(0, str(_research_dir))

# 3. Stub out `hyperbolic_embeddings` ONLY IF the real module is not present
#    in the current worktree (nvidia-cloud lacks it; HEAD has it).  If we
#    stubbed unconditionally, sys.modules would shadow the real module and
#    block legitimate symbols like `HyperbolicEmbedding`.  Since bench_eager
#    sets use_hyperbolic=False, identity-function stubs for the two functions
#    referenced at module-import time are sufficient when stubbing is needed.
if (
    "hyperbolic_embeddings" not in sys.modules
    and not (_research_dir / "hyperbolic_embeddings.py").exists()
):
    _stub = types.ModuleType("hyperbolic_embeddings")

    def _identity(x):  # never actually called (use_hyperbolic=False)
        return x

    _stub.poincare_to_tangent = _identity
    _stub.tangent_to_poincare = _identity
    sys.modules["hyperbolic_embeddings"] = _stub


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
    # We try three import paths to cover both current branch (research.*
    # package) and older branches (bare module names, nvidia-cloud style).
    Model = None
    import importlib, traceback
    errors = []
    for mod_path, cls_name in [
        ("research.tsrn_gist", "TSRNGist"),
        ("tsrn_gist",          "TSRNGist"),  # nvidia-cloud bare-name style
        ("research.tsrn_dml",  "TSRN"),
        ("tsrn_dml",           "TSRN"),
    ]:
        try:
            mod = importlib.import_module(mod_path)
            Model = getattr(mod, cls_name)
            print(f"  using {cls_name} from {mod_path}")
            break
        except Exception as e:
            errors.append((mod_path, cls_name, repr(e),
                           traceback.format_exc(limit=3)))
            continue
    if Model is None:
        print("  ERROR: could not import any model class. Attempts:")
        for mp, cn, er, tb in errors:
            print(f"    - {mp}.{cn}: {er}")
            print("      " + tb.replace("\n", "\n      "))
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
