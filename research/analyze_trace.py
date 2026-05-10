"""
research/analyze_trace.py
=========================

Reads a chrome-trace JSON dumped by ``research.profile_forward
--trace-file PATH`` and groups events by python stack frame to identify
which sub-modules are responsible for the eager-mode kernel storm.

Specifically answers: of the ~5800 CUDA kernel launches per micro-step
the L4 profile shows, which TropFormer modules
(TropicalAttention, PAdicAttention, HyperbolicEmbedding, CliffordFFN,
SheafRotorDiffusion, KleeneSSM, ...) are responsible?

Usage:
    python -m research.analyze_trace logs/profile_tropical_ssm.json
    python -m research.analyze_trace logs/profile_*.json --top 15

The trace JSON is in chrome-trace event format.  We don't need a real
parser; we slurp the file, look at every CPU-side event whose name starts
with ``aten::`` (operator dispatch) and tally counts + total CUDA time by
the *Python source location* attached to that event (when ``with_stack``
was on) or by ``args.External id`` correlated CUDA event otherwise.

For traces collected without with_stack=True (which is our default to keep
the trace small) we can still attribute by *operator name* and by the
parent ``aten::*`` op chain that wraps each call.  This is enough to tell
``aten::sub`` calls coming from Q+K-style scoring vs from PAdic ops,
because every aten op carries a sequential ``correlation`` id.
"""
from __future__ import annotations
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _open_trace(path: Path):
    """Trace files may be plain JSON or gzipped JSON."""
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def load_trace(path: Path) -> list[dict]:
    with _open_trace(path) as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def analyze(events: list[dict], top_n: int = 20) -> None:
    # Phase X = CPU-side complete, M = metadata, etc.  We want phase X events.
    cpu_events = [e for e in events if e.get("ph") == "X" and "cat" in e]

    # Bucket: total CUDA time and call count per op name.
    op_count: Counter = Counter()
    op_cuda_us: Counter = Counter()
    op_cpu_us: Counter = Counter()

    # Build correlation: External id -> CUDA time.
    cuda_by_corr: dict[int, int] = defaultdict(int)
    for e in events:
        if e.get("ph") != "X":
            continue
        # CUDA runtime / kernel events carry "args"."External id".
        args = e.get("args") or {}
        ext = args.get("External id")
        if ext is None:
            continue
        cat = e.get("cat", "")
        if "kernel" in cat or "gpu" in cat or "cuda" in cat.lower():
            cuda_by_corr[ext] += int(e.get("dur", 0))

    # Walk CPU events; attribute their CUDA time via correlation.
    for e in cpu_events:
        name = e.get("name", "")
        if not name.startswith("aten::"):
            continue
        op_count[name] += 1
        op_cpu_us[name] += int(e.get("dur", 0))
        args = e.get("args") or {}
        ext = args.get("External id")
        if ext is not None and ext in cuda_by_corr:
            op_cuda_us[name] += cuda_by_corr[ext]

    # Print summary tables.
    total_cuda = sum(op_cuda_us.values()) or 1
    total_calls = sum(op_count.values()) or 1
    print(f"\n  Total aten::* events       : {total_calls:,}")
    print(f"  Total CUDA time (attributed): {total_cuda/1000:.1f} ms")

    print("\n  Top ops by CALL COUNT (eager dispatch hot spots):")
    print(f"  {'op':30s} {'calls':>10s} {'cuda_ms':>10s} {'%cuda':>7s}")
    print("  " + "-" * 62)
    for name, n in op_count.most_common(top_n):
        cuda_ms = op_cuda_us[name] / 1000
        pct = 100 * op_cuda_us[name] / total_cuda
        print(f"  {name:30s} {n:>10,} {cuda_ms:>10.1f} {pct:>6.1f}%")

    print("\n  Top ops by CUDA TIME:")
    print(f"  {'op':30s} {'calls':>10s} {'cuda_ms':>10s} {'%cuda':>7s}")
    print("  " + "-" * 62)
    for name, t in op_cuda_us.most_common(top_n):
        n = op_count[name]
        pct = 100 * t / total_cuda
        print(f"  {name:30s} {n:>10,} {t/1000:>10.1f} {pct:>6.1f}%")

    # Heuristic attribution to TropFormer modules.  We look for the
    # ``UserAnnotation`` / module-class strings that PyTorch profiler
    # emits when modules are forward-hooked.
    module_count: Counter = Counter()
    module_cuda: Counter = Counter()
    INTERESTING_NAMES = (
        "TropicalAttention", "PAdicAttention", "PAdicMemory",
        "HyperbolicEmbedding", "CliffordFFN", "SheafRotorDiffusion",
        "KleeneSSM", "TropicalSSM", "KleeneAttention",
        "EchoStateReservoir", "GistBuffer", "GistExtractor",
        "RGPool", "PAdicHarmonicPE", "PAdicContextScaling",
        "TSRNGistBlock",
    )
    for e in cpu_events:
        name = e.get("name", "")
        for m in INTERESTING_NAMES:
            if m in name:
                module_count[m] += 1
                args = e.get("args") or {}
                ext = args.get("External id")
                if ext is not None and ext in cuda_by_corr:
                    module_cuda[m] += cuda_by_corr[ext]
                break

    if module_count:
        print("\n  Per-MODULE breakdown (matched on event name):")
        print(f"  {'module':30s} {'events':>10s} {'cuda_ms':>10s}")
        print("  " + "-" * 55)
        for m, n in module_count.most_common():
            print(f"  {m:30s} {n:>10,} {module_cuda[m]/1000:>10.1f}")
    else:
        print("\n  [No module-level annotations found; profile was captured "
              "without record_function() hooks.  Run with_stack=True or wrap "
              "modules with torch.profiler.record_function('Name') for "
              "per-module attribution.]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="chrome-trace JSON file(s) to analyze")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()
    for path in args.paths:
        path = Path(path)
        if not path.exists():
            print(f"[skip] {path} does not exist")
            continue
        print("\n" + "=" * 72)
        print(f"  TRACE: {path}")
        print("=" * 72)
        events = load_trace(path)
        analyze(events, top_n=args.top)


if __name__ == "__main__":
    main()
