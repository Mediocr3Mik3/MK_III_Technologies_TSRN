"""
research/profile_forward.py
===========================

One-shot forward/backward profiler for TSRNGist.  Identifies which
sub-module dominates the eager-mode ~3-4s/micro-step on L4.

Usage:
    python -m research.profile_forward --preset small_24gb [--use-kleene-ssm --tier nano]

Prints a torch.profiler key-averages table sorted by self CUDA time, then
a manual per-section CUDA-event timing breakdown for the top suspects:
gist_extractor, gist_buffer, sheaf_pe, s1 blocks, rg_pool, s2 loop, head.
"""
from __future__ import annotations
import argparse
import importlib
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Make 'research' importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.tsrn_gist import TSRNGist


def _load_preset(name: str) -> dict:
    mod = importlib.import_module(f"research.cloud.configs.{name}")
    return dict(mod.CONFIG)


def _build_model(cfg: dict, args, device):
    model_cfg = None
    if args.use_kleene_ssm or args.tier:
        from research.model_config import nano_config, pro_config, kyro_config
        from dataclasses import replace
        TIERS = {"nano": nano_config, "pro": pro_config, "kyro": kyro_config}
        model_cfg = TIERS[args.tier or "nano"]()
        model_cfg = replace(
            model_cfg,
            d_model=cfg["d_model"],
            context_len=cfg["context_len"],
            n_heads=cfg["n_heads"],
            top_k=cfg.get("top_k", 16),
            tropical_matmul_mode=args.tropical_mode,
            tropical_matmul_h=args.tropical_h,
        )
    model = TSRNGist(
        vocab=205,
        d_model=cfg["d_model"],
        context_len=cfg["context_len"],
        gradient_checkpoint=False,                # off so timings are clean
        n_blocks=cfg["n_blocks"],
        top_k=cfg.get("top_k", 16),
        n_heads=cfg["n_heads"],
        mem_depth=cfg.get("mem_depth", 7),
        max_gists=cfg.get("max_gists", 64),
        gist_top_k=cfg.get("gist_top_k", 4),
        dropout=0.0,
        config=model_cfg,
    ).to(device)
    return model


@torch.no_grad()
def _section(label: str, fn, *args, n=5, warmup=2, **kw):
    """Time a callable with CUDA events; print mean ms over n iters."""
    for _ in range(warmup):
        fn(*args, **kw)
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(n):
        fn(*args, **kw)
    ender.record()
    torch.cuda.synchronize()
    ms = starter.elapsed_time(ender) / n
    print(f"  {label:30s} {ms:8.2f} ms")
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small_24gb")
    p.add_argument("--use-kleene-ssm", action="store_true")
    p.add_argument("--tier", default=None)
    p.add_argument("--tropical-mode", default="auto",
                   choices=["auto", "soft", "triton", "naive"])
    p.add_argument("--tropical-h", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for this profiler.")
        return

    device = torch.device("cuda")
    cfg = _load_preset(args.preset)
    print(f"\n[profile] preset={args.preset}  kleene={args.use_kleene_ssm}  "
          f"tier={args.tier}  mode={args.tropical_mode}  h={args.tropical_h}\n")

    model = _build_model(cfg, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    B, T = cfg["batch_size"], cfg["context_len"]
    x = torch.randint(0, 200, (B, T), device=device, dtype=torch.long)
    y = torch.randint(0, 200, (B, T), device=device, dtype=torch.long)

    amp = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # --- Warm up + fwd+bwd timing ----------------------------------------
    print("\n[warmup]")
    for _ in range(3):
        with amp:
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    print("\n[end-to-end fwd+bwd, mean over %d iters]" % args.steps)
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
    print(f"  fwd+bwd: {ms:.2f} ms / micro-step  (target on L4: ~150-300 ms)")

    # --- torch.profiler key-averages -------------------------------------
    print("\n[torch.profiler key-averages]")
    from torch.profiler import profile, ProfilerActivity, schedule
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        schedule=schedule(wait=0, warmup=1, active=3, repeat=1),
    ) as prof:
        for _ in range(4):
            with amp:
                _, loss = model(x, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            prof.step()
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=20,
    ))


if __name__ == "__main__":
    main()
