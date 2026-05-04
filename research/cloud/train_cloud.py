"""
train_cloud — auto-detecting cloud trainer for TSRNGist.
========================================================

Single executable that runs on:

  * 1×  GPU  (RTX 4090 / A100-40 / A100-80 / H100 / L40S / V100 / T4)
  * N×  GPU  via ``torchrun --nproc_per_node=N``  (DDP, NCCL all-reduce)
  * CPU      (smoke-test only)

Auto-detects:
  - Device + GPU count
  - bf16 (Ampere+) vs fp16 (older) AMP dtype
  - Whether it is the rank-0 process (only rank-0 logs / saves)
  - Memory budget → batch size + grad-accum (when ``--auto-batch`` set)

Architectural fidelity:
  Same TSRNGist model as the DML branch (research/tsrn_gist.py).
  Same Maslov-h cycling, sheaf-harmonic PE, RG fixed-point S2.
  Same eval / checkpoint cadence as research/tsrn_convergence_gist.py.

Differences from the DML trainer:
  - ``torch.optim.AdamW(fused=True)``        — replaces AdamWDML
  - ``torch.cummax`` via fastpath patch       — replaces Hillis-Steele scan
  - ``torch.cuda.amp.autocast`` + GradScaler — replaces fp32-only training
  - ``torch.nn.parallel.DistributedDataParallel``
  - rank-0-gated I/O                         — no double-saves under DDP

Launch
------
Single GPU::

    python -m research.cloud.train_cloud --preset medium_40gb \
        --steps 100000 --tag run0

Multi-GPU (8×A100)::

    torchrun --standalone --nproc_per_node=8 \
        -m research.cloud.train_cloud --preset multi_8xa100 \
        --steps 100000 --tag multi8

Resume::

    python -m research.cloud.train_cloud --preset medium_40gb \
        --resume checkpoints/<tag>_step050000.pt
"""

from __future__ import annotations

import argparse
import datetime
import gc
import importlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# The legacy DML stack imports each other as top-level modules
# (``from tsrn_dml import ...``).  At repo root there is also a stale
# top-level ``tsrn_dml.py`` that shadows ``research/tsrn_dml.py`` on
# ``sys.path``, so we explicitly prepend the research directory.
_RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

# Install CUDA fastpaths *before* importing the model, so TropicalSSM picks up
# torch.cummax instead of the Hillis-Steele scan inside research.tsrn_dml.
from research.cloud import tsrn_cuda
tsrn_cuda.install_cuda_fastpaths()
from research.cloud.wandb_logger import WandBLogger

# Now import model + dataset + eval helpers (legacy top-level style)
from tsrn_gist import TSRNGist, load_enwik8                               # type: ignore
from tsrn_dml import (                                                    # type: ignore
    CharDataset, evaluate, evaluate_sequential, get_lr,
)
try:
    from model_config import ModelConfig, nano_config, pro_config, kyro_config  # type: ignore
    _TIER_MAP = {"nano": nano_config, "pro": pro_config, "kyro": kyro_config}
except Exception:
    ModelConfig = None  # type: ignore
    _TIER_MAP = {}
# Import v2.0 components
from tropical_tokenizer import TropicalMergingTokenizer                      # type: ignore
try:
    from tsrn_inference import generate_with_stats, run_quality_benchmark  # type: ignore
except Exception:  # pragma: no cover
    generate_with_stats = None
    run_quality_benchmark = None


# ---------------------------------------------------------------------------
#  Config presets
# ---------------------------------------------------------------------------

PRESET_MODULES = {
    "small_24gb":   "research.cloud.configs.small_24gb",
    "medium_40gb":  "research.cloud.configs.medium_40gb",
    "large_80gb":   "research.cloud.configs.large_80gb",
    "multi_8xa100": "research.cloud.configs.multi_8xa100",
}


def load_preset(name: str) -> Dict[str, Any]:
    if name not in PRESET_MODULES:
        raise ValueError(f"Unknown preset {name!r}. Choices: {list(PRESET_MODULES)}")
    mod = importlib.import_module(PRESET_MODULES[name])
    return dict(mod.CONFIG)


# ---------------------------------------------------------------------------
#  Filename helpers
# ---------------------------------------------------------------------------

def build_run_tag(user_tag: str = "") -> str:
    date = datetime.datetime.now().strftime("%Y%m%d")
    script = "train_cloud"
    suffix = f"_{user_tag}" if user_tag else ""
    return f"{date}_{script}{suffix}"


def ckpt_path(run_tag: str, step: int, kind: str = "step") -> str:
    if kind == "best":
        return f"checkpoints/{run_tag}_best.pt"
    if kind == "final":
        return f"checkpoints/{run_tag}_final_step{step:06d}.pt"
    return f"checkpoints/{run_tag}_step{step:06d}.pt"


def results_path(run_tag: str, step: int, kind: str = "progress") -> str:
    if kind == "final":
        return f"results/{run_tag}_final_step{step:06d}.json"
    return f"results/{run_tag}_progress_step{step:06d}.json"


# ---------------------------------------------------------------------------
#  NEXUS Maslov-h schedule (mirrors research/tsrn_convergence_gist.py)
# ---------------------------------------------------------------------------

def maslov_h_schedule(step: int, n_steps: int,
                      h_warm: float = 1.5, h_cool: float = 0.3,
                      n_cycles: int = 3) -> float:
    t = max(0.0, min(1.0, step / max(1, n_steps)))
    mid = 0.5 * (h_warm + h_cool)
    amp = 0.5 * (h_warm - h_cool)
    return float(mid + amp * math.cos(2.0 * math.pi * n_cycles * t))


# ---------------------------------------------------------------------------
#  Save helpers (rank-0 only)
# ---------------------------------------------------------------------------

def _unwrap(model: nn.Module) -> nn.Module:
    """Return underlying module if wrapped in DDP/compile."""
    if hasattr(model, "module"):
        return model.module  # DDP
    if hasattr(model, "_orig_mod"):
        return model._orig_mod  # torch.compile
    return model


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: Optional[torch.amp.GradScaler], step: int,
                    best_val_bpc: float, log: List[Dict],
                    config: Dict, run_tag: str) -> None:
    inner = _unwrap(model)
    cpu_state = {k: v.detach().cpu() for k, v in inner.state_dict().items()}
    blob = {
        "model_state_dict":   cpu_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":  scaler.state_dict() if scaler is not None else None,
        "step":               step,
        "best_val_bpc":       best_val_bpc,
        "log":                log,
        "config":             config,
        "run_tag":            run_tag,
    }
    torch.save(blob, path)
    del cpu_state, blob
    gc.collect()


def save_results(path: str, log: List[Dict], step: int, params: int,
                 best_val_bpc: float, run_tag: str, extra: Optional[Dict] = None) -> None:
    payload = {
        "run_tag": run_tag,
        "step": step,
        "params": params,
        "best_val_bpc": round(best_val_bpc, 4),
        "log": log,
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
#  Trainer
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    # --- distributed setup ---
    dist_dev = tsrn_cuda.init_distributed()
    if dist_dev is not None:
        device = dist_dev
    else:
        device = tsrn_cuda.detect_cuda_device(verbose=tsrn_cuda.is_main_process())
    is_main = tsrn_cuda.is_main_process()
    world = tsrn_cuda.get_world_size()
    local_rank = tsrn_cuda.get_local_rank()

    tsrn_cuda.set_global_seed(cfg.get("seed", 42) + tsrn_cuda.get_global_rank())

    # --- AMP policy ---
    amp_dtype, needs_scaler = tsrn_cuda.amp_components(device)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if (use_amp and needs_scaler) else None

    if is_main:
        gpu_mem = tsrn_cuda.get_gpu_memory_gib(device)
        print(f"\n  AMP     : {amp_dtype} (scaler={'on' if scaler else 'off'})")
        print(f"  World   : {world} rank{'s' if world > 1 else ''}, local rank {local_rank}")
        print(f"  GPU mem : {gpu_mem:.1f} GiB")

    # --- dataset ---
    if is_main:
        print(f"\n-- Loading enwik8 (byte-level, standard protocol) --")
    dataset = load_enwik8(context_len=cfg["context_len"])
    V = dataset.vocab_sz

    # --- TMT tokenizer (v2.0) ---
    tmt_tokenizer = None
    if cfg.get("use_tmt_tokenizer", False) and cfg.get("tmt_path"):
        if is_main:
            print(f"-- Loading TMT tokenizer from {cfg['tmt_path']} --")
        tmt_tokenizer = TropicalMergingTokenizer.load(cfg["tmt_path"])
        V = tmt_tokenizer.vocab_size

    # --- ModelConfig (Kleene/tier wiring) ---
    model_cfg = None
    if _TIER_MAP and (cfg.get("use_kleene_ssm") or cfg.get("tier")):
        tier_name = cfg.get("tier", "nano")
        model_cfg = _TIER_MAP.get(tier_name)
        if model_cfg is not None and cfg.get("use_kleene_ssm"):
            from dataclasses import replace  # type: ignore
            model_cfg = replace(model_cfg,
                                d_model=cfg["d_model"],
                                n_heads=cfg["n_heads"],
                                context_len=cfg["context_len"],
                                top_k=cfg.get("top_k", 16))

    # --- model ---
    model = TSRNGist(
        vocab=V, d_model=cfg["d_model"], context_len=cfg["context_len"],
        gradient_checkpoint=cfg.get("gradient_checkpoint", True),
        n_blocks=cfg["n_blocks"], top_k=cfg.get("top_k", 16),
        n_heads=cfg["n_heads"], mem_depth=cfg.get("mem_depth", 7),
        max_gists=cfg.get("max_gists", 64), gist_top_k=cfg.get("gist_top_k", 4),
        dropout=cfg.get("dropout", 0.1),
        # v2.0 features
        use_hyperbolic=cfg.get("use_hyperbolic", False),
        gist_chaining=cfg.get("gist_chaining", False),
        config=model_cfg,
    )

    # --- resume model weights only (optimizer state restored after build) ---
    start_step = 0
    _resume_blob: Optional[Dict] = None
    if args.resume and os.path.exists(args.resume):
        if is_main:
            print(f"  Resume  : {args.resume}")
        _resume_blob = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(_resume_blob["model_state_dict"])
        start_step = _resume_blob.get("step", 0)

    model.to(device)
    model.train()

    # --- compile (optional, single-GPU-only by default) ---
    compile_enabled = cfg.get("compile", False) and world == 1
    if compile_enabled:
        model = tsrn_cuda.maybe_compile(model, mode=cfg.get("compile_mode", "reduce-overhead"))

    # --- DDP wrap ---
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    # --- optimizer ---
    optimizer = tsrn_cuda.make_optimizer(
        _unwrap(model),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.1),
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
        use_8bit=cfg.get("use_8bit_optimizer", False),
    )

    if _resume_blob is not None:
        if "optimizer_state_dict" in _resume_blob:
            try:
                optimizer.load_state_dict(_resume_blob["optimizer_state_dict"])
                if is_main:
                    print(f"  Optim   : restored from checkpoint")
            except Exception as e:
                if is_main:
                    print(f"  Optim   : FRESH (load failed: {type(e).__name__})")
        if scaler is not None and _resume_blob.get("scaler_state_dict"):
            scaler.load_state_dict(_resume_blob["scaler_state_dict"])

    log: List[Dict] = list(_resume_blob.get("log", [])) if _resume_blob else []
    best_val_bpc = float(_resume_blob.get("best_val_bpc", float("inf"))) if _resume_blob else float("inf")
    best_ckpt_path: Optional[str] = None

    if _resume_blob is not None:
        del _resume_blob
        gc.collect()

    # --- run tag and dirs ---
    run_tag = build_run_tag(args.tag)
    if is_main:
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    # --- WandB logger (no-op unless --wandb-project given, rank-0 only) ---
    wb = WandBLogger.maybe_init(args, cfg, run_tag, is_main=is_main,
                                model=_unwrap(model), world_size=world)

    # --- header ---
    n_steps = cfg["steps"]
    batch_size = cfg["batch_size"]
    grad_accum = cfg.get("grad_accum_steps", 1)
    eff_batch = batch_size * grad_accum * world
    eval_every = cfg.get("eval_every", max(1, n_steps // 50))
    ckpt_every = cfg.get("ckpt_every", 5000)

    if is_main:
        n_params = _unwrap(model).count_params()
        print(f"\n  TSRNGist: {n_params:,} parameters")
        print(f"  Vocab   : {V}  |  Context: {cfg['context_len']}  |  d_model: {cfg['d_model']}")
        print(f"  Steps   : {n_steps}  |  Batch: {batch_size}*{grad_accum}*{world}={eff_batch}")
        print(f"  LR      : {cfg['lr']}  |  Eval/{eval_every} steps  |  Ckpt/{ckpt_every}")
        print(f"  Run tag : {run_tag}")
        print(f"\n{'='*88}")
        print(f"  TSRNGist Cloud Trainer  —  enwik8 byte-level  —  {n_steps} steps")
        print(f"{'='*88}")
        print(f"{'Step':>6}  {'TrLoss':>10}  {'TrBPC':>8}  "
              f"{'ValLoss':>9}  {'ValPPL':>9}  {'ValBPC':>8}  "
              f"{'GNorm':>7}  {'ms/step':>10}")
        print(f"{'-'*88}")

    # --- training loop ---
    t0 = time.time()
    last_completed = start_step

    try:
        for step in range(start_step + 1, n_steps + 1):
            # LR schedule (cosine warmup-decay)
            warmup = cfg.get("warmup_steps", min(n_steps // 10, 4000))
            lr = get_lr(step, warmup, n_steps, cfg["lr"], cfg["lr"] * 0.1)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # NEXUS #3 — Maslov-h cycling (per-step scalar)
            h_now = maslov_h_schedule(
                step, n_steps,
                h_warm=cfg.get("maslov_h_warm", 1.5),
                h_cool=cfg.get("maslov_h_cool", 0.3),
                n_cycles=cfg.get("maslov_n_cycles", 3),
            )
            _unwrap(model).set_maslov_h(h_now)

            # Periodic gist-buffer reset
            if step % cfg.get("gist_reset_every", 100) == 1:
                _unwrap(model).gist_buffer.reset()

            # --- gradient accumulation step ---
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for ga in range(grad_accum):
                x, y = dataset.batch("train", batch_size, device)
                # Skip DDP all-reduce on intermediate accum steps
                ddp_sync_ctx = (
                    model.no_sync() if (world > 1 and ga < grad_accum - 1)
                    else _nullcontext()
                )
                with ddp_sync_ctx:
                    if use_amp:
                        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                            _, loss = model(x, y)
                        loss = loss / grad_accum
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    else:
                        _, loss = model(x, y)
                        loss = loss / grad_accum
                        loss.backward()
                accum_loss += loss.item() * grad_accum

            loss_val = accum_loss / grad_accum

            # Gradient clipping (unscale fp16 first)
            if scaler is not None:
                scaler.unscale_(optimizer)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # gc to limit fragmentation on long runs (cheap on CUDA, ~1ms)
            if step % cfg.get("gc_every", 500) == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # --- periodic eval ---
            if (step % eval_every == 0 or step == 1) and is_main:
                _unwrap(model).gist_buffer.reset()
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    val_loss, val_ppl, val_bpc = evaluate(
                        _unwrap(model), dataset, device,
                        batch_size=min(batch_size, 16))
                tr_bpc = loss_val / math.log(2)
                elapsed = time.time() - t0
                ms_step = elapsed / max(1, step - start_step) * 1000
                
                # v2.0 logging: PAdicPE norms and PaCS effective context
                padic_pe_norm = None
                pacs_effective_ctx = None
                if cfg.get("log_padic_pe", False) and hasattr(_unwrap(model), 'sheaf_pe'):
                    with torch.no_grad():
                        pe = _unwrap(model).sheaf_pe(cfg["context_len"], device, torch.float32)
                        padic_pe_norm = pe.norm().item()
                
                if cfg.get("log_pacs", False):
                    pacs_effective_ctx = cfg.get("inference_ctx", cfg["context_len"])
                
                log_entry = {
                    "step": step,
                    "train_loss": round(loss_val, 5),
                    "train_bpc": round(tr_bpc, 4),
                    "val_loss": round(val_loss, 5),
                    "val_ppl": round(val_ppl, 3),
                    "val_bpc": round(val_bpc, 4),
                    "grad_norm": round(float(gnorm), 4),
                    "lr": round(lr, 7),
                    "wall_time_s": round(elapsed, 1),
                    "ms_per_step": round(ms_step, 1),
                    "maslov_h": round(h_now, 4),
                }
                if padic_pe_norm is not None:
                    log_entry["padic_pe_norm"] = round(padic_pe_norm, 4)
                if pacs_effective_ctx is not None:
                    log_entry["pacs_effective_ctx"] = pacs_effective_ctx
                log.append(log_entry)
                wb.log(log_entry, step=step)
                
                print(f"{step:>6}  {loss_val:>10.4f}  {tr_bpc:>8.4f}  "
                      f"{val_loss:>9.4f}  {val_ppl:>9.2f}  {val_bpc:>8.4f}  "
                      f"{float(gnorm):>7.3f}  {ms_step:>8.1f}ms")
                if padic_pe_norm is not None:
                    print(f"         PAdicPE norm: {padic_pe_norm:.4f}")
                if pacs_effective_ctx is not None:
                    print(f"         PaCS effective ctx: {pacs_effective_ctx}")

                if val_bpc < best_val_bpc:
                    best_val_bpc = val_bpc
                    best_ckpt_path = ckpt_path(run_tag, step=0, kind="best")
                    save_checkpoint(
                        best_ckpt_path, model, optimizer, scaler,
                        step, best_val_bpc, log, cfg, run_tag,
                    )
                    wb.log({"best_val_bpc": best_val_bpc}, step=step)

            # --- periodic checkpoint + sample ---
            if step % ckpt_every == 0 and is_main:
                cpath = ckpt_path(run_tag, step, kind="step")
                save_checkpoint(cpath, model, optimizer, scaler,
                                step, best_val_bpc, log, cfg, run_tag)
                print(f"  >> Checkpoint: {cpath}")

                # Inference sample
                if generate_with_stats is not None:
                    _unwrap(model).eval()
                    _unwrap(model).gist_buffer.reset()
                    try:
                        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                            gen = generate_with_stats(
                                _unwrap(model), dataset, device,
                                prompt="The history of ", n_tokens=200,
                                temperature=0.8, top_p=0.9,
                                top_k=40, repetition_penalty=1.1)
                        sample = gen["full_text"][:200].replace("\n", " ")
                        safe = sample.encode("ascii", errors="replace").decode("ascii")
                        print(f"  >> Sample @ step {step}: {safe}")
                        wb.log_text("sample", safe, step=step)
                    except Exception as e:
                        print(f"  >> Sample failed: {e}")
                    _unwrap(model).train()

                save_results(
                    results_path(run_tag, step, kind="progress"),
                    log, step, _unwrap(model).count_params(),
                    best_val_bpc, run_tag,
                    extra={"maslov_h": round(float(_unwrap(model).get_maslov_h()), 4)},
                )

            last_completed = step

    except KeyboardInterrupt:
        if is_main:
            ipath = f"checkpoints/{run_tag}_interrupt_step{last_completed:06d}.pt"
            print(f"\n\n  [INTERRUPT] step {last_completed} -> {ipath}")
            try:
                save_checkpoint(ipath, model, optimizer, scaler,
                                last_completed, best_val_bpc, log, cfg, run_tag)
                save_results(
                    results_path(run_tag, last_completed, kind="progress"),
                    log, last_completed, _unwrap(model).count_params(),
                    best_val_bpc, run_tag,
                )
                print(f"  [INTERRUPT] Resume with: --resume {ipath}")
            except Exception as e:
                print(f"  [INTERRUPT] Save failed: {e}")
        tsrn_cuda.cleanup_distributed()
        return

    # --- final eval (rank 0) ---
    if not is_main:
        tsrn_cuda.cleanup_distributed()
        return

    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
        print(f"\n  >> Loading best model: {best_ckpt_path} (val_bpc={best_val_bpc:.4f})")
        blob = torch.load(best_ckpt_path, map_location="cpu")
        _unwrap(model).load_state_dict(blob["model_state_dict"])
        del blob
        gc.collect()
        _unwrap(model).to(device)

    print(f"\n{'='*80}")
    print(f"  Final Evaluation (best val_bpc={best_val_bpc:.4f})")
    print(f"{'='*80}")
    _unwrap(model).eval()
    _unwrap(model).gist_buffer.reset()
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        val_loss, val_ppl, val_bpc = evaluate(
            _unwrap(model), dataset, device,
            n_batches=200, batch_size=min(batch_size, 16), split="val")
    print(f"  Val:  PPL={val_ppl:.3f}  BPC={val_bpc:.4f}")

    _unwrap(model).gist_buffer.reset()
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        test_loss, test_ppl, test_bpc = evaluate(
            _unwrap(model), dataset, device,
            n_batches=200, batch_size=min(batch_size, 16), split="test")
    print(f"  Test: PPL={test_ppl:.3f}  BPC={test_bpc:.4f}")

    final_cpath = ckpt_path(run_tag, n_steps, kind="final")
    save_checkpoint(final_cpath, model, optimizer, scaler,
                    n_steps, best_val_bpc, log, cfg, run_tag)
    print(f"  Final checkpoint: {final_cpath}")

    # Quality benchmark + final results
    if run_quality_benchmark is not None:
        try:
            _unwrap(model).gist_buffer.reset()
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                quality = run_quality_benchmark(_unwrap(model), dataset, device, "TSRNGist")
        except Exception as e:
            print(f"  Quality benchmark failed: {e}")
            quality = None
    else:
        quality = None

    total_time = time.time() - t0
    out = results_path(run_tag, n_steps, kind="final")
    save_results(out, log, n_steps, _unwrap(model).count_params(),
                 best_val_bpc, run_tag, extra={
                     "test_bpc": round(test_bpc, 4),
                     "test_ppl": round(test_ppl, 3),
                     "val_bpc": round(val_bpc, 4),
                     "val_ppl": round(val_ppl, 3),
                     "wall_time_hours": round(total_time / 3600, 2),
                     "preset": args.preset,
                     "world_size": world,
                     "amp_dtype": str(amp_dtype),
                     "config": cfg,
                     "quality_benchmark": quality,
                 })
    print(f"\n  Results saved: {out}")
    print(f"\n{'='*80}")
    print(f"  Test BPC: {test_bpc:.4f}  |  PPL: {test_ppl:.3f}")
    print(f"  Best Val: {best_val_bpc:.4f}")
    print(f"  Time:     {total_time/3600:.2f} hours on {world}× GPU")
    print(f"{'='*80}")

    # --- WandB final summary + cleanup ---
    wb.log_summary(
        test_bpc=float(test_bpc),
        test_ppl=float(test_ppl),
        val_bpc=float(val_bpc),
        val_ppl=float(val_ppl),
        best_val_bpc=float(best_val_bpc),
        wall_time_hours=round(total_time / 3600, 2),
        n_params=_unwrap(model).count_params(),
        n_steps=n_steps,
    )
    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
        wb.save_artifact(best_ckpt_path, name=f"{run_tag}_best", artifact_type="model")
    wb.finish()

    tsrn_cuda.cleanup_distributed()


# ---------------------------------------------------------------------------
#  No-op context manager (Python ≥3.7 has contextlib.nullcontext but we want
#  to be explicit and dependency-free here).
# ---------------------------------------------------------------------------
class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="TSRNGist cloud trainer — auto-detects single GPU vs DDP")
    p.add_argument("--preset", default="medium_40gb",
                   choices=list(PRESET_MODULES.keys()),
                   help="Configuration preset (model + batch + LR + steps).")
    p.add_argument("--steps", type=int, default=None,
                   help="Override total steps from preset.")
    p.add_argument("--batch", type=int, default=None,
                   help="Override per-GPU micro-batch size.")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="Override gradient-accumulation steps.")
    p.add_argument("--lr", type=float, default=None,
                   help="Override peak learning rate.")
    p.add_argument("--context", type=int, default=None,
                   help="Override context length.")
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile even if preset enables it.")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="Disable gradient checkpointing (uses more VRAM).")
    p.add_argument("--use-8bit-optim", action="store_true",
                   help="Use bitsandbytes AdamW8bit (saves optimizer VRAM).")
    # v2.0 parameters
    p.add_argument("--use-tmt-tokenizer", action="store_true",
                   help="Use Tropical Merging Tokenization (v2.0)")
    p.add_argument("--tmt-path", type=str, default=None,
                   help="Path to TMT tokenizer file")
    p.add_argument("--use-hyperbolic", action="store_true",
                   help="Use hyperbolic gist vectors (v2.0)")
    p.add_argument("--gist-chaining", action="store_true",
                   help="Enable gist chaining for infinite context (v2.0)")
    p.add_argument("--log-padic-pe", action="store_true",
                   help="Log PAdicPE norms during training (v2.0)")
    p.add_argument("--log-pacs", action="store_true",
                   help="Log PaCS effective context during training (v2.0)")
    # Kleene / tier flags
    p.add_argument("--use-kleene-ssm", action="store_true",
                   help="Enable KleeneSSM (replaces TropicalSSM). All tiers default on.")
    p.add_argument("--tier", type=str, default=None, choices=["nano", "pro", "kyro"],
                   help="ModelConfig tier. Sets Kleene flags automatically.")
    # WandB flags
    p.add_argument("--wandb-project", type=str, default=None,
                   help="WandB project name. If unset, WandB logging is disabled.")
    p.add_argument("--wandb-entity", type=str, default=None,
                   help="WandB entity (team/username). Optional.")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="WandB run name. Defaults to the auto-generated run tag.")
    args = p.parse_args(argv)

    cfg = load_preset(args.preset)
    # CLI overrides
    overrides = {
        "steps":           args.steps,
        "batch_size":      args.batch,
        "grad_accum_steps": args.grad_accum,
        "lr":              args.lr,
        "context_len":     args.context,
        "ckpt_every":      args.ckpt_every,
        "eval_every":      args.eval_every,
        "compile":         not args.no_compile,
        "gradient_checkpoint": not args.no_grad_ckpt,
        "use_8bit_optimizer": args.use_8bit_optim,
        # v2.0 overrides
        "use_tmt_tokenizer": args.use_tmt_tokenizer,
        "tmt_path": args.tmt_path,
        "use_hyperbolic": args.use_hyperbolic,
        "gist_chaining": args.gist_chaining,
        "log_padic_pe": args.log_padic_pe,
        "log_pacs": args.log_pacs,
        "use_kleene_ssm": args.use_kleene_ssm or None,
        "tier": args.tier,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if args.no_grad_ckpt:
        cfg["gradient_checkpoint"] = False
    if args.use_8bit_optim:
        cfg["use_8bit_optimizer"] = True

    train(cfg, args)


if __name__ == "__main__":
    main()
