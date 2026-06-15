"""
Azure pretraining trainer for TropFormer/Kyro.

Differs from ``research/cloud/train_cloud.py`` (which is enwik8-only) in:

  * Uses :class:`TokenShardStream` over the 92B-token mixture
  * TMT tokenizer required (``vocab_size`` from tokenizer)
  * Distributed-aware streaming sampler
  * Checkpoints + logs written to blob mount

Launch (single-node 8x H100)::

    torchrun --standalone --nproc_per_node=8 \\
        -m research.cloud.azure.train_pretrain_cloud \\
        --config pretrain_h100x8 --tag run0
"""

from __future__ import annotations

import argparse
import datetime
import gc
import importlib
import json
import logging
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.cloud import tsrn_cuda
from research.cloud.wandb_logger import WandBLogger
from research.cloud.azure.data.streaming_dataset import TokenShardStream
from research.tropical_tokenizer import TropicalMergingTokenizer
from research.tsrn_gist import TSRNGist

try:
    from research.model_config import ModelConfig as _ModelConfig  # noqa: F401
    from research.model_config import NANO_CONFIG, PRO_CONFIG, KYRO_CONFIG
    _TIER_MAP = {"nano": NANO_CONFIG, "pro": PRO_CONFIG, "kyro": KYRO_CONFIG}
except Exception:
    _TIER_MAP = {}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------------

CONFIG_MODULES = {
    "pretrain_h100x8": "research.cloud.azure.configs.pretrain_h100x8",
    "pretrain_flex":   "research.cloud.azure.configs.pretrain_flex",
    "pretrain_smoke":  "research.cloud.azure.configs.pretrain_smoke",
}

# Env vars that override numeric/bool config knobs at load time, so a single
# config adapts to whatever GPU SKU is available (we are NOT guaranteed H100s).
_ENV_INT_OVERRIDES = {
    "TROPFORMER_WORLD_SIZE":   "world_size",
    "TROPFORMER_BATCH_SIZE":   "batch_size",
    "TROPFORMER_GRAD_ACCUM":   "grad_accum_steps",
    "TROPFORMER_CONTEXT_LEN":  "context_len",
    "TROPFORMER_STEPS":        "steps",
    "TROPFORMER_WARMUP_STEPS": "warmup_steps",
}
_ENV_BOOL_OVERRIDES = {
    "TROPFORMER_COMPILE":              "compile",
    "TROPFORMER_GRADIENT_CHECKPOINT":  "gradient_checkpoint",
}


def apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply TROPFORMER_* numeric/bool env overrides onto a config dict."""
    for env, key in _ENV_INT_OVERRIDES.items():
        v = os.environ.get(env)
        if v:
            cfg[key] = int(v)
    for env, key in _ENV_BOOL_OVERRIDES.items():
        v = os.environ.get(env)
        if v is not None and v != "":
            cfg[key] = v.strip().lower() in ("1", "true", "yes", "on")
    return cfg


def load_config(name: str) -> Dict[str, Any]:
    if name not in CONFIG_MODULES:
        raise ValueError(f"unknown config {name}; choices {list(CONFIG_MODULES)}")
    return dict(importlib.import_module(CONFIG_MODULES[name]).CONFIG)


def build_run_tag(user_tag: str = "") -> str:
    date = datetime.datetime.now().strftime("%Y%m%d")
    suffix = f"_{user_tag}" if user_tag else ""
    return f"{date}_pretrain{suffix}"


def _unwrap(m: nn.Module) -> nn.Module:
    """Strip every torch.compile (`._orig_mod`) and DDP (`.module`) layer.
    Safe under any stacking order (compile(DDP(m)) or DDP(compile(m)))."""
    seen = 0
    while seen < 8:  # bounded to prevent any pathological cycle
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif hasattr(m, "module"):
            m = m.module
        else:
            return m
        seen += 1
    return m


def get_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def maslov_h(step: int, total: int, warm: float, cool: float, cycles: int) -> float:
    t = max(0.0, min(1.0, step / max(1, total)))
    mid = 0.5 * (warm + cool)
    amp = 0.5 * (warm - cool)
    return float(mid + amp * math.cos(2 * math.pi * cycles * t))


# ---------------------------------------------------------------------------
# Validation: small fixed eval set, one batch per dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_eval(model: nn.Module, loader: DataLoader, device: torch.device,
               n_batches: int = 32, amp_dtype: torch.dtype = torch.bfloat16) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    losses: List[float] = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["targets"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            _, loss = _unwrap(model)(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    if not losses:
        return {"val_loss": float("inf"), "val_ppl": float("inf"), "val_bpc": float("inf")}
    val_loss = sum(losses) / len(losses)
    return {
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss),
        "val_bpc": val_loss / math.log(2),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    # distributed setup
    dist_dev = tsrn_cuda.init_distributed()
    device = dist_dev or tsrn_cuda.detect_cuda_device(verbose=tsrn_cuda.is_main_process())
    is_main = tsrn_cuda.is_main_process()
    world = tsrn_cuda.get_world_size()
    rank = tsrn_cuda.get_global_rank()
    local_rank = tsrn_cuda.get_local_rank()

    tsrn_cuda.set_global_seed(cfg.get("seed", 42) + rank)
    amp_dtype, needs_scaler = tsrn_cuda.amp_components(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if (use_amp and needs_scaler) else None

    # tokenizer
    if not cfg.get("use_tmt_tokenizer") or not cfg.get("tmt_path"):
        raise ValueError("pretrain config must specify use_tmt_tokenizer + tmt_path")
    tokenizer = TropicalMergingTokenizer.load(cfg["tmt_path"])
    V = tokenizer.vocab_size
    if is_main:
        logger.info("loaded TMT tokenizer (vocab=%d) from %s",
                    V, cfg["tmt_path"])

    # Component-resonant curriculum: train stream reads the manifest's
    # `curriculum:` block and is driven by a file-based progress signal the
    # trainer publishes each step (works across DataLoader worker processes).
    curriculum_enabled = cfg.get("curriculum_enabled", True)
    progress_path = cfg.get("curriculum_progress_path") or str(
        Path(cfg.get("output_dir", "/mnt/blob/checkpoints/pretrain"))
        / "curriculum_progress.txt")
    if curriculum_enabled and is_main:
        Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
        # seed the file; the resume-aware value is written just before the loop
        TokenShardStream.write_progress(progress_path, 0.0)

    # streaming dataset
    train_stream = TokenShardStream(
        manifest=cfg["manifest"],
        tokens_dir=cfg["tokens_dir"],
        context_len=cfg["context_len"],
        seed=cfg.get("seed", 42),
        rank=rank,
        world=world,
        eos_id=tokenizer.vocab_to_id.get("<eos>", 2),
        curriculum=None if curriculum_enabled else {"mode": "off"},
        progress_path=progress_path if curriculum_enabled else None,
    )
    # validation uses the full static mixture (curriculum off) so eval BPC
    # reflects all components regardless of training phase.
    val_stream = TokenShardStream(
        manifest=cfg["manifest"],
        tokens_dir=cfg["tokens_dir"],
        context_len=cfg["context_len"],
        seed=cfg.get("seed", 42) + 9999,
        rank=rank,
        world=world,
        eos_id=tokenizer.vocab_to_id.get("<eos>", 2),
        curriculum={"mode": "off"},
    )
    _nw = cfg.get("num_workers", 4)
    train_loader = DataLoader(train_stream, batch_size=cfg["batch_size"],
                              num_workers=_nw, pin_memory=True,
                              persistent_workers=_nw > 0)
    val_loader = DataLoader(val_stream, batch_size=cfg["batch_size"],
                            num_workers=2, pin_memory=True)

    # model config
    model_cfg = None
    if _TIER_MAP and cfg.get("tier"):
        from dataclasses import replace
        base = _TIER_MAP.get(cfg["tier"])
        if base is not None:
            model_cfg = replace(base,
                                d_model=cfg["d_model"],
                                n_heads=cfg["n_heads"],
                                context_len=cfg["context_len"],
                                top_k=cfg.get("top_k", 16))

    model = TSRNGist(
        vocab=V, d_model=cfg["d_model"], context_len=cfg["context_len"],
        gradient_checkpoint=cfg.get("gradient_checkpoint", True),
        n_blocks=cfg["n_blocks"], top_k=cfg.get("top_k", 16),
        n_heads=cfg["n_heads"], mem_depth=cfg.get("mem_depth", 7),
        max_gists=cfg.get("max_gists", 64), gist_top_k=cfg.get("gist_top_k", 4),
        dropout=cfg.get("dropout", 0.0),
        use_hyperbolic=cfg.get("use_hyperbolic", False),
        gist_chaining=cfg.get("gist_chaining", False),
        config=model_cfg,
    )

    # resume
    start_step = 0
    resume_blob: Optional[Dict[str, Any]] = None
    if args.resume and os.path.exists(args.resume):
        resume_blob = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_blob["model_state_dict"])
        start_step = resume_blob.get("step", 0)
        if is_main:
            logger.info("resumed from %s @ step %d", args.resume, start_step)

    model.to(device).train()

    # DDP first, then compile (PyTorch >=2.0 recommended order for
    # mode=max-autotune; the DDP allreduce hooks are graph-captured
    # inside the compiled region).
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, gradient_as_bucket_view=True,
        )
    if cfg.get("compile"):
        model = tsrn_cuda.maybe_compile(
            model, mode=cfg.get("compile_mode", "max-autotune"))
        if is_main:
            logger.info("torch.compile enabled (mode=%s, world=%d)",
                        cfg.get("compile_mode", "max-autotune"), world)

    optimizer = tsrn_cuda.make_optimizer(
        _unwrap(model), lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.1),
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
        use_8bit=cfg.get("use_8bit_optimizer", False),
    )
    if resume_blob is not None and "optimizer_state_dict" in resume_blob:
        try:
            optimizer.load_state_dict(resume_blob["optimizer_state_dict"])
        except Exception as e:
            if is_main:
                logger.warning("optim restore failed: %s", e)

    run_tag = build_run_tag(args.tag)
    output_dir = Path(cfg.get("output_dir", "/mnt/blob/checkpoints/pretrain"))
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    wb = WandBLogger.maybe_init(args, cfg, run_tag, is_main=is_main,
                                model=_unwrap(model), world_size=world)

    n_steps = cfg["steps"]
    micro_bs = cfg["batch_size"]
    grad_accum = cfg.get("grad_accum_steps", 1)
    eff_batch = micro_bs * grad_accum * world
    eval_every = cfg.get("eval_every", 2000)
    ckpt_every = cfg.get("ckpt_every", 5000)

    if is_main:
        n_params = _unwrap(model).count_params()
        print(f"\n{'='*88}")
        print(f"  Pretrain  —  TSRNGist {n_params:,} params  —  vocab {V}  ctx {cfg['context_len']}")
        print(f"  Steps {n_steps}  |  micro_bs {micro_bs}  |  grad_accum {grad_accum}  |  world {world}  |  eff_batch {eff_batch}")
        print(f"  Tokens/step ~ {eff_batch * cfg['context_len']:,}")
        print(f"  Run tag: {run_tag}")
        print(f"{'='*88}\n")
        print(f"{'Step':>7}  {'TrLoss':>9}  {'TrBPC':>7}  {'ValBPC':>7}  "
              f"{'LR':>9}  {'GNorm':>7}  {'tok/s':>10}  {'ms/step':>9}")

    train_iter = iter(train_loader)
    log: List[Dict[str, Any]] = list(resume_blob.get("log", [])) if resume_blob else []
    best_val_bpc = float(resume_blob.get("best_val_bpc", float("inf"))) if resume_blob else float("inf")
    if resume_blob is not None:
        del resume_blob
        gc.collect()

    # publish resume-aware curriculum progress before workers begin iterating
    if curriculum_enabled and is_main:
        TokenShardStream.write_progress(progress_path, start_step / max(1, n_steps))

    t0 = time.time()
    for step in range(start_step + 1, n_steps + 1):
        # publish curriculum progress (single writer; workers poll the file)
        if curriculum_enabled and is_main and step % cfg.get("curriculum_update_every", 50) == 0:
            TokenShardStream.write_progress(progress_path, step / n_steps)

        lr = get_lr(step, cfg.get("warmup_steps", 4000),
                    n_steps, cfg["lr"], cfg.get("min_lr", cfg["lr"] * 0.1))
        for g in optimizer.param_groups:
            g["lr"] = lr

        h = maslov_h(step, n_steps,
                     cfg.get("maslov_h_warm", 1.5),
                     cfg.get("maslov_h_cool", 0.3),
                     cfg.get("maslov_n_cycles", 3))
        _unwrap(model).set_maslov_h(h)

        if step % cfg.get("gist_reset_every", 100) == 1:
            _unwrap(model).gist_buffer.reset()

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for ga in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["targets"].to(device, non_blocking=True)
            ctx = (model.no_sync() if (world > 1 and ga < grad_accum - 1)
                   else nullcontext())
            with ctx:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                loss = loss / grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            accum_loss += loss.item() * grad_accum

        loss_val = accum_loss / grad_accum
        if scaler is not None:
            scaler.unscale_(optimizer)
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % cfg.get("gc_every", 500) == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if step % eval_every == 0 or step == n_steps:
            metrics = quick_eval(model, val_loader, device,
                                 n_batches=cfg.get("eval_batches", 32),
                                 amp_dtype=amp_dtype)
            elapsed = time.time() - t0
            ms_step = elapsed / max(1, step - start_step) * 1000
            tok_s = eff_batch * cfg["context_len"] / (ms_step / 1000)
            tr_bpc = loss_val / math.log(2)
            entry = {
                "step": step, "train_loss": round(loss_val, 5),
                "train_bpc": round(tr_bpc, 4),
                "val_loss": round(metrics["val_loss"], 5),
                "val_ppl": round(metrics["val_ppl"], 3),
                "val_bpc": round(metrics["val_bpc"], 4),
                "grad_norm": round(float(gnorm), 4),
                "lr": round(lr, 7), "ms_per_step": round(ms_step, 1),
                "tok_per_sec": round(tok_s, 1), "maslov_h": round(h, 4),
            }
            log.append(entry)
            if is_main:
                print(f"{step:7d}  {loss_val:9.4f}  {tr_bpc:7.4f}  "
                      f"{metrics['val_bpc']:7.4f}  {lr:9.2e}  "
                      f"{float(gnorm):7.3f}  {tok_s:10.0f}  {ms_step:9.1f}")
                wb.log(entry)

            if metrics["val_bpc"] < best_val_bpc and is_main:
                best_val_bpc = metrics["val_bpc"]
                ckpt = output_dir / f"{run_tag}_best.pt"
                _save_ckpt(ckpt, _unwrap(model), optimizer, scaler, step,
                           best_val_bpc, log, cfg, run_tag)

        if step % ckpt_every == 0 and is_main:
            ckpt = output_dir / f"{run_tag}_step{step:07d}.pt"
            _save_ckpt(ckpt, _unwrap(model), optimizer, scaler, step,
                       best_val_bpc, log, cfg, run_tag)
            _prune_ckpts(output_dir, run_tag, keep=cfg.get("keep_last_ckpts", 5))

    if is_main:
        ckpt = output_dir / f"{run_tag}_final_step{n_steps:07d}.pt"
        _save_ckpt(ckpt, _unwrap(model), optimizer, scaler, n_steps,
                   best_val_bpc, log, cfg, run_tag)
        wb.log_summary(best_val_bpc=float(best_val_bpc),
                       n_params=_unwrap(model).count_params(),
                       n_steps=n_steps,
                       wall_time_hours=round((time.time() - t0) / 3600, 2))


def _save_ckpt(path: Path, model: nn.Module, optim, scaler, step: int,
               best_val_bpc: float, log, cfg, run_tag: str) -> None:
    blob = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optim.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step, "best_val_bpc": best_val_bpc,
        "log": log, "config": cfg, "run_tag": run_tag,
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)
    del blob
    gc.collect()


def _prune_ckpts(output_dir: Path, run_tag: str, keep: int = 5) -> None:
    ckpts = sorted(output_dir.glob(f"{run_tag}_step*.pt"))
    for old in ckpts[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pretrain_h100x8",
                   choices=list(CONFIG_MODULES.keys()))
    p.add_argument("--tag", default="")
    p.add_argument("--resume", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--no-compile", action="store_true")
    # WandB
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config(args.config)

    # AML input/output path overrides — aml_pretrain.yaml sets these env vars
    # so the trainer works whether run on a bare VM (/mnt/blob/...) or via
    # AzureML (dynamic input/output mount paths).
    if os.environ.get("TROPFORMER_TOKENS_DIR"):
        cfg["tokens_dir"] = os.environ["TROPFORMER_TOKENS_DIR"]
    if os.environ.get("TROPFORMER_TMT_PATH"):
        cfg["tmt_path"] = os.environ["TROPFORMER_TMT_PATH"]
    if os.environ.get("TROPFORMER_OUTPUT_DIR"):
        cfg["output_dir"] = os.environ["TROPFORMER_OUTPUT_DIR"]
    if os.environ.get("TROPFORMER_RESUME") and args.resume is None:
        args.resume = os.environ["TROPFORMER_RESUME"]

    # Numeric/bool overrides so one config adapts to the available GPU SKU.
    cfg = apply_env_overrides(cfg)

    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.no_compile:
        cfg["compile"] = False
    if args.wandb_project is None:
        args.wandb_project = cfg.get("wandb_project")
    train(cfg, args)


if __name__ == "__main__":
    main()
