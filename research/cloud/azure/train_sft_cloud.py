"""Azure SFT trainer (curriculum-aware).

Order: reasoning -> instruction -> tool -> kyro.

Reads JSONL shards from blob, formats by ``format`` field, tokenizes with
TMT on-the-fly, masks prompt tokens to IGNORE in labels, packs to
context_len, runs DDP training.
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
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, IterableDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.cloud import tsrn_cuda
from research.cloud.wandb_logger import WandBLogger
from research.tropical_tokenizer import TropicalMergingTokenizer
from research.tsrn_gist import TSRNGist

logger = logging.getLogger(__name__)

CONFIG_MODULES = {"sft_h100x8": "research.cloud.azure.configs.sft_h100x8"}
IGNORE = -100


def load_config(name: str) -> Dict[str, Any]:
    return dict(importlib.import_module(CONFIG_MODULES[name]).CONFIG)


# ---------------------------------------------------------------------------
# Formatting helpers — return (prompt_text, response_text)
# ---------------------------------------------------------------------------

ROLE_PREFIX = {"system": "[SYS] ", "user": "[USR] ", "assistant": "[AST] ",
               "tool": "[TOOL] ", "function": "[FN] "}


def _fmt_messages(msgs: List[Dict[str, str]]) -> Tuple[str, str]:
    if not msgs:
        return "", ""
    parts: List[str] = []
    for m in msgs[:-1]:
        prefix = ROLE_PREFIX.get(m.get("role", "user"), "[USR] ")
        parts.append(prefix + m.get("content", "") + "\n")
    last = msgs[-1]
    parts.append(ROLE_PREFIX.get(last.get("role", "assistant"), "[AST] "))
    return "".join(parts), last.get("content", "")


def _fmt_instruction_response(ex: Dict[str, Any]) -> Tuple[str, str]:
    instr = ex.get("instruction") or ex.get("question") or ex.get("prompt") or ""
    resp = ex.get("response") or ex.get("answer") or ex.get("output") or ""
    inp = ex.get("input", "")
    if inp:
        prompt = "[USR] " + str(instr) + "\n" + str(inp) + "\n[AST] "
    else:
        prompt = "[USR] " + str(instr) + "\n[AST] "
    return prompt, str(resp)


def _fmt_tool_call(ex: Dict[str, Any]) -> Tuple[str, str]:
    tools = ex.get("tools") or ex.get("functions") or []
    user = ex.get("user") or ex.get("query") or ex.get("prompt") or ""
    answer = ex.get("answer") or ex.get("response") or ex.get("tool_calls") or ""
    if isinstance(answer, (dict, list)):
        answer = json.dumps(answer)
    if isinstance(tools, (dict, list)):
        tools_str = json.dumps(tools)
    else:
        tools_str = str(tools)
    prompt = "[SYS] tools=" + tools_str + "\n[USR] " + str(user) + "\n[AST] "
    return prompt, str(answer)


def _format_example(rec: Dict[str, Any], fmt: str) -> Optional[Tuple[str, str]]:
    try:
        if fmt == "messages":
            msgs = rec.get("messages") or rec.get("conversations") or []
            return _fmt_messages(msgs)
        if fmt == "instruction_response":
            return _fmt_instruction_response(rec)
        if fmt == "tool_call":
            return _fmt_tool_call(rec)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Dataset: streams JSONL shards from a list of dataset entries
# ---------------------------------------------------------------------------

class SFTStream(IterableDataset):
    def __init__(self, entries: List[Dict[str, Any]], tokens_dir: Path,
                 tokenizer: TropicalMergingTokenizer, context_len: int,
                 seed: int, rank: int, world: int) -> None:
        super().__init__()
        self.entries = entries
        self.tokens_dir = tokens_dir
        self.tok = tokenizer
        self.context_len = context_len
        self.seed = seed
        self.rank = rank
        self.world = world
        self.pad_id = tokenizer.vocab_to_id.get("<pad>", 0)
        self.eos_id = tokenizer.vocab_to_id.get("<eos>", 2)
        self._weights = [float(e["weight"]) for e in entries]
        s = sum(self._weights) or 1.0
        self._weights = [w / s for w in self._weights]
        # one open file per (rank-sharded) entry; rotated on EOF
        self._shards_per_entry: List[List[Path]] = []
        for e in entries:
            ds_dir = tokens_dir / e["name"]
            shards = sorted(ds_dir.glob("*.jsonl*"))
            if world > 1:
                shards = [s for i, s in enumerate(shards) if i % world == rank] or shards
            self._shards_per_entry.append(shards)

    def _iter_records(self, shards: List[Path]) -> Iterator[Dict[str, Any]]:
        for path in shards:
            if path.suffix == ".zst":
                import zstandard as zstd  # type: ignore
                with open(path, "rb") as fh:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(fh) as rdr:
                        text = rdr.read().decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
            else:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = random.Random(self.seed + 1000 * self.rank + wid)
        # round-robin generators per entry
        gens = [iter(self._iter_records(sh)) for sh in self._shards_per_entry]
        while True:
            i = rng.choices(range(len(self.entries)), weights=self._weights, k=1)[0]
            try:
                rec = next(gens[i])
            except StopIteration:
                gens[i] = iter(self._iter_records(self._shards_per_entry[i]))
                try:
                    rec = next(gens[i])
                except StopIteration:
                    continue
            fmt = self.entries[i].get("format", "instruction_response")
            pr = _format_example(rec, fmt)
            if pr is None:
                continue
            prompt, response = pr
            if not response:
                continue
            prompt_ids = self.tok.encode(prompt)
            resp_ids = self.tok.encode(response) + [self.eos_id]
            full = prompt_ids + resp_ids
            if len(full) > self.context_len + 1:
                # truncate prompt first
                drop = len(full) - (self.context_len + 1)
                prompt_ids = prompt_ids[drop:]
                full = prompt_ids + resp_ids
                if len(full) > self.context_len + 1:
                    full = full[: self.context_len + 1]
                    resp_ids = full[len(prompt_ids):]
            # pad
            pad_n = (self.context_len + 1) - len(full)
            full = full + [self.pad_id] * pad_n
            x = torch.tensor(full[:-1], dtype=torch.long)
            y = torch.tensor(full[1:], dtype=torch.long)
            # mask prompt + pads in labels
            labels = y.clone()
            mask = torch.ones_like(labels)
            mask[: max(0, len(prompt_ids) - 1)] = 0
            mask[len(full) - 1 - pad_n:] = 0  # zero pad region
            labels[mask == 0] = IGNORE
            sw = float(self.entries[i].get("sample_weight_multiplier", 1.0))
            yield {"input_ids": x, "labels": labels,
                   "sample_weight": torch.tensor(sw, dtype=torch.float32),
                   "dataset": self.entries[i]["name"]}


# ---------------------------------------------------------------------------
# Curriculum split
# ---------------------------------------------------------------------------

def _curriculum_groups(manifest: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    order = manifest.get("curriculum_order",
                         ["reasoning", "instruction", "tool", "kyro"])
    grouped: Dict[str, List[Dict[str, Any]]] = {k: [] for k in order}
    for e in manifest["mixture"]:
        prio = e.get("priority", "instruction")
        if prio not in grouped:
            grouped[prio] = []
        grouped[prio].append(e)
    return [(k, grouped[k]) for k in order if grouped[k]]


def _steps_per_group(groups, total_steps: int) -> List[int]:
    sizes = [sum(e.get("examples_k", 1) for e in g) for _, g in groups]
    s = sum(sizes) or 1
    out = [max(1, int(total_steps * sz / s)) for sz in sizes]
    # rectify to total
    out[-1] += total_steps - sum(out)
    return out


# ---------------------------------------------------------------------------
# Loss with optional sample weighting
# ---------------------------------------------------------------------------

def _ce_loss(logits: torch.Tensor, labels: torch.Tensor,
             sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, T, V = logits.shape
    flat_logits = logits.view(B * T, V)
    flat_labels = labels.view(B * T)
    losses = nn.functional.cross_entropy(
        flat_logits, flat_labels, ignore_index=IGNORE, reduction="none")
    losses = losses.view(B, T)
    valid = (labels != IGNORE).float()
    per_seq = (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    if sample_weight is not None:
        per_seq = per_seq * sample_weight
        return per_seq.sum() / sample_weight.sum().clamp_min(1e-6)
    return per_seq.mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    dist_dev = tsrn_cuda.init_distributed()
    device = dist_dev or tsrn_cuda.detect_cuda_device(verbose=tsrn_cuda.is_main_process())
    is_main = tsrn_cuda.is_main_process()
    world = tsrn_cuda.get_world_size()
    rank = tsrn_cuda.get_global_rank()
    local_rank = tsrn_cuda.get_local_rank()

    tsrn_cuda.set_global_seed(cfg.get("seed", 43) + rank)
    amp_dtype, needs_scaler = tsrn_cuda.amp_components(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if (use_amp and needs_scaler) else None

    tokenizer = TropicalMergingTokenizer.load(cfg["tmt_path"])
    V = tokenizer.vocab_size

    with open(cfg["manifest"], "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    groups = _curriculum_groups(manifest)
    if is_main:
        logger.info("curriculum: %s",
                    ", ".join(f"{k}({len(v)})" for k, v in groups))

    # model
    model_cfg = None
    try:
        from dataclasses import replace
        from research.model_config import NANO_CONFIG, PRO_CONFIG, KYRO_CONFIG
        tier_map = {"nano": NANO_CONFIG, "pro": PRO_CONFIG, "kyro": KYRO_CONFIG}
        base = tier_map.get(cfg.get("tier", "pro"))
        if base is not None:
            model_cfg = replace(base, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                                context_len=cfg["context_len"], top_k=cfg.get("top_k", 16))
    except Exception:
        pass

    model = TSRNGist(
        vocab=V, d_model=cfg["d_model"], context_len=cfg["context_len"],
        gradient_checkpoint=cfg.get("gradient_checkpoint", True),
        n_blocks=cfg["n_blocks"], top_k=cfg.get("top_k", 16),
        n_heads=cfg["n_heads"], mem_depth=cfg.get("mem_depth", 7),
        max_gists=cfg.get("max_gists", 64), gist_top_k=cfg.get("gist_top_k", 4),
        dropout=cfg.get("dropout", 0.05),
        use_hyperbolic=cfg.get("use_hyperbolic", False),
        gist_chaining=cfg.get("gist_chaining", False),
        config=model_cfg,
    )

    # init from pretrained
    init_from = args.init_from or cfg.get("init_from")
    if init_from and os.path.exists(init_from):
        blob = torch.load(init_from, map_location="cpu")
        model.load_state_dict(blob["model_state_dict"], strict=False)
        if is_main:
            logger.info("init from %s", init_from)
        del blob
        gc.collect()
    elif is_main:
        logger.warning("no pretrained checkpoint at %s; training from scratch",
                       init_from)

    model.to(device).train()
    # DDP first, then compile.
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, gradient_as_bucket_view=True)
    if cfg.get("compile"):
        model = tsrn_cuda.maybe_compile(
            model, mode=cfg.get("compile_mode", "reduce-overhead"))
        if is_main:
            logger.info("torch.compile enabled (mode=%s, world=%d)",
                        cfg.get("compile_mode", "reduce-overhead"), world)

    optimizer = tsrn_cuda.make_optimizer(
        _unwrap(model), lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.05),
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
        use_8bit=cfg.get("use_8bit_optimizer", False))

    # determine total steps and per-group split
    total_steps = args.steps or _estimate_total_steps(cfg, manifest, world)
    group_steps = _steps_per_group(groups, total_steps)

    run_tag = _run_tag(args.tag)
    output_dir = Path(cfg.get("output_dir", "/mnt/blob/checkpoints/sft"))
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    wb = WandBLogger.maybe_init(args, cfg, run_tag, is_main=is_main,
                                model=_unwrap(model), world_size=world)

    if is_main:
        print(f"\n  SFT  vocab={V}  ctx={cfg['context_len']}  total_steps={total_steps}")
        print(f"  Curriculum:")
        for (gname, _), gs in zip(groups, group_steps):
            print(f"    {gname:>12s}  {gs:>8d} steps")
        print()

    global_step = 0
    log: List[Dict[str, Any]] = []
    t0 = time.time()
    grad_accum = cfg.get("grad_accum_steps", 1)
    micro_bs = cfg["batch_size"]

    for (gname, gentries), gsteps in zip(groups, group_steps):
        if is_main:
            logger.info("=== curriculum stage: %s (%d steps) ===", gname, gsteps)
        ds = SFTStream(gentries, Path(cfg["tokens_dir"]), tokenizer,
                       cfg["context_len"], seed=cfg.get("seed", 43) + global_step,
                       rank=rank, world=world)
        loader = DataLoader(ds, batch_size=micro_bs,
                            num_workers=cfg.get("num_workers", 4),
                            pin_memory=True)
        train_iter = iter(loader)
        for local_step in range(1, gsteps + 1):
            global_step += 1
            lr = _cosine_lr(global_step, cfg.get("warmup_steps", 200),
                            total_steps, cfg["lr"], cfg.get("min_lr", cfg["lr"] * 0.1))
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for ga in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(loader)
                    batch = next(train_iter)
                x = batch["input_ids"].to(device, non_blocking=True)
                lbl = batch["labels"].to(device, non_blocking=True)
                sw = (batch["sample_weight"].to(device, non_blocking=True)
                      if cfg.get("use_sample_weight_multiplier") else None)
                ctx = (model.no_sync() if (world > 1 and ga < grad_accum - 1)
                       else nullcontext())
                with ctx:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        logits, _ = model(x)
                        loss = _ce_loss(logits, lbl, sample_weight=sw)
                    loss = loss / grad_accum
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                accum_loss += loss.item() * grad_accum

            loss_val = accum_loss / grad_accum
            if scaler is not None:
                scaler.unscale_(optimizer)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(),
                                             cfg.get("grad_clip", 1.0))
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if global_step % cfg.get("eval_every", 1000) == 0 and is_main:
                elapsed = time.time() - t0
                ms_step = elapsed / max(1, global_step) * 1000
                entry = {
                    "step": global_step, "stage": gname,
                    "train_loss": round(loss_val, 5), "lr": round(lr, 7),
                    "grad_norm": round(float(gnorm), 4),
                    "ms_per_step": round(ms_step, 1),
                }
                log.append(entry)
                print(f"  [{gname:>12s}] step {global_step:>7d}  "
                      f"loss {loss_val:.4f}  lr {lr:.2e}  ms {ms_step:.0f}")
                wb.log(entry)

            if global_step % cfg.get("ckpt_every", 2000) == 0 and is_main:
                ckpt = output_dir / f"{run_tag}_step{global_step:07d}.pt"
                _save_ckpt(ckpt, _unwrap(model), optimizer, scaler,
                           global_step, log, cfg, run_tag)

    if is_main:
        ckpt = output_dir / f"{run_tag}_final.pt"
        _save_ckpt(ckpt, _unwrap(model), optimizer, scaler,
                   global_step, log, cfg, run_tag)
        wb.log_summary(n_steps=global_step,
                       wall_time_hours=round((time.time() - t0) / 3600, 2))


def _unwrap(m: nn.Module) -> nn.Module:
    """Strip every torch.compile (`._orig_mod`) and DDP (`.module`) layer."""
    seen = 0
    while seen < 8:
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif hasattr(m, "module"):
            m = m.module
        else:
            return m
        seen += 1
    return m


def _cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(1, warmup)
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def _estimate_total_steps(cfg: Dict[str, Any], manifest: Dict[str, Any],
                          world: int) -> int:
    total_examples = sum(int(e.get("examples_k", 0)) * 1000
                         for e in manifest["mixture"])
    epochs = cfg.get("epochs", 3)
    eff_batch = cfg["batch_size"] * cfg.get("grad_accum_steps", 1) * world
    return max(1000, total_examples * epochs // eff_batch)


def _run_tag(user_tag: str = "") -> str:
    date = datetime.datetime.now().strftime("%Y%m%d")
    suffix = f"_{user_tag}" if user_tag else ""
    return f"{date}_sft{suffix}"


def _save_ckpt(path: Path, model: nn.Module, optim, scaler, step: int,
               log, cfg, run_tag: str) -> None:
    blob = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optim.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step, "log": log, "config": cfg, "run_tag": run_tag,
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)
    del blob
    gc.collect()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft_h100x8", choices=list(CONFIG_MODULES))
    p.add_argument("--tag", default="")
    p.add_argument("--init-from", default=None,
                   help="Override pretrained checkpoint path")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config(args.config)
    if args.wandb_project is None:
        args.wandb_project = cfg.get("wandb_project")
    train(cfg, args)


if __name__ == "__main__":
    main()
