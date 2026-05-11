"""Azure DPO trainer.

Loads (prompt, chosen, rejected) triples from the DPO manifest, runs
forward pass on both chosen and rejected with policy + frozen reference,
computes the DPO loss, and updates only the policy weights.

Launch::

    torchrun --standalone --nproc_per_node=8 \\
        -m research.cloud.azure.train_dpo_cloud \\
        --config dpo_h100x8 --tag run0
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
import torch.nn.functional as F
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

CONFIG_MODULES = {"dpo_h100x8": "research.cloud.azure.configs.dpo_h100x8"}
IGNORE = -100


def load_config(name: str) -> Dict[str, Any]:
    return dict(importlib.import_module(CONFIG_MODULES[name]).CONFIG)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DPOStream(IterableDataset):
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
        self._shards: List[List[Path]] = []
        for e in entries:
            ds_dir = tokens_dir / e["name"]
            shards = sorted(ds_dir.glob("*.jsonl*"))
            if world > 1:
                shards = [s for i, s in enumerate(shards) if i % world == rank] or shards
            self._shards.append(shards)

    def _stream(self, shards: List[Path]) -> Iterator[Dict[str, Any]]:
        for path in shards:
            if path.suffix == ".zst":
                import zstandard as zstd  # type: ignore
                with open(path, "rb") as fh:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(fh) as rdr:
                        text = rdr.read().decode("utf-8", errors="replace")
                for line in text.splitlines():
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            pass
            else:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                pass

    def _encode_pair(self, prompt: str, chosen: str, rejected: str
                     ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        p_ids = self.tok.encode(prompt)
        c_ids = self.tok.encode(chosen) + [self.eos_id]
        r_ids = self.tok.encode(rejected) + [self.eos_id]

        def pack(p, r):
            full = p + r
            if len(full) > self.context_len:
                # truncate prompt
                drop = len(full) - self.context_len
                p2 = p[drop:]
                full = p2 + r
                if len(full) > self.context_len:
                    full = full[: self.context_len]
                return full, len(p2)
            return full + [self.pad_id] * (self.context_len - len(full)), len(p)

        c_seq, c_p_len = pack(p_ids, c_ids)
        r_seq, r_p_len = pack(p_ids, r_ids)
        return (
            torch.tensor(c_seq, dtype=torch.long),
            torch.tensor(r_seq, dtype=torch.long),
            c_p_len, r_p_len,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = random.Random(self.seed + 1000 * self.rank + wid)
        gens = [iter(self._stream(sh)) for sh in self._shards]
        while True:
            i = rng.choices(range(len(self.entries)), weights=self._weights, k=1)[0]
            try:
                rec = next(gens[i])
            except StopIteration:
                gens[i] = iter(self._stream(self._shards[i]))
                try:
                    rec = next(gens[i])
                except StopIteration:
                    continue
            prompt = rec.get("prompt") or rec.get("question") or ""
            chosen = rec.get("chosen") or rec.get("response_chosen") or ""
            rejected = rec.get("rejected") or rec.get("response_rejected") or ""
            if not (prompt and chosen and rejected):
                continue
            c_seq, r_seq, c_pl, r_pl = self._encode_pair(prompt, chosen, rejected)
            yield {
                "chosen_ids": c_seq, "rejected_ids": r_seq,
                "chosen_prompt_len": c_pl, "rejected_prompt_len": r_pl,
            }


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def _logp_completion(logits: torch.Tensor, labels: torch.Tensor,
                     prompt_len: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Sum log p of completion tokens only (mask prompt + pad)."""
    # logits: (B, T, V); labels: (B, T) is the input shifted
    B, T, _ = logits.shape
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = labels[:, 1:]  # next-token targets
    gather = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # mask: 1 for completion tokens, 0 for prompt + pad
    idx = torch.arange(T - 1, device=logits.device).unsqueeze(0)  # (1, T-1)
    is_completion = idx >= prompt_len.unsqueeze(1) - 1  # shifted
    is_not_pad = targets != pad_id
    mask = is_completion & is_not_pad
    return (gather * mask.float()).sum(dim=1)


def dpo_loss(policy_chosen_lp: torch.Tensor,
             policy_rejected_lp: torch.Tensor,
             ref_chosen_lp: torch.Tensor,
             ref_rejected_lp: torch.Tensor,
             beta: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    pi = policy_chosen_lp - policy_rejected_lp
    ref = ref_chosen_lp - ref_rejected_lp
    margin = beta * (pi - ref)
    losses = -F.logsigmoid(margin)
    loss = losses.mean()
    metrics = {
        "margin_mean": margin.detach().mean().item(),
        "policy_chosen_lp": policy_chosen_lp.detach().mean().item(),
        "policy_rejected_lp": policy_rejected_lp.detach().mean().item(),
        "accuracy": (margin > 0).float().mean().item(),
    }
    return loss, metrics


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

    tsrn_cuda.set_global_seed(cfg.get("seed", 44) + rank)
    amp_dtype, _ = tsrn_cuda.amp_components(device)
    use_amp = device.type == "cuda"

    tokenizer = TropicalMergingTokenizer.load(cfg["tmt_path"])
    V = tokenizer.vocab_size

    with open(cfg["manifest"], "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    # build policy and reference (same arch)
    def _build() -> nn.Module:
        return TSRNGist(
            vocab=V, d_model=cfg["d_model"], context_len=cfg["context_len"],
            gradient_checkpoint=cfg.get("gradient_checkpoint", True),
            n_blocks=cfg["n_blocks"], top_k=cfg.get("top_k", 16),
            n_heads=cfg["n_heads"], mem_depth=cfg.get("mem_depth", 7),
            max_gists=cfg.get("max_gists", 64), gist_top_k=cfg.get("gist_top_k", 4),
            dropout=cfg.get("dropout", 0.0),
            use_hyperbolic=cfg.get("use_hyperbolic", False),
            gist_chaining=cfg.get("gist_chaining", False),
        )

    policy = _build()
    reference = _build()

    init_path = args.init_from or cfg.get("init_from")
    ref_path = cfg.get("reference_from", init_path)
    if init_path and os.path.exists(init_path):
        blob = torch.load(init_path, map_location="cpu")
        policy.load_state_dict(blob["model_state_dict"], strict=False)
        del blob
    if ref_path and os.path.exists(ref_path):
        blob = torch.load(ref_path, map_location="cpu")
        reference.load_state_dict(blob["model_state_dict"], strict=False)
        del blob
    gc.collect()

    policy.to(device).train()
    reference.to(device).eval()
    for p in reference.parameters():
        p.requires_grad = False

    if world > 1:
        policy = nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, gradient_as_bucket_view=True)
    if cfg.get("compile"):
        policy = tsrn_cuda.maybe_compile(
            policy, mode=cfg.get("compile_mode", "reduce-overhead"))
        if is_main:
            logger.info("torch.compile enabled (mode=%s, world=%d)",
                        cfg.get("compile_mode", "reduce-overhead"), world)

    optimizer = tsrn_cuda.make_optimizer(
        _unwrap(policy), lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
        use_8bit=cfg.get("use_8bit_optimizer", False))

    ds = DPOStream(manifest["mixture"], Path(cfg["tokens_dir"]),
                   tokenizer, cfg["context_len"],
                   seed=cfg.get("seed", 44), rank=rank, world=world)
    loader = DataLoader(ds, batch_size=cfg["batch_size"],
                        num_workers=cfg.get("num_workers", 2), pin_memory=True)

    # total steps
    total_pairs = sum(int(e.get("sampled_pairs_k") or e.get("pairs_k", 0)) * 1000
                      for e in manifest["mixture"])
    epochs = cfg.get("epochs", 3)
    eff_batch = cfg["batch_size"] * cfg.get("grad_accum_steps", 1) * world
    total_steps = args.steps or max(500, total_pairs * epochs // eff_batch)

    run_tag = _run_tag(args.tag)
    output_dir = Path(cfg.get("output_dir", "/mnt/blob/checkpoints/dpo"))
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    wb = WandBLogger.maybe_init(args, cfg, run_tag, is_main=is_main,
                                model=_unwrap(policy), world_size=world)

    if is_main:
        print(f"\n  DPO  vocab={V}  ctx={cfg['context_len']}  total_steps={total_steps}")
        print(f"  beta={cfg.get('beta', 0.1)}  eff_batch={eff_batch}\n")

    grad_accum = cfg.get("grad_accum_steps", 1)
    train_iter = iter(loader)
    log: List[Dict[str, Any]] = []
    t0 = time.time()
    pad_id = tokenizer.vocab_to_id.get("<pad>", 0)

    for step in range(1, total_steps + 1):
        lr = _cosine_lr(step, cfg.get("warmup_steps", 100), total_steps,
                        cfg["lr"], cfg.get("min_lr", cfg["lr"] * 0.1))
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_metrics: Dict[str, float] = {}
        for ga in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)
            c = batch["chosen_ids"].to(device, non_blocking=True)
            r = batch["rejected_ids"].to(device, non_blocking=True)
            c_pl = batch["chosen_prompt_len"].to(device, non_blocking=True)
            r_pl = batch["rejected_prompt_len"].to(device, non_blocking=True)

            ctx = (policy.no_sync() if (world > 1 and ga < grad_accum - 1)
                   else nullcontext())
            with ctx:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    pol_c_logits, _ = policy(c)
                    pol_r_logits, _ = policy(r)
                    with torch.no_grad():
                        ref_c_logits, _ = reference(c)
                        ref_r_logits, _ = reference(r)
                    pol_c_lp = _logp_completion(pol_c_logits, c, c_pl, pad_id)
                    pol_r_lp = _logp_completion(pol_r_logits, r, r_pl, pad_id)
                    ref_c_lp = _logp_completion(ref_c_logits, c, c_pl, pad_id)
                    ref_r_lp = _logp_completion(ref_r_logits, r, r_pl, pad_id)
                    loss, metrics = dpo_loss(pol_c_lp, pol_r_lp,
                                             ref_c_lp, ref_r_lp,
                                             beta=cfg.get("beta", 0.1))
                loss = loss / grad_accum
                loss.backward()
            accum_loss += loss.item() * grad_accum
            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v / grad_accum

        gnorm = nn.utils.clip_grad_norm_(policy.parameters(),
                                         cfg.get("grad_clip", 1.0))
        optimizer.step()

        if step % cfg.get("eval_every", 500) == 0 and is_main:
            elapsed = time.time() - t0
            ms_step = elapsed / max(1, step) * 1000
            entry = {"step": step, "loss": round(accum_loss / grad_accum, 5),
                     "lr": round(lr, 9), "grad_norm": round(float(gnorm), 4),
                     "ms_per_step": round(ms_step, 1), **{k: round(v, 4) for k, v in accum_metrics.items()}}
            log.append(entry)
            print(f"  step {step:>6d}  loss {entry['loss']:.4f}  "
                  f"acc {accum_metrics.get('accuracy', 0):.3f}  "
                  f"margin {accum_metrics.get('margin_mean', 0):+.3f}  "
                  f"ms {ms_step:.0f}")
            wb.log(entry)

        if step % cfg.get("ckpt_every", 1000) == 0 and is_main:
            ckpt = output_dir / f"{run_tag}_step{step:07d}.pt"
            _save_ckpt(ckpt, _unwrap(policy), optimizer, step, log, cfg, run_tag)

    if is_main:
        ckpt = output_dir / f"{run_tag}_final.pt"
        _save_ckpt(ckpt, _unwrap(policy), optimizer, total_steps, log, cfg, run_tag)
        wb.log_summary(n_steps=total_steps,
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


def _run_tag(user_tag: str = "") -> str:
    date = datetime.datetime.now().strftime("%Y%m%d")
    suffix = f"_{user_tag}" if user_tag else ""
    return f"{date}_dpo{suffix}"


def _save_ckpt(path: Path, model: nn.Module, optim, step: int,
               log, cfg, run_tag: str) -> None:
    blob = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optim.state_dict(),
        "step": step, "log": log, "config": cfg, "run_tag": run_tag,
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)
    del blob
    gc.collect()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="dpo_h100x8", choices=list(CONFIG_MODULES))
    p.add_argument("--tag", default="")
    p.add_argument("--init-from", default=None)
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
