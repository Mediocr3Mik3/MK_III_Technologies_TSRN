"""
TSRN Local Pretraining — DirectML single-GPU.
================================================
Overtrain a ~22M param TSRNGist on the full tokenized corpus.

Usage:
    python research/train_pretrain_local.py \
        --steps 250000 --batch 8 --context 256 --grad-accum 4 \
        --d-model 256 --n-blocks 3 --n-heads 4 \
        --lr 0.0003 --tag v2_kleene_directml

Resume:
    python research/train_pretrain_local.py \
        --resume checkpoints/.../step_XXXXXXX.pt ...
"""

from __future__ import annotations

import argparse
import gc
import json
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

# Ensure research/ modules resolve correctly
_RESEARCH_DIR = Path(__file__).resolve().parent
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR))

from model_config import nano_directml_config
from tsrn_gist import TSRNGist
from tsrn_dml import AdamWDML
from cloud.azure.data.streaming_dataset import TokenShardStream
from cloud.azure.data.special_tokens import build_special_tokens


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


@torch.no_grad()
def quick_eval(model: nn.Module, loader: DataLoader, device: torch.device,
               n_batches: int = 50) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    losses: List[float] = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch["input_ids"].to(device)
        y = batch["targets"].to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    if was_training:
        model.train()
    if not losses:
        return {"val_loss": float("inf"), "val_ppl": float("inf"), "val_bpc": float("inf")}
    val_loss = sum(losses) / len(losses)
    return {"val_loss": val_loss, "val_ppl": math.exp(val_loss), "val_bpc": val_loss / math.log(2)}


REASONING_TESTS = [
    {"name": "bracket_tracking", "prompt": "Complete this Wikipedia-style markup correctly:\n[[Eiffel Tower|",
     "check": lambda s: "]]" in s},
    {"name": "constraint_resolution", "prompt": "Rule: If A then B. Rule: If B then C. Given: A is true. Therefore:",
     "check": lambda s: "c" in s.lower() or "true" in s.lower()},
    {"name": "code_structure", "prompt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
     "check": lambda s: "fibonacci" in s or "n-1" in s or "n - 1" in s},
    {"name": "nested_structure", "prompt": "{{Infobox person\n| name = Albert Einstein\n| birth_date = ",
     "check": lambda s: any(c.isdigit() for c in s[:20])},
    {"name": "math_chain", "prompt": "If x = 5 and y = x + 3, then y = 8. If z = y * 2, then z =",
     "check": lambda s: "16" in s[:30]},
]


@torch.no_grad()
def run_reasoning_tests(model: TSRNGist, tokenizer, device) -> Dict[str, Any]:
    model.eval()
    passed = 0
    results = {}
    for test in REASONING_TESTS:
        try:
            ids = tokenizer.encode(test["prompt"])
            x = torch.tensor([ids], dtype=torch.long, device=device)
            gen = []
            for _ in range(50):
                if x.shape[1] > 512:
                    x = x[:, -512:]
                logits, _ = model(x)
                nxt = logits[0, -1, :].argmax().item()
                gen.append(nxt)
                x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
            cont = tokenizer.decode(gen)
            ok = test["check"](cont)
            results[test["name"]] = {"passed": ok, "continuation": cont[:100]}
            if ok:
                passed += 1
        except Exception as e:
            results[test["name"]] = {"passed": False, "error": str(e)}
    model.train()
    return {"passed": passed, "total": len(REASONING_TESTS), "pass_rate": passed / len(REASONING_TESTS), "tests": results}


@torch.no_grad()
def generate_sample(model: TSRNGist, tokenizer, device, prompt: str = "The history of", max_tokens: int = 150) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_tokens):
        if x.shape[1] > 512:
            x = x[:, -512:]
        logits, _ = model(x)
        probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    model.train()
    return tokenizer.decode(x[0, len(ids):].tolist())


def _save_ckpt(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
               step: int, best_val_bpc: float, log: List[Dict], cfg: Any, run_tag: str) -> None:
    blob = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_bpc": best_val_bpc,
        "log": log,
        "config": cfg.__dict__,
        "run_tag": run_tag,
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


def detect_device():
    try:
        import torch_directml
        dev = torch_directml.device()
        print(f"  DirectML: {dev}")
        return dev
    except ImportError:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {dev}")
        return dev


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=250_000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--mem-depth", type=int, default=5)
    p.add_argument("--max-gists", type=int, default=32)
    p.add_argument("--ckpt-every", type=int, default=2_000)
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--sample-every", type=int, default=5_000)
    p.add_argument("--log-every", type=int, default=50,
                   help="Steps between averaged log rows (default 50)")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--tag", type=str, default="v2_kleene_directml")
    p.add_argument("--manifest", type=str,
                   default="research/cloud/azure/data/manifests/pretrain_mix.yaml")
    p.add_argument("--tokens-dir", type=str, default="D:/ml/tsrn_data/shards/pretrain")
    p.add_argument("--output-dir", type=str, default="D:/ml/tsrn_data/checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int,
                   default=0 if os.name == "nt" else 2,
                   help="DataLoader workers (default 0 on Windows, 2 on Linux)")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--no-reservoir", action="store_true",
                   help="Ablate the echo-state reservoir (s1 block 0); KleeneSSM carries state mixing")
    p.add_argument("--linear-attn", action="store_true",
                   help="Use exact O(T) linear tropical attention instead of the O(T^2) path")
    p.add_argument("--oscillatory", action="store_true",
                   help="Use damped oscillatory state-space memory (LinOSS) instead of the GRU-ESN reservoir")
    args = p.parse_args(argv)

    device = detect_device()
    torch.manual_seed(args.seed)

    run_tag = f"{time.strftime('%Y%m%d')}_{args.tag}_d{args.d_model}_ctx{args.context}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  TSRN Local Pretrain — {run_tag}")
    print(f"{'='*72}")
    print(f"  Steps: {args.steps:,}  |  Batch: {args.batch}x{args.grad_accum}={args.batch*args.grad_accum}")
    print(f"  Context: {args.context}  |  d_model: {args.d_model}  |  Heads: {args.n_heads}")
    print(f"  LR: {args.lr}  |  Curriculum: {'OFF' if args.no_curriculum else 'ON'}")

    # Tokenizer
    from transformers import GPT2TokenizerFast
    tok_path = Path("D:/ml/tsrn_data/tokenizer/gpt2_tsrn_base")
    tokenizer = GPT2TokenizerFast.from_pretrained(str(tok_path))
    vocab_size = len(tokenizer)
    eos_id = tokenizer.eos_token_id
    print(f"  Vocab: {vocab_size:,}")

    # Model config
    cfg = nano_directml_config(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        context_len=args.context,
        top_k=args.top_k,
        padic_depth=args.mem_depth,
        max_gists=args.max_gists,
        use_reservoir=not args.no_reservoir,
        reservoir_kind=("oscillatory" if args.oscillatory else "echo"),
        use_linear_attention=args.linear_attn,
        use_mixed_precision=False,
    )

    # Build model
    model = TSRNGist(
        vocab=vocab_size,
        d_model=cfg.d_model,
        context_len=cfg.context_len,
        gradient_checkpoint=False,
        n_blocks=cfg.n_blocks,
        top_k=cfg.top_k,
        n_heads=cfg.n_heads,
        mem_depth=cfg.padic_depth,
        max_gists=cfg.max_gists,
        gist_top_k=cfg.gist_top_k,
        dropout=cfg.dropout,
        use_hyperbolic=False,
        gist_chaining=False,
        config=cfg,
    )
    model.to(device)
    model.train()
    n_params = model.count_params()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer — AdamWDML avoids the aten::lerp DirectML CPU fallback that
    # plain torch.optim.AdamW triggers every step (per-param CPU round-trip).
    # No weight decay on 1-D params (biases, LayerNorm), decay on matrices.
    decay_params, no_decay_params = [], []
    for _n, _p in model.named_parameters():
        if not _p.requires_grad:
            continue
        (decay_params if _p.dim() >= 2 else no_decay_params).append(_p)
    optimizer = AdamWDML(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Data streams
    progress_path = str(output_dir / "curriculum_progress.txt")
    curriculum_cfg = None if args.no_curriculum else {"mode": "soft"}

    train_stream = TokenShardStream(
        manifest=args.manifest,
        tokens_dir=args.tokens_dir,
        context_len=args.context,
        seed=args.seed,
        eos_id=eos_id,
        curriculum=curriculum_cfg,
        progress_path=progress_path if not args.no_curriculum else None,
    )
    val_stream = TokenShardStream(
        manifest=args.manifest,
        tokens_dir=args.tokens_dir,
        context_len=args.context,
        seed=args.seed + 9999,
        eos_id=eos_id,
        curriculum={"mode": "off"},
    )

    nw = args.num_workers
    train_loader = DataLoader(train_stream, batch_size=args.batch,
                              num_workers=nw, pin_memory=False,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_stream, batch_size=args.batch,
                            num_workers=min(2, nw), pin_memory=False)

    # Resume
    start_step = 0
    best_val_bpc = float("inf")
    log: List[Dict] = []
    resume_blob = None
    if args.resume and os.path.exists(args.resume):
        print(f"\n  Resuming from {args.resume}")
        resume_blob = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_blob["model_state_dict"])
        optimizer.load_state_dict(resume_blob["optimizer_state_dict"])
        start_step = resume_blob.get("step", 0)
        best_val_bpc = resume_blob.get("best_val_bpc", float("inf"))
        log = list(resume_blob.get("log", []))
        print(f"  Resumed at step {start_step:,}, best BPC {best_val_bpc:.4f}")

    model.to(device).train()

    # Write curriculum progress before loop
    if not args.no_curriculum:
        TokenShardStream.write_progress(progress_path, start_step / max(1, args.steps))

    print(f"\n  {'Step':>7}  {'TrLoss':>8}  {'TrBPC':>7}  {'ValBPC':>7}  {'LR':>9}  {'GNorm':>7}  {'ms/step':>8}")
    print(f"  {'-'*68}")

    train_iter = iter(train_loader)
    train_loss_acc = 0.0
    step_times = []
    t_start = time.perf_counter()

    for step in range(start_step, args.steps):
        lr = get_lr(step, 2000, args.steps, args.lr, args.lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = lr

        h = maslov_h(step, args.steps, 1.5, 0.3, 3)
        model.set_maslov_h(h)

        if step % 100 == 1:
            model.gist_buffer.reset()

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for ga in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x = batch["input_ids"].to(device)
            y = batch["targets"].to(device)
            _, loss = model(x, y)
            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item() * args.grad_accum

        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        t_end = time.perf_counter()
        step_ms = (t_end - t_start) * 1000
        step_times.append(step_ms)
        t_start = t_end
        train_loss_acc += accum_loss / args.grad_accum

        # Curriculum progress
        if not args.no_curriculum and step % 50 == 0:
            TokenShardStream.write_progress(progress_path, step / max(1, args.steps))

        # Logging. Per-step heartbeat for the first few steps gives immediate
        # proof-of-life + a measured ms/step (DirectML is slow to the first
        # averaged row); then settle into the averaged --log-every cadence.
        if (step - start_step) < 5:
            inst_loss = accum_loss / args.grad_accum
            print(f"  {step+1:>7}  {inst_loss:>8.4f}  {inst_loss/math.log(2):>7.4f}  "
                  f"{'':>7}  {lr:>9.2e}  {float(gnorm):>7.3f}  {step_ms:>8.1f}")
        if (step + 1) % args.log_every == 0:
            avg_loss = train_loss_acc / args.log_every
            avg_bpc = avg_loss / math.log(2)
            avg_ms = sum(step_times[-100:]) / len(step_times[-100:])
            print(f"  {step+1:>7}  {avg_loss:>8.4f}  {avg_bpc:>7.4f}  {'':>7}  {lr:>9.2e}  {float(gnorm):>7.3f}  {avg_ms:>8.1f}")
            train_loss_acc = 0.0

        # Validation
        if (step + 1) % args.eval_every == 0:
            metrics = quick_eval(model, val_loader, device)
            print(f"  {step+1:>7}  {'':>8}  {'':>7}  {metrics['val_bpc']:>7.4f}  {'':>9}  {'':>7}  {'VAL':>8}")
            log.append({"step": step + 1, **metrics, "lr": lr, "gnorm": float(gnorm)})
            if metrics["val_bpc"] < best_val_bpc:
                best_val_bpc = metrics["val_bpc"]
                _save_ckpt(output_dir / f"{run_tag}_best.pt", model, optimizer,
                           step + 1, best_val_bpc, log, cfg, run_tag)

        # Checkpoint
        if (step + 1) % args.ckpt_every == 0:
            ckpt_path = output_dir / f"{run_tag}_step{step+1:07d}.pt"
            _save_ckpt(ckpt_path, model, optimizer, step + 1,
                       best_val_bpc, log, cfg, run_tag)
            print(f"  >> Checkpoint: {ckpt_path.name}")
            _prune_ckpts(output_dir, run_tag, keep=5)

        # Reasoning tests + sample
        if (step + 1) % args.sample_every == 0:
            sample = generate_sample(model, tokenizer, device)
            print(f"  >> Sample @ {step+1}: {sample[:120]}")
            reasoning = run_reasoning_tests(model, tokenizer, device)
            print(f"  >> Reasoning: {reasoning['passed']}/{reasoning['total']} passed ({reasoning['pass_rate']:.0%})")
            for name, res in reasoning["tests"].items():
                status = "OK" if res["passed"] else "XX"
                print(f"       {status} {name}: {res.get('continuation', '')[:60]}")
            with open(log_dir / f"{run_tag}_reasoning.json", "w") as f:
                json.dump({"step": step + 1, "sample": sample, **reasoning}, f, indent=2)

    # Final
    print(f"\n{'='*72}")
    print(f"  Training complete. Best val BPC: {best_val_bpc:.4f}")
    print(f"  Final checkpoint: {output_dir}/{run_tag}_best.pt")
    _save_ckpt(output_dir / f"{run_tag}_final.pt", model, optimizer,
               args.steps, best_val_bpc, log, cfg, run_tag)


if __name__ == "__main__":
    main()
