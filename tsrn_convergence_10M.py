"""
TSRN 100k-step convergence run on enwik8 (10M parameter model).
- TSRN with 9.1M parameters (d_model=256, n_blocks=5, n_heads=4)
- Periodic checkpointing every 10k steps
- Inference samples at each checkpoint
- Final test-set evaluation and quality benchmark
"""

import os, sys, json, time, math, argparse
import torch
import torch.nn as nn

from tsrn_dml import (
    TSRN, CharDatasetSplit, load_enwik8,
    detect_device, evaluate, get_lr,
)
from tsrn_inference import generate_with_stats, run_quality_benchmark

# ---------------------------------------------------------------------------
#  Training loop with periodic checkpointing + inference samples
# ---------------------------------------------------------------------------

def train_convergence(dataset, device, n_steps=100000, batch_size=8,
                      lr_max=2e-4, d_model=256, context_len=256,
                      n_blocks=5, n_heads=4,
                      top_k=16, mem_depth=7, dropout=0.1,
                      ckpt_every=10000, resume_from=None, tag="",
                      grad_accum_steps=1, gradient_checkpoint=False):
    V = dataset.vocab_sz
    ctx = context_len

    torch.manual_seed(42)
    model = TSRN(vocab=V, d_model=d_model, context_len=ctx,
                 gradient_checkpoint=gradient_checkpoint,
                 n_blocks=n_blocks, top_k=top_k, n_heads=n_heads,
                 mem_depth=mem_depth, dropout=dropout)

    start_step = 0
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"  Resumed from {resume_from} at step {start_step}")

    model.to(device)
    model.train()

    print(f"\n  TSRN: {model.count_params():,} parameters")
    print(f"  Vocab: {V}  |  Context: {ctx}  |  d_model: {d_model}")
    eff_batch = batch_size * grad_accum_steps
    print(f"  Steps: {n_steps}  |  Batch: {batch_size}x{grad_accum_steps}={eff_batch}  |  LR: {lr_max}")
    print(f"  Checkpoint every {ckpt_every} steps")

    decay = {p for n, p in model.named_parameters()
             if p.requires_grad and p.dim() >= 2}
    no_decay = {p for p in model.parameters()
                if p.requires_grad and p not in decay}
    optimizer = torch.optim.AdamW([
        {"params": list(decay), "weight_decay": 0.1},
        {"params": list(no_decay), "weight_decay": 0.0},
    ], lr=lr_max, betas=(0.9, 0.95))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    log = []
    best_val_bpc = float("inf")
    best_model_state = None
    eval_every = max(1, n_steps // 50)  # ~50 eval points over run
    t0 = time.time()

    print(f"\n{'='*80}")
    print(f"  TSRN 100k-step Run — enwik8 byte-level (10M params)")
    print(f"{'='*80}")
    print(f"{'Step':>6}  {'TrLoss':>10}  {'TrBPC':>8}  "
          f"{'ValLoss':>9}  {'ValPPL':>9}  {'ValBPC':>8}  {'GNorm':>7}  {'ms/step':>10}")
    print(f"{'-'*85}")

    for step in range(start_step + 1, n_steps + 1):
        lr = get_lr(step, min(n_steps // 10, 4000), n_steps, lr_max, lr_max * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ga in range(grad_accum_steps):
            x, y = dataset.batch("train", batch_size, device)
            _, loss = model(x, y)
            (loss / grad_accum_steps).backward()
            accum_loss += loss.item()
        loss_val = accum_loss / grad_accum_steps
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Periodic evaluation
        if step % eval_every == 0 or step == 1:
            val_loss, val_ppl, val_bpc = evaluate(
                model, dataset, device, batch_size=min(batch_size, 16))
            tr_bpc = loss_val / math.log(2)
            elapsed = time.time() - t0
            ms_step = elapsed / (step - start_step) * 1000
            print(f"{step:>6}  {loss_val:>10.4f}  {tr_bpc:>8.4f}  "
                  f"{val_loss:>9.4f}  {val_ppl:>9.2f}  {val_bpc:>8.4f}  "
                  f"{float(gnorm):>7.3f}  {ms_step:>8.1f}ms")
            log.append({
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
            })

            if val_bpc < best_val_bpc:
                best_val_bpc = val_bpc
                best_model_state = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}

        # Periodic checkpoint + inference sample
        if step % ckpt_every == 0:
            sfx = f"_{tag}" if tag else ""
            ckpt_path = f"checkpoints/tsrn_enwik8_10M{sfx}_{step}steps.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {"vocab": V, "d_model": d_model, "context_len": ctx,
                           "n_blocks": n_blocks, "top_k": top_k,
                           "n_heads": n_heads, "mem_depth": mem_depth},
                "step": step,
                "best_val_bpc": best_val_bpc,
                "log": log,
            }, ckpt_path)
            print(f"  >> Checkpoint saved: {ckpt_path}")

            # Quick inference sample
            model.eval()
            try:
                gen = generate_with_stats(
                    model, dataset, device, prompt="The history of ",
                    n_tokens=200, temperature=0.8, top_p=0.9,
                    top_k=40, repetition_penalty=1.1)
                sample = gen["full_text"][:200].replace('\n', ' ')
                safe = sample.encode('ascii', errors='replace').decode('ascii')
                print(f"  >> Sample @ step {step}: {safe}")
            except Exception as e:
                print(f"  >> Sample failed: {e}")
            model.train()

            # Save incremental results
            _save_results(log, step, model, best_val_bpc, tag)

    # Save best model checkpoint
    sfx = f"_{tag}" if tag else ""
    if best_model_state is not None:
        best_ckpt_path = f"checkpoints/tsrn_enwik8_10M{sfx}_best.pt"
        torch.save({
            "model_state_dict": best_model_state,
            "config": {"vocab": V, "d_model": d_model, "context_len": ctx,
                       "n_blocks": n_blocks, "top_k": top_k,
                       "n_heads": n_heads, "mem_depth": mem_depth},
            "best_val_bpc": round(best_val_bpc, 4),
        }, best_ckpt_path)
        print(f"\n  >> Best model saved: {best_ckpt_path} (val_bpc={best_val_bpc:.4f})")
        # Load best model for final evaluation
        model.load_state_dict(best_model_state)
        model.to(device)

    # Final evaluation
    print(f"\n{'='*80}")
    print(f"  Final Evaluation (best model, val_bpc={best_val_bpc:.4f})")
    print(f"{'='*80}")
    model.eval()
    val_loss, val_ppl, val_bpc = evaluate(
        model, dataset, device, n_batches=200,
        batch_size=min(batch_size, 16), split="val")
    print(f"  Val:  PPL={val_ppl:.3f}  BPC={val_bpc:.4f}")

    test_loss, test_ppl, test_bpc = evaluate(
        model, dataset, device, n_batches=200,
        batch_size=min(batch_size, 16), split="test")
    print(f"  Test: PPL={test_ppl:.3f}  BPC={test_bpc:.4f}")

    # Save final checkpoint
    sfx = f"_{tag}" if tag else ""
    ckpt_path = f"checkpoints/tsrn_enwik8_10M{sfx}_final_{n_steps}steps.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"vocab": V, "d_model": d_model, "context_len": ctx,
                   "n_blocks": n_blocks, "top_k": top_k,
                   "n_heads": n_heads, "mem_depth": mem_depth},
        "step": n_steps,
        "test_bpc": round(test_bpc, 4),
        "test_ppl": round(test_ppl, 3),
        "best_val_bpc": round(best_val_bpc, 4),
        "log": log,
    }, ckpt_path)
    print(f"  Final checkpoint: {ckpt_path}")

    # Quality benchmark
    print(f"\n  Running quality benchmark...")
    quality = run_quality_benchmark(model, dataset, device, "TSRN")

    # Wall-clock time summary
    total_time = time.time() - t0
    gpu_energy_kwh = total_time / 3600 * 0.186  # RX 6750 XT TBP = 186W

    # Save final results
    results = {
        "run": "tsrn_enwik8_10M_convergence",
        "config": {
            "d_model": d_model, "context": ctx, "n_blocks": n_blocks,
            "n_heads": n_heads, "top_k": top_k, "mem_depth": mem_depth,
            "n_steps": n_steps, "batch_size": batch_size, "lr": lr_max,
            "vocab_size": V, "tokenization": "byte-level (latin-1)",
        },
        "params": model.count_params(),
        "val": {"bpc": round(val_bpc, 4), "ppl": round(val_ppl, 3)},
        "test": {"bpc": round(test_bpc, 4), "ppl": round(test_ppl, 3)},
        "best_val_bpc": round(best_val_bpc, 4),
        "training_log": log,
        "quality_benchmark": quality,
        "hardware": {
            "gpu": "AMD RX 6750 XT (12GB, DirectML)",
            "gpu_tdp_watts": 186,
            "wall_time_hours": round(total_time / 3600, 2),
            "gpu_energy_kwh": round(gpu_energy_kwh, 3),
            "ms_per_step_avg": round(total_time / n_steps * 1000, 1),
        },
    }

    sfx = f"_{tag}" if tag else ""
    out = f"results/tsrn_enwik8_10M{sfx}_{n_steps}steps.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out}")

    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Model:    TSRN ({model.count_params():,} params)")
    print(f"  Dataset:  enwik8 (byte-level, standard 90M/5M/5M)")
    print(f"  Steps:    {n_steps:,}")
    print(f"  Test BPC: {test_bpc:.4f}  (PPL: {test_ppl:.3f})")
    print(f"  Best Val: {best_val_bpc:.4f}")
    print(f"  Time:     {total_time/3600:.1f} hours")
    print(f"  Energy:   {gpu_energy_kwh:.2f} kWh (GPU @ 186W TBP)")
    print(f"{'='*80}")

    return results


def _save_results(log, step, model, best_val_bpc, tag=""):
    """Save incremental results to disk."""
    sfx = f"_{tag}" if tag else ""
    out = f"results/tsrn_enwik8_10M{sfx}_progress_{step}steps.json"
    with open(out, "w") as f:
        json.dump({
            "step": step,
            "params": model.count_params(),
            "best_val_bpc": round(best_val_bpc, 4),
            "log": log,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TSRN 10M Convergence Run")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for checkpoint/results filenames")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch * grad_accum)")
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="Enable gradient checkpointing on TSRN blocks to save VRAM")
    args = parser.parse_args()

    device = detect_device()
    print(f"\n  Device: {device}")

    print(f"\n-- Loading enwik8 (byte-level, standard protocol) --")
    dataset = load_enwik8(context_len=args.context)

    train_convergence(
        dataset, device,
        n_steps=args.steps,
        batch_size=args.batch,
        d_model=args.d_model,
        context_len=args.context,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        resume_from=args.resume,
        ckpt_every=args.ckpt_every,
        tag=args.tag,
        grad_accum_steps=args.grad_accum,
        gradient_checkpoint=args.grad_ckpt,
    )


if __name__ == "__main__":
    main()
