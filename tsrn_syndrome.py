"""
TSRN syndrome decoder: train on toric code syndrome data, report logical error rates.

Trains one TSRN model per code distance (d=3, 5, 7).
Evaluates at multiple physical error rates for direct comparison to published MWPM.
Checkpoints the best model for each distance.

Usage:
  python tsrn_syndrome.py --steps 20000 --d-model 256 --n-blocks 2
"""

import os, sys, json, time, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tsrn_dml import TSRN, detect_device, get_lr
from syndrome_data import (
    SyndromeDataset, ToricCode, generate_syndrome_data,
    evaluate_decoder_accuracy, evaluate_at_error_rates,
    mwpm_sweep,
)


# ---------------------------------------------------------------------------
#  Training loop for one code distance
# ---------------------------------------------------------------------------

def train_decoder(dataset: SyndromeDataset, device, d: int,
                  n_steps: int = 20000, batch_size: int = 32,
                  lr_max: float = 2e-4, d_model: int = 256,
                  n_blocks: int = 2, n_heads: int = 8,
                  top_k: int = 8, mem_depth: int = 5, dropout: float = 0.1,
                  ckpt_every: int = 2000, resume_from: str = None):
    """Train TSRN decoder for a single code distance."""

    V = dataset.vocab_sz
    ctx = dataset.seq_len  # exact sequence length (d^2 + 1)

    torch.manual_seed(42)
    model = TSRN(vocab=V, d_model=d_model, context_len=ctx,
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

    n_params = model.count_params()
    print(f"\n  TSRN Syndrome Decoder (d={d}): {n_params:,} params")
    print(f"  Vocab: {V}  |  SeqLen: {ctx}  |  d_model: {d_model}  |  Blocks: {n_blocks}")
    print(f"  Steps: {n_steps}  |  Batch: {batch_size}  |  LR: {lr_max}")
    if start_step > 0:
        print(f"  Resuming from step {start_step}")

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
    best_val_acc = 0.0
    best_model_state = None
    eval_every = max(1, n_steps // 50)
    t0 = time.time()

    label_pos = dataset.n_synd  # index where label appears in target

    print(f"\n{'='*80}")
    print(f"  Toric Code d={d} — Syndrome Decoder Training — {n_steps} steps")
    print(f"{'='*80}")
    print(f"{'Step':>6}  {'Loss':>8}  {'ValLoss':>8}  {'ValAcc':>8}  "
          f"{'BestAcc':>8}  {'GNorm':>7}  {'ms/step':>10}")
    print(f"{'-'*70}")

    for step in range(start_step + 1, n_steps + 1):
        lr = get_lr(step, min(n_steps // 10, 2000), n_steps, lr_max, lr_max * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = dataset.batch("train", batch_size, device)
        logits, ar_loss = model(x, y)

        # Extra loss on label position for classification signal
        label_logits = logits[:, label_pos, :2]  # only 0/1 tokens
        label_targets = y[:, label_pos]
        cls_loss = F.cross_entropy(label_logits, label_targets)
        loss = ar_loss + cls_loss  # combined: autoregressive + classification

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Periodic evaluation
        if step % eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                # Validation loss
                vx, vy = dataset.batch("val", min(batch_size, 256), device)
                _, vloss = model(vx, vy)

                # Classification accuracy on val set (sample, batched)
                n_eval = min(2000, len(dataset.val_x))
                ix = torch.randint(len(dataset.val_x), (n_eval,))
                eval_correct = 0
                eval_bs = 128
                for ei in range(0, n_eval, eval_bs):
                    eix = ix[ei:ei+eval_bs]
                    ex = dataset.val_x[eix].to(device)
                    ey = dataset.val_y[eix].to(device)
                    elogits, _ = model(ex)
                    epreds = elogits[:, label_pos, :2].argmax(dim=-1)
                    eval_correct += (epreds == ey[:, label_pos]).sum().item()
                val_acc = eval_correct / n_eval

            elapsed = time.time() - t0
            ms_step = elapsed / step * 1000

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}

            print(f"{step:>6}  {loss.item():>8.4f}  {vloss.item():>8.4f}  "
                  f"{val_acc:>7.1%}  {best_val_acc:>7.1%}  "
                  f"{float(gnorm):>7.3f}  {ms_step:>8.1f}ms")

            log.append({
                "step": step,
                "train_loss": round(loss.item(), 5),
                "val_loss": round(vloss.item(), 5),
                "val_accuracy": round(val_acc, 4),
                "best_val_accuracy": round(best_val_acc, 4),
                "grad_norm": round(float(gnorm), 4),
                "lr": round(lr, 7),
                "wall_time_s": round(elapsed, 1),
                "ms_per_step": round(ms_step, 1),
            })
            model.train()

        # Checkpoint
        if step % ckpt_every == 0:
            ckpt_path = f"checkpoints/tsrn_syndrome_d{d}_{step}steps.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"vocab": V, "d_model": d_model, "context_len": ctx,
                           "n_blocks": n_blocks, "top_k": top_k,
                           "n_heads": n_heads, "mem_depth": mem_depth,
                           "code_distance": d},
                "step": step,
                "best_val_accuracy": best_val_acc,
                "log": log,
            }, ckpt_path)
            print(f"  >> Checkpoint: {ckpt_path}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    model.eval()

    # Save best checkpoint
    best_ckpt = f"checkpoints/tsrn_syndrome_d{d}_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"vocab": V, "d_model": d_model, "context_len": ctx,
                   "n_blocks": n_blocks, "top_k": top_k,
                   "n_heads": n_heads, "mem_depth": mem_depth,
                   "code_distance": d},
        "step": n_steps,
        "best_val_accuracy": best_val_acc,
    }, best_ckpt)
    print(f"  >> Best model saved: {best_ckpt} (val_acc={best_val_acc:.1%})")

    # Final test accuracy at training error rate
    test_result = evaluate_decoder_accuracy(model, dataset, device, split="test")
    print(f"  Test accuracy:       {test_result['accuracy']:.1%}")
    print(f"  Test logical error:  {test_result['logical_error_rate']:.4f}")

    # Sweep error rates
    print(f"\n  Error rate sweep (d={d}):")
    error_rates = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    sweep = evaluate_at_error_rates(model, d, device,
                                     error_rates=error_rates,
                                     n_samples=10_000)
    print(f"  {'p':>6}  {'Raw p_L':>8}  {'TSRN p_L':>9}  {'Acc':>7}")
    print(f"  {'-'*35}")
    for p in error_rates:
        r = sweep[p]
        print(f"  {p:>6.2f}  {r['raw_logical_rate']:>8.4f}  "
              f"{r['logical_error_rate']:>9.4f}  {r['accuracy']:>6.1%}")

    total_time = time.time() - t0
    gpu_energy = total_time / 3600 * 0.186

    results = {
        "code_distance": d,
        "config": {
            "d_model": d_model, "context_len": ctx, "n_blocks": n_blocks,
            "n_heads": n_heads, "top_k": top_k, "mem_depth": mem_depth,
            "n_steps": n_steps, "batch_size": batch_size, "lr": lr_max,
            "vocab_size": V, "p_train": dataset.p_train,
        },
        "params": n_params,
        "best_val_accuracy": round(best_val_acc, 4),
        "test": test_result,
        "error_rate_sweep": {str(k): v for k, v in sweep.items()},
        "training_log": log,
        "hardware": {
            "gpu": "AMD RX 6750 XT (12GB, DirectML)",
            "wall_time_hours": round(total_time / 3600, 2),
            "gpu_energy_kwh": round(gpu_energy, 3),
            "ms_per_step_avg": round(total_time / n_steps * 1000, 1),
        },
    }

    return model, results


# ---------------------------------------------------------------------------
#  Main: train for d=3, 5, 7 and save all results
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TSRN Syndrome Decoder")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mem-depth", type=int, default=5)
    parser.add_argument("--p-train", type=float, default=0.05,
                        help="Physical error rate for training data")
    parser.add_argument("--n-train", type=int, default=200000)
    parser.add_argument("--distances", type=str, default="3,5,7",
                        help="Comma-separated code distances")
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best checkpoint for each distance")
    parser.add_argument("--mwpm", action="store_true",
                        help="Run MWPM baseline comparison")
    args = parser.parse_args()

    distances = [int(x) for x in args.distances.split(",")]

    device = detect_device()
    print(f"\n  Device: {device}")

    all_results = {}

    for d in distances:
        print(f"\n{'#'*80}")
        print(f"  TORIC CODE d={d} — Training TSRN Syndrome Decoder")
        print(f"{'#'*80}")

        # Context length: d^2 syndrome bits + 1 SEP token
        ctx = d * d + 1

        dataset = SyndromeDataset(
            d=d, p_train=args.p_train,
            n_train=args.n_train, n_val=args.n_train // 10,
            n_test=args.n_train // 10,
            context_len=ctx, seed=42 + d,
        )

        resume_path = None
        if args.resume:
            resume_path = f"checkpoints/tsrn_syndrome_d{d}_best.pt"
            if not os.path.exists(resume_path):
                print(f"  No checkpoint found at {resume_path}, training from scratch")
                resume_path = None

        model, results = train_decoder(
            dataset, device, d=d,
            n_steps=args.steps,
            batch_size=args.batch,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            top_k=args.top_k,
            mem_depth=args.mem_depth,
            ckpt_every=args.ckpt_every,
            resume_from=resume_path,
        )

        all_results[d] = results

        # Save per-distance results
        out = f"results/tsrn_syndrome_d{d}_{args.steps}steps.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: {out}")

    # Save combined results
    combined_out = f"results/tsrn_syndrome_combined_{args.steps}steps.json"
    with open(combined_out, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\n  Combined results: {combined_out}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY: TSRN Syndrome Decoder — Toric Code Bit-Flip")
    print(f"{'='*80}")
    print(f"  {'d':>3}  {'Params':>10}  {'BestValAcc':>11}  {'TestAcc':>8}  "
          f"{'Test p_L':>9}  {'Time':>8}")
    print(f"  {'-'*60}")
    for d in distances:
        r = all_results[d]
        print(f"  {d:>3}  {r['params']:>10,}  "
              f"{r['best_val_accuracy']:>10.1%}  "
              f"{r['test']['accuracy']:>7.1%}  "
              f"{r['test']['logical_error_rate']:>9.4f}  "
              f"{r['hardware']['wall_time_hours']:>7.2f}h")

    print(f"\n  Error rate sweep:")
    print(f"  {'p':>6}", end="")
    for d in distances:
        print(f"  {'d='+str(d)+' p_L':>10}", end="")
    print()
    print(f"  {'-'*6}", end="")
    for _ in distances:
        print(f"  {'-'*10}", end="")
    print()
    for p_val in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        print(f"  {p_val:>6.2f}", end="")
        for d in distances:
            sweep = all_results[d]["error_rate_sweep"]
            p_key = str(p_val)
            if p_key in sweep:
                print(f"  {sweep[p_key]['logical_error_rate']:>10.4f}", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()

    # MWPM baseline comparison
    if args.mwpm:
        print(f"\n  Running MWPM baseline for comparison...")
        mwpm_results = {}
        for d in distances:
            print(f"    MWPM d={d}...", end="", flush=True)
            mw = mwpm_sweep(d, error_rates=[0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12],
                            n_samples=10000)
            mwpm_results[d] = mw
            print(f" acc@p=0.05: {mw[0.05]['accuracy']:.1%}")

        print(f"\n  TSRN vs MWPM at p=0.05 (training error rate):")
        print(f"  {'d':>3}  {'MWPM':>8}  {'TSRN val':>10}  {'TSRN test':>10}  {'Winner':>8}")
        print(f"  {'-'*48}")
        for d in distances:
            mwpm_acc = mwpm_results[d][0.05]['accuracy']
            tsrn_val = all_results[d]['best_val_accuracy']
            tsrn_test = all_results[d]['test']['accuracy']
            winner = "TSRN" if tsrn_test > mwpm_acc else "MWPM"
            print(f"  {d:>3}  {mwpm_acc:>7.1%}  {tsrn_val:>9.1%}  {tsrn_test:>9.1%}  {winner:>8}")

        # Save MWPM results
        mwpm_out = f"results/mwpm_baseline.json"
        with open(mwpm_out, "w") as f:
            json.dump({str(d): {str(k): v for k, v in sw.items()}
                       for d, sw in mwpm_results.items()}, f, indent=2)
        print(f"  MWPM results saved: {mwpm_out}")

    print(f"\n  Compare against published MWPM threshold: ~10.3% (bit-flip, code capacity)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
