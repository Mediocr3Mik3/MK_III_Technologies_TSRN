"""
TSRN-Gist convergence run on enwik8 (byte-level, standard protocol).
- TSRNGist model (Clifford multivector gists + sheaf rotor diffusion)
- Periodic checkpointing every 5k steps
- Inference samples at each checkpoint
- Final test-set evaluation and quality benchmark
- NEXUS Innovation #3 Maslov temperature cycling on tropical attention
- NEXUS Innovation #4 Sheaf Harmonic positional encoding (model-side)
- NEXUS Innovation #5 RG fixed-point weight sharing for Scale-2 (model-side)
"""

import os, sys, json, time, math, argparse, datetime, gc
import torch
import torch.nn as nn

from tsrn_gist import (
    TSRNGist, load_enwik8,
    detect_device, evaluate, get_lr,
)
from tsrn_dml import AdamWDML
from tsrn_inference import generate_with_stats, run_quality_benchmark


# ---------------------------------------------------------------------------
#  Filename tag — YYYYMMDD_<script>[_<user_tag>]
# ---------------------------------------------------------------------------

def build_run_tag(user_tag: str = "") -> str:
    """Build run identifier: YYYYMMDD_<script>[_<user_tag>]."""
    date = datetime.datetime.now().strftime("%Y%m%d")
    script = os.path.splitext(os.path.basename(sys.argv[0] or "run"))[0]
    suffix = f"_{user_tag}" if user_tag else ""
    return f"{date}_{script}{suffix}"


def ckpt_path(run_tag: str, step: int, kind: str = "step") -> str:
    """kind in {'step', 'best', 'final'}.  Layout under checkpoints/."""
    if kind == "best":
        return f"checkpoints/{run_tag}_best.pt"
    if kind == "final":
        return f"checkpoints/{run_tag}_final_step{step:06d}.pt"
    return f"checkpoints/{run_tag}_step{step:06d}.pt"


def results_path(run_tag: str, step: int, kind: str = "progress") -> str:
    """kind in {'progress', 'final'}.  Layout under results/."""
    if kind == "final":
        return f"results/{run_tag}_final_step{step:06d}.json"
    return f"results/{run_tag}_progress_step{step:06d}.json"


# ---------------------------------------------------------------------------
#  Maslov temperature schedule — NEXUS Innovation #3
# ---------------------------------------------------------------------------

def maslov_h_schedule(step: int, n_steps: int,
                      h_warm: float = 1.5, h_cool: float = 0.3,
                      n_cycles: int = 3) -> float:
    """Cycle Maslov h from warm (classical) to cool (tropical) over training.

    Cosine cycling: h(t) = (h_warm+h_cool)/2 + (h_warm-h_cool)/2 * cos(2*pi*c*t)
    with c = n_cycles / n_steps.  Starts warm, ends cool.
    """
    t = max(0.0, min(1.0, step / max(1, n_steps)))
    mid = 0.5 * (h_warm + h_cool)
    amp = 0.5 * (h_warm - h_cool)
    return float(mid + amp * math.cos(2.0 * math.pi * n_cycles * t))

# ---------------------------------------------------------------------------
#  Training loop with periodic checkpointing + inference samples
# ---------------------------------------------------------------------------

def train_convergence(dataset, device, n_steps=70000, batch_size=8,
                      lr_max=2e-4, d_model=512, context_len=256,
                      n_blocks=3, n_heads=8,
                      top_k=16, mem_depth=7, dropout=0.1,
                      max_gists=64, gist_top_k=4,
                      ckpt_every=5000, resume_from=None, tag="",
                      grad_accum_steps=1, gradient_checkpoint=False):
    V = dataset.vocab_sz
    ctx = context_len

    torch.manual_seed(42)
    model = TSRNGist(vocab=V, d_model=d_model, context_len=ctx,
                     gradient_checkpoint=gradient_checkpoint,
                     n_blocks=n_blocks, top_k=top_k, n_heads=n_heads,
                     mem_depth=mem_depth, max_gists=max_gists,
                     gist_top_k=gist_top_k, dropout=dropout)

    # Load model weights first (optimizer state restored below, after it's built).
    start_step = 0
    _resume_ckpt = None
    if resume_from and os.path.exists(resume_from):
        _resume_ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(_resume_ckpt["model_state_dict"])
        start_step = _resume_ckpt.get("step", 0)

    model.to(device)
    model.train()

    # Run tag for all output artifacts: YYYYMMDD_tsrn_convergence_gist[_tag]
    run_tag = build_run_tag(tag)

    print(f"\n  TSRNGist: {model.count_params():,} parameters")
    print(f"  Vocab: {V}  |  Context: {ctx}  |  d_model: {d_model}")
    eff_batch = batch_size * grad_accum_steps
    print(f"  Steps: {n_steps}  |  Batch: {batch_size}x{grad_accum_steps}={eff_batch}  |  LR: {lr_max}")
    print(f"  Gists: {max_gists} buffer, top-{gist_top_k} retrieval")
    print(f"  Checkpoint every {ckpt_every} steps")
    print(f"  Run tag: {run_tag}")
    print(f"  NEXUS: Maslov cycling (h: 1.5->0.3, 3 cycles), "
          f"Sheaf Harmonic PE, RG fixed-point S2 (max_iters={model.s2_max_iters}, eps={model.s2_eps})")

    # Build the two param groups as LISTS, not sets.
    # Sets iterate in hash order; for nn.Parameter, hash defaults to id()
    # = memory address, which Windows ASLR randomizes across processes.
    # That made optimizer.load_state_dict() positionally misalign moments
    # with parameters on resume, causing shape-mismatch crashes.
    # named_parameters() iterates in module-registration order (fully
    # deterministic within and across runs given the same architecture).
    decay_params, no_decay_params = [], []
    for _n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    optimizer = AdamWDML([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr_max, betas=(0.9, 0.95))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    log = []
    best_val_bpc = float("inf")
    best_ckpt_path = None  # path on disk; we no longer hold the dict in RAM

    # Full resume: restore optimizer moments, best-so-far BPC, and log.
    # Without optimizer state, Adam's exp_avg/exp_avg_sq reset to zero
    # at the resume step, producing a large effective-LR spike for the
    # first ~100-500 steps.
    if _resume_ckpt is not None:
        opt_status = "no state in ckpt"
        if "optimizer_state_dict" in _resume_ckpt:
            try:
                optimizer.load_state_dict(_resume_ckpt["optimizer_state_dict"])
                opt_status = "restored"
            except Exception as e:
                # Checkpoint produced before the set->list ordering fix will
                # have an optimizer state whose positional indices correspond
                # to the original (non-deterministic) hash order.  We cannot
                # reconstruct that order, so fall back to fresh moments.
                # Expect a small LR spike for ~100 steps after resume; loss
                # will recover quickly.
                opt_status = f"FRESH (load failed: {type(e).__name__})"
        best_val_bpc = _resume_ckpt.get("best_val_bpc", float("inf"))
        log = list(_resume_ckpt.get("log", []))
        print(f"  Resumed from {resume_from} at step {start_step} "
              f"(best_val_bpc so far = {best_val_bpc:.4f}, "
              f"opt state = {opt_status})")
        del _resume_ckpt  # release CPU memory

    eval_every = max(1, n_steps // 50)  # ~50 eval points over run
    t0 = time.time()

    print(f"\n{'='*80}")
    print(f"  TSRNGist Convergence — enwik8 byte-level — {n_steps} steps")
    print(f"{'='*80}")
    print(f"{'Step':>6}  {'TrLoss':>10}  {'TrBPC':>8}  "
          f"{'ValLoss':>9}  {'ValPPL':>9}  {'ValBPC':>8}  {'GNorm':>7}  {'ms/step':>10}")
    print(f"{'-'*85}")

    # Track the most recent completed step for interrupt-time checkpointing.
    _last_completed_step = start_step
    try:
      for step in range(start_step + 1, n_steps + 1):
        lr = get_lr(step, min(n_steps // 10, 4000), n_steps, lr_max, lr_max * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # NEXUS Innovation #3 — Maslov temperature cycling (per-step scalar).
        # h is a buffer inside every TropicalAttention layer.  Updating it is
        # a cheap scalar op and preserves causality (causal mask applied after).
        h_now = maslov_h_schedule(step, n_steps)
        model.set_maslov_h(h_now)

        # Reset gist buffer periodically to avoid stale gists
        if step % 100 == 1:
            model.gist_buffer.reset()

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

        # Periodic gc to limit DirectML heap fragmentation on long runs.
        # DML has no empty_cache() API and does not compact its allocator,
        # so Python refs to intermediate tensors outliving a single step
        # gradually fragment the heap and eventually OOM on trivially
        # small allocations.  With --grad-ckpt enabled, each backward
        # pass spawns extra recomputed intermediates, so we collect more
        # aggressively (every 200 steps).  Cheap (~1 ms on a clean graph).
        gc_every = 200 if gradient_checkpoint else 500
        if step % gc_every == 0:
            gc.collect()

        # Periodic evaluation
        if step % eval_every == 0 or step == 1:
            model.gist_buffer.reset()
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
                "maslov_h": round(h_now, 4),
                "s2_iters_used": int(model._last_s2_iters.item()),
            })

            if val_bpc < best_val_bpc:
                best_val_bpc = val_bpc
                # Write best checkpoint to disk immediately and drop the
                # CPU dict.  Holding ``best_model_state`` for the full run
                # cost ~88-400 MB of resident CPU memory and forced a
                # full GPU→CPU sweep on every val improvement (frequent
                # early in training).
                best_ckpt_path = ckpt_path(run_tag, step=0, kind="best")
                cpu_state = {k: v.detach().cpu()
                             for k, v in model.state_dict().items()}
                torch.save({
                    "model_state_dict": cpu_state,
                    "config": {"vocab": V, "d_model": d_model, "context_len": ctx,
                               "n_blocks": n_blocks, "top_k": top_k,
                               "n_heads": n_heads, "mem_depth": mem_depth,
                               "max_gists": max_gists, "gist_top_k": gist_top_k},
                    "best_val_bpc": round(best_val_bpc, 4),
                    "step": step,
                    "run_tag": run_tag,
                }, best_ckpt_path)
                del cpu_state
                gc.collect()

        # Periodic checkpoint + inference sample
        if step % ckpt_every == 0:
            cpath = ckpt_path(run_tag, step, kind="step")
            # Move state_dicts to CPU explicitly before save so torch.save
            # does not pin GPU memory for serialization.  Drop refs after.
            cpu_model = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            cpu_opt = optimizer.state_dict()  # already CPU-mappable, but copies on save
            torch.save({
                "model_state_dict": cpu_model,
                "optimizer_state_dict": cpu_opt,
                "config": {"vocab": V, "d_model": d_model, "context_len": ctx,
                           "n_blocks": n_blocks, "top_k": top_k,
                           "n_heads": n_heads, "mem_depth": mem_depth,
                           "max_gists": max_gists, "gist_top_k": gist_top_k},
                "step": step,
                "best_val_bpc": best_val_bpc,
                "log": log,
                "run_tag": run_tag,
            }, cpath)
            del cpu_model, cpu_opt
            gc.collect()
            print(f"  >> Checkpoint saved: {cpath}")

            # Quick inference sample
            model.eval()
            model.gist_buffer.reset()
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
            _save_results(log, step, model, best_val_bpc, run_tag)

        _last_completed_step = step
    except KeyboardInterrupt:
        interrupt_step = _last_completed_step
        ipath = f"checkpoints/{run_tag}_interrupt_step{interrupt_step:06d}.pt"
        print(f"\n\n  [INTERRUPT] Ctrl+C caught at step {interrupt_step}. "
              f"Saving emergency checkpoint -> {ipath}")
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {"vocab": V, "d_model": d_model, "context_len": ctx,
                           "n_blocks": n_blocks, "top_k": top_k,
                           "n_heads": n_heads, "mem_depth": mem_depth,
                           "max_gists": max_gists, "gist_top_k": gist_top_k},
                "step": interrupt_step,
                "best_val_bpc": best_val_bpc,
                "log": log,
                "run_tag": run_tag,
                "interrupted": True,
            }, ipath)
            _save_results(log, interrupt_step, model, best_val_bpc, run_tag)
            print(f"  [INTERRUPT] Saved. Resume with: --resume {ipath}")
        except Exception as e:
            print(f"  [INTERRUPT] Save failed: {e}")
        return  # skip best-model save + final eval; exit cleanly

    # Best model already saved to disk during training.  Load it for final eval.
    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
        print(f"\n  >> Best model on disk: {best_ckpt_path} (val_bpc={best_val_bpc:.4f})")
        best_blob = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(best_blob["model_state_dict"])
        del best_blob
        gc.collect()
        model.to(device)

    # Final evaluation
    print(f"\n{'='*80}")
    print(f"  Final Evaluation (best model, val_bpc={best_val_bpc:.4f})")
    print(f"{'='*80}")
    model.eval()
    model.gist_buffer.reset()
    val_loss, val_ppl, val_bpc = evaluate(
        model, dataset, device, n_batches=200,
        batch_size=min(batch_size, 16), split="val")
    print(f"  Val:  PPL={val_ppl:.3f}  BPC={val_bpc:.4f}")

    model.gist_buffer.reset()
    test_loss, test_ppl, test_bpc = evaluate(
        model, dataset, device, n_batches=200,
        batch_size=min(batch_size, 16), split="test")
    print(f"  Test: PPL={test_ppl:.3f}  BPC={test_bpc:.4f}")

    # Save final checkpoint
    final_cpath = ckpt_path(run_tag, step=n_steps, kind="final")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"vocab": V, "d_model": d_model, "context_len": ctx,
                   "n_blocks": n_blocks, "top_k": top_k,
                   "n_heads": n_heads, "mem_depth": mem_depth,
                   "max_gists": max_gists, "gist_top_k": gist_top_k},
        "step": n_steps,
        "test_bpc": round(test_bpc, 4),
        "test_ppl": round(test_ppl, 3),
        "best_val_bpc": round(best_val_bpc, 4),
        "log": log,
        "run_tag": run_tag,
    }, final_cpath)
    print(f"  Final checkpoint: {final_cpath}")

    # Quality benchmark
    print(f"\n  Running quality benchmark...")
    model.gist_buffer.reset()
    quality = run_quality_benchmark(model, dataset, device, "TSRNGist")

    # Wall-clock time summary
    total_time = time.time() - t0
    gpu_energy_kwh = total_time / 3600 * 0.186  # RX 6750 XT TBP = 186W

    # Save final results
    results = {
        "run": "tsrn_gist_enwik8_convergence",
        "config": {
            "d_model": d_model, "context": ctx, "n_blocks": n_blocks,
            "n_heads": n_heads, "top_k": top_k, "mem_depth": mem_depth,
            "max_gists": max_gists, "gist_top_k": gist_top_k,
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

    out = results_path(run_tag, n_steps, kind="final")
    # Embed run_tag and innovation flags for traceability
    results["run_tag"] = run_tag
    results["innovations"] = {
        "maslov_cycling": True,
        "sheaf_harmonic_pe": True,
        "rg_fixed_point_s2": True,
        "s2_max_iters": model.s2_max_iters,
        "s2_eps": model.s2_eps,
    }
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out}")

    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Model:    TSRNGist ({model.count_params():,} params)")
    print(f"  Dataset:  enwik8 (byte-level, standard 90M/5M/5M)")
    print(f"  Steps:    {n_steps:,}")
    print(f"  Test BPC: {test_bpc:.4f}  (PPL: {test_ppl:.3f})")
    print(f"  Best Val: {best_val_bpc:.4f}")
    print(f"  Time:     {total_time/3600:.1f} hours")
    print(f"  Energy:   {gpu_energy_kwh:.2f} kWh (GPU @ 186W TBP)")
    print(f"{'='*80}")

    return results


def _save_results(log, step, model, best_val_bpc, run_tag: str):
    """Save incremental progress results to disk with stable naming."""
    out = results_path(run_tag, step, kind="progress")
    with open(out, "w") as f:
        json.dump({
            "run_tag": run_tag,
            "step": step,
            "params": model.count_params(),
            "best_val_bpc": round(best_val_bpc, 4),
            "maslov_h": round(float(model.get_maslov_h()), 4),
            "s2_iters_used": int(model._last_s2_iters.item()),
            "log": log,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TSRNGist Convergence Run")
    parser.add_argument("--steps", type=int, default=70000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--max-gists", type=int, default=64,
                        help="Max gists in ring buffer")
    parser.add_argument("--gist-top-k", type=int, default=4,
                        help="Number of gists to retrieve per forward pass")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for checkpoint/results filenames")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch * grad_accum)")
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="Enable gradient checkpointing on TSRNGist blocks to save VRAM")
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
        max_gists=args.max_gists,
        gist_top_k=args.gist_top_k,
        resume_from=args.resume,
        ckpt_every=args.ckpt_every,
        tag=args.tag,
        grad_accum_steps=args.grad_accum,
        gradient_checkpoint=args.grad_ckpt,
    )


if __name__ == "__main__":
    main()
