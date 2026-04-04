"""Full sequential test evaluation for enwik8 — airtight BPC measurement.

Loads the best TSRN checkpoint and evaluates deterministically over the
ENTIRE 5M byte test set using non-overlapping windows (standard protocol).
"""
import sys, os, json, time, torch, math
sys.path.insert(0, os.path.dirname(__file__))

from tsrn_dml import TSRN, load_enwik8, evaluate, evaluate_sequential

def main():
    # Find best checkpoint
    ckpt_path = "checkpoints/tsrn_enwik8_v2_100k_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found. Tried:")
        print(f"  checkpoints/tsrn_enwik8_v2_100k_best.pt")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # Detect device
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Device: DirectML (AMD GPU)")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

    # Load dataset with same context length as training
    ctx = cfg["context_len"]
    print(f"\nLoading enwik8 (ctx={ctx})...")
    dataset = load_enwik8(context_len=ctx)

    # Build model
    model = TSRN(
        vocab=cfg["vocab"], d_model=cfg["d_model"],
        context_len=cfg["context_len"], n_blocks=cfg["n_blocks"],
        top_k=cfg["top_k"], n_heads=cfg["n_heads"],
        mem_depth=cfg["mem_depth"], dropout=0.1  # Match training dropout
    )

    # Verify what's actually in the checkpoint
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Checkpoint path stored: {ckpt_path}")
    print(f"Best val BPC stored in ckpt: {ckpt.get('best_val_bpc', 'NOT FOUND')}")
    print(f"Config stored: {ckpt.get('config', 'NOT FOUND')}")

    # Load with strict checking
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=True
    )
    if missing:
        print(f"MISSING KEYS: {missing}")
    if unexpected:
        print(f"UNEXPECTED KEYS: {unexpected}")
    assert not missing and not unexpected, "State dict mismatch — weights did not load correctly"
    
    # CRITICAL FIX: Force eval mode on ALL submodules
    model.eval()
    for module in model.modules():
        module.eval()
    
    model.to(device)
    print(f"Model: TSRN ({model.count_params():,} params)")
    print(f"Best val BPC from training: {ckpt.get('best_val_bpc', '?')}")

    # Verify eval mode worked
    all_eval = all(not m.training for m in model.modules())
    print(f"All modules in eval mode: {all_eval}")

    # Sanity check: one batch should give ~0.79 BPC
    x, y = dataset.batch("test", 8, device)
    _, loss = model(x, y)
    sanity_bpc = loss.item() / math.log(2)
    print(f"Sanity check BPC: {sanity_bpc:.4f}  (expected ~0.79)")
    if sanity_bpc > 1.5:
        print(f"WARNING: Sanity check failed - model giving {sanity_bpc:.4f} BPC instead of ~0.79")
        print("This indicates a model architecture or evaluation code issue")
        print("Continuing with evaluation to document the discrepancy...")

    # --- Random sampling evaluation (for comparison) ---
    print(f"\n{'='*70}")
    print(f"  Random Sampling Evaluation (200 batches x 16, ~16% of test set)")
    print(f"{'='*70}")
    t0 = time.time()
    rand_loss, rand_ppl, rand_bpc = evaluate(
        model, dataset, device, n_batches=200, batch_size=8, split="test")
    t_rand = time.time() - t0
    print(f"  Test BPC (random):     {rand_bpc:.4f}  (PPL: {rand_ppl:.3f})")
    print(f"  Time: {t_rand:.1f}s")

    # --- Full sequential evaluation ---
    print(f"\n{'='*70}")
    print(f"  Full Sequential Evaluation (entire 5M byte test set)")
    print(f"{'='*70}")
    n_windows = (len(dataset.test) - 1 - ctx) // ctx
    print(f"  Windows: {n_windows:,} non-overlapping x {ctx} bytes = "
          f"{n_windows * ctx:,} bytes / {len(dataset.test):,} total")

    t0 = time.time()
    seq_loss, seq_ppl, seq_bpc = evaluate_sequential(
        model, dataset, device, batch_size=8, split="test")
    t_seq = time.time() - t0
    print(f"  Test BPC (sequential): {seq_bpc:.4f}  (PPL: {seq_ppl:.3f})")
    print(f"  Time: {t_seq:.1f}s")

    # --- Also do val set for completeness ---
    print(f"\n  Val set (sequential)...")
    val_loss, val_ppl, val_bpc = evaluate_sequential(
        model, dataset, device, batch_size=8, split="val")
    print(f"  Val BPC (sequential):  {val_bpc:.4f}  (PPL: {val_ppl:.3f})")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Full Sequential Evaluation")
    print(f"{'='*70}")
    print(f"  Model:      TSRN ({model.count_params():,} params)")
    print(f"  Dataset:    enwik8 (byte-level, standard 90M/5M/5M)")
    print(f"  Context:    {ctx} bytes")
    print(f"  Protocol:   Non-overlapping windows, full test set coverage")
    print(f"")
    print(f"  Val BPC:    {val_bpc:.4f}  (sequential, full 5M)")
    print(f"  Test BPC:   {seq_bpc:.4f}  (sequential, full 5M)")
    print(f"  Test PPL:   {seq_ppl:.3f}")
    print(f"")
    print(f"  Comparison:")
    print(f"    Random sampling BPC:     {rand_bpc:.4f}")
    print(f"    Sequential BPC:          {seq_bpc:.4f}")
    print(f"    Difference:              {abs(seq_bpc - rand_bpc):.4f}")
    print(f"{'='*70}")

    # Save results
    results = {
        "evaluation": "full_sequential",
        "checkpoint": ckpt_path,
        "model_params": model.count_params(),
        "context_len": ctx,
        "protocol": "non-overlapping windows, full test set coverage",
        "dataset": "enwik8 byte-level (latin-1), standard 90M/5M/5M split",
        "vocab_size": cfg["vocab"],
        "test": {
            "bpc_sequential": round(seq_bpc, 4),
            "ppl_sequential": round(seq_ppl, 3),
            "bpc_random_sampling": round(rand_bpc, 4),
            "ppl_random_sampling": round(rand_ppl, 3),
            "difference_bpc": round(abs(seq_bpc - rand_bpc), 4),
        },
        "val": {
            "bpc_sequential": round(val_bpc, 4),
            "ppl_sequential": round(val_ppl, 3),
        },
        "best_val_bpc_training": ckpt.get("best_val_bpc"),
        "config": cfg,
    }
    out_path = "results/tsrn_enwik8_sequential_eval.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
