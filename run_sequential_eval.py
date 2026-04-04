"""Full sequential test evaluation for enwik8 — airtight BPC measurement.

Loads the best TSRN checkpoint and evaluates deterministically over the
ENTIRE 5M byte test set using non-overlapping windows (standard protocol).
"""
import sys, os, json, time, torch
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

    # Verify what's actually in the checkpoint
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Checkpoint path stored: {ckpt_path}")
    print(f"Best val BPC stored in ckpt: {ckpt.get('best_val_bpc', 'NOT FOUND')}")
    print(f"Config stored: {cfg}")

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
    # FIX 1: Use dataset.vocab_sz instead of the missing cfg["vocab"]
    # FIX 2: Safely fallback for n_heads using the same logic as the training script
    n_heads = cfg.get("n_heads", max(4, cfg["d_model"] // 64))
    
    model = TSRN(
        vocab=dataset.vocab_sz, 
        d_model=cfg["d_model"],
        context_len=cfg["context_len"], 
        n_blocks=cfg["n_blocks"],
        top_k=cfg["top_k"], 
        n_heads=n_heads,
        mem_depth=cfg["mem_depth"], 
        dropout=0.0  # no dropout for eval
    )
    
    # FIX 3: Strip out "module." or "_orig_mod." prefixes if the model was trained 
    # using DataParallel or torch.compile
    raw_state_dict = ckpt["model_state_dict"]
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        clean_k = k.replace("module.", "").replace("_orig_mod.", "")
        clean_state_dict[clean_k] = v
        
    # FIX 4: Use strict=False to bypass tied-weight strictness issues
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if missing:
        print(f"MISSING KEYS: {missing}")
    if unexpected:
        print(f"UNEXPECTED KEYS: {unexpected}")
    
    model.to(device)
    model.eval()

    # Sanity check: one batch should give ~0.79 BPC
    x, y = dataset.batch("test", 8, device)
    _, loss = model(x, y)
    sanity_bpc = loss.item() / 0.6931  # math.log(2)
    print(f"Sanity check BPC: {sanity_bpc:.4f}  (expected ~0.79)")
    if sanity_bpc > 1.5:
        print(f"WARNING: Sanity check failed — model weights may not be loaded correctly")
    print(f"Model: TSRN ({model.count_params():,} params)")
    print(f"Best val BPC from training: {ckpt.get('best_val_bpc', '?')}")

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
