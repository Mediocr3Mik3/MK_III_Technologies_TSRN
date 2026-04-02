#!/usr/bin/env python3
"""
TSRN Interactive Inference Test
================================
Loads a trained TSRN model and provides an interactive chat-like interface
for testing text generation quality. Also runs automated quality benchmarks
including tropical-math-aware coherence tests.

Usage:
  python tsrn_inference.py --dataset wikitext103 --preset 50m
  python tsrn_inference.py --dataset enwik8 --preset 50m --steps 50000
  python tsrn_inference.py --interactive  (after training)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Import TSRN components
from tsrn_dml import (
    TSRN, VanillaTransformer, CharDatasetSplit,
    load_wikitext103, load_wikitext2, load_enwik8,
    detect_device, generate, evaluate, train_model,
    get_lr, CharDataset, generate_synthetic_data,
)

# ---------------------------------------------------------------------------
#  Enhanced generation with multiple strategies
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_stats(model, dataset, device, prompt="The ",
                        n_tokens=500, temperature=0.8, top_p=0.9,
                        top_k=40, repetition_penalty=1.1) -> Dict:
    """Generate text with detailed statistics about the generation process."""
    model.eval()
    ids = [dataset.stoi.get(c, 0) for c in prompt]
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    T = dataset.ctx

    generated_ids = []
    entropies = []
    top1_probs = []
    t0 = time.time()

    for step in range(n_tokens):
        idx_cond = ids[:, -T:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Repetition penalty
        if repetition_penalty != 1.0 and len(generated_ids) > 0:
            recent = set(generated_ids[-50:])
            for token_id in recent:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        probs = torch.softmax(logits, dim=-1)

        # Track entropy (uncertainty of prediction)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        entropies.append(entropy)
        top1_probs.append(probs.max().item())

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
            probs[indices_to_remove] = 0.0

        # Nucleus (top-p) sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()

        next_id = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]
        next_id = next_id.view(1, 1)
        ids = torch.cat([ids, next_id], dim=1)
        generated_ids.append(next_id.item())

    elapsed = time.time() - t0
    text = dataset.decode(ids[0].tolist())
    generated_text = dataset.decode(generated_ids)

    model.train()
    return {
        "full_text": text,
        "generated_text": generated_text,
        "prompt": prompt,
        "n_tokens": n_tokens,
        "chars_per_sec": n_tokens / elapsed,
        "elapsed_s": elapsed,
        "mean_entropy": sum(entropies) / len(entropies),
        "mean_top1_prob": sum(top1_probs) / len(top1_probs),
        "min_entropy": min(entropies),
        "max_entropy": max(entropies),
    }


# ---------------------------------------------------------------------------
#  Quality benchmark suite
# ---------------------------------------------------------------------------

def compute_text_metrics(text: str) -> Dict:
    """Compute various text quality metrics."""
    words = text.split()
    n_words = len(words)

    # Unique word ratio (vocabulary diversity)
    unique_words = set(w.lower() for w in words)
    unique_ratio = len(unique_words) / max(n_words, 1)

    # Average word length
    avg_word_len = sum(len(w) for w in words) / max(n_words, 1)

    # Repetition: count 3-gram repetitions
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    unique_trigrams = set(trigrams)
    trigram_repeat_ratio = 1.0 - len(unique_trigrams) / max(len(trigrams), 1)

    # Character-level: ratio of alphabetic chars
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)

    # Sentence structure: count periods and newlines
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_len = n_words / max(sentence_count, 1)

    # Detect obvious degeneration
    # Check for long runs of the same character
    max_run = 1
    current_run = 1
    for i in range(1, len(text)):
        if text[i] == text[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    return {
        "n_chars": len(text),
        "n_words": n_words,
        "unique_word_ratio": round(unique_ratio, 3),
        "avg_word_length": round(avg_word_len, 2),
        "trigram_repeat_ratio": round(trigram_repeat_ratio, 4),
        "alpha_ratio": round(alpha_ratio, 3),
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_len, 1),
        "max_char_run": max_run,
        "degenerate": max_run > 10 or trigram_repeat_ratio > 0.5,
    }


def run_quality_benchmark(model, dataset, device, label="TSRN") -> Dict:
    """Run a comprehensive quality benchmark with multiple prompts."""
    prompts = [
        # Wikipedia-style factual
        "The history of mathematics ",
        "In the year 1776, ",
        "The quantum mechanical ",
        # Narrative
        "Once upon a time, there was a ",
        "The old man walked slowly through the ",
        # Technical
        "A system of linear equations can be solved by ",
        "The algorithm processes each element in ",
        # Abstract
        "The relationship between structure and function ",
    ]

    results = {"label": label, "prompts": []}
    all_metrics = []

    print(f"\n{'='*70}")
    print(f"  Quality Benchmark: {label}")
    print(f"{'='*70}")

    for prompt in prompts:
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  {'-'*60}")

        gen = generate_with_stats(
            model, dataset, device, prompt=prompt,
            n_tokens=300, temperature=0.8, top_p=0.9,
            top_k=40, repetition_penalty=1.1
        )

        metrics = compute_text_metrics(gen["generated_text"])
        all_metrics.append(metrics)

        # Print generated text (first 200 chars)
        display = gen["full_text"][:250].replace('\n', ' ')
        print(f"  Output: {display.encode('ascii', errors='replace').decode('ascii')}")
        print(f"  Stats: {metrics['n_words']} words, "
              f"unique={metrics['unique_word_ratio']:.2f}, "
              f"entropy={gen['mean_entropy']:.2f}, "
              f"top1={gen['mean_top1_prob']:.3f}, "
              f"{gen['chars_per_sec']:.0f} char/s")

        if metrics["degenerate"]:
            print(f"  WARNING: Degenerate output detected!")

        results["prompts"].append({
            "prompt": prompt,
            "generated": gen["generated_text"][:500],
            "generation_stats": {
                "chars_per_sec": round(gen["chars_per_sec"], 1),
                "mean_entropy": round(gen["mean_entropy"], 3),
                "mean_top1_prob": round(gen["mean_top1_prob"], 4),
            },
            "text_metrics": metrics,
        })

    # Aggregate metrics
    agg = {
        "avg_unique_word_ratio": round(
            sum(m["unique_word_ratio"] for m in all_metrics) / len(all_metrics), 3),
        "avg_trigram_repeat": round(
            sum(m["trigram_repeat_ratio"] for m in all_metrics) / len(all_metrics), 4),
        "avg_word_length": round(
            sum(m["avg_word_length"] for m in all_metrics) / len(all_metrics), 2),
        "degenerate_count": sum(1 for m in all_metrics if m["degenerate"]),
        "total_prompts": len(prompts),
    }
    results["aggregate"] = agg

    print(f"\n  {'='*60}")
    print(f"  Aggregate: unique_ratio={agg['avg_unique_word_ratio']:.3f}  "
          f"repeat={agg['avg_trigram_repeat']:.4f}  "
          f"degen={agg['degenerate_count']}/{agg['total_prompts']}")

    return results


# ---------------------------------------------------------------------------
#  Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode(model, dataset, device):
    """Interactive chat-like interface for text generation."""
    print("\n" + "=" * 70)
    print("  TSRN Interactive Text Generation")
    print("  Type a prompt and press Enter. Type 'quit' to exit.")
    print("  Commands: /temp N  /topp N  /topk N  /len N  /rep N")
    print("=" * 70)

    temperature = 0.8
    top_p = 0.9
    top_k = 40
    n_tokens = 300
    rep_penalty = 1.1

    while True:
        try:
            prompt = input(f"\n[T={temperature} P={top_p} K={top_k} L={n_tokens}] > ")
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt or prompt.strip().lower() == 'quit':
            break

        # Handle commands
        if prompt.startswith('/'):
            parts = prompt.split()
            cmd = parts[0].lower()
            try:
                val = float(parts[1]) if len(parts) > 1 else None
            except ValueError:
                val = None

            if cmd == '/temp' and val is not None:
                temperature = max(0.1, min(2.0, val))
                print(f"  Temperature set to {temperature}")
            elif cmd == '/topp' and val is not None:
                top_p = max(0.1, min(1.0, val))
                print(f"  Top-p set to {top_p}")
            elif cmd == '/topk' and val is not None:
                top_k = max(1, int(val))
                print(f"  Top-k set to {top_k}")
            elif cmd == '/len' and val is not None:
                n_tokens = max(10, int(val))
                print(f"  Generation length set to {n_tokens}")
            elif cmd == '/rep' and val is not None:
                rep_penalty = max(1.0, min(2.0, val))
                print(f"  Repetition penalty set to {rep_penalty}")
            else:
                print("  Unknown command. Use /temp /topp /topk /len /rep")
            continue

        gen = generate_with_stats(
            model, dataset, device, prompt=prompt,
            n_tokens=n_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=rep_penalty
        )

        print(f"\n{gen['full_text']}")
        print(f"\n  [{gen['chars_per_sec']:.0f} char/s | "
              f"entropy={gen['mean_entropy']:.2f} | "
              f"top1={gen['mean_top1_prob']:.3f}]")


# ---------------------------------------------------------------------------
#  Train + Evaluate pipeline for enwik8 convergence
# ---------------------------------------------------------------------------

def train_to_convergence(args, cfg, dataset, device):
    """Train TSRN on enwik8 to convergence, with periodic checkpointing."""
    d = cfg["d_model"]
    ctx = cfg["context"]
    n_blocks = cfg["n_blocks"]
    n_heads = cfg.get("n_heads", max(4, d // 64))
    top_k = cfg["top_k"]
    mem_depth = cfg["mem_depth"]
    n_steps = cfg["steps"]
    batch_size = cfg["batch"]
    lr = cfg["lr"]
    n_layers_tf = cfg.get("n_layers_tf", 4)
    dropout = 0.1
    V = dataset.vocab_sz

    print(f"\n-- Training to convergence: {n_steps} steps --")

    # TSRN
    torch.manual_seed(42)
    tsrn = TSRN(vocab=V, d_model=d, context_len=ctx, n_blocks=n_blocks,
                top_k=top_k, n_heads=n_heads, mem_depth=mem_depth,
                dropout=dropout)

    # Transformer baseline
    torch.manual_seed(42)
    d_ff = d * 4
    transformer = VanillaTransformer(
        vocab=V, d_model=d, n_layers=n_layers_tf,
        n_heads=n_heads, d_ff=d_ff, context_len=ctx, dropout=dropout)

    print(f"\n  TSRN: {tsrn.count_params():,} params")
    print(f"  Transformer: {transformer.count_params():,} params")

    # Train transformer
    log_trans = train_model(
        transformer, dataset, device,
        n_steps=n_steps, batch_size=batch_size, lr_max=lr,
        lr_warmup=min(n_steps // 10, 2000), label="Vanilla Transformer",
        eval_every=max(1, n_steps // 20),
    )
    transformer.cpu()

    # Train TSRN
    log_tsrn = train_model(
        tsrn, dataset, device,
        n_steps=n_steps, batch_size=batch_size, lr_max=lr,
        lr_warmup=min(n_steps // 10, 2000), label="TSRN (full)",
        eval_every=max(1, n_steps // 20),
    )

    # Test evaluation
    print(f"\n-- Test-set evaluation --")
    transformer.to(device)
    t_loss, t_ppl, t_bpc = evaluate(transformer, dataset, device,
                                     n_batches=100, batch_size=min(batch_size, 16),
                                     split="test")
    print(f"  Transformer: PPL={t_ppl:.3f}  BPC={t_bpc:.4f}")
    transformer.cpu()

    s_loss, s_ppl, s_bpc = evaluate(tsrn, dataset, device,
                                     n_batches=100, batch_size=min(batch_size, 16),
                                     split="test")
    print(f"  TSRN:        PPL={s_ppl:.3f}  BPC={s_bpc:.4f}")

    # Save checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/tsrn_{args.dataset}_{n_steps}steps.pt"
    torch.save({
        "model_state_dict": tsrn.state_dict(),
        "config": {"vocab": V, "d_model": d, "context_len": ctx,
                   "n_blocks": n_blocks, "top_k": top_k,
                   "n_heads": n_heads, "mem_depth": mem_depth},
        "test_bpc": round(s_bpc, 4),
        "dataset": args.dataset,
        "steps": n_steps,
    }, ckpt_path)
    print(f"  Checkpoint saved -> {ckpt_path}")

    # Quality benchmark
    quality = run_quality_benchmark(tsrn, dataset, device, "TSRN")
    tsrn.cpu()

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "config": {
            "dataset": args.dataset,
            "preset": args.preset,
            "d_model": d, "context": ctx, "n_blocks": n_blocks,
            "n_steps": n_steps, "batch_size": batch_size, "lr": lr,
        },
        "transformer": {
            "params": transformer.count_params(),
            "log": log_trans,
            "test": {"loss": round(t_loss, 5), "ppl": round(t_ppl, 3), "bpc": round(t_bpc, 4)},
        },
        "tsrn": {
            "params": tsrn.count_params(),
            "log": log_tsrn,
            "test": {"loss": round(s_loss, 5), "ppl": round(s_ppl, 3), "bpc": round(s_bpc, 4)},
        },
        "quality_benchmark": quality,
    }

    out = f"results/tsrn_{args.dataset}_{n_steps}steps.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  FINAL: {args.dataset} @ {n_steps} steps")
    print(f"{'='*70}")
    print(f"  Transformer: {transformer.count_params():,} params  "
          f"Test BPC={t_bpc:.4f}  PPL={t_ppl:.3f}")
    print(f"  TSRN:        {tsrn.count_params():,} params  "
          f"Test BPC={s_bpc:.4f}  PPL={s_ppl:.3f}")
    if s_bpc < t_bpc:
        pct = (1 - s_bpc/t_bpc) * 100
        print(f"  >> TSRN wins by {pct:.1f}% BPC reduction")
    print(f"{'='*70}")

    return results


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TSRN Inference & Quality Test")
    parser.add_argument("--dataset", default="enwik8",
                        choices=["wikitext2", "wikitext103", "enwik8"],
                        help="Dataset to use")
    parser.add_argument("--preset", default="50m", choices=["2m", "50m", "quick"])
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps (default: use preset)")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive generation mode after training")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Only run quality benchmark (skip training)")
    parser.add_argument("--no-train-transformer", action="store_true",
                        help="Skip transformer training (TSRN only)")
    args = parser.parse_args()

    # Presets
    PRESETS = {
        "2m": {
            "d_model": 256, "context": 256, "n_blocks": 1,
            "n_heads": 4, "top_k": 16, "mem_depth": 6,
            "n_layers_tf": 4, "steps": 3000, "batch": 32, "lr": 3e-4,
        },
        "50m": {
            "d_model": 512, "context": 256, "n_blocks": 3,
            "n_heads": 8, "top_k": 16, "mem_depth": 7,
            "n_layers_tf": 8, "steps": 5000, "batch": 8, "lr": 2e-4,
        },
        "quick": {
            "d_model": 128, "context": 64, "n_blocks": 1,
            "n_heads": 2, "top_k": 8, "mem_depth": 5,
            "n_layers_tf": 2, "steps": 200, "batch": 16, "lr": 3e-4,
        },
    }

    cfg = dict(PRESETS[args.preset])
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.batch is not None:
        cfg["batch"] = args.batch

    device = detect_device()

    # Load dataset
    print(f"\n-- Loading dataset: {args.dataset}")
    if args.dataset == "wikitext103":
        dataset = load_wikitext103(context_len=cfg["context"])
    elif args.dataset == "wikitext2":
        dataset = load_wikitext2(context_len=cfg["context"])
    elif args.dataset == "enwik8":
        dataset = load_enwik8(context_len=cfg["context"])

    # Train and evaluate
    results = train_to_convergence(args, cfg, dataset, device)

    # Interactive mode if requested
    if args.interactive:
        ckpt_path = f"checkpoints/tsrn_{args.dataset}_{cfg['steps']}steps.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            c = ckpt["config"]
            tsrn = TSRN(vocab=c["vocab"], d_model=c["d_model"],
                         context_len=c["context_len"], n_blocks=c["n_blocks"],
                         top_k=c["top_k"], n_heads=c["n_heads"],
                         mem_depth=c["mem_depth"], dropout=0.0)
            tsrn.load_state_dict(ckpt["model_state_dict"])
            tsrn.to(device)
            print(f"  Loaded checkpoint: {ckpt_path} (BPC={ckpt.get('test_bpc','?')})")
        else:
            print(f"  No checkpoint found at {ckpt_path}, using freshly trained model")
            V = dataset.vocab_sz
            torch.manual_seed(42)
            tsrn = TSRN(vocab=V, d_model=cfg["d_model"],
                         context_len=cfg["context"], n_blocks=cfg["n_blocks"],
                         top_k=cfg["top_k"], n_heads=cfg["n_heads"],
                         mem_depth=cfg["mem_depth"], dropout=0.1)
            tsrn.to(device)
        interactive_mode(tsrn, dataset, device)


if __name__ == "__main__":
    main()
