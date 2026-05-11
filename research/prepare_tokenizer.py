"""
Prepare TMT Tokenizer
======================

Bootstrap TMT from raw bytes and save the final tokenizer.

Usage:
    python -m research.prepare_tokenizer --corpus data/enwik8 --output tokenizers/tmt_32k.json
"""

from __future__ import annotations

import argparse
import os

from tropical_tokenizer import bootstrap_from_bytes


def main():
    parser = argparse.ArgumentParser(description="Bootstrap TMT tokenizer from bytes")
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/enwik8",
        help="Path to training corpus",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizers/tmt_32k.json",
        help="Output path for tokenizer",
    )
    parser.add_argument(
        "--model-steps",
        type=int,
        default=5000,
        help="Steps to train model in Phase 0",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=14,
        help="Number of TMT rounds",
    )
    parser.add_argument(
        "--k-pairs",
        type=int,
        default=2,
        help="Pairs to merge per round",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size (default 32000)",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="+",
        default=None,
        help="Special tokens to add (e.g., --special-tokens <pad> <eos>)",
    )

    args = parser.parse_args()

    print(f"Bootstrapping TMT from {args.corpus}")
    print(f"Output: {args.output}")
    print(f"Model steps: {args.model_steps}")
    print(f"Rounds: {args.n_rounds}")
    print(f"Pairs per round: {args.k_pairs}")
    print(f"Target vocab size: {args.vocab_size}")
    if args.special_tokens:
        print(f"Special tokens: {args.special_tokens}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tokenizer = bootstrap_from_bytes(
        corpus_path=args.corpus,
        output_dir=output_dir,
        model_steps=args.model_steps,
        n_rounds=args.n_rounds,
        k_pairs=args.k_pairs,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    print(f"\nFinal vocab size: {tokenizer.current_vocab_size}")
    print(f"Total merges: {len(tokenizer.merges)}")


if __name__ == "__main__":
    main()
