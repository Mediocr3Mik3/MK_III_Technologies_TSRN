"""
Generate synthetic token shards for smoke testing.

Creates tiny .bin/.idx.npy files that TokenShardStream can read,
without needing real data or the full download pipeline.

Usage:
    python -m research.cloud.azure.data.smoke_tokens --output /mnt/blob/tokens/smoke
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_shard(
    output_dir: Path,
    shard_name: str,
    vocab_size: int,
    context_len: int,
    num_sequences: int,
) -> None:
    """
    Generate a single synthetic token shard (.bin + .idx.npy).

    Tokens are random integers in [0, vocab_size). Sequences are packed
    contiguously with no padding (same format as tokenize_shard.py).
    """
    shard_dir = output_dir / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Random tokens
    total_tokens = num_sequences * context_len
    tokens = np.random.randint(0, vocab_size, total_tokens, dtype=np.uint16)

    # Save packed tokens (.bin)
    bin_path = shard_dir / f"{shard_name}.bin"
    tokens.tofile(bin_path)
    logger.info(f"  wrote {bin_path} ({total_tokens} tokens, {bin_path.stat().st_size / 1024:.1f} KB)")

    # Save index (.idx.npy) — each sequence start offset
    offsets = np.arange(0, total_tokens, context_len, dtype=np.int64)
    idx_path = shard_dir / f"{shard_name}.idx.npy"
    np.save(idx_path, offsets)
    logger.info(f"  wrote {idx_path} ({len(offsets)} sequences)")

    # Save metadata.json for compatibility
    meta = {
        "shard": shard_name,
        "num_sequences": num_sequences,
        "context_len": context_len,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
    }
    meta_path = shard_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate synthetic token shards for smoke testing")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/blob/tokens/smoke"),
        help="Output directory for token shards",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)",
    )
    p.add_argument(
        "--context-len",
        type=int,
        default=256,
        help="Sequence length (default: 256)",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of shards to generate (default: 8)",
    )
    p.add_argument(
        "--seqs-per-shard",
        type=int,
        default=1000,
        help="Sequences per shard (default: 1000)",
    )
    args = p.parse_args(argv)

    logger.info(f"Generating {args.num_shards} smoke test shards to {args.output}")
    logger.info(f"  vocab_size={args.vocab_size}, context_len={args.context_len}")
    logger.info(f"  sequences per shard={args.seqs_per_shard}")

    args.output.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_shards):
        shard_name = f"shard_{i:04d}"
        generate_shard(
            args.output,
            shard_name,
            args.vocab_size,
            args.context_len,
            args.seqs_per_shard,
        )

    logger.info(f"Done. Total tokens: {args.num_shards * args.seqs_per_shard * args.context_len:,}")


if __name__ == "__main__":
    main()
