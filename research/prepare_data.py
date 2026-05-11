"""
Prepare Data with TMT Tokenizer
================================

Load TMT tokenizer and tokenize training corpus into binary format
for fast loading during training.

Usage:
    python -m research.prepare_data --tokenizer tokenizers/tmt_32k.json --corpus data/ --output data/tmt_tokenized/
"""

from __future__ import annotations

import argparse
import os
from typing import List

from tropical_tokenizer import TropicalMergingTokenizer


def tokenize_file(
    tokenizer: TropicalMergingTokenizer,
    input_path: str,
    output_path: str,
) -> None:
    """
    Tokenize a single file and save to binary format.

    Args:
        tokenizer: TMT tokenizer
        input_path: Input text file
        output_path: Output binary file
    """
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize
    token_ids = tokenizer.encode(text)

    # Save as numpy array (more efficient than JSON)
    import numpy as np
    ids_array = np.array(token_ids, dtype=np.uint16)  # Assumes vocab < 65536

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, ids_array)

    print(f"  Tokenized {input_path} -> {len(token_ids)} tokens -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize corpus with TMT")
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to TMT tokenizer JSON",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to corpus directory or file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for tokenized data",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000000,
        help="Tokens per shard (default: 1M)",
    )

    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = TropicalMergingTokenizer.load(args.tokenizer)
    print(f"Vocab size: {tokenizer.current_vocab_size}")

    os.makedirs(args.output, exist_ok=True)

    # Check if corpus is a file or directory
    if os.path.isfile(args.corpus):
        files = [args.corpus]
    else:
        # Find all text files
        files = []
        for root, dirs, filenames in os.walk(args.corpus):
            for fname in filenames:
                if fname.endswith(".txt") or fname.endswith(".json"):
                    files.append(os.path.join(root, fname))

    print(f"Found {len(files)} files to tokenize")

    for i, filepath in enumerate(files):
        output_path = os.path.join(args.output, f"shard_{i:06d}.npy")
        tokenize_file(tokenizer, filepath, output_path)

    print(f"\nTokenization complete. Output in {args.output}")


if __name__ == "__main__":
    main()
