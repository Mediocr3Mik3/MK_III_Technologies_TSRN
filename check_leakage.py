"""
Data leakage diagnostic for TSRN WikiText-2 benchmark.
Checks whether the CharDataset's random 90/10 split causes train/val overlap
when the original HuggingFace train+val texts are concatenated.
"""
import math
from pathlib import Path
from datasets import load_dataset


def check_wikitext2_leakage():
    print("=" * 72)
    print("  DATA LEAKAGE CHECK: WikiText-2")
    print("=" * 72)

    # 1. Load official splits
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(ds["train"]["text"])
    val_text = "\n".join(ds["validation"]["text"])
    test_text = "\n".join(ds["test"]["text"])

    print(f"\n  Official split sizes:")
    print(f"    Train: {len(train_text):>12,} chars")
    print(f"    Val:   {len(val_text):>12,} chars")
    print(f"    Test:  {len(test_text):>12,} chars")

    # 2. Reproduce the bug: what tsrn_dml.py load_wikitext2() does
    combined = train_text + val_text  # lines 126-127 in tsrn_dml.py
    total = len(combined)
    split_point = int(total * 0.9)  # CharDataset val_split=0.1

    our_train = combined[:split_point]
    our_val = combined[split_point:]

    print(f"\n  tsrn_dml.py CharDataset split (combined -> 90/10):")
    print(f"    Our train: {len(our_train):>12,} chars")
    print(f"    Our val:   {len(our_val):>12,} chars")

    # 3. Check how much official val text leaked into our training split
    official_val_start = len(train_text)  # where val text begins in combined
    official_val_in_train = max(0, split_point - official_val_start)
    official_val_in_val = len(val_text) - official_val_in_train

    print(f"\n  LEAKAGE ANALYSIS:")
    print(f"    Official train text length:        {len(train_text):>12,}")
    print(f"    Our train split point:             {split_point:>12,}")
    print(f"    Official val text starts at:       {official_val_start:>12,}")

    if official_val_in_train > 0:
        pct = official_val_in_train / len(val_text) * 100
        print(f"\n    ** LEAKAGE DETECTED **")
        print(f"    Official val chars in OUR train:   {official_val_in_train:>12,} ({pct:.1f}% of val)")
    else:
        print(f"\n    No direct positional leakage (split point is before val start)")

    # 4. Check n-gram overlap between our train and our val
    # Use 50-char chunks as a proxy for content overlap
    chunk_size = 50
    train_chunks = set()
    for i in range(0, len(our_train) - chunk_size, chunk_size // 2):
        train_chunks.add(our_train[i:i + chunk_size])

    overlap_count = 0
    total_val_chunks = 0
    for i in range(0, len(our_val) - chunk_size, chunk_size // 2):
        total_val_chunks += 1
        if our_val[i:i + chunk_size] in train_chunks:
            overlap_count += 1

    if total_val_chunks > 0:
        overlap_pct = overlap_count / total_val_chunks * 100
        print(f"\n  N-GRAM OVERLAP (50-char chunks):")
        print(f"    Val chunks found in train: {overlap_count:,}/{total_val_chunks:,} ({overlap_pct:.1f}%)")

    # 5. Check official splits for overlap (should be 0)
    print(f"\n  OFFICIAL SPLIT OVERLAP CHECK:")
    official_train_chunks = set()
    for i in range(0, len(train_text) - chunk_size, chunk_size // 2):
        official_train_chunks.add(train_text[i:i + chunk_size])

    official_overlap = 0
    official_val_chunks = 0
    for i in range(0, len(val_text) - chunk_size, chunk_size // 2):
        official_val_chunks += 1
        if val_text[i:i + chunk_size] in official_train_chunks:
            official_overlap += 1

    if official_val_chunks > 0:
        pct = official_overlap / official_val_chunks * 100
        print(f"    Official val chunks in official train: {official_overlap:,}/{official_val_chunks:,} ({pct:.1f}%)")

    # 6. Demonstrate what proper BPC would look like
    print(f"\n  BPC REFERENCE VALUES:")
    print(f"    PPL 1.01  -> BPC = {math.log2(1.01):.4f}  (impossibly low)")
    print(f"    PPL 2.00  -> BPC = {math.log2(2.00):.4f}")
    print(f"    PPL 4.44  -> BPC = {math.log2(4.44):.4f}")
    print(f"    PPL 6.00  -> BPC = {math.log2(6.00):.4f}")
    print(f"    PPL 10.0  -> BPC = {math.log2(10.0):.4f}")
    print(f"    SOTA char-level LM on WT-103: ~1.0-1.2 BPC")
    print(f"    Our TSRN claim of PPL 1.01: {math.log2(1.01):.4f} BPC")
    print(f"    This is ~60x better than SOTA -- clearly data leakage.")

    print(f"\n{'='*72}")
    print(f"  CONCLUSION: The load_wikitext2() function concatenates official")
    print(f"  train + val text, then does a random 90/10 split. This causes")
    print(f"  massive data leakage. The fix: use HuggingFace's official splits.")
    print(f"{'='*72}")


if __name__ == "__main__":
    check_wikitext2_leakage()
