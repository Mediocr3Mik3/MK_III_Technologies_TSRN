"""
Streaming Data Loader for TMT-Tokenized Data
==============================================

Loads pre-tokenized data shards with TMT tokenizer for training.

Supports structurally rich datasets: FineWeb-Edu, OpenWebMath, The Stack.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset


class StreamingDataset(IterableDataset):
    """
    Streaming dataset from pre-tokenized numpy shards.

    Args:
        data_dir: Directory containing .npy shard files
        context_len: Context window length
        shuffle: Whether to shuffle shards
        infinite: Whether to loop infinitely (for training)
    """

    def __init__(
        self,
        data_dir: str,
        context_len: int,
        shuffle: bool = True,
        infinite: bool = True,
    ):
        self.data_dir = data_dir
        self.context_len = context_len
        self.shuffle = shuffle
        self.infinite = infinite

        # Find all shard files
        self.shards = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npy")
        ])

        if not self.shards:
            raise ValueError(f"No .npy shards found in {data_dir}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            if self.shuffle:
                np.random.shuffle(self.shards)

            for shard_path in self.shards:
                # Load shard
                data = np.load(shard_path)  # (N,) uint16

                # Convert to tensor
                tokens = torch.from_numpy(data).long()

                # Yield context windows
                for i in range(0, len(tokens) - self.context_len, self.context_len):
                    x = tokens[i : i + self.context_len]
                    y = tokens[i + 1 : i + self.context_len + 1]
                    yield x, y

            if not self.infinite:
                break


def get_dataloader(
    data_dir: str,
    context_len: int,
    batch_size: int,
    shuffle: bool = True,
    infinite: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from streaming dataset.

    Args:
        data_dir: Directory containing .npy shard files
        context_len: Context window length
        batch_size: Batch size
        shuffle: Whether to shuffle shards
        infinite: Whether to loop infinitely
        num_workers: Number of worker processes

    Returns:
        DataLoader yielding (x, y) tuples
    """
    dataset = StreamingDataset(data_dir, context_len, shuffle, infinite)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


class MixedStreamingDataset(IterableDataset):
    """
    Mixed dataset from multiple sources with specified ratios.

    For TMT pretraining with structurally rich corpora:
    - FineWeb-Edu: 52%
    - OpenWebMath: 12%
    - Stack Python: 13%
    - Stack JS: 5%
    - Stack Shell: 3%
    - Dolma Books: 5%
    - Simple Wikipedia: 5%
    - Markdown Docs: 5%

    Args:
        data_dirs: Dict mapping source name to directory path
        ratios: Dict mapping source name to mixing ratio
        context_len: Context window length
        shuffle: Whether to shuffle shards
        infinite: Whether to loop infinitely
    """

    def __init__(
        self,
        data_dirs: dict[str, str],
        ratios: dict[str, float],
        context_len: int,
        shuffle: bool = True,
        infinite: bool = True,
    ):
        self.context_len = context_len
        self.shuffle = shuffle
        self.infinite = infinite

        # Create datasets for each source
        self.datasets = {
            name: StreamingDataset(dir, context_len, shuffle, infinite)
            for name, dir in data_dirs.items()
        }

        # Normalize ratios
        total = sum(ratios.values())
        self.ratios = {name: r / total for name, r in ratios.items()}

        # Create iterators
        self.iterators = {
            name: iter(dataset)
            for name, dataset in self.datasets.items()
        }

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            # Sample source according to ratios
            sources = list(self.ratios.keys())
            probs = list(self.ratios.values())
            source = np.random.choice(sources, p=probs)

            try:
                yield next(self.iterators[source])
            except StopIteration:
                # Reinitialize iterator
                self.iterators[source] = iter(self.datasets[source])
                yield next(self.iterators[source])


if __name__ == "__main__":
    # Test streaming dataset
    print("Testing StreamingDataset...")

    # Create dummy data
    import tempfile
    tmpdir = tempfile.mkdtemp()
    for i in range(3):
        data = np.random.randint(0, 1000, size=10000, dtype=np.uint16)
        np.save(os.path.join(tmpdir, f"shard_{i}.npy"), data)

    dataset = StreamingDataset(tmpdir, context_len=512, shuffle=False, infinite=False)
    count = 0
    for x, y in dataset:
        count += 1
        if count >= 5:
            break

    print(f"Yielded {count} batches")
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)

    print("All tests passed!")
