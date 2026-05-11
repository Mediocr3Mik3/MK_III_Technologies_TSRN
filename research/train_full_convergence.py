#!/usr/bin/env python3
"""
TSRN Full Convergence Training Script
======================================
Trains TSRN on tokenized shards from pretraining datasets.
Supports hyperbolic embeddings and gist chaining like tsrn_convergence_gist.py.

Usage:
    python train_full_convergence.py --steps 100000 --batch 8 --d-model 512 --context 256 --n-blocks 3 --n-heads 8 --top-k 16 --mem-depth 7 --max-gists 64 --gist-top-k 4 --use-hyperbolic --gist-chaining --tag prod --shards-dir E:\Tropformer\shards\pretrain
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).parent))

from tsrn_gist import TSRNGist
from tropical_tokenizer import TropicalMergingTokenizer


class ShardedDataset(Dataset):
    """Dataset that loads from tokenized binary shards."""
    
    def __init__(self, shards_dir: Path, split: str = "train", max_shards: Optional[int] = None):
        self.shards_dir = Path(shards_dir) / split
        self.shards = sorted(self.shards_dir.glob("shard_*_input.npy"))
        
        if max_shards:
            self.shards = self.shards[:max_shards]
        
        print(f"Found {len(self.shards)} shards in {self.shards_dir}")
        
        # Load all shard metadata
        self.metadata = []
        self.cumulative_sizes = [0]
        total_sequences = 0
        
        for shard_path in self.shards:
            meta_path = shard_path.parent / shard_path.name.replace("_input.npy", "_meta.json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    self.metadata.append(meta)
                    total_sequences += meta["num_sequences"]
                    self.cumulative_sizes.append(total_sequences)
        
        print(f"Total sequences: {total_sequences}")
        
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        # Find which shard this index belongs to
        shard_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        offset_in_shard = idx - self.cumulative_sizes[shard_idx]
        
        shard_path = self.shards[shard_idx]
        input_path = shard_path
        label_path = shard_path.parent / shard_path.name.replace("_input.npy", "_label.npy")
        
        # Load shard
        input_ids = np.load(input_path)
        label_ids = np.load(label_path)
        
        # Get the specific sequence
        return torch.tensor(input_ids[offset_in_shard], dtype=torch.long), torch.tensor(label_ids[offset_in_shard], dtype=torch.long)


def collate_fn(batch):
    """Collate function with padding."""
    input_ids = [item[0] for item in batch]
    label_ids = [item[1] for item in batch]
    
    # Pad to max length in batch
    max_len = max(ids.size(0) for ids in input_ids)
    
    padded_input = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    padded_label = torch.zeros(len(label_ids), max_len, dtype=torch.long)
    
    for i, (inp, lbl) in enumerate(zip(input_ids, label_ids)):
        padded_input[i, :inp.size(0)] = inp
        padded_label[i, :lbl.size(0)] = lbl
    
    return padded_input, padded_label




def compute_bpc(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Compute bits-per-character on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, label_ids in tqdm(dataloader, desc="Validating"):
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                label_ids.view(-1),
                ignore_index=0
            )
            
            total_loss += loss.item() * label_ids.numel()
            total_tokens += (label_ids != 0).sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    bpc = avg_loss / np.log(2)
    return bpc


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, bpc: float, checkpoint_dir: Path, tag: str):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{tag}_step_{step}.pt"
    
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "bpc": bpc,
    }, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def train_convergence(
    steps: int,
    batch_size: int,
    context_len: int,
    d_model: int,
    n_blocks: int,
    n_heads: int,
    top_k: int,
    mem_depth: int,
    max_gists: int,
    gist_top_k: int,
    ckpt_every: int,
    tag: str,
    shards_dir: Path,
    checkpoint_dir: Path,
    use_hyperbolic: bool = False,
    gist_chaining: bool = False,
    use_padic_pe: bool = False,
    lr: float = 1e-4,
    grad_accum: int = 1,
    resume: Optional[str] = None,
):
    """Main training loop with convergence tracking."""
    
    # Device setup
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Using DirectML device: {device}")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # Load tokenized shards
    print(f"Loading tokenized shards from {shards_dir}...")
    dataset = ShardedDataset(shards_dir, split="train")
    
    # Get vocab size from first shard metadata
    if dataset.metadata:
        vocab_size = dataset.metadata[0].get("vocab_size", 32000)
        print(f"Vocab size from shards: {vocab_size}")
    else:
        vocab_size = 32000
        print("Using default vocab size: 32000")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,  # DirectML doesn't support pin_memory
    )
    
    # Create model
    print("Creating TSRNGist model...")
    model = TSRNGist(
        vocab_size=vocab_size,
        context_len=context_len,
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=n_heads,
        top_k=top_k,
        mem_depth=mem_depth,
        max_gists=max_gists,
        gist_top_k=gist_top_k,
        use_hyperbolic=use_hyperbolic,
        gist_chaining=gist_chaining,
        use_padic_pe=use_padic_pe,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Resume from checkpoint
    start_step = 0
    best_bpc = float("inf")
    if resume:
        print(f"Resuming from {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        best_bpc = checkpoint.get("bpc", float("inf"))
        print(f"Resumed from step {start_step}")
    
    # Training loop
    print(f"\nStarting training for {steps} steps...")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    step = start_step
    optimizer.zero_grad()
    
    # Create iterator from DataLoader
    data_iter = iter(dataloader)
    
    pbar = tqdm(range(start_step, steps), desc="Training")
    for step in pbar:
        # Get batch
        try:
            input_ids, label_ids = next(data_iter)
        except StopIteration:
            # Restart from beginning if dataset exhausted
            data_iter = iter(dataloader)
            input_ids, label_ids = next(data_iter)
        
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            label_ids.view(-1),
            ignore_index=0
        )
        
        # Scale for gradient accumulation
        loss = loss / grad_accum
        loss.backward()
        
        # Update weights
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Update progress bar
        bpc = loss.item() / np.log(2)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "bpc": f"{bpc:.4f}"})
        
        # Checkpoint
        if (step + 1) % ckpt_every == 0:
            print(f"\nStep {step + 1}/{steps}")
            # Validation
            val_bpc = compute_bpc(model, dataloader, device)
            print(f"Validation BPC: {val_bpc:.4f}")
            
            if val_bpc < best_bpc:
                best_bpc = val_bpc
                print(f"New best BPC: {best_bpc:.4f}")
            
            save_checkpoint(model, optimizer, step + 1, best_bpc, checkpoint_dir, tag)
    
    print("\nTraining complete!")
    print(f"Best BPC: {best_bpc:.4f}")
    save_checkpoint(model, optimizer, steps, best_bpc, checkpoint_dir, tag)
    print(f"Final BPC: {best_bpc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train TSRN on mixed pretraining datasets")
    
    # Training arguments
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--context", type=int, default=256, help="Context length")
    parser.add_argument("--n-blocks", type=int, default=3, help="Number of blocks")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--top-k", type=int, default=16, help="Top-k for attention")
    parser.add_argument("--mem-depth", type=int, default=7, help="Memory depth")
    parser.add_argument("--max-gists", type=int, default=64, help="Max gists")
    parser.add_argument("--gist-top-k", type=int, default=4, help="Gist top-k")
    
    # TSRNGist features
    parser.add_argument("--use-hyperbolic", action="store_true", help="Use hyperbolic embeddings")
    parser.add_argument("--gist-chaining", action="store_true", help="Use gist chaining")
    parser.add_argument("--use-padic-pe", action="store_true", help="Use p-adic positional encoding")
    parser.add_argument("--log-pacs", action="store_true", help="Log PaCS")
    
    # Training options
    parser.add_argument("--ckpt-every", type=int, default=5000, help="Checkpoint every N steps")
    parser.add_argument("--tag", type=str, default="convergence", help="Run tag")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Data directory
    parser.add_argument("--shards-dir", type=str, required=True, help="Directory with tokenized shards")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory (default: shards-dir/../checkpoints)")
    
    args = parser.parse_args()
    
    shards_dir = Path(args.shards_dir)
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = shards_dir.parent / "checkpoints"
    
    print("=" * 60)
    print("TSRN Full Convergence Training")
    print("=" * 60)
    print(f"Shards directory: {shards_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch}")
    print(f"Context length: {args.context}")
    print(f"Model dimension: {args.d_model}")
    print(f"Number of blocks: {args.n_blocks}")
    print(f"Use hyperbolic: {args.use_hyperbolic}")
    print(f"Gist chaining: {args.gist_chaining}")
    
    train_convergence(
        steps=args.steps,
        batch_size=args.batch,
        context_len=args.context,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        top_k=args.top_k,
        mem_depth=args.mem_depth,
        max_gists=args.max_gists,
        gist_top_k=args.gist_top_k,
        ckpt_every=args.ckpt_every,
        tag=args.tag,
        shards_dir=shards_dir,
        checkpoint_dir=checkpoint_dir,
        use_hyperbolic=args.use_hyperbolic,
        gist_chaining=args.gist_chaining,
        use_padic_pe=args.use_padic_pe,
        lr=args.lr,
        grad_accum=args.grad_accum,
        resume=args.resume,
    )


if __name__ == "__main__":
    import numpy as np
    main()
