#!/usr/bin/env python3
"""
TSRN Full Pretraining Script
==============================
Loads tokenized shards and trains TSRN on pretrain datasets.

Usage:
    python train_pretrain.py --shards-dir E:\Tropformer\shards\pretrain --checkpoint-dir E:\Tropformer\checkpoints --context 512 --d-model 512 --n-blocks 6 --batch-size 8
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).parent))

from tsrn_dml import TSRN, get_activation_fn


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


def load_shard_dataloader(
    shards_dir: Path,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 0,
    max_shards: Optional[int] = None
) -> DataLoader:
    """Create DataLoader from tokenized shards."""
    dataset = ShardedDataset(shards_dir, split, max_shards)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # DirectML doesn't support pin_memory
    )
    return dataloader


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
            
            # Compute loss (cross-entropy)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                label_ids.view(-1),
                ignore_index=0  # Ignore padding
            )
            
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    
    avg_loss = total_loss / total_tokens
    bpc = avg_loss / np.log(2)  # Convert nats to bits
    return bpc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    bpc: float,
    checkpoint_dir: Path,
    tag: str
):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{tag}_step_{step}.pt"
    
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "bpc": bpc,
    }, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum: int = 1,
    clip_grad: float = 1.0
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (input_ids, label_ids) in enumerate(pbar):
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            label_ids.view(-1),
            ignore_index=0  # Ignore padding
        )
        
        # Scale loss for gradient accumulation
        loss = loss / grad_accum
        
        # Backward pass
        loss.backward()
        
        total_loss += loss.item() * input_ids.numel() * grad_accum
        total_tokens += input_ids.numel()
        
        # Update weights
        if (batch_idx + 1) % grad_accum == 0:
            # Gradient clipping
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Update progress bar
        avg_loss = total_loss / total_tokens
        bpc = avg_loss / np.log(2)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "bpc": f"{bpc:.4f}"})
    
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser(description="Train TSRN on tokenized shards")
    
    # Data arguments
    parser.add_argument("--shards-dir", type=str, required=True, help="Directory with tokenized shards")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory for checkpoints")
    parser.add_argument("--max-shards-train", type=int, default=None, help="Max training shards to load")
    parser.add_argument("--max-shards-val", type=int, default=None, help="Max validation shards to load")
    
    # Model arguments
    parser.add_argument("--context", type=int, default=512, help="Context length")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-blocks", type=int, default=6, help="Number of blocks")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--ckpt-every", type=int, default=10000, help="Checkpoint every N steps")
    parser.add_argument("--val-every", type=int, default=5000, help="Validate every N steps")
    
    # TSRN-specific arguments
    parser.add_argument("--use-padic-pe", action="store_true", help="Use p-adic positional encoding")
    parser.add_argument("--use-tropical-ssm", action="store_true", help="Use tropical SSM")
    parser.add_argument("--use-reservoir", action="store_true", help="Use reservoir")
    parser.add_argument("--use-memory", action="store_true", help="Use hyperbolic memory")
    parser.add_argument("--use-sheaf", action="store_true", help="Use sheaf diffusion")
    parser.add_argument("--top-k", type=int, default=16, help="Top-k for attention")
    parser.add_argument("--mem-depth", type=int, default=7, help="Memory depth")
    parser.add_argument("--max-gists", type=int, default=64, help="Max gists")
    
    # Other
    parser.add_argument("--tag", type=str, default="pretrain", help="Run tag for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Device setup (DirectML)
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Using DirectML device: {device}")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TSRN Pretraining")
    print("=" * 60)
    print(f"Shards directory: {args.shards_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Context length: {args.context}")
    print(f"Model dimension: {args.d_model}")
    print(f"Number of blocks: {args.n_blocks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Training steps: {args.steps}")
    print(f"Learning rate: {args.lr}")
    
    # Load data
    print("\nLoading training data...")
    train_loader = load_shard_dataloader(
        Path(args.shards_dir),
        split="train",
        batch_size=args.batch_size,
        num_workers=0,
        max_shards=args.max_shards_train
    )
    
    print("Loading validation data...")
    val_loader = load_shard_dataloader(
        Path(args.shards_dir),
        split="val",
        batch_size=args.batch_size,
        num_workers=0,
        max_shards=args.max_shards_val
    )
    
    # Create model
    print("\nCreating TSRN model...")
    model = TSRN(
        vocab_size=args.vocab_size,
        context_len=args.context,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        use_padic_pe=args.use_padic_pe,
        use_tropical_ssm=args.use_tropical_ssm,
        use_reservoir=args.use_reservoir,
        use_memory=args.use_memory,
        use_sheaf=args.use_sheaf,
        top_k=args.top_k,
        mem_depth=args.mem_depth,
        max_gists=args.max_gists,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Resume from checkpoint if specified
    start_step = 0
    best_bpc = float("inf")
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        best_bpc = checkpoint.get("bpc", float("inf"))
        print(f"Resumed from step {start_step}")
    
    # Training loop
    print("\nStarting training...")
    step = start_step
    epoch = 0
    
    while step < args.steps:
        epoch += 1
        print(f"\nEpoch {epoch}")
        
        # Train
        avg_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_accum=args.grad_accum,
            clip_grad=args.clip_grad
        )
        
        # Estimate steps from epochs (rough approximation)
        steps_this_epoch = len(train_loader) // args.grad_accum
        step += steps_this_epoch
        
        # Validate
        if step % args.val_every == 0 or step >= args.steps:
            print(f"\nValidating at step {step}...")
            val_bpc = compute_bpc(model, val_loader, device)
            print(f"Validation BPC: {val_bpc:.4f}")
            
            if val_bpc < best_bpc:
                best_bpc = val_bpc
                print(f"New best BPC: {best_bpc:.4f}")
        
        # Checkpoint
        if step % args.ckpt_every == 0 or step >= args.steps:
            print(f"\nSaving checkpoint at step {step}...")
            save_checkpoint(
                model,
                optimizer,
                step,
                avg_loss,
                best_bpc,
                checkpoint_dir,
                args.tag
            )
        
        if step >= args.steps:
            break
    
    print("\nTraining complete!")
    print(f"Final BPC: {best_bpc:.4f}")


if __name__ == "__main__":
    main()
