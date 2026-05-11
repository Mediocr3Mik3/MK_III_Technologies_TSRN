"""
SFT Training Loop with TMT Tokenizer
=====================================

Supervised fine-tuning training loop using TMT tokenizer.

Supports synthetic phone/memory/structure tasks.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tropical_tokenizer import TropicalMergingTokenizer


class SFTDataset(Dataset):
    """
    SFT dataset with instruction-following examples.

    Args:
        data_file: JSON file with SFT examples
        tokenizer: TMT tokenizer
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        data_file: str,
        tokenizer: TropicalMergingTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]

        # Format: instruction + response
        text = f"Instruction: {example['instruction']}\nResponse: {example['response']}"

        # Tokenize
        token_ids = self.tokenizer.encode(text)

        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            token_ids = token_ids + [self.tokenizer.vocab_to_id["<pad>"]] * (
                self.max_length - len(token_ids)
            )

        # Create labels (mask instruction tokens)
        instruction_len = len(self.tokenizer.encode(f"Instruction: {example['instruction']}\n"))
        labels = [-100] * instruction_len + token_ids[instruction_len:]

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(self.max_length, dtype=torch.long),
        }


def train_sft(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 3,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """
    SFT training loop.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs
        grad_accum_steps: Gradient accumulation steps

    Returns:
        Training metrics
    """
    model.train()
    total_loss = 0.0
    total_steps = 0

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss / grad_accum_steps

            # Backward
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            total_steps += 1

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item() * grad_accum_steps:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
        model.train()

    return {"train_loss": total_loss / total_steps, "val_loss": val_loss}


def main():
    parser = argparse.ArgumentParser(description="SFT training with TMT")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to TMT tokenizer")
    parser.add_argument("--train-data", type=str, required=True, help="Training data JSON")
    parser.add_argument("--val-data", type=str, required=True, help="Validation data JSON")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint or path")
    parser.add_argument("--output", type=str, default="checkpoints/sft_model.pt", help="Output checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")

    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = TropicalMergingTokenizer.load(args.tokenizer)
    print(f"Vocab size: {tokenizer.current_vocab_size}")

    print(f"Loading datasets")
    train_dataset = SFTDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = SFTDataset(args.val_data, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loading model from {args.model}")
    # TODO: Load actual TSRN model
    # model = load_model(args.model, vocab_size=tokenizer.current_vocab_size)
    model = None  # Placeholder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model:
        model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters() if model else [], lr=args.lr)

    print("Starting SFT training")
    metrics = train_sft(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum,
    )

    print(f"Training complete: {metrics}")

    if model:
        torch.save(model.state_dict(), args.output)
        print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
