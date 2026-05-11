"""
DPO Training Loop with TMT Tokenizer
====================================

Direct Preference Optimization training loop using TMT tokenizer.

Supports hyperbolic memory integration for preference pairs.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tropical_tokenizer import TropicalMergingTokenizer


class DPODataset(Dataset):
    """
    DPO dataset with preference pairs.

    Args:
        data_file: JSON file with preference examples
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

        # Format prompt + chosen/rejected responses
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = self.tokenizer.encode(chosen)
        rejected_ids = self.tokenizer.encode(rejected)

        # Truncate
        prompt_ids = prompt_ids[: self.max_length // 2]
        chosen_ids = chosen_ids[: self.max_length // 2]
        rejected_ids = rejected_ids[: self.max_length // 2]

        # Combine prompt + response
        chosen_full = prompt_ids + chosen_ids
        rejected_full = prompt_ids + rejected_ids

        # Pad
        def pad(ids):
            if len(ids) > self.max_length:
                return ids[: self.max_length]
            return ids + [self.tokenizer.vocab_to_id["<pad>"]] * (self.max_length - len(ids))

        chosen_full = pad(chosen_full)
        rejected_full = pad(rejected_full)

        return {
            "chosen_input_ids": torch.tensor(chosen_full, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_full, dtype=torch.long),
            "prompt_ids": torch.tensor(pad(prompt_ids), dtype=torch.long),
        }


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO loss.

    Args:
        policy_chosen_logps: Policy log probs for chosen
        policy_rejected_logps: Policy log probs for rejected
        reference_chosen_logps: Reference log probs for chosen
        reference_rejected_logps: Reference log probs for rejected
        beta: DPO temperature

    Returns:
        DPO loss
    """
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    losses = -F.logsigmoid(beta * (policy_logratios - reference_logratios))

    return losses.mean()


def train_dpo(
    model: nn.Module,
    reference_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
    beta: float = 0.1,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """
    DPO training loop.

    Args:
        model: Policy model
        reference_model: Reference model (frozen)
        train_loader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs
        beta: DPO temperature
        grad_accum_steps: Gradient accumulation steps

    Returns:
        Training metrics
    """
    model.train()
    reference_model.eval()

    total_loss = 0.0
    total_steps = 0

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)

            # Get log probs from policy
            policy_chosen_logps = model(chosen_ids).log_softmax(dim=-1).sum(dim=-1)
            policy_rejected_logps = model(rejected_ids).log_softmax(dim=-1).sum(dim=-1)

            # Get log probs from reference (no grad)
            with torch.no_grad():
                reference_chosen_logps = reference_model(chosen_ids).log_softmax(dim=-1).sum(dim=-1)
                reference_rejected_logps = reference_model(rejected_ids).log_softmax(dim=-1).sum(dim=-1)

            # DPO loss
            loss = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta,
            )
            loss = loss / grad_accum_steps

            # Backward
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            total_steps += 1

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item() * grad_accum_steps:.4f}")

    return {"train_loss": total_loss / total_steps}


def main():
    parser = argparse.ArgumentParser(description="DPO training with TMT")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to TMT tokenizer")
    parser.add_argument("--train-data", type=str, required=True, help="Training data JSON")
    parser.add_argument("--model", type=str, required=True, help="Policy model checkpoint")
    parser.add_argument("--reference-model", type=str, required=True, help="Reference model checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/dpo_model.pt", help="Output checkpoint")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")

    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = TropicalMergingTokenizer.load(args.tokenizer)
    print(f"Vocab size: {tokenizer.current_vocab_size}")

    print(f"Loading dataset")
    train_dataset = DPODataset(args.train_data, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Loading models")
    # TODO: Load actual TSRN models
    # model = load_model(args.model, vocab_size=tokenizer.current_vocab_size)
    # reference_model = load_model(args.reference_model, vocab_size=tokenizer.current_vocab_size)
    model = None  # Placeholder
    reference_model = None  # Placeholder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model:
        model = model.to(device)
    if reference_model:
        reference_model = reference_model.to(device)

    optimizer = torch.optim.AdamW(model.parameters() if model else [], lr=args.lr)

    print("Starting DPO training")
    metrics = train_dpo(
        model,
        reference_model,
        train_loader,
        optimizer,
        device,
        epochs=args.epochs,
        beta=args.beta,
        grad_accum_steps=args.grad_accum,
    )

    print(f"Training complete: {metrics}")

    if model:
        torch.save(model.state_dict(), args.output)
        print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
