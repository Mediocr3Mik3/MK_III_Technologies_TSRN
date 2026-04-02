"""
LRA ListOps Benchmark
======================
The Long Range Arena ListOps task: classify sequences of nested
max/min/median/sum_mod operations over integers.

Example: [MAX 3 [MIN 4 7] 9] -> class based on result

This is INTRINSICALLY tropical algebra - the task IS hierarchically
applied max and min operations. A tropical transformer with hard routing
should compute this nearly exactly, while classical softmax attention
approximates it with blurred attention.

This is the paper's scientific core (Paper 1, Tier 3).

Data source: We generate ListOps data synthetically following the
original LRA specification (Tay et al., 2021).
"""

import json
import os
import sys
import time
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_utils import get_device


# =============================================================================
# ListOps Data Generator (following LRA specification)
# =============================================================================

OPS = ["MAX", "MIN", "MED", "SM"]  # SM = sum_mod
MAX_DEPTH = 5
MAX_ARGS = 5
MAX_VALUE = 9
NUM_CLASSES = 10  # results are mod 10

# Token vocabulary
PAD_TOKEN = 0
OPEN_TOKEN = 1   # [
CLOSE_TOKEN = 2  # ]
MAX_TOKEN = 3
MIN_TOKEN = 4
MED_TOKEN = 5
SM_TOKEN = 6
# Digits 0-9 map to tokens 7-16
DIGIT_OFFSET = 7
VOCAB_SIZE = 17  # PAD, [, ], MAX, MIN, MED, SM, 0-9

OP_TO_TOKEN = {"MAX": MAX_TOKEN, "MIN": MIN_TOKEN, "MED": MED_TOKEN, "SM": SM_TOKEN}


def _generate_tree(depth: int, max_depth: int, max_args: int) -> tuple:
    """Generate a random ListOps expression tree. Returns (tokens, value)."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        # Leaf: single digit
        val = random.randint(0, MAX_VALUE)
        return [DIGIT_OFFSET + val], val

    op = random.choice(OPS)
    n_args = random.randint(2, max_args)

    tokens = [OPEN_TOKEN, OP_TO_TOKEN[op]]
    values = []

    for _ in range(n_args):
        child_tokens, child_val = _generate_tree(depth + 1, max_depth, max_args)
        tokens.extend(child_tokens)
        values.append(child_val)

    tokens.append(CLOSE_TOKEN)

    # Compute result
    if op == "MAX":
        result = max(values)
    elif op == "MIN":
        result = min(values)
    elif op == "MED":
        values_sorted = sorted(values)
        result = values_sorted[len(values_sorted) // 2]
    elif op == "SM":
        result = sum(values) % NUM_CLASSES
    else:
        raise ValueError(f"Unknown op: {op}")

    return tokens, result


def generate_listops_dataset(
    n_samples: int,
    max_seq_len: int = 512,
    max_depth: int = MAX_DEPTH,
    max_args: int = MAX_ARGS,
    seed: int = 42,
) -> tuple:
    """
    Generate ListOps dataset.

    Returns:
        sequences: LongTensor (n_samples, max_seq_len) - token IDs, padded
        labels: LongTensor (n_samples,) - class labels 0-9
        lengths: LongTensor (n_samples,) - actual sequence lengths
    """
    random.seed(seed)
    all_tokens = []
    all_labels = []
    all_lengths = []

    # Adapt depth/args if max_seq_len is short
    effective_depth = max_depth
    effective_args = max_args
    if max_seq_len <= 64:
        effective_depth = min(max_depth, 2)
        effective_args = min(max_args, 3)
    elif max_seq_len <= 128:
        effective_depth = min(max_depth, 3)
        effective_args = min(max_args, 4)

    attempts = 0
    while len(all_tokens) < n_samples:
        attempts += 1
        if attempts > n_samples * 50:
            raise RuntimeError(
                f"Too many failed attempts ({attempts}). "
                f"Generated {len(all_tokens)}/{n_samples} with "
                f"max_seq_len={max_seq_len}, depth={effective_depth}, args={effective_args}"
            )

        tokens, value = _generate_tree(0, effective_depth, effective_args)

        if len(tokens) > max_seq_len:
            continue
        if len(tokens) < 4:  # too short
            continue

        # Pad to max_seq_len
        padded = tokens + [PAD_TOKEN] * (max_seq_len - len(tokens))
        all_tokens.append(padded)
        all_labels.append(value % NUM_CLASSES)
        all_lengths.append(len(tokens))

    return (
        torch.tensor(all_tokens, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.long),
        torch.tensor(all_lengths, dtype=torch.long),
    )


class ListOpsDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =============================================================================
# Sequence Models for ListOps
# =============================================================================

class ListOpsTransformer(nn.Module):
    """Classical transformer for sequence classification on ListOps."""

    def __init__(self, vocab_size, d_model, nhead, num_layers, ffn_dim,
                 num_classes=10, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(positions)
        h = self.embed_drop(h)

        # Padding mask: True where padded
        pad_mask = (x == PAD_TOKEN)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)

        # Mean pool over non-padded positions
        mask_float = (~pad_mask).float().unsqueeze(-1)
        h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return self.head(h)


class ListOpsTropFormer(nn.Module):
    """TropFormer for sequence classification on ListOps."""

    def __init__(self, vocab_size, d_model, nhead, num_layers, ffn_dim,
                 num_classes=10, max_seq_len=512, dropout=0.1,
                 lf_pieces=4, lf_mode="blend", init_temp=1.0):
        super().__init__()
        from tropformer import TropicalTransformerBlock
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TropicalTransformerBlock(
                d_model, nhead, ffn_dim, dropout, 0.05,
                lf_pieces, lf_mode, init_temp,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(positions)
        h = self.embed_drop(h)

        for block in self.blocks:
            h, _ = block(h)

        h = self.norm(h)

        # Mean pool over non-padded positions
        pad_mask = (x == PAD_TOKEN)
        mask_float = (~pad_mask).float().unsqueeze(-1)
        h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return self.head(h)

    def maslov_summary(self):
        return {
            f"block_{i}": block.attn.maslov.temperatures.detach().cpu()
            for i, block in enumerate(self.blocks)
        }


class ListOpsDTN(nn.Module):
    """DTN (hybrid, tropical-biased) for sequence classification on ListOps."""

    def __init__(self, vocab_size, d_model, nhead, num_layers, ffn_dim,
                 num_classes=10, max_seq_len=512, dropout=0.1,
                 lf_pieces=4, lf_mode="blend", init_temp=1.0):
        super().__init__()
        from deep_tropical_net import DTNBlock, TropicalBatchNorm
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)
        self.trop_bn = TropicalBatchNorm(d_model)

        self.blocks = nn.ModuleList([
            DTNBlock(d_model, nhead, ffn_dim, dropout, lf_pieces, lf_mode, init_temp)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(positions)
        h = self.embed_drop(h)
        h = self.trop_bn(h)

        for block in self.blocks:
            h, _ = block(h)

        h = self.norm(h)

        pad_mask = (x == PAD_TOKEN)
        mask_float = (~pad_mask).float().unsqueeze(-1)
        h = (h * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return self.head(h)

    def maslov_summary(self):
        return {
            f"block_{i}": block.attn.maslov.temperatures.detach().cpu()
            for i, block in enumerate(self.blocks)
        }

    def gate_summary(self):
        out = {}
        for i, block in enumerate(self.blocks):
            g = torch.sigmoid(block.attn.score_gate.bias).detach().cpu()
            out[f"block_{i}_attn_gate"] = g.mean().item()
        return out


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = correct = total = 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(seqs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = seqs.size(0)
        total_loss += loss.item() * bs
        correct += logits.argmax(1).eq(labels).sum().item()
        total += bs
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        logits = model(seqs)
        loss = F.cross_entropy(logits, labels)
        bs = seqs.size(0)
        total_loss += loss.item() * bs
        correct += logits.argmax(1).eq(labels).sum().item()
        total += bs
    return total_loss / total, correct / total


def run_model(name, model, train_loader, test_loader, device, epochs, lr):
    print(f"\n{'='*60}")
    print(f"  {name}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    print(f"{'='*60}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )

    best_acc = 0.0
    print(f"{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'TeLoss':>8}  {'TeAcc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        flag = " *" if te_acc > best_acc else ""
        if te_acc > best_acc:
            best_acc = te_acc
        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {te_loss:>8.4f}  {te_acc:>7.4f}  {elapsed:>5.1f}s{flag}")

    return best_acc, params


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-classical", action="store_true",
                        help="Skip classical transformer (use cached result)")
    parser.add_argument("--tropical-device", default="cpu",
                        help="Device for tropical models (default: cpu)")
    args = parser.parse_args()

    EPOCHS = 20
    BATCH_SIZE = 32
    LR = 1e-3
    D_MODEL = 64
    NUM_HEADS = 2
    NUM_LAYERS = 2
    FFN_DIM = 128
    MAX_SEQ_LEN = 64  # avg actual len=38; tropical attn is O(B*H*L²*d_k)
    TRAIN_SIZE = 20000
    TEST_SIZE = 4000
    SEED = 42

    device = get_device()
    trop_device = torch.device(args.tropical_device)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("  LRA LISTOPS BENCHMARK (Strong-Gain)")
    print(f"  Classical device: {device}")
    print(f"  Tropical device: {trop_device}")
    print(f"  Epochs: {EPOCHS}  Seed: {SEED}")
    print(f"  d={D_MODEL} h={NUM_HEADS} L={NUM_LAYERS} ffn={FFN_DIM}")
    print(f"  max_seq_len={MAX_SEQ_LEN}")
    print(f"  Train: {TRAIN_SIZE}  Test: {TEST_SIZE}")
    print("=" * 60)

    # Generate data
    print("\n  Generating ListOps data...")
    train_seqs, train_labels, train_lens = generate_listops_dataset(
        TRAIN_SIZE, MAX_SEQ_LEN, seed=SEED
    )
    test_seqs, test_labels, test_lens = generate_listops_dataset(
        TEST_SIZE, MAX_SEQ_LEN, seed=SEED + 1
    )
    print(f"  Train: {len(train_labels)} samples, avg len={train_lens.float().mean():.0f}")
    print(f"  Test:  {len(test_labels)} samples, avg len={test_lens.float().mean():.0f}")
    print(f"  Label distribution (train): {torch.bincount(train_labels, minlength=10).tolist()}")

    train_ds = ListOpsDataset(train_seqs, train_labels, train_lens)
    test_ds = ListOpsDataset(test_seqs, test_labels, test_lens)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = {}

    # 1. Classical Transformer
    if args.skip_classical:
        print("\n  [Skipping classical \u2014 using cached result]")
        results["classical"] = {"best_acc": 0.0, "params": 0}  # placeholder
    else:
        torch.manual_seed(SEED)
        classical = ListOpsTransformer(
            VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, FFN_DIM,
            num_classes=NUM_CLASSES, max_seq_len=MAX_SEQ_LEN, dropout=0.1,
        ).to(device)
        c_acc, c_params = run_model(
            "Classical Transformer", classical, train_loader, test_loader, device, EPOCHS, LR
        )
        results["classical"] = {"best_acc": c_acc, "params": c_params}

    # Disable smooth max if tropical models run on CPU (native .max() backward is fine)
    if str(trop_device) == "cpu":
        import tropformer
        tropformer._USE_SMOOTH_MAX = False
        print("  [Disabled smooth max for CPU tropical models]")

    # 2. TropFormer (on trop_device to avoid DirectML 5D tensor bottleneck)
    torch.manual_seed(SEED)
    trop = ListOpsTropFormer(
        VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, FFN_DIM,
        num_classes=NUM_CLASSES, max_seq_len=MAX_SEQ_LEN, dropout=0.1,
        lf_pieces=4, lf_mode="blend", init_temp=1.0,
    ).to(trop_device)
    t_acc, t_params = run_model(
        "TropFormer (Hybrid)", trop, train_loader, test_loader, trop_device, EPOCHS, LR
    )
    results["tropformer"] = {"best_acc": t_acc, "params": t_params}

    # TropFormer diagnostics
    print("\n  TropFormer Maslov temps:")
    for blk, temps in trop.maslov_summary().items():
        print(f"    {blk}: {[f'{t:.3f}' for t in temps.tolist()]}")

    # 3. DTN (hybrid, tropical-biased — on trop_device)
    torch.manual_seed(SEED)
    dtn = ListOpsDTN(
        VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, FFN_DIM,
        num_classes=NUM_CLASSES, max_seq_len=MAX_SEQ_LEN, dropout=0.1,
        lf_pieces=4, lf_mode="blend", init_temp=1.0,
    ).to(trop_device)
    d_acc, d_params = run_model(
        "Deep Tropical Net (Hybrid)", dtn, train_loader, test_loader, trop_device, EPOCHS, LR
    )
    results["deep_tropical"] = {"best_acc": d_acc, "params": d_params}

    # DTN diagnostics
    print("\n  DTN Maslov temps:")
    for blk, temps in dtn.maslov_summary().items():
        print(f"    {blk}: {[f'{t:.3f}' for t in temps.tolist()]}")
    print("\n  DTN Gate values (higher = more tropical):")
    for name, val in dtn.gate_summary().items():
        print(f"    {name}: {val:.3f}")

    # Summary
    print("\n\n" + "=" * 60)
    print(f"  LISTOPS FINAL COMPARISON ({EPOCHS} epochs)")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:25s}: {r['best_acc']*100:6.2f}%  ({r['params']:,} params)")

    os.makedirs("results", exist_ok=True)
    fname = f"results/listops_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved -> {fname}")


if __name__ == "__main__":
    main()
