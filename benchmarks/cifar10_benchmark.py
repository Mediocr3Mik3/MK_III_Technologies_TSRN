"""
CIFAR-10 Parity Benchmark
==========================
Tests TropFormer, DTN, and Classical on CIFAR-10 to demonstrate parity
on smooth image classification tasks.

CIFAR-10: 32x32 RGB images, 10 classes, 50K train / 10K test.
Patch size 8 -> 16 patches of dim 192.
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_utils import get_device
from tropformer import TropFormer
from deep_tropical_net import DeepTropNet
from classical_transformer import ClassicalTransformer


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = correct = total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = data.size(0)
        total_loss += loss.item() * bs
        correct += logits.argmax(1).eq(target).sum().item()
        total += bs
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        bs = data.size(0)
        total_loss += loss.item() * bs
        correct += logits.argmax(1).eq(target).sum().item()
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
    history = []

    print(f"{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'TeLoss':>8}  {'TeAcc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
            "te_loss": te_loss, "te_acc": te_acc, "time": elapsed
        })

        flag = " *" if te_acc > best_acc else ""
        if te_acc > best_acc:
            best_acc = te_acc

        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {te_loss:>8.4f}  {te_acc:>7.4f}  {elapsed:>5.1f}s{flag}")

    return best_acc, params, history


def main():
    EPOCHS = 30
    BATCH_SIZE = 64
    LR = 3e-3
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    FFN_DIM = 256
    TRAIN_SIZE = None  # Full dataset
    TEST_SIZE = None
    SEED = 42
    IMG_SIZE = 32
    PATCH_SIZE = 8
    IN_CHANNELS = 3

    device = get_device()
    torch.manual_seed(SEED)

    print("=" * 60)
    print("  CIFAR-10 PARITY BENCHMARK")
    print(f"  Device: {device}  Epochs: {EPOCHS}  Seed: {SEED}")
    print(f"  d={D_MODEL} h={NUM_HEADS} L={NUM_LAYERS} ffn={FFN_DIM}")
    print(f"  img={IMG_SIZE} patch={PATCH_SIZE} channels={IN_CHANNELS}")
    print("=" * 60)

    # Data augmentation for CIFAR-10
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    if TRAIN_SIZE:
        torch.manual_seed(SEED)
        train_idx = torch.randperm(len(train_ds))[:TRAIN_SIZE].tolist()
        train_ds = Subset(train_ds, train_idx)
    if TEST_SIZE:
        torch.manual_seed(SEED)
        test_idx = torch.randperm(len(test_ds))[:TEST_SIZE].tolist()
        test_ds = Subset(test_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds)}  Test: {len(test_ds)}")

    results = {}

    # 1. Classical Transformer
    torch.manual_seed(SEED)
    classical = ClassicalTransformer(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS, num_classes=10,
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM, dropout=0.1,
    ).to(device)
    c_acc, c_params, c_hist = run_model(
        "Classical Transformer", classical, train_loader, test_loader, device, EPOCHS, LR
    )
    results["classical"] = {"best_acc": c_acc, "params": c_params}

    # 2. TropFormer (hybrid, classical-biased)
    torch.manual_seed(SEED)
    trop = TropFormer(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS, num_classes=10,
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM, dropout=0.1, trop_dropout=0.05,
        lf_pieces=4, lf_mode="blend", init_temp=1.0,
    ).to(device)
    t_acc, t_params, t_hist = run_model(
        "TropFormer (Hybrid)", trop, train_loader, test_loader, device, EPOCHS, LR
    )
    results["tropformer"] = {"best_acc": t_acc, "params": t_params}

    # 3. Deep Tropical Net (hybrid, tropical-biased)
    torch.manual_seed(SEED)
    dtn = DeepTropNet(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS, num_classes=10,
        d_model=D_MODEL, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM, lf_pieces=4, lf_mode="blend",
        init_temp=1.0, dropout=0.1,
    ).to(device)
    d_acc, d_params, d_hist = run_model(
        "Deep Tropical Net (Hybrid)", dtn, train_loader, test_loader, device, EPOCHS, LR
    )
    results["deep_tropical"] = {"best_acc": d_acc, "params": d_params}

    # Summary
    print("\n\n" + "=" * 60)
    print(f"  CIFAR-10 FINAL COMPARISON ({EPOCHS} epochs)")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:25s}: {r['best_acc']*100:6.2f}%  ({r['params']:,} params)")

    # Diagnostics
    print("\n  TropFormer Maslov temps:")
    for blk, temps in trop.maslov_summary().items():
        print(f"    {blk}: {[f'{t:.3f}' for t in temps.tolist()]}")

    print("\n  DTN Maslov temps:")
    for blk, temps in dtn.maslov_summary().items():
        print(f"    {blk}: {[f'{t:.3f}' for t in temps.tolist()]}")

    print("\n  DTN Gate values (higher = more tropical):")
    for name, val in dtn.gate_summary().items():
        print(f"    {name}: {val:.3f}")

    os.makedirs("results", exist_ok=True)
    fname = f"results/cifar10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved -> {fname}")


if __name__ == "__main__":
    main()
