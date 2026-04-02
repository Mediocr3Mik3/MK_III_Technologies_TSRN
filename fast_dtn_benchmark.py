"""
Fast Deep Tropical Network Benchmark
=====================================
Compare DTN vs TropFormer vs Classical on subset of MNIST.
"""

import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime

from deep_tropical_net import DeepTropNet
from tropformer import TropFormer
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


def run_model(name, model, train_loader, test_loader, device, epochs, lr, use_onecycle=True):
    print(f"\n{'='*55}")
    print(f"  {name}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    print(f"{'='*55}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if use_onecycle:
        total_steps = epochs * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps,
            pct_start=0.1, anneal_strategy="cos",
        )
    else:
        scheduler = None

    best_acc = 0.0
    print(f"{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'TeLoss':>8}  {'TeAcc':>7}  {'Time':>6}")
    print("-" * 55)

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
    EPOCHS     = 10
    BATCH_SIZE = 64
    LR         = 3e-3
    D_MODEL    = 64
    NUM_HEADS  = 2
    NUM_LAYERS = 2
    FFN_DIM    = 128
    TRAIN_SIZE = 10000
    TEST_SIZE  = 2000
    SEED       = 42

    device = "cpu"
    torch.manual_seed(SEED)

    print("=" * 55)
    print("  DEEP TROPICAL NETWORK BENCHMARK")
    print(f"  Device: {device}  Epochs: {EPOCHS}  Seed: {SEED}")
    print(f"  Train: {TRAIN_SIZE}  Test: {TEST_SIZE}")
    print(f"  d={D_MODEL} h={NUM_HEADS} L={NUM_LAYERS} ffn={FFN_DIM}")
    print("=" * 55)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test_full = datasets.MNIST("./data", train=False, download=True, transform=tf)

    torch.manual_seed(SEED)
    train_idx = torch.randperm(len(train_full))[:TRAIN_SIZE].tolist()
    test_idx = torch.randperm(len(test_full))[:TEST_SIZE].tolist()
    train_ds = Subset(train_full, train_idx)
    test_ds = Subset(test_full, test_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = {}

    # 1. Deep Tropical Network
    torch.manual_seed(SEED)
    dtn = DeepTropNet(
        img_size=28, patch_size=7, in_channels=1, num_classes=10,
        d_model=D_MODEL, num_layers=NUM_LAYERS, num_attn_layers=1,
        num_heads=NUM_HEADS, lf_pieces=4, lf_mode="blend",
        ste_temp=1.0, init_temp=1.0, dropout=0.1, trop_dropout=0.05,
    ).to(device)
    dtn_acc, dtn_params = run_model(
        "Deep Tropical Net", dtn, train_loader, test_loader, device, EPOCHS, LR
    )
    results["deep_tropical"] = {"best_acc": dtn_acc, "params": dtn_params}

    print("\n  DTN Maslov temperatures:")
    for name, temps in dtn.maslov_summary().items():
        print(f"    {name}: {[f'{t:.3f}' for t in temps.tolist()]}")

    # 2. TropFormer (hybrid)
    torch.manual_seed(SEED)
    trop = TropFormer(
        img_size=28, patch_size=7, in_channels=1, num_classes=10,
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM, dropout=0.1, trop_dropout=0.05,
        lf_pieces=4, lf_mode="blend", init_temp=1.0,
    ).to(device)
    trop_acc, trop_params = run_model(
        "TropFormer (Hybrid)", trop, train_loader, test_loader, device, EPOCHS, LR
    )
    results["tropformer"] = {"best_acc": trop_acc, "params": trop_params}

    # 3. Classical
    torch.manual_seed(SEED)
    classical = ClassicalTransformer(
        img_size=28, patch_size=7, in_channels=1, num_classes=10,
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM, dropout=0.1,
    ).to(device)
    c_acc, c_params = run_model(
        "Classical Transformer", classical, train_loader, test_loader, device, EPOCHS, LR
    )
    results["classical"] = {"best_acc": c_acc, "params": c_params}

    # Summary
    print("\n\n" + "=" * 55)
    print("  FINAL COMPARISON")
    print("=" * 55)
    for name, r in results.items():
        print(f"  {name:25s}: {r['best_acc']*100:6.2f}%  ({r['params']:,} params)")

    os.makedirs("results", exist_ok=True)
    fname = f"results/dtn_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved -> {fname}")


if __name__ == "__main__":
    main()
