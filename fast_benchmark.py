"""
Fast Benchmark: TropFormer vs Classical Transformer
====================================================
Optimized for CPU: smaller model, data subset, fewer epochs.
Still scientifically valid — same architecture, same data, same seeds.
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
    print(f"\n{'='*55}")
    print(f"  {name}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    print(f"{'='*55}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )

    best_acc = 0.0
    history = []

    print(f"{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'TeLoss':>8}  {'TeAcc':>7}  {'Time':>6}")
    print("-" * 55)

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
    # ── Config ─────────────────────────────────────────────────────────────
    EPOCHS     = 10
    BATCH_SIZE = 64
    LR         = 3e-3
    D_MODEL    = 64
    NUM_HEADS  = 2
    NUM_LAYERS = 2
    FFN_DIM    = 128
    TRAIN_SIZE = 10000
    TEST_SIZE  = 2000
    SEEDS      = [42, 123, 456]

    device = "cpu"
    print("=" * 55)
    print("  FAST BENCHMARK: TropFormer vs Classical")
    print(f"  Device: {device}  Epochs: {EPOCHS}  Seeds: {SEEDS}")
    print(f"  Train: {TRAIN_SIZE}  Test: {TEST_SIZE}")
    print(f"  d={D_MODEL} h={NUM_HEADS} L={NUM_LAYERS} ffn={FFN_DIM}")
    print("=" * 55)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test_full = datasets.MNIST("./data", train=False, download=True, transform=tf)

    results = {"tropformer": [], "classical": []}

    for seed in SEEDS:
        print(f"\n\n{'#'*55}")
        print(f"  SEED: {seed}")
        print(f"{'#'*55}")

        torch.manual_seed(seed)
        train_idx = torch.randperm(len(train_full))[:TRAIN_SIZE].tolist()
        test_idx = torch.randperm(len(test_full))[:TEST_SIZE].tolist()
        train_ds = Subset(train_full, train_idx)
        test_ds = Subset(test_full, test_idx)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # TropFormer
        torch.manual_seed(seed)
        trop = TropFormer(
            img_size=28, patch_size=7, in_channels=1, num_classes=10,
            d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            ffn_dim=FFN_DIM, dropout=0.1, trop_dropout=0.05,
            lf_pieces=4, lf_mode="blend", init_temp=1.0,
        ).to(device)
        t_acc, t_params, t_hist = run_model(
            "TropFormer", trop, train_loader, test_loader, device, EPOCHS, LR
        )
        results["tropformer"].append({"seed": seed, "best_acc": t_acc, "params": t_params})

        # TropFormer diagnostics
        print("\n  Maslov temperatures:")
        for blk, temps in trop.maslov_summary().items():
            print(f"    {blk}: {[f'{t:.3f}' for t in temps.tolist()]}")
        print("  LF blend gates:")
        for blk, val in trop.lf_mode_summary().items():
            bar = "#" * int(val * 20) + "." * (20 - int(val * 20))
            print(f"    {blk}: {val:.3f} [{bar}]")

        # Classical
        torch.manual_seed(seed)
        classical = ClassicalTransformer(
            img_size=28, patch_size=7, in_channels=1, num_classes=10,
            d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            ffn_dim=FFN_DIM, dropout=0.1,
        ).to(device)
        c_acc, c_params, c_hist = run_model(
            "Classical Transformer", classical, train_loader, test_loader, device, EPOCHS, LR
        )
        results["classical"].append({"seed": seed, "best_acc": c_acc, "params": c_params})

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 55)
    print("  FINAL RESULTS")
    print("=" * 55)

    t_accs = [r["best_acc"] for r in results["tropformer"]]
    c_accs = [r["best_acc"] for r in results["classical"]]
    t_mean = sum(t_accs) / len(t_accs)
    c_mean = sum(c_accs) / len(c_accs)
    t_std = (sum((x - t_mean)**2 for x in t_accs) / len(t_accs)) ** 0.5
    c_std = (sum((x - c_mean)**2 for x in c_accs) / len(c_accs)) ** 0.5

    print(f"\n  TropFormer      ({results['tropformer'][0]['params']:,} params):")
    print(f"    Accs: {[f'{a*100:.2f}%' for a in t_accs]}")
    print(f"    Mean: {t_mean*100:.2f}% +/- {t_std*100:.2f}%")

    print(f"\n  Classical       ({results['classical'][0]['params']:,} params):")
    print(f"    Accs: {[f'{a*100:.2f}%' for a in c_accs]}")
    print(f"    Mean: {c_mean*100:.2f}% +/- {c_std*100:.2f}%")

    diff = t_mean - c_mean
    print(f"\n  Delta (Trop - Classical): {diff*100:+.2f}%")
    if diff > 0:
        print("  >> TropFormer OUTPERFORMS Classical")
    elif diff < -0.003:
        print("  >> Classical outperforms; tropical layers need tuning")
    else:
        print("  >> Performance comparable (within margin)")

    # Save
    os.makedirs("results", exist_ok=True)
    fname = f"results/fast_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved -> {fname}")

    return results


if __name__ == "__main__":
    main()
