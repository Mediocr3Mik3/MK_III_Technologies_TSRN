"""
TropFormer vs Classical Transformer Benchmark
==============================================
Runs both models with identical settings and compares results.
Reproducible with fixed seeds for scientific validation.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime

# Import both models
from tropformer import TropFormer, get_mnist_loaders as get_loaders_trop
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


def run_experiment(model, model_name, train_loader, test_loader, device, epochs, lr):
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )
    
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "epoch_time": []}
    best_acc = 0.0
    
    print(f"\n{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'TeLoss':>8}  {'TeAcc':>7}  {'Time':>6}")
    print("-" * 55)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["epoch_time"].append(elapsed)
        
        flag = " *" if te_acc > best_acc else ""
        if te_acc > best_acc:
            best_acc = te_acc
        
        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {te_loss:>8.4f}  {te_acc:>7.4f}  {elapsed:>5.1f}s{flag}")
    
    history["best_acc"] = best_acc
    history["total_time"] = sum(history["epoch_time"])
    
    return history, model


def main():
    # Configuration
    EPOCHS = 10  # Reduced for faster validation; increase for full run
    BATCH_SIZE = 128
    LR = 3e-3
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    FFN_DIM = 256
    DROPOUT = 0.1
    SEEDS = [42, 123, 456]  # Multiple seeds for reproducibility
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("  TropFormer vs Classical Transformer Benchmark")
    print("  Rigorous comparison with multiple seeds for reproducibility")
    print("=" * 70)
    print(f"  Device      : {device}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Seeds       : {SEEDS}")
    print(f"  Architecture: d_model={D_MODEL}, heads={NUM_HEADS}, layers={NUM_LAYERS}")
    
    # Results storage
    results = {
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ffn_dim": FFN_DIM,
            "device": device,
            "timestamp": datetime.now().isoformat(),
        },
        "tropformer": {"runs": [], "best_accs": []},
        "classical": {"runs": [], "best_accs": []},
    }
    
    for seed in SEEDS:
        print(f"\n\n{'#'*70}")
        print(f"  SEED: {seed}")
        print(f"{'#'*70}")
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
        
        # Data loaders (same for both models)
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
        test_ds = datasets.MNIST("./data", train=False, download=True, transform=tf)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # TropFormer
        torch.manual_seed(seed)
        trop_model = TropFormer(
            img_size=28, patch_size=7, in_channels=1, num_classes=10,
            d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            ffn_dim=FFN_DIM, dropout=DROPOUT, trop_dropout=0.05,
            lf_pieces=8, lf_mode="blend", init_temp=1.0,
        ).to(device)
        
        trop_history, trop_model = run_experiment(
            trop_model, "TropFormer (Tropical Geometry)", 
            train_loader, test_loader, device, EPOCHS, LR
        )
        results["tropformer"]["runs"].append(trop_history)
        results["tropformer"]["best_accs"].append(trop_history["best_acc"])
        
        # Print TropFormer diagnostics
        print("\n  TropFormer Diagnostics:")
        print("  Maslov temperatures (tau):")
        for blk, temps in trop_model.maslov_summary().items():
            temps_str = ", ".join([f"h{i}={t:.2f}" for i, t in enumerate(temps.tolist())])
            print(f"    {blk}: {temps_str}")
        
        # Classical Transformer
        torch.manual_seed(seed)
        class_model = ClassicalTransformer(
            img_size=28, patch_size=7, in_channels=1, num_classes=10,
            d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            ffn_dim=FFN_DIM, dropout=DROPOUT,
        ).to(device)
        
        class_history, _ = run_experiment(
            class_model, "Classical Transformer (Baseline)",
            train_loader, test_loader, device, EPOCHS, LR
        )
        results["classical"]["runs"].append(class_history)
        results["classical"]["best_accs"].append(class_history["best_acc"])
    
    # Summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    trop_accs = results["tropformer"]["best_accs"]
    class_accs = results["classical"]["best_accs"]
    
    trop_mean = sum(trop_accs) / len(trop_accs)
    class_mean = sum(class_accs) / len(class_accs)
    trop_std = (sum((x - trop_mean)**2 for x in trop_accs) / len(trop_accs)) ** 0.5
    class_std = (sum((x - class_mean)**2 for x in class_accs) / len(class_accs)) ** 0.5
    
    print(f"\n  TropFormer:")
    print(f"    Best accuracies: {[f'{a*100:.2f}%' for a in trop_accs]}")
    print(f"    Mean: {trop_mean*100:.2f}% +/- {trop_std*100:.2f}%")
    
    print(f"\n  Classical Transformer:")
    print(f"    Best accuracies: {[f'{a*100:.2f}%' for a in class_accs]}")
    print(f"    Mean: {class_mean*100:.2f}% +/- {class_std*100:.2f}%")
    
    diff = trop_mean - class_mean
    print(f"\n  Difference (TropFormer - Classical): {diff*100:+.2f}%")
    
    if diff > 0:
        print("  --> TropFormer OUTPERFORMS Classical Transformer")
    elif diff < -0.003:  # 0.3% threshold from roadmap
        print("  --> Classical outperforms; tropical layers need tuning")
    else:
        print("  --> Performance is comparable (within margin)")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_file = f"results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = main()
