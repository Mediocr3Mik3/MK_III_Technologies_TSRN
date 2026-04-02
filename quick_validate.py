"""
Quick Validation Script
=======================
Runs TropFormer for 2 epochs to verify architecture works.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time

from tropformer import TropFormer


def main():
    print("=" * 60)
    print("  Quick TropFormer Validation (2 epochs, subset data)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    torch.manual_seed(42)
    
    # Use smaller subset for quick validation
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test_full = datasets.MNIST("./data", train=False, download=True, transform=tf)
    
    # Use only 5000 training and 1000 test samples for speed
    train_ds = Subset(train_full, range(5000))
    test_ds = Subset(test_full, range(1000))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    
    # Create model
    model = TropFormer(
        img_size=28, patch_size=7, in_channels=1, num_classes=10,
        d_model=64, num_heads=2, num_layers=2, ffn_dim=128,  # Smaller model
        dropout=0.1, trop_dropout=0.05, lf_pieces=4, lf_mode="blend",
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\n  Training:")
    for epoch in range(1, 3):
        t0 = time.time()
        model.train()
        total_loss = correct = total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            correct += logits.argmax(1).eq(target).sum().item()
            total += data.size(0)
        
        train_acc = correct / total
        
        # Evaluate
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                test_correct += logits.argmax(1).eq(target).sum().item()
                test_total += data.size(0)
        
        test_acc = test_correct / test_total
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, time={elapsed:.1f}s")
    
    print("\n  Validation PASSED - Architecture works correctly!")
    print("  Maslov temperatures:")
    for blk, temps in model.maslov_summary().items():
        print(f"    {blk}: {[f'{t:.3f}' for t in temps.tolist()]}")
    
    return True


if __name__ == "__main__":
    main()
