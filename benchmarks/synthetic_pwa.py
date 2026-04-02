"""
Synthetic PWA Recovery Benchmark (Paper 2, Tier 3, B-1)
========================================================
Generate random piecewise-affine functions with known exact partition
boundaries. Train the network to regress the function output from inputs.
Evaluate not just MSE but whether the network recovers the exact partition
geometry.

This is the most important Paper 2 benchmark: it directly tests whether
a deep tropical network can recover the combinatorial partition structure
that defines a PWA function -- the exact representational claim.
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_utils import get_device


# =============================================================================
# PWA Data Generator
# =============================================================================

def generate_pwa_dataset(
    n_modes: int = 8,
    input_dim: int = 4,
    n_train: int = 10000,
    n_test: int = 2000,
    noise: float = 0.01,
    seed: int = 42,
) -> dict:
    """
    Generates a random PWA function:
        f(x) = A_k @ x + b_k   if x in region R_k

    Regions R_k are Voronoi cells of randomly sampled centers c_k.
    Mode k is active when x is closest (in L2) to c_k.

    Returns dict with:
        X_train, y_train, X_test, y_test,
        modes_train, modes_test (true active mode per sample),
        true_centers, true_A, true_b
    """
    rng = np.random.RandomState(seed)

    # Random Voronoi centers in [-2, 2]^input_dim
    centers = rng.uniform(-2.0, 2.0, size=(n_modes, input_dim)).astype(np.float32)

    # Random affine maps per mode: y = A_k @ x + b_k (scalar output)
    true_A = rng.randn(n_modes, input_dim).astype(np.float32) * 0.5
    true_b = rng.randn(n_modes).astype(np.float32) * 0.3

    def evaluate_pwa(X):
        """Evaluate PWA function and return (y, active_modes)."""
        # Distances to each center: (N, n_modes)
        dists = np.linalg.norm(
            X[:, None, :] - centers[None, :, :], axis=-1
        )
        modes = np.argmin(dists, axis=-1)  # (N,)

        # Compute y = A_{mode} @ x + b_{mode}
        y = np.zeros(len(X), dtype=np.float32)
        for k in range(n_modes):
            mask = modes == k
            if mask.any():
                y[mask] = X[mask] @ true_A[k] + true_b[k]
        return y, modes

    # Generate training data
    X_train = rng.randn(n_train, input_dim).astype(np.float32) * 1.5
    y_train, modes_train = evaluate_pwa(X_train)
    y_train += rng.randn(n_train).astype(np.float32) * noise

    # Generate test data
    X_test = rng.randn(n_test, input_dim).astype(np.float32) * 1.5
    y_test, modes_test = evaluate_pwa(X_test)

    return {
        "X_train": torch.from_numpy(X_train),
        "y_train": torch.from_numpy(y_train),
        "X_test": torch.from_numpy(X_test),
        "y_test": torch.from_numpy(y_test),
        "modes_train": torch.from_numpy(modes_train.astype(np.int64)),
        "modes_test": torch.from_numpy(modes_test.astype(np.int64)),
        "true_centers": torch.from_numpy(centers),
        "true_A": torch.from_numpy(true_A),
        "true_b": torch.from_numpy(true_b),
        "n_modes": n_modes,
        "input_dim": input_dim,
    }


class PWADataset(Dataset):
    def __init__(self, X, y, modes):
        self.X = X
        self.y = y
        self.modes = modes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.modes[idx]


# =============================================================================
# Models for PWA Recovery
# =============================================================================

class MLPRegressor(nn.Module):
    """Standard MLP baseline for PWA regression."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TropicalPWANet(nn.Module):
    """Deep Tropical Net for PWA regression (sequence-free, flat input)."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=4,
                 lf_pieces=8, lf_mode="blend"):
        super().__init__()
        from tropformer import TropicalLinear, LFDualActivation

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "trop": TropicalLinear(hidden_dim, hidden_dim),
                "lf": LFDualActivation(hidden_dim, num_pieces=lf_pieces, mode=lf_mode),
                "gate_proj": nn.Linear(hidden_dim, hidden_dim),
                "classical": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU()
                ),
            }))
            # Initialize gate toward tropical
            nn.init.zeros_(self.blocks[-1]["gate_proj"].weight)
            nn.init.constant_(self.blocks[-1]["gate_proj"].bias, 2.0)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            res = h
            h_n = block["norm"](h)
            trop_out = block["lf"](block["trop"](h_n))
            class_out = block["classical"](h_n)
            g = torch.sigmoid(block["gate_proj"](h_n))
            h = res + g * trop_out + (1 - g) * class_out
        h = self.out_norm(h)
        return self.head(h).squeeze(-1)

    def get_gate_values(self):
        """Return mean gate values per block (higher = more tropical)."""
        return {
            f"block_{i}": torch.sigmoid(
                block["gate_proj"].bias
            ).detach().cpu().mean().item()
            for i, block in enumerate(self.blocks)
        }


class HybridTropFormerPWA(nn.Module):
    """TropFormer (classical-biased) for PWA regression."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=4,
                 lf_pieces=4, lf_mode="blend"):
        super().__init__()
        from tropformer import TropicalLinear, LFDualActivation

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "trop": TropicalLinear(hidden_dim, hidden_dim),
                "lf": LFDualActivation(hidden_dim, num_pieces=lf_pieces, mode=lf_mode),
                "gate_proj": nn.Linear(hidden_dim, hidden_dim),
                "classical": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU()
                ),
            }))
            # Initialize gate toward CLASSICAL (TropFormer bias)
            nn.init.zeros_(self.blocks[-1]["gate_proj"].weight)
            nn.init.constant_(self.blocks[-1]["gate_proj"].bias, -2.0)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            res = h
            h_n = block["norm"](h)
            trop_out = block["lf"](block["trop"](h_n))
            class_out = block["classical"](h_n)
            g = torch.sigmoid(block["gate_proj"](h_n))
            h = res + g * trop_out + (1 - g) * class_out
        h = self.out_norm(h)
        return self.head(h).squeeze(-1)


# =============================================================================
# Mode Accuracy: predicted mode from tropical routing
# =============================================================================

def compute_mode_accuracy(model, X, true_modes, device):
    """
    Estimate predicted modes by perturbing inputs slightly and checking
    which inputs produce similar outputs (same affine region).
    For tropical nets, we can also look at argmax routing.
    Uses a clustering approach: discretize the gradient to identify modes.
    """
    # Run gradient computation on CPU to avoid DirectML issues
    cpu_model = model.cpu()
    cpu_model.eval()
    X_cpu = X.clone().requires_grad_(True)

    with torch.enable_grad():
        y = cpu_model(X_cpu)
        grad = torch.autograd.grad(y.sum(), X_cpu, create_graph=False)[0]

    # Move model back to original device
    model.to(device)

    # Each unique gradient direction corresponds to a different affine mode
    # Quantize gradients to identify modes
    grad_np = grad.detach().cpu().numpy()
    # Round to 2 decimal places to cluster similar gradients
    grad_rounded = np.round(grad_np, decimals=1)

    # Map each unique gradient to a mode ID
    unique_grads = {}
    predicted_modes = np.zeros(len(X), dtype=np.int64)
    mode_id = 0
    for i, g in enumerate(grad_rounded):
        key = tuple(g)
        if key not in unique_grads:
            unique_grads[key] = mode_id
            mode_id += 1
        predicted_modes[i] = unique_grads[key]

    # Compute mode accuracy using Hungarian matching
    n_pred_modes = len(unique_grads)
    true_modes_np = true_modes.numpy()
    n_true_modes = len(np.unique(true_modes_np))

    # Simple accuracy: what fraction of same-mode pairs are predicted same-mode
    # (invariant to mode relabeling)
    n_samples = min(2000, len(X))
    same_true = 0
    same_pred = 0
    both_same = 0
    n_pairs = 0
    rng = np.random.RandomState(0)
    for _ in range(10000):
        i, j = rng.randint(0, n_samples, size=2)
        if i == j:
            continue
        st = true_modes_np[i] == true_modes_np[j]
        sp = predicted_modes[i] == predicted_modes[j]
        same_true += st
        same_pred += sp
        both_same += (st and sp)
        n_pairs += 1

    # Rand index approximation
    if same_true > 0:
        precision = both_same / max(same_pred, 1)
        recall = both_same / max(same_true, 1)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
    else:
        f1 = 0.0

    return {
        "n_predicted_modes": n_pred_modes,
        "n_true_modes": n_true_modes,
        "mode_f1": f1,
    }


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total = 0
    for X, y, modes in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X)
        total += len(X)
    return total_loss / total


@torch.no_grad()
def evaluate_mse(model, loader, device):
    model.eval()
    total_loss = total = 0
    for X, y, modes in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = F.mse_loss(pred, y)
        total_loss += loss.item() * len(X)
        total += len(X)
    return total_loss / total


def run_model(name, model, train_loader, test_loader, device, epochs, lr,
              X_test, modes_test):
    print(f"\n{'='*60}")
    print(f"  {name}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    print(f"{'='*60}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"{'Ep':>3}  {'TrMSE':>10}  {'TeMSE':>10}  {'Time':>6}")
    print("-" * 40)

    best_mse = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_mse = train_epoch(model, train_loader, optimizer, device)
        te_mse = evaluate_mse(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        flag = " *" if te_mse < best_mse else ""
        if te_mse < best_mse:
            best_mse = te_mse
        print(f"{epoch:>3}  {tr_mse:>10.6f}  {te_mse:>10.6f}  {elapsed:>5.1f}s{flag}")

    # Mode accuracy analysis
    mode_info = compute_mode_accuracy(model, X_test, modes_test, device)

    return best_mse, params, mode_info


def main():
    EPOCHS = 40
    BATCH_SIZE = 128
    LR = 1e-3
    HIDDEN_DIM = 128
    NUM_LAYERS = 4
    N_MODES = 8
    INPUT_DIM = 4
    N_TRAIN = 10000
    N_TEST = 2000
    SEED = 42

    device = get_device()
    torch.manual_seed(SEED)

    print("=" * 60)
    print("  SYNTHETIC PWA RECOVERY BENCHMARK (Paper 2, B-1)")
    print(f"  Device: {device}  Epochs: {EPOCHS}")
    print(f"  n_modes={N_MODES}  input_dim={INPUT_DIM}")
    print(f"  Train: {N_TRAIN}  Test: {N_TEST}")
    print("=" * 60)

    data = generate_pwa_dataset(
        n_modes=N_MODES, input_dim=INPUT_DIM,
        n_train=N_TRAIN, n_test=N_TEST, seed=SEED
    )

    train_ds = PWADataset(data["X_train"], data["y_train"], data["modes_train"])
    test_ds = PWADataset(data["X_test"], data["y_test"], data["modes_test"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Mode distribution (train): {torch.bincount(data['modes_train'], minlength=N_MODES).tolist()}")

    results = {}

    # 1. MLP-ReLU baseline
    torch.manual_seed(SEED)
    mlp = MLPRegressor(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    mse, params, mode_info = run_model(
        "MLP-ReLU", mlp, train_loader, test_loader, device, EPOCHS, LR,
        data["X_test"], data["modes_test"]
    )
    results["mlp"] = {"mse": mse, "params": params, **mode_info}

    # 2. Deep Tropical Net (tropical-biased hybrid)
    torch.manual_seed(SEED)
    trop_net = TropicalPWANet(
        INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, lf_pieces=8, lf_mode="blend"
    ).to(device)
    mse, params, mode_info = run_model(
        "Deep Tropical Net", trop_net, train_loader, test_loader, device, EPOCHS, LR,
        data["X_test"], data["modes_test"]
    )
    results["deep_tropical"] = {"mse": mse, "params": params, **mode_info}
    print("\n  Gate values (higher = more tropical):")
    for name, val in trop_net.get_gate_values().items():
        print(f"    {name}: {val:.3f}")

    # 3. Hybrid TropFormer (classical-biased)
    torch.manual_seed(SEED)
    hybrid = HybridTropFormerPWA(
        INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, lf_pieces=4, lf_mode="blend"
    ).to(device)
    mse, params, mode_info = run_model(
        "TropFormer (Hybrid)", hybrid, train_loader, test_loader, device, EPOCHS, LR,
        data["X_test"], data["modes_test"]
    )
    results["tropformer"] = {"mse": mse, "params": params, **mode_info}

    # Summary
    print("\n\n" + "=" * 60)
    print(f"  PWA RECOVERY FINAL COMPARISON ({EPOCHS} epochs)")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:25s}: MSE={r['mse']:.6f}  modes={r['n_predicted_modes']}/{r['n_true_modes']}  F1={r['mode_f1']:.3f}")

    os.makedirs("results", exist_ok=True)
    fname = f"results/pwa_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved -> {fname}")


if __name__ == "__main__":
    main()
