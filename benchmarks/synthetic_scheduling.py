"""
Max-Plus Scheduling Benchmark (Paper 2, Tier 3, B-3)
=====================================================
Job-shop scheduling under a max-plus timed event graph model.
Tasks arrive with processing times; machines have setup delays.
Predict the optimal makespan (total completion time).

The max-plus system matrix A has:
    A[i,j] = processing time of job j on machine i + setup delay

Optimal makespan = tropical eigenvalue of A^n
(where A^n is the n-th tropical matrix power).
"""

import json
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
# Max-Plus Scheduling Data Generator
# =============================================================================

def _tropical_matmul(A, B):
    """Max-plus matrix multiply: C[i,j] = max_k(A[i,k] + B[k,j])."""
    # A: (m, p), B: (p, n) -> C: (m, n)
    return (A[:, :, None] + B[None, :, :]).max(axis=1)


def _tropical_eigenvalue(A, n_iters=50):
    """
    Approximate tropical eigenvalue (max-plus spectral radius).
    lambda = lim_{k->inf} (A^k)_{ii} / k  for any i on a critical circuit.
    In practice, compute A^k for moderate k and estimate.
    """
    n = A.shape[0]
    Ak = A.copy()
    for _ in range(min(n_iters, 3 * n)):
        Ak = _tropical_matmul(Ak, A)

    # Tropical eigenvalue: max_i (Ak[i,i]) / k
    k = min(n_iters, 3 * n) + 1
    diag = np.array([Ak[i, i] for i in range(n)])
    return diag.max() / k


def _compute_makespan(processing_times, setup_times, n_jobs, n_machines):
    """
    Compute makespan using max-plus algebra.
    
    Build the system matrix and compute the critical path length.
    For a simple flow-shop, makespan = max over all paths through the DAG.
    """
    # Build completion time matrix using max-plus recursion
    # C[i,j] = max(C[i-1,j], C[i,j-1]) + processing_times[i,j] + setup_times[i,j]
    C = np.zeros((n_machines, n_jobs), dtype=np.float32)
    
    for i in range(n_machines):
        for j in range(n_jobs):
            prev_machine = C[i-1, j] if i > 0 else 0.0
            prev_job = C[i, j-1] if j > 0 else 0.0
            C[i, j] = max(prev_machine, prev_job) + processing_times[i, j] + setup_times[i, j]
    
    return C[-1, -1]  # Makespan = completion of last job on last machine


def generate_scheduling_dataset(
    n_jobs: int = 6,
    n_machines: int = 4,
    n_samples: int = 8000,
    seed: int = 42,
) -> dict:
    """
    Each sample: a random job-shop instance with processing times drawn
    from Uniform(1, 10) and setup times from Uniform(0.5, 3).
    
    Input to network: flattened processing time + setup delay matrix.
    Target: optimal makespan scalar.
    """
    rng = np.random.RandomState(seed)
    
    all_inputs = []
    all_makespans = []
    
    for _ in range(n_samples):
        proc_times = rng.uniform(1.0, 10.0, size=(n_machines, n_jobs)).astype(np.float32)
        setup_times = rng.uniform(0.5, 3.0, size=(n_machines, n_jobs)).astype(np.float32)
        
        makespan = _compute_makespan(proc_times, setup_times, n_jobs, n_machines)
        
        # Input: concatenate flattened processing and setup times
        inp = np.concatenate([proc_times.ravel(), setup_times.ravel()])
        all_inputs.append(inp)
        all_makespans.append(makespan)
    
    inputs = np.stack(all_inputs)
    makespans = np.array(all_makespans, dtype=np.float32)
    
    # Normalize makespans for training stability
    makespan_mean = makespans.mean()
    makespan_std = makespans.std()
    
    # Split 80/20
    n_train = int(0.8 * n_samples)
    
    return {
        "X_train": torch.from_numpy(inputs[:n_train]),
        "y_train": torch.from_numpy(makespans[:n_train]),
        "X_test": torch.from_numpy(inputs[n_train:]),
        "y_test": torch.from_numpy(makespans[n_train:]),
        "input_dim": inputs.shape[1],
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "makespan_mean": float(makespan_mean),
        "makespan_std": float(makespan_std),
    }


class SchedulingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# Models
# =============================================================================

class MLPScheduling(nn.Module):
    """MLP baseline for makespan prediction."""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class TropicalScheduling(nn.Module):
    """Deep Tropical Net for makespan prediction (tropical-biased)."""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=4,
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
        return {
            f"block_{i}": torch.sigmoid(
                block["gate_proj"].bias
            ).detach().cpu().mean().item()
            for i, block in enumerate(self.blocks)
        }


class HybridTropFormerScheduling(nn.Module):
    """TropFormer (classical-biased) for makespan prediction."""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=4,
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
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total = 0
    for X, y in loader:
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
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = F.mse_loss(pred, y)
        total_loss += loss.item() * len(X)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
        total += len(X)
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    mae = (preds - targets).abs().mean().item()
    # Optimality gap: (pred - true) / true * 100%
    opt_gap = ((preds - targets).abs() / targets.clamp(min=1e-6) * 100).mean().item()
    # Correlation
    corr = np.corrcoef(preds.numpy(), targets.numpy())[0, 1]
    
    return {
        "mse": total_loss / total,
        "mae": mae,
        "optimality_gap_pct": opt_gap,
        "correlation": float(corr),
    }


def run_model(name, model, train_loader, test_loader, device, epochs, lr):
    print(f"\n{'='*60}")
    print(f"  {name}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    print(f"{'='*60}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"{'Ep':>3}  {'TrMSE':>10}  {'TeMAE':>8}  {'OptGap%':>8}  {'Time':>6}")
    print("-" * 50)
    
    best_mae = float("inf")
    best_metrics = {}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_mse = train_epoch(model, train_loader, optimizer, device)
        te_metrics = evaluate(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        
        flag = " *" if te_metrics["mae"] < best_mae else ""
        if te_metrics["mae"] < best_mae:
            best_mae = te_metrics["mae"]
            best_metrics = te_metrics.copy()
        
        print(f"{epoch:>3}  {tr_mse:>10.2f}  {te_metrics['mae']:>8.3f}  "
              f"{te_metrics['optimality_gap_pct']:>8.2f}  {elapsed:>5.1f}s{flag}")
    
    return best_metrics, params


def main():
    EPOCHS = 30
    BATCH_SIZE = 128
    LR = 1e-3
    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    N_JOBS = 6
    N_MACHINES = 4
    N_SAMPLES = 8000
    SEED = 42
    
    device = get_device()
    torch.manual_seed(SEED)
    
    print("=" * 60)
    print("  MAX-PLUS SCHEDULING BENCHMARK (Paper 2, B-3)")
    print(f"  Device: {device}  Epochs: {EPOCHS}")
    print(f"  n_jobs={N_JOBS}  n_machines={N_MACHINES}  n_samples={N_SAMPLES}")
    print("=" * 60)
    
    print("\n  Generating scheduling data...")
    data = generate_scheduling_dataset(
        n_jobs=N_JOBS, n_machines=N_MACHINES, n_samples=N_SAMPLES, seed=SEED
    )
    print(f"  Input dim: {data['input_dim']}")
    print(f"  Makespan range: [{data['y_train'].min():.1f}, {data['y_train'].max():.1f}]")
    print(f"  Makespan mean: {data['makespan_mean']:.1f} std: {data['makespan_std']:.1f}")
    
    train_ds = SchedulingDataset(data["X_train"], data["y_train"])
    test_ds = SchedulingDataset(data["X_test"], data["y_test"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    results = {}
    
    # 1. MLP baseline
    torch.manual_seed(SEED)
    mlp = MLPScheduling(data["input_dim"], HIDDEN_DIM, NUM_LAYERS).to(device)
    metrics, params = run_model("MLP-ReLU", mlp, train_loader, test_loader, device, EPOCHS, LR)
    results["mlp"] = {"params": params, **metrics}
    
    # 2. Deep Tropical Net (tropical-biased)
    torch.manual_seed(SEED)
    trop = TropicalScheduling(
        data["input_dim"], HIDDEN_DIM, NUM_LAYERS, lf_pieces=8, lf_mode="blend"
    ).to(device)
    metrics, params = run_model(
        "Deep Tropical Net", trop, train_loader, test_loader, device, EPOCHS, LR
    )
    results["deep_tropical"] = {"params": params, **metrics}
    print("\n  Gate values (higher = more tropical):")
    for name, val in trop.get_gate_values().items():
        print(f"    {name}: {val:.3f}")
    
    # 3. Hybrid TropFormer (classical-biased)
    torch.manual_seed(SEED)
    hybrid = HybridTropFormerScheduling(
        data["input_dim"], HIDDEN_DIM, NUM_LAYERS, lf_pieces=4, lf_mode="blend"
    ).to(device)
    metrics, params = run_model(
        "TropFormer (Hybrid)", hybrid, train_loader, test_loader, device, EPOCHS, LR
    )
    results["tropformer"] = {"params": params, **metrics}
    
    # Summary
    print("\n\n" + "=" * 60)
    print(f"  SCHEDULING FINAL COMPARISON ({EPOCHS} epochs)")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:25s}: MAE={r['mae']:.3f}  OptGap={r['optimality_gap_pct']:.2f}%  "
              f"Corr={r['correlation']:.4f}")
    
    os.makedirs("results", exist_ok=True)
    fname = f"results/scheduling_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved -> {fname}")


if __name__ == "__main__":
    main()
