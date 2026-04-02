"""
Synthetic Graph Shortest Path Benchmark (Paper 2, Tier 3, B-2)
================================================================
Random graphs with random positive edge weights. Task: predict the
shortest path distance between all node pairs.

Why this is intrinsically tropical:
    The all-pairs shortest path is computed by tropical matrix powers:
        D = A^{(trop, n-1)}
    where A is the weighted adjacency matrix and ^ means tropical matrix
    power (repeated min-plus matrix multiply). A deep tropical net should
    converge to this algorithm.
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
# Shortest Path Data Generator
# =============================================================================

def generate_shortest_path_dataset(
    n_nodes: int = 12,
    edge_prob: float = 0.4,
    n_graphs: int = 5000,
    weight_range: tuple = (0.1, 5.0),
    seed: int = 42,
) -> dict:
    """
    Ground truth computed via Floyd-Warshall (tropical matrix power).
    Input: adjacency matrix with edge weights (0 = no edge).
    Target: all-pairs shortest path distance matrix.
    """
    from scipy.sparse.csgraph import shortest_path as sp_shortest_path

    rng = np.random.RandomState(seed)

    adj_matrices = []
    dist_matrices = []
    valid_masks = []  # 1 where path exists, 0 where inf

    for _ in range(n_graphs):
        # Random Erdos-Renyi graph
        edges = rng.random((n_nodes, n_nodes)) < edge_prob
        np.fill_diagonal(edges, False)
        # Make undirected
        edges = edges | edges.T

        # Random positive weights
        weights = rng.uniform(weight_range[0], weight_range[1],
                              size=(n_nodes, n_nodes)).astype(np.float32)
        adj = np.where(edges, weights, 0.0).astype(np.float32)

        # Compute all-pairs shortest path
        dist = sp_shortest_path(adj, directed=False).astype(np.float32)

        # Mask for valid (non-infinite) paths
        valid = np.isfinite(dist).astype(np.float32)
        dist = np.where(valid > 0, dist, 0.0)  # replace inf with 0

        adj_matrices.append(adj)
        dist_matrices.append(dist)
        valid_masks.append(valid)

    adj_matrices = np.stack(adj_matrices)
    dist_matrices = np.stack(dist_matrices)
    valid_masks = np.stack(valid_masks)

    # Split train/test (80/20)
    n_train = int(0.8 * n_graphs)
    return {
        "adj_train": torch.from_numpy(adj_matrices[:n_train]),
        "dist_train": torch.from_numpy(dist_matrices[:n_train]),
        "mask_train": torch.from_numpy(valid_masks[:n_train]),
        "adj_test": torch.from_numpy(adj_matrices[n_train:]),
        "dist_test": torch.from_numpy(dist_matrices[n_train:]),
        "mask_test": torch.from_numpy(valid_masks[n_train:]),
        "n_nodes": n_nodes,
    }


class ShortestPathDataset(Dataset):
    def __init__(self, adj, dist, mask):
        self.adj = adj
        self.dist = dist
        self.mask = mask

    def __len__(self):
        return len(self.adj)

    def __getitem__(self, idx):
        return self.adj[idx], self.dist[idx], self.mask[idx]


# =============================================================================
# Models for Shortest Path Prediction
# =============================================================================

class MLPShortestPath(nn.Module):
    """Flat MLP baseline: flatten adjacency, predict flattened distance."""

    def __init__(self, n_nodes, hidden_dim=256, num_layers=4):
        super().__init__()
        input_dim = n_nodes * n_nodes
        output_dim = n_nodes * n_nodes
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.n_nodes = n_nodes

    def forward(self, adj):
        B = adj.shape[0]
        x = adj.reshape(B, -1)
        return self.net(x).reshape(B, self.n_nodes, self.n_nodes)


class TropicalShortestPath(nn.Module):
    """
    Deep Tropical Net for shortest path prediction.
    Uses tropical layers that can learn the min-plus matrix power.
    
    Architecture mirrors the Floyd-Warshall structure:
    each layer can learn one step of tropical matrix power.
    """

    def __init__(self, n_nodes, hidden_dim=256, num_layers=4,
                 lf_pieces=8, lf_mode="blend"):
        super().__init__()
        from tropformer import TropicalLinear, LFDualActivation

        input_dim = n_nodes * n_nodes
        output_dim = n_nodes * n_nodes
        self.n_nodes = n_nodes

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
            # Tropical-biased gates (DTN style)
            nn.init.zeros_(self.blocks[-1]["gate_proj"].weight)
            nn.init.constant_(self.blocks[-1]["gate_proj"].bias, 2.0)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, adj):
        B = adj.shape[0]
        x = adj.reshape(B, -1)
        h = self.input_proj(x)
        for block in self.blocks:
            res = h
            h_n = block["norm"](h)
            trop_out = block["lf"](block["trop"](h_n))
            class_out = block["classical"](h_n)
            g = torch.sigmoid(block["gate_proj"](h_n))
            h = res + g * trop_out + (1 - g) * class_out
        h = self.out_norm(h)
        return self.head(h).reshape(B, self.n_nodes, self.n_nodes)

    def get_gate_values(self):
        return {
            f"block_{i}": torch.sigmoid(
                block["gate_proj"].bias
            ).detach().cpu().mean().item()
            for i, block in enumerate(self.blocks)
        }


class HybridTropFormerSP(nn.Module):
    """TropFormer (classical-biased) for shortest path."""

    def __init__(self, n_nodes, hidden_dim=256, num_layers=4,
                 lf_pieces=4, lf_mode="blend"):
        super().__init__()
        from tropformer import TropicalLinear, LFDualActivation

        input_dim = n_nodes * n_nodes
        output_dim = n_nodes * n_nodes
        self.n_nodes = n_nodes

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
            # Classical-biased gates (TropFormer style)
            nn.init.zeros_(self.blocks[-1]["gate_proj"].weight)
            nn.init.constant_(self.blocks[-1]["gate_proj"].bias, -2.0)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, adj):
        B = adj.shape[0]
        x = adj.reshape(B, -1)
        h = self.input_proj(x)
        for block in self.blocks:
            res = h
            h_n = block["norm"](h)
            trop_out = block["lf"](block["trop"](h_n))
            class_out = block["classical"](h_n)
            g = torch.sigmoid(block["gate_proj"](h_n))
            h = res + g * trop_out + (1 - g) * class_out
        h = self.out_norm(h)
        return self.head(h).reshape(B, self.n_nodes, self.n_nodes)


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(pred, target, mask):
    """Compute MAE and optimality rate on valid (non-inf) paths."""
    valid = mask.bool()
    if valid.sum() == 0:
        return {"mae": 0.0, "optimality_rate": 0.0}

    pred_valid = pred[valid]
    target_valid = target[valid]

    mae = (pred_valid - target_valid).abs().mean().item()

    # Optimality rate: fraction within 5% of true shortest path
    relative_error = ((pred_valid - target_valid).abs() /
                      target_valid.clamp(min=1e-6))
    optimality_rate = (relative_error < 0.05).float().mean().item()

    return {"mae": mae, "optimality_rate": optimality_rate}


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total = 0
    for adj, dist, mask in loader:
        adj, dist, mask = adj.to(device), dist.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(adj)
        # Masked MSE loss (only on valid paths)
        valid = mask.bool()
        if valid.sum() > 0:
            loss = F.mse_loss(pred[valid], dist[valid])
        else:
            loss = torch.tensor(0.0, device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * adj.size(0)
        total += adj.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    total_loss = total = 0
    for adj, dist, mask in loader:
        adj, dist, mask = adj.to(device), dist.to(device), mask.to(device)
        pred = model(adj)
        valid = mask.bool()
        if valid.sum() > 0:
            loss = F.mse_loss(pred[valid], dist[valid])
            total_loss += loss.item() * adj.size(0)
        all_preds.append(pred.cpu())
        all_targets.append(dist.cpu())
        all_masks.append(mask.cpu())
        total += adj.size(0)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    masks = torch.cat(all_masks)
    metrics = compute_metrics(preds, targets, masks)
    metrics["mse"] = total_loss / max(total, 1)
    return metrics


def run_model(name, model, train_loader, test_loader, device, epochs, lr):
    print(f"\n{'='*60}")
    print(f"  {name}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    print(f"{'='*60}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"{'Ep':>3}  {'TrMSE':>10}  {'TeMAE':>8}  {'OptRate':>8}  {'Time':>6}")
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

        print(f"{epoch:>3}  {tr_mse:>10.4f}  {te_metrics['mae']:>8.4f}  "
              f"{te_metrics['optimality_rate']:>8.3f}  {elapsed:>5.1f}s{flag}")

    return best_metrics, params


def main():
    EPOCHS = 30
    BATCH_SIZE = 64
    LR = 1e-3
    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    N_NODES = 12
    EDGE_PROB = 0.4
    N_GRAPHS = 5000
    SEED = 42

    device = get_device()
    torch.manual_seed(SEED)

    print("=" * 60)
    print("  GRAPH SHORTEST PATH BENCHMARK (Paper 2, B-2)")
    print(f"  Device: {device}  Epochs: {EPOCHS}")
    print(f"  n_nodes={N_NODES}  edge_prob={EDGE_PROB}  n_graphs={N_GRAPHS}")
    print("=" * 60)

    print("\n  Generating shortest path data...")
    data = generate_shortest_path_dataset(
        n_nodes=N_NODES, edge_prob=EDGE_PROB, n_graphs=N_GRAPHS, seed=SEED
    )
    n_train = len(data["adj_train"])
    n_test = len(data["adj_test"])
    print(f"  Train: {n_train}  Test: {n_test}")

    # Compute stats on valid paths
    valid_frac_train = data["mask_train"].mean().item()
    valid_frac_test = data["mask_test"].mean().item()
    print(f"  Valid path fraction: train={valid_frac_train:.2f} test={valid_frac_test:.2f}")

    train_ds = ShortestPathDataset(data["adj_train"], data["dist_train"], data["mask_train"])
    test_ds = ShortestPathDataset(data["adj_test"], data["dist_test"], data["mask_test"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = {}

    # 1. MLP baseline
    torch.manual_seed(SEED)
    mlp = MLPShortestPath(N_NODES, HIDDEN_DIM, NUM_LAYERS).to(device)
    metrics, params = run_model("MLP-ReLU", mlp, train_loader, test_loader, device, EPOCHS, LR)
    results["mlp"] = {"params": params, **metrics}

    # 2. Deep Tropical Net (tropical-biased)
    torch.manual_seed(SEED)
    trop = TropicalShortestPath(
        N_NODES, HIDDEN_DIM, NUM_LAYERS, lf_pieces=8, lf_mode="blend"
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
    hybrid = HybridTropFormerSP(
        N_NODES, HIDDEN_DIM, NUM_LAYERS, lf_pieces=4, lf_mode="blend"
    ).to(device)
    metrics, params = run_model(
        "TropFormer (Hybrid)", hybrid, train_loader, test_loader, device, EPOCHS, LR
    )
    results["tropformer"] = {"params": params, **metrics}

    # Summary
    print("\n\n" + "=" * 60)
    print(f"  SHORTEST PATH FINAL COMPARISON ({EPOCHS} epochs)")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:25s}: MAE={r['mae']:.4f}  OptRate={r['optimality_rate']:.3f}")

    os.makedirs("results", exist_ok=True)
    fname = f"results/shortest_path_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved -> {fname}")


if __name__ == "__main__":
    main()
