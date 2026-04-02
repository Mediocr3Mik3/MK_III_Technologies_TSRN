"""
8-Point Ablation Study
=======================
Tests the contribution of each architectural component in TropFormer and DTN
on CIFAR-10 (and optionally ListOps).

Paper 1 ablations (TropFormer):
  A1: No tropical scores (gate fixed to 0 → classical-only scores)
  A2: No LF dual activation (replace with GELU)
  A3: No Maslov temperature (freeze τ=1, standard softmax)
  A4: No gated fusion (gate fixed to 0.5, equal blend)

Paper 2 ablations (DTN):
  A5: No STE in TropicalLinear (hard max gradient, no softmax STE)
  A6: No TropicalBatchNorm (replace with LayerNorm)
  A7: No tropical Q/K projections (replace TropicalLinear with nn.Linear)
  A8: Full destabilization (remove STE + TropBN + fix gate to pure tropical)

Each ablation trains the modified model on CIFAR-10 for 15 epochs and
reports best test accuracy, compared to the full model baseline.
"""

import copy
import json
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_utils import get_device
from tropformer import (
    TropFormer,
    TropicalLinear,
    TropicalMultiHeadAttention,
    TropicalHybridFFN,
    LFDualActivation,
    MaslovTemperature,
    _tropical_max,
)
from deep_tropical_net import (
    DeepTropNet,
    DTNAttention,
    DTNHybridFFN,
    DTNBlock,
    TropicalBatchNorm,
)


# =============================================================================
# Training utilities (reused from cifar10_benchmark)
# =============================================================================

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


def run_ablation(name, model, train_loader, test_loader, device, epochs, lr):
    """Train model and return best test accuracy."""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  {name}")
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


# =============================================================================
# Ablation modifiers — each takes a freshly constructed model and modifies it
# =============================================================================

def ablate_no_tropical_scores(model):
    """A1: Fix score gate to 0 → only classical dot-product scores used."""
    for block in model.blocks:
        attn = block.attn
        # Fix gate bias to -20 so sigmoid(-20) ≈ 0 → all classical
        with torch.no_grad():
            attn.score_gate.weight.zero_()
            attn.score_gate.bias.fill_(-20.0)
        # Freeze the gate so it stays at 0
        attn.score_gate.weight.requires_grad_(False)
        attn.score_gate.bias.requires_grad_(False)
    return model


def ablate_no_lf_activation(model):
    """A2: Replace LFDualActivation with GELU in FFN tropical branch."""
    for block in model.blocks:
        ffn = block.ffn
        ffn.lf_act = nn.GELU()
    return model


def ablate_no_maslov_temp(model):
    """A3: Freeze Maslov temperature at τ=1 (standard softmax)."""
    import math
    for block in model.blocks:
        attn = block.attn
        with torch.no_grad():
            attn.maslov.log_temps.fill_(math.log(1.0))
        attn.maslov.log_temps.requires_grad_(False)
    return model


def ablate_fixed_gate(model):
    """A4: Fix gate to 0.5 (equal blend of tropical and classical scores)."""
    for block in model.blocks:
        attn = block.attn
        with torch.no_grad():
            attn.score_gate.weight.zero_()
            attn.score_gate.bias.fill_(0.0)  # sigmoid(0) = 0.5
        attn.score_gate.weight.requires_grad_(False)
        attn.score_gate.bias.requires_grad_(False)
        # Also fix FFN gate to 0.5
        ffn = block.ffn
        with torch.no_grad():
            ffn.gate_proj.weight.zero_()
            ffn.gate_proj.bias.fill_(0.0)
        ffn.gate_proj.weight.requires_grad_(False)
        ffn.gate_proj.bias.requires_grad_(False)
    return model


def ablate_no_ste(model):
    """A5: Disable STE in TropicalLinear — use hard max gradient only."""
    # Set STE temp to 0 effectively by making it very small
    # Actually, we replace the forward to use raw max (no STE)
    def _hard_max_forward(self, x):
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        scores = self.weight.unsqueeze(0) + x_flat.unsqueeze(1)
        out = scores.max(dim=-1).values  # hard max, sparse gradient
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*leading, self.out_features)

    for module in model.modules():
        if isinstance(module, TropicalLinear):
            import types
            module.forward = types.MethodType(_hard_max_forward, module)
    return model


def ablate_no_tropbn(model):
    """A6: Replace TropicalBatchNorm with LayerNorm in DTN."""
    if hasattr(model, 'trop_bn'):
        d_model = model.trop_bn.num_features
        model.trop_bn = nn.LayerNorm(d_model)
    return model


def ablate_no_tropical_qk(model):
    """A7: Replace TropicalLinear Q/K projections with nn.Linear in DTN."""
    for block in model.blocks:
        attn = block.attn
        if isinstance(attn, DTNAttention):
            d_model = attn.d_model
            # Replace TropicalLinear Q/K with nn.Linear
            attn.q_proj = nn.Linear(d_model, d_model)
            attn.k_proj = nn.Linear(d_model, d_model)
    return model


def ablate_full_destabilization(model):
    """A8: Remove all stabilization from DTN (no STE + no TropBN + force pure tropical gate)."""
    # Remove STE
    model = ablate_no_ste(model)
    # Remove TropBN
    model = ablate_no_tropbn(model)
    # Force gate to pure tropical (sigmoid(+20) ≈ 1.0)
    for block in model.blocks:
        attn = block.attn
        if isinstance(attn, DTNAttention):
            with torch.no_grad():
                attn.score_gate.weight.zero_()
                attn.score_gate.bias.fill_(+20.0)
            attn.score_gate.weight.requires_grad_(False)
            attn.score_gate.bias.requires_grad_(False)
        ffn = block.ffn
        if isinstance(ffn, DTNHybridFFN):
            with torch.no_grad():
                ffn.gate_proj.weight.zero_()
                ffn.gate_proj.bias.fill_(+20.0)
            ffn.gate_proj.weight.requires_grad_(False)
            ffn.gate_proj.bias.requires_grad_(False)
    return model


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="8-Point Ablation Study")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per ablation")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline runs, use cached 30-epoch results")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only specific ablation(s), comma-separated: A1,A2,...")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'dml' (default: auto-detect)")
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = 64
    LR = 3e-3
    # Smaller model for ablations — relative differences matter, not absolute acc
    D_MODEL = 64
    NUM_HEADS = 2
    NUM_LAYERS = 2
    FFN_DIM = 128
    SEED = 42
    IMG_SIZE = 32
    PATCH_SIZE = 8
    IN_CHANNELS = 3

    if args.device == "cpu":
        device = torch.device("cpu")
        # Don't enable smooth max on CPU
        import tropformer
        tropformer._USE_SMOOTH_MAX = False
        print("  [Using CPU, smooth max disabled]")
    elif args.device == "dml":
        device = get_device()
    else:
        device = get_device()

    print("=" * 60)
    print("  8-POINT ABLATION STUDY (CIFAR-10)")
    print(f"  Device: {device}  Epochs: {EPOCHS}  Seed: {SEED}")
    print(f"  d={D_MODEL} h={NUM_HEADS} L={NUM_LAYERS} ffn={FFN_DIM}")
    print("=" * 60)

    # Data
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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds)}  Test: {len(test_ds)}")

    results = {}

    # Cached baselines will be populated by running BASE_TF and BASE_DTN
    CACHED_BASELINES = {}

    # Helper to build fresh TropFormer
    def make_tropformer():
        torch.manual_seed(SEED)
        return TropFormer(
            img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS,
            num_classes=10, d_model=D_MODEL, num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS, ffn_dim=FFN_DIM, dropout=0.1,
            trop_dropout=0.05, lf_pieces=4, lf_mode="blend", init_temp=1.0,
        )

    # Helper to build fresh DTN
    def make_dtn():
        torch.manual_seed(SEED)
        return DeepTropNet(
            img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS,
            num_classes=10, d_model=D_MODEL, num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS, ffn_dim=FFN_DIM, lf_pieces=4,
            lf_mode="blend", init_temp=1.0, dropout=0.1,
        )

    # Determine which ablations to run
    selected = None
    if args.only:
        selected = [s.strip().upper() for s in args.only.split(",")]

    def should_run(tag):
        return selected is None or tag in selected

    # Define ablation schedule
    ablations = [
        # (tag, name, model_factory, ablation_fn)
        ("BASE_TF", "TropFormer (baseline)", make_tropformer, None),
        ("A1", "A1: No tropical scores (gate=0)", make_tropformer, ablate_no_tropical_scores),
        ("A2", "A2: No LF activation (GELU only)", make_tropformer, ablate_no_lf_activation),
        ("A3", "A3: No Maslov temp (t=1 fixed)", make_tropformer, ablate_no_maslov_temp),
        ("A4", "A4: Fixed gate (50/50 blend)", make_tropformer, ablate_fixed_gate),
        ("BASE_DTN", "DTN (baseline)", make_dtn, None),
        ("A5", "A5: No STE (hard max grad)", make_dtn, ablate_no_ste),
        ("A6", "A6: No TropicalBatchNorm", make_dtn, ablate_no_tropbn),
        ("A7", "A7: No tropical Q/K (nn.Linear)", make_dtn, ablate_no_tropical_qk),
        ("A8", "A8: Full destabilization", make_dtn, ablate_full_destabilization),
    ]

    for tag, name, factory, ablate_fn in ablations:
        is_baseline = tag.startswith("BASE")

        if is_baseline and args.skip_baselines:
            if tag in CACHED_BASELINES:
                results[tag] = CACHED_BASELINES[tag]
                print(f"\n  [Using cached {tag}: {CACHED_BASELINES[tag]['best_acc']*100:.2f}%]")
            continue

        if not is_baseline and not should_run(tag):
            continue

        model = factory()
        if ablate_fn is not None:
            model = ablate_fn(model)
        model = model.to(device)

        acc, params = run_ablation(
            name, model, train_loader, test_loader, device, EPOCHS, LR
        )
        results[tag] = {"name": name, "best_acc": acc, "params": params}

        # Free GPU memory
        del model
        if hasattr(torch, 'dml'):
            pass  # DirectML doesn't have empty_cache
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n\n" + "=" * 60)
    print("  ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"  {'Tag':<10} {'Name':<35} {'Acc':>7}")
    print("-" * 60)
    for tag, r in results.items():
        print(f"  {tag:<10} {r['name']:<35} {r['best_acc']*100:>6.2f}%")

    # Compute deltas from baselines
    if "BASE_TF" in results:
        base_tf = results["BASE_TF"]["best_acc"]
        print(f"\n  TropFormer ablation deltas (baseline: {base_tf*100:.2f}%):")
        for tag in ["A1", "A2", "A3", "A4"]:
            if tag in results:
                delta = (results[tag]["best_acc"] - base_tf) * 100
                print(f"    {tag}: {delta:+.2f}%  ({results[tag]['name']})")

    if "BASE_DTN" in results:
        base_dtn = results["BASE_DTN"]["best_acc"]
        print(f"\n  DTN ablation deltas (baseline: {base_dtn*100:.2f}%):")
        for tag in ["A5", "A6", "A7", "A8"]:
            if tag in results:
                delta = (results[tag]["best_acc"] - base_dtn) * 100
                print(f"    {tag}: {delta:+.2f}%  ({results[tag]['name']})")

    # Save results
    os.makedirs("results", exist_ok=True)
    fname = f"results/ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved -> {fname}")


if __name__ == "__main__":
    main()
