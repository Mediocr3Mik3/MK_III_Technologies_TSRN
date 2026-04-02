"""
Maslov Temperature Heatmap & Routing Path Visualization
========================================================
Diagnostic tools for inspecting tropical-classical interpolation
in trained TropFormer and DTN models.

Generates:
1. Maslov temperature heatmap: tau per head per layer
2. Gate decisiveness: sigmoid(gate_bias) per layer
3. Routing path visualization: which input index j* wins per output i
"""

import json
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_maslov_temps(model) -> dict:
    """
    Collect Maslov temperature values from a trained model.
    Returns dict: {layer_name: tensor of temps per head}
    """
    if hasattr(model, 'maslov_summary'):
        return model.maslov_summary()

    # Manual collection for wrapped models
    temps = {}
    for name, module in model.named_modules():
        if hasattr(module, 'log_temps'):
            tau = torch.exp(module.log_temps).detach().cpu()
            temps[name] = tau
    return temps


def collect_gate_values(model) -> dict:
    """
    Collect gate bias values (tropical vs classical routing).
    Higher sigmoid(bias) = more tropical.
    """
    if hasattr(model, 'gate_summary'):
        return model.gate_summary()

    gates = {}
    for name, module in model.named_modules():
        if hasattr(module, 'bias') and 'gate' in name.lower():
            sig = torch.sigmoid(module.bias).detach().cpu().mean().item()
            gates[name] = sig
    return gates


def compute_routing_paths(model, x: torch.Tensor) -> list:
    """
    For each TropicalLinear layer, compute which input index j*
    wins for each output i: j* = argmax_j(W_ij + x_j).

    Returns list of dicts with routing info per layer.
    """
    from tropformer import TropicalLinear

    routes = []
    hooks = []

    def make_hook(layer_name, layer):
        def hook_fn(module, input, output):
            x_in = input[0]
            leading = x_in.shape[:-1]
            x_flat = x_in.reshape(-1, module.in_features)
            scores = module.weight.unsqueeze(0) + x_flat.unsqueeze(1)
            winners = scores.argmax(dim=-1)  # (batch, out_features)
            routes.append({
                "layer": layer_name,
                "winners": winners.detach().cpu(),
                "in_features": module.in_features,
                "out_features": module.out_features,
            })
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, TropicalLinear):
            h = module.register_forward_hook(make_hook(name, module))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return routes


def generate_maslov_heatmap_text(temps: dict, model_name: str = "Model") -> str:
    """Generate ASCII heatmap of Maslov temperatures."""
    lines = [f"\n{'='*60}", f"  Maslov Temperature Heatmap: {model_name}", "="*60]

    if not temps:
        lines.append("  No Maslov temperatures found.")
        return "\n".join(lines)

    # Categorize temperatures
    lines.append(f"  {'Layer':<30} {'Heads (tau values)':>30}")
    lines.append("-" * 60)

    for layer_name, tau_values in temps.items():
        if isinstance(tau_values, torch.Tensor):
            tau_list = tau_values.tolist()
        else:
            tau_list = [tau_values]

        # Format with color indicators
        formatted = []
        for t in tau_list:
            if t < 0.3:
                formatted.append(f"[TROP {t:.2f}]")
            elif t < 0.8:
                formatted.append(f"[MIX  {t:.2f}]")
            elif t < 1.5:
                formatted.append(f"[STD  {t:.2f}]")
            else:
                formatted.append(f"[SOFT {t:.2f}]")
        lines.append(f"  {layer_name:<30} {' '.join(formatted)}")

    lines.append("")
    lines.append("  Legend: [TROP] tau<0.3 = near-argmax (tropical routing)")
    lines.append("          [MIX]  tau<0.8 = mixed routing")
    lines.append("          [STD]  tau<1.5 = standard softmax")
    lines.append("          [SOFT] tau>1.5 = near-uniform attention")
    return "\n".join(lines)


def generate_gate_report(gates: dict, model_name: str = "Model") -> str:
    """Generate text report of gate decisiveness."""
    lines = [f"\n{'='*60}", f"  Gate Decisiveness: {model_name}", "="*60]

    if not gates:
        lines.append("  No gates found.")
        return "\n".join(lines)

    lines.append(f"  {'Layer':<30} {'sigmoid(bias)':<12} {'Routing':<15}")
    lines.append("-" * 60)

    for name, val in gates.items():
        if val > 0.8:
            routing = "TROPICAL"
        elif val > 0.5:
            routing = "MIXED-TROP"
        elif val > 0.2:
            routing = "MIXED-CLASS"
        else:
            routing = "CLASSICAL"
        lines.append(f"  {name:<30} {val:<12.3f} {routing:<15}")

    return "\n".join(lines)


def generate_routing_report(routes: list, model_name: str = "Model",
                            max_samples: int = 3) -> str:
    """Generate text report of routing paths (winning indices)."""
    lines = [f"\n{'='*60}", f"  Routing Paths: {model_name}", "="*60]

    if not routes:
        lines.append("  No TropicalLinear layers found.")
        return "\n".join(lines)

    for route_info in routes:
        name = route_info["layer"]
        winners = route_info["winners"]  # (batch, out_features)
        in_f = route_info["in_features"]
        out_f = route_info["out_features"]

        # Statistics over batch
        # How many unique winners per output?
        n_samples = min(winners.shape[0], 100)
        unique_per_output = []
        for o in range(min(out_f, 32)):
            n_unique = len(torch.unique(winners[:n_samples, o]))
            unique_per_output.append(n_unique)

        avg_unique = np.mean(unique_per_output)
        max_unique = max(unique_per_output)

        lines.append(f"\n  {name}: {in_f} -> {out_f}")
        lines.append(f"    Avg unique winners per output: {avg_unique:.1f}/{n_samples}")
        lines.append(f"    Max unique winners: {max_unique}/{n_samples}")

        # Show routing for first few samples
        for s in range(min(max_samples, winners.shape[0])):
            w = winners[s, :min(16, out_f)].tolist()
            lines.append(f"    Sample {s}: j* = {w}{'...' if out_f > 16 else ''}")

        # Routing concentration: what fraction of inputs are ever selected?
        all_winners = winners[:n_samples].reshape(-1)
        n_active = len(torch.unique(all_winners))
        lines.append(f"    Active inputs: {n_active}/{in_f} "
                     f"({100*n_active/in_f:.0f}%)")

    return "\n".join(lines)


def run_diagnostics(model, model_name: str, sample_input: torch.Tensor,
                    save_path: Optional[str] = None):
    """Run all diagnostics on a trained model."""
    print(f"\n{'#'*60}")
    print(f"  DIAGNOSTICS: {model_name}")
    print(f"{'#'*60}")

    # 1. Maslov temperatures
    temps = collect_maslov_temps(model)
    print(generate_maslov_heatmap_text(temps, model_name))

    # 2. Gate values
    gates = collect_gate_values(model)
    print(generate_gate_report(gates, model_name))

    # 3. Routing paths
    routes = compute_routing_paths(model, sample_input)
    print(generate_routing_report(routes, model_name))

    # Save diagnostics
    if save_path:
        diag = {
            "model": model_name,
            "maslov_temps": {k: v.tolist() if isinstance(v, torch.Tensor) else v
                           for k, v in temps.items()},
            "gates": gates,
            "routing_stats": [
                {
                    "layer": r["layer"],
                    "in_features": r["in_features"],
                    "out_features": r["out_features"],
                    "avg_unique_winners": float(
                        np.mean([len(torch.unique(r["winners"][:100, o]))
                                for o in range(min(r["out_features"], 32))])
                    ),
                }
                for r in routes
            ],
        }
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(diag, f, indent=2, default=float)
        print(f"\n  Diagnostics saved -> {save_path}")


def main():
    """Quick test on untrained models to verify the pipeline works."""
    from tropformer import TropFormer
    from deep_tropical_net import DeepTropNet

    torch.manual_seed(42)

    # TropFormer
    trop = TropFormer(
        img_size=32, patch_size=8, in_channels=3, num_classes=10,
        d_model=64, num_heads=2, num_layers=2, ffn_dim=128,
    )
    x = torch.randn(4, 3, 32, 32)
    run_diagnostics(trop, "TropFormer (untrained)", x, "results/diag_tropformer.json")

    # DTN
    dtn = DeepTropNet(
        img_size=32, patch_size=8, in_channels=3, num_classes=10,
        d_model=64, num_layers=2, num_heads=2, ffn_dim=128,
    )
    run_diagnostics(dtn, "DTN (untrained)", x, "results/diag_dtn.json")


if __name__ == "__main__":
    main()
