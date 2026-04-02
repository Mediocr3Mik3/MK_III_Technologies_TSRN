# TropFormer: Tropical Geometry Meets Transformers

A research framework implementing hybrid tropical-classical transformers and deep tropical networks, grounded in max-plus algebra, Maslov dequantization, and Legendre-Fenchel duality.

## Project Structure

```
TropFormer/
├── tropformer.py              # Core TropFormer implementation
├── classical_transformer.py   # Classical baseline for comparison
├── deep_tropical_net.py       # Path B: Deep Tropical Network
├── benchmark_comparison.py    # Reproducible benchmark suite
├── quick_validate.py          # Fast validation script
├── Context.md                 # Theoretical foundations
├── Roadmap.md                 # Development roadmap
├── papers/
│   ├── paper1_tropformer.md   # White paper 1: Hybrid architecture
│   └── paper2_deep_tropical.md # White paper 2: Deep tropical nets
└── results/                   # Benchmark results (generated)
```

## Quick Start

```bash
# Install dependencies
pip install torch torchvision

# Quick validation (2 epochs, small model)
python quick_validate.py

# Full TropFormer training (25 epochs)
python tropformer.py

# Run comparison benchmark
python benchmark_comparison.py

# Deep Tropical Network
python deep_tropical_net.py
```

## Architecture Overview

### TropFormer (Path A: Hybrid)
- **Tropical Q/K projections**: `y_i = max_j(W_ij + x_j)` for hard routing
- **Classical V projection**: Smooth gradient flow through values
- **Maslov Temperature**: Learnable per-head τ interpolates softmax↔argmax
- **LF Dual Activation**: Tropical polynomial + Legendre-Fenchel conjugate
- **Gated Fusion**: Learned blend of tropical and classical paths

### Deep Tropical Net (Path B: Pure Tropical)
- **STE Wrapper**: Exact tropical forward, softmax-weighted backward
- **Tropical BatchNorm**: Prevents polytope collapse/explosion
- **Tropical Residual**: `max(f(x) - center, x)` preserves gradient highway
- **All-tropical attention**: Pure max-plus Q/K/V projections

## Mathematical Foundations

### Tropical Semiring
```
T = (R ∪ {-∞}, ⊕, ⊗)
a ⊕ b = max(a, b)    # tropical addition
a ⊗ b = a + b        # tropical multiplication
```

### Maslov Dequantization
```
τ · log(Σ exp(x/τ)) → max(x) as τ → 0
```
Per-head learnable temperature bridges quantum↔classical.

### Legendre-Fenchel Duality
```
f(x) = max_k(s_k·x + b_k)      # primal: tropical polynomial
f*(y) = max_j(x_j·y - f(x_j))  # dual: LF conjugate (also tropical!)
```

## Validation Gates (from Roadmap)

1. **Training stability**: Loss decreasing, no NaN/Inf
2. **Accuracy**: ≥98.0% on MNIST after 25 epochs
3. **Tropical contribution**: TropFormer ≥ classical ablation
4. **Diagnostic meaning**: Head specialization in Maslov temperatures
5. **Reproducibility**: 3 seeds pass all gates

## Benchmarks

| Task | Dataset | Baseline | Target |
|------|---------|----------|--------|
| Image Classification | MNIST | 98%+ (MLP) | Match |
| Image Classification | CIFAR-10 | 92.4% (ResNet-20) | >90% |
| Sequence Classification | SST-2 | 93.5% (BERT) | >91% |
| Shortest Path | Synthetic | GNN baseline | Superior |
| PWA System ID | Synthetic | MLP | Superior |

## Key Files

- **Context.md**: Full theoretical background for AI agent
- **Roadmap.md**: Development path, white paper outlines, task queues
- **tropformer.py**: Complete implementation with training loop

## Hardware Requirements

- **GPU (recommended)**: CUDA-enabled GPU for fast training
- **CPU**: Supported but slow (~60s/epoch for small models)

## Citation

If using this work, please cite the forthcoming papers:
1. "TropFormer: A Hybrid Tropical-Classical Transformer"
2. "Deep Tropical Networks: Piecewise-Linear Deep Learning"

## License

Research use only. Contact authors for commercial licensing.
