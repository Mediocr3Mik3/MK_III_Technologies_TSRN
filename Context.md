# TropFormer: Agent Context Document

> **Purpose of this document:** This file is a living reference for an AI coding agent
> (Windsurf / Cascade or similar) working on `tropformer.py`. It covers the theoretical
> foundations, design intent, architectural decisions, known trade-offs, and roadmap for
> the project. Read this before touching any code. Reference it when making decisions
> about modifications, extensions, or debugging.

---

## 1. Project Identity

**What this is:** TropFormer is a research-grade hybrid neural network that fuses
*tropical geometry* (max-plus algebra) with *classical linear algebra* inside a
transformer architecture. It is not a standard ViT or MLP — its mathematical
foundations are fundamentally different from any mainstream model.

**What this is not:** This is not a drop-in replacement for a standard transformer.
It is an experiment in whether tropical algebraic structure (piecewise-linear
combinatorial geometry) can provide complementary inductive bias to classical smooth
gradient flow when combined inside a single trainable architecture.

**Primary author context:** The lead engineer has a background in control theory and
advanced engineering mathematics (graduate-level). They are *not* a tropical geometry
specialist by training, but are deeply comfortable with piecewise-affine systems,
Hamiltonian mechanics, optimal control, Bellman equations, and nonsmooth analysis.
Explanations and design choices should be grounded in those concepts wherever possible.

**Current training target:** MNIST (28×28 grayscale, 10 classes). This is intentional
— it is a well-understood benchmark that lets us validate the architecture quickly before
scaling. The architecture is designed to generalize to other datasets with minimal changes.

---

## 2. The Two Mathematical Worlds Being Fused

### 2a. Classical (standard) linear algebra — the familiar world

Standard neural networks operate in **(ℝ, +, ×)**: the real numbers under ordinary
addition and multiplication. Linear layers compute `y = Wx + b`. Activations like ReLU,
GELU, and sigmoid introduce nonlinearity. Gradients flow via the chain rule. Everything
is smooth (or piecewise smooth) and amenable to first-order optimization.

**Inductive bias:** smooth manifold structure. Features are dense vectors. Every weight
contributes to every output. Information aggregation is global.

### 2b. Tropical algebra — the new world

The **tropical semiring** `T = (ℝ ∪ {−∞}, ⊕, ⊗)` replaces the two ring operations:

```
a ⊕ b  =  max(a, b)          tropical addition
a ⊗ b  =  a + b              tropical multiplication
```

The additive identity ("zero") is `−∞` because `max(a, −∞) = a`.
The multiplicative identity ("one") is `0` because `a + 0 = a`.

The distributive law still holds: `max(a, b+c) = max(a,b) + max(a,c)`. The only thing
missing relative to a field is subtraction. This means all of linear algebra (matrix
multiply, eigenvalues, characteristic polynomials) still works — but the answers are
geometrically different.

**Tropical linear map:**
```
(W ⊗ x)_i  =  max_j ( W_ij + x_j )
```
This is: for each output neuron i, find the input neuron j that has the highest
*combined* weight+activation value, and report that maximum.

**Inductive bias:** piecewise-linear geometry. The input space ℝⁿ is partitioned into
polyhedral cells — one cell per "winning" j. Inside each cell, the map is affine.
The boundaries between cells are where two j's tie for the maximum. This is *hard,
sparse routing* — exactly one input drives each output, and the driving index is a
discrete combinatorial choice.

### 2c. Why this duality is interesting

These two worlds have *complementary* failure modes:

- Classical layers: smooth, dense, rich gradients everywhere — but cannot easily learn
  hard routing, discrete mode switching, or combinatorial structure.
- Tropical layers: hard routing, combinatorial geometry, sparse gradients — but
  gradient signal is only available through one winning index at a time (subgradient),
  making optimization slower and noisier.

**The hypothesis of TropFormer:** A learned gate that blends both computations per
feature dimension allows the network to use tropical routing where it is geometrically
natural and classical smoothness where it is not. The gate itself is a classical sigmoid
— always differentiable — so the entire system trains end-to-end.

---

## 3. The Control Theory Analogy (core mental model)

This is the mental model that maps tropical geometry onto the lead engineer's primary
domain. Every tropical neural network layer is a **piecewise-affine (PWA) switched
system** in feature space.

| Tropical neural network concept | PWA switched system equivalent |
|---|---|
| Weight matrix `W` | Collection of affine maps `{Aₖx + bₖ}` |
| Winning index `j* = argmax_j(W_ij + x_j)` | Active discrete mode `k ∈ {1..N}` |
| Polyhedral cell where j* is constant | Switching region / guard |
| Tropical variety (cell boundaries) | Switching manifold / sliding surface |
| Tropical variety in ℝⁿ | Polyhedral complex tiling the state space |
| Backprop through argmax | Subgradient descent on a nonsmooth value function |
| Learned W | Optimal switching law found by gradient descent |

The Bellman optimality equation in dynamic programming:
```
V(x) = min_u [ c(x,u) + V(f(x,u)) ]
```
is a **min-plus** tropical operation (using min instead of max, same algebraic
structure). Running Dijkstra or Bellman-Ford computes tropical matrix powers.
The steady-state throughput of a max-plus timed event graph (manufacturing scheduling)
is the tropical eigenvalue of the system matrix.

The lead engineer has worked with these concepts in control contexts. The key insight
is that TropFormer is learning the PWA partition *end-to-end from data*, rather than
designing it by hand from system knowledge.

---

## 4. Module-by-Module Theory Reference

### 4a. `TropicalLinear` — the core primitive

```python
y_i = max_j( W_ij + x_j ) + b_i
```

**Geometry:** Each output neuron i defines a competition among input neurons. The winner
is the j that contributes the most. The input space is partitioned into regions
(polyhedral cells) based on which j wins for each i.

**Gradient:** Only the winning j* receives gradient signal per (batch element, output i).
This is a *subgradient* of the max function. All other connections get zero gradient in
that step. This is intentional — it implements hard routing — but it means convergence
is slower than classical layers. Use with lower learning rates on the tropical parameters
if instability is observed, or increase the classical branch weight via the gate
initialization.

**Initialization:** Weights are uniform in `[-0.5, 0.5]`. This is important — if weights
are initialized too large, one j dominates from the start and the network never explores
other routing configurations. If weights are initialized too small (close to zero), all
j's tie and the gradient is ambiguous. Uniform `[-0.5, 0.5]` gives moderate spread.

**Shape handling:** Supports arbitrary leading batch dimensions `(..., in_features)` by
flattening and reshaping. This allows it to operate on transformer token sequences
`(B, L, d_model)` as well as flat vectors `(B, d_model)`.

**Caution — do not confuse with ReLU networks:** A tropical linear layer followed by
nothing is already a nonlinear function (because max is nonlinear). Do not add a
separate activation function after `TropicalLinear` unless you specifically intend to
compose two nonlinearities. The nonlinearity is *inside* the tropical operation.

---

### 4b. `TropicalDropout` — tropical zero, not classical zero

```python
TROP_ZERO = -1e9   # practical stand-in for −∞
```

Standard dropout sets masked activations to `0`. In the tropical semiring, `0` is not
the additive identity — `−∞` is, because `max(a, −∞) = a`. Setting a masked activation
to `0` in a tropical context means the masked neuron still participates with value `0`
in downstream max operations, which defeats the purpose of dropout.

`TropicalDropout` sets masked neurons to `−1e9`, which functions as tropical zero. A
dropped neuron has no influence on any downstream max — it is correctly "absent."

**Rate:** Default `p=0.05` (5%). Tropical dropout is more aggressive per-unit than
classical dropout because it completely silences a unit rather than scaling it. Keep
this rate low. Values above `0.15` tend to destabilize training because the subgradient
is already sparse — losing more routing paths hurts convergence.

---

### 4c. `MaslovTemperature` — the quantum-to-tropical bridge

This is the most theoretically deep component. The Maslov dequantization bridge is:

```
τ · log( Σᵢ exp(xᵢ/τ) )  →  max(xᵢ)    as τ → 0
```

In quantum mechanics terms: `τ` plays the role of Planck's constant `ℏ`. The WKB
approximation takes `ℏ → 0` and recovers classical geometric optics from wave optics.
Here, taking `τ → 0` recovers the tropical max from the smooth log-sum-exp.

The gradient of `LSE_τ` with respect to its inputs is `softmax(x/τ)`:
- `τ → 0`: softmax → argmax (one-hot, tropical limit)
- `τ = 1`: standard softmax
- `τ → ∞`: softmax → uniform distribution (maximum entropy, no routing)

**Implementation:** `τ` is parameterized as `exp(log_τ)` to keep it positive under
unconstrained optimization. One `τ` per attention head. Clamped to `[0.02, 10.0]` to
prevent numerical issues.

**Diagnostic value:** After training, the per-head `τ` values tell you what each head
learned to do:
- `τ < 0.4`: head is doing hard token routing (tropical regime)
- `0.4 ≤ τ ≤ 1.5`: head is doing standard contextual attention
- `τ > 1.5`: head is attending diffusely / averaging

The `maslov_summary()` method on `TropFormer` prints these values post-training.

**Connection to optimal control:** The temperature `τ` is the regularization parameter
in the entropy-regularized optimal control problem:
```
V(x) = max_u [ r(x,u) + τ · H(π) + γ·V(x') ]
```
where `H(π)` is the policy entropy. At `τ=0`, this recovers the hard Bellman equation.
At `τ=1`, this is the soft Bellman equation (MaxEnt RL). The learnable `τ` allows each
head to find its own optimal point on this spectrum.

---

### 4d. `LFDualActivation` — Legendre-Fenchel duality as activation

The Legendre-Fenchel (LF) conjugate of a convex function `f` is:
```
f*(y) = sup_x { ⟨x, y⟩ − f(x) }
```

This is the same transform that connects the Lagrangian to the Hamiltonian:
```
H(q, p) = sup_{q̇} [ p·q̇ − L(q, q̇) ]
```

In TropFormer, `f(x) = max_k(sₖ·x + bₖ)` is a tropical polynomial — the upper
envelope of affine pieces. It is a piecewise-linear convex function. Its LF conjugate,
evaluated at a grid of x values `{x_j}`, is:
```
f*(y) = max_j { x_j · y − f(x_j) }
```

**Key fact:** The LF conjugate of a tropical polynomial is another tropical polynomial
in the dual variable. Tropical algebra is self-dual under the LF transform. The primal
partitions x-space by which slope index k wins (Newton polytope faces). The dual
partitions y-space by which grid point x_j wins (dual Newton polytope faces).

**Modes:**
- `'primal'`: apply `f(x)` only. Partitions features by slope regime.
- `'dual'`: apply `f*(x)` only. Partitions features by conjugate grid point.
- `'blend'`: `σ(g) · f(x) + (1 − σ(g)) · f*(x)`. Learned per-channel mix.

**`blend` is the recommended mode.** It gives the network access to both polyhedral
decompositions and lets the optimizer decide which is more useful per feature.

**Parameters:**
- `slopes` and `biases`: the `K` affine pieces of the primal
- `x_grid`: the `K` evaluation points for the dual
- `blend_gate` (blend mode only): per-channel sigmoid gate

**Initialization:** Slopes are linearly spaced over `[-1.5, 1.5]`. Grid is linearly
spaced over `[-3, 3]`. This gives good coverage initially. The optimizer will reshape
both distributions during training.

---

### 4e. `TropicalMultiHeadAttention` — tropical + classical score fusion

Standard scaled dot-product attention computes:
```
score(q, k) = (q · k) / √d_k        [Euclidean inner product]
```

TropFormer replaces this with a blend of two score functions:

**Tropical score** (max-plus inner product):
```
trop_score(q, k) = max_i(q_i + k_i) / √d_k
```
Asks: what is the single feature dimension on which q and k most agree, and by how much?
This is a *hard compatibility measure* — only the peak-aligned dimension contributes.

**Classical score** (Euclidean dot product):
```
class_score(q, k) = (q · k) / √d_k
```
Accumulates evidence from all feature dimensions equally.

**Blended score:**
```
score = g(q) · trop_score + (1 − g(q)) · class_score
```
where `g(q) = σ(W_gate · q_input)` is a per-head, per-query-position gate. The gate
is conditioned on the raw pre-split input so it has access to full context.

**Q, K projections:** Both use `TropicalLinear`. This means the queries and keys
themselves are max-plus functions of the input — they are "routing" projections that
emphasize the most activated input dimensions rather than computing weighted sums.

**V projection:** Classical `nn.Linear`. Values must be smooth and dense to carry rich
gradient signal back through the attention aggregation. Using tropical V would produce
too-sparse value vectors.

**Output projection:** Classical `nn.Linear`. Standard.

**Maslov softmax:** The blended scores are passed through `MaslovTemperature` with
per-head `τ`. This applies `softmax(score/τ)` with the learned temperature.

---

### 4f. `TropicalHybridFFN` — parallel branches with LF activation

Structure:
```
x → TropicalLinear → LFDualActivation → TropicalDropout  ─┐
x → nn.Linear      → GELU                                  ├─ GatedFusion → down_proj → norm + residual
gate = σ(W_gate · x) ──────────────────────────────────────┘
```

The gate is conditioned on the *input* `x` (before either branch), not on the branch
outputs. This is intentional: the gate needs to make a routing decision based on
unmodified input information, not information already shaped by one of the branches.

**Residual connection:** The `TropicalHybridFFN` handles its own residual internally
(`norm(out + x)`). Do not add an external residual around this module.

---

### 4g. `TropicalTransformerBlock` — pre-norm architecture

Uses **pre-LayerNorm** (normalize before the sub-layer, not after), as in GPT-2/Llama.
This is critical for tropical layers: tropical subgradients are sparse and can be large
in magnitude. Post-norm architectures can amplify these into instability. Pre-norm
ensures the tropical layers always see normalized inputs.

Structure:
```
x → LayerNorm → TropicalMHA → + x    (attention, residual outside)
  → TropicalHybridFFN               (FFN handles residual internally)
```

---

### 4h. `TropFormer` — full model

**Patch embedding:** 7×7 patches on 28×28 MNIST → 16 patches of dimension 49 →
embedded to `d_model=128`. The patch embedding is a classical `nn.Linear` — keeping
the embedding classical ensures clean gradient flow from the very first layer.

**CLS token:** Prepended to the token sequence. Classification is performed on this
token after all transformer blocks. Standard ViT approach.

**Positional encoding:** Learnable `nn.Parameter` of shape `(1, num_patches+1, d_model)`.
Initialized with `trunc_normal_(std=0.02)`.

**Default hyperparameters (MNIST):**
```
img_size=28, patch_size=7, in_channels=1, num_classes=10
d_model=128, num_heads=4, num_layers=4, ffn_dim=256
dropout=0.1, trop_dropout=0.05
lf_pieces=8, lf_mode='blend'
init_temp=1.0
```

---

## 5. Training Configuration

### Optimizer
`AdamW` with `weight_decay=1e-4`. The combination of tropical subgradients (sparse)
and classical gradients (dense) means the optimizer sees mixed gradient statistics.
AdamW's per-parameter adaptive step sizes handle this well. SGD is not recommended.

### Scheduler
`OneCycleLR` with 10% warmup and cosine annealing. The warmup is important: tropical
layers need a few steps to find stable routing configurations before the learning rate
peaks. Without warmup, early large steps can lock the network into bad polytope
configurations before it has explored the space.

### Gradient clipping
`clip_grad_norm_(parameters, 1.0)` is applied every step. Tropical subgradients can
occasionally be large (when a routing configuration changes suddenly). Clipping prevents
these from causing instability.

### Mixed precision
AMP (`torch.cuda.amp`) is supported on CUDA. Not used on CPU. The `scaler` argument to
`train_epoch` handles this — pass `None` on CPU.

### Expected performance on MNIST
- ~10 epochs to reach ~97% test accuracy
- ~25 epochs to reach ~98.2–98.6% test accuracy
- Comparable to a standard small ViT, which validates the architecture is training
  correctly (tropical routing is not harming performance)

---

## 6. Diagnostic Tools Built Into the Model

Three post-training analysis methods are available on the `TropFormer` instance:

### `model.maslov_summary()`
Returns per-block, per-head Maslov temperature `τ` values.
```
τ < 0.4  → tropical/argmax regime (hard routing)
0.4–1.5  → standard softmax regime
τ > 1.5  → diffuse/uniform regime
```
Interesting diagnostic: look for specialization across layers. Early layers often stay
near `τ=1` (they need contextual smoothing to build representations). Late layers
sometimes converge toward lower `τ` (they can afford to make harder routing decisions
once representations are established).

### `model.lf_mode_summary()`
Returns mean LF blend gate value per block.
```
→ 1.0  primal dominates (partitioning by slope regime)
→ 0.0  dual dominates  (partitioning by Newton polytope dual)
→ 0.5  equal blend
```

### `model.score_gate_summary()`
Returns mean absolute weight of the attention score gate per block.
```
High value → gate is decisive, strongly routing tropical vs classical
Low value  → gate is passive, outputting ~0.5 blend for all queries
```
A completely passive gate (all ~0.5) suggests the score gate is not contributing
meaningfully. Consider increasing its learning rate or changing initialization.

---

## 7. Known Trade-offs and Design Choices

| Decision | Choice made | Rationale | Alternative |
|---|---|---|---|
| Q/K projection | `TropicalLinear` | Hard routing in query/key space | Classical — less novel but more stable |
| V projection | Classical `nn.Linear` | Smooth gradient path through values | Tropical V — too sparse, hurts convergence |
| Gate conditioning | Input `x`, not branch outputs | Gate needs unbiased routing signal | Branch outputs — creates circular dependency |
| LF mode default | `'blend'` | Access to both polytope decompositions | `'primal'` only — simpler, slightly faster |
| Tropical dropout rate | 0.05 | Sparse subgradients can't afford high dropout | Higher — but destabilizes |
| Pre-norm vs post-norm | Pre-norm | Tropical subgradients are spiky | Post-norm — unstable with tropical layers |
| Patch embed | Classical | Clean gradient source | Tropical — experimental, may be worth trying |
| CLS vs mean pool | CLS token | Standard ViT, easy to interpret | Mean pool over tokens |

---

## 8. Extension Roadmap

The following extensions are planned or worth exploring, roughly in order of priority:

### 8a. Datasets beyond MNIST
To adapt TropFormer to other datasets, change:
- `img_size`, `patch_size`, `in_channels`, `num_classes` in `TropFormer.__init__`
- Update the `get_mnist_loaders` function or replace it entirely
- For CIFAR-10: `img_size=32, patch_size=8, in_channels=3, num_classes=10`
- For ImageNet: significant scaling required — increase `d_model`, `num_layers`, `num_heads`

### 8b. Pure classical baseline comparison
Add a `ClassicalTransformer` class with identical hyperparameters but using only
`nn.Linear` for Q/K/V and `nn.GELU` for FFN. Run both on the same seed and compare:
- Test accuracy curves
- Convergence speed
- Gradient statistics (tropical should show sparser gradients)
- Post-training routing analysis

This comparison is the key scientific experiment — it validates (or disproves) that the
tropical components are contributing positively.

### 8c. Routing visualization
After training, visualize which tropical routing configurations (winning j indices) are
active for each digit class. A heatmap of `argmax_j(W_ij + x_j)` per input, colored
by class label, would show whether the tropical layers are learning class-specific
routing paths — i.e., whether the PWA partition is semantically meaningful.

### 8d. Entmax (sparse softmax, Tsallis entropy)
The Maslov bridge interpolates between max (tropical) and softmax (classical) via the
`τ` temperature. A richer generalization uses the Tsallis α-entmax:
```
α = 1    →  softmax (Shannon entropy)
α = 2    →  sparsemax (L2 projection onto simplex)
α → ∞   →  argmax (tropical limit)
```
This gives a fundamentally different interpolation path with genuine sparsity at
intermediate α, rather than just concentration near a single peak. Worth adding as an
alternative to `MaslovTemperature`.

### 8e. Tropical control policy network
The most direct application of this architecture to the lead engineer's primary domain:
replace the classification head with a continuous action head and train as a policy
network in a piecewise-affine dynamical system environment. The tropical routing layers
would learn the mode partition of the PWA system directly from closed-loop data.
The Maslov temperature per layer would correspond to the regularization parameter in
soft Bellman iteration.

### 8f. Tropical eigenvalue monitoring
In timed event graphs and max-plus linear systems, the tropical eigenvalue `λ` satisfies
`A ⊗ v = λ ⊗ v`, i.e., `max_j(A_ij + v_j) = λ + v_i`. During training, computing the
tropical eigenvalue of each `TropicalLinear` weight matrix gives a measure of the
"throughput rate" of information flow through that layer — analogous to the spectral
radius in classical linear stability analysis. Add this as a training monitor.

### 8g. Tropical residual connection
Currently, residual connections use classical addition: `out = f(x) + x`. A tropical
residual would be: `out = max(f(x), x)` — take the better of the transformed and
identity paths element-wise. This is the tropical identity operation and may help in
very deep tropical networks where gradient flow through the tropical path degrades.

### 8h. Learnable `x_grid` in `LFDualActivation`
The dual grid points `x_j` are currently learnable parameters. An improvement would be
to initialize them at the *current* data quantiles rather than uniform spacing, and
potentially use a separate optimizer group with a lower learning rate for the grid.
This would give the LF dual a better initial approximation of the data distribution.

---

## 9. File Structure

```
tropformer.py              — main source file (all code in one module)
TROPFORMER_CONTEXT.md      — this document
tropformer_best.pt         — saved model checkpoint (best test accuracy)
data/                      — MNIST download cache (auto-created)
```

The entire implementation is intentionally in a single file for portability and
readability. If the project scales significantly, consider splitting into:
```
tropformer/
  __init__.py
  primitives.py     — TropicalLinear, TropicalDropout
  maslov.py         — MaslovTemperature
  lf_dual.py        — LFDualActivation
  attention.py      — TropicalMultiHeadAttention
  ffn.py            — TropicalHybridFFN
  model.py          — TropicalTransformerBlock, TropFormer
  train.py          — training loop, data loaders, main()
  diagnostics.py    — post-training analysis utilities
```

---

## 10. Dependency Notes

```
torch         >= 2.0   (uses torch.cuda.amp.GradScaler, DataLoader pin_memory)
torchvision   >= 0.15  (MNIST dataset loader)
python        >= 3.10  (uses X | Y union type hints in signatures)
```

If running on Python 3.9, replace `torch.Tensor | None` with `Optional[torch.Tensor]`
and add `from typing import Optional` at the top. No other compatibility issues expected.

CUDA is optional but strongly recommended for training beyond MNIST scale. The code
falls back to CPU automatically.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| Tropical semiring | (ℝ ∪ {−∞}, max, +) — the algebraic system replacing (+, ×) |
| Tropical addition | max(a, b) |
| Tropical multiplication | a + b |
| Tropical zero | −∞ (identity under max) |
| Tropical one | 0 (identity under tropical multiplication) |
| Tropical linear map | `y_i = max_j(W_ij + x_j)` |
| Tropical variety | The set of points where two or more terms tie for the max; geometrically, the boundary complex of the polyhedral partition |
| Newton polytope | Convex hull of the exponent vectors of a polynomial; in tropical setting, determines the combinatorial structure of the tropical variety |
| Max-plus inner product | `⟨q, k⟩_trop = max_i(q_i + k_i)` — single peak-aligned dimension |
| Legendre-Fenchel conjugate | `f*(y) = sup_x{⟨x,y⟩ − f(x)}` — same as Lagrangian→Hamiltonian transform |
| Maslov dequantization | The limit `τ·log(Σ exp(x/τ)) → max(x)` as `τ→0`; bridge from smooth to tropical |
| Maslov temperature | The parameter `τ`; analogous to `ℏ` in WKB / quantum-classical bridge |
| Piecewise-affine (PWA) system | Dynamical system with different affine modes active in different state-space regions — the control-theory equivalent of a tropical linear map |
| Subgradient | Generalization of gradient to non-differentiable convex functions; for max, it selects the gradient of the active piece |
| Gated fusion | `σ(Gx)·trop + (1−σ(Gx))·classical` — learned convex combination of both paths |
| CLS token | A learned "classification" token prepended to the input sequence; its output representation is used for classification |
| Pre-norm | LayerNorm applied before the sub-layer (not after residual) — more stable with sparse gradients |

---

*Last updated: generated from design session with lead engineer.*
*Reference script: `tropformer.py`*