# Tropical Path Planning Roadmap
## Tropicalization of the TSRN Architecture via Shortest-Path Optimization

*Status: Planning document — no implementation yet.*
*Target: Full tropicalization of the TSRN optimizer and architecture.*

---

## 1. Executive Summary

The TSRN architecture already uses tropical geometry in its attention scores
(logsumexp ≈ max-plus inner product) and sparse top-k selection.  This
roadmap extends the tropical paradigm **into the optimizer itself**, treating
gradient descent as **shortest-path computation on a tropical polytope**.

The key insight: every piecewise-linear (PWL) neural network layer defines a
tropical polynomial.  Composing layers yields a tropical rational map.  The
loss landscape is therefore a tropical hypersurface, and optimizing it is
equivalent to finding shortest paths in the **tropical dual graph** of that
hypersurface's cell decomposition.

---

## 2. Mathematical Foundations

### 2.1  Tropical Semiring Recap

The tropical semiring (ℝ ∪ {-∞}, ⊕, ⊗) with:
- a ⊕ b = max(a, b)     (tropical addition)
- a ⊗ b = a + b          (tropical multiplication)

All ReLU networks are tropical rational maps (Zhang et al., 2018).
All max-plus recurrences (our TropicalSSM) are tropical linear maps.

### 2.2  Loss Landscape as Tropical Hypersurface

Given parameters θ ∈ ℝ^n, the loss L(θ) for a PWL network is a
tropical polynomial in θ.  The **tropical variety** V(L) partitions
parameter space into convex cells (the linear regions of L).

The **dual graph** G = (V, E) has:
- V = linear regions of L
- E = (σ₁, σ₂) if regions share a codimension-1 face

**Optimization = shortest path in G** from current region to a
region with minimal loss value.

### 2.3  Maslov Dequantization Bridge

Classical:  sum_i exp(f_i(x)/ε)  →  as ε→0  →  max_i f_i(x)

This gives us a continuous interpolation:
- ε = 1:  standard softmax/logsumexp (current training)
- ε → 0:  pure tropical (max-plus, PWL)

The MaslovTemperature module already implements this.  The roadmap
extends it to the **optimizer** via a dequantization schedule.

---

## 3. Phase 1: Tropical Subgradient Descent

**Goal**: Replace classical gradients with tropical subgradients for
layers that are already tropical (TropicalAttention, TropicalSSM).

### 3.1  Theory

For a tropical polynomial f(x) = max_i(a_i + ⟨w_i, x⟩), the
tropical subgradient at x is the convex hull of {w_i : i ∈ argmax}.

This is a finite set — the **tropical gradient** is sparse by
construction, touching only the active linear piece.

### 3.2  Implementation Plan

```
TropicalSubgradient:
    1. Forward: compute f(x) = max_i(a_i + ⟨w_i, x⟩)
    2. Record which piece i* is active (argmax)
    3. Backward: return w_{i*} as the subgradient
    4. If multiple pieces tie: use random tie-breaking
       (stochastic tropical gradient)
```

### 3.3  Integration Points

- `TropicalAttention._attend_chunk`: replace logsumexp backward
  with argmax-of-channels backward (the "winning path")
- `TropicalSSM.forward`: the max() already computes the argmax
  implicitly; expose it as a straight-through estimator
- `CliffordFFN`: not tropical — leave as standard backprop

### 3.4  Expected Outcome

Sparser gradients → fewer parameter updates per step → faster
convergence for the tropical layers.  The non-tropical layers
(CliffordFFN, embeddings, LayerNorm) continue using AdamW.

---

## 4. Phase 2: Tropical Mirror Descent

**Goal**: Use the tropical semiring structure for a principled
optimizer that respects the geometry of max-plus spaces.

### 4.1  Theory

Mirror descent on the tropical polytope:
    θ_{t+1} = argmin_{θ ∈ Δ_T} { ⟨∇_T L(θ_t), θ⟩ + D_φ(θ, θ_t) }

where:
- ∇_T L is the tropical subgradient
- D_φ is the Bregman divergence w.r.t. the negative tropical entropy:
    φ(θ) = max_i θ_i - (1/n) Σ_i θ_i
  (the gap between max and mean — a tropical "variance")
- Δ_T is the tropical simplex: {θ : max_i θ_i = 0}

### 4.2  Implementation Plan

```
TropicalMirrorDescent(optimizer):
    For each parameter group:
        1. Compute tropical subgradient g_T
        2. Dual step: ψ_t = φ'(θ_t) - lr * g_T
        3. Mirror step: θ_{t+1} = (φ')^{-1}(ψ_t)
        4. Project onto tropical simplex if needed
```

### 4.3  Hybrid Strategy

- **Tropical layers** (TropicalAttention, TropicalSSM, RGPool):
  use TropicalMirrorDescent
- **Classical layers** (CliffordFFN, embeddings, LayerNorm):
  use AdamW
- **Bridge layers** (SheafDiffusion, GistExtractor):
  use weighted blend of both, controlled by MaslovTemperature

---

## 5. Phase 3: Shortest-Path Tropical Optimization

**Goal**: Reformulate the full training loop as a shortest-path
problem on the tropical dual graph of the loss landscape.

### 5.1  Theory

The cell complex of a tropical polynomial f has a dual graph
where edge weights are the loss differences between adjacent cells.
The optimal descent path minimizes:

    Path cost = Σ_{edges (u,v) in path} |L(u) - L(v)|

This is solvable in polynomial time via Dijkstra's algorithm on
the dual graph — the tropical analogue of gradient descent.

### 5.2  Practical Approximation

The exact dual graph is intractable for large networks.  We
approximate:

1. **Local cell enumeration**: For each mini-batch, enumerate the
   active linear regions by recording activation patterns.
2. **Cell adjacency**: Two cells are adjacent if they differ in
   exactly one activation (one ReLU/max flips).
3. **Shortest-path step**: Find the adjacent cell with lowest
   loss, move to its centroid.

```
TropicalPathOptimizer:
    For each mini-batch:
        1. Forward pass → record active pieces (binary code)
        2. Enumerate neighbors (flip each bit)
        3. For top-k neighbors, evaluate loss (forward-only)
        4. Move toward the best neighbor's centroid
        5. Fine-tune with AdamW within the new cell
```

### 5.3  Connection to TSRN Architecture

The RG coarse-graining in TSRN already performs a kind of
"cell merging" — adjacent fine cells are pooled into coarse cells.
This creates a natural **multi-resolution dual graph**:

- **Scale 1 (fine)**: Many small cells, local moves
- **Scale 2 (coarse)**: Fewer large cells, global moves

The optimizer can use the RG structure to:
1. Find coarse direction on Scale 2 (cheap)
2. Refine within the chosen coarse cell on Scale 1 (precise)

This is the **tropical multi-grid method**.

---

## 6. Phase 4: Tropical Geodesic Learning Rate

**Goal**: Replace the cosine annealing schedule with a schedule
derived from tropical geodesic distance.

### 6.1  Theory

The tropical metric on ℝ^n/ℝ·1 is:
    d_T(x, y) = max_i(x_i - y_i) - min_i(x_i - y_i)

The learning rate at step t should be proportional to the
tropical distance from current parameters to the nearest
local minimum:

    lr_t = α · d_T(θ_t, θ*) / d_T(θ_0, θ*)

This naturally decays to zero as we approach θ* and adapts
to the tropical geometry of the loss landscape.

### 6.2  Practical Approximation

Estimate d_T(θ_t, θ*) using the tropical variance of gradients:
    d_T ≈ max_i(g_i) - min_i(g_i)

This is cheap to compute and gives a natural "gap" measure
that goes to zero at critical points (where all subgradients
agree).

---

## 7. Phase 5: Full Architecture Tropicalization Map

Component-by-component plan for tropicalizing every module:

### 7.1  Already Tropical
| Module | Tropical Operation | Status |
|--------|-------------------|--------|
| TropicalAttention | logsumexp scores (≈ max-plus inner product) | ✅ In place |
| TropicalSSM | max(A+h, B+x) recurrence | ✅ New (this PR) |
| RGPool | Disentangle + pool (can be tropicalized) | 🔲 Phase 6 |

### 7.2  To Tropicalize
| Module | Current | Tropical Version | Priority |
|--------|---------|-------------------|----------|
| CliffordFFN | r²-i², 2ri, SwiGLU | Tropical Clifford: max(r+r, i+i) for grade-0, etc. | Medium |
| SheafDiffusion | Linear restriction maps | Tropical linear maps: max-plus matrix action | High |
| EchoStateReservoir | GRU gates (sigmoid) | Tropical gates: max(0, x) thresholding | Medium |
| PAdicMemory | Softmax retrieval | Tropical retrieval: argmax similarity | Low |
| PAdicAttention | Softmax over p-adic sim | Tropical softmax (hardmax in limit) | Medium |
| GistExtractor | Softmax pool | Tropical pool: max-weighted selection | High |
| GistBuffer | Tropical retrieval | ✅ Already tropical | Done |
| Embeddings | Learned lookup | Tropical embedding: max-plus projection | Low |
| LayerNorm | Mean/variance | Tropical norm: max - min (tropical variance) | Medium |

### 7.3  Tropicalization via Maslov Schedule

The MaslovTemperature parameter ε controls the classical↔tropical
interpolation.  The schedule:

```
Epoch 1-10:     ε = 1.0   (fully classical — stable training)
Epoch 11-30:    ε anneals 1.0 → 0.1  (gradually tropical)
Epoch 31-50:    ε = 0.1   (mostly tropical — exploit sparsity)
Epoch 51+:      ε anneals 0.1 → 0.01  (near-pure tropical)
```

Each module's forward pass uses:
```python
def tropical_forward(x, W, epsilon):
    # Classical: logsumexp(W + x) ≈ log(sum(exp(W + x)))
    # Tropical:  max(W + x)
    return epsilon * torch.logsumexp((W + x) / epsilon, dim=-1)
```

As ε → 0, this converges to max(W + x) pointwise.

---

## 8. Phase 6: Tropical RG Flow Optimization

**Goal**: Use the RG coarse-graining structure to define a
tropical renormalization flow in parameter space.

### 8.1  Theory

The RG transformation R: θ_fine → θ_coarse defines a flow in
parameter space.  Fixed points of this flow correspond to
scale-invariant models (critical points of the loss).

The tropical RG flow:
    θ^{(l+1)} = max(A_l ⊗ θ^{(l)})

where A_l is the transition matrix at scale l.

### 8.2  Multi-Scale Optimization

1. **Coarsen**: θ_coarse = RG(θ_fine)
2. **Optimize coarse**: θ*_coarse = argmin L_coarse(θ_coarse)
3. **Refine**: θ_fine = RG^{-1}(θ*_coarse)  (inverse RG with residual)
4. **Polish**: fine-tune θ_fine on L_fine

This is analogous to multigrid methods in numerical PDE solving,
but in tropical parameter space.

---

## 9. Implementation Priority & Timeline

| Phase | Description | Effort | Dependencies |
|-------|-------------|--------|-------------|
| 1 | Tropical subgradient | 1 week | None |
| 2 | Tropical mirror descent | 2 weeks | Phase 1 |
| 3 | Shortest-path optimization | 3 weeks | Phase 2 |
| 4 | Tropical geodesic LR | 1 week | Phase 1 |
| 5 | Full architecture tropicalization | 4 weeks | Phases 1-2 |
| 6 | Tropical RG flow optimization | 3 weeks | Phases 3, 5 |

**Total estimated effort: 14 weeks**

### Critical Path:
Phase 1 → Phase 2 → Phase 5 → Phase 6

### Quick Wins (can do immediately):
- Phase 4 (tropical geodesic LR) — standalone, 1 week
- Phase 1 (tropical subgradient) — foundational, 1 week

---

## 10. Validation Criteria

### 10.1  Per-Phase Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| 1 | Gradient sparsity of tropical layers | > 80% zeros |
| 2 | Convergence speed vs AdamW baseline | ≥ 1.2x faster |
| 3 | Loss reduction per compute-FLOP | ≥ 1.5x better |
| 4 | LR schedule tracks true distance | Pearson r > 0.8 |
| 5 | BPC on enwik8 with tropicalized modules | ≤ baseline BPC |
| 6 | Multi-scale optimization speedup | ≥ 2x fewer steps |

### 10.2  Overall Goal

Demonstrate that tropical optimization:
1. Achieves **same or better BPC** as classical AdamW
2. Uses **fewer gradient evaluations** (tropical sparsity)
3. Provides **interpretable optimization paths** (dual graph)
4. Naturally respects the **PWL structure** of the network

---

## 11. Theoretical Connections

### 11.1  To Optimal Transport
Tropical optimal transport (Kantorovich in max-plus):
    min_π max_{(i,j)} { c_{ij} + π_{ij} }

This connects our optimizer to Wasserstein distances between
parameter distributions at different training stages.

### 11.2  To Algebraic Geometry
The Newton polytope of a tropical polynomial determines its
combinatorial type.  As training progresses, the Newton polytope
of the loss function changes — this is the tropical analogue of
"phase transitions" in training.

### 11.3  To Control Theory
TSRN layers are piecewise-affine switched systems.  The tropical
optimizer selects the optimal switching sequence — this is a
tropical version of model predictive control (MPC).

---

## 12. References

- Zhang, L. et al. (2018). "Tropical Geometry of Deep Neural Networks"
- Maragos, P. et al. (2021). "Tropical Geometry and Machine Learning"
- Maslov, V. (1992). "Idempotent Analysis" (dequantization framework)
- Maclagan, D. & Sturmfels, B. (2015). "Introduction to Tropical Geometry"
- Joswig, M. (2021). "Essentials of Tropical Combinatorics"
- Akian, M. et al. (2012). "Tropical Linear Algebra" (max-plus spectral theory)
- Cohen, G. et al. (1999). "Max-Plus Algebra and System Theory"

---

*Document created: 2025-04-17*
*Next action: Implement Phase 1 (Tropical Subgradient) after validating current architectural changes.*
