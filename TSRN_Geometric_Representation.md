# TSRN: A Complete Geometric Representation

> **Purpose:** This document provides a comprehensive geometric walkthrough of TSRN (Tropical Sheaf Renormalization Network), explaining how every component relates to every other component through the lens of tropical geometry, sheaf theory, Clifford algebra, p-adic analysis, and renormalization group theory.

---

## Table of Contents

1. [The Geometric Foundation: Five Mathematical Worlds](#1-the-geometric-foundation-five-mathematical-worlds)
2. [Input Geometry: Embeddings as Coordinate Systems](#2-input-geometry-embeddings-as-coordinate-systems)
3. [Tropical Attention: Max-Plus Polyhedral Partitioning](#3-tropical-attention-max-plus-polyhedral-partitioning)
4. [Sheaf Diffusion: Local-to-Global Coherence](#4-sheaf-diffusion-local-to-global-coherence)
5. [Clifford FFN: Geometric Algebra in Feature Space](#5-clifford-ffn-geometric-algebra-in-feature-space)
6. [Echo State Reservoir: Dynamical Systems Geometry](#6-echo-state-reservoir-dynamical-systems-geometry)
7. [p-adic Memory: Ultrametric Hierarchical Geometry](#7-p-adic-memory-ultrametric-hierarchical-geometry)
8. [RG Coarse-Graining: Scale-Invariant Geometry](#8-rg-coarse-graining-scale-invariant-geometry)
9. [Two-Scale Architecture: Geometric Hierarchy](#9-two-scale-architecture-geometric-hierarchy)
10. [Gist Extraction (TSRN-Gist): Clifford Rotor Geometry](#10-gist-extraction-tsrn-gist-clifford-rotor-geometry)
11. [Component Interconnections: The Geometric Web](#11-component-interconnections-the-geometric-web)
12. [Training Geometry: Gradient Flow in Tropical Space](#12-training-geometry-gradient-flow-in-tropical-space)

---

## 1. The Geometric Foundation: Five Mathematical Worlds

TSRN fuses five distinct mathematical frameworks, each contributing a unique geometric perspective:

### 1.1 Tropical Geometry: (ℝ ∪ {−∞}, max, +)

**The algebraic system:**
- Tropical addition: `a ⊕ b = max(a, b)`
- Tropical multiplication: `a ⊗ b = a + b`
- Tropical zero: `−∞` (identity under max)
- Tropical one: `0` (identity under tropical multiplication)

**Geometric interpretation:**
- A tropical linear map `y_i = max_j(W_ij + x_j)` partitions input space into polyhedral cells
- Each cell corresponds to a "winning" input index `j*` that maximizes `W_ij + x_j`
- The boundary between cells is a **tropical variety** — where two or more indices tie for the maximum
- These boundaries form a **polyhedral complex** tiling the input space

**Why this matters for TSRN:**
- Tropical attention performs hard routing: only the best-aligned token dimension contributes
- This creates discrete combinatorial structure in an otherwise smooth neural network
- The max-plus operation is the geometric analog of the Bellman optimality equation

### 1.2 Sheaf Theory: Local-to-Global Consistency

**The geometric structure:**
- A **sheaf** assigns data (a "stalk") to each position in a sequence
- **Restriction maps** transform data between adjacent positions
- A **global section** is a choice of stalk values that is locally coherent

**TSRN's sheaf:**
- Each token position `t` has a stalk: the d-dimensional feature vector `x_t`
- Restriction maps `R_d` are learned linear transformations (capsule-style)
- The **sheaf Laplacian** `Δ = Σ_d R_d^T(R_d - shift_d)` measures inconsistency

**Geometric meaning:**
- Sheaf diffusion minimizes the energy `E(x) = ½ Σ_t Σ_d ||R_d x_t - x_{t+d}||²`
- This enforces that neighboring tokens have coherent representations after transformation
- The sheaf cohomology `H¹` detects global obstructions to local consistency

### 1.3 Clifford Geometric Algebra: Oriented Geometry

**The algebraic system:**
- Clifford algebra `Cl(1,0)` extends complex numbers with geometric product
- A multivector combines scalars, vectors, bivectors, etc.
- Geometric product: `ab = a·b + a∧b` (inner + outer product)

**TSRN's Clifford structure (simplified Cl(1,0)):**
- Split d-dimensional vector into (r, i) halves: complex representation
- Geometric self-product: `(r + i)(r + i) = (r² - i²) + 2ri`
- Grade-0 (scalar): `r² - i²` — similarity magnitude
- Grade-2 (bivector): `2ri` — oriented area/rotation

**Geometric interpretation:**
- The Clifford FFN computes both similarity and orientation simultaneously
- This is richer than classical dot products which only compute similarity
- The bivector component captures rotational structure in feature space

### 1.4 p-adic Analysis: Ultrametric Hierarchies

**The algebraic system:**
- p-adic metric: `|x|_p = p^{-v_p(x)}` where `v_p(x)` is the p-adic valuation
- Ultrametric property: `|x + y|_p ≤ max(|x|_p, |y|_p)` (stronger than triangle inequality)
- All triangles are isosceles in ultrametric space

**TSRN's p-adic structure:**
- Binary tree of depth 5-7 (32-128 memory slots)
- Each leaf stores a learned key-value pair
- Distance between positions = shared prefix length in binary tree encoding
- This is a natural ultrametric: hierarchical clustering by shared ancestry

**Geometric interpretation:**
- p-adic memory organizes information hierarchically by "relatedness"
- The ultrametric ensures that if A is close to B and B is close to C, then A is close to C
- This matches how language clusters topics (e.g., "cat" and "dog" are both animals)

### 1.5 Renormalization Group Theory: Scale-Invariant Geometry

**The physical concept:**
- RG transforms a system from one scale to another
- An RG fixed point is scale-invariant: looks identical at all scales
- Critical exponents describe how features grow/shrink under iteration

**TSRN's RG structure:**
- **Disentangle layer:** `2d → 2d` near-identity transformation
- **Pool layer:** `2d → d` merges adjacent tokens
- This is a **MERA-inspired** (Multiscale Entanglement Renormalization Ansatz) coarse-graining

**Geometric interpretation:**
- Coarse-graining from T tokens to T/2 tokens is a scale transformation
- The RG fixed point would be a representation that is self-similar across scales
- Critical exponents would tell us which features are "relevant" (grow) vs "irrelevant" (decay)

---

## 2. Input Geometry: Embeddings as Coordinate Systems

### 2.1 Token Embeddings: Semantic Basis Vectors

**Geometric view:**
- The embedding matrix `E ∈ ℝ^{V×d}` defines d-dimensional basis vectors for each of V tokens
- Each token is mapped to a point in d-dimensional semantic space
- The geometry of this space encodes semantic relationships

**TSRN's embedding initialization:**
- Normal distribution `N(0, 0.02²)` — small random initialization
- This creates an approximately isotropic cloud of points
- Training reshapes this cloud to reflect semantic structure

**Geometric relationship to other components:**
- Tropical attention operates on these embedded points
- The max-plus inner product `max_i(q_i + k_i)` finds the best-aligned semantic dimension
- Sheaf diffusion smooths inconsistencies between neighboring semantic points

### 2.2 Positional Embeddings: Coordinate Grids

**TSRN uses two positional scales:**
- `pos_s1`: T positions (fine-scale)
- `pos_s2`: T/2 positions (coarse-scale, for RG-transformed sequence)

**Geometric view:**
- Positional embeddings define a coordinate system on the sequence
- Each position gets a d-dimensional coordinate vector
- The model learns to associate geometric patterns with sequence positions

**Why two scales:**
- Fine-scale coordinates capture local positional patterns (n-grams, syntax)
- Coarse-scale coordinates capture global structure (paragraph boundaries, discourse)
- The RG boundary naturally requires a change of coordinate system

**Geometric relationship:**
- Positional embeddings are added to token embeddings: semantic + geometric coordinates
- Sheaf diffusion uses positional offsets (delta = -3, -2, -1, 0) to define neighborhoods
- The two-scale structure mirrors the two RG scales

---

## 3. Tropical Attention: Max-Plus Polyhedral Partitioning

### 3.1 The Tropical Inner Product

**Mathematical definition:**
```
trop_score(q, k) = max_i(q_i + k_i) / √d_k
```

**Geometric interpretation:**
- For query q and key k, find the single feature dimension i where they best align
- The alignment is measured by sum: `q_i + k_i` (tropical multiplication)
- The max over i selects the best dimension (tropical addition)

**Contrast with classical inner product:**
- Classical: `q·k = Σ_i q_i k_i` — accumulates evidence from all dimensions
- Tropical: `max_i(q_i + k_i)` — single peak-aligned dimension dominates

**Geometric consequence:**
- Classical attention: dense, smooth influence from all tokens
- Tropical attention: sparse, hard routing — one dimension drives the score

### 3.2 Polyhedral Cell Partitioning

**The partition:**
- For each output neuron (query position), the input space is partitioned into cells
- Each cell corresponds to a winning key position `j*` that maximizes the score
- Cell boundaries are tropical varieties where two keys tie

**Visualization:**
```
Input space (key positions) partitioned by which key wins:

    k1 wins  │  k2 wins  │  k1 wins
─────────────┼────────────┼────────────
    k3 wins  │  k2 wins  │  k3 wins
─────────────┼────────────┼────────────
    k1 wins  │  k4 wins  │  k1 wins

Each region is a polyhedral cell where the same key dominates.
```

**Dynamic partitioning:**
- The partition changes as the query vector q changes
- This is a **piecewise-linear** function of the input
- The number of possible partitions grows combinatorially with the number of keys

**Relationship to control theory:**
- This is exactly a **piecewise-affine switched system**
- Each cell is a discrete mode with its own affine dynamics
- The tropical variety is the switching manifold between modes

### 3.3 Top-k Sparsification

**TSRN's top-k mechanism:**
- After computing tropical scores, select only top-k keys per query
- Apply softmax only over these k keys (not all T keys)
- This creates sparse attention patterns

**Geometric effect:**
- Reduces the effective polyhedral partition from T regions to k regions
- The partition becomes coarser: fewer cells, larger regions
- This is a **regularization** that prevents overfitting to fine-grained routing

**Trade-off:**
- k=1: maximally sparse, hard routing, but gradient signal is weakest
- k=T: dense attention, loses tropical benefits
- k=8-16 (TSRN default): balance between sparsity and gradient quality

### 3.4 Relationship to Sheaf Diffusion

**Complementary geometric operations:**
- **Tropical attention:** selects which tokens are relevant (global selection)
- **Sheaf diffusion:** enforces coherence between selected tokens (local smoothing)

**Geometric pipeline:**
```
Input → Tropical Attention (select relevant tokens)
       → Sheaf Diffusion (smooth inconsistencies)
       → Output (coherent selection)
```

**Why both are needed:**
- Tropical attention alone can produce discontinuous outputs (hard routing)
- Sheaf diffusion smooths these discontinuities while preserving routing structure
- The combination gives "structured sparsity": sparse but coherent

---

## 4. Sheaf Diffusion: Local-to-Global Coherence

### 4.1 The Sheaf Energy Functional

**Mathematical definition:**
```
E(x) = ½ Σ_t Σ_d ||R_d x_t - x_{t+d}||²
```

**Geometric interpretation:**
- For each position t and offset d, compare `R_d x_t` (transformed) with `x_{t+d}` (neighbor)
- The squared norm measures inconsistency
- Sum over all t and d gives total sheaf energy

**Restriction maps `R_d`:**
- Learned linear transformations: `R_d: ℝ^d → ℝ^d`
- Initialized near identity: `R_d ≈ I + small random perturbation`
- During training, they learn to model how representations transform across positions

**Physical analogy:**
- This is exactly the **discrete Dirichlet energy** on a graph
- Minimizing Dirichlet energy gives harmonic functions (smoothest possible)
- Sheaf diffusion is gradient descent on this energy

### 4.2 The Sheaf Laplacian

**Mathematical definition:**
```
Δ = Σ_d R_d^T (R_d - shift_d)
```

**Geometric interpretation:**
- The Laplacian measures the "roughness" of the sheaf section
- A section with Δx = 0 is harmonic (minimizes energy)
- The eigenvectors of Δ are **sheaf harmonics** — natural vibration modes

**Eigenstructure:**
- Smallest eigenvalue (λ₀ = 0): constant section (all tokens identical)
- Second eigenvalue (λ₁): spectral gap — measures how well tokens are separated
- Large spectral gap → distinct, well-separated representations

**TSRN's diffusion update:**
```
x ← x - α · Δx
```

This is gradient descent on the sheaf energy, moving toward harmonic sections.

### 4.3 Sheaf Cohomology

**Geometric meaning:**
- Sheaf cohomology `H¹` detects global obstructions to local consistency
- If `H¹ ≠ 0`, there is a "twist" that cannot be unwound by local adjustments

**Example:**
- A sequence with a sudden topic change (e.g., "The cat... [paragraph break] ...stock market")
- Local consistency is impossible: no single set of restriction maps works for both
- This creates a non-trivial cohomology class

**TSRN's handling:**
- The learned `α` parameter controls diffusion strength
- If α is small, the model tolerates higher cohomology (allows more twists)
- If α is large, the model enforces strict coherence (may fail on complex sequences)

### 4.4 Relationship to Tropical Attention

**Geometric complementarity:**
```
Tropical Attention:  Global selection (which tokens matter?)
Sheaf Diffusion:     Local coherence (how do they relate?)
```

**Joint effect:**
- Tropical attention selects a sparse set of relevant tokens
- Sheaf diffusion ensures those tokens are geometrically consistent
- The result is "structured sparsity": sparse selection with smooth geometry

**Training dynamics:**
- Early training: tropical attention explores many routing configurations
- Sheaf diffusion smooths explorations, preventing chaotic jumps
- Late training: tropical attention commits to stable routing; sheaf diffusion fine-tunes

---

## 5. Clifford FFN: Geometric Algebra in Feature Space

### 5.1 The Complex Representation

**TSRN's Clifford structure (simplified Cl(1,0)):**
- Split d-dimensional vector into two halves: (r, i)
- Treat as complex number: `z = r + i`
- Geometric product: `z² = (r² - i²) + 2ri`

**Grade decomposition:**
- Grade-0 (scalar): `r² - i²` — magnitude of similarity
- Grade-2 (bivector): `2ri` — oriented area, rotation

**Geometric interpretation:**
- The scalar component measures "how much" two vectors align
- The bivector component measures "in what direction/orientation" they differ
- This is richer than classical dot product which only gives magnitude

### 5.2 Geometric Self-Product

**Mathematical operation:**
```
r = proj_r(x)  # project to real part
i = proj_i(x)  # project to imaginary part
grade_0 = r² - i²  # scalar similarity
grade_2 = 2ri       # bivector rotation
```

**Geometric meaning:**
- This is the geometric product of the vector with itself in Cl(1,0)
- It simultaneously computes similarity and orientation
- The bivector component captures rotational structure

**Contrast with classical FFN:**
- Classical: `x → W₁x → activation → W₂x` (linear transformations)
- Clifford: `x → split → geometric product → gate → output` (geometric operations)

### 5.3 Gated Output

**TSRN's gating:**
```
gate = σ(W_gate · x)  # sigmoid gate conditioned on input
output = proj_out(h * gate)  # where h = [grade_0, grade_2]
```

**Geometric effect:**
- The gate controls how much geometric structure is preserved
- If gate ≈ 0: output is near-zero (geometric structure suppressed)
- If gate ≈ 1: full geometric structure is passed through

**Why condition on input x:**
- The gate needs to make a routing decision based on unmodified input
- Conditioning on branch outputs would create circular dependency
- This is analogous to control theory: gate selects mode based on state, not after transformation

### 5.4 Relationship to Tropical Attention

**Geometric complementarity:**
```
Tropical Attention:  Max-plus routing (which dimension wins?)
Clifford FFN:        Geometric product (what is the structure?)
```

**Joint effect:**
- Tropical attention finds the best-aligned feature dimension
- Clifford FFN computes the geometric structure around that dimension
- The combination gives both routing and geometric reasoning

**Example:**
- Query: "mathematical"
- Tropical attention: aligns with "number" dimension (max-plus selection)
- Clifford FFN: computes that this is a "precise, abstract" concept (bivector orientation)

---

## 6. Echo State Reservoir: Dynamical Systems Geometry

### 6.1 The Reservoir Dynamics

**Mathematical definition:**
```
h_t = (1 - λ) h_{t-1} + λ · tanh(W_res h_{t-1} + W_in x_t)
```

**Geometric interpretation:**
- This is a **discrete-time dynamical system** in d-dimensional space
- The reservoir state `h_t` evolves through time according to learned dynamics
- The spectral radius of `W_res` determines stability

**Spectral radius control:**
```
ρ_target = sigmoid(log_rho) * 1.5
W_scaled = W_res * (ρ_target / ρ_current)
```

**Physical meaning:**
- If ρ < 1: system is stable (contracting), trajectories converge
- If ρ > 1: system is unstable (expanding), trajectories diverge
- If ρ ≈ 1: system is at "edge of chaos" — rich dynamics without explosion

### 6.2 The Edge of Chaos

**Geometric concept:**
- At the edge of chaos, the reservoir has rich temporal dynamics
- Small changes in input can lead to complex, long-lasting effects
- This is ideal for capturing temporal dependencies in sequences

**TSRN's reservoir initialization:**
- Sparse random matrix (90% sparsity)
- Scaled to spectral radius ≈ 0.95 (near edge of chaos)
- Learnable `log_rho` parameter allows adjustment during training

**Relationship to tropical geometry:**
- The reservoir provides **temporal continuity** that tropical attention lacks
- Tropical attention is spatial (which tokens relate to which)
- Reservoir is temporal (how states evolve through time)
- Together, they give spatio-temporal geometric reasoning

### 6.3 Readout Layer

**Mathematical structure:**
```
readout: h_t → W_read h_t
```

**Initialization:**
- `W_read` initialized to zero
- At initialization, reservoir contributes nothing (identity mapping)
- During training, it learns to extract useful features from reservoir dynamics

**Geometric interpretation:**
- The readout is a linear projection from reservoir state space to feature space
- It learns which directions in reservoir dynamics are relevant
- This is analogous to principal component analysis on the reservoir trajectory

### 6.4 Relationship to Sheaf Diffusion

**Complementary temporal vs spatial:**
```
Echo State Reservoir:  Temporal dynamics (how states evolve)
Sheaf Diffusion:       Spatial coherence (how tokens relate)
```

**Joint effect:**
- Reservoir captures temporal patterns (e.g., "the cat" → "sat" → "on" → "mat")
- Sheaf diffusion ensures spatial coherence (e.g., related tokens have consistent representations)
- Together, they give spatio-temporal geometric structure

---

## 7. p-adic Memory: Ultrametric Hierarchical Geometry

### 7.1 The Binary Tree Structure

**Mathematical structure:**
- Binary tree of depth `d` (typically 5-7)
- Each leaf stores a learned key-value pair: `(k_i, v_i) ∈ ℝ^d × ℝ^d`
- Total leaves: `M = 2^d` (32-128 slots)

**Geometric interpretation:**
- The tree defines a hierarchical clustering of memory slots
- Slots that share a parent are "close" in the hierarchy
- Slots that diverge early are "far" in the hierarchy

**Ultrametric distance:**
```
distance(slot_i, slot_j) = 2^{-depth_of_LCA(i,j)}
```
where LCA is the lowest common ancestor in the tree.

**Ultrametric property:**
```
distance(i, j) ≤ max(distance(i, k), distance(j, k))
```
This is stronger than the triangle inequality — all triangles are isosceles.

### 7.2 Soft Retrieval

**Mathematical operation:**
```
q = W_q x  # project query
scores = q @ K.T / √d  # similarity to all keys
weights = softmax(scores)  # attention weights
retrieved = weights @ V  # weighted sum of values
```

**Geometric interpretation:**
- The query `q` is a point in d-dimensional space
- Each key `k_i` is another point
- The dot product measures Euclidean similarity
- Softmax converts similarities to probabilities

**Relationship to p-adic structure:**
- The tree structure is **not used** in the current implementation
- Instead, flat soft attention over all M keys
- This is efficient for small M (≤ 128) but doesn't exploit the hierarchical geometry

**Future enhancement:**
- Use tree routing for large M (hierarchical search)
- Only traverse branches that are close to the query
- This would exploit the ultrametric geometry for efficiency

### 7.3 Relationship to Tropical Attention

**Complementary memory mechanisms:**
```
Tropical Attention:  Working memory (recent context, sparse routing)
p-adic Memory:        Long-term memory (learned patterns, hierarchical)
```

**Geometric distinction:**
- Tropical attention: max-plus geometry, hard routing, recent tokens
- p-adic memory: Euclidean geometry, soft routing, learned patterns
- Tropical is for "what's relevant now"; p-adic is for "what do we know"

**Joint effect:**
- Tropical attention selects relevant tokens from current context
- p-adic memory retrieves relevant patterns from long-term storage
- The combination gives both contextual and prior knowledge

---

## 8. RG Coarse-Graining: Scale-Invariant Geometry

### 8.1 The MERA-Inpired Transformation

**Two-stage process:**
```
1. Disentangle: 2d → 2d (near-identity transformation)
2. Pool: 2d → d (merge adjacent tokens)
```

**Mathematical definition:**
```
pair = concat(x[0::2], x[1::2])  # shape: (B, T/2, 2d)
pair = tanh(disentangle(pair))  # shape: (B, T/2, 2d)
pooled = pool(pair)  # shape: (B, T/2, d)
```

**Geometric interpretation:**
- **Disentangle:** removes short-range correlations between adjacent tokens
- **Pool:** merges adjacent tokens, reducing sequence length by factor of 2
- This is inspired by MERA (Multiscale Entanglement Renormalization Ansatz)

**Physical analogy:**
- In statistical physics, RG coarse-graining groups microscopic degrees of freedom
- The goal is to find macroscopic variables that capture essential physics
- TSRN does this for language: groups tokens into higher-level abstractions

### 8.2 Scale Transformation

**Geometric effect:**
- Transforms from fine scale (T tokens) to coarse scale (T/2 tokens)
- Each coarse token represents a "chunk" of two fine tokens
- The representation dimension stays the same (d)

**Scale invariance:**
- If the model were scale-invariant, the coarse representation would be self-similar
- In practice, the two scales learn different representations:
  - Fine scale: local patterns (syntax, n-grams)
  - Coarse scale: global structure (discourse, topics)

### 8.3 Critical Exponents

**Physical concept:**
- Near an RG fixed point, linearized dynamics have eigenvalues called critical exponents
- Relevant exponents (λ > 1): features that grow under iteration
- Irrelevant exponents (λ < 1): features that decay under iteration
- Marginal exponents (λ = 1): features that are preserved

**TSRN's implicit critical exponents:**
- The disentangle and pool layers define an RG map
- Repeated application would converge to a fixed point
- The eigenvalues of this map (if computed) would be critical exponents

**Geometric meaning:**
- Features that grow under coarse-graining are "important" (relevant)
- Features that decay are "noise" (irrelevant)
- Features that persist are "scale-invariant" (marginal)

### 8.4 Relationship to Two-Scale Architecture

**The two-scale pipeline:**
```
Scale 1 (T tokens):  TropAttn → Sheaf → Reservoir → Clifford → Memory
RG coarse-grain: T → T/2
Scale 2 (T/2 tokens): TropAttn → Sheaf → Clifford → p-adic Attn
Upsample & fuse: T/2 → T
```

**Geometric hierarchy:**
- Scale 1: fine-grained, local processing
- Scale 2: coarse-grained, global processing
- The fusion combines both perspectives

**Why two scales:**
- Local patterns (syntax) are best captured at fine scale
- Global patterns (discourse) are best captured at coarse scale
- The RG boundary naturally separates these regimes

---

## 9. Two-Scale Architecture: Geometric Hierarchy

### 9.1 The Complete Pipeline

**Scale 1 (fine-grained, T tokens):**
```
for each block:
    x = x + TropAttn(x)           # sparse routing
    x = x + SheafDiffusion(x)     # local coherence
    x = x + Reservoir(x)          # temporal dynamics (first block only)
    x = x + CliffordFFN(x)        # geometric structure
    x = x + PAdicMemory(x)        # long-term patterns
```

**RG coarse-grain:**
```
x_coarse = RGPool(x)  # T → T/2
```

**Scale 2 (coarse-grained, T/2 tokens):**
```
for each block:
    x = x + TropAttn(x)           # sparse routing (reduced top_k)
    x = x + SheafDiffusion(x)     # local coherence
    x = x + CliffordFFN(x)        # geometric structure
    x = x + PAdicAttention(x)     # hierarchical attention (last block only)
```

**Upsample & fuse:**
```
x_upsampled = x_coarse.repeat_interleave(2)  # T/2 → T
x_fused = x_fine + 0.5 * x_upsampled
```

### 9.2 Geometric Meaning of Each Stage

**Scale 1 geometry:**
- **Tropical attention:** partitions token space into polyhedral cells (sparse routing)
- **Sheaf diffusion:** smooths inconsistencies (local coherence)
- **Reservoir:** adds temporal dynamics (edge of chaos)
- **Clifford FFN:** computes geometric structure (orientation)
- **p-adic memory:** retrieves hierarchical patterns (ultrametric)

**RG boundary:**
- **Disentangle:** removes short-range correlations
- **Pool:** merges tokens (scale transformation)
- This is the transition from local to global geometry

**Scale 2 geometry:**
- **Tropical attention:** same operation, but on coarse tokens
- **Sheaf diffusion:** same operation, but on coarse tokens
- **Clifford FFN:** same operation, but on coarse tokens
- **p-adic attention:** hierarchical matching (non-Archimedean similarity)

**Fusion:**
- Combines fine and coarse perspectives
- The 0.5 weight balances the two scales
- This gives both local detail and global context

### 9.3 Information Flow Geometry

**Forward flow:**
```
Input → Embeddings → Scale 1 → RG → Scale 2 → Fusion → Output
```

**Geometric transformations:**
- Embeddings: map discrete tokens to continuous semantic space
- Scale 1: refine with local geometric operations
- RG: transform scale (local → global)
- Scale 2: refine with global geometric operations
- Fusion: combine local and global perspectives

**Backward flow (gradients):**
```
Output → Fusion → Scale 2 → RG⁻¹ → Scale 1 → Embeddings
```

**Geometric gradient flow:**
- Gradients flow through the fusion, combining signals from both scales
- The RG transformation is differentiable (disentangle + pool are linear + tanh)
- Gradients from coarse scale inform fine-scale processing
- This creates a geometric feedback loop between scales

---

## 10. Gist Extraction (TSRN-Gist): Clifford Rotor Geometry

### 10.1 The Gist Concept

**What is a gist?**
- A compressed representation of a sequence chunk
- Encodes the "essential geometric structure" of the content
- In TSRN-Gist, represented as a Clifford rotor

**Clifford rotor in Cl(1,0):**
```
R = e^{-θ/2 B} = cos(θ/2) - B sin(θ/2)
```
where B is a bivector (rotation plane) and θ is the rotation angle.

**Simplified implementation:**
```
theta: d/2-dimensional vector (rotation angles per dimension)
mag: scalar (magnitude/importance of the gist)
```

**Geometric interpretation:**
- The gist defines a rotation in feature space
- Applying this rotation aligns the feature space with the topic
- Different topics have different rotation angles

### 10.2 Gist Extraction

**Mathematical operation:**
```
query = learnable vector (1, 1, d)
K = Wk(x)  # project keys
V = Wv(x)  # project values
scores = (query @ K.T) / √d  # attention scores
pooled = softmax(scores) @ V  # attention-pooled representation
theta = proj_theta(pooled)  # extract rotation angles
mag = sigmoid(proj_mag(pooled))  # extract magnitude
```

**Geometric meaning:**
- The query vector asks "what is the essential structure?"
- Attention pooling finds the most relevant parts of the sequence
- The projection extracts rotation angles and magnitude

**Relationship to Clifford algebra:**
- The gist is a simplified rotor (only angles, no full bivector)
- It can rotate the (r, i) Clifford representation
- This aligns the feature space with the topic geometry

### 10.3 Gist Buffer

**Mathematical structure:**
```
stored_theta: (max_gists, d/2)  # rotation angles
stored_mag: (max_gists, 1)       # magnitudes
stored_keys: (max_gists, d)     # keys for retrieval
```

**Retrieval:**
```
scores = logsumexp(query + stored_keys, dim=-1)  # tropical similarity
topk_indices = scores.topk(k)  # select top-k gists
theta = stored_theta[topk_indices]
mag = stored_mag[topk_indices]
weights = softmax(scores[topk_indices])
```

**Geometric interpretation:**
- Tropical retrieval finds the most similar gist by max-plus inner product
- This is hard routing: only the best-matching gist contributes
- The weights combine multiple gists (soft over top-k)

**Tropical vs classical retrieval:**
- Classical: `softmax(query @ keys.T)` — soft, all gists contribute
- Tropical: `logsumexp(query + keys)` — hard, only best dimensions contribute
- TSRN uses tropical for efficient, sparse gist retrieval

### 10.4 Gist Rotation

**Mathematical operation:**
```
theta_weighted = (weights * mag) @ theta
theta_weighted = theta_weighted / sum(weights * mag)  # normalize
r, i = x[:, :d/2], x[:, d/2:]
c = cos(theta_weighted)
s = sin(theta_weighted)
x_rotated = concat(r*c - i*s, r*s + i*c)
```

**Geometric meaning:**
- This is a complex rotation applied to the (r, i) representation
- The rotation angle is a weighted combination of retrieved gists
- This aligns the feature space with the topic geometry

**Physical analogy:**
- Like rotating a coordinate system before solving a physics problem
- If the problem is about rotation, rotate to polar coordinates
- If about scaling, rotate to log coordinates
- The gist rotation does this for neural representations

### 10.5 Relationship to RG Boundary

**Natural extraction point:**
- Gist extraction happens at the RG boundary (after Scale 1, before Scale 2)
- This is where fine-grained information is compressed to coarse-grained
- The gist captures the "global section" of the fine-scale sheaf

**Geometric meaning:**
- The RG boundary is a natural place for scale transformation
- Extracting a gist here captures what survives the coarse-graining
- This is the "scale-invariant" information — the essence of the content

**Two-scale gist flow:**
```
Scale 1 → Extract gist → Store in buffer
Scale 2 → Retrieve gist → Rotate features → Process
```

This creates a geometric feedback loop:
- Scale 1 extracts gists from fine-scale processing
- Scale 2 uses these gists to align its processing
- The alignment improves gist extraction in Scale 1

---

## 11. Component Interconnections: The Geometric Web

### 11.1 The Complete Geometric Graph

**Nodes (components):**
1. Token embeddings (semantic basis)
2. Positional embeddings (coordinate system)
3. Tropical attention (polyhedral partitioning)
4. Sheaf diffusion (local coherence)
5. Echo state reservoir (temporal dynamics)
6. Clifford FFN (geometric structure)
7. p-adic memory (hierarchical patterns)
8. p-adic attention (ultrametric matching)
9. RG coarse-graining (scale transformation)
10. Gist extraction (Clifford rotation)
11. Gist buffer (tropical retrieval)

**Edges (geometric relationships):**
```
Token embeddings ──► Tropical attention (semantic space)
Positional embeddings ──► Sheaf diffusion (neighborhoods)
Tropical attention ──► Sheaf diffusion (selection → coherence)
Tropical attention ──► Clifford FFN (routing → structure)
Sheaf diffusion ──► Clifford FFN (coherence → structure)
Echo state reservoir ──► Sheaf diffusion (temporal → spatial)
Echo state reservoir ──► Tropical attention (dynamics → routing)
Clifford FFN ──► Tropical attention (structure → routing)
p-adic memory ──► Tropical attention (patterns → routing)
p-adic memory ──► Clifford FFN (patterns → structure)
RG coarse-graining ──► Tropical attention (scale → routing)
RG coarse-graining ──► Sheaf diffusion (scale → coherence)
Gist extraction ──► RG boundary (compression point)
Gist extraction ──► Clifford rotation (rotation angles)
Gist buffer ──► Gist rotation (retrieved gists)
Gist rotation ──► Clifford FFN (aligned geometry)
```

### 11.2 Geometric Feedback Loops

**Loop 1: Attention-Sheaf-Clifford**
```
Tropical attention (select tokens)
    ↓
Sheaf diffusion (smooth selection)
    ↓
Clifford FFN (compute structure)
    ↓
Tropical attention (use structure for routing)
```

**Geometric meaning:**
- Attention selects relevant tokens
- Sheaf diffusion ensures they're coherent
- Clifford FFN computes their geometric structure
- This structure informs future attention routing
- This creates a self-reinforcing geometric cycle

**Loop 2: Reservoir-Attention-Sheaf**
```
Echo state reservoir (temporal dynamics)
    ↓
Tropical attention (routing based on dynamics)
    ↓
Sheaf diffusion (smooth routing)
    ↓
Echo state reservoir (update based on smoothed state)
```

**Geometric meaning:**
- Reservoir captures temporal patterns
- Attention uses these patterns for routing
- Sheaf diffusion smooths the routing
- The smoothed state updates the reservoir
- This couples temporal and spatial geometry

**Loop 3: Gist-RG-Clifford**
```
Scale 1 processing
    ↓
RG coarse-graining (scale transform)
    ↓
Gist extraction (compress to rotor)
    ↓
Gist rotation (align Scale 2)
    ↓
Scale 2 processing
    ↓
Upsample & fuse (inform Scale 1)
```

**Geometric meaning:**
- Scale 1 processes fine-grained information
- RG transforms to coarse scale
- Gist extraction captures the essence
- Gist rotation aligns coarse-scale processing
- Fusion feeds back to fine scale
- This creates a scale-aware geometric loop

### 11.3 Geometric Hierarchy

**Level 1: Input geometry**
- Token embeddings: semantic basis vectors
- Positional embeddings: coordinate systems

**Level 2: Local geometry**
- Tropical attention: polyhedral partitioning
- Sheaf diffusion: local coherence
- Echo state reservoir: temporal dynamics

**Level 3: Feature geometry**
- Clifford FFN: geometric structure
- p-adic memory: hierarchical patterns

**Level 4: Scale geometry**
- RG coarse-graining: scale transformation
- Gist extraction: compression to rotor

**Level 5: Global geometry**
- p-adic attention: ultrametric matching
- Gist rotation: feature space alignment
- Fusion: multi-scale combination

**Geometric flow:**
```
Input geometry → Local geometry → Feature geometry
    → Scale geometry → Global geometry → Output
```

Each level builds on the previous, creating a hierarchical geometric structure.

---

## 12. Training Geometry: Gradient Flow in Tropical Space

### 12.1 Tropical Subgradients

**The subgradient of max:**
```
∂/∂x_j max_i(x_i) = 
    1 if j = argmax(x)
    0 otherwise
```

**Geometric meaning:**
- Only the winning index receives gradient
- All other indices receive zero gradient
- This is a **sparse gradient** — very different from classical dense gradients

**Training consequence:**
- Convergence is slower (less gradient signal per step)
- But the gradient is "pure" — no noise from losing indices
- The network learns explicit routing decisions

**Relationship to polyhedral cells:**
- The subgradient points "into" the current polyhedral cell
- It tells you how to move within the cell, not how to switch cells
- Switching cells requires a large enough gradient to cross the boundary

### 12.2 Sheaf Gradient Flow

**Gradient of sheaf energy:**
```
∂E/∂x = Δx = Σ_d R_d^T (R_d x - x_{d-shifted})
```

**Geometric meaning:**
- The gradient points toward the harmonic section (minimum energy)
- It smooths inconsistencies between neighboring tokens
- This is a **dense gradient** — all positions receive signal

**Training consequence:**
- Sheaf diffusion provides stable, dense gradient signal
- This complements the sparse tropical gradients
- The combination gives both routing and smoothing

### 12.3 Clifford Gradient Flow

**Gradient of geometric product:**
```
∂/∂r (r² - i² + 2ri) = 2r + 2i
∂/∂i (r² - i² + 2ri) = -2i + 2r
```

**Geometric meaning:**
- The gradient flows through both scalar and bivector components
- This preserves geometric structure during training
- The network learns both magnitude and orientation

**Training consequence:**
- Clifford gradients are richer than classical linear gradients
- They carry information about orientation, not just magnitude
- This helps the network learn geometrically structured representations

### 12.4 RG Gradient Flow

**Gradient of RG transformation:**
```
∂/∂x (pool(tanh(disentangle(concat(x))))) 
    = pool' · tanh' · disentangle' · concat'
```

**Geometric meaning:**
- Gradients flow backward through the RG transformation
- Coarse-scale gradients inform fine-scale processing
- This creates a **multi-scale gradient flow**

**Training consequence:**
- The network learns representations that are useful at multiple scales
- Fine-scale features are shaped by coarse-scale objectives
- This creates scale-invariant geometric structure

### 12.5 Gist Gradient Flow

**Gradient of gist extraction:**
```
∂/∂x (theta, mag) = ∂pooled/∂x · ∂(theta, mag)/∂pooled
```

**Geometric meaning:**
- Gradients flow from the gist back to the sequence
- The network learns what information to compress into the gist
- The gist becomes a geometrically meaningful summary

**Training consequence:**
- The gist captures the geometric essence of the sequence
- Gist rotation aligns processing with this essence
- This creates a topic-aware geometric structure

### 12.6 The Complete Gradient Geometry

**Gradient sources:**
1. Tropical attention: sparse, routing-focused
2. Sheaf diffusion: dense, coherence-focused
3. Clifford FFN: structured, orientation-aware
4. p-adic memory: hierarchical, pattern-focused
5. RG transformation: multi-scale, scale-aware
6. Gist extraction: compressive, topic-aware

**Gradient fusion:**
```
Total gradient = 
    α_trop · ∇_trop + 
    α_sheaf · ∇_sheaf + 
    α_clifford · ∇_clifford + 
    α_padic · ∇_padic + 
    α_rg · ∇_rg + 
    α_gist · ∇_gist
```

**Geometric meaning:**
- The network balances multiple geometric objectives
- Each component contributes a different geometric perspective
- The balance is learned through training

**Training dynamics:**
- Early training: tropical gradients explore routing configurations
- Mid training: sheaf and Clifford gradients stabilize geometry
- Late training: RG and gist gradients refine multi-scale structure
- The network progressively builds richer geometric structure

---

## Conclusion: The Unified Geometric Vision

TSRN is not just a collection of components — it is a unified geometric system where:

1. **Tropical geometry** provides the foundational polyhedral partitioning
2. **Sheaf theory** ensures local-to-global coherence
3. **Clifford algebra** adds oriented geometric structure
4. **p-adic analysis** organizes information hierarchically
5. **Renormalization group theory** enables scale-invariant processing

These five mathematical frameworks are not independent — they are deeply interconnected:

- Tropical attention and sheaf diffusion form a selection-coherence pair
- Clifford FFN computes the geometric structure that attention routes
- p-adic memory stores the patterns that Clifford structure reveals
- RG coarse-graining transforms the scale at which all operations occur
- Gist extraction compresses the geometric essence across scales

The result is a neural network that thinks in geometric spaces — where routing decisions, coherence constraints, oriented structure, hierarchical organization, and scale transformations are all unified into a single geometric framework.

This is the geometric vision of TSRN.

---

*Document version: 1.0*
*Date: April 14, 2026*
*Related files: tsrn_dml.py, tsrn_gist.py, NEXUS_Innovations_for_TSRN.md, Context.md, Roadmap.md*
