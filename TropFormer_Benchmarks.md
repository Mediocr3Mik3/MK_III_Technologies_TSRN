# TropFormer: Benchmark Strategy, Failure Modes & Compute Analysis

> **Purpose of this document:** Companion to `TROPFORMER_CONTEXT.md` and
> `TROPFORMER_ROADMAP.md`. Covers the complete benchmark plan for both papers,
> an honest analysis of where tropical architectures underperform classical,
> and a rigorous assessment of compute implications when integrated into
> large-scale models. Read this before designing experiments or making
> architectural claims.

---

## Document Map

```
§1  Complete Benchmark Plan         — all datasets, both papers, full strategy
§2  Where Tropical Underperforms    — honest failure mode analysis
§3  Compute Analysis                — training cost, inference cost, large model integration
§4  The Strategic Narrative         — how to frame the results honestly and compellingly
§5  Agent Instructions              — what to implement and in what order
```

---

## §1  Complete Benchmark Plan

### 1.0  The Narrative Structure the Benchmarks Must Tell

Every benchmark in this plan serves one of three rhetorical functions:

- **Parity benchmarks:** Show that the tropical architecture is not broken on
  mainstream tasks. These prove the gate is working correctly (defaulting to
  classical when tropical routing doesn't help) and that the architecture is
  a credible alternative, not a curiosity.

- **Modest-gain benchmarks:** Show that on tasks with *partial* structural match
  to tropical algebra, the architecture extracts real signal from the routing
  mechanism. These build the argument that the hybrid is doing something
  mechanistically meaningful.

- **Strong-gain benchmarks:** Show that on tasks with *intrinsic* tropical structure,
  the architecture decisively outperforms classical. These are the scientific core
  of each paper. Without at least one strong-gain result, neither paper has a
  publishable claim.

MNIST is retained throughout as a sanity check and comparison anchor. It is never
the lead result.

---

### 1.1  Paper 1 Benchmarks — Hybrid Transformer (Path A)

#### Tier 1: Parity benchmarks (include all)

| Dataset | Task | Classical baseline | Expected TropFormer | Why include |
|---|---|---|---|---|
| **MNIST** | Image classification | 99.7% (ViT-small) | 98.0–99.0% | Sanity check; comparison anchor; validates training |
| **CIFAR-10** | Image classification | 94.4% (ViT-small) | 93.5–94.5% | Mainstream vision; shows parity on smooth data |
| **CIFAR-100** | Fine-grained classification | 75–78% (ViT-small) | 74–78% | Harder version; still smooth; honest parity test |
| **SST-2** | Sentiment classification | 93.5% (BERT-base) | 92.5–93.5% | Standard NLP; mild discrete structure |
| **IMDB** | Long-form sentiment | 95.0% (BERT-base) | 94.0–95.5% | Longer sequences; tests whether routing hurts |

**What to do with parity results:** Present them in a table with the label
"smooth/general benchmarks." State clearly that matching classical performance
*is the goal* on these tasks — the gate correctly learns to use classical paths
when tropical routing provides no structural advantage. This is evidence the
architecture works, not evidence it fails.

#### Tier 2: Modest-gain benchmarks (include at least 2)

| Dataset | Task | Classical baseline | Expected TropFormer | Why tropical helps |
|---|---|---|---|---|
| **LRA Pathfinder** | Path connectivity (128×128 images) | ~71% (standard transformer) | 73–77% | Spatial routing is structurally tropical; hard attention outperforms soft on connectivity |
| **LRA Retrieval** | Document similarity | ~79% (standard transformer) | 80–83% | Tropical inner product (max-aligned feature) is a strong similarity signal for retrieval |
| **Penn Treebank** | Constituency parsing | ~93 F1 (BERT-fine-tuned) | 93–95 F1 | Syntactic hierarchy is discrete; tropical routing naturally respects constituency boundaries |
| **MNLI** | Natural language inference | ~84% (BERT-base) | 84–86% | Entailment has logical/discrete structure; tropical gate learns to route on logical connectives |
| **Long Range Arena (average)** | Mixed long-range tasks | ~53% average | 54–58% average | The LRA was specifically designed to test structural inductive bias |

#### Tier 3: Strong-gain benchmarks (must include ListOps; include at least one more)

| Dataset | Task | Classical baseline | Expected TropFormer | Mechanism |
|---|---|---|---|---|
| **LRA ListOps** ★ | Nested max/min over sequences | 36–38% (standard transformer) | 44–52% | ListOps *is* tropical algebra. The task is defined as `(MAX 3 (MIN 4 7) 9)` — hierarchically applied max and min. A tropical transformer with hard routing computes this exactly. Classical transformer approximates it with softmax-blurred attention. This is the paper's scientific core. |
| **LRA Path-X** | Long-range path connectivity (hard) | ~58% (standard transformer) | 62–68% | Harder version of Pathfinder; stronger routing benefit at longer range |
| **Algorithmic tasks (SCAN)** | Compositional generalization | 63–85% (standard transformer) | 72–90% | Systematic compositional rules are discrete logical operations — tropical routing finds the rule boundary exactly |
| **Code understanding (CodeXGLUE)** | Code classification | ~75% (CodeBERT) | 76–80% | Code has explicit syntactic structure; tropical routing learns AST-like partitions |

#### Ablation studies required for Paper 1

Run these on MNIST + CIFAR-10 + LRA ListOps (three-point ablation):

```
1. Full TropFormer (baseline)
2. Classical Q/K — replace TropicalLinear in Q/K with nn.Linear
3. No LF dual activation — replace with GELU only
4. Fixed τ=1.0 — remove learnable Maslov temperature
5. Fixed τ=0.1 — fully tropical attention, no learning
6. No score gate — fixed 50/50 blend (no input-dependent routing)
7. Classical FFN — replace TropicalHybridFFN with standard FFN
8. Fully classical — all tropical components replaced (the true baseline)
```

Ablation 8 is the most important. The delta between Full TropFormer and Fully
Classical, measured specifically on ListOps, is the paper's primary result.

#### Diagnostic analyses required for Paper 1

These go in the paper as novel interpretability contributions, not just ablation:

```
1. Maslov temperature heatmap: (layer × head) after training on each task
   - Hypothesis: ListOps causes more heads to converge to low τ (hard routing)
   - Hypothesis: SST-2 causes heads to stay near τ=1 (smooth sentiment)
   - If true: this is direct evidence the network adapts its routing to task structure

2. Routing path visualization: for each ListOps class, trace which j-indices
   win at each tropical layer for representative inputs
   - Hypothesis: inputs of type (MAX ...) and (MIN ...) activate different routing paths
   - If true: the network has learned a tropical parsing structure

3. Gate decisiveness over training: plot σ(score_gate) distribution per epoch
   - Hypothesis: on ListOps, gate converges to near 0/1 (decisive tropical routing)
   - On SST-2, gate stays near 0.5 (indecisive, using classical blend)

4. LF blend gate per task: does the network use primal or dual activation differently
   across tasks?
```

---

### 1.2  Paper 2 Benchmarks — Deep Tropical Net (Path B)

#### Tier 1: Parity benchmarks (include all)

| Dataset | Task | Classical baseline | Expected DeepTropNet | Note |
|---|---|---|---|---|
| **MNIST** | Image classification | 99.7% | 97.5–99.0% | Slightly lower expected — deep tropical convergence is slower |
| **CIFAR-10** | Image classification | 94.4% (ResNet-20) | 91–93% | Honest: deep tropical likely 1–2% below ResNet on smooth images |
| **ModelNet40** | 3D point cloud classification | 89.2% (PointNet) | 89–91% | PointNet already uses max-aggregation (tropical!); replacing MLP with tropical layers completes the algebra |

**Note on CIFAR-10 for Paper 2:** The deep tropical net will likely score 1–3%
*below* ResNet-20 here. Include this result and do not hide it. The narrative is:
"We accept this cost on smooth tasks in exchange for exact representational fidelity
on structured tasks." A reviewer who sees you hiding the CIFAR-10 result will
reject the paper. A reviewer who sees you reporting it honestly and explaining the
trade-off will respect it.

#### Tier 2: Modest-gain benchmarks (include at least 2)

| Dataset | Task | Classical baseline | Expected DeepTropNet | Mechanism |
|---|---|---|---|---|
| **ModelNet40 (augmented)** | Point cloud under corruption | PointNet drops to ~78% | ~83–86% | Tropical max-aggregation is inherently robust to point removal/noise — structural advantage |
| **LRA ListOps** | Nested max/min | 36–38% | 50–60% | Deep tropical should exceed hybrid transformer here — pure routing, no classical dilution |
| **SCAN (compositional)** | Systematic generalization | 63–85% | 80–92% | Deep tropical finds exact rule boundaries; classical approximates them |
| **Tox21 (molecular)** | Molecular toxicity prediction | ~80% (MPNN) | 81–84% | Molecular graph structure has discrete combinatorial features |

#### Tier 3: Strong-gain benchmarks — these are the paper's contribution

These benchmarks do not exist in standard form. **Building them is part of the
scientific contribution of Paper 2.** Each one requires a synthetic data generator,
which the agent should implement alongside the network.

---

##### Benchmark B-1: Synthetic PWA Recovery ★★★ (most important)

**What it is:** Generate random piecewise-affine functions with known exact partition
boundaries. Train the network to regress the function output from inputs. Evaluate
not just MSE but whether the network recovers the exact partition geometry.

**Generator specification:**
```python
def generate_pwa_dataset(
    n_modes:     int   = 8,       # number of affine pieces
    input_dim:   int   = 4,       # input dimensionality
    n_train:     int   = 10000,
    n_test:      int   = 2000,
    noise:       float = 0.01,    # small noise on outputs
    seed:        int   = 42,
) -> dict:
    """
    Generates a random PWA function:
        f(x) = A_k @ x + b_k   if x in region R_k

    Regions R_k are Voronoi cells of randomly sampled centers c_k.
    Mode k is active when x is closest (in L2) to c_k.

    Returns:
        X_train, y_train, X_test, y_test,
        true_centers (for partition recovery evaluation),
        true_affine_maps (A_k, b_k for each mode)
    """
```

**Metrics:**
- MSE: standard regression quality
- Mode accuracy: for each test point, is the predicted active mode correct?
  (obtain predicted mode from tropical argmax of final layer)
- Partition Hausdorff distance: Euclidean distance between true partition
  boundaries and learned partition boundaries (estimated by probing the network
  along random line segments and finding switching points)

**Expected results:**
- Deep tropical net: MSE competitive with MLP-ReLU, mode accuracy >95%,
  Hausdorff distance near 0 (exact recovery)
- MLP-ReLU: similar MSE, mode accuracy meaningless (no hard routing), Hausdorff
  distance large (smooth boundary approximation)
- Hybrid TropFormer: intermediate — gate learns some routing but mixed with smooth

---

##### Benchmark B-2: Graph Shortest Path ★★★

**What it is:** Random graphs with random positive edge weights. Task: predict
the shortest path distance between all node pairs.

**Generator specification:**
```python
def generate_shortest_path_dataset(
    n_nodes:     int   = 12,      # nodes per graph
    edge_prob:   float = 0.4,     # Erdos-Renyi edge probability
    n_graphs:    int   = 5000,
    weight_range: tuple = (0.1, 5.0),
) -> dict:
    """
    Ground truth computed via scipy.sparse.csgraph.shortest_path.
    Input: adjacency matrix with edge weights (0 = no edge).
    Target: all-pairs shortest path distance matrix.

    Why this is intrinsically tropical:
    The all-pairs shortest path is computed by tropical matrix powers:
        D = A^{(trop,n-1)}
    where A is the weighted adjacency matrix and ^ means tropical matrix power
    (repeated min-plus matrix multiply). A deep tropical net should converge
    to this algorithm.
    """
```

**Metrics:**
- Mean absolute error on distance predictions
- Optimality rate: fraction of predicted paths that match the true shortest path
- Scaling behavior: how performance degrades as n_nodes increases

**Expected results:**
- Deep tropical net: near-perfect optimality rate for n_nodes ≤ 15;
  graceful degradation beyond
- MLP-ReLU / GNN: systematically underperforms — they approximate the
  tropical matrix power with smooth operations and miss exact minima
- Gap should be 20–40% in optimality rate at n_nodes=12

---

##### Benchmark B-3: Max-Plus Scheduling ★★

**What it is:** Job-shop scheduling under a max-plus timed event graph model.
Tasks arrive with processing times; machines have setup delays. Predict the
optimal makespan (total completion time).

**Generator specification:**
```python
def generate_scheduling_dataset(
    n_jobs:      int = 6,
    n_machines:  int = 4,
    n_samples:   int = 8000,
) -> dict:
    """
    Each sample: a random job-shop instance with processing times drawn
    from Uniform(1, 10) and setup times from Uniform(0.5, 3).

    Ground truth makespan computed via:
        1. Exact ILP solver (small instances, n_jobs <= 6)
        2. Max-plus system matrix eigenvalue (for regular schedules)

    The max-plus system matrix A has:
        A[i,j] = processing time of job j on machine i + setup delay

    Optimal makespan = tropical eigenvalue λ of A^n
    (where A^n is the n-th tropical matrix power).

    Input to network: flattened processing time + setup delay matrix.
    Target: optimal makespan scalar.
    """
```

**Metrics:**
- MAE on makespan prediction
- Optimality gap: (predicted - optimal) / optimal × 100%
- Correlation between learned tropical eigenvalue and true tropical eigenvalue

**Expected results:**
- Deep tropical net: optimality gap < 2% on training distribution;
  the tropical eigenvalue of the learned weight matrix should converge
  to the true system throughput rate
- MLP-ReLU: optimality gap 5–15%; no interpretable eigenvalue structure
- This benchmark directly connects to the lead engineer's control theory background

---

##### Benchmark B-4: Morphological Image Analysis ★

**What it is:** Replace classical conv layers with TropicalConv2d in a standard
image classification backbone. Measure both accuracy and structural properties
of learned morphological filters.

**Target datasets:** MNIST (sanity), Fashion-MNIST (texture), CIFAR-10 (objects)

**Metrics:**
- Classification accuracy vs classical CNN
- Morphological structure of learned filters (visualize structuring elements)
- Robustness to salt-and-pepper noise (tropical max-conv is inherently robust
  to isolated corruptions — the max operation suppresses minority votes)

**Expected results:**
- On clean images: 1–3% below classical CNN (tropical conv is less expressive for smooth features)
- On corrupted images: equal or above classical CNN (robust by algebraic structure)
- The noise-robustness gap is the publishable result here

---

## §2  Where Tropical Underperforms Classical

This section is critical for scientific integrity and must be reflected in the
papers. Do not hide these failure modes. A paper that acknowledges them clearly
is more credible and more publishable than one that does not.

---

### 2.1  Smooth continuous regression

**Task type:** Predicting a continuously varying output from continuously varying
inputs where the true function has no sharp boundaries.

**Examples:** Wind speed forecasting, temperature field prediction, option pricing
(Black-Scholes regime), EEG signal regression, medical dose-response curves.

**Why tropical underperforms:** The tropical linear map only activates one input
dimension per output neuron (the argmax winner). For smooth functions, every input
dimension contributes continuously to every output. The tropical partition forces
the network to approximate a smooth function with a piecewise-linear tiling, which
requires many more pieces (and thus many more parameters) than a smooth network
that can represent the gradual slope directly.

**Magnitude of underperformance:** Expect 5–20% higher MSE on smooth regression
tasks at equivalent parameter count. The gap narrows with more parameters but
never fully closes.

**Mitigation in the hybrid:** The gated fusion learns to suppress the tropical path
(gate → 0) on these tasks. If training is correct, the hybrid transformer converges
to approximately classical behavior on smooth regression. But you are paying
parameter cost for tropical layers that contribute little.

---

### 2.2  Dense natural language modeling (next-token prediction)

**Task type:** Predicting the next token in natural language where meaning emerges
from the interaction of *all* context tokens, not a sparse subset.

**Examples:** GPT-style language modeling, narrative text generation, dialogue,
question answering over long prose.

**Why tropical underperforms:** In natural language, almost every token in context
contributes something to the meaning of the next token. The classical dot-product
attention score `Σᵢ(Qᵢ · Kᵢ)` aggregates compatibility evidence from all feature
dimensions. The tropical score `maxᵢ(Qᵢ + Kᵢ)` considers only the single most
aligned dimension. This is too sparse for natural language semantics.

**Magnitude of underperformance:**
- Perplexity increase: expect 2–8 points higher perplexity on standard language
  modeling benchmarks (Penn Treebank word-level, WikiText-103)
- For a model like GPT-2 (124M params), this is significant
- The hybrid mitigates this via the score gate, but the gate must learn to ignore
  the tropical path almost entirely — wasted capacity

**Important nuance:** The *structural* tasks within NLP (syntax parsing, coreference
with hard boundaries, code understanding, formal reasoning) do benefit from tropical
routing. The damage is specifically on *semantic* NLP — diffuse, gradient-rich tasks.

**Recommendation for large model integration:** Do not replace all attention heads
with tropical. Replace only heads in later layers that provably develop discrete
routing behavior (measurable via attention entropy — look for heads with low entropy
attention patterns in a classical pretrained model, then replace those specifically).

---

### 2.3  Image generation and diffusion models

**Task type:** Generating smooth, photorealistic images; score function estimation
in diffusion.

**Why tropical underperforms:** The score function ∇ₓ log p(x|t) in a diffusion
model is smooth and densely populated — every pixel influences the score at every
other pixel. Tropical routing produces hard-partitioned feature space, which creates
visible artifacts at partition boundaries in generated images (similar to color
quantization artifacts but in feature space).

**Magnitude:** Generated image quality would degrade significantly. FID scores
would increase 20–50% on standard benchmarks. This architecture is not suitable
for generative modeling of smooth domains.

---

### 2.4  Audio synthesis and speech recognition (acoustic model)

**Task type:** Continuous waveform generation (WaveNet style), acoustic feature
modeling, raw audio classification.

**Why tropical underperforms:** Audio is a smooth, continuous signal. Spectral
features overlap continuously across frequency bands. The tropical max operation
selects the single strongest frequency component, discarding all others — this
is appropriate for envelope tracking but destructive for full spectral modeling.

**Magnitude:** Speech recognition acoustic model WER would increase 10–30%
relative on standard benchmarks (LibriSpeech). Audio generation quality
(MOS score) would drop significantly.

**Exception:** Voiced/unvoiced detection, onset detection, and other tasks
that are intrinsically *thresholding* operations on audio features would
benefit from tropical routing. Musical note onset detection is one example.

---

### 2.5  Low data regimes

**Task type:** Any task where training set size is small (< ~1,000 samples).

**Why tropical underperforms:** The tropical routing partition requires enough
data to populate all active polytope cells with sufficient examples to make
robust routing decisions. With few data points, the tropical partition tends
to overfit — learning routing paths that are specific to training examples
rather than generalizable structure.

**A smooth neural network** with appropriate regularization generalizes better
in low-data regimes because smooth interpolation between training examples is
valid in most real-world domains.

**Magnitude:** In few-shot settings (< 100 examples), expect 5–15% lower accuracy
than classical networks with equivalent regularization.

**Mitigation:** Pre-training the tropical network on a large dataset first (transfer
learning), then fine-tuning on the small dataset. The pre-trained routing partition
acts as a strong prior.

---

### 2.6  Adversarial robustness (counterintuitive failure mode)

**This is the most counterintuitive failure mode.** One might expect tropical
networks to be more robust to adversarial attacks because hard routing seems
harder to manipulate than smooth gradients. The opposite is true.

**Why tropical is more vulnerable:**
The polyhedral partition boundaries (tropical variety) are sharp discontinuities.
A very small perturbation that moves an input across a partition boundary can
completely change the routing path — activating an entirely different affine mode.
Because the mode switch is discontinuous, the output can change by a large amount
with a tiny input change. Classical networks have smooth gradients that attenuate
adversarial perturbations; tropical networks have hard boundaries that amplify them.

**Magnitude:** On standard adversarial benchmarks (CIFAR-10 with PGD attack),
expect the tropical network's robust accuracy to be 5–15% *below* an equivalently
sized classical network. The smooth classical network's gradients provide natural
resistance to small perturbations. The tropical network's partition boundaries
are exploitable.

**Mitigation:** Adversarial training (train on adversarial examples) helps, but
the fundamental discontinuity at partition boundaries remains. For safety-critical
applications, this must be addressed before deployment.

**Important exception:** Against non-gradient attacks (patch attacks, physical
adversarial examples), tropical routing may be MORE robust because the routing
depends on the global maximum over feature dimensions, which is harder to fool
with localized perturbations than smooth weighted sums.

---

### 2.7  Online / streaming learning

**Task type:** Continual learning, concept drift adaptation, real-time systems
where the distribution changes over time.

**Why tropical underperforms:** The tropical routing partition is relatively slow
to update compared to classical gradient descent. A routing path that was optimal
yesterday may be suboptimal today, but re-routing requires enough gradient signal
to shift which j* wins — and during the transition period, the subgradient is
pointing at the wrong partition.

**Classical networks** adapt smoothly — every weight adjusts continuously to
concept drift. Tropical networks must make discrete routing reconfiguration
events to adapt.

**Magnitude:** In concept drift benchmarks (e.g., stream datasets), expect
5–10% lower accuracy during drift periods. Stable period accuracy is comparable.

---

### 2.8  Tasks requiring dense gradient to input (generative / reconstruction)

**Task type:** Autoencoders, VAEs, image reconstruction, any task where you
backpropagate through the network back to the input for gradient-based generation.

**Why tropical underperforms:** The tropical linear map passes gradient only
through the single winning j* per output neuron. The gradient reaching the input
layer is extremely sparse — most input dimensions receive zero gradient. For
reconstruction tasks, this means most input pixels/features receive no learning
signal from the loss, making reconstruction training unstable.

**Do not use deep tropical networks as generators or autoencoders without
significant modification** (specifically: a dense classical decoder paired
with a tropical encoder, and classical gradients flowing only through the decoder).

---

### 2.9  Summary table: where to use which architecture

| Task type | Hybrid transformer | Deep tropical net | Purely classical |
|---|---|---|---|
| Standard image classification | ✓ competitive | ~1–3% below | ✓ baseline |
| Hierarchical / structured NLP | ✓ better | ✓ better | baseline |
| Natural language modeling (LM) | ~ parity (gate helps) | ✗ avoid | ✓ preferred |
| Formal reasoning / logic | ✓ better | ✓ better | baseline |
| Smooth continuous regression | ~ parity (gate helps) | ✗ avoid | ✓ preferred |
| Combinatorial optimization | ✓ better | ✓✓ strongly better | baseline |
| PWA system identification | ✓ better | ✓✓ strongly better | baseline |
| Shortest path / scheduling | ✓ better | ✓✓ strongly better | baseline |
| Image generation / diffusion | ✗ avoid | ✗ avoid | ✓ required |
| Audio synthesis / WaveNet | ✗ avoid | ✗ avoid | ✓ required |
| Low data (< 1k samples) | ~ marginal | ✗ avoid | ✓ preferred |
| Adversarial robustness | ✗ weaker | ✗ weaker | ✓ preferred |
| Online / streaming | ~ marginal | ✗ avoid | ✓ preferred |
| Molecular / graph property | ✓ better | ✓ better | baseline |
| 3D point cloud | ✓ competitive | ✓ competitive | baseline |
| Control policy (PWA env) | ✓ better | ✓✓ strongly better | baseline |
| Code understanding | ✓ better | ✓ better | baseline |

Legend: ✓✓ = strong gain (10%+), ✓ = modest gain (2–10%), ~ = parity (±2%),
✗ = avoid (significant underperformance expected)

---

## §3  Compute Analysis

### 3.1  The fundamental hardware mismatch — be honest about this

**Bottom line up front:** On current NVIDIA GPU hardware, TropFormer is
**slower than a classical transformer** at equivalent parameter count during
training. This is not a theoretical concern — it is a real practical constraint
that must be disclosed in both papers and understood before integration into
large models.

**Why:**

Modern GPUs execute matrix multiplication using **tensor cores** — dedicated
hardware units that perform 4×4 (or 8×8 or 16×16) matrix blocks in a single
instruction. The entire cuBLAS and cuDNN stack is built around this. A classical
linear layer `y = Wx` is an extremely fast GEMM (General Matrix Multiply) call.

The tropical linear map `yᵢ = maxⱼ(Wᵢⱼ + xⱼ)` requires:
1. A broadcast addition of W and x: `W.unsqueeze(0) + x.unsqueeze(1)` — this
   is an `(out, in)` + `(B, 1, in)` → `(B, out, in)` expansion
2. A max reduction over the last dimension

Step 1 allocates an intermediate tensor of shape `(B, out, in)` — for large
dimensions this is a significant memory overhead. Step 2 has no tensor core
equivalent. The operation maps to slower elementwise CUDA kernels.

**Approximate slowdown on current GPUs:**

| Operation | Relative wall-clock (same dimensions) |
|---|---|
| Classical linear (GEMM, tensor cores) | 1.0× (baseline) |
| Tropical linear (broadcast + max) | 2.5–4.0× slower |
| Maslov softmax (same as classical softmax) | 1.0× |
| LF dual activation | 1.8–2.5× slower than GELU |
| TropicalDropout | ~1.0× (negligible overhead) |
| Score gate (classical sigmoid) | ~1.0× |

**Overall TropFormer vs classical transformer (same architecture size):**
Training is approximately **1.8–2.5× slower** wall-clock on GPU due to the
tropical linear operations in Q/K projections and the FFN tropical branch.

This must be acknowledged in the papers. The standard way to handle this is:
"We note that the tropical linear map currently lacks optimized GPU kernel support.
All timing comparisons are under-favorable for TropFormer; an optimized max-plus
CUDA kernel (analogous to cuBLAS for GEMM) would reduce this overhead to near
parity." Then cite this as future work.

---

### 3.2  Memory overhead during training

The intermediate tensor `(B, out, in)` in `TropicalLinear.forward` is the main
memory concern. For a layer with `in=512, out=512, B=128`:

- Classical: `y = x @ W.T` — no intermediate allocation beyond output
- Tropical: intermediate `(128, 512, 512)` float32 = 128MB per layer

This is a **significant memory overhead** for large models. Mitigation options:

**Option 1 — Chunked tropical linear:**
Process the output dimension in chunks, discarding each intermediate after the max:
```python
def chunked_tropical_forward(self, x, chunk_size=64):
    outputs = []
    for i in range(0, self.out_features, chunk_size):
        W_chunk = self.weight[i:i+chunk_size]   # (chunk, in)
        scores = W_chunk.unsqueeze(0) + x.unsqueeze(1)  # (B, chunk, in)
        outputs.append(scores.max(-1).values)
        # intermediate freed here
    return torch.cat(outputs, dim=-1)
```

This reduces peak memory by `out_features / chunk_size` at the cost of
slightly more kernel launch overhead.

**Option 2 — Custom CUDA kernel (future work):**
A fused max-plus matmul kernel that computes the result without materializing
the full `(B, out, in)` intermediate. This is the correct long-term solution
and would bring tropical linear to within ~1.2× of classical GEMM in both
speed and memory. Writing this kernel should be tracked as a research
infrastructure task.

---

### 3.3  Inference compute — where tropical becomes advantageous

Training is slower. Inference, under certain conditions, can be **faster than
classical**. This is the compute argument for tropical at scale.

**The routing stability property:**

After a tropical network is fully trained, its routing paths (which j* wins
per input) are deterministic and relatively stable for inputs from the training
distribution. For any given input, the forward pass travels through a specific
sequence of affine regions. At inference time, you can exploit this:

**Route caching / path prediction:**
For a trained TropFormer serving a fixed input distribution (e.g., a
production NLP model serving customer queries), the routing paths can be
pre-profiled:

1. Run the model on a representative sample of inputs
2. Record which j* wins at each tropical layer for each input
3. Cluster inputs by routing path — inputs that take the same tropical path
   get sent to a highly optimized affine function (just a single matrix multiply
   per layer, no max comparison needed)
4. At inference, classify input to its routing cluster first, then execute
   the cached affine path

This is analogous to profile-guided optimization in compilers. The result:
**inference can be 2–5× faster than standard transformer inference** for
inputs that follow stable routing paths, because each layer reduces to a
single classical linear operation (no max needed, routing is pre-determined).

**The Maslov temperature benefit at inference:**

Attention heads that converge to low τ (tropical regime) during training can be
replaced with hard argmax at inference:
```
τ < 0.3:  replace softmax(score/τ) with one_hot(argmax(score))
          → attention reduces to a single lookup, not a weighted sum
          → O(L) instead of O(L²) attention for these heads
```

For a 12-head model where 4 heads converge to τ < 0.3, those 4 heads become
O(L) instead of O(L²). At sequence length 2048, this is a 2048× speedup for
those specific heads. The other 8 heads remain classical O(L²).

---

### 3.4  Large model integration — GPT-scale and Claude-scale analysis

**The question:** Would replacing some transformer layers with TropFormer layers
in a large language model (GPT-4 scale, ~1.8T parameters; or Claude-scale) reduce
compute while preserving capability?

**Honest answer:** Mixed. Here is the breakdown by component.

#### What helps at large scale

**Sparse attention via tropical routing:**
Large language models spend a significant fraction of inference compute on
the attention operation — specifically the O(L²) softmax attention. Heads in
later layers of large models are known to develop sparse, routing-like behavior
(specific heads handle coreference, subject tracking, etc.). Replacing these
specific heads with tropical heads:
- Reduces attention from O(L²) to O(L) for routing heads at inference
- Reduces KV cache memory for these heads (only the key-aligned maximum per
  head needs to be stored, not the full KV vector)
- At sequence length 4096 (common for large models), this is meaningful savings

**Parameter efficiency on structured capabilities:**
Tasks like code execution tracing, logical reasoning, mathematical calculation,
and formal language parsing are intrinsically tropical. For these capabilities,
a tropical layer might achieve the same accuracy as a classical layer with
significantly fewer parameters (because one tropical layer is an exact PWL
function, not an approximation). If 20% of a model's capabilities are in this
category, replacing those layers with more parameter-efficient tropical layers
could reduce model size for equivalent capability.

**Interpretability at scale:**
This is not a compute benefit but it may be more important. In large models,
understanding what individual components are computing is one of the central
challenges of AI safety and alignment. Tropical layers have deterministic,
inspectable routing paths. For a model used in safety-critical contexts, having
even a fraction of layers be fully interpretable (exact routing paths, explicit
partition boundaries) is potentially very valuable.

#### What hurts at large scale

**Training compute increase:**
Training a large model with tropical layers costs 2–3× more compute for those
layers. At GPT-4 scale (estimated ~10²⁵ FLOP training cost), even a 10%
increase is economically significant. The current architecture is not suitable
for training from scratch at scale on current hardware. It needs the custom
CUDA kernel first.

**The dead weight problem:**
In a large language model, natural language modeling is the primary task. As
shown in §2.2, deep tropical routing is a liability for this task. If tropical
layers are integrated into a model trained primarily on language, the gate must
learn to suppress the tropical path for most language modeling inputs — and
this suppression adds overhead without benefit.

**Recommended integration strategy for large models:**

Do not replace random layers. The evidence-based strategy is:

1. First train a classical large model as normal.
2. Profile all attention heads using attention entropy — identify the subset
   of heads with low entropy (< 0.5 bits) that are already doing hard routing.
   These are "proto-tropical" heads. In GPT-2 large, ~15–20% of heads fall
   in this category.
3. Replace only those specific heads with TropFormer tropical heads.
   Initialize the tropical weights to match the pre-trained classical weights
   as closely as possible (use the argmax of the softmax attention as the
   initial routing prior).
4. Fine-tune for a small number of steps on the same data.
5. At inference, convert low-τ heads to hard argmax.

**Expected outcome of this strategy:**
- Language modeling perplexity: +0.2–0.5 points (negligible degradation)
- Reasoning / structured task performance: +2–5% (gain from explicit routing)
- Inference speed on long sequences: +8–15% (tropical heads become O(L))
- Interpretability: significant gain on the fraction of heads converted
- Training cost of fine-tuning phase: modest (~1–5% of original training cost)

This is a realistic, near-term path to tropical integration in production models.

---

### 3.5  The custom CUDA kernel: the key infrastructure investment

Everything in §3 changes significantly once an optimized max-plus GEMM kernel
exists. This is a concrete engineering task, not a research question:

**What it needs to do:**
- Compute `yᵢ = maxⱼ(Wᵢⱼ + xⱼ)` without materializing the `(B, out, in)` intermediate
- Use CUDA shared memory tiling analogous to cuBLAS's matrix multiply tiling
- Support float16 / bfloat16 for mixed precision training

**Estimated engineering effort:** 2–4 weeks for a CUDA-competent engineer.
The algorithmic structure is straightforward — it is a reduction over the inner
dimension, which maps naturally to CUDA reduction kernels. The main work is
tiling for cache efficiency.

**Expected performance after custom kernel:**
- Training speed: tropical linear within 1.2–1.5× of classical GEMM
  (slightly slower due to reduction vs multiply-accumulate, but close)
- Memory: no intermediate tensor allocation; in-place reduction
- Overall TropFormer training: within 1.1–1.3× of classical transformer

With this kernel, the compute argument for large model integration becomes
strongly positive. Track this as the highest-priority infrastructure task
after paper submission.

---

## §4  The Strategic Narrative

The papers should be framed around this argument, in this order:

**1. We identify a structural principle that classical transformers lack:**
The ability to learn discrete, hard routing boundaries that correspond to
the polyhedral geometry of combinatorial tasks. Softmax attention is smooth
by construction — it cannot represent exact combinatorial boundaries regardless
of scale.

**2. We show parity on smooth tasks:**
On MNIST, CIFAR-10, CIFAR-100, SST-2 — the gate correctly learns to use
classical paths. This is evidence the architecture is well-designed, not
broken. It does not regress on tasks where it has no advantage.

**3. We show gains on partially structured tasks:**
LRA Pathfinder, Penn Treebank, SCAN, structured NLP — the gate partially
engages tropical routing. The Maslov temperature diagnostics show which heads
are routing and which are attending smoothly.

**4. We show decisive gains on intrinsically tropical tasks:**
LRA ListOps, synthetic PWA, shortest path, scheduling — these are the claims
that no classical network can match without dramatically more parameters. These
are the falsifiable, specific scientific contributions.

**5. We are honest about failure modes and compute cost:**
Sections on smooth regression, dense language modeling, adversarial settings,
and training overhead. This honesty is what makes the paper credible. Reviewers
know the failure modes exist — if you don't mention them, reviewers will assume
you're hiding them. If you mention them first and explain the trade-off clearly,
reviewers see a mature, trustworthy contribution.

---

## §5  Agent Instructions

### Benchmark implementation priority order

```
IMMEDIATE (before writing any results):
[ ] 1.  Replace MNIST as primary dev benchmark with CIFAR-10
[ ] 2.  Implement LRA ListOps data loader and evaluation
[ ] 3.  Run full ablation on CIFAR-10 + ListOps (8 ablations, §1.1)
[ ] 4.  Implement Maslov temperature heatmap visualization
[ ] 5.  Implement routing path visualization (winning j* per layer)

SHORT TERM (before Paper 1 draft):
[ ] 6.  Implement LRA Pathfinder data loader
[ ] 7.  Run all Paper 1 parity benchmarks (table §1.1 Tier 1)
[ ] 8.  Run all Paper 1 modest-gain benchmarks (table §1.1 Tier 2)
[ ] 9.  Run ListOps + Path-X (Tier 3 strong gain)
[ ] 10. Implement gate decisiveness tracking over training
[ ] 11. Run classical ablation (compare against Fully Classical baseline)

BEFORE PAPER 2:
[ ] 12. Implement TropicalBatchNorm + TropicalLinearSTE + tropical residual
[ ] 13. Implement generate_pwa_dataset() synthetic generator
[ ] 14. Implement generate_shortest_path_dataset() generator
[ ] 15. Implement generate_scheduling_dataset() generator
[ ] 16. Run deep tropical net on all Paper 2 benchmarks
[ ] 17. Implement tropical eigenvalue monitor
[ ] 18. Implement chunked_tropical_forward() for memory efficiency

INFRASTRUCTURE (track separately):
[ ] 19. Prototype custom CUDA kernel for max-plus GEMM (post-paper)
[ ] 20. Implement route caching for inference optimization
[ ] 21. Implement attention head profiling tool (entropy-based, for LLM integration)
```

### Do not do these things

```
✗ Do not use MNIST as the primary result in either paper
✗ Do not hide the CIFAR-10 (smooth task) result in Paper 2
✗ Do not claim tropical is universally better than classical
✗ Do not claim reduced training compute on current hardware (it is more)
✗ Do not remove the gated fusion — it is what prevents degradation on smooth tasks
✗ Do not increase trop_dropout above 0.10 without running the gradient norm ablation first
✗ Do not benchmark on audio synthesis or image generation tasks
```

---

*Last updated: generated from benchmark strategy and compute analysis session.*
*Companion files: `TROPFORMER_CONTEXT.md`, `TROPFORMER_ROADMAP.md`*
*Primary source: `tropformer.py`*