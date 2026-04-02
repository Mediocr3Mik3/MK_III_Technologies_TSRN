# TropFormer: Agent Roadmap, White Paper Outlines & Research Frontiers

> **Purpose of this document:** Companion to `TROPFORMER_CONTEXT.md`. This document
> directs an AI coding agent through two parallel development tracks — a drop-in
> transformer replacement and a ground-up deep tropical network — and provides full
> outlines for two scientific white papers to be written as each track matures.
> Read `TROPFORMER_CONTEXT.md` first for theoretical foundations.

---

## Document Map

```
§1  Validation Criteria          — when the current TropFormer is "ready"
§2  Path A: Drop-in Replacement  — full engineering specification
§3  Path B: Deep Tropical Net    — full engineering specification
§4  White Paper 1 Outline        — hybrid tropical transformer
§5  White Paper 2 Outline        — deep tropical network
§6  Research Frontiers           — other ML/AI domains where this applies
§7  Agent Task Queues            — ordered to-do lists for each path
```

---

## §1  Validation Criteria

Before beginning either development path, the current `tropformer.py` must pass the
following gates. These are the minimum bar for the architecture to be worth scaling.

### Gate 1 — Training stability (non-negotiable)
- Loss curve must be monotonically decreasing (with normal noise) for all 25 epochs
- No NaN or Inf values in loss, activations, or gradients at any point
- Gradient norm (logged per step) must stay bounded: `clip_grad_norm` should rarely
  be triggered (< 5% of steps). If clipping fires constantly, the tropical layers are
  generating pathological gradients and the architecture needs tuning before scaling.

### Gate 2 — Accuracy (non-negotiable)
- Test accuracy ≥ 98.0% on MNIST after 25 epochs
- This is achievable by a simple 2-layer MLP. If TropFormer can't match it, something
  is fundamentally broken in the tropical routing, the LF activation, or the training
  configuration.

### Gate 3 — Tropical layers are contributing (important)
- Run an ablation: replace all `TropicalLinear` with `nn.Linear`, keep everything else
  identical. If the ablated model scores ≥ 0.3% higher test accuracy, the tropical
  layers are hurting and the gate balance / initialization needs revisiting.
- Goal: TropFormer score ≥ ablated classical score. Even parity is acceptable for now.

### Gate 4 — Diagnostic outputs are meaningful (important)
- After training, `maslov_summary()` should show head specialization:
  at least one head with `τ < 0.5` and at least one with `τ > 0.8`. If all heads
  collapse to the same temperature, the per-head learning is not working.
- `lf_mode_summary()` should show variation across blocks (not all 0.5).

### Gate 5 — Reproducibility
- Three independent runs with different seeds must all pass Gates 1–4.
- Log seeds, hyperparameters, and final metrics in a `results/` directory.

---

## §2  Path A: Drop-in Transformer Replacement

### 2.0  Design Philosophy

The goal is full API compatibility with `torch.nn.Transformer` and the HuggingFace
`transformers` library. A downstream user should be able to swap
`nn.TransformerEncoderLayer` for `TropicalEncoderLayer` with no other code changes.
The tropical machinery is an internal implementation detail.

**Constraints:**
- Input/output shapes must match standard transformer conventions
- All mask types that PyTorch supports must be supported
- `batch_first=True` must be the default (modern convention)
- The module must be serializable via `torch.save` / `torch.load`
- HuggingFace `PreTrainedModel` compatibility is the final milestone

### 2.1  New Module Hierarchy

Refactor the single `TropFormer` class into a proper module hierarchy:

```
TropicalEncoderLayer          — single encoder block (attn + FFN)
TropicalDecoderLayer          — single decoder block (self-attn + cross-attn + FFN)
TropicalEncoder               — stack of N TropicalEncoderLayers
TropicalDecoder               — stack of N TropicalDecoderLayers
TropicalTransformer           — encoder + decoder (seq2seq)
TropicalEncoderModel          — encoder-only (BERT-style)
TropicalDecoderModel          — decoder-only (GPT-style)

# Vision front-ends (keep but separate from core)
TropicalViT                   — image patch embedding + TropicalEncoder
TropicalViTForClassification  — TropicalViT + classification head

# HuggingFace wrappers (final milestone)
TropFormerConfig              — PretrainedConfig subclass
TropFormerModel               — PreTrainedModel subclass
TropFormerForSequenceClassification
TropFormerForTokenClassification
TropFormerForCausalLM
```

### 2.2  `TropicalEncoderLayer` — specification

```python
TropicalEncoderLayer(
    d_model:      int,
    num_heads:    int,
    ffn_dim:      int,
    dropout:      float = 0.1,
    trop_dropout: float = 0.05,
    lf_pieces:    int   = 8,
    lf_mode:      str   = 'blend',
    init_temp:    float = 1.0,
    batch_first:  bool  = True,
    norm_first:   bool  = True,    # pre-norm (keep True for tropical stability)
)

def forward(
    self,
    src:                    Tensor,          # (B, L, d) if batch_first
    src_mask:               Tensor | None,   # (L, L) additive mask
    src_key_padding_mask:   Tensor | None,   # (B, L) bool, True = ignore
) -> Tensor
```

**Important mask convention:** Use *additive* masks (add to scores before softmax),
not multiplicative. Set ignored positions to `−1e9` (tropical zero) not `−inf`.
Using `−inf` causes NaN in the tropical score path when `max(−inf + k) = −inf`
propagates through the Maslov softmax.

### 2.3  `TropicalDecoderLayer` — specification

The decoder layer adds cross-attention. The cross-attention gate needs redesign:

```python
TropicalDecoderLayer(
    d_model:      int,
    num_heads:    int,
    ffn_dim:      int,
    dropout:      float = 0.1,
    trop_dropout: float = 0.05,
    lf_pieces:    int   = 8,
    lf_mode:      str   = 'blend',
    init_temp:    float = 1.0,
    batch_first:  bool  = True,
    norm_first:   bool  = True,
)

def forward(
    self,
    tgt:                    Tensor,          # (B, L_tgt, d)
    memory:                 Tensor,          # (B, L_src, d) — encoder output
    tgt_mask:               Tensor | None,   # (L_tgt, L_tgt) causal
    memory_mask:            Tensor | None,   # (L_tgt, L_src) cross-attn mask
    tgt_key_padding_mask:   Tensor | None,   # (B, L_tgt)
    memory_key_padding_mask: Tensor | None,  # (B, L_src)
) -> Tensor
```

**Cross-attention gate redesign:** The score gate in `TropicalMultiHeadAttention`
currently takes `x` (shape `(B, L, d_model)`) as input. In cross-attention, Q comes
from `tgt` and K/V from `memory`. The gate should condition on both:

```python
# Concatenate mean-pooled context from both sequences
tgt_ctx    = tgt.mean(dim=1)        # (B, d_model)
mem_ctx    = memory.mean(dim=1)     # (B, d_model)
gate_input = torch.cat([tgt_ctx, mem_ctx], dim=-1)  # (B, 2*d_model)
# gate_proj input size changes: d_model -> 2*d_model
g = torch.sigmoid(self.score_gate(gate_input))      # (B, num_heads)
```

Update `TropicalMultiHeadAttention.__init__` to accept `cross_attention: bool` flag
and instantiate the appropriate gate projection size.

### 2.4  Causal mask utility

```python
def make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """
    Upper-triangular additive mask for autoregressive decoding.
    Uses -1e9 (tropical zero) not -inf, to avoid NaN in tropical score path.
    Shape: (seq_len, seq_len), broadcast over (B, H, L, L).
    """
    return torch.triu(
        torch.full((seq_len, seq_len), -1e9, device=device),
        diagonal=1
    )
```

### 2.5  KV cache for autoregressive inference

Add a `TropicalKVCache` class for efficient generation:

```python
@dataclass
class TropicalKVCache:
    """
    Caches raw K and V tensors from previous steps.
    K is stored pre-split (before head splitting) so the tropical Q/K score
    can be recomputed across the full cached sequence without re-projecting.

    The tropical score max_i(Q_i + K_i) is correct across the full cache:
    max over i of (q_new_i + k_cached_i) correctly finds the peak-aligned
    dimension across all cached positions. No approximation needed.
    """
    k: Tensor | None = None    # (B, H, L_cached, d_k)
    v: Tensor | None = None    # (B, H, L_cached, d_k)

    def update(self, k_new: Tensor, v_new: Tensor):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = self.v = None
```

Each `TropicalMultiHeadAttention` layer accepts an optional `cache: TropicalKVCache`
argument. During generation, pass a persistent cache object that accumulates K/V
across decoding steps.

### 2.6  HuggingFace integration

```python
from transformers import PretrainedConfig, PreTrainedModel

class TropFormerConfig(PretrainedConfig):
    model_type = "tropformer"

    def __init__(
        self,
        vocab_size:   int   = 32000,
        d_model:      int   = 512,
        num_heads:    int   = 8,
        num_layers:   int   = 6,
        ffn_dim:      int   = 2048,
        dropout:      float = 0.1,
        trop_dropout: float = 0.05,
        lf_pieces:    int   = 8,
        lf_mode:      str   = 'blend',
        init_temp:    float = 1.0,
        max_position_embeddings: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # store all fields as self.X

class TropFormerModel(PreTrainedModel):
    config_class = TropFormerConfig

    def __init__(self, config: TropFormerConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed  = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.encoder    = TropicalEncoder(...)
        self.norm       = nn.LayerNorm(config.d_model)

    def forward(self, input_ids, attention_mask=None, ...):
        ...
```

Register with AutoModel:
```python
from transformers import AutoConfig, AutoModel
AutoConfig.register("tropformer", TropFormerConfig)
AutoModel.register(TropFormerConfig, TropFormerModel)
```

### 2.7  Benchmark plan for Path A validation

Once the drop-in architecture is built, run these benchmarks in order:

| Benchmark | Dataset | Baseline | Goal |
|---|---|---|---|
| Sequence classification | SST-2 (sentiment) | BERT-base 93.5% | > 91% |
| Token classification | CoNLL-2003 (NER) | BERT-base F1 91.1 | > 89 F1 |
| Machine translation | WMT14 En-De | Transformer-base 27.3 BLEU | > 25 BLEU |
| Image classification | CIFAR-10 | ViT-small 94.4% | > 93% |
| Image classification | ImageNet-1k | ViT-base 81.8% | > 79% |

The goal is not to beat the baseline — it is to come within 2% while using the tropical
architecture. If TropFormer is within 2%, the architecture is validated as a
scientifically interesting alternative, not just a curiosity.

---

## §3  Path B: Deep Tropical Network

### 3.0  Design Philosophy

Path B does not try to match existing APIs. It is a ground-up architecture where
tropical algebra is the *primary* computational substrate. Classical operations are
used only where they are thermodynamically necessary:
- At the input boundary (embedding, normalization to initialize stable polytopes)
- At the output boundary (Maslov bridge to probability simplex)
- In the gradient correction system (STE, shadow, or residual)

Everything else — routing, feature transformation, aggregation, positional reasoning —
should be tropical.

**The core hypothesis of Path B:** A sufficiently deep stack of tropical linear layers,
with appropriate gradient stabilization, can learn richer piecewise-linear structure
than either a ReLU network or a hybrid network, because each layer compounds the
polyhedral partition of the previous layer in a combinatorially explosive way. The
number of linear regions in an L-layer tropical network with width n grows as O(n^L),
comparable to ReLU networks but with fundamentally different partition geometry.

### 3.1  The gradient death problem — full solution stack

This is the central engineering challenge of Path B. Deploy all three mechanisms:

#### Mechanism 1: Straight-through estimator (STE) wrapper

Wrap every `TropicalLinear.forward` with STE so that backward sees a smooth approximation:

```python
class TropicalLinearSTE(TropicalLinear):
    """
    Forward:  tropical max-plus computation (exact, hard routing)
    Backward: straight-through estimate via soft-max approximation

    The STE backward uses softmax(W + x, dim=-1) @ x as a smooth stand-in
    for the argmax gradient. This is more informative than pure identity STE
    because it weights the gradient by how close each j was to winning.
    """

    def __init__(self, *args, ste_temp: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ste_temp = ste_temp   # temperature for STE softmax approximation

    def forward(self, x: Tensor) -> Tensor:
        # True tropical forward
        scores = self.weight.unsqueeze(0) + x.unsqueeze(1)  # (B, out, in)
        out_trop = scores.max(dim=-1).values                 # (B, out)

        if self.training:
            # Smooth approximation for backward only
            soft_weights = F.softmax(scores / self.ste_temp, dim=-1)  # (B, out, in)
            out_soft = (soft_weights * x.unsqueeze(1)).sum(dim=-1)    # (B, out)
            # STE: forward uses tropical, backward flows through soft
            out = out_trop.detach() + (out_soft - out_soft.detach())
        else:
            out = out_trop

        if self.bias is not None:
            out = out + self.bias
        return out
```

#### Mechanism 2: Tropical batch normalization

```python
class TropicalBatchNorm(nn.Module):
    """
    Normalizes tropical activations to prevent polytope collapse or explosion.

    Classical BN: (x - mean) / std
    Tropical BN:  (x - trop_max) / trop_range

    trop_max   = max over batch of max(x)  — the "tropical mean"
    trop_range = max(x) - min(x)           — spread of the polytope cells

    After normalization, the most active feature = 0 (tropical one),
    and the spread is standardized to ~1. This prevents any single feature
    from permanently dominating all routing decisions.

    Learnable affine: scale γ and shift β (same as classical BN).
    Running stats maintained for inference (same pattern as nn.BatchNorm1d).
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.gamma        = nn.Parameter(torch.ones(num_features))
        self.beta         = nn.Parameter(torch.zeros(num_features))
        self.eps          = eps
        self.momentum     = momentum
        self.register_buffer('running_max',   torch.zeros(num_features))
        self.register_buffer('running_range', torch.ones(num_features))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, ..., num_features) — arbitrary leading dims
        if self.training:
            trop_max   = x.detach().flatten(0, -2).max(dim=0).values   # (F,)
            trop_range = (x.detach().flatten(0, -2).max(dim=0).values
                        - x.detach().flatten(0, -2).min(dim=0).values).clamp(min=self.eps)
            self.running_max   = (1-self.momentum)*self.running_max   + self.momentum*trop_max
            self.running_range = (1-self.momentum)*self.running_range + self.momentum*trop_range
        else:
            trop_max   = self.running_max
            trop_range = self.running_range

        x_norm = (x - trop_max) / (trop_range + self.eps)
        return self.gamma * x_norm + self.beta
```

#### Mechanism 3: Tropical residual connection

```python
def tropical_residual(fx: Tensor, x: Tensor, center: Tensor) -> Tensor:
    """
    Tropical skip connection: max(f(x) - center, x)

    The learned center parameter prevents the constraint max(f,x)>=x
    from restricting the network's expressiveness. The center shifts the
    transformed path so both transformed and identity can win freely.

    center: (d,) learnable per-feature offset
    """
    return torch.maximum(fx - center, x)
```

### 3.2  Deep Tropical Block

The fundamental building block of Path B:

```python
class DeepTropicalBlock(nn.Module):
    """
    One block of the deep tropical network.

        x → TropicalBatchNorm
          → TropicalLinearSTE
          → LFDualActivation
          → TropicalDropout
          → tropical_residual(out, x)

    No classical linear layer. No gated fusion. Purely tropical.
    The STE wrapper handles gradient flow.
    The tropical residual handles identity gradient path.
    The tropical BN prevents polytope collapse across depth.
    """

    def __init__(
        self,
        dim:          int,
        lf_pieces:    int   = 8,
        lf_mode:      str   = 'blend',
        trop_dropout: float = 0.05,
        ste_temp:     float = 1.0,
    ):
        super().__init__()
        self.trop_bn    = TropicalBatchNorm(dim)
        self.trop_lin   = TropicalLinearSTE(dim, dim, ste_temp=ste_temp)
        self.lf_act     = LFDualActivation(dim, num_pieces=lf_pieces, mode=lf_mode)
        self.trop_drop  = TropicalDropout(trop_dropout)
        self.center     = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        out = self.trop_bn(x)
        out = self.trop_lin(out)
        out = self.lf_act(out)
        out = self.trop_drop(out)
        return tropical_residual(out, x, self.center)
```

### 3.3  Deep Tropical Attention

In Path B, the attention mechanism is redesigned so that *all score computation is
tropical*. The classical dot-product path is removed entirely. Maslov temperature
is the only smoothing mechanism.

```
Q = TropicalLinearSTE(x)         — max-plus query projection
K = TropicalLinearSTE(x)         — max-plus key projection
V = TropicalLinearSTE(x)         — max-plus value projection (novel: V is tropical too)

trop_score(q,k) = max_i(q_i + k_i) / sqrt(d_k)   — pure tropical inner product
attn = MaslovSoftmax(trop_score, τ)                — smooth via temperature only

# Value aggregation: instead of classical attn @ V (matrix multiply),
# use tropical weighted aggregation:
# out_i = max_j ( log(attn_ij) + V_ij )
# = max over attended positions of (log-attention-weight + value)
# This keeps the aggregation in the tropical semiring.
out = (attn.log().unsqueeze(-1) + V.unsqueeze(2)).max(dim=2).values
```

The tropical value aggregation `max_j(log(attn_j) + V_j)` is the max-plus analog of
the weighted sum `Σ_j attn_j · V_j`. It selects the value from the most attended
position rather than blending all values — maximally hard routing.

**V being tropical:** This is the main new risk in Path B. Classical V was kept smooth
to carry dense gradient signal. With STE on the tropical V projection, the gradient
flows through the soft approximation. Monitor value vector norms carefully during early
training — if they collapse to a single constant, the tropical V is not learning.

### 3.4  Morphological convolution layer

For spatial/sequential inputs, replace classical convolution with tropical (morphological) convolution:

```python
class TropicalConv1d(nn.Module):
    """
    Tropical convolution: y[i] = max_j( W[j] + x[i-j] )

    This is morphological dilation — the standard tool in mathematical
    morphology for image analysis and robust signal processing.

    W[j] acts as a structuring element. The operation finds the
    max-plus inner product of the kernel with each window of the input.

    Classical conv:    y[i] = sum_j( W[j] * x[i-j] )   — inner product
    Tropical conv:     y[i] = max_j( W[j] + x[i-j] )   — max-plus product

    Gradient: STE, same as TropicalLinearSTE.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, ste_temp: float = 1.0):
        super().__init__()
        self.kernel    = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size))
        self.ste_temp  = ste_temp
        self.stride    = stride
        self.padding   = padding
        nn.init.uniform_(self.kernel, -0.5, 0.5)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C_in, L)
        # unfold into windows: (B, C_in * kernel_size, L_out)
        windows = F.unfold(x.unsqueeze(-1),
                           kernel_size=(self.kernel.shape[-1], 1),
                           stride=self.stride, padding=self.padding)
        # reshape: (B, C_in, K, L_out)
        B, CK, L_out = windows.shape
        K = self.kernel.shape[-1]
        windows = windows.view(B, -1, K, L_out)

        # tropical conv: max over (C_in, K) of (W + window)
        w = self.kernel.view(1, self.out_ch, -1, 1)   # broadcast
        win = windows.view(B, 1, -1, L_out)
        out_trop = (w + win).max(dim=2).values          # (B, out_ch, L_out)

        # STE for backward
        if self.training:
            soft_w = F.softmax((w + win) / self.ste_temp, dim=2)
            out_soft = (soft_w * win).sum(dim=2)
            return out_trop.detach() + (out_soft - out_soft.detach())
        return out_trop
```

### 3.5  Tropical loss functions

```python
def tropical_cross_entropy(logits: Tensor, targets: Tensor, margin: float = 1.0) -> Tensor:
    """
    Tropical max-margin loss.

    L = max(0,  max_{k != y}(logit_k)  -  logit_y  +  margin)

    This is structurally a max-plus expression:
    - max over incorrect classes is tropical addition
    - subtraction of correct logit is tropical division (subtract in log domain)
    - the outer max with 0 is tropical addition with tropical zero

    Reduces to SVM hinge loss for binary case.
    Strongly penalizes cases where any wrong class is close to the correct class.
    Works well with tropical logits which are already max-scaled (not sum-scaled).
    """
    B = logits.shape[0]
    correct_logits = logits[torch.arange(B), targets]      # (B,)
    logits_copy = logits.clone()
    logits_copy[torch.arange(B), targets] = -1e9           # mask correct class
    max_wrong = logits_copy.max(dim=1).values              # (B,)
    return F.relu(max_wrong - correct_logits + margin).mean()


def tropical_contrastive_loss(z1: Tensor, z2: Tensor, temperature: float = 0.5) -> Tensor:
    """
    Contrastive loss using tropical (max-plus) similarity instead of dot product.

    trop_sim(a, b) = max_i(a_i + b_i)   [max-plus inner product]

    This measures the single best-aligned feature dimension between two
    representations. Useful for self-supervised learning where you want
    the network to find the single most discriminating feature axis between
    a sample and its augmentation.
    """
    B, D = z1.shape
    # All pairwise tropical similarities: (B, B)
    sim = (z1.unsqueeze(1) + z2.unsqueeze(0)).max(dim=-1).values / temperature
    labels = torch.arange(B, device=z1.device)
    return F.cross_entropy(sim, labels)
```

### 3.6  Full Deep Tropical Net architecture

```python
class DeepTropNet(nn.Module):
    """
    Full deep tropical network.

    Pipeline (classification):
        Input
          → classical embedding (boundary: enter tropical domain)
          → TropicalBatchNorm (initialize polytope scale)
          → [DeepTropicalBlock × num_layers]   (pure tropical)
          → [DeepTropicalAttention × attn_layers] (optional, for sequence tasks)
          → TropicalBatchNorm
          → Maslov bridge: softmax(x / τ_out)  (boundary: exit to probability)
          → classical Linear head
          → loss

    The two classical operations (input embedding + output head) are the
    unavoidable boundaries between the tropical interior and the real world.
    Everything between them is tropical.
    """
```

### 3.7  Benchmark plan for Path B

| Benchmark | Metric | Comparison target | Notes |
|---|---|---|---|
| MNIST | Test accuracy | Path A TropFormer | Validate gradient solution |
| CIFAR-10 | Test accuracy | ResNet-20 (92.4%) | First real stress test |
| Synthetic PWA regression | MSE on held-out | MLP-ReLU | Should be natural advantage |
| Shortest path (graph) | Optimality rate | GNN baseline | Direct tropical home territory |
| Scheduling (max-plus) | Makespan | Classical NN | Tropical eigenvalue domain |

The synthetic PWA regression benchmark is critical: generate a random piecewise-affine
function with known mode partitions and test whether the deep tropical net can recover
the exact partition structure. A classical network cannot — it will approximate with
smooth boundaries. The deep tropical net should find the exact linear regions.

---

## §4  White Paper 1: Hybrid Tropical Transformer

**Title (draft):** "TropFormer: A Hybrid Tropical-Classical Transformer via Max-Plus
Algebra, Maslov Dequantization, and Legendre-Fenchel Dual Activations"

**Target venues:** NeurIPS, ICML, ICLR (ML track); or IEEE Transactions on Neural
Networks and Learning Systems (TNNLS) for a longer archival version.

**Narrative arc:** We introduce a transformer architecture where each layer block
fuses tropical (max-plus) and classical linear algebra through a learned gate. We
connect three mathematical structures — tropical geometry, Maslov dequantization, and
Legendre-Fenchel duality — to show that these are not three independent additions but
one unified theoretical framework. We validate on standard benchmarks and use
post-training diagnostics (Maslov temperature, LF gate values) to analyze what the
network learns.

---

### Paper 1 Outline

```
Abstract (250 words)
  - Problem: transformers are entirely smooth; no principled way to learn hard routing
  - Contribution: hybrid architecture with tropical layers, theoretical connections
  - Results: competitive with classical transformers + interpretable routing structure
  - Key claim: Maslov temperature reveals per-head specialization not visible in standard attn

1. Introduction
   1.1  Motivation: piecewise-linear structure in learned representations
   1.2  The standard transformer's smooth inductive bias and its limitations
   1.3  Tropical algebra as a natural complement
   1.4  Paper contributions (numbered list)
   1.5  Paper organization

2. Background
   2.1  Tropical semiring: definition, examples, matrix algebra
       - Shortest paths, Bellman equations, max-plus linear systems
       - Connection to piecewise-affine switched systems
   2.2  Tropical geometry: varieties, Newton polytopes, polyhedral complexes
       - Tropical linear maps as PWA functions
       - Input space partitioning and the tropical Voronoi interpretation
   2.3  Maslov dequantization
       - LSE_τ(x) = τ·log(Σ exp(x/τ)) → max(x) as τ→0
       - Connection to WKB approximation / quantum-classical limit
       - Softmax as the gradient of the smooth max
   2.4  Legendre-Fenchel duality
       - Convex conjugate: f*(y) = sup_x{⟨x,y⟩ − f(x)}
       - Connection to Lagrangian/Hamiltonian mechanics
       - Tropical polynomials are self-dual under LF: f* is also tropical
   2.5  Related work
       - Tropical neural networks (Zhang et al. 2018, Charisopoulos & Maragos 2018)
       - Piecewise-linear networks and maxout networks
       - Sparse/hard attention (sparsemax, entmax)
       - Temperature annealing in attention (various)

3. The TropFormer Architecture
   3.1  Tropical linear layer
       - Definition, geometry, gradient (subgradient via argmax)
       - Comparison to maxout networks
       - Initialization strategy and why it matters
   3.2  Tropical dropout
       - Tropical zero vs classical zero — why -∞ is the correct mask
   3.3  Maslov temperature attention
       - Per-head learnable τ parameterized as exp(log_τ)
       - Tropical score: max_i(q_i + k_i)/√d vs classical q·k/√d
       - Score gate: σ(Gx)·trop + (1−σ)·classic
       - Connection to entropy-regularized optimal control
   3.4  Legendre-Fenchel dual activation
       - Primal f(x) = max_k(sₖx + bₖ): tropical polynomial
       - Dual f*(y) = max_j(xⱼy − f(xⱼ)): LF conjugate, also tropical
       - Blend mode and the per-channel gate
       - Geometric interpretation: primal vs dual Newton polytope
   3.5  Tropical hybrid FFN
       - Parallel tropical and classical branches
       - Gated fusion: input-conditioned, not output-conditioned
   3.6  Full block structure and pre-norm choice
   3.7  Vision front-end for image tasks

4. Theoretical Analysis
   4.1  Expressiveness
       - Number of linear regions: O(n^L) — matches ReLU networks
       - Qualitative difference: combinatorial vs smooth partition geometry
       - Tropical varieties as switching manifolds in learned PWA systems
   4.2  The Maslov spectrum as a learned parameter
       - Theorem: as τ_h → 0, head h computes tropical attention (Bellman max)
       - Theorem: as τ_h → ∞, head h approaches uniform attention (max entropy)
       - Corollary: standard transformer is the τ_h = 1 fixed-point of TropFormer
   4.3  LF duality and the Newton polytope
       - Theorem: f and f* partition the same space by dual polyhedral complexes
       - Corollary: blend mode provides access to both primal and dual partitions
       - Connection to tropical discriminant and decision boundaries
   4.4  Gradient flow analysis
       - Sparsity of tropical subgradients vs density of classical gradients
       - Gate learning dynamics: why gates converge to stable tropical fractions
       - Pre-norm necessity: formal argument for tropical layer stability

5. Experiments
   5.1  Experimental setup
       - Datasets: MNIST, CIFAR-10, SST-2, CoNLL-2003
       - Baselines: standard transformer, ViT, BERT (where applicable)
       - Hyperparameter sweep for d_model, num_heads, τ_init, lf_mode
       - All runs: 3 seeds, report mean ± std
   5.2  Classification results
       - Table: TropFormer vs baselines across all tasks
       - Learning curves: convergence speed comparison
   5.3  Ablation study
       - Remove tropical Q/K → classical Q/K
       - Remove LF dual activation → GELU only
       - Remove Maslov temperature → fixed τ=1
       - Remove gated fusion → fixed 50/50 blend
       - Each ablation run on MNIST and CIFAR-10
   5.4  Diagnostic analysis (novel contribution)
       - Maslov temperature distribution per head per layer (heatmap)
       - LF blend gate distribution across blocks
       - Score gate decisiveness (|W| norm) per block
       - Routing entropy: how many j's receive non-negligible gradient per step
   5.5  Routing visualization
       - For each MNIST digit class: heatmap of dominant tropical routing paths
       - Do tropical layers learn class-specific routing? (hypothesis: yes)
   5.6  Computational cost analysis
       - FLOPs: TropFormer vs standard transformer
       - Wall-clock training time (GPU)
       - Inference latency comparison

6. Discussion
   6.1  What Maslov temperatures reveal about learned representations
   6.2  Connection to optimal control and Bellman equations (for expert readers)
   6.3  When is tropical routing beneficial vs harmful?
   6.4  Limitations: gradient sparsity, convergence speed, lack of CUDA kernel

7. Conclusion
   7.1  Summary of contributions
   7.2  Open questions
   7.3  Path to the deep tropical network (foreshadow Paper 2)

References (target: 40–60 citations)

Appendix
   A.  Full hyperparameter tables
   B.  Proof of Theorem 4.2 (Maslov spectrum)
   C.  Proof of Theorem 4.3 (LF duality preserves tropical structure)
   D.  Additional experimental results
   E.  Code availability and reproducibility statement
```

---

## §5  White Paper 2: Deep Tropical Network

**Title (draft):** "Deep Tropical Networks: Piecewise-Linear Deep Learning via
Max-Plus Algebra, Morphological Convolution, and Tropical Gradient Stabilization"

**Target venues:** NeurIPS, ICML; or Neural Networks journal for the archival version.
If the control/scheduling results are strong: IEEE Transactions on Automatic Control or
Automatica for the applied track.

**Narrative arc:** Having established the hybrid architecture in Paper 1, we push to
the limit: what happens when tropical algebra is the primary substrate, not just a
component? We introduce three new mechanisms to solve the gradient death problem (STE
wrapper, tropical BN, tropical residuals), define tropical convolution and attention
in their pure forms, and show that the resulting architecture has a unique advantage on
tasks that are intrinsically piecewise-affine — including shortest-path problems,
scheduling, and PWA system identification (direct connection to control theory).

---

### Paper 2 Outline

```
Abstract (250 words)
  - Problem: gradient death prevents deep tropical networks from training
  - Contribution: three-mechanism solution; full deep tropical architecture
  - Results: competitive on standard tasks, superior on intrinsically tropical tasks
  - Key claim: for PWA systems and combinatorial structure, deep tropical >> smooth NN

1. Introduction
   1.1  Motivation: the limits of smooth approximation
       - ReLU networks approximate PWA functions — tropical networks ARE PWA
       - Combinatorial tasks that smooth networks approximate badly
   1.2  The gradient death problem — why deep tropical failed before
   1.3  Our solution approach
   1.4  Contributions

2. Background
   2.1  Recap of tropical algebra (brief — refer to Paper 1)
   2.2  Why depth matters in tropical networks
       - Composition of tropical maps: polytope refinement across layers
       - Exponential growth of linear regions with depth
   2.3  Mathematical morphology and tropical convolution
       - Dilation as max-plus convolution: y[i] = max_j(W[j] + x[i-j])
       - Erosion as min-plus convolution: dual operation
       - Classical results: what morphological networks can represent
   2.4  The gradient death problem — formal analysis
       - Routing instability: argmax winner changes between forward passes
       - Gradient path fragmentation across depth
       - Why residuals alone don't solve it for deep stacks
   2.5  Related work
       - Maxout networks, deep ReLU networks, PWA system identification
       - Mathematical morphology networks (Mondal et al., Maragos et al.)
       - Tropical neural networks (prior work)
       - Quantization-aware training and STE (Bengio et al. 2013)
       - Soft decision trees and routing networks

3. Deep Tropical Network Architecture
   3.1  Overview and design philosophy
       - Classical boundaries: input embedding, output Maslov bridge
       - Tropical interior: all feature transformation
   3.2  Tropical batch normalization
       - Motivation: polytope collapse/explosion in deep stacks
       - Definition: (x − trop_max) / trop_range
       - Learnable γ, β; running stats for inference
       - Theorem: tropical BN is equivariant to tropical scaling symmetry
   3.3  Straight-through estimator for tropical layers
       - Forward: exact max-plus (hard routing preserved at inference)
       - Backward: softmax-weighted smooth approximation (not identity STE)
       - STE temperature as a hyperparameter / scheduled annealing
   3.4  Tropical residual connections
       - max(f(x) − c, x) with learned center c
       - Why classical residuals break the tropical algebraic structure
       - Gradient highway analysis: identity path always provides gradient
   3.5  Deep tropical block
       - TropBN → TropLinearSTE → LFDualActivation → TropDropout → TropResidual
   3.6  Tropical convolution (morphological dilation)
       - 1D and 2D versions
       - STE wrapper for gradient flow
       - Connection to structuring element optimization in morphology
   3.7  Deep tropical attention
       - Pure tropical Q/K/V projections
       - Tropical value aggregation: max_j(log(attn_j) + V_j)
       - Maslov bridge as the only smoothing mechanism
   3.8  Tropical loss functions
       - Tropical cross-entropy (max-margin)
       - Tropical contrastive loss (max-plus similarity)
       - When to use tropical vs classical loss

4. Theoretical Analysis
   4.1  Expressiveness of deep tropical networks
       - Theorem: L-layer width-n deep tropical net has ≥ C(n,L) linear regions
       - Comparison to ReLU: same asymptotic count, different geometry
       - The unique property: regions are exact polytopes, not smooth approximations
   4.2  Tropical BN convergence analysis
       - Theorem: with tropical BN, the polytope partition diameter is bounded
         at each layer (prevents collapse / explosion)
       - Corollary: gradient norm through deep tropical stack is bounded
   4.3  STE bias analysis
       - Bias of the softmax-weighted STE approximation vs true subgradient
       - Conditions under which bias → 0 as STE temperature → 0
   4.4  Deep tropical networks as learned PWA systems
       - Theorem: a K-layer deep tropical net is a PWA function with
         polyhedral partition determined by the tropical varieties of each layer
       - Connection to hybrid automaton reachability analysis
   4.5  Tropical eigenvalue monitoring
       - Definition: tropical eigenvalue λ of TropicalLinear weight W satisfies
         max_j(W_ij + v_j) = λ + v_i
       - Interpretation: "throughput rate" of feature routing through layer
       - Theorem: λ bounds the Lipschitz constant of the tropical layer map
       - Implication for stability: monitoring λ during training detects routing collapse

5. Experiments
   5.1  Gradient death baseline: deep tropical without stabilization
       - Show: training failure at depth ≥ 4 without STE + TropBN + TropResidual
       - Ablation of each stabilization mechanism individually
       - Key figure: loss curve, routing entropy, gradient norm vs depth
   5.2  Standard classification benchmarks
       - MNIST, CIFAR-10, CIFAR-100
       - Compare to: MLP-ReLU, classical transformer, Path A TropFormer
   5.3  Intrinsically tropical benchmarks (key contribution)
       5.3.1  Shortest-path regression
           - Synthetic graphs, known optimal paths
           - Deep tropical net vs GNN vs MLP
           - Hypothesis: tropical net learns exact tropical eigenvalue structure
       5.3.2  Piecewise-affine system identification
           - Generate random PWA systems with known mode partitions
           - Train network to predict next state given current state + input
           - Evaluate: does the learned partition match the true partition?
           - Metric: partition boundary Hausdorff distance, per-mode MSE
       5.3.3  Job-shop scheduling (max-plus system)
           - Classic max-plus timed event graph scheduling problem
           - Deep tropical net as an end-to-end schedule predictor
           - Compare to: classical NN, ILP solver (upper bound)
   5.4  Morphological convolution on image tasks
       - Replace classical conv with TropicalConv1d/2d
       - Results on MNIST, CIFAR-10
       - Visualization: learned structuring elements vs classical filters
   5.5  Tropical vs classical loss function comparison
       - Standard cross-entropy vs tropical max-margin loss
       - On all benchmarks: which is better for tropical nets?
   5.6  Tropical eigenvalue trajectories during training
       - Plot λ per layer per epoch
       - Correlation between λ and validation accuracy

6. Connection to Control Theory
   6.1  Deep tropical nets as learned hybrid automata
       - The polyhedral partition is the mode partition of a hybrid system
       - Mode identification from data: tropical routing recovers switching surfaces
   6.2  Bellman equation and tropical value function approximation
       - Deep tropical net as a function approximator for V(x) in optimal control
       - Why PWL approximation of V is better than smooth approximation in
         constrained LQR and MPC (no approximation at boundaries)
   6.3  Max-plus system identification
       - Manufacturing scheduling and timed event graphs
       - Tropical eigenvalue of learned weight = throughput of identified system
   6.4  Robustness and worst-case analysis
       - Morphological erosion as robust feature extraction (min-plus dual)
       - Connection to H-infinity control: worst-case over disturbances is a tropical op
   6.5  Safe reinforcement learning
       - PWL value function from deep tropical net admits exact polytopic CBF
       - Constraint satisfaction via tropical variety as safety boundary

7. Discussion
   7.1  When to use Path A (hybrid) vs Path B (deep tropical)
   7.2  Computational challenges and the case for a tropical CUDA kernel
   7.3  The open problem: tropical backpropagation without STE bias
   7.4  Tropical recurrent networks (future work)
   7.5  Tropical graph neural networks (future work)

8. Conclusion

References (target: 50–70 citations)

Appendix
   A.  Proof of Theorem 4.1 (linear region count)
   B.  Proof of Theorem 4.2 (tropical BN stability)
   C.  Proof of Theorem 4.4 (deep tropical = PWA function)
   D.  Tropical eigenvalue algorithm
   E.  Full architecture hyperparameter tables
   F.  Synthetic PWA benchmark generation procedure
   G.  Code, data, reproducibility statement
```

---

## §6  Research Frontiers: Where Tropical Algebra Applies Across ML/AI

These are the domains where the tropical geometric inductive bias is the strongest
natural fit. Ranked roughly by expected impact and proximity to existing work.

---

### 6.1  Reinforcement Learning and Optimal Control ★★★★★

**Why it fits:** The Bellman optimality equation is a tropical operation in disguise.
Value functions in deterministic control are naturally PWL (especially under linear
constraints). Q-learning computes `max_a Q(s,a)` — that max is tropical addition.

**Specific applications:**
- **Piecewise-affine MPC:** Replace the neural value function approximation in
  model predictive control with a deep tropical net. The learned V(x) is exactly PWL,
  so constraint satisfaction near switching boundaries is exact rather than approximate.
- **Tropical Q-networks:** Replace the Q-network in DQN with a deep tropical net.
  The argmax over actions is already tropical — the value function should be too.
- **Max-plus policy gradient:** Derive a policy gradient theorem in the tropical
  semiring. The score function estimator becomes a subgradient.
- **Hybrid system reachability:** Use tropical layers to learn the reachable set
  polytope of a hybrid automaton from trajectory data.

**Nearest existing work:** MaxEnt RL (Haarnoja et al.), soft Q-learning, PWA system
identification literature.

---

### 6.2  Combinatorial Optimization and Graph Learning ★★★★★

**Why it fits:** Graph algorithms (shortest path, min spanning tree, max flow, matching)
are all tropical operations on adjacency matrices. GNNs approximate these with smooth
message passing — tropical GNNs would compute them exactly.

**Specific applications:**
- **Tropical GNN:** Replace the message aggregation `sum_j(W · h_j)` with
  `max_j(W + h_j)`. This makes the GNN natively compute max-plus matrix powers,
  which correspond exactly to shortest-path distances.
- **Learning combinatorial algorithms:** Train a tropical net to imitate dynamic
  programming (Viterbi, Dijkstra, Bellman-Ford). The network structure should match
  the DP table structure naturally.
- **Neural combinatorial optimization:** The pointer network and attention-based TSP
  solvers use softmax routing. Replace with tropical Maslov attention — the hard
  routing property helps with discrete combinatorial structure.
- **Satisfiability and constraint propagation:** Tropical algebra over Boolean
  semirings is related to logic programming and constraint solving.

**Nearest existing work:** Combinatorial optimization with neural networks (Bello et al.,
Kool et al.), algorithmic reasoning (Veličković et al.).

---

### 6.3  Manufacturing, Logistics, and Scheduling ★★★★☆

**Why it fits:** This is the original application domain of max-plus algebra. Timed
event graphs, job shops, pipeline throughput — all are governed by max-plus linear
systems whose steady-state behavior is characterized by the tropical eigenvalue.

**Specific applications:**
- **End-to-end schedule optimization:** Train a tropical net to predict optimal
  schedules directly from job parameters. The tropical eigenvalue of the learned
  weight matrix should converge to the true system throughput.
- **Supply chain optimization:** Multi-echelon inventory problems involve
  `max(demand, replenishment)` operations — natively tropical.
- **Traffic flow:** Signalized intersection timing and merge decisions are
  max-plus operations on vehicle arrival times.
- **Tropical system identification:** Given observed event sequences from a
  manufacturing line, recover the max-plus system matrix. Deep tropical net
  does this end-to-end from data.

---

### 6.4  Robust and Worst-Case Learning ★★★★☆

**Why it fits:** The min-plus dual of max-plus algebra is the algebraic structure of
worst-case analysis. H-infinity control minimizes the worst-case disturbance-to-output
gain — a min-plus (tropical) problem. Minimax optimization problems over compact sets
are tropical at their core.

**Specific applications:**
- **Adversarial robustness certification:** The worst-case adversarial perturbation
  within an L-inf ball is a min-plus operation over the ball boundary. Tropical layers
  may provide tighter certified bounds than classical interval arithmetic.
- **Distributionally robust optimization:** The worst-case expectation over a
  Wasserstein ball involves a tropical-like supremum.
- **Morphological data augmentation:** Tropical convolution (dilation/erosion)
  augments data by structuring element deformation — this is adversarial augmentation
  in the tropical sense.
- **Minimax game-theoretic ML:** In GANs, the discriminator's objective is
  `max_D min_G` — a nested tropical structure.

---

### 6.5  Signal Processing and Time-Series ★★★☆☆

**Why it fits:** Max-filtering (running maximum over a window) is tropical convolution.
Morphological operators are used in seismic signal processing, biomedical signal
analysis, and audio. Replacing classical conv layers with tropical conv layers in
time-series models gives robustness to outliers (max is robust to all but the largest
value — it ignores small perturbations that don't exceed the maximum).

**Specific applications:**
- **Tropical recurrent networks:** Replace LSTM cell operations `h·W + x·U` with
  tropical analogs. The "forget gate" becomes a tropical mask (set to −∞ = forget).
- **Robust audio feature extraction:** Morphological convolution on spectrograms
  to extract envelope and formant structure robustly.
- **Anomaly detection:** The tropical eigenvalue of a time-series' learned
  system matrix changes when the underlying dynamics change — a natural
  change-point detector.
- **Quantization and compression:** Tropical networks are inherently quantization-
  friendly because the max operation is unchanged by small perturbations of its
  inputs (so long as the winner doesn't change). This may enable very aggressive
  weight quantization.

---

### 6.6  Interpretability and Explainable AI ★★★★☆

**Why it fits:** The defining feature of tropical networks — hard routing through
a polyhedral partition — is interpretable by construction. You can always ask "which
partition cell is this input in?" and "which tropical path did it follow?" This is
fundamentally more interpretable than attention weights or feature attributions in
smooth networks.

**Specific applications:**
- **Decision rule extraction:** The polyhedral cells of a trained tropical net are
  exact linear regions with known boundaries — they are rules in `max(W·x + b)` form,
  directly extractable as human-readable conditions.
- **Routing visualization:** For any input, trace the sequence of winning j-indices
  through all tropical layers. This gives an exact computational path — a "proof"
  of why the network made its prediction.
- **Tropical SHAP values:** Define a tropical analog of Shapley values using the
  tropical semiring's additivity properties. The contribution of feature i to output k
  is well-defined: it is non-zero only if feature i is on the winning path.
- **Concept bottleneck models:** Each tropical cell can be labeled as a semantic
  concept if the routing structure aligns with human-interpretable categories.

---

### 6.7  Spiking Neural Networks and Neuromorphic Computing ★★★☆☆

**Why it fits:** Spiking networks communicate via discrete events (spikes) rather than
continuous activations. The timing of spikes is naturally a max-plus quantity — the
time at which a neuron fires depends on the maximum of its weighted input arrival times.
Max-plus algebra is the native language of timed event systems.

**Specific applications:**
- **Tropical spike-timing-dependent plasticity:** The STDP learning rule updates
  weights based on spike timing differences — a temporal difference that is naturally
  a max-plus expression.
- **Neuromorphic hardware mapping:** Tropical layers map very efficiently to
  neuromorphic hardware (Intel Loihi, IBM TrueNorth) where the max operation
  over synaptic inputs is the native computation.
- **Energy-efficient inference:** On tropical hardware, the argmax computation
  requires no multiplications — only comparisons and additions. This is potentially
  orders of magnitude more energy-efficient than classical matrix multiply.

---

### 6.8  Multi-Agent Systems and Game Theory ★★★☆☆

**Why it fits:** Nash equilibrium computation, minimax game solving, and multi-agent
coordination all involve nested max/min operations — tropical structures.

**Specific applications:**
- **Tropical Nash equilibrium:** The Nash equilibrium of a zero-sum game is a
  fixed point of a max-min tropical map. A deep tropical net could learn approximate
  equilibria more naturally than a classical network.
- **Mechanism design:** Auction theory (VCG, Myerson) involves max over agent
  valuations — tropical operations. Tropical nets for revenue-optimal auction learning.
- **Multi-agent path planning:** Decentralized path planning with collision avoidance
  involves max over agent positions — a natural tropical structure.

---

### Priority Ranking for Next Papers (beyond the two outlined above)

| Rank | Domain | Rationale |
|---|---|---|
| 1 | RL / Optimal Control | Direct connection to lead engineer's background; tropical Bellman has formal grounding |
| 2 | Combinatorial Optimization | Tropical GNNs are a natural next paper after establishing the architecture |
| 3 | Interpretability | Low additional implementation cost; high practical impact |
| 4 | Robust/Worst-Case Learning | Connects to H-infinity control — another direct background link |
| 5 | Scheduling/Max-Plus Systems | The original application domain; tropical eigenvalue monitoring is the hook |

---

## §7  Agent Task Queues

### Path A task queue (ordered)

```
[ ] 1.  Validate tropformer.py against all 5 gates in §1
[ ] 2.  Run 3-seed reproducibility test; create results/ directory with logs
[ ] 3.  Run classical ablation (replace TropicalLinear with nn.Linear); compare
[ ] 4.  Refactor: extract TropicalEncoderLayer from TropFormer
[ ] 5.  Implement TropicalDecoderLayer with cross-attention (§2.3 spec)
[ ] 6.  Implement make_causal_mask() with -1e9 masking (§2.4)
[ ] 7.  Implement TropicalEncoder and TropicalDecoder stacks
[ ] 8.  Add batch_first parameter throughout
[ ] 9.  Implement TropicalKVCache (§2.5)
[ ] 10. Test: seq2seq task (small machine translation dataset)
[ ] 11. Implement TropFormerConfig (PretrainedConfig subclass)
[ ] 12. Implement TropFormerModel (PreTrainedModel subclass)
[ ] 13. Register with AutoModel / AutoConfig
[ ] 14. Benchmark: SST-2, CoNLL-2003 (§2.7 benchmark plan)
[ ] 15. Run full ablation study per Paper 1 §5.3
[ ] 16. Produce routing visualization per Paper 1 §5.5
[ ] 17. Write Paper 1 per §4 outline
```

### Path B task queue (ordered)

```
[ ] 1.  Path A tasks 1–3 must be complete first
[ ] 2.  Implement TropicalBatchNorm (§3.2)
[ ] 3.  Implement TropicalLinearSTE with softmax-weighted backward (§3.1)
[ ] 4.  Implement tropical_residual with learnable center (§3.1)
[ ] 5.  Implement DeepTropicalBlock (§3.2)
[ ] 6.  Test: train 8-layer DeepTropicalBlock stack on MNIST — must train stably
[ ] 7.  Ablation: remove each gradient stabilization mechanism — confirm death
[ ] 8.  Implement TropicalConv1d (§3.4 morphological conv)
[ ] 9.  Implement deep tropical attention with tropical V aggregation (§3.3)
[ ] 10. Implement tropical_cross_entropy and tropical_contrastive_loss (§3.5)
[ ] 11. Implement DeepTropNet full model (§3.6)
[ ] 12. Implement tropical eigenvalue monitor (Paper 2 §5.6)
[ ] 13. Synthetic PWA benchmark: generate dataset, train, measure partition recovery
[ ] 14. Scheduling benchmark (§3.7)
[ ] 15. CIFAR-10 benchmark comparison
[ ] 16. Write Paper 2 per §5 outline
```

---

*Last updated: generated from architecture and roadmap session.*
*Companion file: `TROPFORMER_CONTEXT.md`*
*Primary source: `tropformer.py`*