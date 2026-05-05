# TSRN Data-Leakage Audit

**Branch:** `kleene-star`
**Scope:** Every component in `research/tsrn_dml.py`, `research/tsrn_gist.py`,
`research/padic_context_scaling.py`, plus dataset / evaluation paths.
**Methodology:** Trace every information channel between time positions
(attention, recurrence, pooling, cross-batch buffers, statistics, eval splits)
and verify that output at position *t* depends only on input at positions ≤ *t*
within the same forward pass, and that train/val/test splits never bleed.

---

## Summary

| Severity | Component | Status |
|---|---|---|
| 🔴 **HIGH** | `KleeneSSM.forward` — `delta.mean(dim=(0,1))` | **FIX REQUIRED** |
| 🟡 **MED** | `TSRNGist.forward` — `ctx_summary = x.mean(dim=1)` | **FIX REQUIRED** |
| 🟢 OK | Everything else (12 components verified) | Pass |

Both leaks are **subtle global-statistic leaks**: position-wise outputs depend
on a scalar/vector that pools across **all** time positions of the current
window, including positions ≥ *t*. Neither is a catastrophic teacher-forcing
bug (the dominant signal flow is correct), but both can inflate training
metrics relative to true autoregressive performance.

Both predate the `kleene-star` branch in spirit (the Gist mean-query is in
`main`; the KleeneSSM mean-delta was introduced in this branch as a
deliberate "approximation" comment).

---

## 🔴 HIGH — `KleeneSSM` global-mean selectivity (`research/tsrn_dml.py`, line 1347)

### The leak

```python
delta = F.softplus(self.delta_proj(x))            # (B, T, d_state)
scale = (1.0 - delta.mean(dim=(0, 1))).clamp_min(0.05)  # (d_state,)  ← averages over ALL T
A_scaled = self.A * scale.unsqueeze(-1)
A_star = self.compute_kleene_star(A_scaled)
```

`delta.mean(dim=(0, 1))` is a scalar-per-state-dim **averaged over the entire
time axis**, so `A_scaled` (and therefore `A_star`) carries information from
positions *t' > t* into the computation of `H_t`.

### Author's defense (incorrect)

The class docstring (line 1244) claims:

> Causality: enforced by the structure of U... The Kleene star A* is
> position-independent and does not introduce future info.

`A_star` is technically position-independent, but it is **input-dependent**
on a global statistic of the input — which couples future inputs to past
outputs. The existing test `test_kleene_ssm_causality` measures this leak at
`max_leak ≈ 1e-4` and accepts it under `< 1e-3`, with a comment dismissing
it as "not a true future-info leak". This is a **misclassification** —
it *is* a leak.

### Severity

- **Magnitude per token:** small (the mean is a stable global statistic; one
  perturbed token shifts it by ~1/T).
- **Cumulative:** the model can learn to encode a few bits of "what comes
  later in the window" into the global mean during training, then exploit
  those bits at every position.
- **Impact:** train BPC may be optimistically biased; sequential eval
  (windowed, no overlap) is unaffected because each window's mean depends
  only on tokens inside that window, but **same-window prediction at
  position *t* still leaks from positions *t+1..T-1***.

### Fix (applied in commit X)

Replace the global-mean scaling with **per-position causal selectivity** on
the diagonal of A, which still factors through a parallel prefix scan:

```python
delta = F.softplus(self.delta_proj(x))                     # (B, T, ds)
A_diag_const = torch.diagonal(self.A).clamp(max=0.0)       # (ds,)
# Per-position decay (input-dependent, but pointwise per t — fully causal).
A_diag_t = A_diag_const.unsqueeze(0).unsqueeze(0) * (1.0 - delta).clamp_min(0.05)
# Inclusive cumulative sum: S_t = sum_{r=0..t} A_diag_t[r]
S = A_diag_t.cumsum(dim=1)                                 # (B, T, ds)
shifted = U - S                                            # (B, T, ds)
cum = prefix_max(shifted, dim=1)                           # (B, T, ds)
H_diag = S + cum                                           # (B, T, ds)

# Kleene star uses input-INDEPENDENT A — no future leak through cross-state mixing.
A_star = self.compute_kleene_star(self.A)                  # (ds, ds)
```

Trade-off: cross-state mixing structure is no longer adapted to the input
window. The author's "approximation" was meant to give input-dependent
mixing cheaply; this fix sacrifices that for correctness. Per-position
per-state Kleene would cost `O(T·d²·log d)` (rejected for cost). The
diagonal recurrence retains the per-position selectivity gradients via
`delta`, which is the bulk of the Mamba S6 benefit.

---

## 🟡 MED — `TSRNGist` gist-retrieval query (`research/tsrn_gist.py`, line 731)

### The leak

```python
ctx_summary = x.mean(dim=1)  # B d  (embedding mean; no future info here)
gist_theta, gist_mag, gist_w = self.gist_buffer.retrieve(
    self.gist_buffer.key_proj(ctx_summary), top_k=self.gist_top_k)
```

The comment "no future info here" is wrong. `x.mean(dim=1)` averages over
**all T positions of the current window**. This summary is then projected
to a query that selects which past-window gists are retrieved. The
*content* of the retrieved gists is causal (stored from strictly past
windows). The *selection* is non-causal: at position *t*, the model
attends to a soft-weighted mixture of past gists where the weights
depend on tokens at positions *t' > t* of the current window.

### Severity

- The gist contribution is gated through `gist_rotation` and a
  `GistCrossAttention` whose `Wo.weight` is zero-initialized (line 411) —
  so at start-of-training the leak amplitude is zero, and the model has
  to *learn* to use this channel.
- Effect during training: the model can learn to encode a few bits of
  "what's in the rest of this window" into the gist soft-weights and
  exploit them at each position.
- Effect during eval: `gist_buffer.reset()` is called before evaluation
  (verified in `train_cloud.py` lines 429, 553, 560), so eval starts from
  an empty buffer, but the **per-window mean-query leak still applies**
  at every eval window after the first.

### Fix (applied in commit X)

Replace the mean with the **first-position embedding**, which every
position can causally see:

```python
ctx_summary = x[:, 0, :]
```

Trade-off: `x[:, 0, :]` is a less informative summary than the mean.
Counter-arguments:

- For long context windows, position 0 is often a section/paragraph
  start — already a strong topical anchor.
- Gist storage uses `forward_single(x)` which still attends causally to
  the entire current window — that storage key is correct because by the
  time the *next* batch retrieves it, the entire window IS in the past.
  Only the within-window query needed the fix.
- Alternative: a learnable position-independent query parameter (zero
  input dependence). Rejected — would lose all per-window adaptiveness.

---

## 🟢 Verified causal — full list

Every component below was traced and confirmed not to leak:

### Attention paths
- **`TropicalAttention`** — strict triangular mask before topk + softmax;
  cross-window cache only enabled at inference (default off).
- **`KleeneAttention`** — `causal_mask` (-1e9 upper triangle) added BEFORE
  Kleene-star squaring. Mathematical proof: for upper-triangle target
  `(i,j)` with `j>i`, every two-hop path requires `result[i,k]` finite
  (`k≤i`) AND `result[k,j]` finite (`j≤k`). Together: `j≤k≤i`, but `j>i`
  has no solution. So upper-triangle stays at -1e9 forever. ✓
- **`PAdicAttention`** — `triu(diagonal=1)` mask before softmax. ✓
- **`GistCrossAttention`** Scale-1 path — attends to retrieved past-window
  gists (K-shaped, K << T); past windows are strictly causal. ✓
- **`GistCrossAttention`** Scale-2 path — diagonal/elementwise gating
  (`x * gate(gist_repr)`), no cross-position mixing. ✓
- **`GistExtractor.forward`** — explicit `t > 2j` mask: coarse position
  *j* sees fine positions 0..2j only. ✓
- **`GistExtractor.forward_single`** — used only for **storage** into the
  cross-batch buffer. The stored gist represents the entire window, but
  retrieval happens in *future* batches where the window is in the past. ✓

### Recurrence paths
- **`TropicalSSM`** — pointwise `gate_B(x_t)`, `gate_out(x_t)`; cumulative
  prefix-max along time; no cross-position statistics. ✓
- **`EchoStateReservoir`** — explicit Python loop `for t in range(T): h =
  …`; spectral-radius power iter uses detached W only. ✓

### Pooling / coarse-graining
- **`RGPool`** — explicit causal pairing via left-zero-pad shift: coarse
  *j* depends on `(x_{2j-1}, x_{2j})`; latest fine index `2j ≤ 2j`. ✓
- **`SheafRotorDiffusion`** — `causal=True` default → `offsets =
  range(-window, 1)` (only past + current neighbors). ✓
- **`SheafHarmonicPE`** — position-only, no input mixing. ✓
- **`RotaryPositionEmbedding`** — pointwise per-token rotation. ✓

### Memory
- **`PAdicMemory`** — leaf K/V are learnable parameters (no input
  history), per-token attention; ✓.

### Cross-batch state
- **`PersistentCrossWindowMemory`** — disabled by default; only enabled
  outside training; caches *after* the current window's attention has
  finished. ✓
- **`GistBuffer`** — retrieve-then-store ordering inside
  `TSRNGist.forward` (lines 732 → 764) ensures the current window never
  retrieves itself. ✓

### Position scaling
- **`PAdicContextScaling` (PaCS)** — only active at inference
  (`not self.training` AND `inference_ctx > training_ctx`); position
  function only, no input mixing. ✓

### Data
- **`load_enwik8`** — exact byte split 90M / 5M / 5M (community standard,
  Al-Rfou 2019 / Transformer-XL / SHA-RNN). ✓
- **`CharDatasetSplit.batch`** — selects exclusively from train, val, or
  test depending on `split` arg; no cross-split sampling. ✓
- **Vocabulary built from train+val+test** — standard for char-level LM
  (each byte gets a deterministic ID); not a leak.
- **`evaluate`** — random sampling within a single split; ✓.
- **`evaluate_sequential`** — non-overlapping `[i, i+ctx)` windows over
  exactly the requested split, weighted-mean over actual token counts;
  Transformer-XL / SHA-RNN protocol. ✓

---

## Test changes

After the fixes:

- `test_kleene_ssm_causality` threshold tightened from `< 1e-3` to
  `< 1e-5` (any remaining residual is float32 precision only).
- New test `test_tsrn_gist_query_causality` — perturbs `x[:, t0, :]` for
  `t0 > 0`, asserts `ctx_summary` is unchanged.
- New test `test_kleene_ssm_per_position_selectivity` — verifies that
  changing input at position *t* DOES change output at position *t*
  (otherwise the fix would have removed the selectivity entirely).

---

## What was *not* audited

Out of scope for this audit:

- The DPO / SFT / pretrain scripts under `research/sft_train.py`,
  `research/dpo_train.py`, `research/train_pretrain.py`. These were
  staged but uncommitted on `kleene-star` and are unrelated to the
  current cloud workflow.
- The `download_all.py` data preparation pipeline (this just stages bytes
  on disk; any leakage would be in how the trainer slices them, which IS
  audited above for enwik8).
- v2.0 hyperbolic-gist chaining buffer beyond noting it stores hyperbolic
  projections of the same causal `forward_single` output; same causality
  guarantee transitively applies.
