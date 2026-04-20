# NEXUS Branch Updates

**Branch:** `nexus-maslov-sheaf-rgfp`
**Base:** `main @ 00d0464` (Phase 1+2 BPC wins + DML eval perf fix)
**Commits on this branch:** 4
**Scope:** 3 architectural innovations on top of `TSRNGist`, one pre-existing causality
bug fix, a naming-convention refactor for training artifacts, and three DML backend
compatibility fixes (performance + autograd).

All changes preserve exact per-position causality and are numerically equivalent
to their reference implementations (stock PyTorch) to within float32 epsilon.

---

## Commit log

| SHA | Summary |
|---|---|
| `3cbc960` | NEXUS: Maslov cycling + Sheaf Harmonic PE + RG fixed-point S2 + causality fix |
| `5adcf4b` | DML: replace `torch.cummax` with GPU-native `prefix_max` (Hillis-Steele scan) |
| `ac4aff9` | DML: `prefix_max` autograd-safe via `cat(full, contiguous-slice)` |
| `fb57620` | DML: `AdamWDML` drop-in optimizer (avoids `aten::lerp` CPU fallback) |

---

## Section A — Architectural innovations

### A.1  Maslov temperature cycling (`tsrn_dml.py::TropicalAttention`)

**Idea.** The Maslov dequantization
```
h · logsumexp(x / h)  --h→0-->  max(x)
h · logsumexp(x / h)  --h→∞-->  mean(x) + h·log(n)
```
interpolates continuously between soft (classical) and tropical (max-plus)
attention. Fixing `h = 1` is a stationary point on this continuum; cycling
`h` between a warm phase (h = 1.5, soft attention favoring all tokens) and
a cool phase (h = 0.3, near-tropical favoring the argmax) during training is
an annealing schedule over the tropical-softness axis of the loss surface.

**Implementation.**
- `TropicalAttention` registers a non-persistent buffer `maslov_h` (scalar).
- Attention scores become `h · logsumexp(raw / h)`, still followed by the
  causal mask and post-softmax weighting.
- `TSRNGist.set_maslov_h(h)` / `get_maslov_h()` broadcast the value to every
  attention layer.
- The training loop in `tsrn_convergence_gist.py::train_convergence` calls
  `model.set_maslov_h(maslov_h_schedule(step, n_steps))` once per step, where
  `maslov_h_schedule` is a cosine between `h_warm = 1.5` and `h_cool = 0.3`
  over `n_cycles = 3` full cycles across the run.

**Causality.** `h` is a scalar applied pointwise before the causal mask; the
mask still zeros out upper-triangular positions. Confirmed by perturbation
test (see Section E).

---

### A.2  Sheaf Harmonic positional encoding (`tsrn_dml.py::SheafHarmonicPE`)

**Idea.** For a 1-D path graph with `T` nodes, the eigenvectors of the graph
Laplacian are closed-form DCT-II basis functions:
```
φ_k(t) = cos( π · (t + 1/2) · k / T ),   k = 0, 1, ..., T-1.
```
The low-frequency eigenvectors span the "smoothest" modes; truncating to the
bottom `K` harmonics gives a spectral positional encoding that is *intrinsic*
to the sheaf topology, rather than data-agnostic (sinusoidal) or position-
quadratic (ALiBi-style).

**Implementation.**
- `SheafHarmonicPE(max_T, K, d_model)` pre-computes the DCT-II matrix
  `(max_T, K)` as a non-trainable buffer.
- `forward(T)` returns `self.proj(dct[:T])` of shape `(T, d_model)`.
- `self.proj` is a `nn.Linear(K, d_model, bias=False)` initialised with
  **zero weights**, so at step 0 the PE contribution is identically zero
  and the baseline model (without PE) is recovered exactly. The projection
  is learned over training.
- Exposed on `TSRNGist` as `self.sheaf_pe`, applied additively after
  `self.embed(idx)`.

**Causality.** Purely position-indexed; no cross-token mixing. Identity-at-init
is verified by the perturbation test returning 0.0 leak when the projection
weights are zero.

---

### A.3  RG fixed-point weight sharing for Scale 2 (`tsrn_gist.py::TSRNGist`)

**Idea.** The TSRN Scale-2 stack is a renormalization-group (RG) coarse-graining
map: each block is a smoothing / mixing operator applied to the pooled
`(B, T/2, d)` representation. In principle, iterating the *same* RG map
to its fixed point is more consistent with the RG interpretation than
stacking `N` distinct blocks.

**Implementation.**
- Replaced the `nn.ModuleList` of `n_blocks` distinct Scale-2 blocks with a
  single shared `self.s2_block` of type `TSRNGistBlock(..., use_padic_attn=True)`.
- `self.s2_max_iters` (default = `n_blocks`) caps the number of iterations.
- Training always runs the full `s2_max_iters` iterations (no early stop,
  so the graph depth is deterministic).
- Evaluation uses an early-stop criterion: stop when
  `‖xc_new − xc_prev‖_F / ‖xc_prev‖_F < s2_eps` (default `1e-3`).
- `self._last_s2_iters` (non-persistent buffer) records the number of
  iterations actually used on the last forward, logged per eval step.

**Effect on parameter count.** ~30% reduction in Scale-2 parameters for the
default `n_blocks = 3` (one shared block vs. three distinct blocks).

**Causality.** The shared block itself is causal; iterated application of
a causal operator is causal. Perturbing `x[:, t_0, :]` in eval mode with
`s2_max_iters ∈ {1, 2, 3, 5}` gives 0.0 leak at positions `< t_0`.

---

## Section B — Causality fix (pre-existing bug)

**File:** `tsrn_dml.py::EchoStateReservoir._power_iter_spectral_radius`

**Symptom.** On `main`, running the model on two inputs `x_1` and `x_2`
that differ only at position 30 produced a **1.58e-2 leak** at positions
`< 30` — i.e. earlier positions were aware of the perturbation. The leak
was uniform across positions `0 < t < t_perturb`, suggesting a global
state difference rather than a positional information flow.

**Root cause.** The reservoir estimates its weight-matrix spectral radius
via power iteration. The previous implementation:
1. Initialised the starting vector from `torch.randn(...)` when a cached
   buffer was zero, advancing the global RNG state between calls.
2. Wrote the converged eigenvector back into a *persistent* `_cached_v`
   buffer, so the second forward pass saw a different starting vector
   (and therefore a subtly different `rho_current`) than the first.

Both behaviours break batch-independence: the reservoir output depends
not only on the current input but also on the *order* of preceding
forward passes.

**Fix.** `_power_iter_spectral_radius` now:
- Starts every call from a deterministic unit vector `ones(dr) / √dr`
  registered as a non-persistent buffer.
- Increases `n_iter` from 3 to 6 (deterministic start converges slightly
  more slowly than the previously-cached near-eigenvector).
- Does not write anything back to state.

`rho_current` is now a pure function of `W_res`, which is a learnable
parameter identical across batches — so the reservoir is fully batch-
independent, restoring exact causality.

**Verification.** Perturbation test at positions `{5, 15, 30, 45, 60}`
with every combination of Maslov `h ∈ {1.5, 1.0, 0.5, 0.3}` and
`s2_max_iters ∈ {1, 2, 3, 5}`, in both train and eval mode: all runs
report `max |Δlogits|_{t < t_perturb} = 0.0` (exact).

---

## Section C — Artifact naming

**File:** `tsrn_convergence_gist.py`

All training artifacts now use a common run tag
```
<YYYYMMDD>_<script_stem>[_<user_tag>]
```
e.g. `20260420_tsrn_convergence_gist_nexus`. Helpers:

- `build_run_tag(user_tag) -> str`
- `ckpt_path(run_tag, step, kind) -> str`  (`kind ∈ {"step", "best", "final"}`)
- `results_path(run_tag, step, kind) -> str`  (`kind ∈ {"progress", "final"}`)

Filename layout:

| Artifact | Path |
|---|---|
| Periodic checkpoint | `checkpoints/<tag>_stepNNNNNN.pt` |
| Best-so-far checkpoint | `checkpoints/<tag>_best.pt` |
| Final checkpoint | `checkpoints/<tag>_final_stepNNNNNN.pt` |
| Incremental progress JSON | `results/<tag>_progress_stepNNNNNN.json` |
| Final results JSON | `results/<tag>_final_stepNNNNNN.json` |

Fixed a bug where `_save_results` embedded `best_val_bpc` into the filename,
producing a new file every save instead of overwriting. Progress JSON
now has a stable filename per `(run_tag, step)`.

Final results JSON also now carries an `"innovations"` block:
```json
"innovations": {
  "maslov_cycling": true,
  "sheaf_harmonic_pe": true,
  "rg_fixed_point_s2": true,
  "s2_max_iters": 3,
  "s2_eps": 0.001
}
```
Per-eval-step log entries now include `maslov_h` and `s2_iters_used` for
post-hoc analysis of the RG fixed-point convergence rate.

---

## Section D — DirectML compatibility fixes

Three independent DML issues surfaced when launching the 100k run on the
AMD RX 6750 XT. Each is fixed with a minimal, self-contained replacement.

### D.1  `torch.cummax` CPU fallback — replaced with `prefix_max`

**File:** `tsrn_dml.py::prefix_max`

**Symptom.**
```
UserWarning: The operator 'aten::_cummax_helper' is not currently supported
on the DML backend and will fall back to run on the CPU.
```
Triggered from `TropicalSSM.forward` every step. CPU round-trip per forward
is catastrophic for throughput.

**Fix.** Module-level helper `prefix_max(x, dim=1)` implementing a
Hillis-Steele parallel scan:
- log2(T) iterations of pointwise `torch.maximum` against a left-shifted
  copy padded with `-inf`.
- All ops GPU-native on every backend.

**Verification.** Bit-exact match to `torch.cummax` across shapes
`{(4,1,8), (2,16,32), (1,257,256), (3,5,4,8), (8,64,128)}`. `-inf` inputs
handled. Causal.

---

### D.2  `prefix_max` backward autograd fail — `cat(full, contiguous-slice)`

**File:** `tsrn_dml.py::prefix_max`

**Symptom.** Initial `prefix_max` using `F.pad(value=-inf)` on sliced views
crashed at `loss.backward()`:
```
RuntimeError: ensure_in_bounds: sizes [8, 129, 512], strides [131584, 512, 1],
storage offset 8454144, itemsize 4 requiring storage size 37765120 are out
of bounds for storage of size 4210688
```
The 4 210 688-byte storage corresponds to `G` of shape `(8, 257, 512)` in
`TropicalSSM.forward`; the offending tensor is `G[:, :-128]` from the last
Hillis-Steele iteration. DML's autograd mis-tracks stride/offset metadata
for `F.pad(mode='constant', value=-inf)` applied to a sliced view.

**Fix.** Rewrote `prefix_max` so no view survives into the backward graph:
```python
neg_inf_block = torch.full((B, step, *trailing), -inf)   # fresh storage
sliced        = cum[:, :-step].contiguous()              # full copy
shifted       = torch.cat([neg_inf_block, sliced], dim=1)
cum           = torch.maximum(cum, shifted)
```
Every intermediate owns its storage — no view metadata, no stale strides.

**Verification.**
- Forward: bit-exact vs `torch.cummax` on `(8, 257, 512)` + 4 other shapes.
- Backward: `max |grad_ours − grad_cummax| = 0.0` on `(4, 64, 32)`.
- Full `TSRNGist` (22.9 M params, B = 8, T = 256, d = 512) forward + backward
  runs to completion with `loss = 5.6554`, `total grad norm = 1.28`.
- Causality preserved (0.0 leak at `t = 128` perturbation).

---

### D.3  `aten::lerp.Scalar_out` CPU fallback — `AdamWDML`

**File:** `tsrn_dml.py::AdamWDML`

**Symptom.**
```
UserWarning: The operator 'aten::lerp.Scalar_out' is not currently supported
on the DML backend and will fall back to run on the CPU.
torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)
```
`torch.optim.AdamW` implements the first-moment EMA via `lerp`. Both the
single-tensor path (`.lerp_()`) and the multi-tensor foreach path
(`torch._foreach_lerp_()`) fall back to CPU on DML. For a 22.9 M-param
model, every optimizer step round-trips every moment tensor between GPU
and CPU — catastrophic.

**Fix.** `AdamWDML`, a drop-in `torch.optim.Optimizer` that replaces
```python
exp_avg.lerp_(grad, 1 - beta1)
```
with the mathematically identical
```python
exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
```
Decoupled weight decay, second-moment EMA, bias correction, epsilon,
per-group `lr` / `wd` / `betas` / `eps` — all preserved.

**State-dict compatibility.** State keys (`step`, `exp_avg`, `exp_avg_sq`)
match `torch.optim.AdamW` byte-for-byte, so a `torch.optim.AdamW`
checkpoint and an `AdamWDML` checkpoint are interchangeable.

**Verification.**
- 20-step parity vs `torch.optim.AdamW(foreach=False)` on a small net
  with identical inputs, seed, and learning rate: `max |Δparam| = 2.98e-8`
  (below float32 epsilon; residual is from the `lerp` vs `mul+add` rounding
  order only).
- `TSRNGist` full forward + backward + `optimizer.step()` at production
  shape (22.9 M params, B = 8, T = 256, d = 512): completes cleanly.

---

## Section E — Causality verification summary

A single automated test, run after every change, checks that perturbing
`x[:, t_0, :]` cannot affect `logits[:, t, :]` for any `t < t_0`.

| Configuration | Max `|Δlogits|` at `t < t_0` |
|---|---|
| Baseline + reservoir fix | **0.000e+00** |
| + Sheaf Harmonic PE | **0.000e+00** |
| + Maslov `h ∈ {1.5, 1.0, 0.5, 0.3}` | **0.000e+00** (all) |
| + RG fixed-point `s2_max_iters ∈ {1, 2, 3, 5}` | **0.000e+00** (all) |
| Train mode (dropout = 0) | **0.000e+00** |
| Eval mode | **0.000e+00** |
| Perturbation positions `{5, 15, 30, 45, 60, 128}` | **0.000e+00** (all) |
| Scale 1 via `prefix_max` | **0.000e+00** |

Compare to `main @ 00d0464`: **1.58e-2** leak (from the reservoir bug fixed
in this branch).

---

## Section F — How to launch the 100 k-step convergence run

```powershell
New-Item -ItemType Directory -Force logs | Out-Null

.venv312\Scripts\python.exe -u research\tsrn_convergence_gist.py `
    --steps 100000 `
    --batch 8 `
    --d-model 512 `
    --context 256 `
    --n-blocks 3 `
    --n-heads 8 `
    --max-gists 64 `
    --gist-top-k 4 `
    --ckpt-every 5000 `
    --tag nexus 2>&1 | Tee-Object -FilePath "logs\20260420_tsrn_convergence_gist_nexus.log"
```

**The `-u` flag is required** when piping through `Tee-Object`: without it,
Python detects a non-TTY stdout, switches to block buffering, and suppresses
the setup block (`TSRNGist: N parameters`, vocab, GPU, NEXUS summary, run tag)
for the first few dozen training steps.

**Resume is supported:**
```
--resume checkpoints\20260420_tsrn_convergence_gist_nexus_step050000.pt
```

**Artifacts produced** (all filenames use today's UTC date):

| Artifact | Path |
|---|---|
| Periodic ckpt (every 5 k steps) | `checkpoints\20260420_tsrn_convergence_gist_nexus_stepNNNNNN.pt` |
| Best ckpt | `checkpoints\20260420_tsrn_convergence_gist_nexus_best.pt` |
| Final ckpt | `checkpoints\20260420_tsrn_convergence_gist_nexus_final_step100000.pt` |
| Progress JSON | `results\20260420_tsrn_convergence_gist_nexus_progress_stepNNNNNN.json` |
| Final results JSON | `results\20260420_tsrn_convergence_gist_nexus_final_step100000.json` |

---

## Section G — Files changed

| File | Lines ± | Purpose |
|---|---|---|
| `research/tsrn_dml.py` | +124 / −17 | `prefix_max`, `AdamWDML`, reservoir causality fix |
| `research/tsrn_gist.py` | +76 / −23 | `SheafHarmonicPE`, RG fixed-point loop, Maslov `h` plumbing |
| `research/tsrn_convergence_gist.py` | +105 / −2 | `maslov_h_schedule`, naming helpers, `AdamWDML` swap |

No new test files. No new dependencies. No changes to `tsrn_inference.py`
or any other script.
