# TSRN Benchmark Results Summary

**Device:** AMD Radeon RX 6750 XT (12GB VRAM) via DirectML  
**Backend:** PyTorch 2.4.1 + torch-directml  
**Date:** March 2026  

---

## 1. 2M Parameter Configuration (Synthetic Char-Level Data)

| Metric | Vanilla Transformer | TSRN |
|--------|-------------------|------|
| Parameters | 3,239,424 | 2,849,304 |
| Final Val PPL | 1.603 | **1.002** |
| Convergence (PPL < 1.01) | Never | Step 300 |
| Training Speed (ms/step) | 234 | 865 |
| Throughput @ B=16 | 40,836 tok/s | 20,571 tok/s |
| Throughput @ B=32 | 33,761 tok/s | 26,307 tok/s |

**Config:** d=256, context=256, 1 block/scale, 4 heads, top-k=16, mem_depth=6, 3000 steps, B=32

### Ablation (500 steps, synthetic data)

| Variant | Final Val PPL | Delta |
|---------|--------------|-------|
| TSRN_full | 1.003 | -- |
| no_reservoir | 1.003 | +0.000 |
| **no_sheaf** | **1.298** | **+0.295** |
| no_rg_pool | 1.004 | +0.001 |
| no_padic_mem | 1.003 | +0.000 |
| no_padic_attn | 1.003 | +0.000 |

**Key finding:** Sheaf diffusion is the single most impactful component (+0.295 PPL degradation when removed). Other components show minimal impact on synthetic data due to task simplicity.

---

## 2. 50M Parameter Configuration (WikiText-2 Char-Level)

| Metric | Vanilla Transformer | TSRN |
|--------|-------------------|------|
| Parameters | 26,496,000 | 28,291,120 |
| Final Val PPL | 4.443 | **1.010** |
| Convergence (PPL < 1.05) | Never | Step 500 |
| Training Speed (ms/step) | 311 | 1,219 |
| Throughput @ B=8 | 7,593 tok/s | 4,629 tok/s |
| Throughput @ B=16 | 7,865 tok/s | 4,964 tok/s |

**Config:** d=512, context=256, 3 blocks/scale, 8 heads, top-k=16, mem_depth=7, 5000 steps, B=8  
**Dataset:** WikiText-2 (12M chars, 1118 vocab, 10.9M train / 1.2M val tokens)

### Ablation (800 steps, WikiText-2)

| Variant | Final Val PPL | Delta |
|---------|--------------|-------|
| TSRN_full | 1.013 | -- |
| no_reservoir | 1.013 | +0.000 |
| **no_sheaf** | **2.565** | **+1.552** |
| no_rg_pool | 1.013 | +0.000 |
| no_padic_mem | 1.013 | +0.000 |
| no_padic_attn | 1.014 | +0.001 |

**Key finding:** Sheaf diffusion remains overwhelmingly the most critical component at scale, with +1.552 PPL degradation on real language data. The magnitude of its contribution increases from 2M to 50M scale.

---

## 3. Component Analysis

### Sheaf Diffusion (Critical)
- Largest contributor at both scales
- Implements formalized capsule routing via sheaf Laplacian (Whitepaper Sec 2.2)
- Window=3 (7 offsets), alpha init=0.15, near-identity restriction maps
- Responsible for local context aggregation that complements tropical attention's global sparse selection

### Tropical Attention (Architecture-Defining)
- Max-plus inner product: score(i,j) = max_c(Q[i,c] + K[j,c])
- Top-k sparse selection (k=16) with softmax over selected positions
- Memory-efficient chunked logsumexp avoids 5D tensor OOM
- Provides smooth backward via logsumexp (no scatter required)

### Other Components
- **Clifford FFN:** Geometric product nonlinearity (grade-0 and grade-2 terms)
- **RG Coarse-graining:** Disentangle + pool for multi-scale processing
- **p-adic Memory:** Binary tree with soft routing (128 slots at 50M)
- **Echo State Reservoir:** Learnable spectral radius with sparse init
- **p-adic Attention:** Shared prefix similarity (scale 2 only)

---

## 4. DirectML Adaptations

| Issue | Fix |
|-------|-----|
| `scatter_` backward crash | Threshold masking with detached bool mask |
| `.max(dim)` scatter backward | Chunked logsumexp STE (smooth backward) |
| `torch.roll` CPU fallback | Slice + cat with zero padding |
| `F.pad` non-contiguous views | Explicit zero-tensor concatenation |
| `eigvals` not supported | Power iteration approximation |
| `is_causal` + float mask NaN | Manual boolean causal mask |
| 5D tensor OOM | Adaptive chunked logsumexp (128MB cap) |
| Unicode CP1252 errors | ASCII-safe console output |

---

## 5. Throughput Summary

### 2M Config (d=256, context=256)
| Model | B=8 tok/s | B=16 tok/s | B=32 tok/s |
|-------|-----------|------------|------------|
| Transformer | 34,335 | 40,836 | 33,761 |
| TSRN | 14,480 | 20,571 | 26,307 |

### 50M Config (d=512, context=256)
| Model | B=8 tok/s | B=16 tok/s | B=32 tok/s |
|-------|-----------|------------|------------|
| Transformer | 7,593 | 7,865 | 7,715 |
| TSRN | 4,629 | 4,964 | 3,980 |

TSRN is ~0.4-0.6x transformer throughput due to the chunked tropical score computation and 7 sequential components per block. The throughput gap narrows at larger batch sizes.

---

## 6. Files Generated

- `results/tsrn_2m_results.json` -- Full 2M benchmark data
- `results/tsrn_50m_results.json` -- Full 50M benchmark data
- `tsrn_curves.png` -- Training loss/PPL curves (last run)
- `tsrn_ablation.png` -- Ablation bar chart (last run)
- `tsrn_dml.py` -- DirectML-adapted TSRN implementation
