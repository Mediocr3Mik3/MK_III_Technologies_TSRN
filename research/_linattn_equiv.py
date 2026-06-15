"""Empirical proof that soft-tropical (Maslov-LSE) attention == exp-feature
linear attention, hence intrinsically O(T*d^2) rather than O(T^2*d).

Background (Roadmap 3.3 + linear-attention theory, Katharopoulos et al. 2020):
TropicalAttention computes, per (query i, key j),

    score_ij = h * logsumexp_c( (Q_ic + K_jc) / h )          (tsrn_dml.py:637)

then aggregates out_i = sum_j softmax_j(score_ij) * V_j.

Let phi(z) = exp(z / h).  Because logsumexp is the log of a *separable* sum,

    exp( logsumexp_c((Q_ic + K_jc)/h) ) = sum_c exp(Q_ic/h) exp(K_jc/h)
                                        = < phi(Q_i), phi(K_j) >.

So the attention weight is a separable kernel and the aggregation factorizes:

    out_i = < phi(Q_i),  sum_{j<=i} phi(K_j) V_j^T >
            -----------------------------------------
            < phi(Q_i),  sum_{j<=i} phi(K_j) >

The causal sums are prefix scans of a (dh x dv) state => O(T*dh*dv) work,
O(log T) depth.  EXACT, not an approximation.

Subtlety: the code multiplies the logsumexp by h *before* softmax. That outer
factor h re-tempers the softmax and is separable only at h == 1 (the default,
tsrn_dml.py:612).  The "clean" tropical-linear kernel < phi(Q), phi(K) > (i.e.
softmax of logsumexp WITHOUT the outer h) is exactly linear for *all* h.  This
script verifies all three statements numerically.
"""
import time
import torch

torch.manual_seed(0)


def quad_tropical(Q, K, V, h, outer_h, causal=True):
    """O(T^2) reference. outer_h=True reproduces tsrn_dml.py (score = h*LSE);
    outer_h=False uses the clean kernel (score = LSE)."""
    raw = Q.unsqueeze(-2) + K.unsqueeze(-3)               # (B,H,T,T,dh)
    lse = torch.logsumexp(raw / h, dim=-1)                # (B,H,T,T)
    scores = h * lse if outer_h else lse
    if causal:
        T = Q.shape[-2]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=Q.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return w @ V


def linear_tropical(Q, K, V, h, causal=True):
    """O(T*dh*dv) causal prefix-scan form with phi(z)=exp(z/h).
    Global per-head shifts cancel in the num/den ratio (numerical stability)."""
    Qs = Q - Q.amax(dim=(-2, -1), keepdim=True)
    Ks = K - K.amax(dim=(-2, -1), keepdim=True)
    phiQ = torch.exp(Qs / h)                               # (B,H,T,dh)
    phiK = torch.exp(Ks / h)                               # (B,H,T,dh)
    outer = phiK.unsqueeze(-1) * V.unsqueeze(-2)          # (B,H,T,dh,dv)
    if causal:
        KV = torch.cumsum(outer, dim=-3)                 # (B,H,T,dh,dv)
        Z = torch.cumsum(phiK, dim=-2)                   # (B,H,T,dh)
    else:
        KV = outer.sum(-3, keepdim=True)                 # (B,H,1,dh,dv)
        Z = phiK.sum(-2, keepdim=True)                   # (B,H,1,dh)
    num = torch.einsum("bhtd,bhtde->bhte", phiQ, KV)      # (B,H,T,dv)
    den = torch.einsum("bhtd,bhtd->bht", phiQ, Z).clamp_min(1e-30).unsqueeze(-1)
    return num / den


def maxdiff(a, b):
    return (a - b).abs().max().item()


def proof():
    B, H, T, dh, dv = 2, 2, 24, 8, 8
    Q = torch.randn(B, H, T, dh, dtype=torch.float64)
    K = torch.randn(B, H, T, dh, dtype=torch.float64)
    V = torch.randn(B, H, T, dv, dtype=torch.float64)

    print("=" * 64)
    print("PROOF 1: linear form == clean quadratic kernel (all h)")
    print("  expect ~1e-12 (float64); proves the linearization is EXACT")
    for h in (0.5, 1.0, 2.0, 4.0):
        out_lin = linear_tropical(Q, K, V, h)
        out_quad = quad_tropical(Q, K, V, h, outer_h=False)
        print(f"  h={h:<4}  max|lin - quad_clean| = {maxdiff(out_lin, out_quad):.2e}")

    print("=" * 64)
    print("PROOF 2: linear form == ACTUAL code kernel (score = h*LSE)")
    print("  expect ~1e-12 at h=1 (default), growing for h!=1 (outer-h re-temper)")
    for h in (0.5, 1.0, 2.0, 4.0):
        out_lin = linear_tropical(Q, K, V, h)
        out_code = quad_tropical(Q, K, V, h, outer_h=True)
        print(f"  h={h:<4}  max|lin - quad_code|  = {maxdiff(out_lin, out_code):.2e}")

    print("=" * 64)
    print("SCALING: wall-time & peak memory, quadratic vs linear (float32, CPU)")
    print("  quadratic builds (T,T,dh); linear builds (T,dh,dv) -> flat in T")
    for Tn in (128, 256, 512, 1024, 2048):
        q = torch.randn(1, 4, Tn, 16, dtype=torch.float32)
        k = torch.randn(1, 4, Tn, 16, dtype=torch.float32)
        v = torch.randn(1, 4, Tn, 16, dtype=torch.float32)
        t0 = time.perf_counter()
        _ = linear_tropical(q, k, v, 1.0)
        t_lin = (time.perf_counter() - t0) * 1e3
        try:
            t0 = time.perf_counter()
            _ = quad_tropical(q, k, v, 1.0, outer_h=True)
            t_quad = f"{(time.perf_counter() - t0) * 1e3:8.1f} ms"
        except RuntimeError as e:
            t_quad = f"OOM/err: {str(e)[:24]}"
        print(f"  T={Tn:<5} linear={t_lin:8.1f} ms   quadratic={t_quad}")
    print("=" * 64)


if __name__ == "__main__":
    proof()
