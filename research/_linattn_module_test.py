"""Validate TropicalAttention._linear_attend (chunked O(T) scan) against a
brute-force quadratic reference with the SAME ALiBi decay + phi(z)=exp(z/h),
and verify strict causality. CPU/float64.

Run: .venv312\\Scripts\\python.exe research\\_linattn_module_test.py
"""
import torch
from tsrn_dml import TropicalAttention

torch.manual_seed(0)


def ref(Q, K, V, h, gamma):
    """O(T^2) reference: w_ij = <phi(Q_i),phi(K_j)> * gamma^(i-j) for j<=i."""
    B, H, T, dh = Q.shape
    phiQ = torch.exp((Q - Q.amax((-2, -1), keepdim=True)) / h)
    phiK = torch.exp((K - K.amax((-2, -1), keepdim=True)) / h)
    Kern = torch.einsum("bhid,bhjd->bhij", phiQ, phiK)        # (B,H,T,T)
    i = torch.arange(T).view(T, 1)
    j = torch.arange(T).view(1, T)
    rel = (i - j)
    decay = gamma.view(1, H, 1, 1) ** rel.clamp(min=0).double()
    W = Kern * decay * (rel >= 0).double()
    num = torch.einsum("bhij,bhjd->bhid", W, V)
    den = W.sum(-1, keepdim=True).clamp_min(1e-20)
    return num / den                                          # (B,H,T,dv)


def main():
    B, H, T, dh = 2, 4, 24, 8
    d = H * dh
    att = TropicalAttention(d_model=d, n_heads=H, linear_attn=True).double().eval()
    gamma = torch.exp(-att.alibi_slopes.double())

    Q = torch.randn(B, H, T, dh, dtype=torch.float64)
    K = torch.randn(B, H, T, dh, dtype=torch.float64)
    V = torch.randn(B, H, T, dh, dtype=torch.float64)

    print("Equivalence of chunked scan vs quadratic reference (float64):")
    for h in (1.0, 0.7, 1.5):
        att.maslov_h.fill_(h)
        out_ref = ref(Q, K, V, h, gamma)                      # (B,H,T,dv)
        for C in (T, 8, 7, 1):
            att.linear_chunk = C
            o = att._linear_attend(Q, K, V)                   # (B,T,d)
            o = o.view(B, T, H, dh).permute(0, 2, 1, 3)       # (B,H,T,dv)
            err = (o - out_ref).abs().max().item()
            flag = "OK" if err < 1e-8 else "FAIL"
            print(f"  h={h:<4} chunk={C:<3} max|err|={err:.2e}  [{flag}]")

    print("Causality (perturbing future tokens must not change past outputs):")
    att.maslov_h.fill_(1.0)
    att.linear_chunk = 8
    o1 = att._linear_attend(Q, K, V).view(B, T, H, dh)
    Kp, Vp = K.clone(), V.clone()
    t_cut = 17
    Kp[:, :, t_cut:, :] += 5.0 * torch.randn_like(Kp[:, :, t_cut:, :])
    Vp[:, :, t_cut:, :] += 5.0 * torch.randn_like(Vp[:, :, t_cut:, :])
    o2 = att._linear_attend(Q, Kp, Vp).view(B, T, H, dh)
    past_err = (o1[:, :t_cut] - o2[:, :t_cut]).abs().max().item()
    future_changed = (o1[:, t_cut:] - o2[:, t_cut:]).abs().max().item()
    print(f"  max|change| at positions < {t_cut}: {past_err:.2e}  "
          f"[{'CAUSAL' if past_err < 1e-10 else 'LEAK!'}]")
    print(f"  max|change| at positions >={t_cut}: {future_changed:.2e}  (expected > 0)")


if __name__ == "__main__":
    main()
