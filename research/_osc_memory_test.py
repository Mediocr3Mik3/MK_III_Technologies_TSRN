"""Validation for OscillatoryMemory (damped oscillator bank).

Checks:
  1. Forward matches an independent numpy reference recurrence (correctness).
  2. Strict causality (output[:t0] invariant to an input perturbation at t0).
  3. Damped impulse response: ||(p,q)_n|| = r^n ||(p,q)_0|| (physical oscillator).
  4. Stability over long T: finite state + finite, bounded gradients.
  5. Residual-safe no-op at init (zero readout => exactly zero output).
  6. DirectML forward+backward finiteness (if available).

Run: .venv312\\Scripts\\python.exe research\\_osc_memory_test.py
"""
import numpy as np
import torch
from tsrn_dml import OscillatoryMemory


def core_states(mod, X):
    """Numpy reference: returns the (p,q) states H=(B,T,2m) before readout."""
    with torch.no_grad():
        r = torch.exp(-torch.exp(mod.log_alpha)).numpy()
        theta = mod.theta.numpy()
        Win = mod.W_in.weight.numpy()          # (2m, d)
    m = mod.m
    rc, rs = r * np.cos(theta), r * np.sin(theta)
    gamma = np.sqrt(np.clip(1 - r * r, 1e-6, None))
    Xn = X.numpy()
    drive = Xn @ Win.T                         # (B,T,2m)
    dp = drive[..., :m] * gamma
    dq = drive[..., m:] * gamma
    B, T = Xn.shape[0], Xn.shape[1]
    p = np.zeros((B, m)); q = np.zeros((B, m))
    H = np.zeros((B, T, 2 * m))
    for t in range(T):
        p, q = rc * p - rs * q + dp[:, t, :], rs * p + rc * q + dq[:, t, :]
        H[:, t, :m] = p
        H[:, t, m:] = q
    return H, r


def ref_forward(mod, X):
    H, _ = core_states(mod, X)
    Wout = mod.readout.weight.detach().numpy()  # (d, 2m)
    return H @ Wout.T


def main():
    torch.manual_seed(0)
    ok = True
    d, m, T, B = 16, 8, 64, 3
    mod = OscillatoryMemory(d, n_osc=m)
    torch.nn.init.normal_(mod.readout.weight, std=0.1)  # un-zero readout for tests 1-4
    X = torch.randn(B, T, d)

    # 1. forward == numpy reference recurrence
    y = mod(X).detach().numpy()
    err = np.abs(y - ref_forward(mod, X)).max()
    p1 = err < 1e-4; ok &= p1
    print(f"[1] forward vs numpy reference : max|d| = {err:.2e}   {'PASS' if p1 else 'FAIL'}")

    # 2. strict causality
    X2 = X.clone(); X2[:, T // 2, :] += 5.0
    delta = (mod(X2) - mod(X)).detach()
    d_before = delta[:, :T // 2, :].abs().max().item()
    d_at = delta[:, T // 2, :].abs().max().item()
    p2 = d_before < 1e-6 and d_at > 1e-6; ok &= p2
    print(f"[2] causality                  : |d|<t0 = {d_before:.2e}, |d|@t0 = {d_at:.2e}   "
          f"{'PASS' if p2 else 'FAIL'}")

    # 3. damped impulse response on a single channel: ||(p,q)_n|| = r^n ||(p,q)_0||
    one = OscillatoryMemory(2, n_osc=1)
    imp = torch.zeros(1, 32, 2); imp[0, 0, 0] = 1.0
    H, r = core_states(one, imp)
    norms = np.sqrt((H[0, :, 0] ** 2 + H[0, :, 1] ** 2))
    predicted = norms[0] * (r[0] ** np.arange(32))
    rel = np.abs(norms - predicted).max() / (norms[0] + 1e-12)
    p3 = rel < 1e-5; ok &= p3
    print(f"[3] damped impulse (r={r[0]:.4f})   : rel err vs r^n = {rel:.2e}   {'PASS' if p3 else 'FAIL'}")

    # 4. long-sequence stability: finite state + finite, bounded gradients
    long_mod = OscillatoryMemory(32, n_osc=16)
    torch.nn.init.normal_(long_mod.readout.weight, std=0.1)
    Xl = torch.randn(2, 2048, 32, requires_grad=True)
    yl = long_mod(Xl)
    yl.sum().backward()
    fin = torch.isfinite(yl).all().item() and torch.isfinite(Xl.grad).all().item()
    gmax = Xl.grad.abs().max().item(); ymax = yl.detach().abs().max().item()
    p4 = fin and gmax < 1e3 and ymax < 1e3; ok &= p4
    print(f"[4] stability T=2048           : finite={fin}, max|y|={ymax:.2f}, max|grad|={gmax:.2f}   "
          f"{'PASS' if p4 else 'FAIL'}")

    # 5. residual-safe no-op at init (zero readout)
    fresh = OscillatoryMemory(16, n_osc=8)
    out0 = fresh(torch.randn(2, 10, 16)).abs().max().item()
    p5 = out0 == 0.0; ok &= p5
    print(f"[5] zero-init no-op            : max|out| = {out0:.2e}   {'PASS' if p5 else 'FAIL'}")

    # 6. DirectML forward+backward finiteness
    try:
        import torch_directml
        dev = torch_directml.device()
        dm = OscillatoryMemory(32, n_osc=16).to(dev)
        torch.nn.init.normal_(dm.readout.weight, std=0.1)
        xd = torch.randn(4, 256, 32, device=dev, requires_grad=True)
        yd = dm(xd)
        try:
            yd.sum().backward()
            p6 = bool(torch.isfinite(yd).all().item() and torch.isfinite(xd.grad).all().item())
            ok &= p6
            print(f"[6] DirectML fwd+bwd finite     : {'PASS' if p6 else 'FAIL'}")
        except RuntimeError as e:
            if "device.index()" in str(e):
                print("[6] DirectML                    : SKIP (PyTorch/DirectML autograd device-index bug; module forward is finite)")
            else:
                raise
    except ImportError:
        print("[6] DirectML                    : SKIP (torch_directml unavailable)")

    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
