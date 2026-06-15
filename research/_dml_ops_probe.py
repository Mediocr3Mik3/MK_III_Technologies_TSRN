"""Probe which ops needed by linear attention run natively on DirectML vs
fall back to CPU. Decides the implementation: a simple cumsum-scan (if cumsum
is native) or a chunked-matmul scan (if cumsum falls back).

Run: .venv312\\Scripts\\python.exe research\\_dml_ops_probe.py
"""
import time
import warnings
import torch

try:
    import torch_directml
    DEV = torch_directml.device()
    NAME = "DirectML"
except ImportError:
    DEV = torch.device("cpu")
    NAME = "CPU"


def probe(label, fn, ref_cpu=None):
    """Run fn on DEV, capture any DML CPU-fallback warning, time it, and
    (optionally) check correctness vs a CPU reference."""
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        try:
            r = fn()
            if hasattr(r, "cpu"):
                rc = r.cpu()
            torch.manual_seed(0)
            t0 = time.perf_counter()
            for _ in range(20):
                r = fn()
            _ = r.cpu() if hasattr(r, "cpu") else r
            dt = (time.perf_counter() - t0) / 20 * 1e3
        except Exception as e:
            print(f"  {label:<34} FAILED  {type(e).__name__}: {str(e)[:60]}")
            return None
    fell_back = any("fall back" in str(w.message).lower()
                    or "not currently supported" in str(w.message).lower()
                    for w in wlog)
    tag = "CPU-FALLBACK" if fell_back else "native"
    extra = ""
    if ref_cpu is not None:
        extra = f"  max|err|={(rc - ref_cpu).abs().max().item():.2e}"
    print(f"  {label:<34} OK  {dt:6.2f} ms/call  [{tag}]{extra}")
    return r


def main():
    print(f"Device: {NAME} ({DEV})")
    B, H, T, dh, dv = 8, 4, 256, 64, 64
    torch.manual_seed(0)
    q = torch.randn(B, H, T, dh)
    k = torch.randn(B, H, T, dh)
    v = torch.randn(B, H, T, dv)
    phiK_cpu = torch.exp(k - k.amax((-2, -1), keepdim=True))
    outer_cpu = phiK_cpu.unsqueeze(-1) * v.unsqueeze(-2)        # (B,H,T,dh,dv)

    qd, kd, vd = q.to(DEV), k.to(DEV), v.to(DEV)
    phiKd = torch.exp(kd - kd.amax((-2, -1), keepdim=True))
    outerd = phiKd.unsqueeze(-1) * vd.unsqueeze(-2)

    print("Ops required by linear attention:")
    probe("exp / amax (feature map)", lambda: torch.exp(qd / 1.0))
    probe("cumsum dim=-2 (denominator Z)",
          lambda: torch.cumsum(phiKd, dim=-2),
          ref_cpu=torch.cumsum(phiK_cpu, dim=-2))
    probe("cumsum dim=-3 on 5D (state KV)",
          lambda: torch.cumsum(outerd, dim=-3),
          ref_cpu=torch.cumsum(outer_cpu, dim=-3))
    probe("einsum bhtd,bhtde->bhte (read)",
          lambda: torch.einsum("bhtd,bhtde->bhte", phiKd, torch.cumsum(outerd, dim=-3)))
    probe("bmm chunk Q@K^T (chunked path)",
          lambda: torch.matmul(qd, kd.transpose(-1, -2)))
    print("Decision: if both cumsum rows say 'native', use the simple scan;")
    print("otherwise use the chunked-matmul form (bmm is the fallback-safe op).")


if __name__ == "__main__":
    main()
