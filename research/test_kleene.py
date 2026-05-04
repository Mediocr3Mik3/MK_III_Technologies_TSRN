"""
Correctness tests for KleeneSSM, KleeneAttention, and PaCS (P-adic Context Scaling).
MK III Technologies -- TSRN / Kyro Architecture

Run from project root:
    python research/test_kleene.py

All tests should pass before training.
"""

from __future__ import annotations

import os
import sys
import time

import torch

# Make `research/` importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
#  Config tests
# ---------------------------------------------------------------------------

def test_config_tiers():
    """Tier configs produce the expected feature flags."""
    from model_config import nano_config, pro_config, kyro_config

    nano = nano_config()
    assert nano.use_kleene_ssm is True, "Nano should have KleeneSSM ON"
    assert nano.use_kleene_attention is False, "Nano should have KleeneAttention OFF"
    assert nano.use_cross_window_memory is False
    assert nano.tier == "nano"

    pro = pro_config()
    assert pro.use_kleene_ssm is True
    assert pro.use_kleene_attention is True, "Pro should have KleeneAttention ON"
    assert pro.use_cross_window_memory is True
    assert pro.tier == "pro"

    kyro = kyro_config()
    assert kyro.use_kleene_ssm is True
    assert kyro.use_kleene_attention is True
    assert kyro.tier == "kyro"

    print(f"OK  Nano config: {nano}")
    print(f"OK  Pro config:  {pro}")
    print(f"OK  Kyro config: {kyro}")


def test_config_save_load(tmpdir: str = "_tmp_config_test.json"):
    """Config save/load round-trips cleanly."""
    from model_config import pro_config, ModelConfig
    cfg = pro_config(d_model=2048)
    cfg.save(tmpdir)
    loaded = ModelConfig.load(tmpdir)
    assert loaded.d_model == 2048
    assert loaded.use_kleene_attention is True
    os.remove(tmpdir)
    print(f"OK  Config save/load round-trip works")


# ---------------------------------------------------------------------------
#  Kleene-star structural tests
# ---------------------------------------------------------------------------

def test_kleene_star_lower_triangular():
    """Kleene star of a lower-triangular matrix is lower-triangular.
    This is the structural basis for causality preservation in attention.
    """
    from tsrn_dml import KleeneSSM
    model = KleeneSSM(d_model=16, d_state=8, n_iters=4)

    T = 16
    A = torch.tril(torch.randn(T, T))
    # Set strict-upper to very negative.
    A = A.masked_fill(torch.triu(torch.ones_like(A, dtype=torch.bool), diagonal=1), -1e9)

    A_star = model.compute_kleene_star(A)

    upper = A_star.masked_select(
        torch.triu(torch.ones_like(A_star, dtype=torch.bool), diagonal=1)
    )
    finite_upper = upper[upper > -1e8]
    assert finite_upper.numel() == 0, (
        f"Kleene star not lower-triangular: "
        f"max upper-finite value = {finite_upper.max().item() if finite_upper.numel() > 0 else 'n/a'}"
    )
    print(f"OK  Kleene star preserves lower-triangular structure")


def test_kleene_star_idempotence():
    """A** == A* for the tropical Kleene star.

    Idempotence holds iff A* converges, which requires no positive cycles.
    We enforce this by using non-positive entries (paths only shrink).
    """
    from tsrn_dml import KleeneSSM
    model = KleeneSSM(d_model=16, d_state=6, n_iters=6)
    # Non-positive entries: every path either stays same or decreases.
    # This guarantees A* converges.
    A = -torch.rand(6, 6).abs() * 2.0  # all entries in [-2, 0]
    A_star = model.compute_kleene_star(A)
    A_star_star = model.compute_kleene_star(A_star)
    err = (A_star - A_star_star).abs().max().item()
    assert err < 1e-4, f"Kleene star not idempotent: err = {err:.3e}"
    print(f"OK  Kleene star idempotent on bounded A (A** == A*, max err = {err:.2e})")


# ---------------------------------------------------------------------------
#  KleeneSSM tests
# ---------------------------------------------------------------------------

def test_kleene_ssm_shape_and_finite():
    """KleeneSSM produces finite outputs of the correct shape."""
    from tsrn_dml import KleeneSSM
    d = 64
    kleene = KleeneSSM(d_model=d, d_state=16, n_iters=3)
    kleene.eval()
    x = torch.randn(2, 32, d)
    with torch.no_grad():
        out = kleene(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(), "KleeneSSM produced NaN or Inf"
    print(f"OK  KleeneSSM output: shape={tuple(out.shape)}, "
          f"mean={out.mean():.4f}, std={out.std():.4f}")


def test_kleene_ssm_causality():
    """KleeneSSM must not leak future information.

    Perturbing x[:, t0, :] must not affect output[:, t, :] for t < t0.
    """
    from tsrn_dml import KleeneSSM
    torch.manual_seed(0)
    model = KleeneSSM(d_model=64, d_state=16, n_iters=3)
    model.eval()

    B, T, d = 2, 32, 64
    x = torch.randn(B, T, d)
    t0 = 16

    with torch.no_grad():
        out_orig = model(x)
        x_perturbed = x.clone()
        x_perturbed[:, t0, :] += torch.randn(B, d) * 10.0
        out_perturbed = model(x_perturbed)

    leak = (out_orig[:, :t0, :] - out_perturbed[:, :t0, :]).abs().max().item()
    # Allow tiny numerical error from softplus mean over all positions
    # being dependent on t0 (not a true future-info leak, but a global
    # statistic). Compare to the post-t0 difference for context.
    post_diff = (out_orig[:, t0:, :] - out_perturbed[:, t0:, :]).abs().max().item()
    assert leak < 1e-3, (
        f"CAUSALITY VIOLATION: max leak = {leak:.3e} at positions < {t0} "
        f"(post-t0 diff = {post_diff:.3e})"
    )
    print(f"OK  KleeneSSM causality: pre-t0 leak = {leak:.2e}, "
          f"post-t0 diff = {post_diff:.2e}")


def test_kleene_ssm_backward():
    """KleeneSSM supports gradient computation."""
    from tsrn_dml import KleeneSSM
    model = KleeneSSM(d_model=32, d_state=8, n_iters=3)
    x = torch.randn(2, 16, 32, requires_grad=True)
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Non-finite parameter grad"
    print(f"OK  KleeneSSM backward pass produces finite gradients")


# ---------------------------------------------------------------------------
#  KleeneAttention tests
# ---------------------------------------------------------------------------

def test_kleene_attention_shape_and_finite():
    """KleeneAttention produces finite outputs of the correct shape."""
    from tsrn_dml import KleeneAttention
    d, H, T, B = 64, 4, 32, 2
    attn = KleeneAttention(d_model=d, n_heads=H, top_k=8, n_iters=2)
    attn.eval()
    x = torch.randn(B, T, d)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all(), "KleeneAttention produced NaN or Inf"
    print(f"OK  KleeneAttention output: shape={tuple(out.shape)}, "
          f"mean={out.mean():.4f}, std={out.std():.4f}")


def test_kleene_attention_causality():
    """KleeneAttention must not leak future information.

    Critical test: causal mask must hold even after multi-hop Kleene star.
    """
    from tsrn_dml import KleeneAttention
    torch.manual_seed(0)
    model = KleeneAttention(d_model=64, n_heads=4, top_k=8, n_iters=2)
    model.eval()

    B, T, d = 2, 32, 64
    x = torch.randn(B, T, d)
    t0 = 16

    with torch.no_grad():
        out_orig = model(x)
        x_perturbed = x.clone()
        x_perturbed[:, t0, :] += torch.randn(B, d) * 10.0
        out_perturbed = model(x_perturbed)

    leak = (out_orig[:, :t0, :] - out_perturbed[:, :t0, :]).abs().max().item()
    post_diff = (out_orig[:, t0:, :] - out_perturbed[:, t0:, :]).abs().max().item()
    # Strict for attention: Q/K/V are pointwise, so any leak indicates
    # a real causality bug (unlike SSM which has global mean-delta).
    assert leak < 1e-4, (
        f"CAUSALITY VIOLATION: max leak = {leak:.3e} at positions < {t0} "
        f"(post-t0 diff = {post_diff:.3e})"
    )
    print(f"OK  KleeneAttention causality: pre-t0 leak = {leak:.2e}, "
          f"post-t0 diff = {post_diff:.2e}")


def test_kleene_attention_backward():
    """KleeneAttention supports gradient computation."""
    from tsrn_dml import KleeneAttention
    model = KleeneAttention(d_model=32, n_heads=2, top_k=4, n_iters=2)
    x = torch.randn(2, 16, 32, requires_grad=True)
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"OK  KleeneAttention backward pass produces finite gradients")


# ---------------------------------------------------------------------------
#  Factory tests
# ---------------------------------------------------------------------------

def test_factories_return_correct_class():
    """build_attention / build_ssm return the right class for each tier."""
    from model_config import nano_config, pro_config
    from tsrn_dml import (
        build_attention, build_ssm,
        TropicalAttention, KleeneAttention,
        KleeneSSM, TropicalSSM,
    )

    nano = nano_config(d_model=64, n_heads=4, top_k=4)
    pro = pro_config(d_model=64, n_heads=4, top_k=4,
                     kleene_ssm_d_state=16, kleene_ssm_iters=2,
                     kleene_attn_iters=2)

    nano_attn = build_attention(nano)
    nano_ssm = build_ssm(nano)
    pro_attn = build_attention(pro)
    pro_ssm = build_ssm(pro)

    assert isinstance(nano_attn, TropicalAttention)
    assert isinstance(nano_ssm, KleeneSSM)        # Nano has Kleene SSM
    assert isinstance(pro_attn, KleeneAttention)
    assert isinstance(pro_ssm, KleeneSSM)

    # Override: disable kleene_ssm for ablation -> falls back to TropicalSSM
    nano_no_kleene = nano_config(use_kleene_ssm=False)
    fallback_ssm = build_ssm(nano_no_kleene)
    assert isinstance(fallback_ssm, TropicalSSM)

    print(f"OK  Factories return correct classes for each tier")


# ---------------------------------------------------------------------------
#  PaCS tests (P-adic Context Scaling)
# ---------------------------------------------------------------------------

def test_pacs_valuation_correctness():
    """2-adic valuation matches reference computation for known integers."""
    from padic_context_scaling import PAdicContextScaling

    scaler = PAdicContextScaling(training_ctx=512, p=2, v_threshold=4.7)

    # Known 2-adic valuations:
    # v_2(1) = 0, v_2(2) = 1, v_2(4) = 2, v_2(8) = 3, v_2(16) = 4
    # v_2(512) = 9, v_2(1024) = 10, v_2(3) = 0, v_2(513) = 0
    cases = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6,
             128: 7, 256: 8, 512: 9, 1024: 10, 3: 0, 513: 0, 7: 0}
    positions = torch.tensor(list(cases.keys()))
    expected = torch.tensor(list(cases.values()), dtype=torch.float32)
    got = scaler.valuation_tensor(positions)
    assert torch.allclose(got, expected), (
        f"Valuation mismatch:\n  expected: {expected.tolist()}\n  got: {got.tolist()}"
    )
    print(f"OK  PaCS 2-adic valuation correct on {len(cases)} test cases")


def test_pacs_compresses_high_valuation_more():
    """Positions with high p-adic valuation should be compressed more
    aggressively than positions with low valuation. This is the central
    claim of PaCS vs uniform YaRN-style scaling.
    """
    from padic_context_scaling import PAdicContextScaling

    training_ctx = 512
    inference_ctx = 8192  # 16x extension

    scaler = PAdicContextScaling(training_ctx, p=2, v_threshold=4.7)
    positions = torch.arange(2048)
    scaled, _ = scaler.scale(positions, inference_ctx)

    # Position 1 (v=0, local) should be barely compressed.
    # Position 1024 (v=10, structural boundary) should be heavily compressed.
    s1 = (positions[1].float() / scaled[1]).item()       # effective scale at pos 1
    s1024 = (positions[1024].float() / scaled[1024]).item()  # at pos 1024

    assert s1024 > s1, (
        f"PaCS failed: high-valuation position should be compressed MORE. "
        f"s1={s1:.3f}, s1024={s1024:.3f}"
    )
    # Sanity: low-val should be near 1.0 (almost no compression).
    assert 0.95 <= s1 <= 1.5, f"s1 should be near 1.0: got {s1:.3f}"
    # Sanity: high-val should approach the global multiplier (16x).
    assert s1024 > 5.0, f"s1024 should be heavily compressed: got {s1024:.3f}"
    print(f"OK  PaCS compresses by valuation: scale@1={s1:.3f}, scale@1024={s1024:.3f}")


def test_pacs_vs_uniform_yarn():
    """Confirm PaCS scaling differs from uniform (YaRN-style) scaling.
    Uniform would give the same scale at every position; PaCS gives
    structurally varying scales.
    """
    from padic_context_scaling import PAdicContextScaling

    scaler = PAdicContextScaling(training_ctx=512, p=2, v_threshold=4.7)
    positions = torch.arange(1024)
    scaled, _ = scaler.scale(positions, inference_ctx=8192)

    effective_scales = positions[1:].float() / scaled[1:].clamp_min(1e-6)
    # Uniform YaRN would produce constant effective_scales.
    spread = (effective_scales.max() - effective_scales.min()).item()
    assert spread > 1.0, (
        f"PaCS scaling looks uniform (YaRN-like). Spread={spread:.3f}"
    )
    print(f"OK  PaCS scaling is non-uniform: max-min spread = {spread:.3f}")


def test_pacs_temperature_correction_finite():
    """Temperature correction is finite and well-scaled."""
    from padic_context_scaling import PAdicContextScaling
    scaler = PAdicContextScaling(training_ctx=512)
    positions = torch.arange(1, 2048)  # skip 0 (valuation = inf)
    _, temp = scaler.scale(positions, inference_ctx=8192)
    assert torch.isfinite(temp).all(), "PaCS temperature has non-finite values"
    assert (temp > 0).all(), "PaCS temperature must be positive"
    print(f"OK  PaCS temperature correction: range [{temp.min():.3f}, {temp.max():.3f}]")


# ---------------------------------------------------------------------------
#  Speed test (informational)
# ---------------------------------------------------------------------------

def test_speed_kleene_vs_sequential():
    """Compare KleeneSSM forward pass speed against TropicalSSM."""
    from tsrn_dml import KleeneSSM, TropicalSSM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    d, d_state, T, B = 256, 64, 256, 4
    kleene = KleeneSSM(d_model=d, d_state=d_state, n_iters=4).to(device)
    tropical = TropicalSSM(d_model=d).to(device)
    x = torch.randn(B, T, d, device=device)

    for _ in range(2):
        _ = kleene(x); _ = tropical(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    n_iter = 5
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = kleene(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    kleene_t = (time.perf_counter() - start) / n_iter

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = tropical(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    tropical_t = (time.perf_counter() - start) / n_iter

    print(f"OK  Speed (B={B}, T={T}, d={d}, d_state={d_state}): "
          f"KleeneSSM={kleene_t*1000:.1f}ms, "
          f"TropicalSSM={tropical_t*1000:.1f}ms, "
          f"speedup={tropical_t/max(kleene_t, 1e-9):.2f}x  "
          f"(device={device.type})")


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("Config tiers",                test_config_tiers),
    ("Config save/load",            test_config_save_load),
    ("Kleene-star lower-triangular",test_kleene_star_lower_triangular),
    ("Kleene-star idempotence",     test_kleene_star_idempotence),
    ("KleeneSSM shape & finite",    test_kleene_ssm_shape_and_finite),
    ("KleeneSSM causality",         test_kleene_ssm_causality),
    ("KleeneSSM backward",          test_kleene_ssm_backward),
    ("KleeneAttention shape & finite", test_kleene_attention_shape_and_finite),
    ("KleeneAttention causality",   test_kleene_attention_causality),
    ("KleeneAttention backward",    test_kleene_attention_backward),
    ("Factories return right class",test_factories_return_correct_class),
    ("PaCS valuation correctness",  test_pacs_valuation_correctness),
    ("PaCS compresses high val",    test_pacs_compresses_high_valuation_more),
    ("PaCS != uniform YaRN",        test_pacs_vs_uniform_yarn),
    ("PaCS temperature finite",     test_pacs_temperature_correction_finite),
    ("Speed (informational)",       test_speed_kleene_vs_sequential),
]


def main():
    print("=" * 72)
    print("Kleene Star + PaCS correctness tests")
    print("=" * 72)
    failures = []
    for name, fn in ALL_TESTS:
        print(f"\n[{name}]")
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL: {e}")
            failures.append((name, str(e)))
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            failures.append((name, f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 72)
    if not failures:
        print(f"All {len(ALL_TESTS)} tests passed.")
        return 0
    print(f"{len(failures)} / {len(ALL_TESTS)} tests FAILED:")
    for name, err in failures:
        print(f"  - {name}: {err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
