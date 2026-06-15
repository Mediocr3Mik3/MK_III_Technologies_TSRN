"""Short-sequence generation smoke test (DirectML).

Reproduces the step-5000 generate_sample crash (SheafRotorDiffusion zero-length
torch.cat when T <= sheaf window) and verifies the fix, plus any cascade failure
in the downstream small-T path (RG pool, gist extractor, scale-2 at T//2, ...).

Random weights are fine: we are testing shapes / DirectML op support, not output.

Run:
    .venv312/Scripts/python.exe research/_gen_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_R = Path(__file__).resolve().parent
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))

import torch  # noqa: E402

from model_config import nano_directml_config  # noqa: E402
from tsrn_gist import TSRNGist  # noqa: E402


def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def main():
    dev = get_device()
    print(f"device: {dev}")
    cfg = nano_directml_config()
    model = TSRNGist(
        vocab=cfg.vocab_size, d_model=cfg.d_model, context_len=cfg.context_len,
        n_blocks=cfg.n_blocks, top_k=cfg.top_k, n_heads=cfg.n_heads,
        mem_depth=cfg.padic_depth, max_gists=cfg.max_gists,
        gist_top_k=cfg.gist_top_k, dropout=0.0, config=cfg,
    ).to(dev)
    model.eval()

    all_ok = True
    with torch.no_grad():
        for T in [1, 2, 3, 4, 5, 8, 16, 32, 64]:
            try:
                x = torch.randint(0, cfg.vocab_size, (1, T), device=dev)
                logits, _ = model(x)
                shape_ok = logits.shape[0] == 1 and logits.shape[1] == T
                print(f"  T={T:>3}  {'OK ' if shape_ok else 'BAD'}  logits={tuple(logits.shape)}")
                all_ok &= shape_ok
            except Exception as e:
                all_ok = False
                print(f"  T={T:>3}  FAIL {type(e).__name__}: {str(e)[:140]}")

    print("\nALL PASS" if all_ok else "\nFAILURES PRESENT")


if __name__ == "__main__":
    main()
