"""Minimal DirectML forward+backward smoke test for TSRNGist.

Isolates the model from the data pipeline so we can pinpoint any
DirectML-incompatible op. Anomaly detection makes the backward traceback
point at the *forward* line that created the offending autograd node.

Run:
    .venv312/Scripts/python.exe research/_dml_smoke.py
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
from tsrn_dml import AdamWDML  # noqa: E402


def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        print("  (torch_directml not found — falling back to CPU)")
        return torch.device("cpu")


def main():
    dev = get_device()
    print(f"device: {dev}")
    torch.manual_seed(0)

    cfg = nano_directml_config(d_model=256, n_heads=4, n_blocks=3,
                               context_len=256, max_gists=32)
    model = TSRNGist(
        vocab=cfg.vocab_size, d_model=cfg.d_model, context_len=cfg.context_len,
        n_blocks=cfg.n_blocks, top_k=cfg.top_k, n_heads=cfg.n_heads,
        mem_depth=cfg.padic_depth, max_gists=cfg.max_gists,
        gist_top_k=cfg.gist_top_k, dropout=0.0, config=cfg,
    ).to(dev)
    model.train()
    print(f"params: {model.count_params():,}")

    B, T = 2, cfg.context_len
    optimizer = AdamWDML(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    torch.autograd.set_detect_anomaly(True)
    n_steps = 6
    for step in range(n_steps):
        # Exercise the buffer-reset path (mirrors trainer: step % 100 == 1).
        if step == 3:
            model.gist_buffer.reset()
            print(f"  [step {step}] gist_buffer.reset()")

        x = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
        y = torch.randint(0, cfg.vocab_size, (B, T), device=dev)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # count grows each step, so step>=1 exercises the non-empty
        # retrieve() gather path and the store() masked-write path.
        gist_count = int(model.gist_buffer.count.item())
        print(f"  step {step}: loss={loss.item():.4f} gnorm={float(gnorm):.3f} "
              f"gists={gist_count}")

    print("OK: multi-step train loop succeeded on DirectML")


if __name__ == "__main__":
    main()
