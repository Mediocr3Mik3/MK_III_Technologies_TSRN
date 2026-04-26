"""
Modal Labs serverless launcher for TSRNGist cloud training.
============================================================

Modal gives ~$30/month free credits with no card on signup.  GPU types you
can request: T4, L4, A10G, A100-40, A100-80, H100, H200.

Setup (one-time)::

    pip install modal
    modal token new

Smoke test (CPU container, just verifies the import graph)::

    modal run research/cloud/launch/modal_app.py::smoke

Single-GPU training run (A100-40, 100K steps)::

    modal run research/cloud/launch/modal_app.py::train_a100_40 --tag a100_run0

Multi-GPU (1× H100 — Modal does not yet expose torchrun-style multi-GPU
pods on the free tier; for true DDP use RunPod / Lambda)::

    modal run research/cloud/launch/modal_app.py::train_h100 --tag h100_run0

Checkpoints & results persist in a Modal Volume named ``tropformer-vol``.
Pull them locally::

    modal volume get tropformer-vol checkpoints/<run_tag>_best.pt ./checkpoints/
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
#  Image — built from the same Dockerfile but composed in Modal's DSL so that
#  changes to source don't trigger a full image rebuild.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]   # c:\...\TropFormer

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime",
        add_python="3.11",
    )
    .apt_install("git", "wget", "build-essential")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .pip_install_from_requirements(
        str(REPO_ROOT / "research" / "cloud" / "requirements_cuda.txt"))
    # flash-attn — best-effort; SDPA is the fallback
    .run_commands(
        "pip install --no-build-isolation flash-attn>=2.6 "
        "|| echo 'flash-attn skipped'"
    )
    .add_local_dir(str(REPO_ROOT), remote_path="/workspace")
    .workdir("/workspace")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Persistent volume for checkpoints + results + downloaded enwik8
vol = modal.Volume.from_name("tropformer-vol", create_if_missing=True)

app = modal.App("tropformer-train")


def _run_train(preset: str, tag: str, extra_args: list[str]) -> None:
    """Common entry: cd into the mounted workspace and invoke the trainer."""
    os.chdir("/workspace")
    cmd = [
        "python", "-m", "research.cloud.train_cloud",
        "--preset", preset, "--tag", tag,
        *extra_args,
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    vol.commit()


# ---------------------------------------------------------------------------
#  Smoke test (CPU — just checks the import graph compiles)
# ---------------------------------------------------------------------------

@app.function(image=image, cpu=2, memory=4096, timeout=600)
def smoke() -> None:
    print("[smoke] importing torch...")
    import torch
    print(f"[smoke] torch={torch.__version__}, cuda available={torch.cuda.is_available()}")
    print("[smoke] importing TSRNGist...")
    from research.tsrn_gist import TSRNGist
    print(f"[smoke] TSRNGist class: {TSRNGist}")
    print("[smoke] OK")


# ---------------------------------------------------------------------------
#  Single-GPU presets
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="t4",                         # cheapest GPU on Modal — small_24gb fits
    timeout=60 * 60 * 24,             # up to 24h
    volumes={"/workspace/checkpoints": vol},
)
def train_t4(tag: str = "t4_run0", steps: int | None = None) -> None:
    extra = []
    if steps:
        extra += ["--steps", str(steps)]
    _run_train("small_24gb", tag, extra)


@app.function(
    image=image,
    gpu="a10g",                       # 24 GB Ampere — small/medium presets
    timeout=60 * 60 * 24,
    volumes={"/workspace/checkpoints": vol},
)
def train_a10g(tag: str = "a10g_run0", steps: int | None = None) -> None:
    extra = []
    if steps:
        extra += ["--steps", str(steps)]
    _run_train("small_24gb", tag, extra)


@app.function(
    image=image,
    gpu="a100-40gb",
    timeout=60 * 60 * 24,
    volumes={"/workspace/checkpoints": vol},
)
def train_a100_40(tag: str = "a100_40_run0", steps: int | None = None) -> None:
    extra = []
    if steps:
        extra += ["--steps", str(steps)]
    _run_train("medium_40gb", tag, extra)


@app.function(
    image=image,
    gpu="a100-80gb",
    timeout=60 * 60 * 24,
    volumes={"/workspace/checkpoints": vol},
)
def train_a100_80(tag: str = "a100_80_run0", steps: int | None = None) -> None:
    extra = []
    if steps:
        extra += ["--steps", str(steps)]
    _run_train("large_80gb", tag, extra)


@app.function(
    image=image,
    gpu="h100",
    timeout=60 * 60 * 24,
    volumes={"/workspace/checkpoints": vol},
)
def train_h100(tag: str = "h100_run0", steps: int | None = None) -> None:
    extra = []
    if steps:
        extra += ["--steps", str(steps)]
    _run_train("large_80gb", tag, extra)


# ---------------------------------------------------------------------------
#  Resume helper
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="a100-40gb",
    timeout=60 * 60 * 24,
    volumes={"/workspace/checkpoints": vol},
)
def resume_a100_40(checkpoint: str, tag: str = "resumed") -> None:
    """Resume from /workspace/checkpoints/<checkpoint>."""
    _run_train("medium_40gb", tag, ["--resume", f"checkpoints/{checkpoint}"])


# ---------------------------------------------------------------------------
#  Local entrypoint dispatcher
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    target: str = "smoke",
    tag: str = "modal_run0",
    steps: int | None = None,
    checkpoint: str | None = None,
) -> None:
    """Dispatcher.  Usage::

        modal run research/cloud/launch/modal_app.py --target smoke
        modal run research/cloud/launch/modal_app.py --target train_a100_40 \
            --tag a100_run0 --steps 100000
    """
    fn = {
        "smoke":          smoke,
        "train_t4":       train_t4,
        "train_a10g":     train_a10g,
        "train_a100_40":  train_a100_40,
        "train_a100_80":  train_a100_80,
        "train_h100":     train_h100,
        "resume_a100_40": resume_a100_40,
    }.get(target)
    if fn is None:
        raise SystemExit(f"unknown target {target!r}; choices: smoke, train_t4, "
                         "train_a10g, train_a100_40, train_a100_80, train_h100, "
                         "resume_a100_40")

    if target == "smoke":
        fn.remote()
    elif target.startswith("resume"):
        if checkpoint is None:
            raise SystemExit("--checkpoint required for resume_*")
        fn.remote(checkpoint=checkpoint, tag=tag)
    else:
        fn.remote(tag=tag, steps=steps)
