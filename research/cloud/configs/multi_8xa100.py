"""
Preset: multi_8xa100
====================
Target: 8 × A100 (40 or 80 GB) DDP node.  Also fine for 4× / 2× — just adjust
nproc_per_node when launching.

Total effective batch = 16 (per-GPU) × 4 (accum) × 8 (GPUs) = 512.
A100-80 will fit batch=24 per GPU; A100-40 needs batch=12.

~1.5h / 100K steps on 8×A100-80.

Launch::
    torchrun --standalone --nproc_per_node=8 \
        -m research.cloud.train_cloud --preset multi_8xa100 --tag multi8
"""

CONFIG = {
    # Model — 100M target so it fits 40GB cards even at batch 12+
    "d_model":         768,
    "n_blocks":        10,
    "n_heads":         12,
    "context_len":     1024,
    "top_k":           20,
    "mem_depth":       7,
    "max_gists":       96,
    "gist_top_k":      6,
    "dropout":         0.1,

    # Training — large effective batch needs higher LR
    "steps":           100_000,
    "batch_size":      16,           # per-GPU micro-batch (A100-80); drop to 12 on A100-40
    "grad_accum_steps": 4,
    "lr":              4e-4,         # scaled for eff batch 512
    "betas":           (0.9, 0.95),
    "weight_decay":    0.1,
    "warmup_steps":    4000,

    # System
    "gradient_checkpoint": True,
    "compile":         False,        # DDP + compile is finicky in PyTorch 2.4
    "use_8bit_optimizer": False,
    "gc_every":        500,
    "eval_every":      2000,
    "ckpt_every":      5000,
    "gist_reset_every": 100,

    # NEXUS innovations
    "maslov_h_warm":   1.5,
    "maslov_h_cool":   0.3,
    "maslov_n_cycles": 3,

    "seed":            42,
}
