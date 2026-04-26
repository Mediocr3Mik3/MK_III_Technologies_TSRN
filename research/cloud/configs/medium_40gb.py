"""
Preset: medium_40gb
===================
Target: A100 40GB, A6000 (48 GB).

Fits a ~50M-param TSRNGist with context 1024 in ~28-32 GB peak (bf16 + grad-ckpt).
Effective batch = 16 × 2 = 32.

Single-GPU.  ~6h / 100K steps on A100-40.
"""

CONFIG = {
    # Model — scale up depth + context vs small
    "d_model":         512,
    "n_blocks":        7,           # 50M-param target
    "n_heads":         8,
    "context_len":     1024,
    "top_k":           16,
    "mem_depth":       7,
    "max_gists":       96,
    "gist_top_k":      6,
    "dropout":         0.1,

    # Training
    "steps":           100_000,
    "batch_size":      16,
    "grad_accum_steps": 2,
    "lr":              2.5e-4,
    "betas":           (0.9, 0.95),
    "weight_decay":    0.1,
    "warmup_steps":    4000,

    # System
    "gradient_checkpoint": True,
    "compile":         True,        # PyTorch 2.x compile gives ~15-20% speedup here
    "compile_mode":    "reduce-overhead",
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
