"""
Preset: small_24gb
==================
Target: RTX 4090 (24 GB), RTX 3090 (24 GB), A10 (24 GB), L4 (24 GB).

Fits a ~22M-param TSRNGist with context 512 in ~16-18 GB peak (bf16 + grad-ckpt).
Effective batch = 8 × 4 (grad-accum) = 32.

Single-GPU only.  ~3.5h / 100K steps on RTX 4090.
"""

CONFIG = {
    # Model
    "d_model":         512,
    "n_blocks":        3,
    "n_heads":         8,
    "context_len":     512,
    "top_k":           16,
    "mem_depth":       7,
    "max_gists":       64,
    "gist_top_k":      4,
    "dropout":         0.1,

    # Training
    "steps":           100_000,
    "batch_size":      8,
    "grad_accum_steps": 4,         # effective batch 32
    "lr":              2e-4,
    "betas":           (0.9, 0.95),
    "weight_decay":    0.1,
    "warmup_steps":    4000,

    # System
    "gradient_checkpoint": True,
    "compile":         False,       # too slow on small models
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
