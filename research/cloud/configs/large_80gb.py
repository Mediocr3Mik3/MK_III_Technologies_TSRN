"""
Preset: large_80gb
==================
Target: A100 80GB, H100 80GB, H200, L40S (48GB but use this if 4090 too small).

Fits a ~150M-param TSRNGist with context 2048 in ~55-65 GB peak (bf16 + grad-ckpt).
Effective batch = 32 × 2 = 64.

Single-GPU.  ~10h / 100K steps on H100.
"""

CONFIG = {
    # Model — Roadmap target: 100-150M params
    "d_model":         768,
    "n_blocks":        12,
    "n_heads":         12,
    "context_len":     2048,
    "top_k":           24,
    "mem_depth":       8,
    "max_gists":       128,
    "gist_top_k":      8,
    "dropout":         0.1,

    # Training
    "steps":           150_000,
    "batch_size":      32,
    "grad_accum_steps": 2,           # effective batch 64
    "lr":              3e-4,
    "betas":           (0.9, 0.95),
    "weight_decay":    0.1,
    "warmup_steps":    6000,

    # System
    "gradient_checkpoint": True,
    "compile":         True,
    "compile_mode":    "reduce-overhead",
    "use_8bit_optimizer": False,     # plenty of memory
    "gc_every":        500,
    "eval_every":      3000,
    "ckpt_every":      5000,
    "gist_reset_every": 100,

    # NEXUS innovations
    "maslov_h_warm":   1.5,
    "maslov_h_cool":   0.3,
    "maslov_n_cycles": 3,

    "seed":            42,
}
