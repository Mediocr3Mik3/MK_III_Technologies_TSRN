"""
DPO config — 8x H100 80GB.

25K preference pairs, ~3 epochs. Reference and policy share weights at
init; reference is frozen on the SFT checkpoint.
"""

CONFIG = {
    # ---- Model (must match SFT) ----
    "tier":            "pro",
    "d_model":         768,
    "n_blocks":        12,
    "n_heads":         12,
    "context_len":     4096,
    "top_k":           24,
    "mem_depth":       8,
    "max_gists":       256,
    "gist_top_k":      8,
    "dropout":         0.0,
    "use_kleene_ssm":  True,
    "use_hyperbolic":  True,
    "gist_chaining":   True,

    # ---- Tokenizer ----
    "use_tmt_tokenizer": True,
    "tmt_path":          "/mnt/blob/tokenizers/tmt_32k.json",
    "vocab_size":        32000,

    # ---- Training ----
    "init_from":          "/mnt/blob/checkpoints/sft/best.pt",
    "reference_from":     "/mnt/blob/checkpoints/sft/best.pt",
    "epochs":             3,
    "batch_size":         2,           # 2 prompts * 2 (chosen+rejected) = 4 fwd
    "grad_accum_steps":   8,
    "world_size":         8,
    "lr":                 5e-7,
    "min_lr":             5e-8,
    "betas":              (0.9, 0.95),
    "weight_decay":       0.0,
    "warmup_steps":       100,
    "lr_schedule":        "cosine",
    "grad_clip":          1.0,
    "beta":               0.1,         # DPO temperature
    "label_smoothing":    0.0,

    # ---- System ----
    "gradient_checkpoint": True,
    "compile":             False,
    "use_8bit_optimizer":  False,
    "amp_dtype":           "bfloat16",
    "gc_every":            200,
    "eval_every":          500,
    "ckpt_every":          1000,
    "keep_last_ckpts":     3,

    # ---- Data ----
    "manifest":   "research/cloud/azure/data/manifests/dpo_mix.yaml",
    "tokens_dir": "/mnt/blob/tokens/dpo",

    # ---- Logging / output ----
    "wandb_project": "tropformer-dpo",
    "output_dir":    "/mnt/blob/checkpoints/dpo",

    "seed": 44,
}
