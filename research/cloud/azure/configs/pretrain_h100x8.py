"""
Pretrain config — 8x H100 80GB (Azure ND_H100_v5 / NDm_H100_v5).

Target model: TSRNGist Pro tier, ~150M params, context 2048, vocab 32k (TMT).
92B-token pretraining corpus -> ~46K optimizer steps at effective batch 1M tokens.

Effective tokens / step:
    micro_batch * grad_accum * world * context_len
    8           * 4          * 8     * 2048        = 524,288

92B tokens / 524288 = ~175,000 steps. We schedule a few extra for safety.
"""

CONFIG = {
    # ---- Model ----
    "tier":            "pro",
    "d_model":         768,
    "n_blocks":        12,
    "n_heads":         12,
    "context_len":     2048,
    "top_k":           24,
    "mem_depth":       8,
    "max_gists":       256,
    "gist_top_k":      8,
    "dropout":         0.0,            # disabled during pretrain
    "use_kleene_ssm":  True,
    "use_hyperbolic":  True,
    "gist_chaining":   True,

    # ---- Tokenizer (frozen 48k TMT with the 92 TSRN special tokens) ----
    "use_tmt_tokenizer": True,
    "tmt_path":          "/mnt/blob/tokenizers/tmt_48k.json",
    "vocab_size":        48000,

    # ---- Training schedule ----
    "steps":           180_000,
    "batch_size":      8,              # per GPU
    "grad_accum_steps": 4,             # effective per-rank micro-batch 32
    "world_size":      8,              # 8 H100s
    "lr":              4e-4,
    "min_lr":          4e-5,
    "betas":           (0.9, 0.95),
    "weight_decay":    0.1,
    "warmup_steps":    4000,
    "lr_schedule":     "cosine",
    "grad_clip":       1.0,

    # ---- System ----
    "gradient_checkpoint": True,
    "compile":             True,
    "compile_mode":        "max-autotune",
    "use_8bit_optimizer":  False,
    "amp_dtype":           "bfloat16",
    "gc_every":            500,
    "eval_every":          2000,
    "ckpt_every":          5000,
    "keep_last_ckpts":     5,
    "gist_reset_every":    100,

    # ---- NEXUS innovations ----
    "maslov_h_warm":   1.5,
    "maslov_h_cool":   0.3,
    "maslov_n_cycles": 6,

    # ---- Data ----
    "manifest":   "research/cloud/azure/data/manifests/pretrain_mix.yaml",
    "tokens_dir": "/mnt/blob/tokens/pretrain",
    "val_tokens_per_dataset": 2_000_000,

    # ---- Logging / output ----
    "wandb_project": "tropformer-pretrain",
    "output_dir":    "/mnt/blob/checkpoints/pretrain",

    "seed": 42,
}
