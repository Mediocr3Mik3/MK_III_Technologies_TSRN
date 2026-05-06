"""
SFT config — 8x H100 80GB.

Curriculum (sequential):
    reasoning -> instruction -> tool -> kyro

Total ~6.5M examples * 3 epochs / (micro_batch * world * grad_accum) steps.
"""

CONFIG = {
    # ---- Model (must match pretrained) ----
    "tier":            "pro",
    "d_model":         768,
    "n_blocks":        12,
    "n_heads":         12,
    "context_len":     4096,           # SFT often longer than pretrain
    "top_k":           24,
    "mem_depth":       8,
    "max_gists":       256,
    "gist_top_k":      8,
    "dropout":         0.05,
    "use_kleene_ssm":  True,
    "use_hyperbolic":  True,
    "gist_chaining":   True,

    # ---- Tokenizer ----
    "use_tmt_tokenizer": True,
    "tmt_path":          "/mnt/blob/tokenizers/tmt_32k.json",
    "vocab_size":        32000,

    # ---- Training ----
    "init_from":   "/mnt/blob/checkpoints/pretrain/best.pt",
    "epochs":      3,
    "batch_size":  4,                  # per GPU; long context
    "grad_accum_steps": 4,
    "world_size":  8,
    "lr":          1e-5,
    "min_lr":      1e-6,
    "betas":       (0.9, 0.95),
    "weight_decay": 0.05,
    "warmup_steps": 200,
    "lr_schedule": "cosine",
    "grad_clip":   1.0,

    # ---- Curriculum ----
    "curriculum_order": ["reasoning", "instruction", "tool", "kyro"],
    "curriculum_steps_split": "by_examples",  # split steps proportionally to example counts
    "loss_mask_prompt": True,
    "use_sample_weight_multiplier": True,

    # ---- System ----
    "gradient_checkpoint": True,
    "compile":             False,      # SFT shapes are ragged; compile breaks
    "use_8bit_optimizer":  False,
    "amp_dtype":           "bfloat16",
    "gc_every":            500,
    "eval_every":          1000,
    "ckpt_every":          2000,
    "keep_last_ckpts":     5,

    # ---- Data ----
    "manifest":   "research/cloud/azure/data/manifests/sft_mix.yaml",
    "tokens_dir": "/mnt/blob/tokens/sft",
    "val_examples_per_dataset": 500,

    # ---- Logging / output ----
    "wandb_project": "tropformer-sft",
    "output_dir":    "/mnt/blob/checkpoints/sft",

    "seed": 43,
}
