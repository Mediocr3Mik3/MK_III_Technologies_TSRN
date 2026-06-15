"""
GPU-agnostic pretrain config for TSRNGist (no H100 assumption).
MK III Technologies.

We are NOT guaranteed H100s, so this config carries conservative single-GPU
defaults (sized to fit ~24 GB VRAM with gradient checkpointing) and exposes
every scaling knob via TROPFORMER_* environment variables, applied at load
time by train_pretrain_cloud.apply_env_overrides(). Set them in the AzureML
job for the SKU you actually get:

    TROPFORMER_WORLD_SIZE     number of GPUs (process_count_per_instance)
    TROPFORMER_BATCH_SIZE     per-GPU micro-batch
    TROPFORMER_GRAD_ACCUM     gradient accumulation steps
    TROPFORMER_CONTEXT_LEN    sequence length
    TROPFORMER_STEPS          optimizer steps
    TROPFORMER_WARMUP_STEPS   warmup steps
    TROPFORMER_COMPILE        "1"/"0"  (torch.compile; keep off on non-CUDA)
    TROPFORMER_GRADIENT_CHECKPOINT  "1"/"0"

Effective tokens/step = batch_size * grad_accum_steps * world_size * context_len.
Defaults below: 4 * 16 * 1 * 1024 = 65,536 tokens/step.
A ~28B-token budget => ~427k steps on one GPU; scale steps down / GPUs up to fit
your wall-clock. The trainer always reads the true vocab from the tokenizer, so
vocab_size here is informational only.
"""

CONFIG = {
    # ---- Model (TSRNGist Pro, ~150M params; fits 24 GB w/ grad ckpt) ----
    "tier":            "pro",
    "d_model":         768,
    "n_blocks":        12,
    "n_heads":         12,
    "context_len":     1024,           # override via TROPFORMER_CONTEXT_LEN
    "top_k":           24,
    "mem_depth":       8,
    "max_gists":       256,
    "gist_top_k":      8,
    "dropout":         0.0,
    "use_kleene_ssm":  True,
    "use_hyperbolic":  True,
    "gist_chaining":   True,

    # ---- Tokenizer (frozen 48k TMT with the 92 TSRN special tokens) ----
    "use_tmt_tokenizer": True,
    "tmt_path":          "/mnt/blob/tokenizers/tmt_48k.json",
    "vocab_size":        48000,

    # ---- Training schedule (conservative single-GPU defaults) ----
    "steps":           100_000,        # override via TROPFORMER_STEPS
    "batch_size":      4,              # per GPU; override via TROPFORMER_BATCH_SIZE
    "grad_accum_steps": 16,            # override via TROPFORMER_GRAD_ACCUM
    "world_size":      1,              # override via TROPFORMER_WORLD_SIZE
    "lr":              4e-4,
    "min_lr":          4e-5,
    "betas":           (0.9, 0.95),
    "weight_decay":    0.1,
    "warmup_steps":    2000,
    "lr_schedule":     "cosine",
    "grad_clip":       1.0,

    # ---- System (compile OFF by default: safest across SKUs/backends) ----
    "gradient_checkpoint": True,
    "compile":             False,      # override via TROPFORMER_COMPILE
    "compile_mode":        "default",
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

    # ---- Curriculum (component-resonant phased mixture) ----
    "curriculum_enabled":     True,
    "curriculum_update_every": 50,

    # ---- Logging / output ----
    "wandb_project": "tropformer-pretrain",
    "output_dir":    "/mnt/blob/checkpoints/pretrain",

    "seed": 42,
}
