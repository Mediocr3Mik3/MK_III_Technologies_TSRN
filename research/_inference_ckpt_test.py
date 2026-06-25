"""Quick inference test for a saved TSRN checkpoint.

Loads the checkpoint from 20260614, builds a model with the saved hyperparams,
loads the state dict, and runs a short generation sample.
"""
import sys, os, torch

_RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)

from tsrn_gist import TSRNGist
from model_config import ModelConfig
from transformers import GPT2TokenizerFast

CKPT_PATH = r"D:\ml\tsrn_data\checkpoints\20260614_v2_kleene_directml_d256_ctx256_best.pt"
TOK_PATH = r"D:\ml\tsrn_data\tokenizer\gpt2_tsrn_base"
PROMPT = "The tropical geometry of neural networks reveals that"
MAX_NEW = 64
TEMP = 0.8


def load_and_generate():
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    step = ckpt.get("step", "?")
    best_bpc = ckpt.get("best_val_bpc", "?")
    print(f"  Trained step: {step} | Best val BPC: {best_bpc}")
    print(f"  Config: d={cfg['d_model']}, blocks={cfg['n_blocks']}, heads={cfg['n_heads']}, "
          f"ctx={cfg['context_len']}, vocab={cfg['vocab_size']}")

    tokenizer = GPT2TokenizerFast.from_pretrained(TOK_PATH)
    print(f"  Tokenizer vocab: {len(tokenizer)}")

    # Reconstruct ModelConfig from saved dict.  The checkpoint predates
    # reservoir_kind, so current defaults (echo reservoir) match the old
    # architecture.  Filter to fields that actually exist in ModelConfig.
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    model_cfg = ModelConfig(**{k: v for k, v in cfg.items() if k in valid_fields})
    # Ensure new defaults are present for fields the old config lacked.
    if not hasattr(model_cfg, "reservoir_kind"):
        model_cfg.reservoir_kind = "echo"
    if not hasattr(model_cfg, "use_linear_attention"):
        model_cfg.use_linear_attention = False
    print(f"  Reconstructed config: reservoir_kind={model_cfg.reservoir_kind}, "
          f"kleene_ssm={model_cfg.use_kleene_ssm}")

    model = TSRNGist(
        vocab=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        context_len=model_cfg.context_len,
        gradient_checkpoint=False,
        n_blocks=model_cfg.n_blocks,
        top_k=model_cfg.top_k,
        n_heads=model_cfg.n_heads,
        mem_depth=getattr(model_cfg, "padic_depth", 5),
        max_gists=getattr(model_cfg, "max_gists", 32),
        gist_top_k=getattr(model_cfg, "gist_top_k", 4),
        dropout=getattr(model_cfg, "dropout", 0.0),
        use_hyperbolic=False,
        gist_chaining=False,
        config=model_cfg,
    )

    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  MISSING keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  UNEXPECTED keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        print("  State dict loaded cleanly — no key mismatches.")

    device = torch.device("cpu")
    if torch_directml := sys.modules.get("torch_directml") or None:
        try:
            import torch_directml as _dml
            device = _dml.device()
            print(f"  Using DirectML device: {device}")
        except Exception:
            pass
    else:
        try:
            import torch_directml as _dml
            device = _dml.device()
            print(f"  Using DirectML device: {device}")
        except ImportError:
            print("  Using CPU (torch_directml not available)")

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    # encode prompt
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    print(f"\nPrompt: {PROMPT}")
    print(f"Input ids: {ids}")

    with torch.no_grad():
        for i in range(MAX_NEW):
            if input_ids.size(1) > cfg["context_len"]:
                input_ids = input_ids[:, -cfg["context_len"]:]
            logits, _ = model(input_ids)
            next_logits = logits[:, -1, :] / TEMP
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            ids.append(next_id)
            input_ids = torch.tensor([ids], device=device)
            if next_id == tokenizer.eos_token_id:
                break

    generated = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"\nGenerated ({len(ids) - len(tokenizer.encode(PROMPT, add_special_tokens=False))} new tokens):")
    print(generated)


if __name__ == "__main__":
    load_and_generate()
