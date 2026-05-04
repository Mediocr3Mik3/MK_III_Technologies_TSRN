# research/model_config.py
# MK III Technologies
# Central configuration for TSRN model family.
# All architectural feature flags live here.
# Import this everywhere instead of hardcoding architecture choices.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Complete configuration for a TSRN model instance.
    Controls architecture, size, and which features are active.

    Feature flags follow this convention:
      use_*: bool = True means enabled, False means disabled
      Nano defaults: conservative, fast, low memory
      Pro defaults: full feature set, Kleene star, higher expressivity
      Kyro defaults: same as Pro with larger dimensions
    """

    # -- Identity ----------------------------------------------------------
    model_name: str = "tsrn_gist"
    tier: str = "nano"               # "nano" | "pro" | "kyro"
    version: str = "1.0"

    # -- Dimensions --------------------------------------------------------
    vocab_size: int = 32000          # TMT/BBPE vocabulary
    d_model: int = 512               # main model dimension
    n_heads: int = 8                 # attention heads
    n_blocks: int = 3                # Scale 1 blocks
    context_len: int = 512           # training context window
    dropout: float = 0.0

    # -- Attention ---------------------------------------------------------
    top_k: int = 16                  # sparse attention top-k
    use_rope: bool = True            # rotary position embeddings
    use_alibi: bool = True           # ALiBi distance bias
    use_differential_attention: bool = False  # noise-cancelling attention
                                              # Pro: True, Nano: False

    # -- Kleene Star -------------------------------------------------------
    # KleeneSSM: replaces TropicalSSM. Recommended ON for all tiers.
    # Eliminates sequential loop, removes cummax CPU fallback.
    use_kleene_ssm: bool = True
    kleene_ssm_d_state: int = 64     # state dimension for KleeneSSM
    kleene_ssm_iters: int = 4        # Kleene star iterations (log2 of d_state)

    # KleeneAttention: replaces TropicalAttention. Pro and above only.
    # Increases per-layer cost by ~k* but reduces required depth.
    # Set False for Nano to keep O(Tkd) attention cost.
    use_kleene_attention: bool = False
    kleene_attn_iters: int = 3       # Kleene star iterations in attention
                                     # 3 iterations captures 8-hop dependencies

    # -- Memory Components -------------------------------------------------
    use_reservoir: bool = True       # GRU-gated echo state reservoir
    d_reservoir: int = 256           # reservoir dimension (d_model // 2 for Nano)
    use_padic_memory: bool = True    # p-adic hierarchical key-value memory
    padic_depth: int = 7             # tree depth: 7=Nano, 9=Pro, 12=Kyro
    use_padic_attention: bool = True # non-Archimedean path attention

    # -- Sheaf Components --------------------------------------------------
    use_sheaf_diffusion: bool = True
    sheaf_window: int = 4            # causal restriction map window
    use_sheaf_harmonic_pe: bool = True  # DCT-II + p-adic positional encoding
    sheaf_pe_harmonics: int = 32     # number of DCT harmonics

    # -- Gist Memory -------------------------------------------------------
    use_gist: bool = True
    max_gists: int = 64              # Nano: 64, Pro: 128, Kyro: 256
    gist_top_k: int = 4              # retrieved gists per forward pass

    # -- RG Coarse-Graining (Scale 2) --------------------------------------
    use_rg_coarsening: bool = True
    s2_max_iters: int = 3            # fixed-point iterations
    s2_eps: float = 0.01             # convergence threshold

    # -- Cross-Window Memory -----------------------------------------------
    use_cross_window_memory: bool = False  # Pro: True, Nano: False
    cross_window_size: int = 512     # tokens from previous window to cache

    # -- Maslov Temperature Cycling ----------------------------------------
    use_maslov_cycling: bool = True
    maslov_h_warm: float = 1.5
    maslov_h_cool: float = 0.3
    maslov_cycles: int = 3           # full cycles over the training run

    # -- Context Extension (inference only) --------------------------------
    use_padic_context_scaling: bool = True
    inference_context_multiplier: int = 8  # Nano: 8x, Pro: 16x, Kyro: 32x
    sink_tokens: int = 4             # permanent sink tokens for long context
    pacs_v_threshold: float = 4.7    # p-adic valuation threshold separating
                                     # local from global structure

    # -- Training ----------------------------------------------------------
    use_mixed_precision: bool = True  # bfloat16 on CUDA
    use_gradient_checkpointing: bool = False  # saves memory, slower
    weight_tying: bool = True         # tie input/output embeddings

    # -- Quantization ------------------------------------------------------
    quantization_bits: int = 32       # 32=full, 8=int8, 4=int4
    use_qat: bool = False             # quantization-aware training

    # -- Methods -----------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def count_active_features(self) -> dict:
        """Return dict of which features are active. Useful for logging."""
        return {k: v for k, v in self.__dict__.items() if k.startswith("use_")}

    def __repr__(self) -> str:
        active = [k.replace("use_", "") for k, v in self.__dict__.items()
                  if k.startswith("use_") and v]
        return (f"ModelConfig(tier={self.tier}, d_model={self.d_model}, "
                f"active=[{', '.join(active)}])")


# -- Pre-built tier configurations -----------------------------------------

def nano_config(**overrides) -> ModelConfig:
    """
    Nano: 50-80M parameters. Always-on phone model.
    Optimized for speed and memory. Runs locally 100% of the time.
    KleeneSSM: ON (faster than sequential SSM)
    KleeneAttention: OFF (too expensive for phone)
    """
    cfg = ModelConfig(
        tier="nano",
        d_model=512,
        n_heads=8,
        n_blocks=3,
        context_len=512,
        d_reservoir=256,
        padic_depth=7,
        max_gists=64,
        gist_top_k=4,
        kleene_ssm_d_state=64,
        kleene_ssm_iters=4,
        use_kleene_ssm=True,          # ON: eliminates sequential loop
        use_kleene_attention=False,   # OFF: too expensive for phone
        use_differential_attention=False,
        use_cross_window_memory=False,
        inference_context_multiplier=8,
        quantization_bits=8,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def pro_config(**overrides) -> ModelConfig:
    """
    Pro: 300-400M parameters. On-device flagship or fast cloud.
    Full Kleene star architecture. Released first as paid tier.
    KleeneSSM: ON
    KleeneAttention: ON (Pro has compute budget for it)
    """
    cfg = ModelConfig(
        tier="pro",
        d_model=1024,
        n_heads=16,
        n_blocks=6,
        context_len=1024,
        d_reservoir=512,
        padic_depth=9,
        max_gists=128,
        gist_top_k=8,
        kleene_ssm_d_state=128,
        kleene_ssm_iters=5,
        use_kleene_ssm=True,          # ON
        use_kleene_attention=True,    # ON: full Kleene architecture
        kleene_attn_iters=3,
        use_differential_attention=True,
        use_cross_window_memory=True,
        cross_window_size=1024,
        inference_context_multiplier=16,
        s2_max_iters=4,
        quantization_bits=8,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def kyro_config(**overrides) -> ModelConfig:
    """
    Kyro: 3-5B parameters. Cloud-only capability ceiling.
    Full architecture, maximum context, maximum expressivity.
    """
    cfg = ModelConfig(
        tier="kyro",
        d_model=2048,
        n_heads=32,
        n_blocks=12,
        context_len=2048,
        d_reservoir=1024,
        padic_depth=12,
        max_gists=256,
        gist_top_k=16,
        kleene_ssm_d_state=256,
        kleene_ssm_iters=6,
        use_kleene_ssm=True,
        use_kleene_attention=True,
        kleene_attn_iters=4,
        use_differential_attention=True,
        use_cross_window_memory=True,
        cross_window_size=2048,
        inference_context_multiplier=32,
        s2_max_iters=6,
        s2_eps=0.001,
        quantization_bits=8,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
