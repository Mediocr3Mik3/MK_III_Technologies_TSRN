"""
TSRN special-token inventory (FROZEN, single source of truth).
MK III Technologies.

Both tokenizer trainers import build_special_tokens() from here so the frozen
token IDs are IDENTICAL regardless of which trainer produced the vocabulary:
  * research.cloud.azure.data.train_tmt          (tropical-synergy, cloud/local)
  * E:/ml/tsrn_data/pipeline/extend_tokenizer.py (frequency-BPE, offline fallback)

Changing this list invalidates every tokenized shard and requires a full
re-tokenize, so do NOT reorder or remove entries. New control tokens should
consume the reserved_* block to keep existing IDs stable.

Component-resonant inventory (relative order is part of the freeze):
  padic    : level_0 .. level_15        (hierarchy depth)
  rg       : scale_0 .. scale_7         (multi-scale)
  kleene   : dep                        (dependency edge)
  sheaf    : consistent, contradict     (agreement / violation)
  gist     : gist, summary              (compression)
  reasoning: think, /think, answer, uncertain
  world    : world_state, update
  chat     : system, user, assistant, tool
  tool-call: tool_call, tool_response, tool_name, args, end
  reserved : reserved_0 .. reserved_47  (future control tokens; freezes IDs)
"""

from __future__ import annotations

from typing import List


def _ctl(name: str) -> str:
    """Wrap a plain name as a TSRN control token: foo -> the bracketed form."""
    return "<|" + name + "|>"


def build_special_tokens() -> List[str]:
    names: List[str] = []
    # p-adic hierarchy depth markers
    names += [f"level_{i}" for i in range(16)]
    # RG multi-scale markers
    names += [f"scale_{i}" for i in range(8)]
    # kleene / sheaf / gist component markers
    names += ["dep", "consistent", "contradict", "gist", "summary"]
    # reasoning + world-model control tokens
    names += ["think", "/think", "answer", "uncertain", "world_state", "update"]
    # chat roles
    names += ["system", "user", "assistant", "tool"]
    # tool-call structural tokens (tools themselves are JSON-encoded payloads)
    names += ["tool_call", "tool_response", "tool_name", "args", "end"]
    tokens = [_ctl(n) for n in names]
    # reserved block to freeze IDs for future control tokens without re-sharding
    tokens += [_ctl(f"reserved_{i}") for i in range(48)]
    # de-dupe, preserve order
    seen: set = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


SPECIAL_TOKENS: List[str] = build_special_tokens()
NUM_SPECIAL_TOKENS: int = len(SPECIAL_TOKENS)
