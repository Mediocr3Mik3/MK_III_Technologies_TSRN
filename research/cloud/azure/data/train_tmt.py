"""
Train the Tropical Merging Tokenizer (TMT) on a sample of the pretraining
corpus.

Usage::

    python -m research.cloud.azure.data.train_tmt \\
        --raw-dir   /mnt/blob/raw/pretrain \\
        --output    /mnt/blob/tokenizers/tmt_32k.json \\
        --vocab-size 32000 \\
        --sample-bytes 5e9 \\
        --bootstrap-steps 5000 \\
        --rounds 14

Pipeline (per `tropical_tokenizer.py`):
  Phase 0 — sample N bytes from raw shards, count adjacent pairs
  Phase 1-3 — repeated tropical-synergy merges, with TSRN re-training
              between rounds to refresh embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

import torch

# Make the repo importable when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.tropical_tokenizer import TropicalMergingTokenizer  # noqa: E402
from research.cloud.azure.data.special_tokens import (  # noqa: E402
    build_special_tokens, NUM_SPECIAL_TOKENS)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus sampling
# ---------------------------------------------------------------------------

def _iter_shard_lines(raw_dir: Path) -> Iterator[str]:
    """Yield ``text`` field of every record across all shards.

    Supports ``.jsonl`` and ``.jsonl.zst`` shards (output of download.py).
    """
    shards = sorted(raw_dir.rglob("*shard_*.jsonl*"))
    if not shards:
        raise FileNotFoundError(f"no shards under {raw_dir}")
    for shard in shards:
        if shard.suffix == ".zst":
            try:
                import zstandard as zstd  # type: ignore
            except ImportError as ex:
                raise RuntimeError("install zstandard to read .zst shards") from ex
            with open(shard, "rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as rdr:
                    text_io = (rdr.read().decode("utf-8", errors="replace"))
                    for line in text_io.splitlines():
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        yield rec.get("text", "")
        else:
            with open(shard, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield rec.get("text", "")


def sample_corpus(raw_dir: Path, target_bytes: int) -> List[List[str]]:
    """Collect documents up to ``target_bytes`` of UTF-8 text.

    Returns each document as a list of single-character "byte" tokens —
    matching what TMT expects in ``update_pair_counts``.
    """
    docs: List[List[str]] = []
    total = 0
    t0 = time.time()
    for text in _iter_shard_lines(raw_dir):
        b = len(text.encode("utf-8"))
        if b == 0:
            continue
        docs.append(list(text))
        total += b
        if total >= target_bytes:
            break
    logger.info("sampled %d docs (%.2f GiB) in %.1fs",
                len(docs), total / 2**30, time.time() - t0)
    return docs


# ---------------------------------------------------------------------------
# Bootstrap embeddings (Phase 0)
# ---------------------------------------------------------------------------

def _bootstrap_embeddings(vocab_size: int, d_model: int = 512) -> torch.Tensor:
    """Initialize embeddings.

    For the cloud TMT run we use small-variance random init followed by a
    short TSRN training run on the byte-tokenized sample.  That training
    happens inside the merge loop (between rounds) so here we just produce
    a starting tensor.
    """
    g = torch.Generator().manual_seed(0)
    return torch.randn(vocab_size, d_model, generator=g) * 0.05


# ---------------------------------------------------------------------------
# Embedding refresh (Phase 3)
# ---------------------------------------------------------------------------

def _refresh_embeddings(
    embeddings: torch.Tensor,
    pair_counts,
    vocab_to_id,
    new_pairs,
    *,
    lr: float = 0.02,
    steps: int = 250,
) -> torch.Tensor:
    """Cheap embedding refresh between merge rounds.

    Pulls each merged token's embedding toward the tropical max of its parts:
        e_merged ← elementwise_max(e_a, e_b)  (max-plus blend)

    This is much cheaper than re-training a TSRN inside a tokenizer loop and
    keeps the tropical geometry consistent.  A real Phase 3 (full TSRN
    retrain) is invoked separately by the orchestration layer when the
    pretrain stage is launched.
    """
    eps = 1e-3
    for a, b, merged in new_pairs:
        if merged not in vocab_to_id:
            continue
        i_m = vocab_to_id[merged]
        i_a = vocab_to_id.get(a)
        i_b = vocab_to_id.get(b)
        if i_a is None or i_b is None:
            continue
        target = torch.maximum(embeddings[i_a], embeddings[i_b])
        embeddings[i_m] = (1 - lr * steps / 100) * embeddings[i_m] + (lr * steps / 100) * target
    # tiny noise to break ties next round
    embeddings = embeddings + torch.randn_like(embeddings) * eps
    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True,
                   help="Directory of raw shards (output of download.py)")
    p.add_argument("--output", required=True,
                   help="Where to save tmt_*.json (and _embeddings.pt)")
    p.add_argument("--vocab-size", type=int, default=48000)
    p.add_argument("--sample-bytes", type=float, default=5e9,
                   help="Bytes of text to sample for pair counts (default 5GiB)")
    p.add_argument("--bootstrap-steps", type=int, default=5000,
                   help="Phase 0 model training steps (logged but not run here)")
    p.add_argument("--rounds", type=int, default=22)
    p.add_argument("--k-pairs-per-round", type=int, default=2300,
                   help="Merges per round. ~21 rounds * 2300 reaches 48k vocab.")
    p.add_argument("--no-special-tokens", action="store_true",
                   help="Skip the frozen TSRN special-token inventory (debug only).")
    p.add_argument("--theta", type=float, default=0.1,
                   help="Synergy threshold for accepting a merge")
    p.add_argument("--d-model", type=int, default=512,
                   help="Bootstrap embedding dim (matches model d_model)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 0: sampling corpus (%.1f GiB)...",
                args.sample_bytes / 2**30)
    docs = sample_corpus(raw_dir, int(args.sample_bytes))

    logger.info("Phase 0: initializing TMT and counting pairs...")
    specials = None if args.no_special_tokens else build_special_tokens()
    if specials:
        logger.info("reserving %d frozen TSRN special tokens", len(specials))
    tok = TropicalMergingTokenizer(
        vocab_size=args.vocab_size,
        theta=args.theta,
        special_tokens=specials,
    )
    tok.update_pair_counts(docs)

    logger.info("Phase 0: bootstrap embeddings (d=%d)", args.d_model)
    embeddings = _bootstrap_embeddings(len(tok.vocab), d_model=args.d_model)
    tok.embeddings = embeddings

    target = args.vocab_size
    for r in range(args.rounds):
        if tok.current_vocab_size >= target:
            logger.info("target vocab reached (%d)", tok.current_vocab_size)
            break
        logger.info("Round %d/%d  vocab=%d",
                    r + 1, args.rounds, tok.current_vocab_size)
        merges = tok.merge_round(embeddings, k_pairs=args.k_pairs_per_round)
        if not merges:
            logger.warning("no merges this round (theta=%.3f too strict?)",
                           args.theta)
            break
        embeddings = tok.embeddings  # merge_round may have appended rows
        embeddings = _refresh_embeddings(
            embeddings, tok.pair_counts, tok.vocab_to_id, merges
        )
        tok.embeddings = embeddings
        logger.info("  merged %d pairs, vocab now %d",
                    len(merges), tok.current_vocab_size)

    # sanity: every frozen special token must be in the final vocabulary
    if specials:
        missing = [t for t in specials if t not in tok.vocab_to_id]
        if missing:
            raise RuntimeError(
                f"{len(missing)} special tokens missing from vocab: {missing[:5]}...")
        logger.info("verified %d special tokens present in vocab", len(specials))

    tok.save(str(out_path))
    logger.info("saved tokenizer -> %s (vocab=%d, specials=%d)",
                out_path, tok.current_vocab_size,
                0 if not specials else NUM_SPECIAL_TOKENS)


if __name__ == "__main__":
    main()
