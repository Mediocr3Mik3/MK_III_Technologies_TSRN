"""
Tokenize raw text shards with TMT and write packed token shards.

Output format (per shard):
    <name>_tok_<NNNNN>.bin   uint32 little-endian, document-packed,
                              EOS-delimited (token id of "<eos>")
    <name>_tok_<NNNNN>.idx   numpy array of (start, length) for each document

Usage::

    python -m research.cloud.azure.data.tokenize_shard \\
        --raw-dir   /mnt/blob/raw/pretrain \\
        --tokenizer /mnt/blob/tokenizers/tmt_32k.json \\
        --output    /mnt/blob/tokens/pretrain \\
        --workers 8 \\
        --shard-tokens 200000000   # 200M tokens per .bin (~800 MiB at uint32)
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.tropical_tokenizer import TropicalMergingTokenizer  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shard reading
# ---------------------------------------------------------------------------

def _iter_records(shard_path: Path) -> Iterator[dict]:
    if shard_path.suffix == ".zst":
        import zstandard as zstd  # type: ignore
        with open(shard_path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as rdr:
                buf = rdr.read().decode("utf-8", errors="replace")
        for line in buf.splitlines():
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    else:
        with open(shard_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_TOKENIZER: Optional[TropicalMergingTokenizer] = None
_EOS_ID: Optional[int] = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER, _EOS_ID
    _TOKENIZER = TropicalMergingTokenizer.load(tokenizer_path)
    _EOS_ID = _TOKENIZER.vocab_to_id.get("<eos>", 2)


def _tokenize_shard(args) -> dict:
    shard_path, out_dir, name, shard_idx, shard_tokens = args
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert _TOKENIZER is not None and _EOS_ID is not None

    bin_path = out_dir / f"{name}_tok_{shard_idx:05d}.bin"
    idx_path = out_dir / f"{name}_tok_{shard_idx:05d}.idx.npy"
    tmp_bin = bin_path.with_suffix(".bin.tmp")
    tmp_idx = idx_path.with_suffix(".npy.tmp")

    if bin_path.exists() and idx_path.exists():
        return {"shard": shard_idx, "skipped": True}

    starts: List[int] = []
    lengths: List[int] = []
    total = 0
    t0 = time.time()
    with open(tmp_bin, "wb") as fout:
        for rec in _iter_records(Path(shard_path)):
            text = rec.get("text") or ""
            if not text:
                continue
            ids = _TOKENIZER.encode(text)
            if not ids:
                continue
            ids.append(_EOS_ID)
            arr = np.asarray(ids, dtype=np.uint32)
            starts.append(total)
            lengths.append(len(arr))
            total += len(arr)
            fout.write(arr.tobytes())
            if total >= shard_tokens:
                break

    np.save(tmp_idx, np.stack([np.asarray(starts, dtype=np.int64),
                               np.asarray(lengths, dtype=np.int64)],
                              axis=1))
    os.replace(tmp_bin, bin_path)
    os.replace(tmp_idx, idx_path)
    return {
        "shard": shard_idx,
        "tokens": int(total),
        "docs": len(starts),
        "elapsed_s": round(time.time() - t0, 1),
        "path": str(bin_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    p.add_argument("--shard-tokens", type=int, default=200_000_000,
                   help="Approx tokens per output shard")
    p.add_argument("--only", nargs="*", default=None,
                   help="Only tokenize these dataset names")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    raw_root = Path(args.raw_dir)
    out_root = Path(args.output)

    # group shards by dataset (subdir name)
    work = []
    for ds_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        if args.only and ds_dir.name not in args.only:
            continue
        shards = sorted(ds_dir.glob("*shard_*.jsonl*"))
        for i, shard in enumerate(shards):
            work.append((str(shard),
                         str(out_root / ds_dir.name),
                         ds_dir.name,
                         i,
                         args.shard_tokens))

    if not work:
        logger.error("no shards found in %s", raw_root)
        sys.exit(1)

    logger.info("Tokenizing %d shards with %d workers...",
                len(work), args.workers)

    if args.workers <= 1:
        _init_worker(args.tokenizer)
        results = [_tokenize_shard(w) for w in work]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers,
                      initializer=_init_worker,
                      initargs=(args.tokenizer,)) as pool:
            results = pool.map(_tokenize_shard, work)

    total_tokens = sum(r.get("tokens", 0) for r in results)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "_summary.json").write_text(
        json.dumps({"shards": results, "total_tokens": total_tokens}, indent=2)
    )
    logger.info("Done. %d shards, %.2fB tokens.",
                len(results), total_tokens / 1e9)


if __name__ == "__main__":
    main()
