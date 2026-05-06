"""
Download HuggingFace datasets according to a YAML manifest.

Usage::

    python -m research.cloud.azure.data.download \\
        --manifest research/cloud/azure/data/manifests/pretrain_mix.yaml \\
        --output  /mnt/blob/raw/pretrain \\
        --workers 4

What it does
------------
* Reads the manifest, iterates each dataset entry
* For each entry, streams from HF Hub (avoids materializing the full dataset
  on disk first) and writes ``raw_text/<name>/shard_NNNN.jsonl.zst``
* Stops once ``tokens_b`` (or ``examples_k`` for SFT/DPO) target is hit
* Skips entries with ``local_path`` (proprietary data is staged elsewhere)
* Resumes safely: shards are atomically renamed after flush

Notes
-----
* We don't tokenize here — that's :mod:`tokenize_shard`.
* "Tokens" here are word-tokens for budgeting; we use a 4-byte/token
  approximation when the manifest says ``tokens_b``. The tokenize step
  enforces the real per-stage token budget.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Approximate bytes-per-token for budgeting.  Refined by tokenize step.
APPROX_BYTES_PER_TOKEN = 4
SHARD_BYTES = 256 * 1024 * 1024      # 256 MiB per shard (uncompressed)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

@dataclass
class DatasetEntry:
    name: str
    hf_dataset: Optional[str]
    hf_config: Optional[str]
    split: Optional[str]
    text_field: Optional[str]
    target_bytes: Optional[int]      # for pretrain (tokens_b * 4 GiB-ish)
    target_examples: Optional[int]   # for SFT/DPO (examples_k * 1000)
    local_path: Optional[str]
    filter_expr: Optional[str]
    raw: Dict[str, Any]


def _entries_from_manifest(manifest_path: str) -> List[DatasetEntry]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)

    out: List[DatasetEntry] = []
    for e in m["mixture"]:
        tokens_b = e.get("tokens_b")
        examples_k = e.get("examples_k") or e.get("pairs_k")
        out.append(DatasetEntry(
            name=e["name"],
            hf_dataset=e.get("hf_dataset"),
            hf_config=e.get("hf_config"),
            split=e.get("split"),
            text_field=e.get("text_field", "text"),
            target_bytes=int(tokens_b * 1e9 * APPROX_BYTES_PER_TOKEN) if tokens_b else None,
            target_examples=int(examples_k * 1000) if examples_k else None,
            local_path=e.get("local_path"),
            filter_expr=e.get("filter"),
            raw=e,
        ))
    return out


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """JSONL writer that rotates shards at SHARD_BYTES.

    Uses ``.tmp`` -> rename for atomicity on blobfuse.
    """

    def __init__(self, out_dir: Path, name: str, compress: bool = True):
        self.out_dir = out_dir
        self.name = name
        self.compress = compress
        self.shard_idx = 0
        self.bytes_in_shard = 0
        self.total_bytes = 0
        self.total_records = 0
        self._fp: Optional[io.BufferedWriter] = None
        self._raw: Optional[io.BufferedWriter] = None
        self._cur_path: Optional[Path] = None
        self._cur_tmp: Optional[Path] = None
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _open_shard(self) -> None:
        ext = ".jsonl.zst" if self.compress else ".jsonl"
        self._cur_path = self.out_dir / f"{self.name}_shard_{self.shard_idx:05d}{ext}"
        self._cur_tmp = self._cur_path.with_suffix(self._cur_path.suffix + ".tmp")
        if self.compress:
            try:
                import zstandard as zstd  # type: ignore
            except ImportError as ex:
                raise RuntimeError(
                    "zstandard not installed; pass --no-compress or "
                    "`pip install zstandard`"
                ) from ex
            self._raw = open(self._cur_tmp, "wb")
            cctx = zstd.ZstdCompressor(level=3, threads=2)
            self._fp = cctx.stream_writer(self._raw)
        else:
            self._raw = open(self._cur_tmp, "wb")
            self._fp = self._raw

    def write(self, record: Dict[str, Any]) -> None:
        if self._fp is None:
            self._open_shard()
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        assert self._fp is not None
        self._fp.write(line)
        self.bytes_in_shard += len(line)
        self.total_bytes += len(line)
        self.total_records += 1
        if self.bytes_in_shard >= SHARD_BYTES:
            self._rotate()

    def _rotate(self) -> None:
        self.close()
        self.shard_idx += 1
        self.bytes_in_shard = 0

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            finally:
                if self._raw is not None and self._raw is not self._fp:
                    self._raw.close()
            assert self._cur_tmp is not None and self._cur_path is not None
            os.replace(self._cur_tmp, self._cur_path)
            logger.info("wrote %s (%.1f MiB)",
                        self._cur_path.name,
                        self.bytes_in_shard / (1024 * 1024))
            self._fp = None
            self._raw = None
            self._cur_tmp = None
            self._cur_path = None


# ---------------------------------------------------------------------------
# Streaming download
# ---------------------------------------------------------------------------

def _stream_hf(entry: DatasetEntry) -> Iterator[Dict[str, Any]]:
    """Stream rows from HuggingFace datasets in `streaming=True` mode."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as ex:
        raise RuntimeError(
            "huggingface `datasets` not installed; "
            "`pip install datasets zstandard pyyaml`"
        ) from ex

    ds = load_dataset(
        entry.hf_dataset,
        entry.hf_config,
        split=entry.split,
        streaming=True,
    )
    if entry.filter_expr:
        # Very small DSL: only supports `field == 'value'` and `or`
        ds = ds.filter(_compile_filter(entry.filter_expr))

    for ex in ds:
        text = ex.get(entry.text_field or "text")
        if not text:
            continue
        yield {"text": text, "meta": {k: v for k, v in ex.items() if k != entry.text_field}}


def _compile_filter(expr: str):
    """Compile a tiny boolean expression: ``field == 'val' or field == 'val2'``."""
    import re
    parts = [p.strip() for p in re.split(r"\s+or\s+", expr)]
    rules = []
    for p in parts:
        m = re.match(r"(\w+)\s*==\s*'([^']*)'", p)
        if not m:
            raise ValueError(f"unsupported filter expr: {p}")
        rules.append((m.group(1), m.group(2)))
    def fn(ex: Dict[str, Any]) -> bool:
        return any(ex.get(k) == v for k, v in rules)
    return fn


def _stream_local(entry: DatasetEntry) -> Iterator[Dict[str, Any]]:
    """Stream rows from a local JSONL file (proprietary data)."""
    assert entry.local_path is not None
    if not Path(entry.local_path).exists():
        logger.warning("local_path %s does not exist, skipping %s",
                       entry.local_path, entry.name)
        return iter(())
    with open(entry.local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield rec


# ---------------------------------------------------------------------------
# Per-entry download
# ---------------------------------------------------------------------------

def download_entry(entry: DatasetEntry, out_dir: Path,
                   compress: bool = True,
                   max_records: Optional[int] = None,
                   resume: bool = True) -> Dict[str, Any]:
    entry_dir = out_dir / entry.name
    entry_dir.mkdir(parents=True, exist_ok=True)

    # Resume: if a manifest.json exists with the target met, skip
    manifest_path = entry_dir / "manifest.json"
    if resume and manifest_path.exists():
        try:
            done = json.loads(manifest_path.read_text())
            if done.get("complete"):
                logger.info("[skip] %s already complete (%d records, %.1f GiB)",
                            entry.name, done["records"], done["bytes"] / 2**30)
                return done
        except Exception:
            pass

    logger.info("[download] %s -> %s", entry.name, entry_dir)
    writer = ShardWriter(entry_dir, entry.name, compress=compress)

    src_iter: Iterator[Dict[str, Any]]
    if entry.local_path:
        src_iter = _stream_local(entry)
    else:
        src_iter = _stream_hf(entry)

    t0 = time.time()
    last_log = t0
    for rec in src_iter:
        writer.write(rec)
        if max_records is not None and writer.total_records >= max_records:
            break
        if entry.target_bytes and writer.total_bytes >= entry.target_bytes:
            break
        if entry.target_examples and writer.total_records >= entry.target_examples:
            break
        now = time.time()
        if now - last_log > 30:
            mbps = writer.total_bytes / (now - t0) / 1e6
            logger.info("  [%s] %d records, %.1f GiB, %.1f MB/s",
                        entry.name, writer.total_records,
                        writer.total_bytes / 2**30, mbps)
            last_log = now

    writer.close()
    summary = {
        "name": entry.name,
        "records": writer.total_records,
        "bytes": writer.total_bytes,
        "shards": writer.shard_idx + 1,
        "elapsed_s": round(time.time() - t0, 1),
        "complete": True,
    }
    manifest_path.write_text(json.dumps(summary, indent=2))
    logger.info("[done] %s: %d recs, %.1f GiB in %.1fs",
                entry.name, writer.total_records,
                writer.total_bytes / 2**30, summary["elapsed_s"])
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="YAML mixture manifest path")
    p.add_argument("--output", required=True,
                   help="Output dir (e.g. /mnt/blob/raw/pretrain)")
    p.add_argument("--workers", type=int, default=2,
                   help="Concurrent dataset downloads")
    p.add_argument("--no-compress", action="store_true",
                   help="Disable zstd compression")
    p.add_argument("--only", nargs="*", default=None,
                   help="Only download these entry names")
    p.add_argument("--max-records-per-entry", type=int, default=None,
                   help="Cap records per dataset (debugging)")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-download even if already complete")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    entries = _entries_from_manifest(args.manifest)
    if args.only:
        entries = [e for e in entries if e.name in set(args.only)]
        if not entries:
            logger.error("No entries match --only %s", args.only)
            sys.exit(1)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    if args.workers <= 1:
        for e in entries:
            summaries.append(download_entry(
                e, out,
                compress=not args.no_compress,
                max_records=args.max_records_per_entry,
                resume=not args.no_resume,
            ))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(download_entry, e, out,
                          not args.no_compress,
                          args.max_records_per_entry,
                          not args.no_resume): e.name
                for e in entries
            }
            for fut in as_completed(futs):
                summaries.append(fut.result())

    # Aggregate manifest
    agg = {
        "manifest": args.manifest,
        "datasets": summaries,
        "total_bytes": sum(s["bytes"] for s in summaries),
        "total_records": sum(s["records"] for s in summaries),
    }
    (out / "_aggregate.json").write_text(json.dumps(agg, indent=2))
    logger.info("All done. Total: %d records, %.1f GiB",
                agg["total_records"], agg["total_bytes"] / 2**30)


if __name__ == "__main__":
    main()
