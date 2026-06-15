"""
Weighted streaming dataset over packed token shards.

Reads ``*_tok_*.bin`` (uint32) + ``*.idx.npy`` produced by
:mod:`tokenize_shard`, and yields contiguous windows of length
``context_len + 1`` so the trainer can split into ``input_ids`` / ``targets``.

Sampling
--------
At each step we pick a dataset with probability proportional to
``manifest.weight[i]``, then a random valid offset inside that dataset's
token stream.  Windows can span document boundaries — consistent with how
typical pretraining stacks pack documents end-to-end.

DDP-aware
---------
``rank``/``world`` shards the dataset by *shard* (not by token), so workers
don't read the same .bin file at the same time.

Usage::

    ds = TokenShardStream(
        manifest="research/cloud/azure/data/manifests/pretrain_mix.yaml",
        tokens_dir="/mnt/blob/tokens/pretrain",
        context_len=2048,
        seed=42, rank=0, world=1,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class _DatasetSpec:
    name: str
    weight: float
    shards: List[Path] = field(default_factory=list)
    sample_weight_multiplier: float = 1.0
    component: str = "general"


class Curriculum:
    """Soft scheduled component-resonant curriculum.

    Each component becomes "active" at the start of the earliest phase that
    emphasizes it, ramping from ``floor_weight`` to 1.0 over ``ramp`` of
    training progress, and staying engaged afterwards (monotonic — avoids
    catastrophic forgetting). Before activation it sits at the floor so no
    component is fully starved.
    """

    def __init__(self, cfg: dict, ramp: float = 0.05):
        self.floor = float(cfg.get("floor_weight", 0.15))
        self.ramp = max(1e-6, float(cfg.get("ramp", ramp)))
        phases = cfg.get("phases", [])
        # activation start (fraction) per component = start of first phase
        # that emphasizes it; phase start = previous phase's `until` (0 first).
        self.activation: Dict[str, float] = {}
        prev_until = 0.0
        for ph in phases:
            start = prev_until
            for comp in ph.get("emphasize", []):
                if comp not in self.activation:
                    self.activation[comp] = start
            prev_until = float(ph.get("until", prev_until))

    def multiplier(self, component: str, progress: float) -> float:
        act = self.activation.get(component)
        if act is None:
            return self.floor  # never emphasized -> stays at floor
        if progress >= act:
            return 1.0
        # ramp up over [act - ramp, act]
        x = (progress - (act - self.ramp)) / self.ramp
        x = min(1.0, max(0.0, x))
        return self.floor + (1.0 - self.floor) * x


class TokenShardStream(IterableDataset):
    def __init__(
        self,
        manifest: str,
        tokens_dir: str,
        context_len: int = 2048,
        seed: int = 42,
        rank: int = 0,
        world: int = 1,
        eos_id: int = 2,
        only: Optional[List[str]] = None,
        curriculum: Optional[dict] = None,
        progress_path: Optional[str] = None,
        progress_refresh: int = 512,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.seed = seed
        self.rank = rank
        self.world = world
        self.eos_id = eos_id
        self.progress_path = progress_path
        self.progress_refresh = max(1, int(progress_refresh))

        with open(manifest, "r", encoding="utf-8") as f:
            m = yaml.safe_load(f)

        # curriculum: explicit arg overrides manifest's `curriculum:` block.
        cur_cfg = curriculum if curriculum is not None else m.get("curriculum")
        self.curriculum: Optional[Curriculum] = (
            Curriculum(cur_cfg) if cur_cfg and cur_cfg.get("mode", "soft") != "off"
            else None)

        tokens_root = Path(tokens_dir)
        self.specs: List[_DatasetSpec] = []
        for entry in m["mixture"]:
            name = entry["name"]
            if only and name not in set(only):
                continue
            ds_dir = tokens_root / name
            if not ds_dir.exists():
                logger.warning("missing %s under %s, skipping", name, tokens_dir)
                continue
            shards = sorted(ds_dir.glob(f"{name}_tok_*.bin"))
            if not shards:
                logger.warning("no shards for %s, skipping", name)
                continue
            # rank-shard the .bin files so workers don't overlap
            if world > 1:
                shards = [s for i, s in enumerate(shards) if i % world == rank]
                if not shards:
                    # rank starvation: fall back to all shards
                    shards = sorted(ds_dir.glob(f"{name}_tok_*.bin"))
            self.specs.append(_DatasetSpec(
                name=name,
                weight=float(entry["weight"]),
                shards=shards,
                sample_weight_multiplier=float(
                    entry.get("sample_weight_multiplier", 1.0)),
                component=str(entry.get("component", "general")),
            ))

        if not self.specs:
            raise RuntimeError("no datasets loaded; check manifest + tokens_dir")

        # normalize weights
        total = sum(s.weight for s in self.specs)
        for s in self.specs:
            s.weight = s.weight / total
        self._base_weights = np.asarray([s.weight for s in self.specs],
                                        dtype=np.float64)
        self._weights = self._base_weights.copy()
        # curriculum progress cache (per-worker). Start at refresh so the first
        # pick computes curriculum weights immediately.
        self._cached_progress = 0.0
        self._picks_since_refresh = self.progress_refresh

        # mmap each shard once (lazy by mmap)
        self._mmaps: Dict[Path, np.memmap] = {}
        self._open_mmaps()

    def _open_mmaps(self) -> None:
        """Re-populate self._mmaps from self.specs (used by __setstate__)."""
        self._mmaps.clear()
        for s in self.specs:
            for sh in s.shards:
                self._mmaps[sh] = np.memmap(sh, dtype=np.uint32, mode="r")

    def __getstate__(self):
        """Drop unpicklable memmaps; re-open via __setstate__."""
        state = self.__dict__.copy()
        state.pop("_mmaps", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_mmaps()

    @staticmethod
    def write_progress(path: str, frac: float) -> None:
        """Trainer calls this periodically to publish training progress [0,1]."""
        try:
            tmp = str(path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(f"{max(0.0, min(1.0, float(frac))):.6f}")
            import os
            os.replace(tmp, path)
        except OSError:
            pass

    def _read_progress(self) -> float:
        if not self.progress_path:
            return 0.0
        try:
            with open(self.progress_path, "r", encoding="utf-8") as f:
                return float(f.read().strip() or 0.0)
        except (OSError, ValueError):
            return self._cached_progress

    def _refresh_curriculum_weights(self) -> None:
        if self.curriculum is None:
            return
        self._picks_since_refresh += 1
        if self._picks_since_refresh < self.progress_refresh:
            return
        self._picks_since_refresh = 0
        p = self._read_progress()
        self._cached_progress = p
        mult = np.asarray(
            [self.curriculum.multiplier(s.component, p) for s in self.specs],
            dtype=np.float64)
        w = self._base_weights * mult
        total = w.sum()
        self._weights = (w / total) if total > 0 else self._base_weights

    def _pick_window(self, rng: random.Random) -> Tuple[np.ndarray, str]:
        self._refresh_curriculum_weights()
        spec_idx = rng.choices(range(len(self.specs)), weights=self._weights, k=1)[0]
        spec = self.specs[spec_idx]
        shard = rng.choice(spec.shards)
        arr = self._mmaps[shard]
        n = len(arr)
        if n <= self.context_len + 1:
            # very small shard: just pad-loop it
            pad = np.full(self.context_len + 1 - n, self.eos_id, dtype=np.uint32)
            window = np.concatenate([arr, pad])
        else:
            start = rng.randint(0, n - self.context_len - 2)
            window = np.asarray(arr[start:start + self.context_len + 1])
        return window, spec.name

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        nworkers = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + 1000 * self.rank + wid)
        # also sub-shard within DDP rank by worker:
        # (kept simple — sampling is random anyway, this just diversifies seeds)
        del nworkers

        while True:
            window, ds_name = self._pick_window(rng)
            x = torch.from_numpy(window[:-1].astype(np.int64))
            y = torch.from_numpy(window[1:].astype(np.int64))
            sample_weight = next(
                (s.sample_weight_multiplier for s in self.specs if s.name == ds_name),
                1.0,
            )
            yield {
                "input_ids": x,
                "targets": y,
                "dataset": ds_name,
                "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            }
