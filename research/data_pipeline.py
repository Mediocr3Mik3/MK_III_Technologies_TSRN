"""
TSRN Three-Stage Training Data Pipeline
=========================================
Stage 1: Foundation Pretraining  (~75B tokens, streaming)
Stage 2: Supervised Fine-Tuning   (instruction datasets)
Stage 3: Preference Alignment       (DPO pairs)

All stages use HuggingFace ``datasets`` streaming mode to keep RAM low.
On a single Windows workstation with DirectML this is essential.

Dependencies (install before use):
  pip install datasets transformers tokenizers huggingface_hub
"""

import os
import random
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Iterator, Callable
from itertools import cycle

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# ---------------------------------------------------------------------------
#  Optional imports with graceful degradation
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset, interleave_datasets
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# ---------------------------------------------------------------------------
#  Tokenizer Wrapper
# ---------------------------------------------------------------------------

class TokenizerWrapper:
    """Wraps byte-level (vocab=256) or HuggingFace BPE tokenizers."""

    def __init__(self, vocab_size: int = 256, hf_tokenizer_name: Optional[str] = None,
                 context_len: int = 512):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self._hf = None
        self._is_byte = False

        if hf_tokenizer_name and _HAS_TRANSFORMERS:
            self._hf = AutoTokenizer.from_pretrained(hf_tokenizer_name, use_fast=True)
            self._hf.model_max_length = context_len
            self.vocab_size = len(self._hf)
        else:
            self._is_byte = True
            self.vocab_size = 256

    def encode(self, text: str) -> List[int]:
        if self._is_byte:
            return list(text.encode("utf-8", errors="ignore"))
        return self._hf.encode(text, add_special_tokens=False)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        if self._is_byte:
            return [self.encode(t) for t in texts]
        return [out["input_ids"] for out in self._hf(texts, add_special_tokens=False)]

    def decode(self, ids: List[int]) -> str:
        if self._is_byte:
            return bytes(ids).decode("utf-8", errors="ignore")
        return self._hf.decode(ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
#  Chat format helpers (phone-specific SFT template)
# ---------------------------------------------------------------------------

CHAT_TEMPLATE = "<|system|>{system}<|user|>{user}<|assistant|>{assistant}<|end|>"

SFT_SYSTEM_PROMPT = (
    "You are a helpful phone assistant. You have access to: "
    "calendar, contacts, messages, reminders, web search, maps. "
    "Respond concisely. When a tool call is needed, output the exact tool name "
    "followed by JSON arguments."
)

DPO_SYSTEM_PROMPT = SFT_SYSTEM_PROMPT


def format_sft(system: str, user: str, assistant: str) -> str:
    return CHAT_TEMPLATE.format(system=system, user=user, assistant=assistant)


def format_dpo_prompt(system: str, user: str) -> str:
    return f"<|system|>{system}<|user|>{user}<|assistant|>"


# ---------------------------------------------------------------------------
#  Stage-1: Foundation Pretraining — Streaming Mixed Dataset
# ---------------------------------------------------------------------------

# Dataset registry: name -> (HF path, split, weight, text_column, subset)
PRETRAIN_REGISTRY: Dict[str, Tuple[str, str, float, str, Optional[str]]] = {
    "fineweb_edu":    ("HuggingFaceFW/fineweb-edu",    "train", 0.55,  "text", None),
    "open_web_math":  ("open-web-math/open-web-math",  "train", 0.10,  "text", None),
    "the_stack":      ("bigcode/the-stack",            "train", 0.20,  "content", "Python"),
    "wikipedia":      ("wikimedia/wikipedia",          "train", 0.10,  "text", "20231101.en"),
    "openhermes_raw": ("teknium/OpenHermes-2.5",       "train", 0.05,  "text", None),
}


def _resolve_split(path: str, subset: Optional[str]) -> str:
    """Try a few common splits and fall back to 'train'."""
    for split in ["train", "training", "all"]:
        try:
            ds = load_dataset(path, subset, split=split, streaming=True, trust_remote_code=True)
            next(iter(ds))  # trigger remote resolution
            return split
        except Exception:
            continue
    return "train"


class PretrainDatasetMixer(IterableDataset):
    """
    Streaming pretraining corpus with probabilistic mixing.

    Parameters
    ----------
    tokenizer : TokenizerWrapper
    context_len : int
        Block size (including the label shift).
    datasets : dict
        Subset of PRETRAIN_REGISTRY keys to actually load.
    weights : dict or None
        Override default mixture weights.  Must sum to 1.0.
    buffer_size : int
        Shuffle buffer for each sub-dataset before mixing.
    max_tokens : int or None
        Hard cap on total tokens yielded (for debugging / short runs).
    """

    def __init__(self,
                 tokenizer: TokenizerWrapper,
                 context_len: int = 512,
                 datasets: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 buffer_size: int = 10000,
                 max_tokens: Optional[int] = None,
                 ):
        super().__init__()
        if not _HAS_DATASETS:
            raise RuntimeError("HuggingFace ``datasets`` library is required.  Install: pip install datasets")
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens
        self._datasets = datasets or list(PRETRAIN_REGISTRY.keys())
        self._weights = weights or {k: v[2] for k, v in PRETRAIN_REGISTRY.items() if k in self._datasets}
        total_w = sum(self._weights.values())
        self._weights = {k: v / total_w for k, v in self._weights.items()}
        self._streams: Dict[str, Iterator] = {}

    def _open_stream(self, key: str) -> Iterator[Dict]:
        path, split, _, text_col, subset = PRETRAIN_REGISTRY[key]
        try:
            ds = load_dataset(path, subset, split=split, streaming=True, trust_remote_code=True)
        except Exception:
            split = _resolve_split(path, subset)
            ds = load_dataset(path, subset, split=split, streaming=True, trust_remote_code=True)
        # small local shuffle via buffer
        buffer: List[Dict] = []
        for ex in ds:
            buffer.append(ex)
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                for _ in range(self.buffer_size // 2):
                    yield buffer.pop()
        random.shuffle(buffer)
        for ex in buffer:
            yield ex

    def _tokenized_blocks(self, key: str) -> Iterator[torch.Tensor]:
        path, _, _, text_col, _ = PRETRAIN_REGISTRY[key]
        buf: List[int] = []
        for ex in self._open_stream(key):
            text = ex.get(text_col, "")
            if not text:
                continue
            ids = self.tokenizer.encode(text)
            buf.extend(ids)
            while len(buf) >= self.context_len + 1:
                block = buf[:self.context_len + 1]
                buf = buf[self.context_len:]
                yield torch.tensor(block, dtype=torch.long)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield (input_ids, label_ids) pairs."""
        workers = get_worker_info()
        worker_id = workers.id if workers else 0
        random.seed(42 + worker_id)
        # Build round-robin iterators for each dataset
        iters = {k: self._tokenized_blocks(k) for k in self._datasets}
        token_count = 0
        while True:
            if self.max_tokens and token_count >= self.max_tokens:
                break
            key = random.choices(list(self._weights.keys()),
                                 weights=list(self._weights.values()))[0]
            try:
                block = next(iters[key])
            except StopIteration:
                # restart exhausted stream
                iters[key] = self._tokenized_blocks(key)
                try:
                    block = next(iters[key])
                except StopIteration:
                    continue
            x = block[:-1]
            y = block[1:]
            token_count += x.numel()
            yield x, y


# ---------------------------------------------------------------------------
#  Stage-2: Supervised Fine-Tuning Dataset
# ---------------------------------------------------------------------------

SFT_REGISTRY: Dict[str, Tuple[str, Optional[str], str, Callable]] = {
    "openhermes":  ("teknium/OpenHermes-2.5",  None,       "train", lambda ex: (ex.get("system",""), ex.get("user",""), ex.get("assistant",""))),
    "sharegpt":    ("sharenlp/ShareGPT_Vicuna_unfiltered", None, "train", lambda ex: (SFT_SYSTEM_PROMPT, ex.get("human",""), ex.get("gpt",""))),
    "toolbench":   ("OpenBMB/ToolBench",       None,       "train", lambda ex: (SFT_SYSTEM_PROMPT, ex.get("query",""), ex.get("answer",""))),
    "toolalpaca":  ("TIGER-Lab/ToolAlpaca",    None,       "train", lambda ex: (SFT_SYSTEM_PROMPT, ex.get("instruction",""), ex.get("output",""))),
    # "systemchat":  placeholder – replace with actual HF path when available
}


class SFTDataset(IterableDataset):
    def __init__(self,
                 tokenizer: TokenizerWrapper,
                 context_len: int = 512,
                 datasets: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 synthetic_jsonl: Optional[str] = None,
                 buffer_size: int = 5000):
        super().__init__()
        if not _HAS_DATASETS:
            raise RuntimeError("HuggingFace ``datasets`` library is required.")
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.buffer_size = buffer_size
        self._datasets = datasets or [k for k in SFT_REGISTRY.keys() if k != "systemchat"]
        self._weights = weights
        self._synthetic_path = synthetic_jsonl

    def _hf_stream(self, key: str) -> Iterator[Tuple[str, str, str]]:
        path, subset, split, fmt = SFT_REGISTRY[key]
        try:
            ds = load_dataset(path, subset, split=split, streaming=True, trust_remote_code=True)
        except Exception:
            split = _resolve_split(path, subset)
            ds = load_dataset(path, subset, split=split, streaming=True, trust_remote_code=True)
        for ex in ds:
            try:
                system, user, assistant = fmt(ex)
                if user and assistant:
                    yield system, user, assistant
            except Exception:
                continue

    def _synthetic_stream(self) -> Iterator[Tuple[str, str, str]]:
        if not self._synthetic_path or not Path(self._synthetic_path).exists():
            return
        import json
        with open(self._synthetic_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    yield obj.get("system", SFT_SYSTEM_PROMPT), obj["user"], obj["assistant"]
                except Exception:
                    continue

    def _tokenized_stream(self, key: str) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for system, user, assistant in self._hf_stream(key):
            text = format_sft(system or SFT_SYSTEM_PROMPT, user, assistant)
            ids = self.tokenizer.encode(text)
            if len(ids) > self.context_len + 1:
                ids = ids[:self.context_len + 1]
            elif len(ids) < 2:
                continue
            block = ids + [0] * (self.context_len + 1 - len(ids))  # pad with 0 (null byte or pad_id)
            block_t = torch.tensor(block, dtype=torch.long)
            yield block_t[:-1], block_t[1:]

    def _synthetic_tokenized(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for system, user, assistant in self._synthetic_stream():
            text = format_sft(system or SFT_SYSTEM_PROMPT, user, assistant)
            ids = self.tokenizer.encode(text)
            if len(ids) > self.context_len + 1:
                ids = ids[:self.context_len + 1]
            elif len(ids) < 2:
                continue
            block = ids + [0] * (self.context_len + 1 - len(ids))
            block_t = torch.tensor(block, dtype=torch.long)
            yield block_t[:-1], block_t[1:]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        workers = get_worker_info()
        worker_id = workers.id if workers else 0
        random.seed(42 + worker_id)

        weights = self._weights
        if weights is None:
            n = len(self._datasets)
            if self._synthetic_path and Path(self._synthetic_path).exists():
                n += 1
            weights = {k: 1.0 / n for k in self._datasets}
            if self._synthetic_path and Path(self._synthetic_path).exists():
                weights["synthetic"] = 1.0 / n

        iters: Dict[str, Iterator] = {k: self._tokenized_stream(k) for k in self._datasets}
        if self._synthetic_path and Path(self._synthetic_path).exists():
            iters["synthetic"] = self._synthetic_tokenized()

        while True:
            key = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
            try:
                yield next(iters[key])
            except StopIteration:
                # refresh exhausted stream
                if key == "synthetic":
                    iters[key] = self._synthetic_tokenized()
                else:
                    iters[key] = self._tokenized_stream(key)
                try:
                    yield next(iters[key])
                except StopIteration:
                    continue


# ---------------------------------------------------------------------------
#  Stage-3: DPO Preference Dataset
# ---------------------------------------------------------------------------

class DPODataset(torch.utils.data.Dataset):
    """
    Loads pre-generated DPO preference pairs.

    Each sample is a dict with keys:
        prompt   : str   (the user query + system prompt, up to assistant tag)
        chosen   : str   (preferred assistant response)
        rejected : str   (less-preferred assistant response)

    If ``jsonl_path`` does not exist, the dataset is empty and you must
    generate pairs first (see ``generate_dpo_pairs`` below).
    """

    def __init__(self,
                 tokenizer: TokenizerWrapper,
                 context_len: int = 512,
                 jsonl_path: str = "data/dpo_pairs.jsonl"):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples: List[Dict[str, str]] = []
        if Path(jsonl_path).exists():
            import json
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        self.samples.append(json.loads(line))
                    except Exception:
                        continue

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_pair(self, prompt: str, chosen: str, rejected: str) -> Tuple[torch.Tensor, ...]:
        p_ids = self.tokenizer.encode(prompt)
        c_ids = self.tokenizer.encode(chosen)
        r_ids = self.tokenizer.encode(rejected)

        def make_block(prefix: List[int], suffix: List[int]):
            ids = prefix + suffix
            if len(ids) > self.context_len + 1:
                ids = ids[:self.context_len + 1]
            ids += [0] * (self.context_len + 1 - len(ids))
            t = torch.tensor(ids, dtype=torch.long)
            return t[:-1], t[1:]

        p_x, p_y = make_block(p_ids, [])
        c_x, c_y = make_block(p_ids, c_ids)
        r_x, r_y = make_block(p_ids, r_ids)
        return p_x, p_y, c_x, c_y, r_x, r_y

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return self._encode_pair(s["prompt"], s["chosen"], s["rejected"])


def generate_dpo_pairs(sft_model, tokenizer, prompts: List[str],
                       n_samples_per_prompt: int = 4,
                       output_path: str = "data/dpo_pairs.jsonl",
                       device: str = "cpu",
                       temperature: float = 0.8,
                       max_new_tokens: int = 256,
                       rank_fn: Optional[Callable[[str], float]] = None):
    """
    Generate DPO preference pairs by sampling the SFT model and ranking outputs.

    Parameters
    ----------
    sft_model : nn.Module
        Trained Stage-2 model in eval mode.
    tokenizer : TokenizerWrapper
    prompts : list of str
        Diverse prompts covering the phone-assistant domain.
    n_samples_per_prompt : int
        How many completions to draw per prompt.
    rank_fn : callable or None
        Scoring function rank_fn(text) -> float (higher = better).
        If None, uses a simple heuristics-based fallback (length + formatting checks).
    """
    import json
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sft_model.eval()
    pairs = []

    if rank_fn is None:
        def rank_fn(text: str) -> float:
            score = 0.0
            if text.startswith("{") and "tool" in text:
                score += 5.0
            if len(text) > 20:
                score += min(len(text) / 100.0, 3.0)
            if "<|user|>" not in text and "<|system|>" not in text:
                score += 2.0
            return score

    for prompt in prompts:
        completions = []
        for _ in range(n_samples_per_prompt):
            ids = tokenizer.encode(format_dpo_prompt(DPO_SYSTEM_PROMPT, prompt))
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                # Greedy / sampling generation stub – replace with your generate() call
                out = sft_model(inp) if hasattr(sft_model, "forward") else sft_model(inp)
                # naive: just take argmax of logits for the next token repeatedly
                # in practice, wire this to your actual generate() function
                gen_ids = out.argmax(dim=-1)[0, len(ids):len(ids)+max_new_tokens].tolist()
            text = tokenizer.decode(gen_ids)
            completions.append((text, rank_fn(text)))

        completions.sort(key=lambda x: x[1], reverse=True)
        if len(completions) >= 2:
            pairs.append({
                "prompt": format_dpo_prompt(DPO_SYSTEM_PROMPT, prompt),
                "chosen": completions[0][0],
                "rejected": completions[-1][0],
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} DPO pairs to {output_path}")


# ---------------------------------------------------------------------------
#  Public factory
# ---------------------------------------------------------------------------

def get_stage_dataloader(stage: str,
                         tokenizer: TokenizerWrapper,
                         batch_size: int = 8,
                         context_len: int = 512,
                         num_workers: int = 0,
                         pin_memory: bool = False,
                         **kwargs) -> DataLoader:
    """
    Build a DataLoader for the requested training stage.

    stage : {"pretrain", "sft", "dpo"}
    tokenizer : TokenizerWrapper
    batch_size : int
    context_len : int
    num_workers : int
        >0 possible only if datasets is pickle-safe (streaming datasets often are not).
    pin_memory : bool
    kwargs : forwarded to dataset constructors (e.g., datasets=[...], weights={...},
             synthetic_jsonl=..., max_tokens=..., jsonl_path=...)

    Returns
    -------
    torch.utils.data.DataLoader
    """
    if stage == "pretrain":
        ds = PretrainDatasetMixer(tokenizer, context_len=context_len, **kwargs)
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=pin_memory)
    elif stage == "sft":
        ds = SFTDataset(tokenizer, context_len=context_len, **kwargs)
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=pin_memory)
    elif stage == "dpo":
        ds = DPODataset(tokenizer, context_len=context_len,
                        jsonl_path=kwargs.get("jsonl_path", "data/dpo_pairs.jsonl"))
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    else:
        raise ValueError(f"Unknown stage: {stage}")


# ---------------------------------------------------------------------------
#  CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    tok = TokenizerWrapper(context_len=256)
    if len(sys.argv) < 2:
        print("Usage: python data_pipeline.py <pretrain|sft|dpo>")
        sys.exit(1)
    stage = sys.argv[1]
    dl = get_stage_dataloader(stage, tok, batch_size=2, context_len=256,
                              max_tokens=1024 if stage == "pretrain" else None)
    for i, batch in enumerate(dl):
        if stage in ("pretrain", "sft"):
            x, y = batch
            print(f"Batch {i}: x.shape={tuple(x.shape)}  y.shape={tuple(y.shape)}")
        else:
            p_x, p_y, c_x, c_y, r_x, r_y = batch
            print(f"Batch {i}: prompt={tuple(p_x.shape)} chosen={tuple(c_x.shape)}")
        if i >= 2:
            break
