"""
Tropical Merging Tokenization (TMT)
===================================

Tokenization native to TSRN's tropical geometry. Merges pairs based on
tropical synergy score (max-plus inner product coherence) rather than
frequency. This produces a vocabulary that is algebraically coherent
in the model's max-plus space.

Algorithm:
  Phase 0: Initialize with byte-level vocabulary (205 tokens)
          Train TSRN for 5,000 steps to get initial tropical embeddings
  Phase 1: Compute tropical synergy score for all adjacent pairs
  Phase 2: Merge top-k pairs, update embeddings
  Phase 3: Retrain TSRN for 2,000 steps
  Repeat Phases 1-3 for N rounds until vocabulary size = 32,000

Reference: TSRN Agent Model v2.0 Specification, Section 2.2
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class TropicalMergingTokenizer:
    """
    Tropical Merging Tokenizer.

    Merges token pairs based on tropical synergy score:
        score_trop(a, b) = max_c(e(a)_c + e(b)_c) - max_c(e(a)_c) - max_c(e(b)_c) + λ × log_freq(a, b)

    This measures how much additional information the tropical inner product
    of the merged token captures beyond the sum of its parts.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        p: int = 2,
        lambda_freq: float = 0.01,
        theta: float = 0.1,
        base_vocab: Optional[List[str]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Args:
            vocab_size: Target vocabulary size (default 32,000)
            p: Prime for p-adic structure (default 2)
            lambda_freq: Frequency regularizer weight (small, makes frequency a weak tiebreaker)
            theta: Synergy threshold for merging
            base_vocab: Optional base vocabulary (default: bytes 0-204)
            special_tokens: Multi-character atomic tokens (e.g. "<|level_0|>")
                that must never be split into characters during tokenization.
                The 4 defaults (<pad>/<unk>/<eos>/<bos>) are always treated as special.
        """
        self.target_vocab_size = vocab_size
        self.p = p
        self.lambda_freq = lambda_freq
        self.theta = theta

        # Initialize with byte-level vocabulary
        if base_vocab is None:
            self.base_vocab = [chr(i) for i in range(256)]  # bytes 0-255
            # Add special tokens
            self.base_vocab = ["<pad>", "<unk>", "<eos>", "<bos>"] + self.base_vocab
        else:
            self.base_vocab = list(base_vocab)

        # Special (atomic) tokens — never character-split. Always include the
        # 4 control defaults; longest-first so overlapping markers match greedily.
        specials = ["<pad>", "<unk>", "<eos>", "<bos>"] + list(special_tokens or [])
        seen: set = set()
        self.special_tokens: List[str] = []
        for t in specials:
            if t not in seen:
                seen.add(t)
                self.special_tokens.append(t)
        # Ensure every special token is present in the base vocabulary.
        for t in self.special_tokens:
            if t not in self.base_vocab:
                self.base_vocab.append(t)

        # Current vocabulary (starts as base vocab, grows with merges)
        self.vocab: List[str] = list(self.base_vocab)
        self.vocab_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}

        # Merge rules: (a, b) -> merged_token
        self.merges: List[Tuple[str, str, str]] = []

        # Token embeddings (will be loaded from model)
        self.embeddings: Optional[torch.Tensor] = None

        # Pair frequencies (for lambda term)
        self.pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    @property
    def current_vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        """Alias for current_vocab_size — what trainers expect when querying the live
        vocabulary size. Use ``target_vocab_size`` for the training target."""
        return len(self.vocab)

    def compute_tropical_synergy(
        self,
        embeddings: torch.Tensor,
        pairs: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute tropical synergy score for all adjacent pairs.

        Args:
            embeddings: (V, d) tensor of token embeddings
            pairs: List of (token_a, token_b) pairs to score

        Returns:
            Dict mapping (a, b) -> synergy score
        """
        device = embeddings.device
        scores = {}

        for a, b in tqdm(pairs, desc="Computing tropical synergy"):
            if a not in self.vocab_to_id or b not in self.vocab_to_id:
                continue

            id_a = self.vocab_to_id[a]
            id_b = self.vocab_to_id[b]

            e_a = embeddings[id_a]  # (d,)
            e_b = embeddings[id_b]  # (d,)

            # Tropical inner product: max_c(e(a)_c + e(b)_c)
            tropical_inner = torch.max(e_a + e_b).item()

            # Individual tropical norms: max_c(e(a)_c), max_c(e(b)_c)
            norm_a = torch.max(e_a).item()
            norm_b = torch.max(e_b).item()

            # Base synergy
            synergy = tropical_inner - norm_a - norm_b

            # Frequency regularizer (weak tiebreaker)
            freq = self.pair_counts.get((a, b), 0)
            freq_term = self.lambda_freq * math.log(freq + 1)

            scores[(a, b)] = synergy + freq_term

        return scores

    def merge_round(
        self,
        embeddings: torch.Tensor,
        k_pairs: int = 2,
    ) -> List[Tuple[str, str, str]]:
        """
        Merge top-k pairs by tropical synergy score.

        Args:
            embeddings: Current token embeddings
            k_pairs: Number of pairs to merge in this round

        Returns:
            List of (a, b, merged_token) for this round
        """
        # Collect all adjacent pairs from corpus (or sample)
        # For efficiency, sample from pair_counts
        pairs = list(self.pair_counts.keys())

        if len(pairs) == 0:
            return []

        # Compute synergy scores
        scores = self.compute_tropical_synergy(embeddings, pairs)

        # Sort by score, take top-k
        sorted_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_pairs = sorted_pairs[:k_pairs]

        round_merges = []
        for (a, b), score in top_pairs:
            if score < self.theta:
                continue  # Skip if below threshold

            # Create merged token
            merged_token = a + b

            # Check if already exists
            if merged_token in self.vocab_to_id:
                continue

            # Add to vocabulary
            merged_id = len(self.vocab)
            self.vocab.append(merged_token)
            self.vocab_to_id[merged_token] = merged_id

            # Initialize embedding as tropical mean
            if a in self.vocab_to_id and b in self.vocab_to_id:
                id_a = self.vocab_to_id[a]
                id_b = self.vocab_to_id[b]
                e_merged = (embeddings[id_a] + embeddings[id_b]) / 2
                # Extend embeddings tensor
                embeddings = torch.cat([embeddings, e_merged.unsqueeze(0)], dim=0)

            # Record merge
            self.merges.append((a, b, merged_token))
            round_merges.append((a, b, merged_token))

        self.embeddings = embeddings
        return round_merges

    def update_pair_counts(self, corpus: List[List[str]]) -> None:
        """
        Update pair frequency counts from corpus.

        Args:
            corpus: List of tokenized sequences
        """
        self.pair_counts.clear()
        for seq in tqdm(corpus, desc="Counting pairs"):
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                self.pair_counts[(a, b)] += 1

    def _split_specials(self, text: str) -> List[str]:
        """Split text into a list of segments where special tokens are kept
        atomic and everything else is a raw substring to be char-tokenized.
        Longest specials are matched first to avoid prefix collisions."""
        if not self.special_tokens:
            return [text]
        specials = sorted(self.special_tokens, key=len, reverse=True)
        segments: List[str] = [text]
        for sp in specials:
            new_segments: List[str] = []
            for seg in segments:
                if seg in self.special_tokens:
                    new_segments.append(seg)
                    continue
                parts = seg.split(sp)
                for i, part in enumerate(parts):
                    if part:
                        new_segments.append(part)
                    if i < len(parts) - 1:
                        new_segments.append(sp)
            segments = new_segments
        return segments

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using current vocabulary and merge rules. Special tokens
        (e.g. "<|level_0|>") are kept atomic and never character-split.

        Args:
            text: Input string

        Returns:
            List of tokens
        """
        out: List[str] = []
        for seg in self._split_specials(text):
            if seg in self.special_tokens:
                out.append(seg)
                continue
            # Start with bytes
            tokens = list(seg)

            # Apply merges greedily
            changed = True
            while changed:
                changed = False
                for a, b, merged in self.merges:
                    # Find and replace all occurrences
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i] == a and tokens[i + 1] == b:
                            tokens[i:i + 2] = [merged]
                            changed = True
                        i += 1
            out.extend(tokens)
        return out

    def encode(self, text: str) -> List[int]:
        """Tokenize and convert to IDs."""
        tokens = self.tokenize(text)
        ids = [self.vocab_to_id.get(t, self.vocab_to_id["<unk>"]) for t in tokens]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert IDs back to text."""
        tokens = [self.vocab[i] if i < len(self.vocab) else "<unk>" for i in ids]
        return "".join(tokens)

    def save(self, path: str) -> None:
        """Save tokenizer state to JSON."""
        state = {
            "target_vocab_size": self.target_vocab_size,
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "pair_counts": {json.dumps([a, b]): count for (a, b), count in self.pair_counts.items()},
        }

        # Only create the parent dir if there is one (path may be a bare filename).
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        # Save embeddings if available
        if self.embeddings is not None:
            emb_path = path.replace(".json", "_embeddings.pt")
            torch.save(self.embeddings, emb_path)

    @classmethod
    def load(cls, path: str) -> "TropicalMergingTokenizer":
        """Load tokenizer state from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        # Backwards-compat: older saves wrote `vocab_size` (which was the target).
        target_size = state.get("target_vocab_size", state.get("vocab_size", 32000))
        tokenizer = cls(
            vocab_size=target_size,
            base_vocab=state["vocab"][:256],  # Reconstruct base
        )
        tokenizer.vocab = state["vocab"]
        tokenizer.vocab_to_id = {tok: i for i, tok in enumerate(tokenizer.vocab)}
        tokenizer.merges = state["merges"]
        if state.get("special_tokens"):
            tokenizer.special_tokens = list(state["special_tokens"])

        # Reconstruct pair_counts
        tokenizer.pair_counts = defaultdict(int)
        for key, count in state["pair_counts"].items():
            # Try JSON format first (new format)
            try:
                a, b = json.loads(key)
            except json.JSONDecodeError:
                # Fallback to old "|" delimiter format
                parts = key.split("|")
                if len(parts) == 2:
                    a, b = parts
                else:
                    # If still not 2 parts, skip this entry
                    continue
            tokenizer.pair_counts[(a, b)] = count

        # Load embeddings if available
        emb_path = path.replace(".json", "_embeddings.pt")
        if os.path.exists(emb_path):
            tokenizer.embeddings = torch.load(emb_path)

        return tokenizer


def bootstrap_from_bytes(
    corpus_path: str,
    output_dir: str = "tokenizers",
    model_steps: int = 5000,
    n_rounds: int = 14,
    k_pairs: int = 2,
    vocab_size: int = 32000,
    special_tokens: Optional[List[str]] = None,
) -> TropicalMergingTokenizer:
    """
    Bootstrap TMT from byte-level vocabulary.

    Phase 0: Train TSRN for model_steps to get initial embeddings
    Phases 1-3: Run n_rounds of TMT training

    Args:
        corpus_path: Path to training corpus
        output_dir: Directory to save tokenizer
        model_steps: Steps to train model in Phase 0
        n_rounds: Number of TMT rounds
        k_pairs: Pairs to merge per round
        vocab_size: Target vocabulary size (default 32,000)
        special_tokens: Optional list of special tokens to add

    Returns:
        Trained TropicalMergingTokenizer
    """
    print(f"Phase 0: Bootstrapping from byte-level vocabulary")

    # Initialize tokenizer with byte vocab
    tokenizer = TropicalMergingTokenizer(vocab_size=vocab_size)

    # TODO: Load sample corpus and count pairs
    # For now, use synthetic data
    print("  Loading corpus...")
    corpus = load_sample_corpus(corpus_path, n_samples=100000)
    tokenizer.update_pair_counts(corpus)

    # TODO: Train initial TSRN model to get embeddings
    # For now, use random embeddings
    print(f"  Training initial model for {model_steps} steps...")
    d_model = 512
    embeddings = torch.randn(len(tokenizer.vocab), d_model) * 0.1
    tokenizer.embeddings = embeddings

    print(f"  Starting TMT training ({n_rounds} rounds)...")
    for round_idx in range(n_rounds):
        print(f"\nRound {round_idx + 1}/{n_rounds}")
        print(f"  Current vocab size: {tokenizer.current_vocab_size}")

        # Phase 1: Compute scores (done in merge_round)
        # Phase 2: Merge top-k pairs
        merges = tokenizer.merge_round(embeddings, k_pairs=k_pairs)
        print(f"  Merged {len(merges)} pairs")

        if tokenizer.current_vocab_size >= tokenizer.vocab_size:
            print(f"  Target vocab size reached!")
            break

        # Phase 3: Retrain model to update embeddings
        # TODO: Actually retrain model
        # For now, add small noise to simulate retraining
        embeddings = embeddings + torch.randn_like(embeddings) * 0.01
        tokenizer.embeddings = embeddings

        print(f"  Updated embeddings")

    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "tmt_32k.json")
    tokenizer.save(save_path)
    print(f"\nTokenizer saved to {save_path}")

    return tokenizer


def load_sample_corpus(corpus_path: str, n_samples: int = 100000) -> List[List[str]]:
    """
    Load sample corpus for TMT training.

    For now, returns synthetic data. In production, load from actual corpus.
    """
    # TODO: Implement actual corpus loading
    # For now, return synthetic byte sequences
    corpus = []
    for _ in range(n_samples):
        # Generate random byte sequence
        seq = [chr(i) for i in range(256)] * 10
        corpus.append(seq)
    return corpus


# Aliases / backwards compatibility -------------------------------------------
# Several call sites import this class as ``TropicalTokenizer``.
TropicalTokenizer = TropicalMergingTokenizer


if __name__ == "__main__":
    # Test basic functionality
    tokenizer = TropicalMergingTokenizer(vocab_size=1000)  # Small vocab for testing

    # Test tokenization
    text = "Hello world"
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")

    # Test encode/decode
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")

    # Test save/load
    tokenizer.save("test_tokenizer.json")
    loaded = TropicalMergingTokenizer.load("test_tokenizer.json")
    print(f"Loaded vocab size: {loaded.current_vocab_size}")
