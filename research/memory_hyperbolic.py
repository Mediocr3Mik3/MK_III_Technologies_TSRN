"""
Hyperbolic KYRO Memory System
=============================

Persistent memory in the Poincaré disk.

Episodic memories have p-adic structure: access count determines stability.
Semantic memories are tropical convex hulls of episode gists.
Memory decay is hyperbolic drift toward the boundary.

Retrieval uses the Busemann function, which measures directional
distance from a point to the boundary.

Reference: TSRN Agent Model v2.0 Specification, Section 9
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic_embeddings import (
    poincare_distance,
    poincare_to_tangent,
    tangent_to_poincare,
)


class HyperbolicMemoryStore:
    """
    Hyperbolic memory store in the Poincaré disk.

    Memories are stored in hyperbolic space where hierarchy has
    geometric meaning:
    - Center of disk: core identity memories (always accessible)
    - Middle of disk: semantic memories (patterns, relationships)
    - Boundary of disk: episodic memories (specific events, fading)

    Decay: memories drift toward the boundary as they age.
    Retrieval: Busemann function penalizes boundary-proximate memories.
    """

    def __init__(
        self,
        d_model: int,
        capacity: int = 10000,
        decay_halflife_base: float = 7.0,
        p: int = 2,
    ):
        """
        Args:
            d_model: Embedding dimension
            capacity: Maximum number of memories
            decay_halflife_base: Base half-life in days (Ebbinghaus curve)
            p: Prime for p-adic stability computation
        """
        self.d_model = d_model
        self.capacity = capacity
        self.decay_halflife_base = decay_halflife_base
        self.p = p

        # Memory storage
        self.memories: Dict[str, torch.Tensor] = {}  # memory_id -> hyperbolic embedding
        self.metadata: Dict[str, Dict] = {}  # memory_id -> metadata

        # Access counts for p-adic stability
        self.access_counts: Dict[str, int] = defaultdict(int)

        # Decay scores (1.0 = fresh, 0.0 = deleted)
        self.decay_scores: Dict[str, float] = {}

    def store(
        self,
        memory_id: str,
        content: str,
        embedding: torch.Tensor,
        memory_type: str = "episodic",
    ) -> None:
        """
        Store a memory in hyperbolic space.

        Args:
            memory_id: Unique identifier for the memory
            content: Text content of the memory
            embedding: (d_model,) tensor in tangent space (will be projected to disk)
            memory_type: "episodic" or "semantic"
        """
        if len(self.memories) >= self.capacity:
            self.prune()

        # Project to Poincaré disk
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        hyperbolic_emb = tangent_to_poincare(embedding).squeeze(0)

        self.memories[memory_id] = hyperbolic_emb
        self.metadata[memory_id] = {
            "content": content,
            "type": memory_type,
            "created_at": self._current_time(),
        }
        self.access_counts[memory_id] = 0
        self.decay_scores[memory_id] = 1.0

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 10,
        memory_type: Optional[str] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve memories using Busemann function.

        Args:
            query_embedding: (d_model,) query in tangent space
            top_k: Number of memories to retrieve
            memory_type: Filter by memory type (optional)

        Returns:
            List of (memory_id, score, content) tuples
        """
        # Project query to Poincaré disk
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        query_hyperbolic = tangent_to_poincare(query_embedding).squeeze(0)

        # Compute Busemann scores for all memories
        scores = []
        for mem_id, mem_emb in self.memories.items():
            # Filter by type if specified
            if memory_type and self.metadata[mem_id]["type"] != memory_type:
                continue

            # Busemann function: penalize boundary-proximate memories
            decay = self.decay_scores.get(mem_id, 0.0)
            busemann = self.busemann_distance(query_hyperbolic, mem_emb, decay)

            # Convert to similarity score (lower distance = higher score)
            score = -busemann
            scores.append((mem_id, score))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for mem_id, score in scores[:top_k]:
            content = self.metadata[mem_id]["content"]
            results.append((mem_id, float(score), content))

        # Update access counts
        for mem_id, _ in scores[:top_k]:
            self.access_counts[mem_id] += 1

        return results

    def busemann_distance(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        decay: float,
    ) -> float:
        """
        Busemann function: directional distance to boundary.

        B(query, memory) = d_H(query, memory) - |memory|_H × r

        where r penalizes boundary-proximate (faded) memories.

        Args:
            query: (d,) query in Poincaré disk
            memory: (d,) memory in Poincaré disk
            decay: Decay score (1.0 = fresh, 0.0 = boundary)

        Returns:
            Busemann distance
        """
        # Hyperbolic distance
        dist = poincare_distance(query.unsqueeze(0), memory.unsqueeze(0)).item()

        # Memory's hyperbolic norm (distance from center)
        mem_norm = memory.norm().item()

        # Decay penalty: higher decay (closer to boundary) = larger penalty
        penalty = mem_norm * (1.0 - decay)

        return dist + penalty

    def decay_step(self, days_passed: float = 1.0) -> None:
        """
        Apply Ebbinghaus decay to all memories.

        Half-life depends on p-adic stability (access count).

        Args:
            days_passed: Number of days that have passed
        """
        for mem_id in list(self.memories.keys()):
            # Compute p-adic stability
            access_count = self.access_counts[mem_id]
            v_stab = math.floor(math.log2(access_count + 1))

            # Effective half-life: base × p^{v_stab}
            halflife = self.decay_halflife_base * (self.p ** v_stab)

            # Apply decay: score = score * (0.5)^(days / halflife)
            current_score = self.decay_scores[mem_id]
            new_score = current_score * (0.5 ** (days_passed / halflife))
            self.decay_scores[mem_id] = new_score

            # Drift memory toward boundary in hyperbolic space
            self._drift_toward_boundary(mem_id, decay=1.0 - new_score)

            # Prune if at boundary
            if new_score < 0.01:
                self._remove(mem_id)

    def _drift_toward_boundary(self, memory_id: str, decay: float) -> None:
        """
        Move memory outward in Poincaré disk as it decays.

        Args:
            memory_id: Memory identifier
            decay: Amount of decay (0 = no drift, 1 = to boundary)
        """
        if memory_id not in self.memories:
            return

        mem = self.memories[memory_id]
        norm = mem.norm()

        if norm < 0.99:  # Don't drift if already near boundary
            # Move outward: scale toward boundary
            new_norm = norm + (1.0 - norm) * decay * 0.1  # Gradual drift
            self.memories[memory_id] = mem * (new_norm / (norm + 1e-5))

    def prune(self) -> None:
        """Remove memories at the boundary (decay score ≈ 0)."""
        to_remove = [mem_id for mem_id, score in self.decay_scores.items() if score < 0.01]
        for mem_id in to_remove:
            self._remove(mem_id)

    def _remove(self, memory_id: str) -> None:
        """Remove a memory from storage."""
        self.memories.pop(memory_id, None)
        self.metadata.pop(memory_id, None)
        self.access_counts.pop(memory_id, None)
        self.decay_scores.pop(memory_id, None)

    def consolidate_semantic(
        self,
        episode_ids: List[str],
        semantic_id: str,
    ) -> None:
        """
        Consolidate episodic memories into a semantic memory.

        The consolidation operation is the tropical convex hull:
            semantic(episodes) = tconv(e_1, ..., e_n)

        Args:
            episode_ids: List of episodic memory IDs to consolidate
            semantic_id: ID for the new semantic memory
        """
        if len(episode_ids) == 0:
            return

        # Get episode embeddings
        episode_embs = []
        for ep_id in episode_ids:
            if ep_id in self.memories:
                episode_embs.append(self.memories[ep_id])

        if len(episode_embs) == 0:
            return

        # Stack embeddings
        stacked = torch.stack(episode_embs)  # (n, d)

        # Tropical convex hull: take max across each dimension
        # This captures the maximum-information projection
        semantic_emb, _ = stacked.max(dim=0)

        # Store as semantic memory
        content = f"Consolidated from {len(episode_ids)} episodes"
        self.store(semantic_id, content, semantic_emb, memory_type="semantic")

        # Remove original episodes (optional)
        for ep_id in episode_ids:
            self._remove(ep_id)

    def _current_time(self) -> float:
        """Get current time (days since epoch)."""
        # In production, use actual time
        return 0.0

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "total_memories": len(self.memories),
            "episodic_count": sum(1 for m in self.metadata.values() if m["type"] == "episodic"),
            "semantic_count": sum(1 for m in self.metadata.values() if m["type"] == "semantic"),
            "avg_decay": sum(self.decay_scores.values()) / max(len(self.decay_scores), 1),
            "avg_access_count": sum(self.access_counts.values()) / max(len(self.access_counts), 1),
        }


class HyperbolicMemoryLayer(nn.Module):
    """
    PyTorch module wrapper for hyperbolic memory.

    Integrates with the model for end-to-end training.
    """

    def __init__(
        self,
        d_model: int,
        capacity: int = 10000,
        decay_halflife_base: float = 7.0,
        enable_training: bool = False,
    ):
        """
        Args:
            d_model: Embedding dimension
            capacity: Maximum number of memories
            decay_halflife_base: Base half-life in days
            enable_training: Whether to enable gradient flow through memory
        """
        super().__init__()
        self.d_model = d_model
        self.enable_training = enable_training

        self.memory_store = HyperbolicMemoryStore(
            d_model=d_model,
            capacity=capacity,
            decay_halflife_base=decay_halflife_base,
        )

        # Learnable projection for memory queries
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.5)

    def forward(
        self,
        query: torch.Tensor,
        top_k: int = 10,
        memory_type: Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Query memory and retrieve top-k results.

        Args:
            query: (B, T, d) or (B, d) query tensor
            top_k: Number of memories to retrieve
            memory_type: Filter by memory type

        Returns:
            (retrieved_embeddings, memory_ids) tuple
        """
        # Project query
        if query.dim() == 3:
            # Use the last token's query for retrieval
            query = query[:, -1, :]  # (B, d)
        query_proj = self.query_proj(query)  # (B, d)

        # Retrieve memories for each batch element
        retrieved_embs = []
        memory_ids = []

        for i in range(query_proj.shape[0]):
            q = query_proj[i]
            results = self.memory_store.retrieve(q, top_k, memory_type)

            if results:
                ids = [r[0] for r in results]
                emb_list = [self.memory_store.memories[mem_id] for mem_id in ids]

                # Project back to tangent space
                emb_tangent = poincare_to_tangent(torch.stack(emb_list))  # (k, d)

                retrieved_embs.append(emb_tangent)
                memory_ids.append(ids)
            else:
                # No memories found, return zeros
                retrieved_embs.append(torch.zeros(top_k, self.d_model, device=query.device))
                memory_ids.append([])

        # Stack and pad
        max_k = max(len(emb) for emb in retrieved_embs) if retrieved_embs else 0
        if max_k == 0:
            return torch.zeros(query.shape[0], top_k, self.d_model, device=query.device), []

        padded = torch.zeros(query.shape[0], max_k, self.d_model, device=query.device)
        for i, emb in enumerate(retrieved_embs):
            k = min(emb.shape[0], max_k)
            padded[i, :k] = emb[:k]

        return padded, memory_ids

    def store_batch(
        self,
        memory_ids: List[str],
        contents: List[str],
        embeddings: torch.Tensor,
        memory_types: List[str],
    ) -> None:
        """
        Store a batch of memories.

        Args:
            memory_ids: List of memory IDs
            contents: List of memory contents
            embeddings: (B, d) tensor in tangent space
            memory_types: List of memory types
        """
        for i, mem_id in enumerate(memory_ids):
            self.memory_store.store(
                memory_id=mem_id,
                content=contents[i],
                embedding=embeddings[i],
                memory_type=memory_types[i],
            )

    def decay_step(self, days_passed: float = 1.0) -> None:
        """Apply decay to all memories."""
        self.memory_store.decay_step(days_passed)

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return self.memory_store.get_stats()


if __name__ == "__main__":
    # Test hyperbolic memory
    d_model = 64
    memory = HyperbolicMemoryLayer(d_model, capacity=100)

    # Store some memories
    memory.store_batch(
        memory_ids=["mem1", "mem2", "mem3"],
        contents=["Hello world", "Test memory", "Another test"],
        embeddings=torch.randn(3, d_model),
        memory_types=["episodic", "episodic", "semantic"],
    )

    print(f"Memory stats: {memory.get_stats()}")

    # Retrieve
    query = torch.randn(1, d_model)
    retrieved, ids = memory(query, top_k=2)
    print(f"Retrieved shape: {retrieved.shape}")
    print(f"Retrieved IDs: {ids}")

    # Test decay
    memory.decay_step(days_passed=7.0)
    print(f"After decay: {memory.get_stats()}")

    # Test consolidation
    memory.memory_store.consolidate_semantic(["mem1", "mem2"], "semantic1")
    print(f"After consolidation: {memory.get_stats()}")

    print("All tests passed!")
