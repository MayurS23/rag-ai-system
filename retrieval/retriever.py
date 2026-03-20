"""
retrieval/retriever.py
-----------------------
Retrieval pipeline: takes a user query → returns ranked, relevant chunks.

Responsibilities:
  1. Embed the user query (same model as indexing!)
  2. Search the vector store for top-K candidates
  3. Apply MMR (Maximal Marginal Relevance) to reduce redundancy
  4. Return ranked RetrievalResult objects with full provenance

MMR balances:
  - Relevance: How similar is the chunk to the query?
  - Diversity: Have we already retrieved something similar?
This prevents returning 5 chunks from the same paragraph.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ingestion.chunker import Chunk
from ingestion.embedding_generator import EmbeddingGenerator
from retrieval.vector_store import SearchResult, VectorStore, create_vector_store
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """
    Final output of the retrieval stage.
    Passed to the LLM engine as context.
    """
    chunks: List[SearchResult]
    query: str
    total_found: int
    retrieval_time_ms: float
    metadata: dict = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        """Concatenated text of all retrieved chunks, with source labels."""
        parts = []
        for r in self.chunks:
            source = r.chunk.source
            text = r.chunk.text.strip()
            parts.append(f"[Source: {source}]\n{text}")
        return "\n\n---\n\n".join(parts)

    @property
    def sources(self) -> List[str]:
        """Unique source documents referenced in retrieved chunks."""
        seen = set()
        sources = []
        for r in self.chunks:
            if r.chunk.source not in seen:
                seen.add(r.chunk.source)
                sources.append(r.chunk.source)
        return sources

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(chunks={len(self.chunks)}, "
            f"sources={len(self.sources)}, "
            f"time={self.retrieval_time_ms:.1f}ms)"
        )


class Retriever:
    """
    High-level retriever combining embedding lookup + MMR re-ranking.
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        embedding_generator: EmbeddingGenerator = None,
        top_k: int = None,
        mmr_lambda: float = 0.6,
    ):
        """
        Args:
            vector_store:        VectorStore instance (FAISS or Pinecone)
            embedding_generator: EmbeddingGenerator instance
            top_k:               Number of results to return
            mmr_lambda:          MMR trade-off (0=max diversity, 1=max relevance)
        """
        self.store = vector_store or create_vector_store()
        self.embedder = embedding_generator or EmbeddingGenerator()
        self.top_k = top_k or settings.TOP_K_RESULTS
        self.mmr_lambda = mmr_lambda

    async def initialize(self) -> None:
        """Load the vector store from disk/cloud. Call once at startup."""
        await self.store.load()
        logger.info("retriever_initialized")

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[Dict] = None,
        use_mmr: bool = True,
    ) -> RetrievalResult:
        """
        Main retrieval method.

        Args:
            query:           Natural language question
            top_k:           Override default top_k
            filter_metadata: Optional exact-match metadata filters
            use_mmr:         Apply Maximal Marginal Relevance reranking

        Returns:
            RetrievalResult with ranked chunks and metadata
        """
        k = top_k or self.top_k
        start = time.perf_counter()

        # 1. Embed the query
        query_embedding = await self.embedder.embed_query(query)

        # 2. Fetch candidates (overfetch for MMR)
        candidate_k = k * 3 if use_mmr else k
        candidates = await self.store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            filter_metadata=filter_metadata,
        )

        # 3. MMR re-ranking
        if use_mmr and len(candidates) > k:
            final_results = self._apply_mmr(
                query_embedding=query_embedding,
                candidates=candidates,
                top_k=k,
            )
        else:
            final_results = candidates[:k]

        # Re-assign ranks after MMR reordering
        for i, result in enumerate(final_results):
            result.rank = i + 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = RetrievalResult(
            chunks=final_results,
            query=query,
            total_found=len(candidates),
            retrieval_time_ms=round(elapsed_ms, 2),
            metadata={
                "top_k_requested": k,
                "mmr_applied": use_mmr,
                "mmr_lambda": self.mmr_lambda,
                "filter_metadata": filter_metadata,
            },
        )

        logger.info(
            "retrieval_complete",
            query=query[:80],
            candidates=len(candidates),
            returned=len(final_results),
            elapsed_ms=result.retrieval_time_ms,
        )

        return result

    def _apply_mmr(
        self,
        query_embedding: np.ndarray,
        candidates: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Maximal Marginal Relevance (MMR) selection.

        Algorithm:
        1. Start with the highest-relevance candidate
        2. At each step, select the candidate that maximizes:
           MMR = λ * relevance(doc, query) - (1-λ) * max_similarity(doc, selected)
        3. This balances relevance AND diversity

        Reference: Carbonell & Goldstein, 1998
        """
        if not candidates:
            return []

        # Extract embeddings (proxy: use score as 1D representation since
        # we don't re-store embeddings in SearchResult for memory efficiency;
        # for full MMR, you'd store embeddings in the vector store metadata)
        # Here we use score-based MMR as an approximation
        selected: List[SearchResult] = []
        remaining = list(candidates)

        # Always start with best candidate
        selected.append(remaining.pop(0))

        while remaining and len(selected) < top_k:
            best_score = -float("inf")
            best_idx = 0

            for i, candidate in enumerate(remaining):
                # Relevance term
                relevance = candidate.score

                # Redundancy term: max similarity with already-selected chunks
                # Using text overlap as a proxy for embedding similarity
                max_redundancy = max(
                    self._text_overlap(candidate.chunk.text, sel.chunk.text)
                    for sel in selected
                )

                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1 - self.mmr_lambda) * max_redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    @staticmethod
    def _text_overlap(text_a: str, text_b: str) -> float:
        """
        Compute Jaccard similarity between two texts (word-level).
        Used as a lightweight proxy for semantic similarity in MMR.
        Range: [0.0, 1.0]
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
