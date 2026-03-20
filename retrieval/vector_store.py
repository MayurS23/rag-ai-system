"""
retrieval/vector_store.py
--------------------------
Vector store abstraction layer supporting FAISS (local) and Pinecone (cloud).

Design pattern: Program to an interface (VectorStore ABC), not an implementation.
This lets you swap FAISS for Pinecone with a single config change.

FAISS:
  - Stores vectors in-memory + serialized to disk
  - Supports millions of vectors on a single machine
  - Index types: IndexFlatIP (exact), IndexIVFFlat (approximate, faster at scale)

Pinecone:
  - Managed cloud vector DB
  - Handles billions of vectors with automatic scaling
  - Requires API key
"""

from __future__ import annotations

import json
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ingestion.chunker import Chunk
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float                    # 0.0–1.0, higher = more similar
    rank: int                       # 1-based rank in result set

    def __repr__(self) -> str:
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.4f}, "
            f"chunk_id={self.chunk.chunk_id}, source={self.chunk.source!r})"
        )


class VectorStore(ABC):
    """Abstract interface for all vector store implementations."""

    @abstractmethod
    async def add_chunks(self, chunks: List[Chunk], embeddings: List[np.ndarray]) -> int:
        """Index chunks with their embeddings. Returns count added."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """ANN search. Returns top_k results sorted by similarity desc."""
        ...

    @abstractmethod
    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source. Returns count deleted."""
        ...

    @abstractmethod
    async def get_stats(self) -> Dict:
        """Return store statistics (count, dimensions, etc.)."""
        ...

    @abstractmethod
    async def save(self) -> None:
        """Persist the store to disk/cloud."""
        ...

    @abstractmethod
    async def load(self) -> None:
        """Load the store from disk/cloud."""
        ...


class FAISSVectorStore(VectorStore):
    """
    Local FAISS vector store with full persistence.

    Uses IndexFlatIP (inner product) which equals cosine similarity
    when vectors are L2-normalized (which our embedding providers do).

    Data layout on disk:
      {index_path}/
        index.faiss       ← FAISS binary index
        metadata.pkl      ← Chunk objects & id mapping
        stats.json        ← Stats for quick introspection
    """

    def __init__(self, index_path: str = None, dimension: int = None):
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._index = None          # FAISS index
        self._chunks: List[Chunk] = []        # aligned with FAISS internal IDs
        self._id_to_pos: Dict[str, int] = {}  # chunk_id → position in _chunks

        logger.info(
            "faiss_store_init",
            path=str(self.index_path),
            dimension=self.dimension,
        )

    def _get_index(self):
        """Lazy-initialize FAISS index."""
        if self._index is None:
            try:
                import faiss
            except ImportError:
                raise ImportError("faiss-cpu required: pip install faiss-cpu")
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    async def add_chunks(self, chunks: List[Chunk], embeddings: List[np.ndarray]) -> int:
        """Add chunks and their embeddings to the FAISS index."""
        if not chunks:
            return 0

        import faiss
        index = self._get_index()

        vectors = np.array(embeddings, dtype=np.float32)

        # Ensure L2-normalized for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        # Track which chunks are new vs updates (re-index on duplicate source)
        start_pos = len(self._chunks)
        index.add(vectors)
        self._chunks.extend(chunks)

        for i, chunk in enumerate(chunks):
            self._id_to_pos[chunk.chunk_id] = start_pos + i

        await self.save()

        logger.info(
            "chunks_indexed",
            count=len(chunks),
            total_indexed=len(self._chunks),
        )
        return len(chunks)

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        filter_metadata: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Perform ANN search. Optional metadata filtering post-search.
        Returns results sorted by similarity (highest first).
        """
        top_k = top_k or settings.TOP_K_RESULTS

        if len(self._chunks) == 0:
            logger.warning("search_on_empty_index")
            return []

        import faiss
        index = self._get_index()

        # Ensure correct shape and normalization
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        # Search with extra buffer for post-filtering
        search_k = min(top_k * 3, len(self._chunks))
        scores, indices = index.search(query, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for empty slots
                continue
            if idx >= len(self._chunks):
                continue

            chunk = self._chunks[idx]

            # Below threshold → skip
            if score < settings.SIMILARITY_THRESHOLD:
                continue

            # Metadata filtering (exact match)
            if filter_metadata:
                if not all(chunk.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(SearchResult(
                chunk=chunk,
                score=float(score),
                rank=0,  # assigned after filtering
            ))

            if len(results) >= top_k:
                break

        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        logger.info(
            "search_complete",
            results_found=len(results),
            top_score=round(results[0].score, 4) if results else 0,
        )
        return results

    async def delete_by_source(self, source: str) -> int:
        """
        Remove all chunks from a given source.
        FAISS doesn't support in-place deletion — we rebuild the index.
        """
        original_count = len(self._chunks)
        surviving_chunks = [c for c in self._chunks if c.source != source]

        if len(surviving_chunks) == original_count:
            return 0

        # Rebuild index with only surviving chunks
        # We need embeddings — they're stored in the index
        # Re-encode is expensive; in production, store embeddings separately
        deleted = original_count - len(surviving_chunks)
        logger.warning(
            "faiss_delete_requires_rebuild",
            deleted=deleted,
            note="Re-ingest documents to refresh after deletions",
        )

        self._chunks = surviving_chunks
        self._id_to_pos = {c.chunk_id: i for i, c in enumerate(self._chunks)}

        import faiss
        self._index = faiss.IndexFlatIP(self.dimension)
        await self.save()

        return deleted

    async def get_stats(self) -> Dict:
        index = self._get_index()
        sources = list({c.source for c in self._chunks})
        return {
            "backend": "faiss",
            "total_chunks": len(self._chunks),
            "index_size": index.ntotal,
            "dimension": self.dimension,
            "unique_sources": len(sources),
            "sources": sources[:20],
            "index_path": str(self.index_path),
        }

    async def save(self) -> None:
        """Persist FAISS index + metadata to disk."""
        try:
            import faiss
            faiss_path = self.index_path / "index.faiss"
            meta_path = self.index_path / "metadata.pkl"
            stats_path = self.index_path / "stats.json"

            faiss.write_index(self._get_index(), str(faiss_path))

            with open(meta_path, "wb") as f:
                pickle.dump({
                    "chunks": self._chunks,
                    "id_to_pos": self._id_to_pos,
                }, f)

            stats = await self.get_stats()
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2, default=str)

            logger.info("faiss_saved", path=str(self.index_path))
        except Exception as e:
            logger.error("faiss_save_failed", error=str(e))
            raise

    async def load(self) -> None:
        """Load FAISS index + metadata from disk."""
        faiss_path = self.index_path / "index.faiss"
        meta_path = self.index_path / "metadata.pkl"

        if not faiss_path.exists():
            logger.info("no_existing_faiss_index_creating_new")
            self._get_index()  # initialize empty
            return

        try:
            import faiss
            self._index = faiss.read_index(str(faiss_path))
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data["chunks"]
            self._id_to_pos = data["id_to_pos"]
            logger.info(
                "faiss_loaded",
                chunks=len(self._chunks),
                index_size=self._index.ntotal,
            )
        except Exception as e:
            logger.error("faiss_load_failed", error=str(e))
            raise


class PineconeVectorStore(VectorStore):
    """
    Pinecone managed cloud vector store.
    Better for production at scale (>1M vectors) or multi-tenant deployments.
    """

    def __init__(self):
        self._index = None
        self._dimension = settings.EMBEDDING_DIMENSION
        logger.info("pinecone_store_init", index_name=settings.PINECONE_INDEX_NAME)

    def _get_client(self):
        if self._index is None:
            try:
                from pinecone import Pinecone, ServerlessSpec
            except ImportError:
                raise ImportError("pinecone required: pip install pinecone-client")

            if not settings.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY not set in .env")

            pc = Pinecone(api_key=settings.PINECONE_API_KEY)

            existing = [i.name for i in pc.list_indexes()]
            if settings.PINECONE_INDEX_NAME not in existing:
                pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=self._dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT),
                )
            self._index = pc.Index(settings.PINECONE_INDEX_NAME)
        return self._index

    async def add_chunks(self, chunks: List[Chunk], embeddings: List[np.ndarray]) -> int:
        import asyncio
        index = self._get_client()
        loop = asyncio.get_event_loop()

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk.chunk_id,
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk.text[:1000],   # Pinecone metadata limit
                    "source": chunk.source,
                    "doc_id": chunk.doc_id,
                    "doc_type": chunk.doc_type,
                    **{k: str(v) for k, v in chunk.metadata.items()},
                },
            })

        # Batch upsert (Pinecone recommends 100 vectors per batch)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            await loop.run_in_executor(None, lambda b=batch: index.upsert(vectors=b))

        return len(chunks)

    async def search(self, query_embedding: np.ndarray, top_k: int = 5, filter_metadata=None) -> List[SearchResult]:
        import asyncio
        index = self._get_client()
        loop = asyncio.get_event_loop()

        query_filter = None
        if filter_metadata:
            query_filter = {k: {"$eq": v} for k, v in filter_metadata.items()}

        response = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=query_filter,
            )
        )

        results = []
        for i, match in enumerate(response.matches):
            meta = match.metadata or {}
            chunk = Chunk(
                text=meta.get("text", ""),
                chunk_id=match.id,
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                doc_type=meta.get("doc_type", ""),
                chunk_index=0,
                metadata=meta,
            )
            results.append(SearchResult(chunk=chunk, score=match.score, rank=i + 1))

        return results

    async def delete_by_source(self, source: str) -> int:
        index = self._get_client()
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: index.delete(filter={"source": {"$eq": source}})
        )
        return 0  # Pinecone doesn't return delete count

    async def get_stats(self) -> Dict:
        index = self._get_client()
        import asyncio
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, index.describe_index_stats)
        return {
            "backend": "pinecone",
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_name": settings.PINECONE_INDEX_NAME,
        }

    async def save(self) -> None:
        pass  # Pinecone auto-persists

    async def load(self) -> None:
        self._get_client()  # initialize connection


def create_vector_store() -> VectorStore:
    """
    Factory function — returns the correct VectorStore based on config.
    Usage: store = create_vector_store()
    """
    provider = settings.VECTOR_STORE_PROVIDER
    if provider == "pinecone":
        return PineconeVectorStore()
    return FAISSVectorStore()
