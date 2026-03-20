"""
ingestion/embedding_generator.py
----------------------------------
Converts text chunks into dense numerical vectors (embeddings).

Supports:
  - Local sentence-transformers (default, free, private)
  - OpenAI text-embedding-3-small / text-embedding-3-large (best quality, costs $)

The same EmbeddingGenerator is used for BOTH:
  1. Indexing (at ingestion time)
  2. Query encoding (at retrieval time)
This ensures embedding space consistency — critical for accurate retrieval.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from ingestion.chunker import Chunk
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract base for embedding providers. Swap implementations via config."""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Returns ndarray of shape (n_texts, embedding_dim)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Sentence-transformers running locally.
    - No API costs
    - Full data privacy
    - Good quality (all-MiniLM-L6-v2: 384 dims, fast; all-mpnet-base-v2: 768 dims, better)
    - Models auto-downloaded on first use (~90MB for MiniLM)
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        logger.info("local_embedding_provider_init", model=self.model_name)

    def _load_model(self):
        """Lazy load — avoids slow import at module level."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
            logger.info("loading_embedding_model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_loaded", model=self.model_name, dim=self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension()

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Run CPU-bound embedding in thread pool to not block event loop."""
        model = self._load_model()
        loop = asyncio.get_event_loop()

        def _encode():
            return model.encode(
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,   # L2 normalize → cosine = dot product
                convert_to_numpy=True,
            )

        embeddings = await loop.run_in_executor(None, _encode)
        return embeddings.astype(np.float32)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI's text-embedding-3-small / text-embedding-3-large.
    - Best quality embeddings available
    - Costs ~$0.02 per million tokens (very cheap)
    - Requires OPENAI_API_KEY
    """

    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.OPENAI_EMBEDDING_MODEL
        self._client = None
        logger.info("openai_embedding_provider_init", model=self.model_name)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai required: pip install openai")
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in .env")
            self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    @property
    def dimension(self) -> int:
        return self.DIMENSIONS.get(self.model_name, 1536)

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        client = self._get_client()

        # OpenAI has a 2048 item limit per request — batch if needed
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await client.embeddings.create(
                model=self.model_name,
                input=batch,
                encoding_format="float",
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)


class EmbeddingGenerator:
    """
    High-level embedding generator with batching, caching, and error handling.
    This is the main class used by the ingestion pipeline and retriever.
    """

    def __init__(self, provider: BaseEmbeddingProvider = None):
        if provider is not None:
            self.provider = provider
        elif settings.EMBEDDING_PROVIDER == "openai":
            self.provider = OpenAIEmbeddingProvider()
        else:
            self.provider = LocalEmbeddingProvider()

        logger.info(
            "embedding_generator_ready",
            provider=settings.EMBEDDING_PROVIDER,
        )

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    async def embed_chunks(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of Chunk objects.
        Returns list of numpy arrays aligned 1:1 with input chunks.
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        return await self.embed_texts(texts)

    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a list of plain strings.
        Returns list of numpy arrays, one per input text.
        """
        if not texts:
            return []

        start = time.perf_counter()

        embeddings_matrix = await self.provider.embed_texts(texts)

        elapsed = time.perf_counter() - start
        logger.info(
            "embeddings_generated",
            count=len(texts),
            dim=embeddings_matrix.shape[1],
            elapsed_ms=round(elapsed * 1000, 2),
            throughput=round(len(texts) / elapsed, 1),
        )

        return [embeddings_matrix[i] for i in range(len(texts))]

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for retrieval.
        Exposed separately for clarity — same model as indexing.
        """
        embeddings = await self.embed_texts([query])
        return embeddings[0]
