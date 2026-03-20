"""
tests/test_retrieval.py
Tests for the vector store and retrieval pipeline.
"""
import asyncio
import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
async def populated_store():
    """Create a FAISS store populated with test data."""
    from retrieval.vector_store import FAISSVectorStore
    from ingestion.chunker import Chunk

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FAISSVectorStore(index_path=tmpdir, dimension=4)

        chunks = [
            Chunk(text="Machine learning is a subset of AI", chunk_id="c1",
                  doc_id="d1", source="ml_guide.pdf", doc_type="pdf", chunk_index=0),
            Chunk(text="Neural networks are inspired by the brain", chunk_id="c2",
                  doc_id="d1", source="ml_guide.pdf", doc_type="pdf", chunk_index=1),
            Chunk(text="Python is a programming language", chunk_id="c3",
                  doc_id="d2", source="python_intro.md", doc_type="markdown", chunk_index=0),
            Chunk(text="FastAPI is a modern web framework for Python", chunk_id="c4",
                  doc_id="d2", source="python_intro.md", doc_type="markdown", chunk_index=1),
        ]

        # Small 4-dim embeddings for testing
        np.random.seed(42)
        embeddings = [np.random.randn(4).astype(np.float32) for _ in chunks]

        await store.add_chunks(chunks, embeddings)
        yield store, chunks, embeddings


@pytest.mark.asyncio
async def test_faiss_store_add_and_search():
    from retrieval.vector_store import FAISSVectorStore
    from ingestion.chunker import Chunk

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FAISSVectorStore(index_path=tmpdir, dimension=4)
        chunk = Chunk(
            text="Test chunk content",
            chunk_id="test_c1",
            doc_id="test_d1",
            source="test.txt",
            doc_type="text",
            chunk_index=0,
        )
        embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        count = await store.add_chunks([chunk], [embedding])
        assert count == 1

        results = await store.search(embedding, top_k=1)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "test_c1"


@pytest.mark.asyncio
async def test_faiss_store_persistence():
    """Index should survive save/load cycle."""
    from retrieval.vector_store import FAISSVectorStore
    from ingestion.chunker import Chunk

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and populate
        store1 = FAISSVectorStore(index_path=tmpdir, dimension=4)
        chunk = Chunk(text="Persistent chunk", chunk_id="p1", doc_id="pd1",
                      source="persist.txt", doc_type="text", chunk_index=0)
        embedding = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        await store1.add_chunks([chunk], [embedding])

        # Reload in a new store instance
        store2 = FAISSVectorStore(index_path=tmpdir, dimension=4)
        await store2.load()

        stats = await store2.get_stats()
        assert stats["total_chunks"] == 1


@pytest.mark.asyncio
async def test_faiss_search_top_k():
    from retrieval.vector_store import FAISSVectorStore
    from ingestion.chunker import Chunk

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FAISSVectorStore(index_path=tmpdir, dimension=2)
        chunks = [
            Chunk(text=f"Chunk {i}", chunk_id=f"c{i}", doc_id=f"d{i}",
                  source="test.txt", doc_type="text", chunk_index=i)
            for i in range(10)
        ]
        embeddings = [np.random.randn(2).astype(np.float32) for _ in chunks]
        await store.add_chunks(chunks, embeddings)

        query = embeddings[0]
        results = await store.search(query, top_k=3)
        assert len(results) <= 3


@pytest.mark.asyncio
async def test_search_empty_store():
    from retrieval.vector_store import FAISSVectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FAISSVectorStore(index_path=tmpdir, dimension=4)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = await store.search(query, top_k=5)
        assert results == []


@pytest.mark.asyncio
async def test_retriever_full_pipeline():
    """Integration test: embed a query and retrieve from populated store."""
    from retrieval.retriever import Retriever
    from retrieval.vector_store import FAISSVectorStore
    from ingestion.embedding_generator import EmbeddingGenerator
    from ingestion.chunker import Chunk

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FAISSVectorStore(index_path=tmpdir, dimension=384)
        embedder = EmbeddingGenerator()

        texts = [
            "Python is a high-level programming language",
            "FastAPI is used to build REST APIs with Python",
            "Machine learning algorithms learn from data",
            "Deep learning uses neural networks with many layers",
        ]
        chunks = [
            Chunk(text=t, chunk_id=f"c{i}", doc_id="d1",
                  source="test.pdf", doc_type="pdf", chunk_index=i)
            for i, t in enumerate(texts)
        ]
        embeddings = await embedder.embed_chunks(chunks)
        await store.add_chunks(chunks, embeddings)

        retriever = Retriever(
            vector_store=store,
            embedding_generator=embedder,
            top_k=2,
        )

        result = await retriever.retrieve("What is Python used for?")
        assert len(result.chunks) <= 2
        assert result.query == "What is Python used for?"
        assert result.retrieval_time_ms > 0
        # Python-related chunks should rank higher
        sources_text = " ".join(r.chunk.text for r in result.chunks)
        assert "Python" in sources_text or "FastAPI" in sources_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
