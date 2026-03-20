"""
tests/test_ingestion.py
Tests for the document loading and chunking pipeline.
"""
import asyncio
import os
import tempfile
import pytest
from pathlib import Path


# ── Document Loader Tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_text_file():
    from ingestion.document_loader import DocumentLoader
    loader = DocumentLoader()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test document about artificial intelligence and machine learning.")
        path = f.name

    try:
        docs = await loader.load(path)
        assert len(docs) == 1
        assert "artificial intelligence" in docs[0].content
        assert docs[0].doc_type == "text"
        assert docs[0].doc_id.startswith("doc_")
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_load_markdown_file():
    from ingestion.document_loader import DocumentLoader
    loader = DocumentLoader()

    content = "# RAG System\n\nThis is a **test** markdown document.\n\n## Section 2\nMore content here."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        docs = await loader.load(path)
        assert len(docs) == 1
        assert docs[0].doc_type == "markdown"
        assert "RAG System" in docs[0].content
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_load_python_file():
    from ingestion.document_loader import DocumentLoader
    loader = DocumentLoader()

    code = '''
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name

    try:
        docs = await loader.load(path)
        assert len(docs) == 1
        assert docs[0].doc_type == "code"
        assert docs[0].metadata["language"] == "python"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_load_directory():
    from ingestion.document_loader import DocumentLoader
    loader = DocumentLoader()

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            with open(os.path.join(tmpdir, f"doc_{i}.txt"), "w") as f:
                f.write(f"Document {i} content with some useful text about topic {i}.")

        docs = await loader.load(tmpdir)
        assert len(docs) == 3


@pytest.mark.asyncio
async def test_load_nonexistent_file():
    from ingestion.document_loader import DocumentLoader, DocumentLoadError
    loader = DocumentLoader()

    with pytest.raises(DocumentLoadError):
        await loader.load("/nonexistent/path/file.txt")


# ── Chunker Tests ────────────────────────────────────────────────────────────

def test_recursive_chunker_basic():
    from ingestion.chunker import TextChunker
    from ingestion.document_loader import Document

    chunker = TextChunker(chunk_size=100, chunk_overlap=20, strategy="recursive")
    doc = Document(
        content="This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three.",
        source="test.txt",
        doc_type="text",
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.doc_id == doc.doc_id
        assert chunk.source == "test.txt"
        assert len(chunk.text) > 0


def test_chunker_preserves_metadata():
    from ingestion.chunker import TextChunker
    from ingestion.document_loader import Document

    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    doc = Document(
        content="A" * 500,
        source="test_source.pdf",
        doc_type="pdf",
        metadata={"filename": "test_source.pdf", "page_count": 10},
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.get("filename") == "test_source.pdf"


def test_chunk_ids_are_unique():
    from ingestion.chunker import TextChunker
    from ingestion.document_loader import Document

    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    doc = Document(
        content="Word " * 200,
        source="test.txt",
        doc_type="text",
    )
    chunks = chunker.chunk_document(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_code_chunker():
    from ingestion.chunker import TextChunker
    from ingestion.document_loader import Document

    code = "\n".join([f"def function_{i}():\n    return {i}\n" for i in range(20)])
    chunker = TextChunker(chunk_size=50, chunk_overlap=5, strategy="recursive")
    doc = Document(content=code, source="test.py", doc_type="code")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1


# ── Embedding Generator Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_texts():
    from ingestion.embedding_generator import EmbeddingGenerator
    import numpy as np

    gen = EmbeddingGenerator()
    texts = ["What is machine learning?", "How do neural networks work?"]
    embeddings = await gen.embed_texts(texts)

    assert len(embeddings) == 2
    assert isinstance(embeddings[0], np.ndarray)
    assert embeddings[0].shape[0] > 0
    assert embeddings[0].dtype == np.float32


@pytest.mark.asyncio
async def test_embed_query():
    from ingestion.embedding_generator import EmbeddingGenerator
    import numpy as np

    gen = EmbeddingGenerator()
    embedding = await gen.embed_query("What is RAG?")

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1


@pytest.mark.asyncio
async def test_embeddings_are_normalized():
    """Normalized embeddings should have magnitude ~1.0"""
    from ingestion.embedding_generator import EmbeddingGenerator
    import numpy as np

    gen = EmbeddingGenerator()
    embeddings = await gen.embed_texts(["test sentence"])
    magnitude = np.linalg.norm(embeddings[0])
    assert abs(magnitude - 1.0) < 0.01, f"Expected ~1.0 magnitude, got {magnitude}"


@pytest.mark.asyncio
async def test_similar_sentences_closer_than_different():
    """Semantically similar sentences should have higher cosine similarity."""
    from ingestion.embedding_generator import EmbeddingGenerator
    import numpy as np

    gen = EmbeddingGenerator()
    embeddings = await gen.embed_texts([
        "The cat sat on the mat",
        "A feline rested on the rug",  # semantically similar
        "Quantum mechanics and particle physics",  # different
    ])

    sim_similar = float(np.dot(embeddings[0], embeddings[1]))
    sim_different = float(np.dot(embeddings[0], embeddings[2]))

    assert sim_similar > sim_different, (
        f"Similar sentences ({sim_similar:.3f}) should have "
        f"higher similarity than different ones ({sim_different:.3f})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
