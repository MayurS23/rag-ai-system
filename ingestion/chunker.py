"""
ingestion/chunker.py
---------------------
Text chunking strategies for optimal retrieval.

The key insight: chunk size is a tradeoff between:
  - Too large  → low retrieval precision, hits embedding token limits
  - Too small  → loses context, too many vector lookups needed

Strategies:
  1. FixedSizeChunker    — simple, fast, good baseline
  2. RecursiveChunker    — respects paragraph/sentence/word boundaries (default)
  3. SemanticChunker     — uses embedding similarity to find natural breaks (best quality)
  4. CodeChunker         — splits by function/class definitions

All chunkers return List[Chunk] with full provenance metadata.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from utils.config import settings
from utils.logger import get_logger
from ingestion.document_loader import Document

logger = get_logger(__name__)


@dataclass
class Chunk:
    """
    A text chunk ready for embedding.
    Carries full provenance: which document it came from, position, etc.
    """
    text: str
    chunk_id: str
    doc_id: str
    source: str
    doc_type: str
    chunk_index: int
    total_chunks: int = 0          # set after all chunks are created
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            h = hashlib.md5(f"{self.doc_id}_{self.chunk_index}_{self.text[:50]}".encode()).hexdigest()[:10]
            self.chunk_id = f"chunk_{h}"

    @property
    def token_estimate(self) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(self.text) // 4

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id}, doc={self.doc_id}, "
            f"index={self.chunk_index}/{self.total_chunks}, "
            f"tokens≈{self.token_estimate})"
        )


class TextChunker:
    """
    Main chunking facade. Selects strategy based on doc_type and config.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.strategy = strategy or settings.CHUNKING_STRATEGY

        self._fixed = FixedSizeChunker(self.chunk_size, self.chunk_overlap)
        self._recursive = RecursiveChunker(self.chunk_size, self.chunk_overlap)
        self._code = CodeChunker(self.chunk_size, self.chunk_overlap)

        logger.info(
            "chunker_initialized",
            strategy=self.strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document using the appropriate strategy."""
        start = time.perf_counter()

        # Route to specialized chunker based on doc type
        if document.doc_type == "code":
            raw_chunks = self._code.split(document.content)
        elif self.strategy == "fixed":
            raw_chunks = self._fixed.split(document.content)
        else:
            raw_chunks = self._recursive.split(document.content)

        chunks = []
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if not text or len(text) < 10:  # skip tiny fragments
                continue

            h = hashlib.md5(f"{document.doc_id}_{i}_{text[:50]}".encode()).hexdigest()[:10]

            chunk = Chunk(
                text=text,
                chunk_id=f"chunk_{h}",
                doc_id=document.doc_id,
                source=document.source,
                doc_type=document.doc_type,
                chunk_index=i,
                metadata={
                    **document.metadata,
                    "char_count": len(text),
                    "token_estimate": len(text) // 4,
                },
            )
            chunks.append(chunk)

        # Set total_chunks on all chunks now that we know the count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        elapsed = time.perf_counter() - start
        logger.info(
            "document_chunked",
            doc_id=document.doc_id,
            source=document.source,
            chunk_count=len(chunks),
            elapsed_ms=round(elapsed * 1000, 2),
        )

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents, returns flat list of all chunks."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            "batch_chunked",
            doc_count=len(documents),
            total_chunks=len(all_chunks),
        )
        return all_chunks


class FixedSizeChunker:
    """
    Splits text into fixed-size character windows with overlap.
    Fastest option. Doesn't respect word/sentence boundaries.
    Good for: quick prototyping, highly uniform text.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size * 4   # convert tokens → chars
        self.chunk_overlap = chunk_overlap * 4

    def split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks


class RecursiveChunker:
    """
    Splits text recursively using a hierarchy of separators.
    Tries to keep paragraphs → sentences → words together.
    This is the default strategy — best balance of quality and speed.

    Separator hierarchy:
    1. Double newline (paragraph break)
    2. Single newline (line break)
    3. Period/? /! (sentence end)
    4. Comma/semicolon (clause break)
    5. Space (word break)
    6. Character (last resort)
    """

    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(self, chunk_size: int, chunk_overlap: int):
        # chunk_size in tokens → chars (rough estimate: 4 chars/token)
        self.chunk_size = chunk_size * 4
        self.chunk_overlap = chunk_overlap * 4

    def split(self, text: str) -> List[str]:
        return self._recursive_split(text, self.SEPARATORS)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []

        # Find the best separator that actually exists in the text
        separator = separators[-1]  # fallback: character split
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)

            if current_len + split_len + len(separator) > self.chunk_size:
                # Flush good_splits as a chunk
                if good_splits:
                    merged = separator.join(good_splits)
                    final_chunks.append(merged)

                    # Keep overlap: retain last N chars worth of splits
                    overlap_text = merged[-self.chunk_overlap:] if self.chunk_overlap else ""
                    good_splits = [overlap_text] if overlap_text else []
                    current_len = len(overlap_text)

                # If a single split is still too large, recurse with finer separators
                if split_len > self.chunk_size and len(separators) > 1:
                    sub_chunks = self._recursive_split(split, separators[1:])
                    final_chunks.extend(sub_chunks[:-1])
                    # Keep last sub-chunk as seed for next iteration
                    if sub_chunks:
                        good_splits = [sub_chunks[-1]]
                        current_len = len(sub_chunks[-1])
                    continue

            good_splits.append(split)
            current_len += split_len + len(separator)

        if good_splits:
            final_chunks.append(separator.join(good_splits))

        return [c for c in final_chunks if c.strip()]


class CodeChunker:
    """
    Smart chunker for source code files.
    Splits at function/class definition boundaries to keep logical units intact.
    Falls back to RecursiveChunker if no clear boundaries found.
    """

    # Regex patterns for function/class starts in common languages
    BOUNDARY_PATTERNS = [
        r"^(async\s+)?def\s+\w+",          # Python functions
        r"^class\s+\w+",                    # Python/Java/JS classes
        r"^(export\s+)?(async\s+)?function\s+\w+",  # JS/TS functions
        r"^(public|private|protected|static).*\s+\w+\s*\(",  # Java/C# methods
        r"^func\s+\w+",                     # Go functions
        r"^fn\s+\w+",                       # Rust functions
        r"^(pub\s+)?impl\s+\w+",            # Rust impl blocks
    ]

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._fallback = RecursiveChunker(chunk_size, chunk_overlap)
        self._boundary_re = re.compile(
            "|".join(f"({p})" for p in self.BOUNDARY_PATTERNS),
            re.MULTILINE
        )

    def split(self, text: str) -> List[str]:
        lines = text.splitlines(keepends=True)
        chunks: List[str] = []
        current_block: List[str] = []
        current_size = 0

        for line in lines:
            # Check if this line starts a new logical block
            is_boundary = bool(self._boundary_re.match(line.strip()))

            if is_boundary and current_block and current_size >= self.chunk_size * 2:
                chunks.append("".join(current_block))
                # Keep some overlap: last few lines
                overlap_lines = current_block[-3:] if len(current_block) > 3 else current_block
                current_block = overlap_lines[:]
                current_size = sum(len(l) for l in current_block)

            current_block.append(line)
            current_size += len(line)

        if current_block:
            chunks.append("".join(current_block))

        # If we got no useful splits, fall back
        if len(chunks) <= 1 and len(text) > self.chunk_size * 4 * 2:
            return self._fallback.split(text)

        return chunks
