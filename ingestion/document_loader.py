"""
ingestion/document_loader.py
-----------------------------
Loads documents from multiple sources into a unified Document schema.

Supported sources:
  - PDF files          (.pdf)
  - Markdown files     (.md, .mdx)
  - Plain text files   (.txt)
  - Code files         (.py, .js, .ts, .java, .go, .rs, .cpp, etc.)
  - Web URLs           (HTML → plain text via BeautifulSoup)
  - Git repositories   (clones and loads all text files)

Every loader returns List[Document] — a normalized contract for downstream
chunking and embedding stages.
"""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import aiofiles
import chardet
import requests
from bs4 import BeautifulSoup

from utils.logger import get_logger

logger = get_logger(__name__)

# Supported code file extensions
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".scala", ".r", ".sh", ".bash", ".yaml", ".yml", ".json",
    ".toml", ".xml", ".sql", ".graphql",
}


@dataclass
class Document:
    """
    Normalized document representation.
    This is the contract between the loader and all downstream components.
    """
    content: str                          # Raw text content
    source: str                           # Original source path / URL
    doc_type: str                         # "pdf" | "markdown" | "text" | "code" | "web"
    metadata: dict = field(default_factory=dict)
    doc_id: str = field(default="")

    def __post_init__(self):
        if not self.doc_id:
            # Deterministic ID based on source + content hash
            content_hash = hashlib.md5(
                f"{self.source}{self.content[:200]}".encode()
            ).hexdigest()[:12]
            self.doc_id = f"doc_{content_hash}"

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        return len(self.content)

    def __repr__(self) -> str:
        return (
            f"Document(id={self.doc_id}, type={self.doc_type}, "
            f"source={self.source!r}, words={self.word_count})"
        )


class DocumentLoader:
    """
    Unified document loader with async support.
    Routes to the correct specialized loader based on source type.
    """

    def __init__(self, request_timeout: int = 30):
        self.request_timeout = request_timeout

    # ── Public API ────────────────────────────────────────────────────────

    async def load(self, source: str) -> List[Document]:
        """
        Main entry point. Accepts a file path, URL, or directory path.
        Returns a list of Document objects (one file can produce multiple docs
        in the case of a directory or repository).
        """
        start = time.perf_counter()

        try:
            if self._is_url(source):
                docs = await self._load_url(source)
            elif self._is_git_repo(source):
                docs = await self._load_git_repo(source)
            elif os.path.isdir(source):
                docs = await self._load_directory(source)
            else:
                docs = await self._load_file(source)

            elapsed = time.perf_counter() - start
            logger.info(
                "documents_loaded",
                source=source,
                doc_count=len(docs),
                elapsed_ms=round(elapsed * 1000, 2),
            )
            return docs

        except Exception as e:
            logger.error("document_load_failed", source=source, error=str(e))
            raise DocumentLoadError(f"Failed to load '{source}': {e}") from e

    async def load_many(self, sources: List[str]) -> List[Document]:
        """Load multiple sources concurrently."""
        tasks = [self.load(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: List[Document] = []
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.warning("source_load_skipped", source=source, error=str(result))
            else:
                docs.extend(result)

        return docs

    # ── File Router ───────────────────────────────────────────────────────

    async def _load_file(self, path: str) -> List[Document]:
        """Route to the correct loader based on file extension."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = p.suffix.lower()

        if ext == ".pdf":
            return await self._load_pdf(path)
        elif ext in {".md", ".mdx"}:
            return await self._load_markdown(path)
        elif ext == ".txt":
            return await self._load_text(path)
        elif ext in CODE_EXTENSIONS:
            return await self._load_code(path)
        else:
            # Try to load as text with encoding detection
            return await self._load_text(path)

    async def _load_directory(self, dir_path: str) -> List[Document]:
        """Recursively load all supported files in a directory."""
        supported_exts = {".pdf", ".md", ".mdx", ".txt"} | CODE_EXTENSIONS
        files = [
            str(p) for p in Path(dir_path).rglob("*")
            if p.is_file() and p.suffix.lower() in supported_exts
            and not any(part.startswith(".") for part in p.parts)  # skip hidden
            and "__pycache__" not in p.parts
            and "node_modules" not in p.parts
        ]

        logger.info("loading_directory", path=dir_path, file_count=len(files))
        return await self.load_many(files)

    # ── Specialized Loaders ───────────────────────────────────────────────

    async def _load_pdf(self, path: str) -> List[Document]:
        """Extract text from PDF using pypdf with page-level metadata."""
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required: pip install pypdf")

        documents = []
        reader = pypdf.PdfReader(path)
        full_text_parts = []

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                full_text_parts.append(page_text)

        full_text = "\n\n".join(full_text_parts)
        if full_text.strip():
            documents.append(Document(
                content=full_text,
                source=path,
                doc_type="pdf",
                metadata={
                    "filename": Path(path).name,
                    "page_count": len(reader.pages),
                    "file_path": str(Path(path).absolute()),
                },
            ))

        return documents

    async def _load_markdown(self, path: str) -> List[Document]:
        """Load markdown file, preserve raw markdown text."""
        async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as f:
            content = await f.read()

        return [Document(
            content=content,
            source=path,
            doc_type="markdown",
            metadata={
                "filename": Path(path).name,
                "file_path": str(Path(path).absolute()),
                "extension": Path(path).suffix,
            },
        )]

    async def _load_text(self, path: str) -> List[Document]:
        """Load plain text file with auto encoding detection."""
        with open(path, "rb") as f:
            raw = f.read()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"

        content = raw.decode(encoding, errors="replace")

        return [Document(
            content=content,
            source=path,
            doc_type="text",
            metadata={
                "filename": Path(path).name,
                "encoding": encoding,
                "file_path": str(Path(path).absolute()),
            },
        )]

    async def _load_code(self, path: str) -> List[Document]:
        """Load source code file with language metadata."""
        with open(path, "rb") as f:
            raw = f.read()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        content = raw.decode(encoding, errors="replace")

        ext = Path(path).suffix.lower()
        language_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp",
            ".c": "c", ".cs": "csharp", ".rb": "ruby", ".php": "php",
            ".swift": "swift", ".kt": "kotlin", ".sql": "sql",
        }

        return [Document(
            content=content,
            source=path,
            doc_type="code",
            metadata={
                "filename": Path(path).name,
                "language": language_map.get(ext, "unknown"),
                "extension": ext,
                "file_path": str(Path(path).absolute()),
                "line_count": content.count("\n"),
            },
        )]

    async def _load_url(self, url: str) -> List[Document]:
        """Fetch a webpage and extract clean text using BeautifulSoup."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(url, timeout=self.request_timeout, headers={
                "User-Agent": "Mozilla/5.0 (compatible; RAG-Bot/1.0)"
            })
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script, style, nav, footer noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else urlparse(url).netloc

        # Extract main content
        main = soup.find("main") or soup.find("article") or soup.find("body")
        content = main.get_text(separator="\n", strip=True) if main else ""

        # Clean up excessive whitespace
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        content = "\n".join(lines)

        return [Document(
            content=content,
            source=url,
            doc_type="web",
            metadata={
                "url": url,
                "title": title_text,
                "domain": urlparse(url).netloc,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
            },
        )]

    async def _load_git_repo(self, repo_url: str) -> List[Document]:
        """Clone a git repo to a temp dir and load all supported files."""
        try:
            import git
        except ImportError:
            raise ImportError("gitpython is required: pip install gitpython")

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("cloning_repo", url=repo_url, target=tmpdir)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: git.Repo.clone_from(repo_url, tmpdir, depth=1)
            )
            docs = await self._load_directory(tmpdir)
            # Update sources to show original repo URL
            for doc in docs:
                doc.metadata["repo_url"] = repo_url
            return docs

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_url(source: str) -> bool:
        try:
            result = urlparse(source)
            return result.scheme in {"http", "https"}
        except Exception:
            return False

    @staticmethod
    def _is_git_repo(source: str) -> bool:
        return (
            source.endswith(".git") or
            ("github.com" in source and not source.endswith((".pdf", ".md", ".txt")))
        )


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded."""
    pass
