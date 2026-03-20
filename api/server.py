"""
api/server.py
-------------
FastAPI application factory with full production middleware stack:
  - CORS
  - Rate limiting (slowapi)
  - In-memory response caching
  - Structured request logging with request_id
  - Global exception handling
  - Startup/shutdown lifecycle (loads vector index on startup)
  - OpenAPI docs at /docs
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.config import settings
from utils.logger import get_logger, set_request_id

logger = get_logger(__name__)

# ── App state (shared across requests) ───────────────────────────────────────
_app_state: Dict[str, Any] = {}


def get_app_state() -> Dict[str, Any]:
    return _app_state


# ── In-Memory Cache ───────────────────────────────────────────────────────────

class InMemoryCache:
    """Thread-safe TTL cache for query responses."""

    def __init__(self, ttl: int = 300, max_size: int = 500):
        self._store: Dict[str, tuple] = {}
        self.ttl = ttl
        self.max_size = max_size

    def _make_key(self, query: str) -> str:
        import hashlib
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[dict]:
        key = self._make_key(query)
        if key in self._store:
            value, expires_at = self._store[key]
            if time.time() < expires_at:
                return value
            else:
                del self._store[key]
        return None

    def set(self, query: str, value: dict) -> None:
        if len(self._store) >= self.max_size:
            # Evict oldest entry
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        key = self._make_key(query)
        self._store[key] = (value, time.time() + self.ttl)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# ── Ingestion Service (coordinates loader + chunker + embedder + store) ───────

class IngestionService:
    """
    Orchestrates the full document ingestion pipeline:
    load → chunk → embed → store
    """

    def __init__(self, retriever, embedding_generator):
        self.retriever = retriever
        self.embedding_generator = embedding_generator

        from ingestion.document_loader import DocumentLoader
        from ingestion.chunker import TextChunker
        self.loader = DocumentLoader()
        self.chunker = TextChunker()

    async def ingest_file(self, file_path: str) -> dict:
        return await self._ingest(file_path)

    async def ingest_url(self, url: str) -> dict:
        return await self._ingest(url)

    async def _ingest(self, source: str) -> dict:
        # 1. Load
        documents = await self.loader.load(source)
        if not documents:
            raise ValueError(f"No content extracted from: {source}")

        # 2. Chunk
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            raise ValueError(f"No chunks produced from: {source}")

        # 3. Embed
        embeddings = await self.embedding_generator.embed_chunks(chunks)

        # 4. Store
        await self.retriever.store.add_chunks(chunks, embeddings)

        return {
            "doc_ids": list({c.doc_id for c in chunks}),
            "chunk_count": len(chunks),
            "sources": list({c.source for c in chunks}),
        }


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all heavy resources once at startup.
    This runs before any request is served.
    """
    logger.info("rag_system_starting", version=settings.APP_VERSION)

    # Initialize components
    from ingestion.embedding_generator import EmbeddingGenerator
    from retrieval.retriever import Retriever
    from generation.llm_engine import LLMEngine

    embedding_generator = EmbeddingGenerator()
    retriever = Retriever(embedding_generator=embedding_generator)
    await retriever.initialize()

    llm_engine = LLMEngine()

    ingestion_service = IngestionService(
        retriever=retriever,
        embedding_generator=embedding_generator,
    )

    cache = InMemoryCache(
        ttl=settings.CACHE_TTL,
        max_size=500,
    ) if settings.ENABLE_CACHE else None

    _app_state.update({
        "retriever": retriever,
        "llm_engine": llm_engine,
        "embedding_generator": embedding_generator,
        "ingestion_service": ingestion_service,
        "cache": cache,
    })

    logger.info(
        "rag_system_ready",
        provider=settings.LLM_PROVIDER,
        model=settings.effective_llm_model,
        vector_store=settings.VECTOR_STORE_PROVIDER,
        embedding=settings.EMBEDDING_PROVIDER,
        cache_enabled=settings.ENABLE_CACHE,
    )

    yield  # Application runs here

    # Shutdown: persist index
    logger.info("rag_system_shutting_down")
    try:
        await retriever.store.save()
        logger.info("index_persisted_on_shutdown")
    except Exception as e:
        logger.error("shutdown_save_failed", error=str(e))


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production-grade Retrieval-Augmented Generation (RAG) API",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate Limiting ─────────────────────────────────────────────────────
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address

        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("rate_limiting_enabled", requests=settings.RATE_LIMIT_REQUESTS)
    except ImportError:
        logger.warning("slowapi_not_installed_rate_limiting_disabled")

    # ── Request Logging Middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Callable):
        request_id = set_request_id()
        request.state.request_id = request_id
        start = time.perf_counter()

        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_ms=round(elapsed_ms, 2),
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"
        return response

    # ── Global Exception Handler ───────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.DEBUG else "Contact support",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    # ── Register Routes ────────────────────────────────────────────────────
    from api.routes import router
    app.include_router(router)

    # ── Root redirect ──────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": f"{settings.APP_NAME} v{settings.APP_VERSION}", "docs": "/docs"}

    return app


# ── Entry point ────────────────────────────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
