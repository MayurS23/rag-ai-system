"""
api/routes.py
-------------
All FastAPI route definitions for the RAG system.

Endpoints:
  POST /api/v1/ingest          — Upload & index a document
  POST /api/v1/ingest/url      — Ingest from a URL
  POST /api/v1/query           — Query the RAG pipeline
  POST /api/v1/query/stream    — Streaming query (SSE)
  GET  /api/v1/documents       — List all indexed documents
  DELETE /api/v1/documents     — Remove a document by source
  GET  /api/v1/health          — Health check
  GET  /api/v1/stats           — Index statistics
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl

from utils.logger import get_logger, set_request_id

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")


# ── Request / Response Models ─────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: str = Field(..., description="URL to fetch and index")
    metadata: dict = Field(default_factory=dict, description="Optional metadata tags")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks")
    filter_metadata: Optional[dict] = Field(default=None, description="Filter by metadata")
    use_mmr: bool = Field(default=True, description="Apply MMR diversity reranking")
    stream: bool = Field(default=False, description="Stream the response")


class DocumentInfo(BaseModel):
    source: str
    doc_type: str
    chunk_count: int
    metadata: dict


class IngestResponse(BaseModel):
    success: bool
    doc_ids: List[str]
    chunk_count: int
    sources: List[str]
    elapsed_ms: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list
    sources: List[str]
    model: str
    provider: str
    usage: dict
    performance: dict
    request_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    model: str


# ── Dependency injection ───────────────────────────────────────────────────────

def get_pipeline():
    """Inject the RAG pipeline from app state."""
    from api.server import get_app_state
    return get_app_state()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, summary="Ingest a document file")
async def ingest_file(
    file: UploadFile = File(...),
    pipeline=Depends(get_pipeline),
):
    """
    Upload a file (PDF, MD, TXT, code) and index it into the vector store.
    The file is temporarily saved, loaded, chunked, embedded, and indexed.
    """
    request_id = set_request_id()
    start = time.perf_counter()

    logger.info("ingest_request", filename=file.filename, content_type=file.content_type)

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save to temp file
    import tempfile, os, shutil
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = await pipeline["ingestion_service"].ingest_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    elapsed = (time.perf_counter() - start) * 1000
    return IngestResponse(
        success=True,
        doc_ids=result["doc_ids"],
        chunk_count=result["chunk_count"],
        sources=result["sources"],
        elapsed_ms=round(elapsed, 2),
    )


@router.post("/ingest/url", response_model=IngestResponse, summary="Ingest from a URL")
async def ingest_url(
    request: IngestURLRequest,
    pipeline=Depends(get_pipeline),
):
    """Fetch a webpage or direct file URL and index it."""
    request_id = set_request_id()
    start = time.perf_counter()

    logger.info("ingest_url_request", url=request.url)

    try:
        result = await pipeline["ingestion_service"].ingest_url(request.url)
    except Exception as e:
        logger.error("ingest_url_failed", url=request.url, error=str(e))
        raise HTTPException(status_code=422, detail=str(e))

    elapsed = (time.perf_counter() - start) * 1000
    return IngestResponse(
        success=True,
        doc_ids=result["doc_ids"],
        chunk_count=result["chunk_count"],
        sources=result["sources"],
        elapsed_ms=round(elapsed, 2),
    )


@router.post("/query", response_model=QueryResponse, summary="Query the knowledge base")
async def query(
    request: QueryRequest,
    pipeline=Depends(get_pipeline),
):
    """
    Ask a question against the indexed knowledge base.
    Returns a grounded answer with citations and source references.
    """
    request_id = set_request_id()
    start = time.perf_counter()

    logger.info("query_request", query=request.query[:80], top_k=request.top_k)

    # Check cache
    cache = pipeline.get("cache")
    if cache:
        cached = cache.get(request.query)
        if cached:
            logger.info("cache_hit", query=request.query[:40])
            cached["request_id"] = request_id
            return QueryResponse(**cached)

    try:
        # Retrieve relevant chunks
        retrieval_result = await pipeline["retriever"].retrieve(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
            use_mmr=request.use_mmr,
        )

        # Generate answer
        gen_response = await pipeline["llm_engine"].generate(
            query=request.query,
            retrieval_result=retrieval_result,
        )

    except Exception as e:
        logger.error("query_failed", error=str(e), query=request.query[:80])
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    response_data = gen_response.to_dict()
    response_data["query"] = request.query
    response_data["request_id"] = request_id

    # Cache successful response
    if cache:
        cache.set(request.query, response_data)

    return QueryResponse(**response_data)


@router.post("/query/stream", summary="Streaming query response (SSE)")
async def query_stream(
    request: QueryRequest,
    pipeline=Depends(get_pipeline),
):
    """
    Stream the LLM response token-by-token using Server-Sent Events.
    Connect with EventSource in the frontend.
    """
    set_request_id()

    async def event_generator():
        try:
            retrieval_result = await pipeline["retriever"].retrieve(
                query=request.query,
                top_k=request.top_k,
                use_mmr=request.use_mmr,
            )

            # First, stream sources metadata
            sources_json = str(retrieval_result.sources)
            yield f"data: {{\"type\": \"sources\", \"sources\": {sources_json}}}\n\n"

            # Then stream tokens
            async for token in pipeline["llm_engine"].generate_stream(
                query=request.query,
                retrieval_result=retrieval_result,
            ):
                escaped = token.replace('"', '\\"').replace('\n', '\\n')
                yield f'data: {{"type": "token", "content": "{escaped}"}}\n\n'

            yield 'data: {"type": "done"}\n\n'

        except Exception as e:
            logger.error("stream_failed", error=str(e))
            yield f'data: {{"type": "error", "message": "{str(e)}"}}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/documents", summary="Remove a document by source")
async def delete_document(
    source: str,
    pipeline=Depends(get_pipeline),
):
    """Remove all chunks from the given source path/URL from the vector index."""
    deleted = await pipeline["retriever"].store.delete_by_source(source)
    return {"deleted": deleted, "source": source}


@router.get("/stats", summary="Vector store statistics")
async def get_stats(pipeline=Depends(get_pipeline)):
    """Return statistics about the current index."""
    stats = await pipeline["retriever"].store.get_stats()
    return stats


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(pipeline=Depends(get_pipeline)):
    """Liveness probe — returns 200 if service is up."""
    from utils.config import settings
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        provider=settings.LLM_PROVIDER,
        model=settings.effective_llm_model,
    )
