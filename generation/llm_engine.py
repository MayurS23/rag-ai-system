"""
generation/llm_engine.py
-------------------------
LLM generation layer: builds RAG prompts, calls LLM APIs, extracts citations.

Supports:
  - Anthropic Claude (claude-sonnet-4-20250514, etc.)
  - OpenAI GPT-4o / GPT-4 Turbo
  - Ollama (local models: llama3, mistral, etc.)

Prompt engineering:
  - System prompt instructs strict grounding in context
  - Context injected with source labels for citation extraction
  - Fallback detection: "I don't know" when context is insufficient
  - Streaming support for real-time token generation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from retrieval.retriever import RetrievalResult
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── RAG System Prompt ────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a precise, helpful AI knowledge assistant.

Your task is to answer the user's question using ONLY the provided context documents.

Rules you must follow:
1. Base your answer STRICTLY on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, respond with:
   "I don't have enough information in the provided documents to answer this question."
3. Always cite your sources using [Source: <source_name>] inline after each claim.
4. If multiple sources support a claim, cite all of them.
5. Be concise but complete. Prefer clear, structured answers with bullet points when listing facts.
6. Never fabricate information or infer beyond what the context explicitly states.
7. If asked for opinions or speculation, clarify you can only speak to what the documents say.

Format citations as: [Source: filename.pdf] or [Source: https://example.com]
"""


# ── Response Data Models ──────────────────────────────────────────────────────

@dataclass
class Citation:
    source: str
    chunk_id: str
    relevance_score: float
    excerpt: str       # Short excerpt from the chunk

    def __repr__(self) -> str:
        return f"Citation(source={self.source!r}, score={self.relevance_score:.3f})"


@dataclass
class GenerationResponse:
    """Full response from the LLM generation pipeline."""
    answer: str
    citations: List[Citation]
    query: str
    retrieval_result: Optional[RetrievalResult] = None
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def unique_sources(self) -> List[str]:
        """Deduplicated list of all cited sources."""
        seen = set()
        result = []
        for c in self.citations:
            if c.source not in seen:
                seen.add(c.source)
                result.append(c.source)
        return result

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [
                {
                    "source": c.source,
                    "chunk_id": c.chunk_id,
                    "relevance_score": round(c.relevance_score, 4),
                    "excerpt": c.excerpt[:200],
                }
                for c in self.citations
            ],
            "sources": self.unique_sources,
            "model": self.model,
            "provider": self.provider,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "performance": {
                "generation_time_ms": self.generation_time_ms,
            },
        }

    def __repr__(self) -> str:
        return (
            f"GenerationResponse(sources={len(self.unique_sources)}, "
            f"tokens={self.total_tokens}, time={self.generation_time_ms:.0f}ms)"
        )


# ── LLM Engine ────────────────────────────────────────────────────────────────

class LLMEngine:
    """
    Production RAG generation engine.
    Assembles prompts, calls LLM, extracts citations, handles errors.
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.effective_llm_model
        logger.info("llm_engine_init", provider=self.provider, model=self.model)

    async def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        stream: bool = False,
        custom_system_prompt: Optional[str] = None,
    ) -> GenerationResponse:
        """
        Generate a grounded answer from retrieved context.

        Args:
            query:              User's question
            retrieval_result:   Retrieved chunks from vector store
            stream:             If True, streams tokens (returns full response after completion)
            custom_system_prompt: Override default RAG system prompt

        Returns:
            GenerationResponse with answer, citations, and usage metadata
        """
        start = time.perf_counter()

        # Build the prompt
        context_block = self._build_context_block(retrieval_result)
        user_message = self._build_user_message(query, context_block)
        system_prompt = custom_system_prompt or RAG_SYSTEM_PROMPT

        # Dispatch to correct provider
        if self.provider == "anthropic":
            raw_answer, usage = await self._call_anthropic(system_prompt, user_message)
        elif self.provider == "openai":
            raw_answer, usage = await self._call_openai(system_prompt, user_message)
        elif self.provider == "ollama":
            raw_answer, usage = await self._call_ollama(system_prompt, user_message)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract citations from retrieved chunks
        citations = self._build_citations(retrieval_result)

        response = GenerationResponse(
            answer=raw_answer,
            citations=citations,
            query=query,
            retrieval_result=retrieval_result,
            model=self.model,
            provider=self.provider,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            generation_time_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "generation_complete",
            provider=self.provider,
            model=self.model,
            tokens=response.total_tokens,
            citations=len(citations),
            elapsed_ms=response.generation_time_ms,
        )

        return response

    async def generate_stream(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        custom_system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens as they're generated. Yields string chunks.
        Usage: async for token in engine.generate_stream(query, result): ...
        """
        context_block = self._build_context_block(retrieval_result)
        user_message = self._build_user_message(query, context_block)
        system_prompt = custom_system_prompt or RAG_SYSTEM_PROMPT

        if self.provider == "anthropic":
            async for token in self._stream_anthropic(system_prompt, user_message):
                yield token
        elif self.provider == "openai":
            async for token in self._stream_openai(system_prompt, user_message):
                yield token
        else:
            # Fallback: non-streaming call yielded in one chunk
            answer, _ = await self._call_ollama(system_prompt, user_message)
            yield answer

    # ── Prompt Building ────────────────────────────────────────────────────

    def _build_context_block(self, retrieval_result: RetrievalResult) -> str:
        """Build the context section of the prompt with numbered sources."""
        if not retrieval_result.chunks:
            return "No relevant context found in the knowledge base."

        parts = ["=== CONTEXT DOCUMENTS ===\n"]
        for i, result in enumerate(retrieval_result.chunks, 1):
            chunk = result.chunk
            source_label = chunk.source
            text = chunk.text.strip()
            score = result.score

            parts.append(
                f"[Document {i}] Source: {source_label} | Relevance: {score:.3f}\n"
                f"{text}\n"
            )

        parts.append("=== END OF CONTEXT ===")
        return "\n".join(parts)

    def _build_user_message(self, query: str, context_block: str) -> str:
        return f"{context_block}\n\nQuestion: {query}\n\nAnswer:"

    # ── Anthropic ──────────────────────────────────────────────────────────

    async def _call_anthropic(self, system: str, user_msg: str):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic required: pip install anthropic")

        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")

        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

        message = await client.messages.create(
            model=self.model,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            temperature=settings.LLM_TEMPERATURE,
        )

        answer = message.content[0].text
        usage = {
            "prompt_tokens": message.usage.input_tokens,
            "completion_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        }
        return answer, usage

    async def _stream_anthropic(self, system: str, user_msg: str) -> AsyncIterator[str]:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

        async with client.messages.stream(
            model=self.model,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            temperature=settings.LLM_TEMPERATURE,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    # ── OpenAI ─────────────────────────────────────────────────────────────

    async def _call_openai(self, system: str, user_msg: str):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai required: pip install openai")

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env")

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )

        answer = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return answer, usage

    async def _stream_openai(self, system: str, user_msg: str) -> AsyncIterator[str]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        stream = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    # ── Ollama (local) ─────────────────────────────────────────────────────

    async def _call_ollama(self, system: str, user_msg: str):
        import httpx

        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": f"System: {system}\n\nUser: {user_msg}",
            "stream": False,
            "options": {"temperature": settings.LLM_TEMPERATURE},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        answer = data.get("response", "")
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }
        return answer, usage

    # ── Citation Extraction ────────────────────────────────────────────────

    def _build_citations(self, retrieval_result: RetrievalResult) -> List[Citation]:
        """Build Citation objects from all retrieved chunks."""
        citations = []
        for search_result in retrieval_result.chunks:
            chunk = search_result.chunk
            # Use first 200 chars as excerpt
            excerpt = chunk.text[:200].strip()
            if len(chunk.text) > 200:
                excerpt += "..."

            citations.append(Citation(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                relevance_score=search_result.score,
                excerpt=excerpt,
            ))

        return citations
