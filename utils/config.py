"""
utils/config.py
---------------
Central configuration management using Pydantic BaseSettings.
All config values are loaded from environment variables / .env file.
This is the single source of truth for ALL configuration in the system.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Production-grade settings with full type validation.
    Environment variables override .env file values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = "RAG AI System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── LLM Provider ─────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["anthropic", "openai", "ollama"] = "anthropic"
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    OPENAI_MODEL: str = "gpt-4o"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # ── Embeddings ────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: Literal["local", "openai"] = "local"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # ── Vector Store ──────────────────────────────────────────────────────
    VECTOR_STORE_PROVIDER: Literal["faiss", "pinecone"] = "faiss"
    FAISS_INDEX_PATH: Path = Path("./data/faiss_index")
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "rag-index"
    PINECONE_ENVIRONMENT: str = "us-east-1"

    # ── Chunking ──────────────────────────────────────────────────────────
    CHUNK_SIZE: int = Field(default=512, ge=64, le=4096)
    CHUNK_OVERLAP: int = Field(default=100, ge=0, le=512)
    CHUNKING_STRATEGY: Literal["fixed", "recursive", "semantic"] = "recursive"

    # ── Retrieval ─────────────────────────────────────────────────────────
    TOP_K_RESULTS: int = Field(default=5, ge=1, le=20)
    SIMILARITY_THRESHOLD: float = Field(default=0.3, ge=0.0, le=1.0)

    # ── API Server ────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    API_WORKERS: int = 1

    # ── Rate Limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # ── Caching ───────────────────────────────────────────────────────────
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 300  # seconds
    CACHE_BACKEND: Literal["memory", "redis"] = "memory"
    REDIS_URL: str = "redis://localhost:6379"

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"
    LOG_FILE: Optional[Path] = Path("./data/rag_system.log")

    @field_validator("FAISS_INDEX_PATH", "LOG_FILE", mode="before")
    @classmethod
    def ensure_parent_dir(cls, v):
        if v:
            path = Path(v)
            path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def effective_llm_api_key(self) -> Optional[str]:
        if self.LLM_PROVIDER == "anthropic":
            return self.ANTHROPIC_API_KEY
        elif self.LLM_PROVIDER == "openai":
            return self.OPENAI_API_KEY
        return None

    @property
    def effective_llm_model(self) -> str:
        if self.LLM_PROVIDER == "anthropic":
            return self.ANTHROPIC_MODEL
        elif self.LLM_PROVIDER == "openai":
            return self.OPENAI_MODEL
        return self.OLLAMA_MODEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    Using lru_cache ensures we only parse .env once across the app lifetime.
    """
    return Settings()


# Convenience alias used throughout the codebase
settings = get_settings()
