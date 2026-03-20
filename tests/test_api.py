"""
tests/test_api.py
API integration tests using FastAPI TestClient.
Tests run against the full app with mocked LLM + real embeddings.
"""
import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def test_client():
    """Create a FastAPI test client with mocked LLM."""
    from fastapi.testclient import TestClient
    from api.server import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client


def test_health_endpoint(test_client):
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_stats_endpoint(test_client):
    response = test_client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data or "total_vectors" in data


def test_ingest_text_file(test_client):
    content = b"This is a test document about machine learning and AI systems."
    response = test_client.post(
        "/api/v1/ingest",
        files={"file": ("test_doc.txt", content, "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["chunk_count"] >= 1


def test_ingest_markdown_file(test_client):
    content = b"# Test Document\n\nThis is markdown content about RAG systems.\n\n## Section\nMore content here."
    response = test_client.post(
        "/api/v1/ingest",
        files={"file": ("test.md", content, "text/markdown")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_query_requires_content(test_client):
    """Query endpoint should return error for empty query."""
    response = test_client.post(
        "/api/v1/query",
        json={"query": ""},
    )
    assert response.status_code == 422  # Validation error


def test_response_has_request_id(test_client):
    """Every response should have X-Request-ID header."""
    response = test_client.get("/api/v1/health")
    assert "x-request-id" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
