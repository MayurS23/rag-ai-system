#  RAG AI System — Production-Grade Knowledge Assistant

A fully production-ready **Retrieval-Augmented Generation (RAG)** system built with Python, FastAPI, FAISS, and support for Anthropic Claude, OpenAI GPT-4, and local Ollama models.

Ingest any document → ask questions → get accurate, cited answers grounded in your own data.

---

##  System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAG AI SYSTEM                             │
│                                                                  │
│  INGESTION PIPELINE              QUERY PIPELINE                  │
│  ─────────────────               ──────────────                  │
│  PDF / MD / URL / Code           User Question                   │
│        │                               │                         │
│        ▼                               ▼                         │
│   DocumentLoader               EmbeddingGenerator                │
│        │                               │                         │
│        ▼                               ▼                         │
│    TextChunker                   VectorStore.search()            │
│        │                               │                         │
│        ▼                               ▼                         │
│  EmbeddingGenerator            Retriever (MMR rerank)            │
│        │                               │                         │
│        ▼                               ▼                         │
│    VectorStore ──────────────► LLMEngine (prompt + generate)     │
│    (FAISS / Pinecone)                  │                         │
│                                        ▼                         │
│                              Answer + Citations                   │
└──────────────────────────────────────────────────────────────────┘
                          FastAPI REST Layer
                   /ingest  /query  /health  /stats
```

---

##  Features

| Feature | Details |
|---|---|
| **Multi-format ingestion** | PDF, Markdown, plain text, source code (20+ languages), URLs, Git repos |
| **Smart chunking** | Fixed-size, Recursive (default), Code-aware strategies with configurable overlap |
| **Embedding providers** | Local `sentence-transformers` (free) or OpenAI `text-embedding-3-small` |
| **Vector stores** | FAISS (local, free) or Pinecone (managed cloud) — swappable via config |
| **LLM providers** | Anthropic Claude, OpenAI GPT-4o, Ollama (local) — all switchable via `.env` |
| **MMR re-ranking** | Maximal Marginal Relevance for diverse, non-redundant results |
| **Streaming** | Server-Sent Events (SSE) for real-time token streaming |
| **Response caching** | In-memory or Redis cache with configurable TTL |
| **Rate limiting** | Per-IP request rate limiting via SlowAPI |
| **Structured logging** | JSON logs with request_id tracing (Datadog/CloudWatch ready) |
| **Full test suite** | Async pytest tests for every pipeline stage |
| **Docker support** | One-command containerized deployment |
| **Dark-mode UI** | Drag-and-drop frontend with citation display |

---

##  Project Structure

```
rag-ai-system/
│
├── ingestion/
│   ├── document_loader.py      # PDF, URL, Markdown, code → Document objects
│   ├── chunker.py              # Fixed / Recursive / Code-aware text splitting
│   └── embedding_generator.py # Text → dense vectors (local or OpenAI)
│
├── retrieval/
│   ├── vector_store.py         # FAISS + Pinecone abstraction layer
│   └── retriever.py            # Top-K semantic search with MMR reranking
│
├── generation/
│   └── llm_engine.py           # Prompt builder + LLM caller + citation extractor
│
├── api/
│   ├── server.py               # FastAPI app factory + middleware + lifecycle
│   └── routes.py               # /ingest, /query, /stream, /health, /stats
│
├── utils/
│   ├── config.py               # Pydantic BaseSettings — single source of truth
│   └── logger.py               # Structured JSON logging with request tracing
│
├── tests/
│   ├── test_ingestion.py       # Document loading + chunking + embedding tests
│   ├── test_retrieval.py       # Vector store + retriever tests
│   └── test_api.py             # FastAPI endpoint integration tests
│
├── frontend/
│   └── index.html              # Drag-drop UI with chat, citations, dark mode
│
├── data/                       # FAISS index + logs (auto-created)
├── .env.example                # All configuration keys with defaults
├── requirements.txt            # Pinned Python dependencies
├── Dockerfile                  # Production container
├── docker-compose.yml          # App + Redis stack
└── main.py                     # Application entry point
```

---

##  Quick Start

### 1. Clone and Set Up

```bash
# Clone the project
git clone <your-repo-url>
cd rag-ai-system

# Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Open `.env` and set your API key:

```env
# Choose your LLM provider
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Embeddings (local is free, no key needed)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector store (faiss is local, no key needed)
VECTOR_STORE_PROVIDER=faiss
```

### 3. Run the Server

```bash
python main.py
```

You'll see:
```
╔══════════════════════════════════════════════════╗
║       RAG AI System  v1.0.0                      ║
║                                                  ║
║    API:   http://localhost:8000                ║
║    Docs:  http://localhost:8000/docs           ║
║    UI:    Open frontend/index.html in browser  ║
╚══════════════════════════════════════════════════╝
```

### 4. Open the UI

Open `frontend/index.html` in your browser — no build step needed.

---

##  API Reference

### POST `/api/v1/ingest` — Upload a file

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{
  "success": true,
  "doc_ids": ["doc_a3f8c12b"],
  "chunk_count": 42,
  "sources": ["document.pdf"],
  "elapsed_ms": 1247.3
}
```

---

### POST `/api/v1/ingest/url` — Ingest from a URL

```bash
curl -X POST http://localhost:8000/api/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.python.org/3/tutorial/index.html"}'
```

---

### POST `/api/v1/query` — Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is retrieval-augmented generation?",
    "top_k": 5,
    "use_mmr": true
  }'
```

**Response:**
```json
{
  "query": "What is retrieval-augmented generation?",
  "answer": "Retrieval-Augmented Generation (RAG) is a technique that... [Source: rag_overview.pdf]",
  "citations": [
    {
      "source": "rag_overview.pdf",
      "chunk_id": "chunk_a1b2c3",
      "relevance_score": 0.9312,
      "excerpt": "RAG combines dense retrieval with conditional generation..."
    }
  ],
  "sources": ["rag_overview.pdf", "transformer_paper.pdf"],
  "model": "claude-sonnet-4-20250514",
  "provider": "anthropic",
  "usage": {
    "prompt_tokens": 1842,
    "completion_tokens": 387,
    "total_tokens": 2229
  },
  "performance": {
    "generation_time_ms": 1240.5
  }
}
```

---

### POST `/api/v1/query/stream` — Streaming response (SSE)

```javascript
const evtSource = new EventSource('/api/v1/query/stream');
evtSource.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.type === 'token') process.stdout.write(data.content);
  if (data.type === 'done') evtSource.close();
};
```

---

### GET `/api/v1/stats` — Index statistics

```bash
curl http://localhost:8000/api/v1/stats
```

```json
{
  "backend": "faiss",
  "total_chunks": 247,
  "dimension": 384,
  "unique_sources": 8,
  "sources": ["guide.pdf", "readme.md", "..."]
}
```

---

### DELETE `/api/v1/documents?source=<path>` — Remove a document

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents?source=guide.pdf"
```

---

### GET `/api/v1/health` — Health check

```bash
curl http://localhost:8000/api/v1/health
# {"status": "healthy", "version": "1.0.0", "provider": "anthropic", "model": "claude-sonnet-4-20250514"}
```

---

##  Python SDK Usage

You can also use the pipeline directly in Python without the API:

```python
import asyncio
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import TextChunker
from ingestion.embedding_generator import EmbeddingGenerator
from retrieval.vector_store import FAISSVectorStore
from retrieval.retriever import Retriever
from generation.llm_engine import LLMEngine

async def main():
    # 1. Load documents
    loader = DocumentLoader()
    docs = await loader.load("./my_documents/")

    # 2. Chunk
    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)

    # 3. Embed
    embedder = EmbeddingGenerator()
    embeddings = await embedder.embed_chunks(chunks)

    # 4. Index
    store = FAISSVectorStore()
    await store.add_chunks(chunks, embeddings)
    await store.save()

    # 5. Retrieve + Generate
    retriever = Retriever(vector_store=store, embedding_generator=embedder)
    result = await retriever.retrieve("What is the main topic?")

    engine = LLMEngine()
    response = await engine.generate("What is the main topic?", result)

    print(response.answer)
    print("Sources:", response.unique_sources)

asyncio.run(main())
```

---

##  Configuration Reference

All settings live in `.env`. Every value has a safe default.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` \| `ollama` |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model name |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `EMBEDDING_PROVIDER` | `local` | `local` \| `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `EMBEDDING_DIMENSION` | `384` | Must match the embedding model |
| `VECTOR_STORE_PROVIDER` | `faiss` | `faiss` \| `pinecone` |
| `FAISS_INDEX_PATH` | `./data/faiss_index` | Where FAISS saves its index |
| `CHUNK_SIZE` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `CHUNKING_STRATEGY` | `recursive` | `fixed` \| `recursive` \| `semantic` |
| `TOP_K_RESULTS` | `5` | Default retrieval count |
| `SIMILARITY_THRESHOLD` | `0.3` | Min cosine similarity to include |
| `ENABLE_CACHE` | `true` | Enable query response caching |
| `CACHE_TTL` | `300` | Cache expiry in seconds |
| `RATE_LIMIT_REQUESTS` | `60` | Requests per window per IP |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `LOG_FORMAT` | `json` | `json` (production) \| `console` (dev) |

---

##  Switching LLM Providers

Change one line in `.env` — no code changes required:

```env
# Use OpenAI instead
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key

# Use local Ollama (free, private)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
```

---

##  Switching to Pinecone

```env
VECTOR_STORE_PROVIDER=pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=rag-index
PINECONE_ENVIRONMENT=us-east-1
EMBEDDING_DIMENSION=384
```

---

##  Docker Deployment

```bash
# Build and start everything (API + Redis)
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f rag-api

# Stop
docker-compose down
```

---

##  Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ingestion.py -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# Run only fast tests (skip embedding model load)
pytest tests/test_ingestion.py::test_recursive_chunker_basic -v
```

**Test coverage:**

| Module | Tests |
|---|---|
| `document_loader` | Text, Markdown, Python files, directories, error handling |
| `chunker` | Recursive splitting, metadata preservation, unique IDs |
| `embedding_generator` | Shape validation, normalization, semantic similarity ordering |
| `vector_store` | Add/search/persist/reload, empty store, top-K limits |
| `retriever` | Full pipeline integration, MMR diversity |
| `api` | Health check, file upload, query validation, request headers |

---

##  Architecture Decisions

### Why FAISS over Pinecone by default?
FAISS runs locally with zero cost, zero latency overhead, and full data privacy. It scales to millions of vectors on a single machine. Pinecone is the right choice for multi-tenant production systems at billions of vectors.

### Why sentence-transformers over OpenAI embeddings?
`all-MiniLM-L6-v2` is free, runs locally, and produces 384-dim vectors that are excellent for semantic search. OpenAI's `text-embedding-3-small` produces better embeddings but costs money and requires an internet connection. The abstraction layer makes swapping trivial.

### Why MMR for retrieval?
Without MMR, the top-5 results might all be from the same paragraph, giving the LLM redundant context. MMR (Maximal Marginal Relevance) balances relevance to the query *and* diversity across results, using the formula:
```
MMR = λ * relevance(doc, query) - (1-λ) * max_similarity(doc, selected)
```

### Why recursive chunking by default?
Recursive chunking tries to split at paragraph → sentence → word boundaries in order, preserving semantic coherence. Fixed-size chunking is faster but may cut sentences mid-way, reducing retrieval quality.

---

##  Performance Benchmarks

Tested on a MacBook Pro M2 with `all-MiniLM-L6-v2`:

| Operation | Throughput / Latency |
|---|---|
| Embedding (local) | ~500 chunks/sec |
| FAISS indexing | ~100k vectors/sec |
| FAISS search (10k vectors) | < 1ms |
| FAISS search (1M vectors) | ~10ms |
| Full query pipeline (local) | ~200–500ms |
| Full query pipeline (with Claude) | ~1–3 seconds |

---

##  Roadmap / Phase 10 Optimizations

- [ ] **Hybrid search** — BM25 keyword + dense vector, fused with RRF
- [ ] **Re-ranking** — CrossEncoder reranker for second-stage precision
- [ ] **Semantic chunking** — Embedding-similarity-based chunk boundary detection
- [ ] **Multi-modal** — Image and table extraction from PDFs
- [ ] **Conversation memory** — Multi-turn chat with history compression
- [ ] **Eval framework** — RAGAS metrics (faithfulness, answer relevancy, context recall)
- [ ] **Auth layer** — API key management per tenant
- [ ] **Observability** — OpenTelemetry traces + Prometheus metrics

---

##  Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/hybrid-search`
3. Write tests for your changes
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request

---

## 📄 License

MIT License — free for personal and commercial use.

---

##  Acknowledgements

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) — async Python web framework
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI similarity search
- [sentence-transformers](https://www.sbert.net/) — state-of-the-art embeddings
- [Anthropic Claude](https://www.anthropic.com/) — LLM generation
- [Pydantic](https://docs.pydantic.dev/) — data validation and settings management
- [structlog](https://www.structlog.org/) — structured logging

##  Author

**Mayur S** · [@MayurS23](https://github.com/MayurS23)

---

<div align="center">

 If this project helped you learn something, please star it!

</div>

<div align="center"> Built with ❤️ by <a href="https://github.com/MayurS23">MayurS23</a> </div>
