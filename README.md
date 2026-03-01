# ofir-ai-brain

Production-ready RAG system with Google Drive sync, hybrid retrieval (Chroma + BM25 + MMR), and weekly reporting.

## Features

- **FastAPI** endpoints: `POST /query`, `POST /reindex`, `GET /health`
- **Google Drive sync** with hash-based change detection for `/ofir_brain`
- **Embeddings**: Gemini text-embedding-004 (abstract provider for swap)
- **Vector store**: Chroma with persistent storage and metadata filtering
- **Hybrid retrieval**: Dense embeddings + BM25 fallback + MMR reranking
- **Observability**: JSONL query logs with scores and metadata
- **Evaluation**: LLM-as-judge scoring (relevance, hallucination risk, strategic alignment)
- **Weekly report**: Executive summary from reflections, energy log, projects log

## Setup

1. Copy `.env.example` to `.env` and fill in secrets
2. Create `db/` and `logs/` directories (or they are created on first run)
3. Install: `pip install -r requirements.txt`
4. Run: `uvicorn app.main:app --reload`

## Environment Variables

See `.env.example` for all options.
# ofir-ai-brain
