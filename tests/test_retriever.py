"""Tests for hybrid retriever (MMR, fusion)."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider returning fixed dim vectors."""
    provider = MagicMock()
    dim = 768  # Gemini text-embedding-004

    def embed(texts):
        return [[0.1 + hash(t) % 100 * 0.001] * dim for t in texts]

    provider.embed = embed
    return provider


@pytest.fixture
def mock_vector_store():
    """Mock vector store returning fixed results."""
    store = MagicMock()

    def query(**kwargs):
        n = min(kwargs.get("n_results", 10), 3)
        ids = ["id1", "id2", "id3"][:n]
        docs = ["Doc 1 content", "Doc 2 content", "Doc 3 content"][:n]
        metas = [{"file_name": "a.md"}, {"file_name": "b.md"}, {"file_name": "c.md"}][:n]
        dists = [0.1, 0.2, 0.3][:n]
        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }

    store.query = query
    return store


@pytest.fixture
def bm25_index_path(temp_db_dir):
    """Create a minimal BM25 index file."""
    from rank_bm25 import BM25Okapi

    docs = ["hello world", "goodbye world", "hello universe"]
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    path = Path(temp_db_dir) / "bm25_index.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": ["id1", "id2", "id3"],
            "docs": docs,
            "metadatas": [{"file_name": f"f{i}.txt"} for i in range(3)],
        }, f)
    return str(path)


def test_mmr_rerank_diversifies():
    """Test that MMR reduces redundancy (conceptual)."""
    from app.retriever import _mmr_rerank

    q_emb = [0.1] * 768
    doc_embs = [
        [0.1] * 768,
        [0.1001] * 768,
        [0.5] * 768,
    ]
    results = _mmr_rerank(
        query_embedding=q_emb,
        doc_embeddings=doc_embs,
        doc_ids=["a", "b", "c"],
        doc_contents=["A", "B", "C"],
        doc_metadatas=[{}, {}, {}],
        hybrid_scores=[1.0, 0.9, 0.8],
        top_k=2,
        lambda_=0.5,
    )
    assert len(results) == 2
    assert results[0]["id"] in ["a", "b", "c"]
    assert results[1]["id"] in ["a", "b", "c"]


def test_rrf_score():
    """Test RRF score calculation."""
    from app.retriever import _rrf_score

    assert _rrf_score([1, 1]) > _rrf_score([1, 10])
    assert _rrf_score([1]) > _rrf_score([5])


def test_hybrid_retriever_search_no_bm25(temp_db_dir, mock_embedding_provider, mock_vector_store):
    """Test retriever when BM25 index doesn't exist (dense only)."""
    from app.retriever import HybridRetriever

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        embedding_provider=mock_embedding_provider,
        bm25_index_path=str(Path(temp_db_dir) / "nonexistent.pkl"),
        mmr_lambda=0.2,
        mmr_fetch_factor=3,
        rrf_k=60,
        vector_weight=0.5,
    )
    results = retriever.search("hello", top_k=2)
    assert len(results) <= 2
    for r in results:
        assert "content" in r
        assert "metadata" in r
        assert "score" in r


def test_hybrid_retriever_search_with_bm25(
    temp_db_dir, mock_embedding_provider, mock_vector_store, bm25_index_path
):
    """Test retriever with BM25 index."""
    from app.retriever import HybridRetriever

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        embedding_provider=mock_embedding_provider,
        bm25_index_path=bm25_index_path,
        mmr_lambda=0.2,
    )
    results = retriever.search("hello world", top_k=2)
    assert len(results) <= 2
    assert all("content" in r for r in results)


def test_hybrid_retriever_debug_mode(mock_embedding_provider, mock_vector_store, bm25_index_path):
    """Test retriever debug mode returns scores."""
    from app.retriever import HybridRetriever

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        embedding_provider=mock_embedding_provider,
        bm25_index_path=bm25_index_path,
        mmr_lambda=0.2,
    )
    results = retriever.search("hello", top_k=2, debug=True)
    for r in results:
        assert "scores" in r
        assert "vector_score" in r["scores"]
        assert "bm25_score" in r["scores"]
        assert "hybrid_score" in r["scores"]
        assert "mmr_score" in r["scores"]


def test_metadata_filter_applied(mock_embedding_provider, mock_vector_store, bm25_index_path):
    """Test metadata filtering via filters param."""
    from app.retriever import HybridRetriever, _build_chroma_where

    assert _build_chroma_where({"category": "reflections"}) == {"category": "reflections"}
    assert _build_chroma_where({"source_file": "a.md"}) == {"source_file": "a.md"}
    assert _build_chroma_where({"last_updated": "2024-01-01"}) == {"last_updated": "2024-01-01"}
    assert _build_chroma_where({"unknown_key": "x"}) is None
