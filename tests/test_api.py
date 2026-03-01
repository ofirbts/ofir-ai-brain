"""API tests with TestClient."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(temp_db_dir, env_overrides):
    """FastAPI test client with temp db."""
    from app.main import app

    return TestClient(app)


# --- Health ---
def test_health_returns_ok(client: TestClient):
    """Test GET /health returns 200 with status and vector_store."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["vector_store"] in ("connected", "disconnected")


def test_health_response_model(client: TestClient):
    """Test health response has correct schema."""
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()
    assert "vector_store" in r.json()
    assert len(r.json()) == 2


# --- Query ---
def test_query_valid(client: TestClient):
    """Test POST /query with valid request returns results list."""
    r = client.post("/query", json={"query": "hello world", "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert isinstance(data["results"], list)


def test_query_includes_timing_header(client: TestClient):
    """Test query endpoint adds X-Response-Time-Ms header."""
    r = client.post("/query", json={"query": "test"})
    assert r.status_code == 200
    assert "X-Response-Time-Ms" in r.headers


def test_query_empty_fails_validation(client: TestClient):
    """Test POST /query with empty query returns 422."""
    r = client.post("/query", json={"query": ""})
    assert r.status_code == 422


def test_query_missing_body_fails(client: TestClient):
    """Test POST /query without body returns 422."""
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_query_with_filters(client: TestClient):
    """Test POST /query with metadata filters (source_file, category, last_updated)."""
    r = client.post(
        "/query",
        json={"query": "reflections", "top_k": 3, "filters": {"category": "reflections"}},
    )
    assert r.status_code == 200
    assert "results" in r.json()


def test_query_debug_mode(client: TestClient):
    """Test POST /query with debug=true returns scores per result."""
    r = client.post("/query", json={"query": "test", "debug": True})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    for item in data["results"]:
        assert "scores" in item
        assert "vector_score" in item["scores"]
        assert "bm25_score" in item["scores"]
        assert "hybrid_score" in item["scores"]
        assert "mmr_score" in item["scores"]


# --- Reindex ---
def test_reindex_returns_indexed_skipped(client: TestClient):
    """Test POST /reindex returns indexed and skipped counts."""
    r = client.post("/reindex")
    assert r.status_code == 200
    data = r.json()
    assert "indexed" in data
    assert "skipped" in data
    assert isinstance(data["indexed"], int)
    assert isinstance(data["skipped"], int)


def test_reindex_idempotent(client: TestClient):
    """Test POST /reindex called twice returns consistent structure."""
    r1 = client.post("/reindex")
    r2 = client.post("/reindex")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert set(r1.json().keys()) == set(r2.json().keys())


def test_reindex_includes_timing_header(client: TestClient):
    """Test reindex endpoint adds X-Response-Time-Ms header."""
    r = client.post("/reindex")
    assert r.status_code == 200
    assert "X-Response-Time-Ms" in r.headers


# --- Evaluate (optional, ensures no unhandled exceptions) ---
def test_evaluate_valid(client: TestClient):
    """Test POST /evaluate with valid body."""
    r = client.post(
        "/evaluate",
        json={
            "query": "test",
            "retrieved_chunks": [{"content": "relevant text", "metadata": {}}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "retrieval_relevance" in data
    assert "answer_faithfulness" in data
    assert "hallucination_risk" in data
    assert "strategic_usefulness" in data


def test_evaluate_empty_chunks_fails(client: TestClient):
    """Test POST /evaluate with empty chunks returns 422."""
    r = client.post(
        "/evaluate",
        json={"query": "test", "retrieved_chunks": []},
    )
    assert r.status_code == 422


# --- Metrics ---
def test_metrics_summary(client: TestClient):
    """Test GET /metrics/summary returns aggregates."""
    r = client.get("/metrics/summary")
    assert r.status_code == 200
    data = r.json()
    assert "total_queries" in data
    assert "latency_ms" in data
    assert "token_usage" in data
    assert "judge_scores" in data or data.get("total_evaluations") == 0


# --- Weekly report ---
def test_weekly_report(client: TestClient):
    """Test POST /weekly-report returns report string."""
    r = client.post("/weekly-report", json={})
    assert r.status_code == 200
    data = r.json()
    assert "report" in data
    assert isinstance(data["report"], str)


# --- Weekly Strategic Intelligence ---
def test_weekly_intelligence(client: TestClient):
    """Test POST /reports/weekly returns structured report with path and trends."""
    r = client.post("/reports/weekly", json={})
    assert r.status_code == 200
    data = r.json()
    assert "report" in data
    assert "path" in data
    assert "trends" in data
    assert "sources_loaded" in data
    assert isinstance(data["report"], str)
    assert isinstance(data["trends"], list)
    assert "reflections" in data["sources_loaded"]
    assert "energy" in data["sources_loaded"]
    assert "projects" in data["sources_loaded"]
    assert "opportunities" in data["sources_loaded"]
