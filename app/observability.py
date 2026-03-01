"""Enterprise observability: query logging, evaluation logging, token estimation."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for typical text."""
    return max(1, len(text) // 4)


def log_query(
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
    results: list[dict[str, Any]],
    latency_ms: float,
    logs_dir: str | None = None,
    judge_scores: dict[str, Any] | None = None,
) -> None:
    """
    Append a query log entry with full observability:
    - latency
    - retrieved chunk ids
    - similarity scores (mmr/final score per chunk)
    - token usage (estimated)
    - judge scores (when evaluated)
    """
    from app.config import get_settings

    settings = get_settings()
    log_dir = Path(logs_dir or settings.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    today = datetime.utcnow().strftime("%Y%m%d")
    log_file = log_dir / f"queries_{today}.jsonl"

    chunk_ids = [r.get("chunk_id", "") for r in results]
    similarity_scores = [r.get("score", 0) for r in results]
    query_tokens = _estimate_tokens(query)
    result_tokens = sum(_estimate_tokens(r.get("content", "") or "") for r in results)
    total_tokens = query_tokens + result_tokens

    entry: dict[str, Any] = {
        "timestamp": timestamp,
        "query": query,
        "top_k": top_k,
        "filters": filters or {},
        "latency_ms": round(latency_ms, 2),
        "chunk_ids": chunk_ids,
        "similarity_scores": [round(s, 4) for s in similarity_scores],
        "token_usage": {
            "query_tokens": query_tokens,
            "result_tokens": result_tokens,
            "total_tokens": total_tokens,
        },
        "results": [
            {
                "chunk_id": r.get("chunk_id"),
                "content_preview": (r.get("content", "") or "")[:500],
                "metadata": r.get("metadata", {}),
                "score": r.get("score", 0),
            }
            for r in results
        ],
    }
    if judge_scores:
        entry["judge_scores"] = judge_scores

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_evaluation(
    query: str,
    chunk_ids: list[str],
    scores: dict[str, Any],
    logs_dir: str | None = None,
) -> None:
    """Append evaluation result to logs/evaluations.jsonl."""
    from app.config import get_settings

    settings = get_settings()
    log_dir = Path(logs_dir or settings.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "evaluations.jsonl"

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "chunk_ids": chunk_ids,
        "retrieval_relevance": scores.get("retrieval_relevance", 0),
        "answer_faithfulness": scores.get("answer_faithfulness", 0),
        "hallucination_risk": scores.get("hallucination_risk", 0),
        "strategic_usefulness": scores.get("strategic_usefulness", 0),
        "model_used": scores.get("model_used", ""),
        "error": scores.get("error"),
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def should_sample_evaluation() -> bool:
    """Return True with probability = evaluation_sample_rate."""
    from app.config import get_settings

    return random.random() < get_settings().evaluation_sample_rate
