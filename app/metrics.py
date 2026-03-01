"""Metrics aggregation from query and evaluation logs."""

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file, return list of parsed objects."""
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def get_metrics_summary(logs_dir: str | Path | None = None) -> dict[str, Any]:
    """
    Aggregate metrics from logs/queries_*.jsonl and logs/evaluations.jsonl.
    Returns summary with totals, averages, percentiles.
    """
    from app.config import get_settings

    log_dir = Path(logs_dir or get_settings().logs_dir)
    if not log_dir.exists():
        return _empty_summary()

    query_files = sorted(log_dir.glob("queries_*.jsonl"))
    eval_file = log_dir / "evaluations.jsonl"

    all_queries: list[dict[str, Any]] = []
    for f in query_files:
        all_queries.extend(_read_jsonl(f))

    all_evals = _read_jsonl(eval_file)

    total_queries = len(all_queries)
    if total_queries == 0:
        return _empty_summary()

    latencies = [q.get("latency_ms", 0) for q in all_queries if "latency_ms" in q]
    total_tokens = sum(
        q.get("token_usage", {}).get("total_tokens", 0) for q in all_queries
    )
    queries_with_judge = [q for q in all_queries if "judge_scores" in q]

    def safe_avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def p95(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        idx = int(len(s) * 0.95) - 1
        return s[max(0, idx)]

    summary: dict[str, Any] = {
        "total_queries": total_queries,
        "total_evaluations": len(all_evals),
        "queries_with_judge": len(queries_with_judge),
        "latency_ms": {
            "avg": round(safe_avg(latencies), 2),
            "p95": round(p95(latencies), 2),
            "min": round(min(latencies), 2) if latencies else 0,
            "max": round(max(latencies), 2) if latencies else 0,
        },
        "token_usage": {
            "total": total_tokens,
            "avg_per_query": round(total_tokens / total_queries, 1) if total_queries else 0,
        },
    }

    if all_evals:
        rr = [e.get("retrieval_relevance", 0) for e in all_evals]
        af = [e.get("answer_faithfulness", 0) for e in all_evals]
        hr = [e.get("hallucination_risk", 0) for e in all_evals]
        su = [e.get("strategic_usefulness", 0) for e in all_evals]
        summary["judge_scores"] = {
            "retrieval_relevance": {"avg": round(safe_avg(rr), 2), "count": len(rr)},
            "answer_faithfulness": {"avg": round(safe_avg(af), 2), "count": len(af)},
            "hallucination_risk": {"avg": round(safe_avg(hr), 2), "count": len(hr)},
            "strategic_usefulness": {"avg": round(safe_avg(su), 2), "count": len(su)},
        }
    else:
        summary["judge_scores"] = None

    if queries_with_judge:
        jr = [q["judge_scores"].get("retrieval_relevance", 0) for q in queries_with_judge]
        summary["inline_judge_avg"] = {
            "retrieval_relevance": round(safe_avg(jr), 2),
        }

    return summary


def _empty_summary() -> dict[str, Any]:
    return {
        "total_queries": 0,
        "total_evaluations": 0,
        "queries_with_judge": 0,
        "latency_ms": {"avg": 0, "p95": 0, "min": 0, "max": 0},
        "token_usage": {"total": 0, "avg_per_query": 0},
        "judge_scores": None,
    }
