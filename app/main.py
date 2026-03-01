"""FastAPI application for ofir-ai-brain."""

from pathlib import Path
import sys

# Ensure project root is on Python path (for Streamlit Cloud, Docker, etc.)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings, validate_startup_config
from app.exceptions import AppError, ConfigurationError
from app.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
    RetrievalResult,
    WeeklyIntelligenceRequest,
    WeeklyIntelligenceResponse,
    WeeklyReportRequest,
    WeeklyReportResponse,
)

_reindex_lock = Lock()


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to log request timing."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    from app.logging_config import setup_logging

    setup_logging(get_settings().log_format)
    try:
        validate_startup_config()
    except ConfigurationError as e:
        raise RuntimeError(f"Startup validation failed: {e.message}") from e
    get_settings().ensure_dirs()
    yield
    # Shutdown (if needed)


app = FastAPI(
    title="ofir-ai-brain",
    description="Production RAG with Drive sync, hybrid retrieval, and weekly reporting",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(TimingMiddleware)


@app.exception_handler(AppError)
def app_error_handler(request: Request, exc: AppError):
    """Handle structured application errors."""
    from fastapi.responses import JSONResponse

    status_code = 422 if isinstance(exc, ConfigurationError) else 500
    return JSONResponse(
        status_code=status_code,
        content={"error": exc.message, "code": exc.code, "details": exc.details},
    )


@app.exception_handler(Exception)
def generic_error_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions - never leak internals."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "code": "internal_error", "details": {}},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check. Returns status and vector store connectivity."""
    try:
        from app.vector_store import get_vector_store

        vs = get_vector_store()
        _ = vs.count()
        return HealthResponse(status="ok", vector_store="connected")
    except Exception:
        return HealthResponse(status="ok", vector_store="disconnected")


@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest) -> QueryResponse:
    """
    Run hybrid RAG retrieval. Supports metadata filters: source_file, category, last_updated.
    Set debug=true to see per-result scores (vector, bm25, hybrid, mmr).
    Queries are logged with full observability; a configurable fraction are auto-evaluated.
    """
    from app.evaluator import evaluate
    from app.observability import log_evaluation, log_query, should_sample_evaluation
    from app.rag_pipeline import query as rag_query

    start = time.perf_counter()
    results = rag_query(
        query_text=body.query,
        top_k=body.top_k,
        filters=body.filters,
        debug=body.debug,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    judge_scores: dict[str, Any] | None = None
    if results and should_sample_evaluation():
        chunks_for_eval = [{"content": r.get("content", ""), "metadata": r.get("metadata", {})} for r in results]
        eval_result = evaluate(body.query, chunks_for_eval)
        judge_scores = {k: v for k, v in eval_result.items() if k != "error" and isinstance(v, (int, float))}
        log_evaluation(
            query=body.query,
            chunk_ids=[r.get("chunk_id", "") for r in results],
            scores=eval_result,
        )

    log_query(
        query=body.query,
        top_k=body.top_k,
        filters=body.filters,
        results=results,
        latency_ms=elapsed_ms,
        judge_scores=judge_scores,
    )

    retrieval_results = [
        RetrievalResult(
            content=r.get("content", ""),
            metadata=r.get("metadata", {}),
            score=r.get("score", 0.0),
            scores=r.get("scores"),
        )
        for r in results
    ]
    return QueryResponse(results=retrieval_results)


@app.post("/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    """
    Trigger Drive sync and incremental reindex.
    Idempotent: multiple calls with same data yield same result.
    """
    from app.drive_sync import sync_folder

    acquired = _reindex_lock.acquire(blocking=False)
    if not acquired:
        return ReindexResponse(
            indexed=0,
            skipped=0,
            message="Reindex already in progress",
        )

    try:
        result = sync_folder()
        return ReindexResponse(indexed=result.indexed, skipped=result.skipped)
    except Exception as e:
        return ReindexResponse(
            indexed=0,
            skipped=0,
            message=f"Drive sync failed: {e!s}. Check credentials and OFIR_BRAIN_FOLDER_ID.",
        )
    finally:
        try:
            _reindex_lock.release()
        except RuntimeError:
            pass


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(body: EvaluateRequest) -> EvaluateResponse:
    """LLM-as-judge evaluation of retrieval quality."""
    from app.evaluator import evaluate

    result = evaluate(
        query=body.query,
        retrieved_chunks=body.retrieved_chunks,
        ground_truth=body.ground_truth,
    )
    return EvaluateResponse(
        retrieval_relevance=result.get("retrieval_relevance", 0),
        answer_faithfulness=result.get("answer_faithfulness", 0),
        hallucination_risk=result.get("hallucination_risk", 0),
        strategic_usefulness=result.get("strategic_usefulness", 0),
        model_used=result.get("model_used", "unknown"),
        error=result.get("error"),
    )


@app.get("/metrics/summary")
def metrics_summary() -> dict[str, Any]:
    """Return aggregated metrics from query and evaluation logs."""
    from app.metrics import get_metrics_summary

    return get_metrics_summary()


@app.post("/weekly-report", response_model=WeeklyReportResponse)
def weekly_report_endpoint(body: WeeklyReportRequest = WeeklyReportRequest()) -> WeeklyReportResponse:
    """Generate weekly executive summary from reflections, energy log, projects log."""
    from app.weekly_report import generate_weekly_report

    sync_root = body.sync_root
    report = generate_weekly_report(sync_root=sync_root)
    return WeeklyReportResponse(report=report)


@app.post("/reports/weekly", response_model=WeeklyIntelligenceResponse)
def weekly_intelligence_endpoint(
    body: WeeklyIntelligenceRequest = WeeklyIntelligenceRequest(),
) -> WeeklyIntelligenceResponse:
    """
    Generate Weekly Strategic Intelligence report.
    Parses reflections, energy log, projects log, opportunity pipeline.
    Saves to logs/weekly_reports/{date}.md.
    Detects trends across previous weeks.
    """
    from app.weekly_intelligence import run_pipeline

    result = run_pipeline(sync_root=body.sync_root, date_str=body.date)
    return WeeklyIntelligenceResponse(
        report=result["report"],
        path=result["path"],
        trends=result["trends"],
        sources_loaded=result["sources_loaded"],
    )
