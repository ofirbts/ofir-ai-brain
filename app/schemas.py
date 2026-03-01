"""Pydantic request/response models for API endpoints."""

from typing import Any

from pydantic import BaseModel, Field


# --- Health ---
class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(..., description="Service status")
    vector_store: str = Field(..., description="Vector store connection status")


# --- Query ---
class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Metadata filters: source_file, category, last_updated",
    )
    debug: bool = Field(
        default=False,
        description="If true, include retrieval scores (vector, bm25, rrf, hybrid, mmr) per result",
    )


class RetrievalResult(BaseModel):
    """Single retrieval result."""

    content: str = Field(..., description="Retrieved chunk content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    score: float = Field(..., description="Final relevance score (MMR)")
    scores: dict[str, float] | None = Field(
        default=None,
        description="Per-score breakdown when debug=true",
    )


class QueryResponse(BaseModel):
    """Response for POST /query."""

    results: list[RetrievalResult] = Field(..., description="Retrieved chunks")


# --- Reindex ---
class ReindexResponse(BaseModel):
    """Response for POST /reindex."""

    indexed: int = Field(..., ge=0, description="Number of files indexed")
    skipped: int = Field(..., ge=0, description="Number of files skipped")
    message: str | None = Field(default=None, description="Optional status message")


# --- Evaluate ---
class EvaluateRequest(BaseModel):
    """Request body for POST /evaluate."""

    query: str = Field(..., min_length=1, description="Original query")
    retrieved_chunks: list[dict[str, Any]] = Field(..., min_length=1, description="Retrieved chunks to evaluate")
    ground_truth: str | None = Field(default=None, description="Optional ground truth")


class EvaluateResponse(BaseModel):
    """Response for POST /evaluate."""

    retrieval_relevance: int = Field(..., ge=0, le=10)
    answer_faithfulness: int = Field(..., ge=0, le=10)
    hallucination_risk: int = Field(..., ge=0, le=10)
    strategic_usefulness: int = Field(..., ge=0, le=10)
    model_used: str = Field(...)
    error: str | None = Field(default=None)


# --- Weekly Report ---
class WeeklyReportRequest(BaseModel):
    """Request body for POST /weekly-report."""

    sync_root: str | None = Field(default=None, description="Optional path to sync root")


class WeeklyReportResponse(BaseModel):
    """Response for POST /weekly-report."""

    report: str = Field(..., description="Generated executive summary")


# --- Weekly Strategic Intelligence ---
class WeeklyIntelligenceRequest(BaseModel):
    """Request body for POST /reports/weekly."""

    sync_root: str | None = Field(default=None, description="Optional path to sync root")
    date: str | None = Field(default=None, description="Report date (YYYY-MM-DD), default today")


class WeeklyIntelligenceResponse(BaseModel):
    """Response for POST /reports/weekly."""

    report: str = Field(..., description="Structured markdown report")
    path: str = Field(..., description="Path where report was saved")
    trends: list[str] = Field(..., description="Detected trends from report")
    sources_loaded: dict[str, bool] = Field(..., description="Which sources had data")
