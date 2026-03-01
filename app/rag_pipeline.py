"""RAG pipeline: retrieval-only (no LLM generation)."""

from typing import Any


def query(
    query_text: str,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant chunks for a query.
    Returns list of {content, metadata, score, scores?}.
    When debug=True, each result includes scores: {vector_score, bm25_score, rrf_score, hybrid_score, mmr_score}.
    """
    from app.retriever import get_retriever

    retriever = get_retriever()
    return retriever.search(query=query_text, top_k=top_k, filters=filters, debug=debug)
