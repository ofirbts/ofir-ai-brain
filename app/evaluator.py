"""LLM-as-judge evaluation: retrieval relevance, answer faithfulness, hallucination risk, strategic usefulness."""

from typing import Any


EVALUATION_PROMPT = """You are evaluating a retrieval system for a personal knowledge base ("ofir_brain").

Given:
- Query: {query}
- Retrieved chunks: {chunks}
- Ground truth (optional): {ground_truth}

Score each dimension from 1 to 10 (10 = best):

1. RETRIEVAL_RELEVANCE: Do the retrieved chunks directly address or support answering the query? (1=not relevant, 10=highly relevant)
2. ANSWER_FAITHFULNESS: If one answered using only these chunks, would the answer stay faithful to the source material? (1=would fabricate, 10=fully grounded)
3. HALLUCINATION_RISK: How likely would someone hallucinate or make unsupported claims using only these chunks? (1=high risk, 10=low risk)
4. STRATEGIC_USEFULNESS: How useful are these chunks for strategic planning and decision-making? (1=not useful, 10=highly actionable)

Respond ONLY with valid JSON in this exact format:
{{"retrieval_relevance": <int>, "answer_faithfulness": <int>, "hallucination_risk": <int>, "strategic_usefulness": <int>}}
"""


def evaluate(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    ground_truth: str | None = None,
) -> dict[str, Any]:
    """
    LLM-as-judge evaluation. Returns:
    {retrieval_relevance, answer_faithfulness, hallucination_risk, strategic_usefulness, model_used}
    """
    import json
    import re

    from google import genai

    from app.config import get_settings

    settings = get_settings()
    if not settings.google_api_key:
        return _empty_scores("GOOGLE_API_KEY not set")

    chunks_text = "\n\n---\n\n".join(
        f"[{i+1}] {c.get('content', '')[:800]}" for i, c in enumerate(retrieved_chunks[:10])
    )
    gt = ground_truth or "N/A"

    prompt = EVALUATION_PROMPT.format(
        query=query,
        chunks=chunks_text or "(no chunks)",
        ground_truth=gt,
    )

    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={"temperature": 0},
    )
    content = (response.text or "") if hasattr(response, "text") else str(response)

    try:
        match = re.search(r"\{[^{}]*\}", content)
        if match:
            data = json.loads(match.group())
            return {
                "retrieval_relevance": int(data.get("retrieval_relevance", 0)),
                "answer_faithfulness": int(data.get("answer_faithfulness", 0)),
                "hallucination_risk": int(data.get("hallucination_risk", 0)),
                "strategic_usefulness": int(data.get("strategic_usefulness", 0)),
                "model_used": "gemini-2.0-flash",
            }
    except (json.JSONDecodeError, ValueError):
        pass

    return _empty_scores(f"Parse failed: {content[:200]}")


def _empty_scores(error: str) -> dict[str, Any]:
    return {
        "retrieval_relevance": 0,
        "answer_faithfulness": 0,
        "hallucination_risk": 0,
        "strategic_usefulness": 0,
        "model_used": "gemini-2.0-flash",
        "error": error,
    }
