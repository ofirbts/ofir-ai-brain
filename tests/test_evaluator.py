"""Tests for LLM-as-judge evaluator."""

from unittest.mock import MagicMock, patch

import pytest


def test_evaluate_parse_response():
    """Test that evaluate parses LLM JSON response."""
    with patch("google.genai.Client") as mock_genai_class, patch("app.config.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(google_api_key="fake-key")
        mock_client = MagicMock()
        mock_genai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = '{"retrieval_relevance": 8, "answer_faithfulness": 7, "hallucination_risk": 6, "strategic_usefulness": 5}'
        mock_client.models.generate_content.return_value = mock_response

        from app.evaluator import evaluate

        result = evaluate(
            query="What did I reflect on?",
            retrieved_chunks=[{"content": "I reflected on X", "metadata": {}}],
        )
        assert result["retrieval_relevance"] == 8
        assert result["answer_faithfulness"] == 7
        assert result["hallucination_risk"] == 6
        assert result["strategic_usefulness"] == 5
        assert result["model_used"] == "gemini-2.0-flash"


def test_evaluate_no_api_key():
    """Test evaluate when OPENAI_API_KEY not set."""
    with patch("app.config.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(google_api_key=None)

        from app.evaluator import evaluate

        result = evaluate("q", [{"content": "x"}])
        assert result["retrieval_relevance"] == 0
        assert result["answer_faithfulness"] == 0
        assert "error" in result
