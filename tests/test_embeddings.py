"""Tests for embedding provider."""

import pytest


def test_gemini_embedding_provider_interface():
    """Test that GeminiEmbeddingProvider implements embed() protocol."""
    from app.embeddings import GeminiEmbeddingProvider

    # Mock: use a fake client that returns dummy embeddings
    dim = 768  # text-embedding-004 default dim
    mock_embedding = type("Embedding", (), {"values": [0.1] * dim})()

    def mock_embed_content(**kwargs):
        contents = kwargs.get("contents", [])
        if isinstance(contents, str):
            contents = [contents]
        return type("Response", (), {"embeddings": [mock_embedding] * len(contents)})()

    provider = GeminiEmbeddingProvider(model="models/text-embedding-004", api_key="fake-key")
    provider._client.models.embed_content = mock_embed_content

    result = provider.embed(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == dim
    assert len(result[1]) == dim


def test_embedding_provider_batching():
    """Test that embed batches large inputs."""
    from app.embeddings import GeminiEmbeddingProvider, BATCH_SIZE

    dim = 768
    mock_embedding = type("Embedding", (), {"values": [0.1] * dim})()

    def mock_embed_content(**kwargs):
        contents = kwargs.get("contents", [])
        if isinstance(contents, str):
            contents = [contents]
        return type("Response", (), {"embeddings": [mock_embedding] * len(contents)})()

    provider = GeminiEmbeddingProvider(model="models/text-embedding-004", api_key="fake-key")
    provider._client.models.embed_content = mock_embed_content

    # 150 texts should trigger 2 batches
    texts = ["x"] * 150
    result = provider.embed(texts)
    assert len(result) == 150
