"""Embedding provider abstraction with Gemini implementation."""

from typing import Protocol

BATCH_SIZE = 100


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of embedding vectors."""
        ...


class GeminiEmbeddingProvider:
    """Gemini text-embedding-004 embedding provider."""

    def __init__(self, model: str = "models/text-embedding-004", api_key: str | None = None):
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches of BATCH_SIZE."""
        results: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = self._client.models.embed_content(
                model=self.model,
                contents=batch,
            )
            if hasattr(response, "embeddings") and response.embeddings:
                for emb in response.embeddings:
                    vals = getattr(emb, "values", None) or (emb if isinstance(emb, (list, tuple)) else [])
                    results.append(list(vals))
            elif hasattr(response, "embedding"):
                emb = response.embedding
                vals = getattr(emb, "values", None) or (emb if isinstance(emb, (list, tuple)) else [])
                results.append(list(vals))
        return results


def get_embedding_provider(model: str | None = None, api_key: str | None = None) -> EmbeddingProvider:
    """Factory for embedding provider. Uses Gemini by default."""
    from app.config import get_settings

    settings = get_settings()
    m = model or settings.embedding_model
    key = api_key or settings.google_api_key
    return GeminiEmbeddingProvider(model=m, api_key=key)
