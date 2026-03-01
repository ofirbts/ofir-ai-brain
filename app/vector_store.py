"""Chroma vector store with persistent storage and metadata filtering."""

from pathlib import Path
from typing import Any

import chromadb

COLLECTION_NAME = "ofir_brain"


class VectorStore:
    """Chroma-based vector store with metadata support."""

    def __init__(self, persist_path: str = "./db"):
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Add documents to the collection."""
        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        self._collection.add(**kwargs)

    def query(
        self,
        query_embeddings: list[list[float]] | None = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Query the collection. Provide either query_embeddings or query_texts."""
        kwargs: dict[str, Any] = {"n_results": n_results}
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        elif query_texts is not None:
            kwargs["query_texts"] = query_texts
        else:
            raise ValueError("Must provide query_embeddings or query_texts")
        if where is not None:
            kwargs["where"] = where
        if include is not None:
            kwargs["include"] = include
        return self._collection.query(**kwargs)

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        self._collection.delete(ids=ids)

    def count(self) -> int:
        """Return number of documents in the collection."""
        return self._collection.count()

    def get_all_ids(self) -> list[str]:
        """Get all document IDs in the collection."""
        result = self._collection.get(include=[])
        return result["ids"]

    def get_all_documents(self) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Get all ids, documents, and metadatas. For BM25 index rebuild."""
        result = self._collection.get(include=["documents", "metadatas"])
        return result["ids"], result["documents"] or [], result["metadatas"] or []


def get_vector_store(persist_path: str | Path | None = None) -> VectorStore:
    """Factory for vector store. Uses absolute path for persistence across restarts."""
    from pathlib import Path

    from app.config import get_settings

    settings = get_settings()
    if persist_path is not None:
        path = Path(persist_path).resolve() if not isinstance(persist_path, Path) else persist_path
    else:
        path = settings.resolve_chroma_path()
    return VectorStore(persist_path=str(path))
