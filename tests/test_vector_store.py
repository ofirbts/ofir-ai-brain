"""Tests for Chroma vector store."""

import pytest


@pytest.fixture
def vector_store(temp_db_dir):
    """Create vector store with temp path."""
    from app.vector_store import VectorStore

    return VectorStore(persist_path=temp_db_dir)


def test_vector_store_add_and_query(vector_store):
    """Test add and query operations."""
    ids = ["doc1", "doc2"]
    docs = ["Hello world", "Goodbye world"]
    metas = [{"file_name": "a.txt"}, {"file_name": "b.txt"}]
    # Use dummy embeddings (Chroma can compute if we pass documents)
    emb1 = [0.1] * 768  # Gemini text-embedding-004 dim
    emb2 = [0.2] * 768
    embeddings = [emb1, emb2]

    vector_store.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    result = vector_store.query(
        query_embeddings=[emb1],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )
    assert len(result["ids"][0]) >= 1
    assert "Hello" in (result["documents"][0][0] or "")


def test_vector_store_metadata_filter(vector_store):
    """Test metadata filtering."""
    ids = ["d1", "d2", "d3"]
    docs = ["A", "B", "C"]
    metas = [
        {"category": "reflections", "file_name": "r.md"},
        {"category": "energy", "file_name": "e.csv"},
        {"category": "reflections", "file_name": "r2.md"},
    ]
    dim = 768  # Gemini text-embedding-004
    embeddings = [[0.1] * dim, [0.2] * dim, [0.3] * dim]
    vector_store.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    result = vector_store.query(
        query_embeddings=[[0.1] * dim],
        n_results=10,
        where={"category": "reflections"},
    )
    assert len(result["ids"][0]) == 2
    for m in result.get("metadatas", [[]])[0] or []:
        assert m.get("category") == "reflections"


def test_vector_store_delete(vector_store):
    """Test delete operation."""
    vector_store.add(
        ids=["x"],
        documents=["test"],
        metadatas=[{}],
        embeddings=[[0.1] * 768],
    )
    assert vector_store.count() == 1
    vector_store.delete(["x"])
    assert vector_store.count() == 0


def test_vector_store_get_all_documents(vector_store):
    """Test get_all_documents for BM25 rebuild."""
    vector_store.add(
        ids=["a", "b"],
        documents=["doc a", "doc b"],
        metadatas=[{"f": "1"}, {"f": "2"}],
        embeddings=[[0.1] * 768, [0.2] * 768],
    )
    ids, docs, metas = vector_store.get_all_documents()
    assert ids == ["a", "b"]
    assert "doc a" in docs[0]
    assert metas[0]["f"] == "1"


def test_vector_store_persistence_across_restarts(temp_db_dir):
    """Test that Chroma data persists when store is recreated (simulates server restart)."""
    from app.vector_store import VectorStore

    # First instance: add data
    vs1 = VectorStore(persist_path=temp_db_dir)
    vs1.add(
        ids=["persist_test"],
        documents=["Survives restart"],
        metadatas=[{"test": "true"}],
        embeddings=[[0.1] * 768],
    )
    assert vs1.count() == 1
    del vs1  # Simulate process exit

    # Second instance: same path (like server restart)
    vs2 = VectorStore(persist_path=temp_db_dir)
    assert vs2.count() == 1
    result = vs2.query(query_embeddings=[[0.1] * 768], n_results=1)
    assert result["ids"][0]
    assert "Survives restart" in (result["documents"][0][0] or "")
