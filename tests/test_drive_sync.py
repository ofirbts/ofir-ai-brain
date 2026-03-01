"""Tests for Drive sync (mocked)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_chunk_text():
    """Test chunking with tiktoken."""
    from app.drive_sync import _chunk_text

    text = "Hello world. " * 100  # ~300 tokens
    chunks = _chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) >= 2
    assert "Hello" in chunks[0]


def test_tokenize_for_bm25():
    """Test BM25 tokenization."""
    from app.drive_sync import _tokenize_for_bm25

    tokens = _tokenize_for_bm25("Hello World! 123")
    assert "hello" in tokens
    assert "world" in tokens


def test_sync_state_save_load(temp_db_dir):
    """Test sync state persistence."""
    from app.drive_sync import _load_sync_state, _save_sync_state

    state = {"file1": {"hash": "abc", "modified": "2024-01-01"}}
    _save_sync_state(temp_db_dir, state)
    loaded = _load_sync_state(temp_db_dir)
    assert loaded == state


def test_infer_category():
    """Test category inference from path."""
    from app.drive_sync import _infer_category

    assert _infer_category("weekly_reflections.md") == "reflections"
    assert _infer_category("energy_log.csv") == "energy"
    assert _infer_category("projects_log.md") == "projects"
    assert _infer_category("random.txt") == "general"


def test_write_local_copy(temp_db_dir):
    """Test _write_local_copy persists file content."""
    from app.drive_sync import FileInfo, _write_local_copy

    fi = FileInfo(
        id="x",
        name="weekly_reflections.md",
        mime_type="text/markdown",
        modified_time="",
        content_hash="abc",
        parent_path="",
    )
    _write_local_copy(temp_db_dir, fi, "# Reflection\n\nGood week.")
    path = Path(temp_db_dir) / "ofir_brain" / "weekly_reflections.md"
    assert path.exists()
    assert "Good week" in path.read_text()
