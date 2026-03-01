"""Pytest fixtures and configuration."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_db_dir():
    """Temporary directory for Chroma and test data."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def env_overrides(temp_db_dir):
    """Set test environment variables."""
    old = {}
    vars_to_set = {
        "CHROMA_PERSIST_PATH": temp_db_dir,
        "LOGS_DIR": str(Path(temp_db_dir) / "logs"),
        "GOOGLE_API_KEY": "test-fake-key",
        "EVALUATION_SAMPLE_RATE": "0",  # Disable auto-eval in tests
    }
    for k, v in vars_to_set.items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    yield vars_to_set
    for k in vars_to_set:
        if old.get(k) is not None:
            os.environ[k] = old[k]
        else:
            os.environ.pop(k, None)
