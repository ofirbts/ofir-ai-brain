"""Configuration via environment variables."""

import os
import tomllib
from pathlib import Path
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.exceptions import ConfigurationError


def _inject_streamlit_secrets() -> None:
    """Load Streamlit secrets into os.environ so pydantic-settings can read them."""
    candidates = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            for key, value in _flatten_toml(data).items():
                if key not in os.environ and isinstance(value, str) and value:
                    os.environ[key] = value
        except Exception:
            pass
        break


# Env vars our app expects - when found in any TOML section, inject at top level
_EXPECTED_ENV_VARS = {"GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "OFIR_BRAIN_FOLDER_ID"}


def _flatten_toml(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten TOML into env vars. Handles flat keys and nested sections."""
    result: dict[str, str] = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}".upper().replace(".", "_") if prefix else key.upper()
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, str):
                    nested_key = f"{full_key}_{k}".upper()
                    result[nested_key] = v
                    if k.upper() in _EXPECTED_ENV_VARS:
                        result[k.upper()] = v  # top-level for pydantic-settings
        elif isinstance(value, str):
            result[full_key] = value
    return result


_inject_streamlit_secrets()


class Settings(BaseSettings):
    """Application settings from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google AI / Gemini
    google_api_key: Optional[str] = Field(default=None, description="Google API key for Gemini (embeddings, chat)")

    # Google Drive
    google_application_credentials: Optional[str] = Field(
        default=None, description="Path to Google credentials JSON"
    )
    google_drive_credentials_json: Optional[str] = Field(
        default=None, description="Inline Google credentials JSON (alternative to file path)"
    )
    ofir_brain_folder_id: Optional[str] = Field(
        default=None, description="Drive folder ID for /ofir_brain"
    )

    # Storage
    chroma_persist_path: str = Field(default="./db", description="Chroma persistent storage path")
    logs_dir: str = Field(default="./logs", description="Directory for JSONL query logs")

    # Logging (production: json)
    log_format: str = Field(
        default="text",
        description="Log format: text or json (json for cloud/production)",
    )

    # Embeddings (Gemini model)
    embedding_model: str = Field(
        default="models/text-embedding-004", description="Gemini embedding model name"
    )

    # Hybrid retrieval
    mmr_lambda: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="MMR diversity (0=relevance only, 1=max diversity)",
    )
    mmr_fetch_factor: int = Field(
        default=3,
        ge=1,
        le=10,
        description="MMR candidate pool size = top_k * this factor",
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF constant k (higher = less rank influence)")
    vector_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity in hybrid (1-this = BM25 weight)",
    )

    # Chunking
    chunk_size: int = Field(default=512, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks in tokens")

    # Evaluation
    evaluation_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of queries to auto-evaluate (0=never, 1=always)",
    )

    # Weekly report source files (paths relative to synced root)
    weekly_reflections_path: str = Field(
        default="weekly_reflections.md", description="Path to weekly reflections"
    )
    energy_log_path: str = Field(default="energy_log.csv", description="Path to energy log")
    projects_log_path: str = Field(default="projects_log.md", description="Path to projects log")
    opportunity_pipeline_path: str = Field(
        default="opportunity_pipeline.md",
        description="Path to opportunity pipeline (optional)",
    )
    weekly_reports_dir: str = Field(
        default="weekly_reports",
        description="Subdir under logs_dir for saved weekly reports",
    )
    trend_lookback_weeks: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Number of prior weeks to use for trend detection",
    )

    def ensure_dirs(self) -> None:
        """Create db and logs directories if they don't exist."""
        Path(self.chroma_persist_path).mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)

    def resolve_chroma_path(self) -> Path:
        """Resolve Chroma persist path to absolute for consistent persistence across restarts."""
        p = Path(self.chroma_persist_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()


def get_settings() -> Settings:
    """Get application settings (lazy singleton)."""
    return Settings()


def validate_startup_config() -> None:
    """Validate required configuration on startup. Raises ConfigurationError if invalid."""
    settings = get_settings()
    if not settings.google_api_key or not settings.google_api_key.strip():
        raise ConfigurationError(
            "GOOGLE_API_KEY is required for embeddings, evaluation, and weekly report. "
            "Set it in .env or environment.",
            details={"hint": "Get a key from https://aistudio.google.com/"},
        )
