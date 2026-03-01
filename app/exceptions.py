"""Structured API exceptions and error handling."""

from typing import Any


class AppError(Exception):
    """Base exception for application errors."""

    def __init__(self, message: str, code: str = "internal_error", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ConfigurationError(AppError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="configuration_error", details=details)


class VectorStoreError(AppError):
    """Raised when vector store operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="vector_store_error", details=details)


class DriveSyncError(AppError):
    """Raised when Google Drive sync fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="drive_sync_error", details=details)


class EmbeddingError(AppError):
    """Raised when embedding operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, code="embedding_error", details=details)
