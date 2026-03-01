# ofir-ai-brain - Production RAG service
# Multi-stage build for slim image

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Production stage ---
FROM python:3.11-slim

# Security: non-root user
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app -m --shell /bin/bash app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/app/.local

# Copy application
COPY app/ ./app/
COPY pyproject.toml .

# Ensure correct ownership
RUN chown -R app:app /app /home/app/.local

# Default paths for volume mounts
ENV CHROMA_PERSIST_PATH=/data
ENV LOGS_DIR=/app/logs
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/app/.local/bin:$PATH

# Create dirs (will be overridden by volume mounts)
RUN mkdir -p /data /app/logs && chown -R app:app /data /app/logs

USER app

EXPOSE 8000

# Health check: GET /health must return 200
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Single worker for concurrency-safe reindex (lock is process-local)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
