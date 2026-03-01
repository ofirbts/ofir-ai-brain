# Cloud Deployment Guide

ofir-ai-brain is ready for deployment on Railway, Render, and Google Cloud Run.

## Prerequisites

- Docker (for local testing)
- Account on your chosen platform
- `GOOGLE_API_KEY` (required)
- Google Drive credentials (optional, for `/reindex`)

---

## Local Docker Run

```bash
# Build
docker build -t ofir-ai-brain .

# Run with volume for persistent Chroma DB
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_key \
  -v ofir-ai-brain-data:/data \
  -v ofir-ai-brain-logs:/app/logs \
  ofir-ai-brain
```

Health check: `curl http://localhost:8000/health`

---

## Railway

### 1. Deploy from GitHub

1. Connect your repo at [railway.app](https://railway.app)
2. Add **New Project** → **Deploy from GitHub** → select ofir-ai-brain
3. Railway auto-detects the Dockerfile

### 2. Environment Variables

In **Variables**, add:

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Gemini API key |
| `GOOGLE_DRIVE_CREDENTIALS_JSON` | No | Inline JSON for Drive (if using reindex) |
| `OFIR_BRAIN_FOLDER_ID` | No | Drive folder ID |
| `CHROMA_PERSIST_PATH` | No | Default `/data` |
| `LOGS_DIR` | No | Default `/app/logs` |
| `LOG_FORMAT` | No | `json` for production logging |

### 3. Volume (Chroma persistence)

1. **Project** → **+ New** → **Volume**
2. Mount path: `/data`
3. Attach to your service

Without a volume, the index is ephemeral and will be lost on redeploy.

### 4. Settings

- **Port**: 8000 (auto-detected)
- **Health check**: Railway uses `/health` if configured; ensure service returns 200

---

## Render

### 1. Create Web Service

1. [render.com](https://render.com) → **New** → **Web Service**
2. Connect repo, select ofir-ai-brain

### 2. Build & Run

| Field | Value |
|-------|-------|
| **Environment** | Docker |
| **Dockerfile Path** | `./Dockerfile` |

### 3. Environment Variables

Add in **Environment**:

- `GOOGLE_API_KEY` (required)
- `GOOGLE_DRIVE_CREDENTIALS_JSON` (optional, for reindex)
- `OFIR_BRAIN_FOLDER_ID` (optional)
- `LOG_FORMAT=json` (recommended)

### 4. Disk (Chroma persistence)

1. **Disks** → **Add Disk**
2. Mount path: `/data`
3. Size: 1 GB minimum

### 5. Health Check

Render checks `/` by default. Configure **Health Check Path**: `/health`

---

## Google Cloud Run

### 1. Build and Push

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and push to Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ofir-ai-brain
```

### 2. Deploy

```bash
gcloud run deploy ofir-ai-brain \
  --image gcr.io/YOUR_PROJECT_ID/ofir-ai-brain \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_API_KEY=your_key" \
  --memory 1Gi \
  --set-env-vars "CHROMA_PERSIST_PATH=/tmp/chroma"
```

### 3. Persistent Storage

Cloud Run is stateless. For persistence:

**Option A: Cloud Storage FUSE (advanced)**  
Mount a GCS bucket as a filesystem for Chroma.

**Option B: Ephemeral (simple)**  
Use `/tmp` for Chroma. Data is lost on cold start; run `/reindex` after deploy or use a startup job.

**Option C: Cloud Run Jobs**  
Run reindex as a one-off job that writes to a shared store (GCS, etc.).

### 4. Health Check

Cloud Run uses startup probes. Ensure `/health` returns 200. The Dockerfile `HEALTHCHECK` is used when running locally.

---

## Health Check Compatibility

All platforms expect:

- **Path**: `GET /health`
- **Success**: HTTP 200
- **Response**: `{"status": "ok", "vector_store": "connected" | "disconnected"}`

The service returns 200 even when `vector_store` is `disconnected` (e.g. no index yet).

---

## Concurrency-Safe Indexing

The service uses a process-local lock for `/reindex`. It runs with **1 worker** so that:

- Concurrent `POST /reindex` calls are serialized
- No race conditions during Drive sync and indexing

For multi-worker setups, use an external lock (e.g. Redis) or keep a single worker.

---

## Production Logging

Set `LOG_FORMAT=json` for structured JSON logs (timestamp, level, message). Useful for:

- Cloud Logging (GCP)
- Datadog, Logtail, etc.
- Log aggregation and search

---

## Minimal .env for Cloud

```env
GOOGLE_API_KEY=your_key
LOG_FORMAT=json
CHROMA_PERSIST_PATH=/data
```
