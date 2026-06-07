# Semantic Video & Audio Search — Production Prototype

A cost-effective, production-ready semantic multimedia search system. Every service — including FFmpeg — runs in Docker with a single command.

| Component | Technology | Notes |
|-----------|-----------|-------|
| Scene Detection | PySceneDetect | Content-aware boundary detection, no GPU needed |
| Chunking | FFmpeg (in Docker) | Scene-aligned clips, 2–30 s bounds |
| Transcription | Gemini Flash | Cloud-native audio/video understanding, no local model |
| Embeddings | Gemini Embedding 2 | Text + video + audio in one shared vector space |
| Metadata | Gemini Flash | Title / summary / keywords per chunk |
| Vector DB | Qdrant (Docker) | Self-hosted, horizontally scalable |
| API | FastAPI | Async, production-ready |
| Frontend | Single-file HTML/JS | Open directly in your browser |

---

## Architecture

```
Video In (URL or Upload)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Ingest Pipeline                                       │
│                                                        │
│  1. Scene Detection  ← PySceneDetect                   │
│     ContentDetector, threshold=27.0                    │
│                                                        │
│  2. Chunking         ← FFmpeg (Docker)                 │
│     Scene-aligned clips, 2–30 s bounds                 │
│                                                        │
│  3. For each chunk (parallel, up to MAX_PARALLEL):     │
│     a. Gemini Flash    → transcript text               │
│     b. Gemini Embed 2  on video clip  → video_vec      │
│     c. Gemini Embed 2  on transcript  → audio_vec      │
│     d. Gemini Flash    → title, summary, keywords      │
│     e. Gemini Embed 2  on metadata    → meta_vec       │
│                                                        │
│  4. Qdrant upsert — 3 named vectors per chunk          │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Search Pipeline                                       │
│                                                        │
│  1. Intent routing  (visual / speech / topic / auto)   │
│  2. Embed query     → query_vec via Gemini Embed 2     │
│  3. Weighted fusion across Qdrant named indexes:       │
│       video_vec × w1 + audio_vec × w2 + meta_vec × w3 │
│  4. Temporal boundary refinement                       │
│  5. Return ranked segments with timestamps             │
└────────────────────────────────────────────────────────┘
```

---

## Quick Start — Docker (recommended)

All you need installed on your host is **Docker Desktop** (or Docker Engine + Compose plugin).

### 1. Copy and fill in your `.env`

```bash
cp .env.example .env
```

Open `.env` and set your `GOOGLE_API_KEY`. Everything else has sensible defaults.

> **Note:** Do not set `QDRANT_URL` in your `.env` — docker-compose automatically injects
> `http://qdrant:6333` so the API container always finds Qdrant by its service name.

### 2. Build and start all services

```bash
docker compose up --build
```

This starts two containers:

| Container | Port | Purpose |
|-----------|------|---------|
| `ms_qdrant` | 6333 | Qdrant REST API + dashboard |
| `ms_api` | 8000 | FastAPI backend (includes FFmpeg) |

On first run `--build` compiles the image. Subsequent starts are fast:

```bash
docker compose up
```

### 3. Open the frontend

Open `frontend/index.html` directly in your browser (no server required).
The API endpoint field defaults to `http://localhost:8000`.

### 4. Verify everything is healthy

```bash
# API health
curl http://localhost:8000/health

# Qdrant dashboard
open http://localhost:6333/dashboard
```

### Useful commands

```bash
# Follow logs from both services
docker compose logs -f

# Follow API logs only
docker compose logs -f api

# Stop without losing data
docker compose stop

# Destroy containers AND volumes (wipes all indexed data)
docker compose down -v

# Rebuild after code changes
docker compose up --build api
```

---

## Manual Setup (development / no Docker)

Use this path if you need to iterate on code without rebuilding Docker images.

### Prerequisites

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg libgl1 libglib2.0-0
```

### Install Python deps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Start Qdrant only

```bash
docker compose up -d qdrant
```

### Run the API locally

```bash
# QDRANT_URL must point to localhost when running outside Docker
QDRANT_URL=http://localhost:6333 uvicorn main:app --reload --port 8000
```

### Using S3

Remote object storage can be used to store and content, embeddings, metadata and documents. There are some extra steps to configure s3 storage.

#### Install dependencies

uv sync --extra s3

#### Configure s3 variables

| Variable | Default | Effect |
|----------|---------|--------|
| `S3_REGION` | *(required)* | S3 Region |
| `S3_BUCKET` | *(required)* | S3 Bucket name |
| `S3_SOURCE_PREFIX` | *(required)* | S3 Source Content Prefix |
| `CONTENT_DIRECTORY` | *(required)* | Content Destination Prefix |
| `EMBEDDINGS_DIRECTORY` | *(required)* | Embeddings Destination Prefix |
| `METADATA_DIRECTORY` | *(required)* | Metadata Destination Prefix |
| `DOCUMENTS_DIRECTORY` | *(required)* | Documents Destination Prefix |

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest/url` | Ingest video from a public URL |
| `POST` | `/ingest/file` | Ingest an uploaded video file |
| `GET` | `/ingest/{job_id}` | Poll ingest job status + progress |
| `POST` | `/search` | Semantic search |
| `GET` | `/videos` | List all indexed videos |
| `GET` | `/videos/{video_id}` | List all segments for a video |
| `GET` | `/health` | Liveness check |
| `GET` | `/stats` | Qdrant collection stats |

### Ingest from URL

```bash
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/video.mp4", "name": "My Video"}'
```

Response:
```json
{ "job_id": "3fa85f64-...", "status": "downloading", "video_name": "My Video" }
```

### Poll job status

```bash
curl http://localhost:8000/ingest/3fa85f64-...
```

Response:
```json
{
  "job_id": "3fa85f64-...",
  "status": "processing",
  "progress": 62,
  "scene_count": 14,
  "video_name": "My Video"
}
```

Possible `status` values: `downloading` → `detecting_scenes` → `chunking` → `processing` → `indexing` → `complete` | `failed`

### Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "someone explaining machine learning concepts",
    "limit": 5,
    "mode": "auto"
  }'
```

`mode` options:

| Mode | Vector weights | Use when |
|------|---------------|----------|
| `auto` | Inferred from query text | Default — works for most queries |
| `visual` | video 75 / audio 10 / meta 15 | Describe what's on screen |
| `speech` | video 10 / audio 75 / meta 15 | Find what someone said |
| `topic` | video 20 / audio 20 / meta 60 | Broad subject / keyword queries |

---

## MCP Server

The included `mcp_server.py` exposes `search_video` and `ingest_url` as MCP tools so any MCP-compatible agent (Claude Desktop, etc.) can drive the pipeline directly.

```bash
# Run standalone
python mcp_server.py
```

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "multimodal_search": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

---

## Cost Estimation

| Component | Cost |
|-----------|------|
| Gemini Embedding 2 | ~$0.004 / 1K chars |
| Gemini Flash (transcription + metadata) | ~$0.075 / 1M input tokens |
| Qdrant | Free (self-hosted) |
| PySceneDetect + FFmpeg | Free |

**Rough estimate:** A 1-hour video (~120 scenes × 3 embeddings + transcription) costs under **$0.07** to ingest end-to-end.

---

## Configuration Reference

All settings live in `.env`. Docker Compose injects `QDRANT_URL` automatically — do not override it there.

| Variable | Default | Effect |
|----------|---------|--------|
| `GOOGLE_API_KEY` | *(required)* | Google AI Studio or Vertex key |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Embedding model |
| `GEMINI_FLASH_MODEL` | `gemini-2.0-flash-lite` | Transcription + metadata model |
| `EMBEDDING_DIM` | `1024` | Matryoshka dim: 128 / 256 / 512 / **1024** / 2048 / 3072 |
| `SCENE_THRESHOLD` | `27.0` | Lower = more scenes detected |
| `MIN_SCENE_DURATION` | `2.0` | Seconds — merges flash cuts |
| `MAX_SCENE_DURATION` | `30.0` | Seconds — splits long uncut sequences |
| `MAX_PARALLEL_CHUNKS` | `4` | Concurrent chunk workers |
| `SEARCH_VIDEO_WEIGHT` | `0.50` | Visual vector weight in auto mode |
| `SEARCH_AUDIO_WEIGHT` | `0.30` | Transcript vector weight in auto mode |
| `SEARCH_META_WEIGHT` | `0.20` | Metadata vector weight in auto mode |
| `COLLECTION_NAME` | `video_segments` | Qdrant collection name |
