# Semantic Video & Audio Search — Production Prototype

A cost-effective, production-ready semantic video search system using:

- **Scene Detection**: PySceneDetect (content-aware boundary detection)
- **Chunking**: FFmpeg (scene-aligned, not fixed-length)
- **Transcription**: OpenAI Whisper (local, free)
- **Embeddings**: Gemini Embedding 2 (text + video + audio in one shared vector space)
- **Metadata**: Gemini Flash (title / summary / keywords per chunk)
- **Vector DB**: Qdrant (self-hosted, horizontally scalable)
- **API**: FastAPI (async, production-ready)
- **Frontend**: Single-file HTML/JS search UI

---

## Architecture

```
Video In (URL or Upload)
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Ingest Pipeline                                │
│                                                 │
│  1. Scene Detection  ← PySceneDetect            │
│     (ContentDetector, threshold=27.0)           │
│                                                 │
│  2. Chunking         ← FFmpeg                   │
│     (scene-aligned, 2–30s bounds)               │
│                                                 │
│  3. For each chunk (parallel):                  │
│     a. Whisper transcription → transcript text  │
│     b. Gemini Embedding 2 on video bytes        │
│        → video_vec [1024-dim]                   │
│     c. Gemini Embedding 2 on transcript text    │
│        → audio_vec [1024-dim]                   │
│     d. Gemini Flash metadata generation         │
│        → title, summary, keywords               │
│     e. Gemini Embedding 2 on metadata text      │
│        → meta_vec [1024-dim]                    │
│                                                 │
│  4. Qdrant upsert (3 named vectors per chunk)   │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Search Pipeline                                │
│                                                 │
│  1. Intent routing (visual / speech / topic)    │
│  2. Embed query → query_vec                     │
│  3. Weighted multi-vector search across Qdrant  │
│     video_vec × w1 + audio_vec × w2 +           │
│     meta_vec × w3                               │
│  4. Temporal boundary refinement                │
│  5. Return ranked segments with timestamps      │
└─────────────────────────────────────────────────┘
```

---

## Setup

### 1. Prerequisites

```bash
# System deps
brew install ffmpeg         # macOS
# OR: sudo apt-get install ffmpeg  # Ubuntu

# Python
python -m venv .venv && source .venv/bin/activate
```

### 2. Clone & Install

```bash
cd backend
pip install -r requirements.txt
```

### 3. Environment

```bash
cp .env.example .env
# Edit .env — add your Google AI API key
```

### 4. Start Qdrant

```bash
docker-compose up -d qdrant
```

### 5. Run the API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Open the frontend

Open `frontend/index.html` directly in your browser.  
Point it at `http://localhost:8000` (default).

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest/url` | Ingest video from URL |
| `POST` | `/ingest/file` | Ingest uploaded video file |
| `GET` | `/ingest/{job_id}` | Poll job status |
| `POST` | `/search` | Semantic search |
| `GET` | `/videos` | List indexed videos |
| `GET` | `/health` | Health check |

### Ingest from URL

```bash
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/video.mp4", "name": "My Video"}'
```

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

---

## Cost Estimation

| Component | Cost |
|-----------|------|
| Gemini Embedding 2 | ~$0.004 / 1K chars |
| Gemini Flash (metadata) | ~$0.075 / 1M input tokens |
| Whisper | Free (local) |
| Qdrant | Free (self-hosted) |
| PySceneDetect + FFmpeg | Free |

**Rough estimate**: A 1-hour video (~120 scenes × 3 embeddings) costs under $0.05 to ingest.

---

## Configuration Tuning

| Setting | Default | Effect |
|---------|---------|--------|
| `SCENE_THRESHOLD` | 27.0 | Lower = more scenes detected |
| `MIN_SCENE_DURATION` | 2.0s | Avoids flash-cut noise |
| `MAX_SCENE_DURATION` | 30.0s | Splits long scenes (Gemini limit) |
| `EMBEDDING_DIM` | 1024 | 512 for speed, 3072 for max accuracy |
| `WHISPER_MODEL` | `base` | `small`/`medium` for better accuracy |
| `SEARCH_VIDEO_WEIGHT` | 0.50 | Visual relevance importance |
| `SEARCH_AUDIO_WEIGHT` | 0.30 | Speech/transcript importance |
| `SEARCH_META_WEIGHT` | 0.20 | Topic/keyword importance |