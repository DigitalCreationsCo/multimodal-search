"""
main.py — FastAPI application entry point.

Routes:
  POST /ingest/url              — Ingest media from a public URL
  POST /ingest/file             — Ingest an uploaded media file
  GET  /ingest/{job_id}         — Poll ingest job status + progress
  POST /search                  — Semantic multimodal search
  GET  /documents               — List all indexed documents
  GET  /documents/{document_id} — Get full document with segments
  GET  /health                  — Liveness check
  GET  /stats                   — OpenSearch index stats
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import structlog
from multimodal_search.config import settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from multimodal_search.models import (
    DocumentListItem,
    IngestURLRequest,
    SearchRequest,
    SegmentSearchResult,
)
from multimodal_search.pipeline.embed import embed_query
from multimodal_search.pipeline.ingest import get_job, start_ingest_from_file, start_ingest_from_url
from multimodal_search.search.index import INDEX_NAME, ensure_index
from multimodal_search.search.search import get_opensearch_client, list_documents, multimodal_search

# ── Logging ───────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Startup: ensure OpenSearch index exists ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the OpenSearch index on first boot if it doesn't exist."""
    try:
        client = get_opensearch_client()
        ensure_index(client)
        logger.info("OpenSearch index ready: %s", INDEX_NAME)
    except Exception as exc:
        # Log but don't crash — OpenSearch may still be initialising
        logger.warning("OpenSearch index setup deferred: %s", exc)
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal Search",
    description="Production-grade semantic search over video and audio content.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve thumbnails from temp dir for local-storage deployments.
# In remote-storage (S3) deployments thumbnailUri points to S3 directly
# and this mount is unused — but it is harmless to keep it.
os.makedirs(settings.temp_dir, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=settings.temp_dir), name="thumbnails")


# ── Intent Router ─────────────────────────────────────────────────────────────

_SPEECH_SIGNALS = {
    "say",
    "said",
    "says",
    "talk",
    "spoke",
    "mentioned",
    "explains",
    "narrate",
    "voice",
    "audio",
    "words",
    "hear",
    "dialogue",
    "quote",
}
_VISUAL_SIGNALS = {
    "show",
    "see",
    "look",
    "appear",
    "scene",
    "visual",
    "footage",
    "clip",
    "frame",
    "watch",
    "display",
    "screen",
    "view",
}


def _route_intent(query: str, mode: str) -> dict:
    """
    Return per-embedding-type search weights based on explicit mode or query signals.

    Keys match those expected by multimodal_search():
      "video" — visual embedding weight
      "audio" — transcript embedding weight
      "text"  — generated metadata embedding weight

    Mode presets override signal detection:
      visual  — emphasise video embedding
      speech  — emphasise audio/transcript embedding
      topic   — emphasise text/metadata embedding
      auto    — infer from keywords in the query
    """
    if mode == "visual":
        return {"video": 0.75, "audio": 0.10, "text": 0.15}
    if mode == "speech":
        return {"video": 0.10, "audio": 0.75, "text": 0.15}
    if mode == "topic":
        return {"video": 0.20, "audio": 0.20, "text": 0.60}

    # Auto: count signal words and bias weights accordingly
    tokens = set(query.lower().split())
    speech_hits = len(tokens & _SPEECH_SIGNALS)
    visual_hits = len(tokens & _VISUAL_SIGNALS)

    if speech_hits > visual_hits:
        return {"video": 0.20, "audio": 0.60, "text": 0.20}
    if visual_hits > speech_hits:
        return {"video": 0.60, "audio": 0.20, "text": 0.20}

    # Balanced default — falls back to .env weights
    return {
        "video": settings.search_video_weight,
        "audio": settings.search_audio_weight,
        "text": settings.search_meta_weight,
    }


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/")
def serve_frontend():
    """Serve the single-page HTML application frontend."""
    # Adjust "frontend" and "index.html" to match your exact file structure paths
    html_path = os.path.join(".", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    raise HTTPException(
        status_code=404, detail="index.html file not found in frontend directory"
    )


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/stats")
def stats():
    """Return OpenSearch index statistics."""
    try:
        client = get_opensearch_client()
        info = client.indices.stats(index=INDEX_NAME)
        totals = info.get("_all", {}).get("total", {})
        return {
            "index": INDEX_NAME,
            "document_count": totals.get("docs", {}).get("count", 0),
            "store_size_bytes": totals.get("store", {}).get("size_in_bytes", 0),
            "status": "ok",
        }
    except Exception as exc:
        logger.warning("Could not retrieve OpenSearch stats: %s", exc)
        raise HTTPException(status_code=503, detail="OpenSearch unavailable")


# ── Ingest ────────────────────────────────────────────────────────────────────


@app.post("/ingest/url", status_code=202)
def ingest_url(req: IngestURLRequest):
    """
    Queue ingestion of a media file from a public URL.
    Returns immediately with a job_id; poll /ingest/{job_id} for progress.
    """
    name = req.name or req.url.split("/")[-1].split("?")[0] or "Untitled"
    job_id = start_ingest_from_url(req.url, name)
    return {"job_id": job_id, "status": "downloading", "fileName": name}


@app.post("/ingest/file", status_code=202)
async def ingest_file(file: UploadFile = File(...)):
    """
    Queue ingestion of a directly uploaded media file (video or audio).
    Returns immediately with a job_id; poll /ingest/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())
    content_dir = Path(settings.local_storage_base_directory) / settings.content_directory
    content_dir.mkdir(parents=True, exist_ok=True)
    dest = str(content_dir / f"{job_id}_{file.filename}")

    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    name = file.filename or "Uploaded File"
    actual_job_id = start_ingest_from_file(dest, name)
    return {"job_id": actual_job_id, "status": "queued", "fileName": name}


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str):
    """
    Poll the status of an ingest job.

    Possible status values:
      downloading → detecting_segments → chunking → processing → indexing
      → complete | failed
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job


# ── Search ────────────────────────────────────────────────────────────────────


@app.post("/search", response_model=List[SegmentSearchResult])
def search(req: SearchRequest):
    """
    Semantic search across all indexed media segments.

    Each result is a specific temporal segment (not a whole file) ranked by
    weighted similarity across three embedding types (video, audio, text).

    mode:
      auto    — Detect intent from signal words in the query (default)
      visual  — Weight the visual embedding heavily
      speech  — Weight the transcript embedding heavily
      topic   — Weight the metadata embedding heavily
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_vec = embed_query(req.query)
    weights = _route_intent(req.query, req.mode)

    logger.info(
        "Search query=%r mode=%s weights=%s limit=%d",
        req.query[:80],
        req.mode,
        weights,
        req.limit,
    )

    client = get_opensearch_client()
    return multimodal_search(
        client=client,
        index_name=INDEX_NAME,
        query_embedding=query_vec,
        weights=weights,
        top_k=req.limit,
        document_id=req.document_id,  # None = search all documents
    )


# ── Documents ─────────────────────────────────────────────────────────────────


@app.get("/documents", response_model=List[DocumentListItem])
def list_all_documents():
    """List all indexed documents with segment counts and duration."""
    client = get_opensearch_client()
    return list_documents(client, INDEX_NAME)


@app.get("/documents/{document_id}")
def get_document(document_id: str):
    """
    Return a full document record including all segment metadata.
    Embedding vectors are excluded from the response payload.
    """
    client = get_opensearch_client()

    try:
        response = client.search(
            index=INDEX_NAME,
            body={
                "query": {"term": {"documentId": document_id}},
                "_source": {
                    "excludes": [
                        "segments.videoEmbedding",
                        "segments.audioEmbedding",
                        "segments.textEmbedding",
                    ]
                },
                "size": 1,
            },
        )
    except Exception as exc:
        logger.error("OpenSearch error fetching document %s: %s", document_id, exc)
        raise HTTPException(status_code=503, detail="Search backend unavailable")

    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        raise HTTPException(
            status_code=404, detail=f"Document {document_id!r} not found"
        )

    return hits[0]["_source"]


if __name__ == "__main__":
    import uvicorn

    # Make sure 'uvicorn' is added to your project dependencies
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
