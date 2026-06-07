"""
mcp_server.py — MCP tool server for Multimodal Search.

Exposes the ingest and search pipeline as MCP tools so any MCP-compatible
host (Claude Desktop, custom agents, etc.) can drive the system directly.

Run standalone:
    python mcp_server.py

Add to claude_desktop_config.json:
    {
      "mcpServers": {
        "multimodal_search": {
          "command": "python",
          "args": ["/absolute/path/to/mcp_server.py"]
        }
      }
    }

Import path note:
    mcp_server.py lives at the repo root; the application package is under
    multimodal-search/. We insert that directory into sys.path so the
    relative imports (config, models, pipeline.*, search.*) resolve correctly
    without needing the package to be installed.
"""

import asyncio
import json
import sys
from pathlib import Path

# ── Resolve package root ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent
_PKG_DIR = _REPO_ROOT / "multimodal_search"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

# ── Application imports (resolved after sys.path patch) ───────────────────────
from mcp.server.fastmcp import FastMCP  # noqa: E402

from multimodal_search.config import settings  # noqa: E402
from multimodal_search.pipeline.embed import embed_query  # noqa: E402
from multimodal_search.pipeline.ingest import (  # noqa: E402
    get_job,
    start_ingest_from_url,
)
from multimodal_search.search.index import INDEX_NAME  # noqa: E402
from multimodal_search.search.search import (  # noqa: E402
    get_opensearch_client,
    list_documents,
    multimodal_search,
)

# ── MCP server instance ───────────────────────────────────────────────────────
mcp = FastMCP("MultimodalSearch")

# ── Intent weights (mirrors main.py _route_intent) ───────────────────────────
_DEFAULT_WEIGHTS = {
    "video": settings.search_video_weight,
    "audio": settings.search_audio_weight,
    "text": settings.search_meta_weight,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool: search
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def search(
    query: str,
    max_results: int = 5,
    mode: str = "auto",
) -> str:
    """
    Semantically search indexed video and audio content.

    Embeds the query with Gemini Embedding 2 and runs a weighted k-NN search
    across three embedding types (visual, transcript, metadata) stored in
    OpenSearch. Results are individual temporal segments ranked by relevance.

    Args:
        query:       Natural language search query. Examples:
                       "someone explaining gradient descent"
                       "outdoor scene with bird sounds"
                       "where does the presenter mention Q3 revenue"
        max_results: Number of results to return (default 5, max 20).
        mode:        Search mode — one of:
                       "auto"   — infer intent from query keywords (default)
                       "visual" — emphasise visual content
                       "speech" — emphasise spoken words / transcript
                       "topic"  — emphasise topic and keyword metadata

    Returns:
        JSON string containing a list of matching segments with timestamps,
        titles, summaries, transcripts, and relevance scores.
    """
    max_results = min(max(1, max_results), 20)

    # Embed query and run search in a thread (both are sync operations)
    query_vec = await asyncio.to_thread(embed_query, query)

    weights = _resolve_weights(mode)

    client = get_opensearch_client()
    results = await asyncio.to_thread(
        multimodal_search,
        client,
        INDEX_NAME,
        query_vec,
        weights,
        max_results,
        None,  # document_id filter — None = search all
    )

    if not results:
        return json.dumps({"results": [], "count": 0, "query": query})

    serialised = [
        {
            "documentId": r.documentId,
            "fileName": r.fileName,
            "contentType": r.contentType,
            "segmentIndex": r.segmentIndex,
            "startSec": r.startSec,
            "endSec": r.endSec,
            "title": r.title,
            "summary": r.summary,
            "transcript": r.transcript[:400] + "…"
            if len(r.transcript) > 400
            else r.transcript,
            "keywords": r.keywords,
            "mood": r.mood,
            "score": r.score,
            "matchedVectors": r.matchedVectors,
            "thumbnailUri": r.thumbnailUri,
        }
        for r in results
    ]

    return json.dumps(
        {"results": serialised, "count": len(serialised), "query": query}, indent=2
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool: ingest_from_url
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def ingest_from_url(url: str, name: str = "") -> str:
    """
    Ingest a video or audio file from a public URL into the search index.

    Triggers the full ingest pipeline asynchronously:
      1. Download the file
      2. Detect semantic segment boundaries
      3. Chunk with FFmpeg
      4. Transcribe each segment with Gemini Flash
      5. Generate title / summary / keywords with Gemini Flash
      6. Produce video + audio + text embeddings with Gemini Embedding 2
      7. Index into OpenSearch

    The job runs in the background. Use check_ingest_status(job_id) to
    monitor progress.

    Args:
        url:  Public URL to a video or audio file (mp4, mov, mp3, wav, etc.)
        name: Human-readable label for the content (optional).
              Defaults to the filename inferred from the URL.

    Returns:
        JSON string with job_id and initial status.
    """
    resolved_name = name.strip() or url.split("/")[-1].split("?")[0] or "Untitled"

    # start_ingest_from_url is synchronous (launches a ThreadPoolExecutor internally)
    job_id = await asyncio.to_thread(start_ingest_from_url, url, resolved_name)

    return json.dumps(
        {
            "job_id": job_id,
            "fileName": resolved_name,
            "status": "downloading",
            "message": f"Ingest job started. Poll check_ingest_status('{job_id}') for progress.",
        },
        indent=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool: check_ingest_status
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def check_ingest_status(job_id: str) -> str:
    """
    Check the current status and progress of an ingest job.

    Args:
        job_id: The job_id returned by ingest_from_url.

    Returns:
        JSON string with status, progress percentage, and segment count
        (once complete). Status values:
          downloading → detecting_segments → chunking → processing
          → indexing → complete | failed
    """
    job = await asyncio.to_thread(get_job, job_id)

    if not job:
        return json.dumps({"error": f"Job '{job_id}' not found."})

    return json.dumps(job, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool: list_documents
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def list_indexed_documents() -> str:
    """
    List all media files currently indexed in the search system.

    Returns:
        JSON string with a list of documents including fileName, contentType,
        duration, segment count, and the date they were ingested.
    """
    client = get_opensearch_client()
    docs = await asyncio.to_thread(list_documents, client, INDEX_NAME)

    if not docs:
        return json.dumps({"documents": [], "count": 0})

    return json.dumps(
        {
            "documents": [d.model_dump() for d in docs],
            "count": len(docs),
        },
        indent=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_weights(mode: str) -> dict:
    """Map search mode string to embedding weight dict."""
    presets = {
        "visual": {"video": 0.75, "audio": 0.10, "text": 0.15},
        "speech": {"video": 0.10, "audio": 0.75, "text": 0.15},
        "topic": {"video": 0.20, "audio": 0.20, "text": 0.60},
    }
    return presets.get(mode, _DEFAULT_WEIGHTS)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Runs the MCP server over stdio — the host process connects via stdin/stdout.
    mcp.run()
