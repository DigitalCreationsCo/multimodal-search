"""
pipeline/embed.py

Gemini Embedding 2 client.

Produces three 1024-dim vectors per chunk, all in the same shared
embedding space:
  - video_vec   → from the raw .mp4 bytes (visual + audio signal)
  - audio_vec   → from the Whisper transcript (speech semantics)
  - meta_vec    → from the LLM-generated metadata string (topic/entity)

Why three separate vectors?
  Searching across all three with intent-weighted routing yields
  30-40 point MRR gains over single-vector approaches (AWS Nova study, 2026).

API notes (Gemini Embedding 2, as of March 2026):
  - Model ID: gemini-embedding-2
  - Video:  MP4 / MOV, up to 128 s — passed via Google Files API
  - Text:   direct string content
  - Dims:   128 | 256 | 512 | 1024 | 2048 | 3072 (Matryoshka)
  - Pricing: ~$0.004 / 1K chars (significantly cheaper than alternatives)
"""

import base64
import logging
import pathlib
import time
from typing import List, Optional

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

# Module-level client — initialised once
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


def _embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Embed a text string using Gemini Embedding 2.

    Returns:
        List of floats (length = settings.embedding_dim).
    """
    if not text or not text.strip():
        # Return a zero vector for empty transcripts
        return [0.0] * settings.embedding_dim

    client = _get_client()

    response = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.embedding_dim,
        ),
    )
    return list(response.embeddings[0].values)


def _embed_video(video_path: str) -> List[float]:
    """
    Embed a video chunk using Gemini Embedding 2 via the Files API.

    The file is uploaded temporarily, embedded, then deleted.
    For chunks longer than the model limit (128 s), we fall back
    to text-only embedding of the transcript.

    Returns:
        List of floats (length = settings.embedding_dim).
    """
    client = _get_client()
    video_path_obj = pathlib.Path(video_path)

    if not video_path_obj.exists():
        logger.warning("Video chunk not found: %s", video_path)
        return [0.0] * settings.embedding_dim

    uploaded_file = None
    try:
        # Upload to Google Files API (files are auto-deleted after 48 h)
        uploaded_file = client.files.upload(
            file=video_path_obj,
            config={"display_name": video_path_obj.name},
        )

        # Wait for processing (video files need a moment)
        _wait_for_file_active(client, uploaded_file.name)

        response = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(
                            file_uri=uploaded_file.uri,
                            mime_type="video/mp4",
                        )
                    )
                ]
            ),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=settings.embedding_dim,
            ),
        )
        return list(response.embeddings[0].values)

    except Exception as exc:
        logger.warning("Video embedding failed for %s: %s", video_path, exc)
        return [0.0] * settings.embedding_dim

    finally:
        # Always clean up the uploaded file
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass  # Non-critical cleanup failure


def _wait_for_file_active(client: genai.Client, file_name: str, timeout: int = 60) -> None:
    """Poll until the uploaded file is ACTIVE (processed by Google)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        file_info = client.files.get(name=file_name)
        state = getattr(file_info, "state", None)
        if state and state.name == "ACTIVE":
            return
        if state and state.name == "FAILED":
            raise RuntimeError(f"File processing failed: {file_name}")
        time.sleep(2)
    raise TimeoutError(f"File {file_name} not active after {timeout}s")


# ── Public API ────────────────────────────────────────────────────────────────

def embed_video_chunk(video_path: str) -> List[float]:
    """Embed raw video bytes → captures visual + audio scene content."""
    logger.debug("Embedding video: %s", video_path)
    return _embed_video(video_path)


def embed_transcript(transcript: str) -> List[float]:
    """Embed Whisper transcript → captures speech semantics."""
    logger.debug("Embedding transcript (%d chars)", len(transcript))
    return _embed_text(transcript, task_type="RETRIEVAL_DOCUMENT")


def embed_metadata(metadata_text: str) -> List[float]:
    """Embed LLM-generated metadata → captures topic / entity signal."""
    logger.debug("Embedding metadata (%d chars)", len(metadata_text))
    return _embed_text(metadata_text, task_type="RETRIEVAL_DOCUMENT")


def embed_query(query: str) -> List[float]:
    """Embed a user search query (uses RETRIEVAL_QUERY task type)."""
    logger.debug("Embedding query: %s", query[:80])
    return _embed_text(query, task_type="RETRIEVAL_QUERY")