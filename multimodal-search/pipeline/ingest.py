"""
pipeline/ingest.py

Orchestrates the full ingest pipeline for a single file:

  detect_scenes → chunk_content → [transcribe | embed_chunk | enrich] → upsert

Chunk-level work runs concurrently up to MAX_PARALLEL_CHUNKS.
Job state is held in-memory (replace with Redis/DB for multi-process deploys).
"""

import logging
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import httpx
from config import settings

from pipeline.chunker import chunk_content
from pipeline.embed import embed_chunk, embed_metadata, embed_transcript
from pipeline.generate_metadata import generate_metadata, metadata_to_embed_string
from pipeline.scene_detect import detect_scenes
from pipeline.transcribe import transcribe

logger = logging.getLogger(__name__)

# ── In-memory job store ───────────────────────────────────────────────────────
# Replace with Redis / Postgres for horizontal scaling
_jobs: Dict[str, Dict[str, Any]] = {}


def get_job(job_id: str) -> Optional[Dict]:
    return _jobs.get(job_id)


def _update_job(job_id: str, **kwargs) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


# ── Download helper ───────────────────────────────────────────────────────────


def _download_content(url: str, dest_path: str) -> None:
    """Download source content from *url* to *dest_path* with streaming."""
    logger.info("Downloading video from %s", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
    logger.info("Download complete: %s", dest_path)


# ── Per-chunk worker ──────────────────────────────────────────────────────────


def _process_chunk(chunk: dict, content_id: str, file_name: str) -> dict:
    """
    Process a single chunk synchronously (runs in a thread pool).

    Steps:
      1. Transcribe audio → text
      2. Enrich metadata via Gemini Flash
      3. Embed content chunk, transcript, metadata (3 separate vectors)

    Returns:
        A dict ready for vector_store.upsert_segment().
    """
    idx = chunk["chunk_index"]
    start = chunk["start_time"]
    end = chunk["end_time"]

    logger.info("Processing chunk %d (%.1f–%.1f s)", idx, start, end)

    # Step 1: transcription
    transcript = transcribe(chunk["audio_path"])

    # Step 2: metadata enrichment
    metadata = generate_metadata(transcript, start, end, idx)
    metadata_string = metadata_to_embed_string(metadata)

    # Step 3: embeddings (video takes longest — do first while text embeds run)
    video_vec = embed_chunk(chunk["content_path"])
    audio_vec = embed_transcript(transcript)
    text_vec = embed_metadata(metadata_string)

    return {
        "content_id": content_id,
        "file_name": file_name,
        "chunk_index": idx,
        "start_time": start,
        "end_time": end,
        "transcript": transcript,
        "title": metadata.title,
        "summary": metadata.summary,
        "keywords": metadata.keywords,
        "mood": metadata.mood,
        "has_speech": metadata.has_speech,
        "video_embedding": video_vec,
        "audio_embedding": audio_vec,
        "text_embedding": text_vec,
        "thumbnail_path": chunk.get("thumbnail_path", ""),
    }


# ── Main ingest function ──────────────────────────────────────────────────────


def run_ingest(
    job_id: str,
    content_path: str,
    file_name: str,
    cleanup: bool = True,
) -> None:
    """
    Full ingest pipeline for a local file.
    Designed to run in a background thread.

    Args:
        job_id:      UUID string for tracking this job.
        content_path:  Path to the local file.
        file_name:  Human-readable name to store with segments.
        cleanup:     If True, delete temp chunks after indexing.
    """
    content_id = job_id  # video_id == job_id for simplicity

    try:
        _update_job(job_id, status="detecting_scenes", progress=5)

        # 1. Scene detection
        scenes = detect_scenes(content_path)
        _update_job(job_id, scene_count=len(scenes), progress=15)

        # 2. Chunking
        _update_job(job_id, status="chunking", progress=20)
        chunks = chunk_content(content_path, scenes, job_id)
        _update_job(job_id, progress=30)

        # 3. Per-chunk processing (parallel)
        _update_job(job_id, status="processing", progress=35)
        results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=settings.max_parallel_chunks) as pool:
            futures = {
                pool.submit(_process_chunk, chunk, content_id, file_name): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                try:
                    segment = future.result()
                    results.append(segment)
                except Exception as exc:
                    chunk = futures[future]
                    logger.error(
                        "Chunk %d failed: %s", chunk["chunk_index"], exc, exc_info=True
                    )
                finally:
                    completed += 1
                    progress = 35 + int((completed / len(chunks)) * 55)
                    _update_job(job_id, progress=progress)

        # 4. Upsert to Qdrant
        _update_job(job_id, status="indexing", progress=92)
        for seg in sorted(results, key=lambda s: s["chunk_index"]):
            store.upsert_segment(seg)

        _update_job(
            job_id,
            status="complete",
            progress=100,
            segment_count=len(results),
            content_id=content_id,
            completed_at=time.time(),
        )
        logger.info(
            "Ingest complete: %d segments indexed for %s", len(results), file_name
        )

    except Exception as exc:
        logger.error("Ingest failed for job %s: %s", job_id, exc, exc_info=True)
        _update_job(job_id, status="failed", error=str(exc))

    finally:
        if cleanup:
            job_dir = os.path.join(settings.temp_dir, job_id)
            if os.path.isdir(job_dir):
                shutil.rmtree(job_dir, ignore_errors=True)
            # Remove downloaded file if it's in the temp dir
            if content_path.startswith(settings.temp_dir):
                try:
                    os.remove(content_path)
                except OSError:
                    pass


def start_ingest_from_url(url: str, name: str) -> str:
    """
    Download a file from *url* and kick off the ingest pipeline.
    Returns a job_id immediately (non-blocking).
    """
    job_id = str(uuid.uuid4())
    temp_path = os.path.join(settings.temp_dir, f"{job_id}_source.mp4")

    _jobs[job_id] = {
        "job_id": job_id,
        "file_name": name,
        "source_url": url,
        "status": "downloading",
        "progress": 0,
        "created_at": time.time(),
    }

    def _run():
        try:
            _download_content(url, temp_path)
            run_ingest(job_id, temp_path, name, cleanup=True)
        except Exception as exc:
            _update_job(job_id, status="failed", error=str(exc))

    thread_pool = ThreadPoolExecutor(max_workers=1)
    thread_pool.submit(_run)
    return job_id


def start_ingest_from_file(file_path: str, name: str) -> str:
    """
    Ingest a locally-uploaded file.
    Returns a job_id immediately (non-blocking).
    """
    job_id = str(uuid.uuid4())

    _jobs[job_id] = {
        "job_id": job_id,
        "file_name": name,
        "status": "queued",
        "progress": 0,
        "created_at": time.time(),
    }

    thread_pool = ThreadPoolExecutor(max_workers=1)
    thread_pool.submit(run_ingest, job_id, file_path, name, False)
    return job_id
