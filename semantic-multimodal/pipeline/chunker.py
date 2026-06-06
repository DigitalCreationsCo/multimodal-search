"""
pipeline/chunker.py

Extracts video segments and thumbnail frames using FFmpeg.
Each chunk preserves both the video and audio streams so the
Gemini Embedding 2 model receives the full multimodal signal.
"""

import logging
import os
import subprocess
from typing import List, Tuple

from config import settings

logger = logging.getLogger(__name__)


def _run_ffmpeg(cmd: List[str]) -> None:
    """Run an FFmpeg command, raising on non-zero exit."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr}")


def extract_chunk(
    path: str,
    output_path: str,
    start: float,
    end: float,
) -> str:
    """
    Extract a content segment from *start* to *end* seconds.

    Uses stream-copy (-c copy) when possible for speed.
    Falls back to re-encode if the chunk would be unplayable
    (e.g. no keyframe at the cut point).

    Returns:
        Path to the extracted chunk file.
    """
    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-c",
        "copy",  # no re-encode for speed
        "-avoid_negative_ts",
        "1",
        output_path,
    ]
    try:
        _run_ffmpeg(cmd)
    except RuntimeError:
        # Re-encode fallback — slower but always works
        logger.warning("Stream-copy failed, re-encoding chunk %s", output_path)
        cmd_enc = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            path,
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            output_path,
        ]
        _run_ffmpeg(cmd_enc)

    return output_path


def extract_thumbnail(
    path: str,
    output_path: str,
    timestamp: float,
    width: int = 320,
) -> str:
    """
    Extract a single JPEG frame at *timestamp* seconds.

    Returns:
        Path to the thumbnail file.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        path,
        "-vframes",
        "1",
        "-vf",
        f"scale={width}:-1",
        output_path,
    ]
    _run_ffmpeg(cmd)
    return output_path


def extract_audio(
    video_path: str,
    output_path: str,
    start: float,
    end: float,
) -> str:
    """
    Extract the audio track of a segment as a WAV file.
    Used as input to Whisper for transcription.

    Returns:
        Path to the WAV file.
    """
    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-vn",  # drop video stream
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",  # Whisper expects 16 kHz
        "-ac",
        "1",  # mono
        output_path,
    ]
    _run_ffmpeg(cmd)
    return output_path


def chunk_content(
    path: str,
    scenes: List[Tuple[float, float]],
    job_id: str,
) -> List[dict]:
    job_dir = os.path.join(settings.temp_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    chunks = []
    for idx, (start, end) in enumerate(scenes):
        prefix = os.path.join(job_dir, f"chunk_{idx:04d}")

        video_chunk = extract_chunk(path, f"{prefix}.mp4", start, end)
        audio_chunk = extract_audio(path, f"{prefix}.wav", start, end)
        thumbnail = extract_thumbnail(
            path,
            f"{prefix}.jpg",
            timestamp=start + (end - start) / 2,  # mid-scene frame
        )

        chunks.append(
            {
                "chunk_index": idx,
                "start_time": start,
                "end_time": end,
                "duration": end - start,
                "video_path": video_chunk,
                "audio_path": audio_chunk,
                "thumbnail_path": thumbnail,
            }
        )

        logger.debug("Extracted chunk %d: %.1f–%.1f s", idx, start, end)

    logger.info("Chunking complete: %d chunks in %s", len(chunks), job_dir)
    return chunks
