"""
pipeline/chunker.py

Extracts content segments and thumbnail frames using FFmpeg.
Each chunk preserves both the video and audio streams so the
Gemini Embedding 2 model receives the full multimodal signal.
"""

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from utilities import Utilities

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
        path,
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
    Used as input for transcription.

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
    media_file_path: str,
    time_segments: List[Tuple[float, float]],
    job_identifier: str,
) -> List[Dict[str, Any]]:

    if not os.path.isfile(media_file_path):
        logger.error(
            "Media file verification failed. Target missing: %s", media_file_path
        )
        raise FileNotFoundError(f"Source media file does not exist: {media_file_path}")

    try:
        media_type = Utilities.determine_media_type(media_file_path)
        logger.debug(
            "Successfully resolved media type '%s' for target: %s",
            media_type,
            media_file_path,
        )
    except Exception as err:
        logger.error("Media type classification failed: %s", str(err), exc_info=True)
        raise

    job_directory_path = os.path.join(settings.temp_dir, job_identifier)
    try:
        os.makedirs(job_directory_path, exist_ok=True)
        logger.debug("Job workspace mapped and verified: %s", job_directory_path)
    except OSError as err:
        logger.error(
            "Filesystem access denied. Could not create directory %s: %s",
            job_directory_path,
            str(err),
            exc_info=True,
        )
        raise

    extracted_chunks: List[Dict[str, Any]] = []

    for index, (start_time, end_time) in enumerate(time_segments):
        # Prevent erroneous processing of inverted or zero-duration segments
        if start_time >= end_time or start_time < 0:
            logger.warning(
                "Invalid time segment detected. Bypassing extraction. Index: %d, Start: %.2f, End: %.2f",
                index,
                start_time,
                end_time,
            )
            continue

        chunk_file_prefix = os.path.join(job_directory_path, f"chunk_{index:04d}")

        extracted_video_path: Optional[str] = None
        extracted_audio_path: Optional[str] = None
        extracted_thumbnail_path: Optional[str] = None

        try:
            # Audio extraction is universal across both video and audio source files
            extracted_audio_path = extract_audio(
                media_file_path, f"{chunk_file_prefix}.wav", start_time, end_time
            )

            # Conditional execution for video streams
            if media_type == "video":
                extracted_video_path = extract_chunk(
                    media_file_path, f"{chunk_file_prefix}.mp4", start_time, end_time
                )
                extracted_thumbnail_path = extract_thumbnail(
                    media_file_path,
                    f"{chunk_file_prefix}.jpg",
                    timestamp=start_time + (end_time - start_time) / 2.0,
                )

            chunk_metadata = {
                "chunk_index": index,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "video_path": extracted_video_path,
                "audio_path": extracted_audio_path,
                "thumbnail_path": extracted_thumbnail_path,
                "media_type": media_type,
            }
            extracted_chunks.append(chunk_metadata)
            logger.debug(
                "Extraction cycle completed successfully for chunk %d: %.1f-%.1f s",
                index,
                start_time,
                end_time,
            )

        except Exception as err:
            logger.error(
                "Extraction cycle failure at chunk %d (%.1f-%.1f) originating from %s. Trace: %s",
                index,
                start_time,
                end_time,
                media_file_path,
                str(err),
                exc_info=True,
            )
            # Re-raise to ensure atomic failure rather than silent corruption of the job state.
            raise RuntimeError(
                f"Critical execution failure during extraction of chunk {index}"
            ) from err

    logger.info(
        "Batch chunking operation completed. Total chunks processed: %d. Workspace: %s",
        len(extracted_chunks),
        job_directory_path,
    )

    return extracted_chunks
