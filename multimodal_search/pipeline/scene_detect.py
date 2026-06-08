"""
pipeline/scene_detect.py

Splits a media file into segments aligning with
natural content transitions. Returns a list of (start_sec, end_sec) tuples.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What "semantic" segmentation means here — and what it doesn't
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PySceneDetect's ContentDetector finds PHYSICAL edit cut points — frames where
the per-channel HSV histogram changes abruptly between consecutive frames.
In edited content (films, structured interviews, presentations) these cuts
are placed by the editor at semantic transitions, so they correlate strongly
with meaning boundaries. They are not derived from understanding the content.

For uncut or sparsely cut content (lectures, podcasts, live recordings) the
detector will find few or no cuts. The secondary interval cap (MAX_SCENE_DURATION)
ensures segments remain within the Gemini Embedding 2 window regardless.

True deep-semantic segmentation (topic modelling on transcripts, LLM-based
boundary detection) is planned but out of scope for v1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audio-only files
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PySceneDetect requires video frames — it cannot process audio-only files.
Audio segmentation uses FFmpeg's `silencedetect` filter instead, which finds
pauses between speech or music. This is the closest audio analogue to a
visual cut: a silence is where a speaker finishes a thought and pauses before
the next, functioning as a natural segment boundary.

Pipeline selection is automatic based on the resolved media type:
  video  →  PySceneDetect ContentDetector  (+ interval cap)
  audio  →  FFmpeg silencedetect           (+ interval cap)
"""

import logging
import math
import re
import subprocess
from typing import List, Literal, Tuple

from scenedetect import ContentDetector, SceneManager, detect, open_video
from scenedetect.scene_manager import save_images

from multimodal_search.config import settings

logger = logging.getLogger(__name__)

# ── Type alias ────────────────────────────────────────────────────────────────
Segment = Tuple[float, float]  # (start_sec, end_sec)


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════


def detect_segments(
    media_path: str,
    media_type: Literal["video", "audio"],
) -> List[Segment]:
    """
    Detect natural segment boundaries in a media file.

    Args:
        media_path:  Absolute path to a local media file.
        media_type:  "video" or "audio" — controls detector selection.

    Returns:
        List of (start_sec, end_sec) tuples, ordered chronologically.
        Always contains at least one entry spanning the full file.
    """
    logger.info("Detecting segments in %s [%s]", media_path, media_type)

    if media_type == "video":
        raw = _detect_video_cuts(media_path)
    else:
        raw = _detect_audio_silences(media_path)

    constrained = _apply_duration_constraints(raw)

    logger.info(
        "Segmentation complete: %d segments (from %d raw boundaries)",
        len(constrained),
        len(raw),
    )
    return constrained


# ═══════════════════════════════════════════════════════════════════════════════
# Video segmentation — PySceneDetect ContentDetector
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_video_cuts(video_path: str) -> List[Segment]:
    """
    Find physical cut points using PySceneDetect's ContentDetector.

    ContentDetector computes the mean per-channel HSV difference between
    consecutive decoded frames. When the score exceeds `scene_threshold`,
    it marks a scene boundary. Lower threshold = more sensitive to soft
    transitions; higher = only hard cuts.

    Falls back to a single full-duration segment if no cuts are found
    (e.g. continuous lecture recording with no editing).
    """
    try:
        raw_scenes = detect(
            video_path,
            ContentDetector(threshold=settings.scene_threshold),
        )
    except Exception as exc:
        logger.warning(
            "PySceneDetect failed for %s (%s) — falling back to full file",
            video_path,
            exc,
        )
        return _full_file_fallback(video_path)

    if not raw_scenes:
        logger.info("No cuts detected — using full file as single segment")
        return _full_file_fallback(video_path)

    segments: List[Segment] = [
        (start.get_seconds(), end.get_seconds()) for start, end in raw_scenes
    ]
    logger.debug("ContentDetector found %d cuts", len(segments))
    return segments


def _full_file_fallback(video_path: str) -> List[Segment]:
    """Return [(0.0, duration)] for files where detection produces nothing."""
    duration = _get_duration_ffprobe(video_path)
    return [(0.0, duration)] if duration > 0 else []


# ═══════════════════════════════════════════════════════════════════════════════
# Audio segmentation — FFmpeg silencedetect
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_audio_silences(audio_path: str) -> List[Segment]:
    """
    Segment an audio file using FFmpeg's silencedetect filter.

    Silence periods are natural pauses between speech turns or musical phrases
    and serve the same role as visual cuts: they mark where one thought ends
    and another begins.

    Config driven by:
        SILENCE_THRESHOLD_DB   — level below which audio is "silent" (default -35 dB)
        SILENCE_MIN_DURATION   — minimum silence length to register (default 0.5 s)

    Returns segments representing the NON-silent regions between silences.
    Falls back to a single full-duration segment if no silence is detected.
    """
    threshold_db = getattr(settings, "silence_threshold_db", -35)
    min_silence_dur = getattr(settings, "silence_min_duration_sec", 0.5)

    cmd = [
        "ffmpeg",
        "-i",
        audio_path,
        "-af",
        f"silencedetect=noise={threshold_db}dB:d={min_silence_dur}",
        "-f",
        "null",
        "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        # silencedetect writes to stderr
        stderr = result.stderr
    except FileNotFoundError:
        logger.error("ffmpeg not found — cannot segment audio")
        raise

    silence_starts = [float(m) for m in re.findall(r"silence_start: ([0-9.]+)", stderr)]
    silence_ends = [float(m) for m in re.findall(r"silence_end: ([0-9.]+)", stderr)]

    total_duration = _get_duration_ffprobe(audio_path)

    if not silence_starts:
        logger.info("No silence detected in audio — using full file as single segment")
        return [(0.0, total_duration)]

    # Build non-silent regions from silence boundaries
    segments: List[Segment] = []
    cursor = 0.0

    for s_start, s_end in zip(silence_starts, silence_ends):
        if s_start > cursor:
            segments.append((cursor, s_start))
        cursor = s_end

    # Tail segment after last silence
    if cursor < total_duration:
        segments.append((cursor, total_duration))

    logger.debug(
        "silencedetect found %d silence periods → %d speech segments",
        len(silence_starts),
        len(segments),
    )
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# Secondary interval constraint
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_duration_constraints(segments: List[Segment]) -> List[Segment]:
    """
    Enforce MIN_SCENE_DURATION and MAX_SCENE_DURATION on a list of segments.

    Step 1 — Merge: segments shorter than MIN are merged into their predecessor
             to avoid embedding single-sentence fragments.
    Step 2 — Split: segments longer than MAX are divided into equal sub-segments
             so they fit within the Gemini Embedding 2 video limit (128 s) and
             produce focused, coherent embeddings.

    This is the ONLY interval-based logic. All primary segmentation is
    content-driven by either ContentDetector or silencedetect above.
    """
    if not segments:
        return []

    min_dur = settings.min_scene_duration
    max_dur = settings.max_scene_duration

    # ── Step 1: merge short segments ─────────────────────────────────────────
    merged: List[Segment] = []
    for start, end in segments:
        dur = end - start
        if merged and dur < min_dur:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)  # extend previous
        else:
            merged.append((start, end))

    # ── Step 2: split long segments ───────────────────────────────────────────
    result: List[Segment] = []
    for start, end in merged:
        dur = end - start
        if dur <= max_dur:
            result.append((start, end))
        else:
            n = math.ceil(dur / max_dur)
            sub_dur = dur / n
            for i in range(n):
                sub_start = start + i * sub_dur
                sub_end = min(start + (i + 1) * sub_dur, end)
                result.append((sub_start, sub_end))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FFprobe helper
# ═══════════════════════════════════════════════════════════════════════════════


def _get_duration_ffprobe(media_path: str) -> float:
    """Return the duration of a media file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", media_path, exc)
        return 0.0


logger = logging.getLogger(__name__)
