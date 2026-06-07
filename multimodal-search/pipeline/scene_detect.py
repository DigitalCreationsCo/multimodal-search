"""
pipeline/scene_detect.py

Detects scene boundaries in a video file using PySceneDetect's
ContentDetector. Returns a list of (start_sec, end_sec) tuples,
each representing one coherent semantic scene.

Key design decisions:
- ContentDetector analyses per-frame HSV differences — fast, no GPU needed.
- min_scene_duration guards against flash cuts creating noise segments.
- max_scene_duration splits long uncut scenes so they stay within the
  Gemini Embedding 2 video limit (128 s) and produce focused embeddings.
"""

import logging
from typing import List, Tuple

from scenedetect import detect, ContentDetector, SceneManager, open_video
from scenedetect.scene_manager import save_images

from config import settings

logger = logging.getLogger(__name__)


def detect_scenes(video_path: str) -> List[Tuple[float, float]]:
    """
    Detect semantically coherent scene boundaries in *video_path*.

    Args:
        video_path: Absolute path to a local video file.

    Returns:
        List of (start_seconds, end_seconds) tuples, one per scene.
        Always contains at least one entry (the whole video if no cuts found).
    """
    logger.info("Running scene detection on %s", video_path)

    raw_scenes = detect(
        video_path,
        ContentDetector(threshold=settings.scene_threshold),
    )

    # Convert PySceneDetect FrameTimecode objects → float seconds
    timestamps: List[Tuple[float, float]] = []
    for start_tc, end_tc in raw_scenes:
        timestamps.append((start_tc.get_seconds(), end_tc.get_seconds()))

    # Apply min/max duration constraints
    timestamps = _apply_duration_constraints(timestamps)

    logger.info("Found %d scenes (after constraints)", len(timestamps))
    return timestamps


def _apply_duration_constraints(
    scenes: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    1. Merge scenes shorter than MIN_SCENE_DURATION with their neighbour.
    2. Split scenes longer than MAX_SCENE_DURATION into equal sub-scenes.
    """
    min_dur = settings.min_scene_duration
    max_dur = settings.max_scene_duration

    # ── Step 1: merge short scenes ──────────────────────────────────
    merged: List[Tuple[float, float]] = []
    for start, end in scenes:
        if merged and (end - start) < min_dur:
            # Extend previous scene's end instead of adding a tiny scene
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # ── Step 2: split long scenes ───────────────────────────────────
    result: List[Tuple[float, float]] = []
    for start, end in merged:
        duration = end - start
        if duration <= max_dur:
            result.append((start, end))
        else:
            # Split into equal chunks that each fit within max_dur
            import math
            n_splits = math.ceil(duration / max_dur)
            chunk_dur = duration / n_splits
            for i in range(n_splits):
                chunk_start = start + i * chunk_dur
                chunk_end = min(start + (i + 1) * chunk_dur, end)
                result.append((chunk_start, chunk_end))

    return result