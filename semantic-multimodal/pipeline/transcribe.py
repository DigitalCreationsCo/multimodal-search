"""
pipeline/transcribe.py

Local speech-to-text using OpenAI Whisper.
Whisper runs entirely on-device (CPU is fine for `base` model),
so there is zero per-call cost and no data leaves your infra.

The transcript feeds two downstream steps:
  1. Gemini Embedding 2 text embed  → audio_vec
  2. Gemini Flash metadata generation
"""

import logging
from functools import lru_cache
from typing import Optional

import whisper

from config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> whisper.Whisper:
    """Load and cache the Whisper model (loaded once per process)."""
    logger.info("Loading Whisper model: %s", settings.whisper_model)
    model = whisper.load_model(settings.whisper_model)
    logger.info("Whisper model ready")
    return model


def transcribe(audio_path: str, language: Optional[str] = None) -> str:
    """
    Transcribe the audio file at *audio_path* to text.

    Args:
        audio_path: Path to a 16 kHz mono WAV file (from chunker.extract_audio).
        language:   ISO-639-1 code to skip language detection (faster).
                    Pass None to auto-detect.

    Returns:
        The full transcript as a single string.
        Returns an empty string if Whisper finds no speech.
    """
    model = _load_model()

    opts = {}
    if language:
        opts["language"] = language

    try:
        result = model.transcribe(
            audio_path,
            fp16=False,        # CPU inference — fp16 unsupported
            verbose=False,
            **opts,
        )
        text: str = result.get("text", "").strip()
        logger.debug("Transcript (%d chars): %s…", len(text), text[:80])
        return text
    except Exception as exc:
        logger.warning("Transcription failed for %s: %s", audio_path, exc)
        return ""