"""
pipeline/transcribe.py

Cloud-based transcription using Gemini Flash.

Replaces the previous Whisper (local) approach.

Why Gemini Flash instead of Whisper?
  - No local model to download or GPU/CPU overhead
  - Native audio + video understanding — Gemini reads the clip directly,
    so it captures speaker tone, pauses, and non-speech audio context
  - Same API call as metadata enrichment (one less dependency, one less
    Docker layer, simpler ops)
  - Cost: ~$0.075 / 1M input tokens — negligible per chunk

API path:
  We upload the audio WAV file to the Google Files API (same pattern
  as video embedding), call generate_content with a transcription prompt,
  then delete the temporary file.
"""

import logging
import pathlib
import time

from multimodal_search.config import get_client, settings
from multimodal_search.pipeline._retry import retry_with_backoff

logger = logging.getLogger(__name__)

_TRANSCRIBE_PROMPT = (
    "Transcribe all spoken words in this audio clip verbatim. "
    "Output only the transcript text with no commentary, timestamps, "
    "or speaker labels. If there is no speech, output an empty string."
)


def _wait_for_file(client, file_name: str, timeout: int = 60) -> None:
    """Poll until the uploaded file reaches ACTIVE state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = client.files.get(name=file_name)
        state = getattr(info, "state", None)
        if state and state.name == "ACTIVE":
            return
        if state and state.name == "FAILED":
            raise RuntimeError(f"File processing failed: {file_name}")
        time.sleep(2)
    raise TimeoutError(f"File {file_name} not ACTIVE after {timeout}s")


def transcribe(audio_path: str) -> str:
    """
    Transcribe a WAV audio file using Gemini Flash.

    Retries with exponential backoff + jitter on failure.

    Args:
        audio_path: Path to a 16 kHz mono WAV file (from chunker.extract_audio).

    Returns:
        Transcript string. Empty string if no speech is detected or on error.
    """
    path = pathlib.Path(audio_path)
    if not path.exists():
        logger.warning("Audio file not found: %s", audio_path)
        return ""

    def _attempt() -> str:
        client = get_client()
        uploaded = None
        try:
            uploaded = client.files.upload(
                file=path,
                config={"display_name": path.name, "mime_type": "audio/wav"},
            )
            _wait_for_file(client, uploaded.name)

            from google.genai import types as gtypes

            response = client.models.generate_content(
                model=settings.gemini_flash_model,
                contents=gtypes.Content(
                    parts=[
                        gtypes.Part(
                            file_data=gtypes.FileData(
                                file_uri=uploaded.uri,
                                mime_type="audio/wav",
                            )
                        ),
                        gtypes.Part(text=_TRANSCRIBE_PROMPT),
                    ]
                ),
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 1024,
                },
            )

            text = (response.text or "").strip()
            logger.debug("Transcript (%d chars): %s…", len(text), text[:80])
            return text

        finally:
            if uploaded:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass  # Non-critical cleanup

    try:
        return retry_with_backoff(
            _attempt,
            label=f"transcription {audio_path}",
        )
    except Exception as exc:
        logger.warning(
            "Transcription failed for %s after %d attempts: %s",
            audio_path,
            settings.max_attempts,
            exc,
        )
        return ""
