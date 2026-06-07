"""
pipeline/generate_metadata.py

Generates per-segment semantic metadata using Gemini Flash.

Bug fixed from previous version:
  The retry loop was a `while True` with a `return` only in the exception's
  else branch. On a successful API call, `metadata` was assigned but never
  returned — the function would run indefinitely. Fixed by adding an explicit
  `return` in the try block after successful parsing.

Schema alignment:
  Now returns SegmentGeneratedMetadata (not the old ContentMetadata which
  mixed file-level and segment-level concerns). `fileName` and `uri` are
  document-level fields and do not belong here.
"""

import json
import logging
import random
import re
import time

from multimodal_search.config import get_client, settings
from multimodal_search.models import SegmentGeneratedMetadata

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a media indexing assistant. Given the transcript and timing of a single
segment from a video or audio file, output ONLY a valid JSON object with these
exact keys — no markdown, no backticks, no preamble:

{
  "title": "Short descriptive title (5–10 words)",
  "summary": "One sentence describing what happens in this segment",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "mood": "one of: informative | emotional | action | technical | conversational | ambient",
  "hasSpeech": true or false,
  "transcript": "Verbatim transcript (copy from input, or empty string)",
  "confidence": 0.0 to 1.0
}

Rules:
- keywords: up to 8 items, lower-case, specific nouns and topics
- mood: pick the single best match from the controlled list above
- hasSpeech: true only if audible speech is present in the transcript
- confidence: your certainty that the metadata is accurate (not a score)
- If transcript is empty, infer what you can from duration and segment index
- Return ONLY the JSON object, nothing else"""


def _safe_parse(raw: str) -> dict:
    """Parse JSON from model output, tolerating minor formatting issues."""
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse metadata JSON from model output")
    return {}


def _fallback_metadata(
    transcript: str,
    segment_index: int,
) -> SegmentGeneratedMetadata:
    """Safe fallback when Gemini is unavailable after all retries."""
    return SegmentGeneratedMetadata(
        title=f"Segment {segment_index + 1}",
        summary=transcript[:120] if transcript else "Media content segment",
        keywords=[],
        mood="ambient",
        hasSpeech=bool(transcript.strip()),
        transcript=transcript,
        confidence=0.1,
    )


def generate_metadata(
    transcript: str,
    segment_index: int,
    start_sec: float,
    end_sec: float,
) -> SegmentGeneratedMetadata:
    """
    Call Gemini Flash to produce structured semantic metadata for one segment.

    Args:
        transcript:     Verbatim transcript from the transcription step.
                        May be empty for silent or non-speech segments.
        segment_index:  Zero-based index of this segment within the file.
        start_sec:      Segment start time in seconds.
        end_sec:        Segment end time in seconds.

    Returns:
        SegmentGeneratedMetadata populated with AI-generated analysis.
    """
    client = get_client()

    user_prompt = (
        f"Segment #{segment_index} — {start_sec:.1f}s to {end_sec:.1f}s "
        f"({end_sec - start_sec:.1f}s duration)\n\n"
        f"Transcript:\n{transcript or '[No speech detected]'}"
    )

    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, settings.max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=settings.gemini_flash_model,
                contents=user_prompt,
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.1,
                    "max_output_tokens": 600,
                },
            )
            raw = response.text or ""
            parsed = _safe_parse(raw)

            if not parsed:
                raise ValueError("Empty or unparseable response from model")

            # ── Normalise and validate ─────────────────────────────────────
            parsed["keywords"] = [str(k).lower() for k in parsed.get("keywords", [])][
                :8
            ]
            parsed["hasSpeech"] = bool(
                parsed.get("hasSpeech", bool(transcript.strip()))
            )
            parsed["transcript"] = transcript  # always use the actual transcript
            parsed["confidence"] = max(
                0.0, min(1.0, float(parsed.get("confidence", 0.8)))
            )

            # ── BUG FIX: explicit return on success ────────────────────────
            return SegmentGeneratedMetadata.model_validate(parsed)

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Metadata generation attempt %d/%d failed for segment %d: %s",
                attempt,
                settings.max_attempts,
                segment_index,
                exc,
            )
            if attempt < settings.max_attempts:
                backoff = (2**attempt) + random.uniform(0, 1)
                logger.info("Retrying in %.1fs…", backoff)
                time.sleep(backoff)

    logger.error(
        "All %d metadata generation attempts exhausted for segment %d: %s",
        settings.max_attempts,
        segment_index,
        last_exc,
    )
    return _fallback_metadata(transcript, segment_index)


def metadata_to_embed_string(metadata: SegmentGeneratedMetadata) -> str:
    """
    Flatten generated metadata into a single string for text embedding.
    Concatenating title + summary + keywords + mood gives the meta_vec
    a rich topic/entity signal that complements the raw video embedding.
    """
    parts = [
        metadata.title,
        metadata.summary,
        " ".join(metadata.keywords),
        metadata.mood,
    ]
    return " | ".join(p for p in parts if p)
