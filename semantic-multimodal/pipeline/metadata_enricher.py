"""
pipeline/metadata_enricher.py

Generates structured metadata per video segment using Gemini Flash.

Why bother with metadata enrichment?
  The meta_vec gives the search pipeline a "topic index" that captures
  named entities, mood, and high-level concepts that the raw video
  embedding may not surface for text queries like "scenes about revenue."

Cost: Gemini Flash Lite is ~$0.075/1M input tokens.
A 200-word transcript costs roughly $0.000015 to enrich.
"""

import json
import logging
import re
from typing import Optional

from google import genai

from config import settings

logger = logging.getLogger(__name__)

_client: Optional[genai.Client] = None

_SYSTEM_PROMPT = """\
You are a video indexing assistant. Given a transcript and timing of a video segment,
output ONLY a valid JSON object (no markdown, no backticks) with these exact keys:

{
  "title": "Short descriptive title (5–10 words)",
  "summary": "One sentence summary of what happens in this segment",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "mood": "one of: informative | emotional | action | technical | conversational | ambient",
  "has_speech": true or false,
  "confidence": 0.0 to 1.0
}

Be factual. If the transcript is empty, infer from timing and context.
Return ONLY the JSON object."""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


def _safe_parse_json(text: str) -> dict:
    """Extract and parse JSON from model output, tolerating minor formatting."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Safe fallback
    logger.warning("Could not parse metadata JSON, using fallback")
    return {
        "title": "Video Segment",
        "summary": "Video content segment",
        "keywords": [],
        "mood": "ambient",
        "has_speech": False,
        "confidence": 0.3,
    }


def enrich_segment(
    transcript: str,
    start_time: float,
    end_time: float,
    chunk_index: int,
) -> dict:
    """
    Call Gemini Flash to generate structured metadata for a segment.

    Args:
        transcript:   Whisper transcript text (may be empty).
        start_time:   Segment start in seconds.
        end_time:     Segment end in seconds.
        chunk_index:  Zero-based scene index within the video.

    Returns:
        Dict with keys: title, summary, keywords, mood, has_speech, confidence.
    """
    client = _get_client()

    user_prompt = (
        f"Segment #{chunk_index} — {start_time:.1f}s to {end_time:.1f}s "
        f"({end_time - start_time:.1f}s duration)\n\n"
        f"Transcript:\n{transcript or '[No speech detected]'}"
    )

    try:
        response = client.models.generate_content(
            model=settings.gemini_flash_model,
            contents=user_prompt,
            config={
                "system_instruction": _SYSTEM_PROMPT,
                "temperature": 0.1,    # Low temp for consistent JSON output
                "max_output_tokens": 300,
            },
        )
        raw = response.text or ""
        metadata = _safe_parse_json(raw)

    except Exception as exc:
        logger.warning("Metadata enrichment failed for chunk %d: %s", chunk_index, exc)
        metadata = {
            "title": f"Segment {chunk_index + 1}",
            "summary": transcript[:120] if transcript else "Video segment",
            "keywords": [],
            "mood": "ambient",
            "has_speech": bool(transcript),
            "confidence": 0.2,
        }

    # Normalise types
    metadata["keywords"] = [str(k) for k in metadata.get("keywords", [])][:8]
    metadata["has_speech"] = bool(metadata.get("has_speech", bool(transcript)))

    return metadata


def metadata_to_embed_string(metadata: dict) -> str:
    """
    Flatten structured metadata into a single string for embedding.
    Repeated keywords boost their signal in the vector space.
    """
    parts = [
        metadata.get("title", ""),
        metadata.get("summary", ""),
        " ".join(metadata.get("keywords", [])),
        metadata.get("mood", ""),
    ]
    return " | ".join(p for p in parts if p)