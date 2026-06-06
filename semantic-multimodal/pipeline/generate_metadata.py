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
import os
import random
import re
import time

from config import get_client, settings
from models import ContentMetadata

logger = logging.getLogger(__name__)

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


def generate_metadata(
    transcript: str,
    start_time: float,
    end_time: float,
    chunk_index: int,
) -> ContentMetadata:
    """
    Generate structured metadata for a segment.

    Args:
        transcript:   Transcript text (may be empty).
        start_time:   Segment start in seconds.
        end_time:     Segment end in seconds.
        chunk_index:  Zero-based scene index within the video.

    Returns:
        Dict with keys: title, summary, keywords, mood, has_speech, confidence.
    """
    client = get_client()

    user_prompt = (
        f"Segment #{chunk_index} — {start_time:.1f}s to {end_time:.1f}s "
        f"({end_time - start_time:.1f}s duration)\n\n"
        f"Transcript:\n{transcript or '[No speech detected]'}"
    )

    retries = 0
    while True:
        try:
            response = client.models.generate_content(
                model=settings.gemini_flash_model,
                contents=user_prompt,
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.1,  # Low temp for consistent JSON output
                    "max_output_tokens": 600,
                },
            )
            raw = response.text or ""
            metadata = _safe_parse_json(raw)

        except Exception as e:
            logger.warning(
                "Metadata enrichment failed for chunk %d: %s", chunk_index, e
            )
            if retries < settings.max_attempts:
                retries += 1
                backoff_time = (2**retries) + random.uniform(
                    0, 1
                )  # Exponential backoff with jitter
                print(
                    f"Throttled. Retrying in {backoff_time:.2f} seconds (attempt {retries})..."
                )
                time.sleep(backoff_time)
            else:
                metadata = ContentMetadata(
                    title=f"Segment {chunk_index + 1}",
                    summary=transcript[:120] if transcript else "Content segment",
                    keywords=[],
                    mood="ambient",
                    has_speech=bool(transcript),
                    confidence=0.2,
                ).model_dump()

                # Normalise types
                metadata["keywords"] = [str(k) for k in metadata.get("keywords", [])][
                    :8
                ]
                metadata["has_speech"] = bool(
                    metadata.get("has_speech", bool(transcript))
                )

                return ContentMetadata.model_validate(metadata)


def metadata_to_embed_string(metadata: ContentMetadata) -> str:
    """
    Flatten structured metadata into a single string for embedding.
    Repeated keywords boost their signal in the vector space.
    """
    parts = [
        metadata.title,
        metadata.summary,
        " ".join(metadata.keywords) if metadata.keywords else "",
        metadata.mood,
    ]
    return " | ".join(p for p in parts if p)


def write_metadata_to_file(metadata: ContentMetadata, local_file_path: str) -> None:
    """Write the video analysis response to a local file.
    Args:
        video_analysis (VideoAnalysis): The video analysis object containing the response.
        local_file_path (str): The local file path where the response will be written.
    """
    with open(local_file_path, "w") as f:
        f.write(metadata.model_dump_json(indent=2))
    print(f"Response written to {local_file_path}")
