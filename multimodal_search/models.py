"""
models.py — Canonical data shapes for the entire system.

Design rules enforced here:
  1. One OpenSearchDocument per content file (atomic).
  2. Segments are the only thing inside that document that varies.
  3. Each segment carries TWO distinct metadata objects:
       contentMetadata  — measurable facts derived from the file itself
       generatedMetadata — AI-inferred analysis from Gemini Flash
  4. Embeddings live inside segments, never at the document root.
  5. API I/O models (request / response) are kept separate from
     storage models to avoid leaking internal shapes to callers.
"""

import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
# Segment-level sub-models
# ═══════════════════════════════════════════════════════════════════════════════


class SegmentContentMetadata(BaseModel):
    """
    Measurable, factual metadata about a segment — derived directly
    from the media file. No AI inference. Never changes after ingest.

    Fields:
        sizeBytes     File size of the extracted segment chunk on disk.
        durationSec   Precise duration computed from start/end timestamps.
        thumbnailUri  Storage URI for the mid-point JPEG frame.
                      None for audio-only segments (no video stream).
        hasVideo      True if a video stream was extracted.
        hasAudio      True if an audio stream is present.
        mediaType     The resolved media type of the parent file.
    """

    sizeBytes: int
    durationSec: float
    thumbnailUri: Optional[str] = None
    hasVideo: bool
    hasAudio: bool
    mediaType: Literal["video", "audio"]


class SegmentGeneratedMetadata(BaseModel):
    """
    AI-generated semantic analysis of a segment, produced by Gemini Flash.
    All fields are inferred from the segment's content. May be regenerated.

    Fields:
        title       Short descriptive label (5–10 words).
        summary     One-sentence description of what happens.
        keywords    Up to 8 indexable topic/entity keywords.
        mood        One of the controlled vocabulary values below.
        hasSpeech   Whether audible speech was detected.
        transcript  Verbatim speech-to-text from Gemini.
        confidence  Model confidence in the generated metadata (0–1).
    """

    title: str
    summary: str
    keywords: List[str] = Field(default_factory=list, max_length=8)
    mood: Literal[
        "informative", "emotional", "action", "technical", "conversational", "ambient"
    ]
    hasSpeech: bool
    transcript: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ContentSegment(BaseModel):
    """
    One temporal slice of a content file.

    Each segment is the unit of retrieval: the search system finds segments,
    not whole files. Three embedding vectors are stored per segment so the
    intent router can weight them independently at query time.

    Note: videoEmbedding is None for audio-only source files.
          audioEmbedding is derived from the transcript text.
          textEmbedding is derived from the generated metadata string.
    """

    segmentIndex: int
    startSec: float
    endSec: float

    contentMetadata: SegmentContentMetadata
    generatedMetadata: SegmentGeneratedMetadata

    # Vectors — excluded from OpenSearch _source returns by default
    videoEmbedding: Optional[List[float]] = None  # None for audio-only
    audioEmbedding: List[float]  # From transcript
    textEmbedding: List[float]  # From generated metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Document model — one per content file
# ═══════════════════════════════════════════════════════════════════════════════


class OpenSearchDocument(BaseModel):
    """
    The canonical document shape indexed into OpenSearch and persisted
    by the storage layer. One document per content file.

    Segments are stored as a nested list. OpenSearch k-NN queries
    target the nested embedding fields via nested queries + inner_hits
    to pinpoint the specific matching segment.

    Fields:
        documentId    UUID, same as the ingest job_id. Primary key.
        fileName      Original filename (e.g. lecture.mp4).
        uri           Canonical storage URI — local path or s3://...
        contentType   "video" or "audio" — controls pipeline routing.
        sizeBytes     Total source file size in bytes.
        durationSec   Total duration of the source file.
        dateIngested  ISO 8601 UTC timestamp of when ingest completed.
        segments      Ordered list of temporal segments.
    """

    documentId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fileName: str
    uri: str
    contentType: Literal["video", "audio"]
    sizeBytes: int
    durationSec: float
    dateIngested: str  # ISO 8601, e.g. "2026-06-07T14:30:00Z"

    segments: List[ContentSegment]


# ═══════════════════════════════════════════════════════════════════════════════
# API request / response models
# ═══════════════════════════════════════════════════════════════════════════════


class IngestURLRequest(BaseModel):
    url: str
    name: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    mode: Literal["auto", "visual", "speech", "topic"] = "auto"
    document_id: Optional[str] = None  # Scope search to a single document
    score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class SegmentSearchResult(BaseModel):
    """
    A single matching segment returned by the search API.
    Heavy embedding vectors are never included in API responses.
    """

    # Parent document identifiers
    documentId: str
    fileName: str
    contentType: str

    # Segment position
    segmentIndex: int
    startSec: float
    endSec: float
    durationSec: float

    # Generated metadata (what the model inferred)
    title: str
    summary: str
    transcript: str
    keywords: List[str]
    mood: str
    hasSpeech: bool

    # Content metadata (facts)
    thumbnailUri: Optional[str] = None
    mediaType: str

    # Retrieval metadata
    score: float
    matchedVectors: List[str]  # Which embedding types contributed


class DocumentListItem(BaseModel):
    """Summary record for the /documents list endpoint."""

    documentId: str
    fileName: str
    contentType: str
    durationSec: float
    sizeBytes: int
    segmentCount: int
    dateIngested: str
