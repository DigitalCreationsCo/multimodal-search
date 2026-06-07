from typing import List, Literal, Optional

from pydantic import BaseModel


class IngestURLRequest(BaseModel):
    url: str
    name: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    mode: Literal["auto", "visual", "speech", "topic"] = "auto"
    video_id: Optional[str] = None
    score_threshold: float = 0.25


class SearchResult(BaseModel):
    video_id: str
    video_name: str
    chunk_index: int
    start_time: float
    end_time: float
    duration: float
    title: str
    summary: str
    transcript: str
    keywords: List[str]
    mood: str
    weighted_score: float
    contributing_vectors: List[str]
    thumbnail_url: Optional[str] = None


class SegmentMetadata(BaseModel):
    segmentIndex: int
    segmentStartSeconds: float
    segmentEndSeconds: float
    thumbnailURL: str
    summary: str
    title: str


class EmbeddingSegment(BaseModel):
    video_embedding: List[float]
    audio_embedding: List[float]
    text_embedding: List[float]
    status: str
    segmentMetadata: SegmentMetadata
    keywords: List[str]
    chunkId: str


class Embeddings(BaseModel):
    title: str
    content_path: str
    thumbnailURL: str
    dateCreated: str
    contentType: str
    sizeBytes: int
    durationSec: float = 0.0
    embeddings: List[EmbeddingSegment]


class ContentMetadata(BaseModel):
    title: str
    summary: str
    keywords: List[str]
    mood: str
    has_speech: bool
    confidence: float


class OpenSearchDocument(BaseModel):
    fileName: str
    content_path: str
    dateCreated: str
    contentType: str
    sizeBytes: int
    durationSec: float = 0.0
    embeddings: List[EmbeddingSegment]
