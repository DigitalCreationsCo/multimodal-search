from typing import List, Literal, Optional

from pydantic import BaseModel, HttpUrl


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


class EmbeddingSegment(BaseModel):
    embedding: List[float]
    status: str
    segmentMetadata: SegmentMetadata


class Embeddings(BaseModel):
    videoName: str
    s3URI: str
    keyframeURL: str
    dateCreated: str
    sizeBytes: int
    durationSec: float = 0.0
    contentType: str
    embeddings: List[EmbeddingSegment]


class ContentMetadata(BaseModel):
    title: str
    summary: str
    keywords: List[str]
    mood: str
    has_speech: bool
    confidence: float
