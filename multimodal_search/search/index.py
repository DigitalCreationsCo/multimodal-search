"""
search/index.py

OpenSearch index mapping definition.

Every other concern (alias management, index lifecycle, document writes)
lives in ``indexer.py``.  This module owns only:
  1. The constant ``INDEX_NAME``.
  2. The mapping builder ``get_index_body()`` (used by ``Indexer``).

Key mapping decisions:
  - ``segments`` is a ``nested`` type so we can store per-segment
    embeddings and retrieve them via ``inner_hits``.
  - Text fields (title, summary, transcript, keywords) get BM25 inverted
    indexes for hybrid (vector + keyword) search.
  - Three ``knn_vector`` fields per segment (video, audio, text) so the
    intent router can weight them independently at query time.
  - Embedding vectors are excluded from ``_source`` by default in queries
    because they are large and wasteful in API responses.
"""

import logging

from multimodal_search.config import settings

logger = logging.getLogger(__name__)

INDEX_NAME = settings.index_name  # default: "multimodal_search"


def get_index_body(embedding_dimension: int) -> dict:
    """
    Build the OpenSearch index mapping and settings dict.

    Segments are stored as nested objects. Each segment carries:
      - Temporal bounds (startSec, endSec, durationSec)
      - contentMetadata   — factual file-derived data (keyword/numeric fields)
      - generatedMetadata — AI analysis (text + keyword fields for hybrid search)
      - videoEmbedding    — 1024-dim knn_vector (None-able for audio-only)
      - audioEmbedding    — 1024-dim knn_vector (from transcript)
      - textEmbedding     — 1024-dim knn_vector (from generated metadata)
    """
    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
                # Number of shards: 1 is fine for <100k documents.
                # Increase for horizontal scaling.
                "number_of_shards": 1,
                "number_of_replicas": 0,  # set to 1+ in production
            }
        },
        "mappings": {
            "properties": {
                # ── Document-level fields ──────────────────────────────────
                "documentId": {"type": "keyword"},
                "fileName": {"type": "keyword"},
                "uri": {"type": "keyword"},
                "contentType": {"type": "keyword"},  # "video" | "audio"
                "sizeBytes": {"type": "long"},
                "durationSec": {"type": "float"},
                "dateIngested": {"type": "date"},
                # ── Segments (nested — one object per temporal chunk) ──────
                "segments": {
                    "type": "nested",
                    "properties": {
                        # Temporal position
                        "segmentIndex": {"type": "integer"},
                        "startSec": {"type": "float"},
                        "endSec": {"type": "float"},
                        # Content metadata (facts)
                        "contentMetadata": {
                            "properties": {
                                "sizeBytes": {"type": "long"},
                                "durationSec": {"type": "float"},
                                "thumbnailUri": {"type": "keyword"},
                                "hasVideo": {"type": "boolean"},
                                "hasAudio": {"type": "boolean"},
                                "mediaType": {"type": "keyword"},
                            }
                        },
                        # Generated metadata (text fields get BM25 indexes)
                        "generatedMetadata": {
                            "properties": {
                                # keyword for exact match / aggregations
                                "mood": {"type": "keyword"},
                                "hasSpeech": {"type": "boolean"},
                                "confidence": {"type": "float"},
                                # text for full-text BM25 search (hybrid)
                                "title": {
                                    "type": "text",
                                    "fields": {"raw": {"type": "keyword"}},
                                },
                                "summary": {"type": "text"},
                                "transcript": {"type": "text"},
                                "keywords": {
                                    "type": "text",
                                    "fields": {"raw": {"type": "keyword"}},
                                },
                            }
                        },
                        # ── Embedding vectors ──────────────────────────────
                        # faiss engine is the supported option in OpenSearch 3.x
                        # (nmslib was deprecated in 2.x and removed in 3.x)
                        "videoEmbedding": {
                            "type": "knn_vector",
                            "dimension": embedding_dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "faiss",
                                "parameters": {
                                    "m": 16,  # HNSW M parameter
                                    "ef_construction": 100,
                                },
                            },
                        },
                        "audioEmbedding": {
                            "type": "knn_vector",
                            "dimension": embedding_dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "faiss",
                                "parameters": {"m": 16, "ef_construction": 100},
                            },
                        },
                        "textEmbedding": {
                            "type": "knn_vector",
                            "dimension": embedding_dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "faiss",
                                "parameters": {"m": 16, "ef_construction": 100},
                            },
                        },
                    },  # end segments.properties
                },  # end segments
            }
        },
    }

