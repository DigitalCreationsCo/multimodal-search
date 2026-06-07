"""
search/index.py

OpenSearch index definition for the multimodal search system.

Key differences from the previous version:
  1. nmslib engine removed — nmslib was deprecated in OpenSearch 2.x and is
     unavailable in OpenSearch 3.x. Using `faiss` instead.
  2. Segments are a proper `nested` type. This is required to store per-segment
     embeddings inside a single document and query them with inner_hits.
  3. Text fields (title, summary, transcript, keywords) are explicitly mapped
     so OpenSearch builds BM25 inverted indexes over them. This enables
     hybrid (vector + keyword) queries — the primary reason to use OpenSearch
     over a pure vector store like Qdrant.
  4. The embedding vector fields live under `segments.*` and are excluded from
     the returned `_source` by default in queries — returned vectors are large
     and wasteful in API responses.
  5. Document-level fields use `keyword` for exact/filter matches and `text`
     for full-text search where appropriate.
"""

import logging

from multimodal_search.config import settings
from opensearchpy import OpenSearch

logger = logging.getLogger(__name__)

INDEX_NAME = settings.collection_name  # default: "multimodal_search"


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


def ensure_index(client: OpenSearch) -> None:
    """
    Create the index if it doesn't exist. Safe to call on every startup.
    """
    if client.indices.exists(index=INDEX_NAME):
        logger.info("Index '%s' already exists — skipping creation", INDEX_NAME)
        return

    body = get_index_body(settings.embedding_dimension)
    client.indices.create(index=INDEX_NAME, body=body)
    logger.info(
        "Created index '%s' (dim=%d, engine=faiss)",
        INDEX_NAME,
        settings.embedding_dimension,
    )


def drop_index(client: OpenSearch) -> None:
    """Destroy and recreate the index. Wipes all data — use carefully."""
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        logger.warning("Deleted index '%s'", INDEX_NAME)
    ensure_index(client)
