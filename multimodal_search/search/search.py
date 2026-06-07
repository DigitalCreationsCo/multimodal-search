"""
search/search.py

Multimodal semantic search over the nested OpenSearch index.

Previous version bugs fixed:
  1. Field references were off the hit root (e.g. hit["content_id"]) — all
     payload data lives under hit["_source"]. Fixed to use _source correctly.
  2. The knn query targeted top-level fields (embeddings.video_embedding) that
     don't exist in the new nested mapping. Fixed to use nested query +
     inner_hits to target segments.* fields.
  3. SearchResult construction referenced renamed/removed fields
     (video_id → documentId, video_name → fileName, etc.).
  4. weighted_score was passed to round() without a ndigits arg — round() with
     one arg returns an int, discarding the fractional score.

Architecture note — nested knn + inner_hits:
  Because segments are a nested type, a standard top-level knn query cannot
  reach into them. The nested query runs the knn search within each document's
  nested segments and returns `inner_hits` — the specific matching segment(s)
  within each document — alongside the parent document's _source.

  Score fusion: three separate nested queries (one per embedding type) are
  issued and their results merged client-side with intent-based weights.
  This is equivalent to the bool/should boost approach but gives finer
  control over normalization.
"""

import logging
from typing import Dict, List, Optional

from multimodal_search.models import DocumentListItem, OpenSearchDocument, SegmentSearchResult
from opensearchpy import OpenSearch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# OpenSearch client factory
# ═══════════════════════════════════════════════════════════════════════════════


def get_opensearch_client() -> OpenSearch:
    """
    Build an OpenSearch client from environment settings.
    Handles the self-signed TLS cert present in the Docker setup.
    """
    from multimodal_search.config import settings

    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=(settings.opensearch_user, settings.opensearch_password),
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        ssl_show_warn=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Core search function
# ═══════════════════════════════════════════════════════════════════════════════


def multimodal_search(
    client: OpenSearch,
    index_name: str,
    query_embedding: List[float],
    weights: Dict[str, float],
    top_k: int = 10,
    document_id: Optional[str] = None,
) -> List[SegmentSearchResult]:
    """
    Search across all three embedding types using nested knn queries and
    merge results client-side using intent-based weights.

    Args:
        client:          OpenSearch client instance.
        index_name:      Target index name.
        query_embedding: The embedded query vector (same for all three searches).
        weights:         Dict with keys "video", "audio", "text" summing to 1.0.
        top_k:           Maximum results to return.
        document_id:     Optional — restrict search to a single document.

    Returns:
        List of SegmentSearchResult, sorted by weighted score descending.
    """
    # Build optional document filter
    doc_filter = None
    if document_id:
        doc_filter = {"term": {"documentId": document_id}}

    # Run one nested knn query per active embedding type
    candidate_pool: Dict[str, dict] = {}  # key: "documentId:segmentIndex"

    vector_fields = {
        "video": "segments.videoEmbedding",
        "audio": "segments.audioEmbedding",
        "text": "segments.textEmbedding",
    }

    for vec_name, field_path in vector_fields.items():
        weight = weights.get(vec_name, 0.0)
        if weight == 0.0:
            continue

        hits = _nested_knn_search(
            client=client,
            index_name=index_name,
            field_path=field_path,
            query_vector=query_embedding,
            k=top_k * 3,  # fetch more for better fusion coverage
            doc_filter=doc_filter,
        )

        for hit in hits:
            source = hit["_source"]
            inner = hit.get("inner_hits", {}).get("segments", {}).get("hits", {})
            best_inner = inner.get("hits", [{}])[0]
            inner_source = best_inner.get("_source", {})
            inner_score = best_inner.get("_score", 0.0)

            doc_id = source.get("documentId", hit["_id"])
            seg_idx = inner_source.get("segmentIndex", 0)
            pool_key = f"{doc_id}:{seg_idx}"

            if pool_key not in candidate_pool:
                candidate_pool[pool_key] = {
                    "source": source,
                    "inner": inner_source,
                    "weighted_score": 0.0,
                    "matched_vectors": [],
                }

            candidate_pool[pool_key]["weighted_score"] += inner_score * weight
            candidate_pool[pool_key]["matched_vectors"].append(vec_name)

    # Sort and slice
    ranked = sorted(
        candidate_pool.values(),
        key=lambda x: x["weighted_score"],
        reverse=True,
    )[:top_k]

    return [_build_result(r) for r in ranked]


def _nested_knn_search(
    client: OpenSearch,
    index_name: str,
    field_path: str,
    query_vector: List[float],
    k: int,
    doc_filter: Optional[dict],
) -> list:
    """
    Execute a nested knn query targeting a single embedding field.

    Returns the raw OpenSearch hits list.
    """
    # Build the nested knn query
    nested_knn = {
        "nested": {
            "path": "segments",
            "query": {
                "knn": {
                    field_path: {
                        "vector": query_vector,
                        "k": k,
                    }
                }
            },
            # inner_hits returns the specific matching segment(s)
            "inner_hits": {
                "size": 1,  # we only need the top matching segment per doc
                "_source": {
                    # Exclude heavy vector arrays from inner_hits source
                    "excludes": [
                        "segments.videoEmbedding",
                        "segments.audioEmbedding",
                        "segments.textEmbedding",
                    ]
                },
            },
            "score_mode": "max",  # use the best-matching segment score
        }
    }

    # Optionally restrict to a single document
    if doc_filter:
        query_body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [nested_knn],
                    "filter": [doc_filter],
                }
            },
            "_source": {
                "excludes": ["segments"]
            },  # parent source only (no segments bulk)
        }
    else:
        query_body = {
            "size": k,
            "query": nested_knn,
            "_source": {"excludes": ["segments"]},
        }

    try:
        response = client.search(index=index_name, body=query_body)
        return response.get("hits", {}).get("hits", [])
    except Exception as exc:
        logger.error("OpenSearch knn query failed for field %s: %s", field_path, exc)
        return []


def _build_result(r: dict) -> SegmentSearchResult:
    """Map raw pool entry to the API response model."""
    source = r["source"]
    inner = r["inner"]
    gen = inner.get("generatedMetadata", {})
    content = inner.get("contentMetadata", {})

    return SegmentSearchResult(
        documentId=source.get("documentId", ""),
        fileName=source.get("fileName", ""),
        contentType=source.get("contentType", ""),
        segmentIndex=inner.get("segmentIndex", 0),
        startSec=inner.get("startSec", 0.0),
        endSec=inner.get("endSec", 0.0),
        durationSec=inner.get("contentMetadata", {}).get("durationSec", 0.0),
        title=gen.get("title", ""),
        summary=gen.get("summary", ""),
        transcript=gen.get("transcript", ""),
        keywords=gen.get("keywords", []),
        mood=gen.get("mood", "ambient"),
        hasSpeech=gen.get("hasSpeech", False),
        thumbnailUri=content.get("thumbnailUri"),
        mediaType=content.get("mediaType", ""),
        score=round(r["weighted_score"], 4),
        matchedVectors=r["matched_vectors"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Document listing
# ═══════════════════════════════════════════════════════════════════════════════


def list_documents(client: OpenSearch, index_name: str) -> List[DocumentListItem]:
    """
    Return one summary record per indexed document.
    Uses a scroll or aggregation-style query — no embeddings returned.
    """
    query_body = {
        "size": 1000,
        "query": {"match_all": {}},
        "_source": {
            "excludes": ["segments"]  # Exclude all segment data (including vectors)
        },
    }

    try:
        response = client.search(index=index_name, body=query_body)
    except Exception as exc:
        logger.error("Failed to list documents: %s", exc)
        return []

    results = []
    for hit in response.get("hits", {}).get("hits", []):
        s = hit.get("_source", {})

        # Count segments by fetching with segments.segmentIndex only
        seg_count = _count_segments(client, index_name, s.get("documentId", ""))

        results.append(
            DocumentListItem(
                documentId=s.get("documentId", hit["_id"]),
                fileName=s.get("fileName", ""),
                contentType=s.get("contentType", ""),
                durationSec=s.get("durationSec", 0.0),
                sizeBytes=s.get("sizeBytes", 0),
                segmentCount=seg_count,
                dateIngested=s.get("dateIngested", ""),
            )
        )

    return results


def _count_segments(client: OpenSearch, index_name: str, document_id: str) -> int:
    """Return the number of segments in a specific document."""
    try:
        body = {
            "size": 1,
            "query": {"term": {"documentId": document_id}},
            "_source": ["segments.segmentIndex"],
        }
        r = client.search(index=index_name, body=body)
        hits = r.get("hits", {}).get("hits", [])
        if hits:
            segs = hits[0].get("_source", {}).get("segments", [])
            return len(segs)
    except Exception:
        pass
    return 0
