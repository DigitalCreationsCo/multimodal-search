"""
Key Advantages of this Approach
Built-in Weighting: The boost parameter handles the multimodal scoring naturally, multiplying the cosine similarity score of each vector by its designated weight before OpenSearch aggregates the final document score.

Payload Efficiency: By explicitly excluding the vector fields in the _source return mapping, you ensure that the API and MCP stdout responses remain incredibly fast and lightweight.
"""

from typing import List

from models import SearchResult
from opensearch import Client as OpenSearch


def multimodal_search(
    client: OpenSearch,
    index_name: str,
    query_embedding,
    video_embedding_weight: float,
    audio_embedding_weight: float,
    text_embedding_weight: float,
    top_k=5,
) -> List[SearchResult]:
    # Apply your specific configuration weights using the 'boost' parameter
    query_body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embeddings.video_embedding": {
                                "vector": query_embedding,
                                "k": top_k,
                                "boost": video_embedding_weight,
                            }
                        }
                    },
                    {
                        "knn": {
                            "embeddings.audio_embedding": {
                                "vector": query_embedding,
                                "k": top_k,
                                "boost": audio_embedding_weight,
                            }
                        }
                    },
                    {
                        "knn": {
                            "embeddings.text_embedding": {
                                "vector": query_embedding,
                                "k": top_k,
                                "boost": text_embedding_weight,
                            }
                        }
                    },
                ]
            }
        },
        # Exclude the heavy vector arrays from the returned results
        "_source": {
            "excludes": [
                "embeddings.video_embedding",
                "embeddings.audio_embedding",
                "embeddings.text_embedding",
            ]
        },
    }

    response = client.search(index=index_name, body=query_body)
    results = []

    for hit in response["hits"]["hits"]:
        thumbnail_url = hit["_source"]["segmentMetadata"]["thumbnailURL"]
        results.append(
            SearchResult(
                content_id=hit["content_id"],
                file_name=hit["file_name"],
                chunk_index=hit["chunk_index"],
                start_time=hit["start_time"],
                end_time=hit["end_time"],
                duration=hit["duration"],
                title=hit["title"],
                summary=hit["summary"],
                transcript=hit["transcript"],
                keywords=hit["keywords"],
                mood=hit["mood"],
                weighted_score=round(hit["weighted_score"]),
                contributing_vectors=hit["contributing_vectors"],
                thumbnail_url=thumbnail_url,
            )
        )

    return results
