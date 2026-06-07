from config import settings
from opensearch import Client as OpenSearch

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
index_name = "semantic-multimodal"


def create_index(client: OpenSearch, os_index: str, embedding_dimension):
    # Define the index mapping with multiple knn_vector fields
    index_body = {
        "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 100}},
        "mappings": {
            # Metadata mapping
            "content_path": {"type": "keyword"},
            "embeddings": {
                "type": "nested",
                "properties": {
                    "video_embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                    "audio_embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                    "text_embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                },
            },
        },
    }

    # Create the index
    if not client.indices.exists(index=os_index):
        client.indices.create(index=os_index, body=index_body)


create_index(client, index_name, settings.embedding_dimension)
