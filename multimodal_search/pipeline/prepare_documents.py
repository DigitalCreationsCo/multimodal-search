import os
import time

from multimodal_search.config import settings
from multimodal_search.models import ContentMetadata, Embeddings, OpenSearchDocument
from multimodal_search.storage.storage_router import StorageRouter


def main():

    storage = StorageRouter.get_storage()
    for metadata_file in storage.list_files(settings.metadata_directory):
        if not metadata_file.endswith(".json"):
            print(f"Skipping {metadata_file}, not a JSON file.")
            continue

        print(f"Generating OpenSearch document for: {metadata_file}")

        # Construct file paths for embeddings and analysis
        embeddings_file_path = os.path.join(
            settings.embeddings_directory, metadata_file
        )
        metadata_file_path = os.path.join(settings.metadata_directory, metadata_file)
        try:
            embeddings = Embeddings(**storage.read_json_file(embeddings_file_path))
            metadata = ContentMetadata(**storage.read_json_file(metadata_file_path))

            # Prepare OpenSearch document
            opensearch_document: OpenSearchDocument = prepare_opensearch_documents(
                embeddings, metadata
            )

            # Write the OpenSearch document to a file
            storage.write_documents(metadata_file, opensearch_document)
        except FileNotFoundError as e:
            print(e)
            continue


def prepare_opensearch_documents(
    embeddings: Embeddings, metadata: ContentMetadata
) -> OpenSearchDocument:
    """Prepare OpenSearch document from content embeddings and metadata.
    :param embeddings: Embeddings object containing multivector embeddings.
    :param analysis: Metadata object containing generated content metadata.
    :return: OpenSearchDocument object ready for indexing.
    """

    document = OpenSearchDocument(
        fileName=metadata.fileName,
        uri=metadata.uri,
        dateCreated=time.strftime("%Y-%m-%dT%H:%M:%S %Z", time.gmtime()),
        contentType=embeddings.contentType,
        sizeBytes=embeddings.sizeBytes,
        durationSec=embeddings.durationSec,
        embeddings=embeddings.embeddings,
    )

    return document


if __name__ == "__main__":
    main()
    print("OpenSearch documents prepared successfully.")
