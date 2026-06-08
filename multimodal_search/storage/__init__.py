import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from multimodal_search.models import OpenSearchDocument

logger = logging.getLogger(__name__)


class StorageService(ABC):
    """
    The Universal Storage Protocol.
    Any storage backend must implement these methods.
    """

    @abstractmethod
    def fetch_media(self) -> List[str]:
        pass

    @abstractmethod
    def fetch_metadata(self, job_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def write_metadata(self, job_id: str, payload: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def fetch_embeddings(self, job_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def write_embeddings(self, job_id: str, payload: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def fetch_documents(self, doc_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def write_documents(self, doc_id: str, payload: OpenSearchDocument) -> None:
        pass

    @abstractmethod
    def read_json_file(self, path: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def cleanup_staging(self, uri: str) -> None:
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """
        Retrieves a list of filenames or object keys from the storage medium.

        Args:
            prefix (str): The local directory name or remote S3 prefix to scan.

        Returns:
            List[str]: A list of filenames found.
        """
        pass


# Deferred imports — subclasses need StorageService to be defined first.
from .file_system_service import FileSystemService  # noqa: E402
from .remote_storage_service import RemoteStorageService  # noqa: E402

__all__ = ["StorageService", "FileSystemService", "RemoteStorageService"]
