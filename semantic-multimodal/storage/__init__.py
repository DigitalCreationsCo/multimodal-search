import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageService(ABC):
    """
    The Universal Storage Protocol.
    Any storage backend must implement these methods.
    """

    @abstractmethod
    def fetch_media(self, uri: str) -> Path:
        """Resolves or downloads the media to a local path for FFmpeg processing."""
        pass

    @abstractmethod
    def write_metadata(self, job_id: str, payload: dict) -> None:
        """Persists the extracted JSON metadata."""
        pass

    @abstractmethod
    def write_embeddings(self, job_id: str, payload: dict) -> None:
        """Persists the generated JSON embeddings."""
        pass

    @abstractmethod
    def cleanup_staging(self, uri: str) -> None:
        """Cleans up any temporary files created during fetch_media."""
        pass
