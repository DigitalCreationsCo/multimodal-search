import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class StorageService(ABC):
    """
    The Universal Storage Protocol.
    Any storage backend must implement these methods.
    """

    @abstractmethod
    def fetch_media(self, uri: str) -> Path:
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
    def write_documents(self, doc_id: str, payload: Dict[str, Any]) -> None:
        pass
