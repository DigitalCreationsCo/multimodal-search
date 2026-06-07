import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from config import settings

from storage import StorageService

logger = logging.getLogger(__name__)


class FilesystemService(StorageService):
    """Handles direct local disk operations."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.analyses_dir = self.base_dir / settings.metadata_directory
        self.embeddings_dir = self.base_dir / settings.embeddings_directory
        self.documents_dir = self.base_dir / settings.documents_directory
        self.content_dir = self.base_dir / settings.content_directory

        # Ensure directories exist
        try:
            self.analyses_dir.mkdir(parents=True, exist_ok=True)
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            self.content_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Local storage directories verified at {self.base_dir}")
        except Exception as e:
            logger.error(f"Failed to create local storage directories: {e}")
            raise

    def fetch_media(self, uri: str) -> Path:
        # Assuming uri is a local path string
        return Path(uri)

    def fetch_metadata(self, job_id: str) -> Dict[str, Any]:
        with open(self.base_dir / "analyses" / f"{job_id}.json", "r") as f:
            return json.load(f)

    def write_metadata(self, job_id: str, payload: Dict[str, Any]) -> None:
        with open(self.base_dir / "analyses" / f"{job_id}.json", "w") as f:
            json.dump(payload, f, indent=2)

    def fetch_embeddings(self, job_id: str) -> Dict[str, Any]:
        with open(self.base_dir / "embeddings" / f"{job_id}.json", "r") as f:
            return json.load(f)

    def write_embeddings(self, job_id: str, payload: Dict[str, Any]) -> None:
        with open(self.base_dir / "embeddings" / f"{job_id}.json", "w") as f:
            json.dump(payload, f, indent=2)

    def fetch_documents(self, doc_id: str) -> Dict[str, Any]:
        with open(self.base_dir / "documents" / f"{doc_id}.json", "r") as f:
            return json.load(f)

    def write_documents(self, doc_id: str, payload: Dict[str, Any]) -> None:
        with open(self.base_dir / "documents" / f"{doc_id}.json", "w") as f:
            json.dump(payload, f, indent=2)

    def cleanup_staging(self, uri: str) -> None:
        logger.debug(f"[Local] No staging cleanup required for local file: {uri}")


os.makedirs(settings.temp_dir, exist_ok=True)
