import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from multimodal_search.config import settings
from multimodal_search.models import OpenSearchDocument

from multimodal_search.storage import StorageService

logger = logging.getLogger(__name__)


class FileSystemService(StorageService):
    """Handles direct local disk operations."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.metadata_dir = self.base_dir / settings.metadata_directory
        self.embeddings_dir = self.base_dir / settings.embeddings_directory
        self.documents_dir = self.base_dir / settings.documents_directory
        self.content_dir = self.base_dir / settings.content_directory
        self.staging_dir = self.base_dir / settings.temp_dir

        # Ensure directories exist
        try:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            self.content_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Local storage directories verified at {self.base_dir}")
        except Exception as e:
            logger.error(f"Failed to create local storage directories: {e}")
            raise

    def fetch_media(self) -> Path:
        # Assuming uri is a local path string
        return self.list_files(settings.content_directory)

    def fetch_metadata(self, job_id: str) -> Dict[str, Any]:
        with open(self.metadata_dir / f"{job_id}.json", "r") as f:
            return json.load(f)

    def write_metadata(self, job_id: str, payload: Dict[str, Any]) -> None:
        with open(self.metadata_dir / f"{job_id}.json", "w") as f:
            json.dump(payload, f, indent=2)

    def fetch_embeddings(self, job_id: str) -> Dict[str, Any]:
        with open(self.embeddings_dir / f"{job_id}.json", "r") as f:
            return json.load(f)

    def write_embeddings(self, job_id: str, payload: Dict[str, Any]) -> None:
        with open(self.embeddings_dir / f"{job_id}.json", "w") as f:
            json.dump(payload, f, indent=2)

    def fetch_documents(self, doc_id: str) -> Dict[str, Any]:
        with open(self.documents_dir / f"{doc_id}.json", "r") as f:
            return json.load(f)

    def write_documents(self, doc_id: str, payload: OpenSearchDocument) -> None:
        with open(self.documents_dir / f"{doc_id}.json", "w") as f:
            json.dump(payload, f, indent=2)

    def cleanup_staging(self, uri: str) -> None:
        logger.info(f"[Remote] Cleaning up staged file for URI: {uri}")
        parsed = urlparse(uri)
        object_key = parsed.path.lstrip("/")
        staging_path = self.staging_dir / object_key

        try:
            if staging_path.exists():
                staging_path.unlink()
                logger.debug(
                    f"[Remote] Successfully removed staged file: {staging_path}"
                )
            else:
                logger.warning(
                    f"[Remote] Staged file not found for cleanup: {staging_path}"
                )
        except Exception as e:
            logger.error(f"[Remote] Failed to clean up staged file {staging_path}: {e}")

    def read_json_file(self, path: str) -> dict[str, Any]:
        """Read a JSON file and return the parsed data.
        :param file_path: Path to the JSON file.
        :return: Parsed JSON data as a dictionary.
        :raises FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as file:
            data = json.load(file)

        return data

    def list_files(self, prefix: str = "") -> List[str]:
        target_dir = self.base_dir / prefix

        try:
            if not target_dir.exists():
                logger.warning(f"[Local] Directory does not exist: {target_dir}")
                return []

            if not target_dir.is_dir():
                raise NotADirectoryError(
                    f"[Local] Path is not a valid directory: {target_dir}"
                )

            # Using list comprehension to filter by files only
            files = [f.name for f in target_dir.iterdir() if f.is_file()]
            logger.info(f"[Local] Found {len(files)} files in {target_dir}")

            return files

        except Exception as e:
            logger.error(
                f"[Local] Root cause analysis: Failed to list directory {target_dir}. Error: {e}"
            )
            raise


os.makedirs(settings.temp_dir, exist_ok=True)
