import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from config import settings

from storage import StorageService

logger = logging.getLogger(__name__)


class FilesystemService(StorageService):
    """Handles direct local disk operations."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.analyses_dir = self.base_dir / "analyses"
        self.embeddings_dir = self.base_dir / "embeddings"

        # Ensure directories exist
        try:
            self.analyses_dir.mkdir(parents=True, exist_ok=True)
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Local storage directories verified at {self.base_dir}")
        except Exception as e:
            logger.error(f"Failed to create local storage directories: {e}")
            raise

    def fetch_media(self, uri: str) -> Path:
        logger.info(f"[Local] Fetching media for URI: {uri}")

        # Parse URI (e.g., "file:///data/video.mp4" or "/data/video.mp4")
        parsed = urlparse(uri)
        file_path = Path(parsed.path) if parsed.scheme == "file" else Path(uri)

        if not file_path.exists():
            logger.error(f"[Local] Edge Case Hit: Media file not found at {file_path}")
            raise FileNotFoundError(f"Media file not found: {file_path}")

        if not file_path.is_file():
            logger.error(f"[Local] Edge Case Hit: Path is not a file {file_path}")
            raise IsADirectoryError(
                f"Path is a directory, expected a file: {file_path}"
            )

        logger.debug(f"[Local] Media resolved successfully at {file_path}")
        return file_path

    def write_metadata(self, job_id: str, payload: dict) -> None:
        file_path = self.analyses_dir / f"{job_id}.json"
        logger.info(f"[Local] Writing metadata for job {job_id} to {file_path}")
        self._write_json(file_path, payload)

    def write_embeddings(self, job_id: str, payload: dict) -> None:
        file_path = self.embeddings_dir / f"{job_id}.json"
        logger.info(f"[Local] Writing embeddings for job {job_id} to {file_path}")
        self._write_json(file_path, payload)

    def _write_json(self, path: Path, data: dict) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"[Local] Successfully wrote {path.name}")
        except IOError as e:
            logger.error(f"[Local] Failed to write JSON to {path}: {e}")
            raise

    def cleanup_staging(self, uri: str) -> None:
        logger.debug(f"[Local] No staging cleanup required for local file: {uri}")


os.makedirs(settings.temp_dir, exist_ok=True)
os.makedirs(settings.content_directory, exist_ok=True)
os.makedirs(settings.embeddings_directory, exist_ok=True)
os.makedirs(settings.metadata_directory, exist_ok=True)
os.makedirs(settings.documents_directory, exist_ok=True)
