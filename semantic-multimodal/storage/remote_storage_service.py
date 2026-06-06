import logging
from pathlib import Path
from urllib.parse import urlparse

from storage import StorageService

logger = logging.getLogger(__name__)


class RemoteObjectProvider(StorageService):
    """Handles cloud storage operations (S3, GCS, etc)."""

    def __init__(self, bucket_name: str, staging_dir: Path):
        self.bucket_name = bucket_name
        self.staging_dir = staging_dir

        try:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"[Remote] Staging directory ready at {self.staging_dir}")
        except Exception as e:
            logger.error(f"[Remote] Failed to create staging directory: {e}")
            raise

    def fetch_media(self, uri: str) -> Path:
        logger.info(f"[Remote] Initiating fetch for remote URI: {uri}")
        parsed = urlparse(uri)
        object_key = parsed.path.lstrip("/")

        staging_path = self.staging_dir / object_key
        staging_path.parent.mkdir(parents=True, exist_ok=True)

        # Edge Case: File already staged
        if staging_path.exists():
            logger.info(f"[Remote] Using existing staged file at {staging_path}")
            return staging_path

        logger.debug(
            f"[Remote] Downloading s3://{self.bucket_name}/{object_key} -> {staging_path}"
        )

        # TODO: Implement actual boto3 download logic here.
        # Stubbing the download for demonstration:
        try:
            with open(staging_path, "w") as f:
                f.write("mock video content")
            logger.debug("[Remote] Download complete.")
        except Exception as e:
            logger.error(f"[Remote] Download failed for {uri}: {e}")
            raise

        return staging_path

    def write_metadata(self, job_id: str, payload: dict) -> None:
        object_key = f"analyses/{job_id}.json"
        logger.info(
            f"[Remote] Uploading metadata for job {job_id} to s3://{self.bucket_name}/{object_key}"
        )
        # TODO: Implement actual boto3 upload logic here

    def write_embeddings(self, job_id: str, payload: dict) -> None:
        object_key = f"embeddings/{job_id}.json"
        logger.info(
            f"[Remote] Uploading embeddings for job {job_id} to s3://{self.bucket_name}/{object_key}"
        )
        # TODO: Implement actual boto3 upload logic here

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
