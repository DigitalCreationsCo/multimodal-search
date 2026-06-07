import json
import logging
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from config import settings

from storage import StorageService

logger = logging.getLogger(__name__)


class RemoteObjectProvider(StorageService):
    """Handles cloud storage operations (S3, GCS, etc)."""

    def __init__(self, bucket_name: str, staging_dir: Path):
        self.bucket_name = bucket_name
        self.staging_dir = staging_dir
        self.s3_client = boto3.client("s3")

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

        try:
            self.s3_client.download_file(
                self.bucket_name, object_key, str(staging_path)
            )
            return staging_path
            logger.debug("[Remote] Download complete.")
        except Exception as e:
            logger.error(f"[Remote] Download failed for {uri}: {e}")
            raise

        return staging_path

    def fetch_metadata(self, job_id: str) -> Dict[str, Any]:
        logger.info(f"[Remote] Fetching metadata for job: {job_id}")
        return self._fetch_json(f"{settings.metadata_directory}/{job_id}.json")

    def write_metadata(self, job_id: str, payload: Dict[str, Any]) -> None:
        logger.info(f"[Remote] Writing metadata for job: {job_id}")
        self._write_json(f"{settings.metadata_directory}/{job_id}.json", payload)

    def fetch_embeddings(self, job_id: str) -> Dict[str, Any]:
        logger.info(f"[Remote] Fetching embeddings for job: {job_id}")
        return self._fetch_json(f"{settings.embeddings_directory}/{job_id}.json")

    def write_embeddings(self, job_id: str, payload: Dict[str, Any]) -> None:
        logger.info(f"[Remote] Writing embeddings for job: {job_id}")
        self._write_json(f"{settings.embeddings_directory}/{job_id}.json", payload)

    def fetch_documents(self, doc_id: str) -> Dict[str, Any]:
        """Fetches raw documents/transcripts."""
        logger.info(f"[Remote] Fetching document: {doc_id}")
        return self._fetch_json(f"{settings.documents_directory}/{doc_id}.json")

    def write_documents(self, doc_id: str, payload: Dict[str, Any]) -> None:
        """Writes raw documents/transcripts."""
        logger.info(f"[Remote] Writing document: {doc_id}")
        self._write_json(f"{settings.documents_directory}/{doc_id}.json", payload)

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

    def _fetch_json(self, key: str) -> Dict[str, Any]:
        """Fetches and decodes JSON from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            logger.error(f"[Remote] Failed to fetch JSON from {key}: {e}")
            raise

    def _write_json(self, key: str, data: Dict[str, Any]) -> None:
        """Serializes and uploads JSON to S3."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=key, Body=json.dumps(data, indent=2)
            )
        except ClientError as e:
            logger.error(f"[Remote] Failed to write JSON to {key}: {e}")
            raise
