import logging
from pathlib import Path

from config import settings

from storage import FileSystemService, RemoteStorageService, StorageService

logger = logging.getLogger("storage_router")


class StorageRouter:
    """Factory class to route storage requests to the correct provider based on URI scheme."""

    @staticmethod
    def get_storage() -> StorageService:
        logger.info("Initializing storage provider based on environment configuration.")

        # Conditional Routing: If S3_BUCKET is defined, route to S3. Otherwise, use local.
        if settings.s3_bucket:
            logger.debug(
                "S3_BUCKET is configured. Initializing Remote Object (S3) provider."
            )

            # Create a staging directory within the local base directory (acting as temp_dir)
            staging_path = Path(settings.local_storage_base_directory) / "staging"
            staging_path.mkdir(parents=True, exist_ok=True)

            return RemoteStorageService(
                bucket_name=settings.s3_bucket,
                staging_dir=staging_path.resolve(),
            )

        else:
            logger.debug(
                "S3_BUCKET is not configured. Defaulting to Local Filesystem provider."
            )
            return FileSystemService(
                base_dir=Path(settings.local_storage_base_directory).resolve()
            )
