"""
config.py — All settings loaded from environment / .env file.
Single source of truth for tunable parameters.
"""

import os
from typing import Optional

from google import genai
from google.genai import types
from pydantic_settings import BaseSettings, SettingsConfigDict

types
Client = genai.Client

# Module-level client — initialised once
_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── Google AI ──────────────────────────────────────────────────
    google_api_key: str
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_flash_model: str = "gemini-2.5-flash"
    embedding_dimension: int = 1024

    # ── OpenSearch ──────────────────────────────────────────────────
    index_name: str = "multimodal_search"
    opensearch_host: str
    opensearch_port: int
    opensearch_user: str
    opensearch_password: str
    opensearch_use_ssl: bool
    opensearch_verify_certs: bool
    index_number_of_shards: int = 1
    index_number_of_replicas: int = 0

    # ── Scene Detection ────────────────────────────────────────────
    scene_threshold: float = 27.0
    min_scene_duration: float = 2.0
    max_scene_duration: float = 30.0
    temp_dir: str = "/tmp/ms_chunks"

    # ── Pipeline ───────────────────────────────────────────────────
    max_parallel_chunks: int = 4
    max_attempts: int = 20

    # ── Search Weights ─────────────────────────────────────────────
    search_video_weight: float = 0.50
    search_audio_weight: float = 0.30
    search_meta_weight: float = 0.20

    # ── File System ────────────────────────────────────────────────────────
    local_storage_base_directory: str = "."
    content_directory: str = "content"
    embeddings_directory: str = "embeddings"
    metadata_directory: str = "metadata"
    documents_directory: str = "documents"

    # ── Object Storage ────────────────────────────────────────────────────────
    s3_region: str
    s3_bucket: str
    s3_content_prefix: str

    # ── API ────────────────────────────────────────────────────────
    cors_origins: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()

# Ensure temp dir exists at startup
os.makedirs(settings.temp_dir, exist_ok=True)
