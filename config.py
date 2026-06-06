"""
config.py — All settings loaded from environment / .env file.
Single source of truth for tunable parameters.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Google AI ──────────────────────────────────────────────────
    google_api_key: str
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_flash_model: str = "gemini-2.0-flash-lite"
    embedding_dim: int = 1024

    # ── Qdrant ─────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "video_segments"

    # ── Whisper ────────────────────────────────────────────────────
    whisper_model: str = "base"

    # ── Scene Detection ────────────────────────────────────────────
    scene_threshold: float = 27.0
    min_scene_duration: float = 2.0
    max_scene_duration: float = 30.0
    temp_dir: str = "/tmp/svs_chunks"

    # ── Pipeline ───────────────────────────────────────────────────
    max_parallel_chunks: int = 4

    # ── Search Weights ─────────────────────────────────────────────
    search_video_weight: float = 0.50
    search_audio_weight: float = 0.30
    search_meta_weight: float = 0.20

    # ── API ────────────────────────────────────────────────────────
    cors_origins: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()

# Ensure temp dir exists at startup
os.makedirs(settings.temp_dir, exist_ok=True)