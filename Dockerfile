# ── Stage: runtime ─────────────────────────────────────────────────────────
# python:3.13-slim gives us a small Debian base.
# We add FFmpeg and the two OpenCV shared-library deps (libgl1, libglib2.0-0)
# that scenedetect needs when running headless (no display).
FROM python:3.13-slim

# Labels
LABEL org.opencontainers.image.title="Semantic Video Search API"
LABEL org.opencontainers.image.description="FastAPI backend with FFmpeg, scenedetect, and Gemini AI"

# ── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    # OpenCV headless runtime (scenedetect uses cv2 under the hood)
    libgl1 \
    libglib2.0-0 \
    # curl for Docker health check
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python env ──────────────────────────────────────────────────────────────
WORKDIR /app

# Copy dependency manifest first — Docker cache busts only when it changes
COPY pyproject.toml ./

# Install the package and all its dependencies
# --no-cache-dir keeps the image lean
RUN pip install --no-cache-dir -e .

# ── Application source ──────────────────────────────────────────────────────
COPY . .

# ── Runtime config ──────────────────────────────────────────────────────────
EXPOSE 8000

# Temp dir for video chunks & thumbnails (overrideable via compose volume)
ENV TEMP_DIR=/tmp/svs_chunks
RUN mkdir -p /tmp/svs_chunks

# Docker-native health check (also used by compose depends_on)
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# 2 workers is safe for most VM sizes — increase for higher concurrency needs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]