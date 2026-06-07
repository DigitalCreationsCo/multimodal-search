FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set configurations for optimized container compilation
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Labels
LABEL org.opencontainers.image.title="Multimodal Search API"
LABEL org.opencontainers.image.description="FastAPI backend with FFmpeg, scenedetect, and Gemini AI"

# Add FFmpeg and the two OpenCV shared-library deps (libgl1, libglib2.0-0)
# that scenedetect needs when running headless (no display).
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

# Install dependencies (leverages Docker layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# ── Application source ──────────────────────────────────────────────────────
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000

ENV TEMP_DIR=/tmp/ms_chunks
RUN mkdir -p /tmp/ms_chunks

# Docker-native health check (also used by compose depends_on)
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# 2 workers is safe for most VM sizes — increase for higher concurrency needs
CMD ["uv", "run", "uvicorn", "multimodal_search.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
