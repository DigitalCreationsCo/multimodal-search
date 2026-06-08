"""
pipeline/downloader.py

URL resolution and content downloading for the ingest pipeline.

Supports two kinds of URLs:
  1. **Video platform URLs** (YouTube, Vimeo, etc.) — resolved to direct media
     URLs via yt-dlp before downloading.
  2. **Direct media URLs** (raw .mp4, .mp3, .wav, etc.) — downloaded as-is,
     with content-type validation to catch misdirected URLs early.

Every download is validated:
  - Content-Type header must indicate media (video/, audio/, octet-stream).
  - File size must exceed a minimum threshold (256 bytes).
  - File extension is determined dynamically from the Content-Type header
    rather than hardcoded to .mp4.
"""

import logging
import os
import re
from typing import NamedTuple

import httpx
import yt_dlp

logger = logging.getLogger(__name__)


# ── Resolved URL result ────────────────────────────────────────────────────────


class ResolvedUrl(NamedTuple):
    """Result of resolving a user-provided URL to a downloadable media URL."""

    url: str  # Direct media URL (downloadable via HTTP GET)
    extension: str  # Suggested file extension, e.g. ".mp4"
    title: str  # Human-readable title (empty string if unknown)


# ── Platform detection ─────────────────────────────────────────────────────────

_PLATFORM_PATTERNS = re.compile(
    r"(youtube\.com|youtu\.be|vimeo\.com|dailymotion\.com|twitch\.tv)",
    re.I,
)

# ── MIME type → file extension mapping ─────────────────────────────────────────

_MIME_EXT_MAP: dict[str, str] = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/x-msvideo": ".avi",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/x-flv": ".flv",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
}

# Bare minimum file size (bytes) — anything smaller is almost certainly
# not a real media file (e.g. an HTML error page or redirect body).
_MIN_FILE_SIZE = 256


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def is_platform_url(url: str) -> bool:
    """Return True if *url* belongs to a known video-platform domain."""
    return bool(_PLATFORM_PATTERNS.search(url))


def resolve_url(url: str) -> ResolvedUrl:
    """
    Resolve a user-provided URL to a directly-downloadable media URL.

    For platform URLs (YouTube, Vimeo, etc.) this uses yt-dlp to extract
    the actual media file URL.  For all other URLs the original URL is
    returned unchanged.

    Returns
    -------
    ResolvedUrl
        ``.url``         — direct media URL (downloadable via HTTP GET).
        ``.extension``   — suggested file extension, e.g. ``".mp4"``.
        ``.title``       — human-readable title, or empty string if unknown.
    """
    if is_platform_url(url):
        logger.info("Resolving platform URL via yt-dlp: %s", url)
        return _extract_with_ytdlp(url)

    ext = _extension_from_url(url)
    logger.debug("Direct media URL — using as-is: %s", url)
    return ResolvedUrl(url=url, extension=ext, title="")


def download_content(url: str, dest_path: str) -> str:
    """
    Download media from *url* to *dest_path* with content validation.

    Validates:
      * HTTP status is 2xx.
      * Content-Type header indicates video, audio, or octet-stream.
      * Downloaded file is larger than ``_MIN_FILE_SIZE`` bytes.

    The file extension in *dest_path* may be overridden based on the
    Content-Type header of the response (e.g. if the caller suggested
    ``.mp4`` but the server returns ``audio/mpeg`` the extension will
    be corrected to ``.mp3``).

    Returns
    -------
    str
        The actual path of the downloaded file (may differ from
        *dest_path* if the extension was corrected).
    """
    logger.info("Downloading media from %s", url)

    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()

        # ── Content-Type validation ────────────────────────────────────
        content_type = resp.headers.get("content-type", "")
        if not _is_media_content_type(content_type):
            _raise_not_media(url, content_type)

        # ── Determine actual file extension ────────────────────────────
        actual_ext = _extension_from_content_type(content_type, url)
        base, _ = os.path.splitext(dest_path)
        if actual_ext != os.path.splitext(dest_path)[1]:
            dest_path = base + actual_ext
            logger.debug("Corrected file extension to %s", actual_ext)

        # ── Stream to disk ─────────────────────────────────────────────
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)

    # ── File-size sanity check ─────────────────────────────────────────
    file_size = os.path.getsize(dest_path)
    if file_size < _MIN_FILE_SIZE:
        os.remove(dest_path)
        raise ValueError(
            f"Downloaded file is only {file_size} bytes "
            f"(minimum expected: {_MIN_FILE_SIZE} bytes). "
            f"The URL may have returned an empty or error page "
            f"instead of media content."
        )

    logger.info("Download complete: %s (%d bytes)", dest_path, file_size)
    return dest_path


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_with_ytdlp(url: str) -> ResolvedUrl:
    """
    Use yt-dlp to obtain a direct media URL, extension, and title.

    Raises
    ------
    ValueError
        If yt-dlp cannot extract a media URL (private video, geo-block,
        removed, etc.).
    """
    ydl_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise ValueError(
            f"Could not extract a downloadable video from this URL. "
            f"You may need to download the file manually and use the "
            f"file-upload endpoint instead.  Details: {exc}"
        ) from exc

    direct_url = info.get("url", "")
    if not direct_url:
        raise ValueError(
            "yt-dlp did not return a downloadable media URL. "
            "The video may be private, age-restricted, or region-locked."
        )

    ext = info.get("ext", "mp4")
    title = info.get("title", "")
    if title:
        logger.info("Resolved: %r (%.2f s)", title, info.get("duration", 0))

    return ResolvedUrl(url=direct_url, extension=f".{ext}", title=title)


def _is_media_content_type(content_type: str) -> bool:
    """Return True if the Content-Type looks like video, audio, or binary."""
    base = content_type.split(";")[0].strip().lower()
    return any(
        base.startswith(prefix)
        for prefix in ("video/", "audio/", "application/octet-stream")
    )


def _raise_not_media(url: str, content_type: str) -> None:
    """Raise a descriptive ``ValueError`` for non-media responses."""
    if "text/html" in content_type.lower():
        msg = (
            f"The URL returned an HTML page (Content-Type: {content_type}) "
            f"instead of a media file. "
        )
        if is_platform_url(url):
            msg += (
                "This looks like a video-platform URL but yt-dlp extraction "
                "may have failed.  Try downloading the file manually and "
                "using the /ingest/file endpoint instead."
            )
        else:
            msg += (
                "This URL does not point directly to a video or audio file. "
                "Provide a URL ending in .mp4, .mp3, .wav, etc., or use "
                "the /ingest/file endpoint to upload a local file."
            )
    else:
        msg = (
            f"The URL returned '{content_type}' instead of a video or audio "
            f"file.  Ensure the URL points directly to a media file."
        )
    raise ValueError(msg)


def _extension_from_content_type(content_type: str, fallback_url: str) -> str:
    """Map Content-Type to a file extension, falling back to the URL suffix."""
    base_type = content_type.split(";")[0].strip().lower()
    ext = _MIME_EXT_MAP.get(base_type)
    if ext:
        return ext
    return _extension_from_url(fallback_url)


def _extension_from_url(url: str) -> str:
    """Extract ``.ext`` from the last path segment of a URL."""
    path = url.split("?")[0].split("#")[0]
    _, ext = os.path.splitext(path)
    if ext and len(ext) <= 5:
        return ext.lower()
    return ".mp4"  # sensible default
