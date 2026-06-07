"""
search/indexer.py

The single write path to OpenSearch. Handles index management, alias-based
zero-downtime reindexing, single-document writes with retry, and bulk
operations with per-document error reporting.

Alias architecture:
  {index_name}_read   — concrete index that search queries hit
  {index_name}_write  — concrete index that writes target

During a reindex with delete_existing=True:
  1. Create fresh versioned index  {index_name}_v{timestamp}
  2. Move write alias to new index
  3. Bulk-index all documents from storage artifacts
  4. Atomically swap read alias from old → new
  5. Delete old index
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError

from multimodal_search.config import settings
from multimodal_search.models import (
    ContentSegment,
    OpenSearchDocument,
    SegmentContentMetadata,
    SegmentGeneratedMetadata,
)
from multimodal_search.search.index import get_index_body
from multimodal_search.storage.storage_router import StorageRouter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Result type
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class IndexResult:
    """Outcome of a single document index or delete operation."""

    document_id: str
    succeeded: bool
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Indexer
# ═══════════════════════════════════════════════════════════════════════════════


class Indexer:
    """
    OpenSearch index manager: aliases, single/bulk writes, zero-downtime reindex.
    """

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        storage: Optional[Any] = None,
    ):
        self.client = client
        self.index_name = index_name
        self.read_alias = f"{index_name}_read"
        self.write_alias = f"{index_name}_write"
        self.storage = storage or StorageRouter.get_storage()

    # ── Alias resolution ────────────────────────────────────────────────

    def resolve_read_index(self) -> str:
        """Resolve the read alias to the concrete index name."""
        try:
            response = self.client.indices.get_alias(name=self.read_alias)
            aliased = list(response.keys())
            if aliased:
                return aliased[0]
        except (NotFoundError, RequestError):
            pass
        # Fallback: the base index_name itself (pre-alias migration)
        return self.index_name

    def resolve_write_index(self) -> str:
        """Resolve the write alias to the concrete index name."""
        try:
            response = self.client.indices.get_alias(name=self.write_alias)
            aliased = list(response.keys())
            if aliased:
                return aliased[0]
        except (NotFoundError, RequestError):
            pass
        return self.index_name

    # ── Index lifecycle ──────────────────────────────────────────────────

    def ensure_index(self) -> str:
        """
        Ensure at least one concrete index and its aliases exist.

        Returns the concrete index name ready for writes.
        Safe to call on every startup.
        """
        # 1. Read alias already resolves → done
        try:
            existing = self.client.indices.get_alias(name=self.read_alias)
            concrete = list(existing.keys())[0]
            logger.info("Read alias '%s' → concrete index '%s'", self.read_alias, concrete)
            return concrete
        except (NotFoundError, RequestError):
            pass

        # 2. Bare index exists (pre-alias migration) → set aliases on it
        if self.client.indices.exists(index=self.index_name):
            logger.info("Migrating bare index '%s' → setting aliases", self.index_name)
            self._set_aliases(self.index_name)
            return self.index_name

        # 3. Nothing exists → create versioned index + aliases
        concrete = self._create_versioned_index()
        self._set_aliases(concrete)
        logger.info("Created index '%s' with read/write aliases", concrete)
        return concrete

    def _create_versioned_index(self) -> str:
        """Create a new concrete index with the full mapping from index.py."""
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        concrete = f"{self.index_name}_{suffix}"
        body = get_index_body(settings.embedding_dimension)
        # Apply configurable shards/replicas from settings
        body["settings"]["index"]["number_of_shards"] = settings.index_number_of_shards
        body["settings"]["index"]["number_of_replicas"] = settings.index_number_of_replicas
        self.client.indices.create(index=concrete, body=body)
        return concrete

    def _set_aliases(self, concrete: str) -> None:
        """Point both aliases at *concrete*."""
        self.client.indices.put_alias(index=concrete, name=self.read_alias)
        self.client.indices.put_alias(index=concrete, name=self.write_alias)

    def _swap_aliases(self, new_index: str, old_index: Optional[str] = None) -> None:
        """
        Atomically move read/write aliases from old_index to new_index.

        This is the zero-downtime hand-off: between the start and end of
        this call, readers may see either index, but the window is atomic
        at the OpenSearch cluster level.
        """
        actions = [
            {"add": {"index": new_index, "alias": self.read_alias}},
            {"add": {"index": new_index, "alias": self.write_alias}},
        ]
        if old_index:
            actions.insert(0, {"remove": {"index": old_index, "alias": self.read_alias}})
            actions.insert(1, {"remove": {"index": old_index, "alias": self.write_alias}})

        self.client.indices.update_aliases(body={"actions": actions})
        logger.info("Aliases swapped: read/write → '%s' (old: %s)", new_index, old_index)

    # ── Single document operations ───────────────────────────────────────

    def index_document(
        self,
        doc: OpenSearchDocument,
        retries: int = 3,
        target_index: Optional[str] = None,
    ) -> IndexResult:
        """
        Index a single document with retry.

        Uses explicit ``_id = doc.documentId`` for idempotency —
        re-indexing the same document overwrites rather than duplicating.

        Args:
            doc: The document to index.
            retries: Number of retry attempts on transient failures.
            target_index: Concrete index name (for reindex flows where
                          the write alias hasn't been fully swapped yet).
                          Defaults to resolve_write_index().
        """
        doc_id = doc.documentId
        write_index = target_index or self.resolve_write_index()
        last_error: Optional[str] = None

        for attempt in range(retries):
            try:
                self.client.index(
                    index=write_index,
                    id=doc_id,
                    body=doc.model_dump(mode="json"),
                    refresh=False,
                )
                return IndexResult(document_id=doc_id, succeeded=True)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Index doc %s attempt %d/%d failed: %s",
                    doc_id,
                    attempt + 1,
                    retries,
                    last_error,
                )
                if attempt < retries - 1:
                    time.sleep(0.5 * (2**attempt))

        logger.error(
            "Failed to index doc %s after %d retries: %s", doc_id, retries, last_error
        )
        return IndexResult(document_id=doc_id, succeeded=False, error=last_error)

    def delete_document(self, document_id: str, retries: int = 3) -> IndexResult:
        """Delete a document by its ``documentId`` field. Idempotent."""
        write_index = self.resolve_write_index()
        last_error: Optional[str] = None

        for attempt in range(retries):
            try:
                self.client.delete_by_query(
                    index=write_index,
                    body={"query": {"term": {"documentId": document_id}}},
                    refresh=False,
                )
                return IndexResult(document_id=document_id, succeeded=True)
            except NotFoundError:
                # Document already gone — idempotent success
                return IndexResult(document_id=document_id, succeeded=True)
            except Exception as exc:
                last_error = str(exc)
                if attempt < retries - 1:
                    time.sleep(0.5 * (2**attempt))

        return IndexResult(
            document_id=document_id, succeeded=False, error=last_error
        )

    # ── Bulk operations ──────────────────────────────────────────────────

    def bulk_index(
        self,
        docs: List[OpenSearchDocument],
        retries: int = 3,
        target_index: Optional[str] = None,
    ) -> List[IndexResult]:
        """
        Bulk-index documents with per-document error reporting.

        Each document uses ``_id = doc.documentId`` so repeated calls
        are idempotent (upsert semantics).

        Returns one ``IndexResult`` per input document. The caller can
        inspect ``succeeded`` and ``error`` for partial-failure handling.
        """
        if not docs:
            return []

        write_index = target_index or self.resolve_write_index()
        results: List[IndexResult] = []

        # Build bulk body as a flat list of action + document pairs
        bulk_body: List[Any] = []
        for doc in docs:
            bulk_body.append({"index": {"_index": write_index, "_id": doc.documentId}})
            bulk_body.append(doc.model_dump(mode="json"))

        # Execute with retry
        response: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None

        for attempt in range(retries):
            try:
                response = self.client.bulk(body=bulk_body, refresh=False)
                break
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Bulk index attempt %d/%d failed: %s",
                    attempt + 1,
                    retries,
                    last_error,
                )
                if attempt < retries - 1:
                    time.sleep(1.0 * (2**attempt))

        if response is None:
            # All retries exhausted — mark every document as failed
            for doc in docs:
                results.append(
                    IndexResult(
                        document_id=doc.documentId,
                        succeeded=False,
                        error=last_error,
                    )
                )
            return results

        # Parse per-item errors from the bulk response
        items = response.get("items", [])
        for i, item in enumerate(items):
            op = item.get("index", {})
            doc_id = op.get("_id", docs[i].documentId if i < len(docs) else "unknown")
            err = op.get("error")
            if err:
                err_msg = str(err)
                results.append(
                    IndexResult(document_id=doc_id, succeeded=False, error=err_msg)
                )
            else:
                results.append(IndexResult(document_id=doc_id, succeeded=True))

        failed = sum(1 for r in results if not r.succeeded)
        if failed:
            logger.error(
                "Bulk index completed with %d/%d errors", failed, len(docs)
            )

        return results

    # ── Reindex ──────────────────────────────────────────────────────────

    def reindex(self, delete_existing: bool = False) -> Dict[str, Any]:
        """
        Rebuild the index from serialized artifacts in storage.

        Reads ``OpenSearchDocument`` JSON files from the storage
        ``documents/`` directory and indexes them.  No AI calls, no
        FFmpeg — just deserialize + bulk index.

        Args:
            delete_existing:
                If True — create a fresh versioned index, bulk-index all
                documents into it, then atomically swap the read alias
                from the old index to the new one, and delete the old.
                If False — bulk-index into the existing index
                (append/update, no downtime, no alias swap).

        Returns:
            Dict with keys: old_index, new_index, docs_found, docs_indexed,
            succeeded, failed, errors.
        """
        old_index: Optional[str] = None
        if delete_existing:
            old_index = self.resolve_read_index()

        result: Dict[str, Any] = {
            "old_index": old_index,
            "new_index": old_index,
            "docs_found": 0,
            "docs_indexed": 0,
            "succeeded": 0,
            "failed": 0,
            "errors": [],
        }

        # 1. Load all document artifacts from storage
        doc_files = self.storage.list_files(settings.documents_directory)
        result["docs_found"] = len(doc_files)

        if not doc_files:
            logger.warning(
                "No document artifacts found in '%s' — nothing to reindex",
                settings.documents_directory,
            )
            return result

        docs: List[OpenSearchDocument] = []
        for fname in doc_files:
            # Strip .json extension if present (list_files returns filenames
            # with extensions, but fetch_documents appends .json internally)
            doc_id = fname
            if doc_id.endswith(".json"):
                doc_id = doc_id[:-5]
            try:
                raw = self.storage.fetch_documents(doc_id)
                docs.append(OpenSearchDocument(**raw))
            except Exception as exc:
                err = f"Failed to load document '{fname}': {exc}"
                logger.error(err)
                result["errors"].append(err)

        if not docs:
            logger.warning("No valid documents could be loaded from storage")
            return result

        # 2. If delete_existing: create new index and do alias dance
        new_index: Optional[str] = None
        if delete_existing:
            new_index = self._create_versioned_index()
            result["new_index"] = new_index

            # Move write alias to new index (reads still hit old_index)
            self.client.indices.update_aliases(
                body={
                    "actions": [
                        {"remove": {"index": old_index, "alias": self.write_alias}},
                        {"add": {"index": new_index, "alias": self.write_alias}},
                    ]
                }
            )
            logger.info(
                "Write alias moved to '%s'; reads still on '%s'",
                new_index,
                old_index,
            )

            # Bulk index into new_index (via explicit target_index)
            index_results = self.bulk_index(docs, target_index=new_index)

            # Atomically swap read alias and remove write from old
            self.client.indices.update_aliases(
                body={
                    "actions": [
                        {"remove": {"index": old_index, "alias": self.read_alias}},
                        {"add": {"index": new_index, "alias": self.read_alias}},
                    ]
                }
            )
            logger.info("Read alias moved to '%s'", new_index)

            # Delete old index
            if self.client.indices.exists(index=old_index):
                self.client.indices.delete(index=old_index)
                logger.info("Deleted old index '%s'", old_index)
        else:
            # In-place index (append/update, no alias changes)
            index_results = self.bulk_index(docs)

        # 3. Aggregate per-document results
        for ir in index_results:
            if ir.succeeded:
                result["succeeded"] += 1
            else:
                result["failed"] += 1
                result["errors"].append(ir.error or "unknown error")
        result["docs_indexed"] = len(index_results)

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Document builder
# ═══════════════════════════════════════════════════════════════════════════════


def build_document_from_results(
    content_id: str,
    file_name: str,
    content_path: str,
    segments: List[Dict[str, Any]],
    storage: Optional[Any] = None,
) -> OpenSearchDocument:
    """
    Assemble an ``OpenSearchDocument`` from the pipeline's per-segment results.

    The pipeline (``_process_chunk``) returns flat dicts.  This function
    maps them into the canonical nested model (``ContentSegment`` →
    ``SegmentContentMetadata`` + ``SegmentGeneratedMetadata`` + 3 embeddings)
    and wraps them in a document with file-level metadata.

    Args:
        content_id:  Unique document identifier (same as the ingest job_id).
        file_name:   Human-readable filename (e.g. ``"lecture.mp4"``).
        content_path: Local or remote path to the source media file.
        segments:    Sorted list of segment dicts from ``_process_chunk``.
        storage:     Optional ``StorageService`` instance for resolving
                     thumbnail URIs.  Created from ``StorageRouter`` if
                     not provided.

    Returns:
        A populated ``OpenSearchDocument`` ready for ``Indexer.index_document()``.
    """
    if storage is None:
        storage = StorageRouter.get_storage()

    # ── File-level metadata ──────────────────────────────────────────────
    try:
        size_bytes = os.path.getsize(content_path) if os.path.exists(content_path) else 0
    except (OSError, TypeError):
        size_bytes = 0

    ext = Path(file_name).suffix.lower()
    if ext in (".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"):
        content_type = "video"
    elif ext in (".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"):
        content_type = "audio"
    else:
        content_type = "video"

    total_duration = segments[-1]["end_time"] if segments else 0.0
    date_ingested = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Build segments ───────────────────────────────────────────────────
    content_segments: List[ContentSegment] = []
    for seg in segments:
        has_video = seg.get("video_embedding") is not None
        thumbnail_path = seg.get("thumbnail_path", "")

        # Resolve thumbnail to a path relative to temp_dir for serving
        thumbnail_uri: Optional[str] = None
        if thumbnail_path:
            try:
                thumbnail_uri = str(Path(thumbnail_path).relative_to(settings.temp_dir))
            except ValueError:
                thumbnail_uri = thumbnail_path

        seg_duration = seg["end_time"] - seg["start_time"]

        content_metadata = SegmentContentMetadata(
            sizeBytes=0,  # chunk-file tracking not currently exposed by pipeline
            durationSec=seg_duration,
            thumbnailUri=thumbnail_uri,
            hasVideo=has_video,
            hasAudio=True,
            mediaType=content_type,
        )

        generated_metadata = SegmentGeneratedMetadata(
            title=seg.get("title", ""),
            summary=seg.get("summary", ""),
            keywords=seg.get("keywords", []),
            mood=seg.get("mood", "ambient"),
            hasSpeech=seg.get("has_speech", True),
            transcript=seg.get("transcript", ""),
            confidence=1.0,
        )

        content_segments.append(
            ContentSegment(
                segmentIndex=seg["chunk_index"],
                startSec=seg["start_time"],
                endSec=seg["end_time"],
                contentMetadata=content_metadata,
                generatedMetadata=generated_metadata,
                videoEmbedding=seg.get("video_embedding"),
                audioEmbedding=seg.get("audio_embedding", []),
                textEmbedding=seg.get("text_embedding", []),
            )
        )

    return OpenSearchDocument(
        documentId=content_id,
        fileName=file_name,
        uri=content_path,
        contentType=content_type,
        sizeBytes=size_bytes,
        durationSec=total_duration,
        dateIngested=date_ingested,
        segments=content_segments,
    )
