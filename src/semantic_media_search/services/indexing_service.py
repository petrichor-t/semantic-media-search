import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from semantic_media_search.domain.models import IndexingProgress, MediaFile
from semantic_media_search.indexing.vector_index import VectorIndex
from semantic_media_search.ml.image_encoder import ImageEncoder
from semantic_media_search.scanning.media_scanner import MediaScanner
from semantic_media_search.storage.file_repository import FileRepository

logger = logging.getLogger(__name__)


def _batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class IndexingService:
    """Orchestrates full image indexing pipeline."""

    def __init__(
        self,
        scanner: MediaScanner,
        repository: FileRepository,
        image_encoder: ImageEncoder,
        index_path: Path,
        model_name: str,
        batch_size: int = 16,
    ) -> None:
        self._scanner = scanner
        self._repository = repository
        self._image_encoder = image_encoder
        self._index_path = index_path
        self._model_name = model_name
        self._batch_size = batch_size

    def rebuild_image_index(
        self,
        root: Path,
        progress_callback: Callable[[IndexingProgress], None] | None = None,
    ) -> int:
        """
        Full pipeline: scan → save metadata → encode → build FAISS index → persist.

        Returns the number of successfully indexed images.
        """
        logger.info("Scanning folder: %s", root)

        # 1. Scan
        files = self._scanner.scan_images(root)
        if not files:
            logger.warning("No images found in %s", root)
            return 0
        logger.info("Found %d images", len(files))

        # 2. Rebuild SQLite records
        self._repository.delete_all()
        files_with_ids = self._repository.save_all(files)
        logger.info("Saved %d file records to database", len(files_with_ids))

        # 3. Build new FAISS index
        dimension = self._image_encoder.dimension
        index = VectorIndex(dimension)
        total = len(files_with_ids)

        indexed_count = 0
        for batch in _batched(files_with_ids, self._batch_size):
            batch: list[MediaFile]
            try:
                paths = [f.path for f in batch]
                embeddings = self._image_encoder.encode_batch(paths)
                file_ids = np.array([f.id for f in batch], dtype=np.int64)
                index.add(embeddings, file_ids)

                successful_ids = [f.id for f in batch]
                self._repository.update_index_status(
                    successful_ids,
                    "indexed",
                    self._model_name,
                )
                indexed_count += len(batch)

            except Exception:
                logger.exception(
                    "Batch encoding failed for %d files, falling back to per-file",
                    len(batch),
                )
                # Fallback: encode one by one
                for f in batch:
                    try:
                        emb = self._image_encoder.encode(f.path)
                        emb_2d = emb.reshape(1, -1)
                        fid = np.array([f.id], dtype=np.int64)
                        index.add(emb_2d, fid)
                        self._repository.update_index_status(
                            [f.id], "indexed", self._model_name
                        )
                        indexed_count += 1
                    except Exception:
                        logger.exception("Failed to encode: %s", f.path)
                        self._repository.update_index_status(
                            [f.id], "failed"
                        )

            # Report progress
            if progress_callback:
                progress_callback(
                    IndexingProgress(
                        processed=min(
                            files_with_ids.index(batch[-1]) + 1, total
                        ),
                        total=total,
                        current_file=str(batch[-1].name) if batch else "",
                    )
                )

        # 4. Save index (atomic via tempfile — safe for Unicode paths)
        index.save(self._index_path)
        logger.info(
            "Index saved: %s (%d vectors)", self._index_path, index.size
        )

        return indexed_count

    def load_or_create_index(self) -> VectorIndex:
        """Load existing index from disk, or create an empty one."""
        if self._index_path.exists():
            logger.info("Loading existing index from %s", self._index_path)
            return VectorIndex.load(self._index_path)
        logger.info("Creating new empty index")
        return VectorIndex(self._image_encoder.dimension)