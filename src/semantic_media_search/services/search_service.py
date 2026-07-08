import logging

import numpy as np

from semantic_media_search.domain.models import MediaFile, SearchResult
from semantic_media_search.indexing.vector_index import VectorIndex
from semantic_media_search.ml.text_encoder import TextEncoder
from semantic_media_search.storage.file_repository import FileRepository

logger = logging.getLogger(__name__)


class SearchService:
    """Orchestrates semantic search: query → embed → FAISS → metadata → results."""

    def __init__(
        self,
        text_encoder: TextEncoder,
        image_index: VectorIndex,
        file_repository: FileRepository,
    ) -> None:
        self._text_encoder = text_encoder
        self._image_index = image_index
        self._file_repository = file_repository

    def update_index(self, image_index: VectorIndex) -> None:
        """Replace the current index with a newly built one."""
        self._image_index = image_index

    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """
        Execute semantic search and return ranked results.

        Args:
            query: Russian/English text query.
            top_k: Maximum number of results to return.
            min_score: Optional minimum cosine similarity threshold.

        Returns:
            List of SearchResult sorted by descending score.
        """
        if not query.strip():
            raise ValueError("Search query must not be empty")

        if self._image_index.size == 0:
            logger.warning("Index is empty, returning no results")
            return []

        # 1. Encode query
        query_vector = self._text_encoder.encode(query)
        logger.info("Encoded query: '%s'", query)

        # 2. FAISS search
        scores, file_ids = self._image_index.search(
            query_vector, top_k=top_k
        )

        # 3. Filter invalid IDs (-1) and build pairs
        ranked_pairs: list[tuple[float, int]] = [
            (float(scores[i]), int(file_ids[i]))
            for i in range(len(scores))
            if file_ids[i] != -1
        ]

        if not ranked_pairs:
            return []

        # 4. Fetch metadata from SQLite
        unique_ids = list({fid for _, fid in ranked_pairs})
        media_files = self._file_repository.find_by_ids(unique_ids)

        files_by_id: dict[int, MediaFile] = {
            mf.id: mf for mf in media_files if mf.id is not None
        }

        # 5. Build results in FAISS ranking order
        results: list[SearchResult] = []
        for score, file_id in ranked_pairs:
            if file_id not in files_by_id:
                continue
            if min_score is not None and score < min_score:
                continue

            mf = files_by_id[file_id]
            results.append(
                SearchResult(
                    file_id=file_id,
                    path=mf.path,
                    name=mf.name,
                    media_type=mf.media_type,
                    score=score,
                )
            )

        logger.info("Search returned %d results for '%s'", len(results), query)
        return results