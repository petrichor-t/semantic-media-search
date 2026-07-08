from pathlib import Path

from semantic_media_search.domain.models import SearchHit
from semantic_media_search.indexing.vector_index import VectorIndex
from semantic_media_search.ml.text_encoder import TextEncoder


class SearchEngine:

    def __init__(
        self,
        text_encoder: TextEncoder,
        vector_index: VectorIndex,
        media_paths: list[Path],
    ) -> None:
        if not media_paths:
            raise ValueError(
                "Media paths must not be empty"
            )

        if vector_index.size != len(media_paths):
            raise ValueError(
                "Vector index size does not match "
                "media paths count"
            )

        self._text_encoder = text_encoder
        self._vector_index = vector_index
        self._media_paths = tuple(media_paths)

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise ValueError(
                "Limit must be greater than zero"
            )

        query_embedding = self._text_encoder.encode(
            query
        )

        scores, indices = self._vector_index.search(
            query_embedding,
            k=min(limit, len(self._media_paths)),
        )

        return [
            SearchHit(
                path=self._media_paths[int(index)],
                score=float(score),
            )
            for score, index in zip(scores, indices)
        ]