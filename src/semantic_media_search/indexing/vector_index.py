import os as _os
import shutil
import tempfile
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray


class VectorIndex:

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        base_index = faiss.IndexFlatIP(dimension)
        self._index = faiss.IndexIDMap2(base_index)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def add(
        self,
        embeddings: NDArray[np.float32],
        file_ids: NDArray[np.int64],
    ) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        ids = np.asarray(file_ids, dtype=np.int64)

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2-dimensional array"
            )

        if ids.ndim != 1:
            raise ValueError(
                "File IDs must be a 1-dimensional array"
            )

        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(
                f"Number of vectors ({vectors.shape[0]}) "
                f"does not match number of IDs ({ids.shape[0]})"
            )

        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension {vectors.shape[1]} "
                f"does not match index dimension {self._dimension}"
            )

        vectors = vectors.copy()
        faiss.normalize_L2(vectors)
        self._index.add_with_ids(vectors, ids)

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 5,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self._index.ntotal == 0:
            raise RuntimeError("Vector index is empty")

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2:
            raise ValueError(
                "Query embedding must be a 1D or 2D array"
            )

        if query.shape[1] != self._dimension:
            raise ValueError(
                f"Query dimension {query.shape[1]} "
                f"does not match index dimension {self._dimension}"
            )

        query = query.copy()
        faiss.normalize_L2(query)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        return scores[0], indices[0]

    def save(self, path: Path) -> None:
        """Save index atomically using a temp file to avoid Unicode path issues."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            suffix=".faiss", prefix="index_tmp_", dir=str(path.parent)
        )
        try:
            faiss.write_index(self._index, tmp_name)
        finally:
            _os.close(tmp_fd)
        shutil.move(tmp_name, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        if not path.exists():
            raise FileNotFoundError(
                f"Index file does not exist: {path}"
            )

        faiss_index = faiss.read_index(str(path))

        if not isinstance(faiss_index, faiss.IndexIDMap2):
            raise ValueError(
                "Loaded index is not an IndexIDMap2"
            )

        base_index = faiss.downcast_index(faiss_index.index)
        if base_index is None:
            base_index = faiss_index.index

        dimension = base_index.d

        obj = cls.__new__(cls)
        obj._dimension = dimension
        obj._index = faiss_index
        return obj