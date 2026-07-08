import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class TextEncoder:

    def __init__(
        self,
        model: SentenceTransformer,
    ) -> None:
        self._model = model

    def encode(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        if not text.strip():
            raise ValueError(
                "Text must not be empty"
            )

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )