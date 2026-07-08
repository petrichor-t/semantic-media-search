from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer


class ImageEncoder:

    def __init__(
        self,
        model: SentenceTransformer,
    ) -> None:
        self._model = model
        self._dimension = model.get_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        image_path: Path,
    ) -> NDArray[np.float32]:
        embeddings = self.encode_batch(
            [image_path]
        )

        return embeddings[0]

    def encode_batch(
        self,
        image_paths: list[Path],
    ) -> NDArray[np.float32]:
        images = [
            self._load_image(image_path)
            for image_path in image_paths
        ]

        embeddings = self._model.encode(
            images,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    @staticmethod
    def _load_image(image_path: Path) -> Image.Image:
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Image path is not a file: {image_path}"
            )

        image = Image.open(image_path)
        return image.convert("RGB")
