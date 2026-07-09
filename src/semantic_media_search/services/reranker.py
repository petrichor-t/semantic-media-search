"""Iterative CLIP cross-encoder for multi-stage search refinement."""
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class IterativeReranker:
    """Loads CLIP model once, reuses it for unlimited rerank calls."""

    def __init__(self, model) -> None:
        self._model = model

    @property
    def device(self) -> str:
        return str(self._model.device)

    def rerank(
        self,
        query: str,
        paths: list[Path],
        timeout_seconds: float | None = None,
    ) -> list[tuple[float, Path]]:
        """
        Score each image against the query using CLIP cross-encoding.

        Args:
            query: Russian/English search query.
            paths: Candidate image paths.
            timeout_seconds: Optional deadline; returns best-so-far if exceeded.

        Returns:
            List of (score, path) sorted descending by score.
        """
        if not paths:
            return []

        import torch

        start = time.perf_counter()
        scored: list[tuple[float, Path]] = []

        for i, path in enumerate(paths):
            # Check timeout
            if timeout_seconds is not None and (time.perf_counter() - start) > timeout_seconds:
                logger.info("Rerank timeout at %d/%d", i, len(paths))
                break

            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                logger.warning("Cannot open image for rerank: %s", path)
                continue

            # Use sentence_transformers encode API (handles CLIP internally)
            with torch.no_grad():
                img_emb = self._model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
                txt_emb = self._model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
                score = float(np.dot(txt_emb, img_emb))

            scored.append((score, path))

        scored.sort(key=lambda x: x[0], reverse=True)
        elapsed = time.perf_counter() - start
        logger.info("Reranked %d images in %.1fs", len(scored), elapsed)
        return scored