"""Adaptive CLIP reranker — warmup → measure → multiple full passes."""
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class AdaptiveReranker:
    """Calibrates speed, then makes as many full passes as time allows."""

    def __init__(self, model) -> None:
        self._model = model
        self._tau: float | None = None  # seconds per image, calibrated

    @property
    def tau(self) -> float | None:
        return self._tau

    def calibrate(self, sample_paths: list[Path], query: str) -> float:
        """Measure seconds-per-image on a small sample. Returns τ."""
        import torch

        n = len(sample_paths)
        start = time.perf_counter()
        for path in sample_paths:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            with torch.no_grad():
                self._model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
        elapsed = time.perf_counter() - start
        self._tau = elapsed / max(n, 1)
        logger.info("Calibrated τ = %.3f s/image (%d samples)", self._tau, n)
        return self._tau

    def score_batch(
        self,
        query: str,
        paths: list[Path],
    ) -> dict[Path, list[float]]:
        """
        Score all paths in one pass. Returns {path: [scores]} — appends, not replaces.
        """
        import torch

        results: dict[Path, list[float]] = {p: [] for p in paths}
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            with torch.no_grad():
                img_emb = self._model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
                txt_emb = self._model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
                score = float(np.dot(txt_emb, img_emb))
            results[path].append(score)
        return results

    def rerank_adaptive(
        self,
        query: str,
        paths: list[Path],
        think_time: float,
    ) -> list[tuple[float, Path]]:
        """
        Multi-pass reranking driven entirely by think_time and pool size.

        Args:
            query: Search query.
            paths: All candidate image paths.
            think_time: Total time budget in seconds.

        Returns:
            (score, path) sorted descending — scores averaged across passes.
        """
        if not paths:
            return []

        N = len(paths)

        # Phase 1: calibrate if needed
        if self._tau is None:
            warmup = paths[: min(5, N)]
            self.calibrate(warmup, query)

        assert self._tau is not None
        τ = self._tau
        P = N * τ  # time per full pass
        budget = max(think_time - 0.5, 0.1)  # reserve 0.5s for overhead

        # Phase 2: compute max passes
        if P <= 0:
            P = 0.1
        k_full = max(0, int(budget / P))
        remainder = budget - k_full * P

        logger.info(
            "Planning: N=%d, τ=%.3f, P=%.2fs, k_full=%d, remainder=%.2fs, budget=%.2fs",
            N, τ, P, k_full, remainder, think_time,
        )

        # Phase 3: accumulate scores over k_full passes
        accumulated: dict[Path, list[float]] = {p: [] for p in paths}

        start_total = time.perf_counter()
        for pass_i in range(k_full):
            # Check if we're running over time
            if time.perf_counter() - start_total > think_time:
                break
            batch = self.score_batch(query, paths)
            for p, scores in batch.items():
                accumulated[p].extend(scores)

        # Phase 4: partial pass on remainder time (process subset)
        if remainder > τ * 3 and k_full == 0:
            # Process as many as we can in the remaining time
            subset_count = min(N, int(remainder / τ))
            subset = paths[:subset_count]
            logger.info("Partial pass: %d of %d images", subset_count, N)
            batch = self.score_batch(query, subset)
            for p, scores in batch.items():
                accumulated[p].extend(scores)

        # Phase 5: average scores and rank
        ranked: list[tuple[float, Path]] = []
        for p, scores in accumulated.items():
            if scores:
                avg = float(np.mean(scores))
                ranked.append((avg, p))

        ranked.sort(key=lambda x: x[0], reverse=True)
        elapsed = time.perf_counter() - start_total
        passes_used = max(len(v) for v in accumulated.values()) if accumulated else 0
        logger.info(
            "Reranked %d/%d images in %.1fs (%d passes)", len(ranked), N, elapsed, passes_used
        )
        return ranked