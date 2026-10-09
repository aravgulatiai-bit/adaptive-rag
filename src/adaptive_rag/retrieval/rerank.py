"""Cross-encoder re-ranking (sentence-transformers CrossEncoder).

A cross-encoder jointly encodes (query, passage) pairs to produce a relevance
score that is typically much more accurate than bi-encoder cosine or BM25, at
higher compute cost. It is applied only for Complex queries, on the already
shortlisted candidates from the hybrid stage.
"""

from __future__ import annotations

import functools
from typing import List

from ..logging_utils import get_logger
from .types import RetrievedDoc

logger = get_logger(__name__)


@functools.lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    """Load and cache the CrossEncoder (CPU)."""
    from sentence_transformers import CrossEncoder

    logger.debug("Loading cross-encoder '%s'", model_name)
    return CrossEncoder(model_name, device="cpu")


class CrossEncoderReranker:
    """Re-ranks retrieved passages with a cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Initialize.

        Args:
            model_name: CrossEncoder model id.
        """
        self.model_name = model_name
        self._model = None

    def rerank(self, query: str, docs: List[RetrievedDoc], top_k: int) -> List[RetrievedDoc]:
        """Re-score and re-order ``docs`` for ``query``, keeping ``top_k``.

        Args:
            query: Query text.
            docs: Candidate passages (from hybrid/dense retrieval).
            top_k: Number of passages to keep after re-ranking.

        Returns:
            Re-ranked top-k :class:`RetrievedDoc` list (cross-encoder scores).
        """
        if not docs:
            return []
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name)

        pairs = [(query, d.text) for d in docs]
        scores = self._model.predict(pairs)
        for doc, score in zip(docs, scores):
            doc.score = float(score)
            doc.source = "cross-encoder"
        reranked = sorted(docs, key=lambda d: d.score, reverse=True)[:top_k]
        logger.debug("Cross-encoder reranked %d docs -> top %d", len(docs), len(reranked))
        return reranked
