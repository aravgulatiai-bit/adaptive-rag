"""Sparse BM25 retrieval (rank_bm25)."""

from __future__ import annotations

from typing import List

import numpy as np

from ..logging_utils import get_logger
from .indexer import CorpusIndex, bm25_tokenize
from .types import RetrievedDoc

logger = get_logger(__name__)


class BM25Retriever:
    """BM25 sparse retriever over the prebuilt BM25 index."""

    def __init__(self, index: CorpusIndex) -> None:
        """Initialize.

        Args:
            index: A loaded :class:`CorpusIndex` with a BM25 index.

        Raises:
            ValueError: If the index has no BM25 index.
        """
        if index.bm25 is None:
            raise ValueError("CorpusIndex has no BM25 index; rebuild with `rag ingest`.")
        self.index = index

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        """Return the ``top_k`` passages by BM25 score.

        Args:
            query: Query text.
            top_k: Number of passages to return.

        Returns:
            Ranked list of :class:`RetrievedDoc` (highest BM25 score first).
        """
        tokens = bm25_tokenize(query)
        scores = np.asarray(self.index.bm25.get_scores(tokens))
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results = [
            RetrievedDoc(
                doc_id=int(i),
                text=self.index.passages[i].text,
                score=float(scores[i]),
                source="bm25",
                title=self.index.passages[i].title,
            )
            for i in top_idx
        ]
        logger.debug("BM25 retrieve(top_k=%d) -> %d docs", top_k, len(results))
        return results
