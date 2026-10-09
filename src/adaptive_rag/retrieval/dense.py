"""Dense retrieval via a sentence-transformers bi-encoder (cosine similarity).

Passage embeddings are precomputed and L2-normalized (see indexer), so cosine
similarity reduces to a dot product against the (also normalized) query vector.
"""

from __future__ import annotations

import functools
from typing import List

import numpy as np

from ..logging_utils import get_logger
from .indexer import CorpusIndex
from .types import RetrievedDoc

logger = get_logger(__name__)


@functools.lru_cache(maxsize=2)
def _load_embedder(model_name: str):
    """Load and cache the SentenceTransformer query encoder (CPU)."""
    from sentence_transformers import SentenceTransformer

    logger.debug("Loading embedding model '%s' for query encoding", model_name)
    return SentenceTransformer(model_name, device="cpu")


class DenseRetriever:
    """Cosine-similarity dense retriever over precomputed embeddings."""

    def __init__(self, index: CorpusIndex, model_name: str | None = None) -> None:
        """Initialize.

        Args:
            index: A loaded :class:`CorpusIndex` with embeddings.
            model_name: Query-encoder model; defaults to the index's model.

        Raises:
            ValueError: If the index has no dense embeddings.
        """
        if index.embeddings is None:
            raise ValueError("CorpusIndex has no embeddings; rebuild with `rag ingest`.")
        self.index = index
        self.model_name = model_name or index.embedding_model_name
        self._embedder = None

    def _encode(self, query: str) -> np.ndarray:
        if self._embedder is None:
            self._embedder = _load_embedder(self.model_name)
        vec = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        return vec[0]

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        """Return the ``top_k`` passages by cosine similarity.

        Args:
            query: Query text.
            top_k: Number of passages to return.

        Returns:
            Ranked list of :class:`RetrievedDoc` (highest similarity first).
        """
        qvec = self._encode(query)
        sims = self.index.embeddings @ qvec  # cosine (both normalized)
        k = min(top_k, len(sims))
        # argpartition for the top-k, then sort those k by score.
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        results = [
            RetrievedDoc(
                doc_id=int(i),
                text=self.index.passages[i].text,
                score=float(sims[i]),
                source="dense",
                title=self.index.passages[i].title,
            )
            for i in top_idx
        ]
        logger.debug("Dense retrieve(top_k=%d) -> %d docs", top_k, len(results))
        return results
