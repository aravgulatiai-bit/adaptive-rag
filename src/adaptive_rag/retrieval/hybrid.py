"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Why RRF (justification)
-----------------------
Dense cosine similarities live in ``[-1, 1]`` while BM25 scores are unbounded and
corpus/query dependent; the two are not directly comparable, and min-max
normalization is brittle (sensitive to outliers and to the candidate-pool size).
RRF fuses by *rank* rather than raw score:

    rrf_score(d) = sum over retrievers of  1 / (k + rank_r(d))

with ``rank`` starting at 1 and a smoothing constant ``k`` (default 60, the value
from the original Cormack et al. RRF paper). This is scale-free, needs no score
normalization or per-retriever weight tuning, is robust to score outliers, and is
a strong, widely-used default for combining heterogeneous rankers.
"""

from __future__ import annotations

from typing import Dict, List

from ..logging_utils import get_logger
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .types import RetrievedDoc

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievedDoc]],
    rrf_k: int,
    top_k: int,
    passages,
) -> List[RetrievedDoc]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: One ranked list per retriever (rank 1 = best).
        rrf_k: RRF smoothing constant.
        top_k: Number of fused results to return.
        passages: The corpus passage list (for text/title lookup).

    Returns:
        The fused top-k :class:`RetrievedDoc` list, sorted by RRF score.
    """
    fused: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            fused[doc.doc_id] = fused.get(doc.doc_id, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    results = [
        RetrievedDoc(
            doc_id=doc_id,
            text=passages[doc_id].text,
            score=score,
            source="rrf",
            title=passages[doc_id].title,
        )
        for doc_id, score in ordered
    ]
    logger.debug(
        "RRF fused %d lists -> %d unique docs, returning top %d",
        len(ranked_lists), len(fused), len(results),
    )
    return results


class HybridRetriever:
    """Combines dense and BM25 retrieval with RRF."""

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        rrf_k: int = 60,
        candidate_pool: int = 50,
    ) -> None:
        """Initialize.

        Args:
            dense: A dense retriever.
            bm25: A BM25 retriever.
            rrf_k: RRF smoothing constant.
            candidate_pool: How many candidates to draw from each retriever
                before fusion.
        """
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        """Retrieve ``top_k`` passages via dense+BM25 RRF fusion.

        Args:
            query: Query text.
            top_k: Number of fused passages to return.

        Returns:
            Fused ranked list of :class:`RetrievedDoc`.
        """
        pool = max(self.candidate_pool, top_k)
        dense_docs = self.dense.retrieve(query, pool)
        bm25_docs = self.bm25.retrieve(query, pool)
        return reciprocal_rank_fusion(
            [dense_docs, bm25_docs],
            rrf_k=self.rrf_k,
            top_k=top_k,
            passages=self.dense.index.passages,
        )
