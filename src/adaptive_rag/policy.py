"""Module 6 — Retrieval Policy Selector.

Map a continuous complexity score in ``[0, 1]`` to a retrieval configuration.
All thresholds and top-k values come from the config (``policy.bands``), never
hardcoded here.

Default bands (from ``config/config.yaml``):

======================  ==========  ======================================
Complexity score        Query class Retrieval strategy
======================  ==========  ======================================
0.00 - 0.35             Simple      Top-2 dense, no re-ranking
0.35 - 0.70             Medium      Top-5 hybrid (BM25 + dense)
0.70 - 1.00             Complex     Top-10 hybrid + cross-encoder re-rank
======================  ==========  ======================================
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PolicyCfg
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalPolicy:
    """The retrieval configuration selected for a query."""

    query_class: str      # "Simple" | "Medium" | "Complex"
    top_k: int
    use_dense: bool
    use_bm25: bool
    rerank: bool

    @property
    def is_hybrid(self) -> bool:
        """True when both dense and sparse retrievers are used."""
        return self.use_dense and self.use_bm25

    def describe(self) -> str:
        """Return a short human-readable summary of the strategy."""
        parts = []
        if self.is_hybrid:
            parts.append(f"Top-{self.top_k} hybrid (BM25 + dense)")
        elif self.use_dense:
            parts.append(f"Top-{self.top_k} dense")
        elif self.use_bm25:
            parts.append(f"Top-{self.top_k} BM25")
        if self.rerank:
            parts.append("+ cross-encoder re-rank")
        else:
            parts.append("no re-ranking")
        return ", ".join(parts)


class PolicySelector:
    """Selects a :class:`RetrievalPolicy` from a complexity score."""

    def __init__(self, cfg: PolicyCfg) -> None:
        """Initialize with policy configuration.

        Args:
            cfg: Policy bands loaded from config, sorted by lower bound.
        """
        self.bands = sorted(cfg.bands, key=lambda b: b.lo)
        if not self.bands:
            raise ValueError("Policy configuration has no bands.")

    def select(self, complexity_score: float) -> RetrievalPolicy:
        """Select a retrieval policy for a complexity score.

        Bands are half-open ``[lo, hi)``. If the score falls outside all bands
        (numerical edge cases), the nearest band is used.

        Args:
            complexity_score: Overall complexity in ``[0, 1]``.

        Returns:
            The selected :class:`RetrievalPolicy`.
        """
        score = max(0.0, min(1.0, complexity_score))
        for band in self.bands:
            if band.lo <= score < band.hi:
                return self._to_policy(band, score)
        # Fallback: score >= last hi (e.g. exactly 1.0 with hi=1.0) -> top band.
        band = self.bands[-1]
        return self._to_policy(band, score)

    def _to_policy(self, band, score: float) -> RetrievalPolicy:
        policy = RetrievalPolicy(
            query_class=band.name,
            top_k=band.top_k,
            use_dense=band.use_dense,
            use_bm25=band.use_bm25,
            rerank=band.rerank,
        )
        logger.debug(
            "Score %.4f -> class=%s (%s)", score, policy.query_class, policy.describe()
        )
        return policy
