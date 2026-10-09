"""Module 7 — Adaptive Retrieval.

Sub-modules:

* :mod:`adaptive_rag.retrieval.indexer` — build / save / load the corpus, the
  dense embedding index, and the BM25 index.
* :mod:`adaptive_rag.retrieval.dense` — dense (bi-encoder cosine) retrieval.
* :mod:`adaptive_rag.retrieval.bm25` — sparse BM25 retrieval.
* :mod:`adaptive_rag.retrieval.hybrid` — Reciprocal Rank Fusion of dense + BM25.
* :mod:`adaptive_rag.retrieval.rerank` — cross-encoder re-ranking.
"""

from .types import RetrievedDoc  # noqa: F401
