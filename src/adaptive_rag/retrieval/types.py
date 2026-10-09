"""Shared retrieval data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    """A retrieved corpus passage with provenance.

    Attributes:
        doc_id: Stable corpus id (index into the corpus).
        text: Passage text.
        score: Retriever score (semantics depend on the retriever/fusion stage).
        source: Which retriever/stage produced or last scored this doc
            (e.g. ``"dense"``, ``"bm25"``, ``"rrf"``, ``"cross-encoder"``).
        title: Optional passage title/metadata.
    """

    doc_id: int
    text: str
    score: float
    source: str = ""
    title: str = ""
