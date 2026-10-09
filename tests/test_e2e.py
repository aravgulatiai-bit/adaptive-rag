"""End-to-end smoke test on a tiny in-memory corpus.

Builds a small dense + BM25 index at runtime (downloading only the small
MiniLM embedding model on first run) and drives the full pipeline with a stub
LLM so the test is fast and offline-friendly. A Simple query is used so the
cross-encoder is never loaded.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_rag.config import load_config
from adaptive_rag.llm import BaseLLM
from adaptive_rag.pipeline import AdaptiveRAGPipeline
from adaptive_rag.retrieval.indexer import CorpusIndex, Passage, bm25_tokenize


class StubLLM(BaseLLM):
    """Deterministic LLM stub that echoes the first retrieved context."""

    def generate(self, prompt: str) -> str:
        return "STUB_ANSWER"


_PASSAGES = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Python is a high-level, general-purpose programming language.",
    "The mitochondrion is the powerhouse of the cell in eukaryotes.",
    "The General Data Protection Regulation (GDPR) governs data protection in the EU.",
    "Mount Everest is Earth's highest mountain above sea level.",
    "Photosynthesis converts light energy into chemical energy in plants.",
]


@pytest.fixture(scope="module")
def small_index():
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    cfg = load_config()
    passages = [Passage(doc_id=i, text=t) for i, t in enumerate(_PASSAGES)]
    model = SentenceTransformer(cfg.models.embedding_model, device="cpu")
    emb = model.encode(
        [p.text for p in passages], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    bm25 = BM25Okapi([bm25_tokenize(p.text) for p in passages])
    return CorpusIndex(passages, emb, bm25, cfg.models.embedding_model)


@pytest.fixture(scope="module")
def pipeline(small_index):
    cfg = load_config()
    return AdaptiveRAGPipeline(cfg=cfg, index=small_index, llm=StubLLM())


def test_end_to_end_simple_query(pipeline):
    """A simple factual query should run end-to-end and retrieve the right doc."""
    resp = pipeline.run("What is the Eiffel Tower?")
    assert resp.answer == "STUB_ANSWER"
    assert resp.policy.query_class == "Simple"
    assert len(resp.documents) == 2  # top-2 dense
    # The Eiffel Tower passage (doc 0) should be retrieved.
    assert any(d.doc_id == 0 for d in resp.documents)
    assert 0.0 <= resp.complexity_score <= 1.0
    assert resp.latency_s >= 0.0


def test_end_to_end_retrieval_only(pipeline):
    """generate=False skips the LLM but still retrieves documents."""
    resp = pipeline.run("What is Python?", generate=False)
    assert resp.answer == ""
    assert len(resp.documents) >= 1
    assert any("Python" in d.text for d in resp.documents)


def test_analyze_produces_full_trace(pipeline):
    parsed, features, subsystem, score, policy = pipeline.analyze("What is GDPR?")
    assert parsed.num_tokens > 0
    assert len(features.as_vector()) == 6
    assert 0.0 <= score <= 1.0
    assert policy.query_class in {"Simple", "Medium", "Complex"}
