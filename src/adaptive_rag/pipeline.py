"""End-to-end orchestration of the Adaptive RAG pipeline (Modules 1-8).

Wires together preprocessing -> feature extraction -> fuzzy engine ->
defuzzification -> policy selection -> adaptive retrieval -> LLM generation, and
returns a structured :class:`RAGResponse` capturing every intermediate for
transparency and debugging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config, load_config
from .features import FeatureExtractor, FeatureVector
from .fuzzy.hierarchical import HierarchicalFuzzyEngine
from .fuzzy.subsystems import ComplexitySubsystems, SubsystemScores
from .llm import BaseLLM, get_llm
from .logging_utils import get_logger
from .policy import PolicySelector, RetrievalPolicy
from .preprocessing import ParsedQuery, QueryPreprocessor
from .retrieval.bm25 import BM25Retriever
from .retrieval.dense import DenseRetriever
from .retrieval.hybrid import HybridRetriever
from .retrieval.indexer import CorpusIndex
from .retrieval.rerank import CrossEncoderReranker
from .retrieval.types import RetrievedDoc

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Structured pipeline output (Module 8).

    Attributes:
        query: The original query.
        answer: The generated answer text.
        complexity_score: Overall complexity in ``[0, 1]``.
        subsystem_scores: Linguistic / Knowledge / Reasoning complexity.
        features: The 6-D feature vector.
        policy: The selected retrieval policy.
        documents: Retrieved passages used as context.
        latency_s: End-to-end wall-clock latency in seconds.
        retrieval_cost: Number of documents retrieved (context size).
        reranked: Whether cross-encoder re-ranking ran.
    """

    query: str
    answer: str
    complexity_score: float
    subsystem_scores: SubsystemScores
    features: FeatureVector
    policy: RetrievalPolicy
    documents: List[RetrievedDoc] = field(default_factory=list)
    latency_s: float = 0.0
    retrieval_cost: int = 0
    reranked: bool = False

    def trace_str(self) -> str:
        """Return a verbose, human-readable trace of every pipeline stage."""
        fv = self.features
        lines = [
            f"Query: {self.query}",
            "",
            "── Features (raw / normalized) ─────────────────────────────",
            f"  query_length          : {fv.query_length_raw:>5} / {fv.query_length_norm:.3f}",
            f"  dependency_depth      : {fv.dependency_depth_raw:>5} / {fv.dependency_depth_norm:.3f}",
            f"  entity_density        : {fv.entity_density_raw:>5.3f} / {fv.entity_density_norm:.3f}",
            f"  domain_keyword_density: {fv.domain_keyword_density_raw:>5.3f} / {fv.domain_keyword_density_norm:.3f}",
            f"  intent_type           : {fv.intent_label:>10} / {fv.intent_type_norm:.3f}",
            f"  semantic_ambiguity    : {fv.semantic_ambiguity_raw:>5.2f} / {fv.semantic_ambiguity_norm:.3f}",
            "",
            "── Subsystem complexity ────────────────────────────────────",
            f"  Linguistic : {self.subsystem_scores.linguistic:.3f}",
            f"  Knowledge  : {self.subsystem_scores.knowledge:.3f}",
            f"  Reasoning  : {self.subsystem_scores.reasoning:.3f}",
            "",
            "── Overall complexity & policy ─────────────────────────────",
            f"  Complexity score : {self.complexity_score:.3f}",
            f"  Query class      : {self.policy.query_class}",
            f"  Strategy         : {self.policy.describe()}",
            f"  Re-ranking ran   : {self.reranked}",
            f"  Docs retrieved   : {self.retrieval_cost}",
            f"  Latency          : {self.latency_s:.3f}s",
            "",
            "── Retrieved documents ─────────────────────────────────────",
        ]
        for i, doc in enumerate(self.documents, 1):
            snippet = doc.text[:140].replace("\n", " ")
            lines.append(f"  [{i}] (id={doc.doc_id}, score={doc.score:.4f}, src={doc.source}) {snippet}...")
        lines.append("")
        lines.append("── Answer ──────────────────────────────────────────────────")
        lines.append(f"  {self.answer}")
        return "\n".join(lines)


class AdaptiveRAGPipeline:
    """The full adaptive RAG pipeline."""

    def __init__(
        self,
        cfg: Optional[Config] = None,
        index: Optional[CorpusIndex] = None,
        llm: Optional[BaseLLM] = None,
        load_retrieval: bool = True,
    ) -> None:
        """Initialize the pipeline and its components.

        Args:
            cfg: Config; loaded from default location if omitted.
            index: Prebuilt corpus index; loaded from disk if omitted and
                ``load_retrieval`` is True.
            llm: Answering LLM; constructed from config if omitted.
            load_retrieval: If False, skip loading the corpus/LLM (useful for
                testing just the complexity engine).
        """
        self.cfg = cfg or load_config()

        # Modules 1-6 (cheap, always constructed).
        self.preprocessor = QueryPreprocessor(self.cfg.models.spacy_model)
        # Load the corpus-derived domain vocabulary if enabled and present.
        domain_weights = None
        if self.cfg.features.use_corpus_domain_vocab:
            from .domain_vocab import load_domain_vocab

            domain_weights = load_domain_vocab(
                self.cfg.paths.domain_vocab_path(self.cfg.features.domain_vocab_file)
            )
            if domain_weights:
                logger.debug("Loaded corpus domain vocab: %d terms", len(domain_weights))
        self.extractor = FeatureExtractor(self.cfg.features, domain_weights=domain_weights)
        self.subsystems = ComplexitySubsystems()
        self.engine = HierarchicalFuzzyEngine()
        self.selector = PolicySelector(self.cfg.policy)

        # Modules 7-8 (heavy; optional).
        self.index = index
        self._dense: Optional[DenseRetriever] = None
        self._bm25: Optional[BM25Retriever] = None
        self._hybrid: Optional[HybridRetriever] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._llm = llm

        if load_retrieval:
            self._init_retrieval()

    # ------------------------------------------------------------------ setup
    def _init_retrieval(self) -> None:
        if self.index is None:
            self.index = CorpusIndex.load(self.cfg)
        self._dense = DenseRetriever(self.index, self.cfg.models.embedding_model)
        self._bm25 = BM25Retriever(self.index)
        self._hybrid = HybridRetriever(
            self._dense,
            self._bm25,
            rrf_k=self.cfg.retrieval.rrf_k,
            candidate_pool=self.cfg.retrieval.candidate_pool,
        )
        self._reranker = CrossEncoderReranker(self.cfg.models.cross_encoder_model)

    @property
    def llm(self) -> BaseLLM:
        """Lazily construct the configured LLM."""
        if self._llm is None:
            self._llm = get_llm(
                self.cfg.models.llm_backend,
                self.cfg.models.llm_model,
                self.cfg.models.llm_max_new_tokens,
            )
        return self._llm

    # ------------------------------------------------------- complexity engine
    def analyze(self, query: str) -> tuple[ParsedQuery, FeatureVector, SubsystemScores, float, RetrievalPolicy]:
        """Run Modules 1-6: parse, feature-extract, fuzzy-infer, select policy.

        Args:
            query: Raw query text.

        Returns:
            Tuple of (parsed, features, subsystem scores, complexity score,
            selected policy).
        """
        parsed = self.preprocessor.parse(query)
        features = self.extractor.extract(parsed)
        subsystem = self.subsystems.infer(
            query_length=features.query_length_norm,
            dependency_depth=features.dependency_depth_norm,
            entity_density=features.entity_density_norm,
            domain_density=features.domain_keyword_density_norm,
            intent_type=features.intent_type_norm,
            semantic_ambiguity=features.semantic_ambiguity_norm,
        )
        score = self.engine.infer(subsystem)
        policy = self.selector.select(score)
        return parsed, features, subsystem, score, policy

    # --------------------------------------------------------------- retrieval
    def retrieve(self, query: str, policy: RetrievalPolicy) -> tuple[List[RetrievedDoc], bool]:
        """Execute the retrieval strategy dictated by ``policy`` (Module 7).

        Args:
            query: Query text.
            policy: The selected retrieval policy.

        Returns:
            Tuple of (retrieved docs, whether re-ranking ran).
        """
        if self._dense is None:
            self._init_retrieval()

        reranked = False
        if policy.is_hybrid:
            # Fetch a candidate pool; if re-ranking, keep the pool wide then trim.
            if policy.rerank:
                pool = max(self.cfg.retrieval.candidate_pool, policy.top_k)
                candidates = self._hybrid.retrieve(query, pool)
                docs = self._reranker.rerank(query, candidates, policy.top_k)
                reranked = True
            else:
                docs = self._hybrid.retrieve(query, policy.top_k)
        elif policy.use_dense:
            docs = self._dense.retrieve(query, policy.top_k)
            if policy.rerank:
                docs = self._reranker.rerank(query, docs, policy.top_k)
                reranked = True
        elif policy.use_bm25:
            docs = self._bm25.retrieve(query, policy.top_k)
        else:  # pragma: no cover - config guarantees at least one retriever
            docs = []

        logger.debug("Retrieved %d docs (reranked=%s)", len(docs), reranked)
        return docs, reranked

    def answer_from_docs(self, query: str, docs: List[RetrievedDoc]) -> str:
        """Generate an answer from retrieved docs, focusing the LLM context.

        Retrieval fetches many passages for recall, but only the top-N
        best-ranked passages (``models.llm_context_top_n``) are placed in the
        prompt — small models degrade when fed many passages.

        Args:
            query: Query text.
            docs: Retrieved documents, best-ranked first.

        Returns:
            The generated answer.
        """
        top_n = self.cfg.models.llm_context_top_n
        contexts = [d.text for d in docs[:top_n]]
        return self.llm.answer(query, contexts)

    # ---------------------------------------------------------------- full run
    def run(self, query: str, generate: bool = True) -> RAGResponse:
        """Run the full pipeline end-to-end.

        Args:
            query: Raw query text.
            generate: If False, skip LLM generation (retrieval-only; useful for
                the evaluation harness's retrieval metrics).

        Returns:
            A populated :class:`RAGResponse`.
        """
        start = time.perf_counter()
        parsed, features, subsystem, score, policy = self.analyze(query)
        docs, reranked = self.retrieve(query, policy)

        answer = ""
        if generate:
            answer = self.answer_from_docs(query, docs)

        latency = time.perf_counter() - start
        response = RAGResponse(
            query=query,
            answer=answer,
            complexity_score=score,
            subsystem_scores=subsystem,
            features=features,
            policy=policy,
            documents=docs,
            latency_s=latency,
            retrieval_cost=len(docs),
            reranked=reranked,
        )
        logger.debug("Pipeline complete for %r in %.3fs", query, latency)
        return response
