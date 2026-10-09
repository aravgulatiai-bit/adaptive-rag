"""Typed configuration loaded from ``config/config.yaml``.

All tunables (thresholds, top-k, model names, normalization caps, paths) live in
the YAML file and are surfaced here as frozen dataclasses so the rest of the
codebase never hardcodes a policy-relevant value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Repository root = two levels up from this file (src/adaptive_rag/config.py).
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Filesystem locations for data and results (resolved to absolute)."""

    data_dir: Path
    results_dir: Path
    corpus_file: str
    questions_file: str
    embeddings_file: str
    bm25_file: str
    meta_file: str

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / self.corpus_file

    @property
    def questions_path(self) -> Path:
        return self.data_dir / self.questions_file

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / self.embeddings_file

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / self.bm25_file

    @property
    def meta_path(self) -> Path:
        return self.data_dir / self.meta_file

    def domain_vocab_path(self, filename: str) -> Path:
        """Path to the corpus-derived domain vocabulary file (in data_dir)."""
        return self.data_dir / filename


@dataclass(frozen=True)
class Models:
    """Model identifiers used across the pipeline."""

    spacy_model: str
    embedding_model: str
    cross_encoder_model: str
    llm_backend: str
    llm_model: str
    llm_max_new_tokens: int
    llm_context_top_n: int = 3


@dataclass(frozen=True)
class DatasetCfg:
    """Ingestion dataset settings."""

    name: str
    split: str
    num_passages: int
    num_questions: int
    questions_per_context: int = 1
    source: str = "squad"
    docs_dir: str = "sample_docs"
    eval_questions_file: str = "sample_docs_eval.jsonl"
    chunk_max_chars: int = 700
    chunk_min_chars: int = 200


@dataclass(frozen=True)
class FeaturesCfg:
    """Feature normalization caps and lookup tables (Module 2)."""

    query_length_cap: float
    dependency_depth_cap: float
    entity_density_cap: float
    domain_density_cap: float
    semantic_ambiguity_cap: float
    intent_scale: Dict[str, float]
    domain_keywords: List[str]
    intent_syntax_aware: bool = True
    use_corpus_domain_vocab: bool = True
    domain_vocab_file: str = "domain_vocab.json"
    domain_vocab_top_n: int = 80


@dataclass(frozen=True)
class PolicyBand:
    """A single retrieval-policy band (Module 6)."""

    name: str
    lo: float
    hi: float
    top_k: int
    use_bm25: bool
    use_dense: bool
    rerank: bool


@dataclass(frozen=True)
class PolicyCfg:
    bands: List[PolicyBand]


@dataclass(frozen=True)
class RetrievalCfg:
    rrf_k: int
    candidate_pool: int


@dataclass(frozen=True)
class EvaluationCfg:
    fixed_simple_top_k: int
    fixed_complex_top_k: int


@dataclass(frozen=True)
class Config:
    """Top-level configuration object."""

    paths: Paths
    models: Models
    dataset: DatasetCfg
    features: FeaturesCfg
    policy: PolicyCfg
    retrieval: RetrievalCfg
    evaluation: EvaluationCfg
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


def _resolve_dir(base: Path, value: str) -> Path:
    """Resolve ``value`` against the repo root unless already absolute."""
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate configuration from YAML.

    Args:
        path: Optional path to a config file. Defaults to
            ``config/config.yaml`` at the repository root, overridable via the
            ``ADAPTIVE_RAG_CONFIG`` environment variable.

    Returns:
        A populated :class:`Config`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required sections are missing.
    """
    cfg_path = Path(path) if path else Path(os.environ.get("ADAPTIVE_RAG_CONFIG", _DEFAULT_CONFIG))
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    for section in ("paths", "models", "dataset", "features", "policy", "retrieval", "evaluation"):
        if section not in raw:
            raise ValueError(f"Missing config section: '{section}'")

    p = raw["paths"]
    paths = Paths(
        data_dir=_resolve_dir(_REPO_ROOT, p["data_dir"]),
        results_dir=_resolve_dir(_REPO_ROOT, p["results_dir"]),
        corpus_file=p["corpus_file"],
        questions_file=p["questions_file"],
        embeddings_file=p["embeddings_file"],
        bm25_file=p["bm25_file"],
        meta_file=p["meta_file"],
    )

    m = raw["models"]
    models = Models(
        spacy_model=m["spacy_model"],
        embedding_model=m["embedding_model"],
        cross_encoder_model=m["cross_encoder_model"],
        llm_backend=m["llm_backend"],
        llm_model=m["llm_model"],
        llm_max_new_tokens=int(m["llm_max_new_tokens"]),
        llm_context_top_n=int(m.get("llm_context_top_n", 3)),
    )

    d = raw["dataset"]
    dataset = DatasetCfg(
        name=d["name"],
        split=d["split"],
        num_passages=int(d["num_passages"]),
        num_questions=int(d["num_questions"]),
        questions_per_context=int(d.get("questions_per_context", 1)),
        source=str(d.get("source", "squad")),
        docs_dir=str(_resolve_dir(_REPO_ROOT, d.get("docs_dir", "sample_docs"))),
        eval_questions_file=str(_resolve_dir(_REPO_ROOT, d.get("eval_questions_file", "sample_docs_eval.jsonl"))),
        chunk_max_chars=int(d.get("chunk_max_chars", 700)),
        chunk_min_chars=int(d.get("chunk_min_chars", 200)),
    )

    f = raw["features"]
    features = FeaturesCfg(
        query_length_cap=float(f["query_length_cap"]),
        dependency_depth_cap=float(f["dependency_depth_cap"]),
        entity_density_cap=float(f["entity_density_cap"]),
        domain_density_cap=float(f["domain_density_cap"]),
        semantic_ambiguity_cap=float(f["semantic_ambiguity_cap"]),
        intent_scale={k: float(v) for k, v in f["intent_scale"].items()},
        domain_keywords=[str(k).lower() for k in f["domain_keywords"]],
        intent_syntax_aware=bool(f.get("intent_syntax_aware", True)),
        use_corpus_domain_vocab=bool(f.get("use_corpus_domain_vocab", True)),
        domain_vocab_file=str(f.get("domain_vocab_file", "domain_vocab.json")),
        domain_vocab_top_n=int(f.get("domain_vocab_top_n", 80)),
    )

    bands = [
        PolicyBand(
            name=b["name"],
            lo=float(b["lo"]),
            hi=float(b["hi"]),
            top_k=int(b["top_k"]),
            use_bm25=bool(b["use_bm25"]),
            use_dense=bool(b["use_dense"]),
            rerank=bool(b["rerank"]),
        )
        for b in raw["policy"]["bands"]
    ]
    policy = PolicyCfg(bands=bands)

    r = raw["retrieval"]
    retrieval = RetrievalCfg(rrf_k=int(r["rrf_k"]), candidate_pool=int(r["candidate_pool"]))

    e = raw["evaluation"]
    evaluation = EvaluationCfg(
        fixed_simple_top_k=int(e["fixed_simple_top_k"]),
        fixed_complex_top_k=int(e["fixed_complex_top_k"]),
    )

    return Config(
        paths=paths,
        models=models,
        dataset=dataset,
        features=features,
        policy=policy,
        retrieval=retrieval,
        evaluation=evaluation,
        raw=raw,
    )
