"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from adaptive_rag.config import load_config
from adaptive_rag.features import FeatureExtractor
from adaptive_rag.fuzzy.hierarchical import HierarchicalFuzzyEngine
from adaptive_rag.fuzzy.subsystems import ComplexitySubsystems
from adaptive_rag.policy import PolicySelector
from adaptive_rag.preprocessing import QueryPreprocessor


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def preprocessor(cfg):
    return QueryPreprocessor(cfg.models.spacy_model)


@pytest.fixture(scope="session")
def extractor(cfg):
    return FeatureExtractor(cfg.features)


@pytest.fixture(scope="session")
def subsystems():
    return ComplexitySubsystems()


@pytest.fixture(scope="session")
def engine():
    return HierarchicalFuzzyEngine()


@pytest.fixture(scope="session")
def selector(cfg):
    return PolicySelector(cfg.policy)


@pytest.fixture(scope="session")
def complexity_of(preprocessor, extractor, subsystems, engine):
    """Return a callable query -> (complexity_score, subsystem_scores, features)."""

    def _fn(query: str):
        parsed = preprocessor.parse(query)
        features = extractor.extract(parsed)
        scores = subsystems.infer(
            query_length=features.query_length_norm,
            dependency_depth=features.dependency_depth_norm,
            entity_density=features.entity_density_norm,
            domain_density=features.domain_keyword_density_norm,
            intent_type=features.intent_type_norm,
            semantic_ambiguity=features.semantic_ambiguity_norm,
        )
        overall = engine.infer(scores)
        return overall, scores, features

    return _fn
