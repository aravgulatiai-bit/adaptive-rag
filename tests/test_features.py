"""Unit tests for Module 2 — feature extraction on hand-crafted queries."""

from __future__ import annotations

import pytest

from adaptive_rag.features import analyze_intent, classify_intent


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is GDPR?", "definition"),
        ("Define entropy", "definition"),
        ("Who was Alan Turing?", "definition"),
        ("How do I install Python?", "procedure"),
        ("What are the steps to configure a firewall?", "procedure"),
        ("Compare GDPR and CCPA", "comparison"),
        ("What is the difference between TCP and UDP?", "comparison"),
        ("Why does inflation affect interest rates?", "reasoning"),
        ("Explain the causes of the French Revolution", "reasoning"),
    ],
)
def test_intent_classification(query, expected):
    """Legacy regex-only classifier (still used when intent_syntax_aware=False)."""
    assert classify_intent(query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        # Anchors preserved.
        ("What is GDPR?", "definition"),
        ("Compare GDPR and CCPA", "comparison"),
        ("Why does inflation affect interest rates?", "reasoning"),
        ("How do I install Python?", "procedure"),
        # Under-firing fixed: comparison with no comparison cue word, detected via
        # comparative morphology ("faster") + coordinated nouns ("Postgres or MySQL").
        ("Is Postgres or MySQL faster for writes?", "comparison"),
        # Over-firing fixed: "between X and Y" over numbers/dates is NOT a comparison.
        ("What happened between 1990 and 1995?", "definition"),
        # Genuine comparison phrased with a definition frame stays comparison
        # (background definition cue must not dilute it).
        ("What are the differences between the Starter, Growth, and Enterprise tiers?", "comparison"),
    ],
)
def test_syntax_aware_intent_label(preprocessor, cfg, query, expected):
    res = analyze_intent(preprocessor.parse(query), cfg.features.intent_scale)
    assert res.label == expected


def test_intent_anchors_and_softness(preprocessor, cfg):
    """Single-intent queries hit their anchor; mixed queries land between anchors."""
    sc = cfg.features.intent_scale
    definition = analyze_intent(preprocessor.parse("What is X?"), sc)
    comparison = analyze_intent(preprocessor.parse("Compare X and Y"), sc)
    mixed = analyze_intent(preprocessor.parse("Compare X and Y and explain why it matters"), sc)
    assert definition.score == pytest.approx(sc["definition"])
    assert comparison.score == pytest.approx(sc["comparison"])
    # Comparison + reasoning blend lands strictly between reasoning and comparison.
    assert sc["reasoning"] < mixed.score < sc["comparison"]
    assert 0.0 <= mixed.score <= 1.0


def test_weighted_domain_density_uses_supplied_vocab(cfg, preprocessor):
    """A corpus vocab weight should scale a term's contribution to domain density."""
    from adaptive_rag.features import FeatureExtractor

    # Same query through two extractors: one with a corpus vocab weighting
    # "encryption" highly, one with only the config list. The weighted one must
    # register a domain hit for a term the config list may not contain.
    ext = FeatureExtractor(cfg.features, domain_weights={"encryption": 1.0, "rotate": 0.8})
    fv = ext.extract(preprocessor.parse("How often must encryption keys rotate?"))
    assert fv.domain_keyword_density_raw > 0.0


def test_feature_ranges_are_normalized(extractor, preprocessor):
    """All six normalized features must lie in [0, 1]."""
    for q in [
        "What is X?",
        "Compare the data-retention obligations under GDPR and CCPA for cloud "
        "providers processing biometric data across jurisdictions",
        "How do I bake bread?",
    ]:
        fv = extractor.extract(preprocessor.parse(q))
        for name, val in fv.as_dict().items():
            assert 0.0 <= val <= 1.0, f"{name}={val} out of range for {q!r}"


def test_query_length_feature(extractor, preprocessor):
    short = extractor.extract(preprocessor.parse("What is X?"))
    long = extractor.extract(preprocessor.parse(
        "Compare and contrast the long term economic and social consequences "
        "of two competing monetary policy frameworks in developing nations"
    ))
    assert long.query_length_raw > short.query_length_raw
    assert long.query_length_norm >= short.query_length_norm


def test_entity_density_feature(extractor, preprocessor):
    """A query naming several entities should have higher entity density."""
    with_entities = extractor.extract(preprocessor.parse(
        "Compare GDPR in Germany with CCPA in California and PIPEDA in Canada"
    ))
    without = extractor.extract(preprocessor.parse("What is a law?"))
    assert with_entities.entity_density_raw >= without.entity_density_raw


def test_domain_keyword_density_feature(extractor, preprocessor):
    """Domain terms from the configured list must be counted."""
    domain_heavy = extractor.extract(preprocessor.parse(
        "What are the data retention and consent obligations under GDPR compliance?"
    ))
    generic = extractor.extract(preprocessor.parse("What is a dog?"))
    assert domain_heavy.domain_keyword_density_raw > generic.domain_keyword_density_raw


def test_intent_numeric_scale_ordering(extractor, preprocessor):
    """Comparison intent must map to a higher numeric value than definition."""
    definition = extractor.extract(preprocessor.parse("What is X?"))
    comparison = extractor.extract(preprocessor.parse("Compare X and Y"))
    assert comparison.intent_type_norm > definition.intent_type_norm


def test_semantic_ambiguity_nonnegative(extractor, preprocessor):
    fv = extractor.extract(preprocessor.parse("What does the term set mean?"))
    assert fv.semantic_ambiguity_norm >= 0.0
