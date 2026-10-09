"""Unit tests for Modules 3-5 — fuzzy subsystems + hierarchical engine.

Covers boundary cases per subsystem, completeness of the hierarchical rule base,
and monotonicity sanity checks.
"""

from __future__ import annotations

import itertools

import pytest

from adaptive_rag.fuzzy.hierarchical import (
    HierarchicalFuzzyEngine,
    aggregate_level,
    build_rule_map,
)
from adaptive_rag.fuzzy.membership import LEVELS
from adaptive_rag.fuzzy.subsystems import ComplexitySubsystems, SubsystemScores


# --------------------------------------------------------------------------- #
# Subsystem boundary cases.
# --------------------------------------------------------------------------- #
def test_subsystem_low_inputs_give_low_output(subsystems):
    """All-low inputs should defuzzify to a low score (< 0.35)."""
    assert subsystems.linguistic.infer(0.0, 0.0) < 0.35
    assert subsystems.knowledge.infer(0.0, 0.0) < 0.35
    assert subsystems.reasoning.infer(0.0, 0.0) < 0.35


def test_subsystem_high_inputs_give_high_output(subsystems):
    """All-high inputs should defuzzify to a high score (> 0.65)."""
    assert subsystems.linguistic.infer(1.0, 1.0) > 0.65
    assert subsystems.knowledge.infer(1.0, 1.0) > 0.65
    assert subsystems.reasoning.infer(1.0, 1.0) > 0.65


def test_reasoning_comparison_intent_forces_high(subsystems):
    """A comparison intent (high) alone must yield high reasoning complexity."""
    # intent_type high (0.95 comparison), semantic ambiguity low.
    assert subsystems.reasoning.infer(0.95, 0.0) > 0.65


def test_subsystem_monotonic_in_first_input(subsystems):
    """Increasing an input should not decrease the subsystem output."""
    low = subsystems.linguistic.infer(0.1, 0.5)
    mid = subsystems.linguistic.infer(0.5, 0.5)
    high = subsystems.linguistic.infer(0.9, 0.5)
    assert low <= mid + 1e-6
    assert mid <= high + 1e-6


def test_subsystem_output_in_unit_interval(subsystems):
    for a, b in itertools.product([0.0, 0.25, 0.5, 0.75, 1.0], repeat=2):
        for sub in (subsystems.linguistic, subsystems.knowledge, subsystems.reasoning):
            out = sub.infer(a, b)
            assert 0.0 <= out <= 1.0


# --------------------------------------------------------------------------- #
# Hierarchical engine: completeness + spec example rules + monotonicity.
# --------------------------------------------------------------------------- #
def test_rule_map_is_complete():
    """Every one of the 27 level-combinations must map to a valid output."""
    rule_map = build_rule_map()
    assert len(rule_map) == 27
    for combo in itertools.product(LEVELS, repeat=3):
        assert rule_map[combo] in LEVELS


def test_aggregate_spec_examples():
    """The three spec example rules must hold under the aggregation."""
    assert aggregate_level("low", "low", "low") == "low"
    assert aggregate_level("high", "high", "low") == "high"      # Ling High & Know High -> High
    assert aggregate_level("high", "high", "high") == "high"
    assert aggregate_level("medium", "medium", "low") == "medium"


def test_engine_no_empty_output(engine):
    """No crisp subsystem-score combination may produce an empty output."""
    grid = [0.0, 0.16, 0.33, 0.5, 0.66, 0.83, 1.0]
    for l, k, r in itertools.product(grid, repeat=3):
        score = engine.infer(SubsystemScores(l, k, r))
        assert 0.0 <= score <= 1.0


def test_engine_all_low_is_low(engine):
    assert engine.infer(SubsystemScores(0.1, 0.1, 0.1)) < 0.35


def test_engine_all_high_is_high(engine):
    assert engine.infer(SubsystemScores(0.9, 0.9, 0.9)) > 0.70


def test_engine_monotonicity(engine):
    """Raising any subsystem score must not lower the overall complexity."""
    base = engine.infer(SubsystemScores(0.2, 0.2, 0.2))
    up_l = engine.infer(SubsystemScores(0.9, 0.2, 0.2))
    up_k = engine.infer(SubsystemScores(0.2, 0.9, 0.2))
    up_r = engine.infer(SubsystemScores(0.2, 0.2, 0.9))
    assert up_l >= base - 1e-6
    assert up_k >= base - 1e-6
    assert up_r >= base - 1e-6


# --------------------------------------------------------------------------- #
# End-to-end complexity monotonicity (spec requirement).
# --------------------------------------------------------------------------- #
def test_complex_query_scores_higher_than_simple(complexity_of):
    """A long, entity-dense comparison query must score higher than 'What is X?'."""
    simple, _, _ = complexity_of("What is X?")
    complex_score, _, _ = complexity_of(
        "Compare the data-retention obligations under GDPR and CCPA for cloud "
        "providers processing biometric data across multiple jurisdictions"
    )
    assert complex_score > simple


def test_gdpr_definition_is_low_and_comparison_is_high(complexity_of):
    """Spec sanity check: 'What is GDPR?' Low; the GDPR/CCPA comparison High."""
    low_score, _, _ = complexity_of("What is GDPR?")
    high_score, _, _ = complexity_of(
        "Compare the data-retention obligations under GDPR and CCPA for cloud "
        "providers processing biometric data"
    )
    assert low_score < 0.35
    assert high_score > 0.70
