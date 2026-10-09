"""Unit tests for Module 6 — retrieval policy selector thresholds."""

from __future__ import annotations

import pytest


def test_simple_band(selector):
    for score in (0.0, 0.1, 0.34999):
        pol = selector.select(score)
        assert pol.query_class == "Simple"
        assert pol.top_k == 2
        assert pol.use_dense and not pol.use_bm25
        assert not pol.rerank


def test_medium_band(selector):
    for score in (0.35, 0.5, 0.699):
        pol = selector.select(score)
        assert pol.query_class == "Medium"
        assert pol.top_k == 5
        assert pol.is_hybrid
        assert not pol.rerank


def test_complex_band(selector):
    for score in (0.70, 0.85, 1.0):
        pol = selector.select(score)
        assert pol.query_class == "Complex"
        assert pol.top_k == 10
        assert pol.is_hybrid
        assert pol.rerank


def test_threshold_boundaries_are_half_open(selector):
    """0.35 -> Medium (not Simple); 0.70 -> Complex (not Medium)."""
    assert selector.select(0.35).query_class == "Medium"
    assert selector.select(0.70).query_class == "Complex"
    assert selector.select(0.3499).query_class == "Simple"
    assert selector.select(0.6999).query_class == "Medium"


def test_out_of_range_scores_clamp(selector):
    assert selector.select(-0.5).query_class == "Simple"
    assert selector.select(1.5).query_class == "Complex"
