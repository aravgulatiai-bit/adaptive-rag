"""Module 2 — Feature Extraction.

Extract exactly six lightweight numeric features from a :class:`ParsedQuery`
(no document retrieval happens here) and normalize each to ``[0, 1]`` using the
documented caps in the config.

Feature summary
---------------
====================  =========  ================================================
Feature               Category   Raw definition
====================  =========  ================================================
query_length          Linguistic non-punctuation token count
dependency_depth      Linguistic max depth of the dependency tree
entity_density        Knowledge  named entities / tokens
domain_keyword_density Knowledge domain-term matches / tokens
intent_type           Reasoning  {definition, procedure, comparison, reasoning}
semantic_ambiguity    Reasoning  mean WordNet polysemy of content words
====================  =========  ================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import FeaturesCfg
from .logging_utils import get_logger
from .preprocessing import ParsedQuery, dependency_depth

logger = get_logger(__name__)

# Intent detection patterns (checked in priority order). Comparison and
# reasoning are checked before procedure/definition because their cue words are
# more specific.
_COMPARISON_PATTERNS = [
    r"\bcompare\b", r"\bcompared\b", r"\bcomparison\b", r"\bversus\b", r"\bvs\.?\b",
    r"\bdifference(s)?\b", r"\bdiffer\b", r"\bcontrast\b", r"\bbetween\b.*\band\b",
    r"\bsimilar(ity|ities)?\b", r"\bbetter\b", r"\bworse\b", r"\bmore .* than\b",
]
_REASONING_PATTERNS = [
    r"\bwhy\b", r"\bexplain\b", r"\bhow come\b", r"\breason(s)?\b", r"\bcause(s)?\b",
    r"\bimplication(s)?\b", r"\bconsequence(s)?\b", r"\bimpact\b", r"\banalyze\b",
    r"\bjustify\b", r"\bwhat if\b", r"\baffect(s|ed)?\b", r"\binfluence\b",
]
_PROCEDURE_PATTERNS = [
    r"\bhow (to|do|does|can|should)\b", r"\bsteps?\b", r"\bprocedure\b", r"\bprocess\b",
    r"\binstall\b", r"\bconfigure\b", r"\bset up\b", r"\bimplement\b", r"\bcreate\b",
    r"\bbuild\b", r"\bmethod\b",
]
_DEFINITION_PATTERNS = [
    r"\bwhat is\b", r"\bwhat are\b", r"\bwho (is|are|was|were)\b", r"\bwhen (is|was)\b",
    r"\bwhere (is|are)\b", r"\bdefine\b", r"\bdefinition\b", r"\bmeaning of\b",
    r"\bwhich\b",
]

_INTENT_ORDER: List[Tuple[str, List[str]]] = [
    ("comparison", _COMPARISON_PATTERNS),
    ("reasoning", _REASONING_PATTERNS),
    ("procedure", _PROCEDURE_PATTERNS),
    ("definition", _DEFINITION_PATTERNS),
]

# POS tags whose lemmas count as "content words" for ambiguity + domain lookup.
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


@dataclass(frozen=True)
class FeatureVector:
    """The 6-D feature vector (Module 2 output).

    ``*_norm`` fields are normalized to ``[0, 1]``; ``*_raw`` fields keep the raw
    measurement for transparency/logging. ``intent_label`` records the discrete
    intent class before it was mapped to ``intent_type_norm``.
    """

    query_length_raw: int
    dependency_depth_raw: int
    entity_density_raw: float
    domain_keyword_density_raw: float
    intent_label: str
    semantic_ambiguity_raw: float

    query_length_norm: float
    dependency_depth_norm: float
    entity_density_norm: float
    domain_keyword_density_norm: float
    intent_type_norm: float
    semantic_ambiguity_norm: float

    def as_vector(self) -> List[float]:
        """Return the six normalized features as an ordered list."""
        return [
            self.query_length_norm,
            self.dependency_depth_norm,
            self.entity_density_norm,
            self.domain_keyword_density_norm,
            self.intent_type_norm,
            self.semantic_ambiguity_norm,
        ]

    def as_dict(self) -> Dict[str, float]:
        """Return a name->value mapping of the normalized features."""
        return {
            "query_length": self.query_length_norm,
            "dependency_depth": self.dependency_depth_norm,
            "entity_density": self.entity_density_norm,
            "domain_keyword_density": self.domain_keyword_density_norm,
            "intent_type": self.intent_type_norm,
            "semantic_ambiguity": self.semantic_ambiguity_norm,
        }


def _clip01(x: float) -> float:
    """Clip a value to ``[0, 1]``."""
    return max(0.0, min(1.0, x))


def classify_intent(query: str) -> str:
    """Classify query intent via ordered regex heuristics.

    Args:
        query: Raw query text.

    Returns:
        One of ``{"definition", "procedure", "comparison", "reasoning"}``.
        Defaults to ``"definition"`` when no cue matches (simplest intent).
    """
    q = query.lower()
    for label, patterns in _INTENT_ORDER:
        if any(re.search(p, q) for p in patterns):
            return label
    return "definition"


# The "between X and Y" cue is ambiguous ("between 1990 and 1995" is not a
# comparison), so it is validated against the parse rather than trusted lexically.
_BETWEEN_AND = r"\bbetween\b.*\band\b"
# Comparative-degree fine-grained POS tags: JJR (bigger), RBR (faster).
_COMPARATIVE_TAGS = {"JJR", "RBR"}
_NOMINAL_POS = {"NOUN", "PROPN"}


@dataclass(frozen=True)
class IntentResult:
    """Outcome of syntax-aware intent analysis.

    Attributes:
        label: The dominant intent (argmax of evidence), for display/logging.
        score: A continuous intent score in ``[0, 1]`` — the evidence-weighted
            mixture of the per-intent scalars in ``intent_scale``. Reduces to the
            single intent's scalar when only one intent has evidence.
        evidence: Per-intent evidence weights (for the verbose trace).
    """

    label: str
    score: float
    evidence: Dict[str, float]


def _structural_comparison_cues(parsed: ParsedQuery) -> List[str]:
    """Detect comparison evidence from the dependency parse (no cue words needed).

    Catches comparisons that lexical patterns miss, e.g. "Is Postgres or MySQL
    faster?" (comparative adjective + coordinated nouns), while ignoring
    non-comparative "between X and Y" over numbers/dates.

    Returns:
        A list of human-readable cue names (empty if none fire).
    """
    cues: List[str] = []
    toks = parsed.tokens

    # 1) Comparative morphology: a comparative adjective/adverb (faster, better).
    if any(t.tag in _COMPARATIVE_TAGS for t in toks):
        cues.append("comparative-morphology")

    # 2) Coordinated nominal alternatives joined by "or" (Postgres or MySQL).
    has_or = any(t.pos == "CCONJ" and t.lemma == "or" for t in toks)
    n_nominal = sum(1 for t in toks if t.pos in _NOMINAL_POS)
    if has_or and n_nominal >= 2:
        cues.append("or-coordination")

    # 3) "between X and Y" only counts when X, Y are nominal (not NUM/dates).
    if re.search(_BETWEEN_AND, parsed.raw.lower()):
        n_num = sum(1 for t in toks if t.pos in ("NUM",) or t.tag in ("CD",))
        if n_nominal >= 2 and n_num < 2:
            cues.append("between-nominals")

    return cues


def analyze_intent(parsed: ParsedQuery, intent_scale: Dict[str, float]) -> IntentResult:
    """Classify intent using lexical cues + the parse, returning a soft score.

    Evidence is accumulated per intent from (a) lexical regex families and
    (b) structural cues from the dependency parse. The final score is the
    evidence-weighted average of the per-intent scalars, so a query that is
    partly comparison and partly reasoning lands between their anchors rather
    than snapping to one bucket. With a single intent's evidence it reproduces
    the legacy hard mapping exactly.

    Args:
        parsed: The parsed query.
        intent_scale: Mapping intent-label -> scalar in ``[0, 1]``.

    Returns:
        An :class:`IntentResult`.
    """
    q = parsed.raw.lower()
    evidence: Dict[str, float] = {k: 0.0 for k in ("definition", "procedure", "reasoning", "comparison")}

    # (a) Lexical evidence: 1.0 if any pattern in a family matches. The ambiguous
    # between-and cue is excluded here and handled structurally below.
    for label, patterns in _INTENT_ORDER:
        for p in patterns:
            if p == _BETWEEN_AND:
                continue
            if re.search(p, q):
                evidence[label] += 1.0
                break

    # (b) Structural evidence for comparison (each cue adds weight).
    cues = _structural_comparison_cues(parsed)
    evidence["comparison"] += float(len(cues))

    # Foreground intents (comparison/reasoning) express what the question is
    # *doing*; background intents (definition/procedure) are often just the
    # question frame ("what is the difference...", "how do X and Y compare?").
    # A fired foreground intent therefore suppresses background dilution — this
    # keeps genuine comparisons at their anchor instead of averaging them down.
    foreground = {"comparison": evidence["comparison"], "reasoning": evidence["reasoning"]}
    fg_total = sum(foreground.values())
    if fg_total > 0.0:
        active = foreground
        total = fg_total
    else:
        active = {"procedure": evidence["procedure"], "definition": evidence["definition"]}
        total = sum(active.values())

    if total <= 0.0:
        # No cue at all -> simplest intent (matches legacy default).
        return IntentResult(label="definition", score=_clip01(intent_scale.get("definition", 0.10)), evidence=evidence)

    # Continuous score = evidence-weighted mixture over the active intent group.
    score = sum(active[k] * intent_scale.get(k, 0.10) for k in active) / total
    label = max(active, key=lambda k: active[k])
    return IntentResult(label=label, score=_clip01(score), evidence=evidence)


class _WordNetPolysemy:
    """Lazy WordNet accessor for polysemy (synset count) lookups.

    Loaded lazily so importing this module does not require the NLTK corpus to
    be present, and so a missing corpus degrades gracefully to a neutral value.
    """

    def __init__(self) -> None:
        self._wn = None
        self._available: bool | None = None

    def _ensure(self) -> None:
        if self._available is not None:
            return
        try:
            from nltk.corpus import wordnet as wn

            wn.ensure_loaded()
            self._wn = wn
            self._available = True
            logger.debug("WordNet corpus loaded for semantic ambiguity feature.")
        except Exception as exc:  # pragma: no cover - environment dependent
            self._available = False
            logger.warning(
                "WordNet unavailable (%s); semantic ambiguity will default to 0.0. "
                "Run: python -c \"import nltk; nltk.download('wordnet')\"",
                exc,
            )

    def polysemy(self, lemma: str) -> int:
        """Return the number of WordNet synsets for a lemma (0 if unknown)."""
        self._ensure()
        if not self._available or self._wn is None:
            return 0
        try:
            return len(self._wn.synsets(lemma))
        except Exception:  # pragma: no cover
            return 0


_POLYSEMY = _WordNetPolysemy()


class FeatureExtractor:
    """Extracts the normalized 6-D feature vector from a parsed query."""

    def __init__(
        self, cfg: FeaturesCfg, domain_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """Initialize with feature configuration.

        Args:
            cfg: Feature normalization caps and lookup tables.
            domain_weights: Optional corpus-derived ``{lemma: weight}`` vocabulary
                (from :mod:`domain_vocab`). Merged with the hand-authored
                ``cfg.domain_keywords`` (each given weight 1.0 unless already
                present with a higher weight). When omitted, only the
                hand-authored list is used, with flat weight 1.0 — reproducing
                the legacy count-based behavior.
        """
        self.cfg = cfg
        # Build the weighted domain lookup: corpus vocab (if any) + config list.
        weights: Dict[str, float] = dict(domain_weights or {})
        for kw in cfg.domain_keywords:
            weights[kw] = max(weights.get(kw, 0.0), 1.0)
        self._domain_weights = weights
        self._domain = set(weights)  # kept for membership checks / tests

    def extract(self, parsed: ParsedQuery) -> FeatureVector:
        """Extract and normalize the six features.

        Args:
            parsed: A :class:`ParsedQuery`.

        Returns:
            A populated :class:`FeatureVector`.
        """
        cfg = self.cfg

        # --- Linguistic features ------------------------------------------------
        n_tokens = max(1, parsed.num_tokens)  # avoid division by zero
        query_length_raw = parsed.num_tokens
        depth_raw = dependency_depth(parsed)

        # --- Knowledge features -------------------------------------------------
        # Entity density: count entity spans (not sub-tokens) relative to tokens.
        entity_density_raw = len(parsed.entities) / n_tokens

        # Weighted domain mass: sum the vocabulary weight of each matching
        # content token (a term's lemma or surface form), divided by tokens.
        # Rare technical terms (weight ~1.0) move this more than common ones.
        domain_mass = sum(
            self._domain_weights.get(t.lemma, self._domain_weights.get(t.text.lower(), 0.0))
            for t in parsed.content_tokens
        )
        domain_density_raw = domain_mass / n_tokens

        # --- Reasoning features -------------------------------------------------
        if cfg.intent_syntax_aware:
            intent_res = analyze_intent(parsed, cfg.intent_scale)
            intent_label = intent_res.label
            intent_norm = intent_res.score
        else:
            intent_label = classify_intent(parsed.raw)
            intent_norm = cfg.intent_scale.get(intent_label, 0.1)

        content_lemmas = [
            t.lemma for t in parsed.content_tokens if t.pos in _CONTENT_POS
        ]
        if content_lemmas:
            polysemies = [_POLYSEMY.polysemy(lem) for lem in content_lemmas]
            mean_polysemy = sum(polysemies) / len(polysemies)
        else:
            mean_polysemy = 0.0

        # --- Normalization ------------------------------------------------------
        fv = FeatureVector(
            query_length_raw=query_length_raw,
            dependency_depth_raw=depth_raw,
            entity_density_raw=entity_density_raw,
            domain_keyword_density_raw=domain_density_raw,
            intent_label=intent_label,
            semantic_ambiguity_raw=mean_polysemy,
            query_length_norm=_clip01(query_length_raw / cfg.query_length_cap),
            dependency_depth_norm=_clip01(depth_raw / cfg.dependency_depth_cap),
            entity_density_norm=_clip01(entity_density_raw / cfg.entity_density_cap),
            domain_keyword_density_norm=_clip01(domain_density_raw / cfg.domain_density_cap),
            intent_type_norm=_clip01(intent_norm),
            semantic_ambiguity_norm=_clip01(mean_polysemy / cfg.semantic_ambiguity_cap),
        )
        logger.debug(
            "Features for %r: length=%d(%.2f) depth=%d(%.2f) ent=%.3f(%.2f) "
            "dom=%.3f(%.2f) intent=%s(%.2f) ambig=%.2f(%.2f)",
            parsed.raw,
            fv.query_length_raw, fv.query_length_norm,
            fv.dependency_depth_raw, fv.dependency_depth_norm,
            fv.entity_density_raw, fv.entity_density_norm,
            fv.domain_keyword_density_raw, fv.domain_keyword_density_norm,
            fv.intent_label, fv.intent_type_norm,
            fv.semantic_ambiguity_raw, fv.semantic_ambiguity_norm,
        )
        return fv
