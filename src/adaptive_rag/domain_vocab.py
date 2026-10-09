"""Corpus-derived domain vocabulary (for the Knowledge / domain-density feature).

Instead of a hand-authored keyword list, this module extracts a weighted domain
vocabulary directly from the ingested corpus — fully offline, no training:

1. Lemmatize every passage with spaCy, keeping only content-word lemmas
   (NOUN/ADJ/VERB), dropping stop words, very short tokens, and any lemma that
   ever appears inside a named-entity span (company / product / place names --
   those are the corpus's subject, already captured by entity density, and are
   excluded robustly here rather than relying on POS tags, which spaCy assigns
   inconsistently to the same word).
2. Fit a TF-IDF vectorizer over the passages (each passage = one document) to get
   each term's corpus salience (its summed TF-IDF mass across passages).
3. **Contrastive re-weighting**: multiply salience by a *specificity* factor that
   is large for words rare in general English and near-zero for common ones,
   using ``wordfreq``'s Zipf frequency. This suppresses corpus-frequent but
   generic words ("day", "company", "year") and promotes genuinely specialized
   terms ("encryption", "biometric", "retention"). If ``wordfreq`` is not
   installed the step is skipped (plain TF-IDF salience is used).
4. Keep the top-N terms; weights are rescaled to ``(0, 1]`` (top term = 1.0).

The result is a ``{lemma: weight}`` map: interpretable (printable ranked list),
corpus-adaptive (a new corpus yields a new vocabulary, no manual editing), and
weighted (rare technical terms move the density more than common ones).
See :func:`build_domain_vocab`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .logging_utils import get_logger

logger = get_logger(__name__)

# Content POS eligible to be domain terms. PROPN (named entities: company /
# product / place names) is deliberately excluded — those are captured by the
# separate entity-density feature, so including them here would double-count and
# inflate every query that names the corpus's own subject. Domain vocabulary is
# specialized *terminology* (common nouns, adjectives, verbs), not entities.
_CONTENT_POS = {"NOUN", "ADJ", "VERB"}
_MIN_LEN = 3  # ignore very short lemmas ("id", "os", ...)


def _lemmatize_passages(texts: List[str], spacy_model: str):
    """Lemmatize passages and collect named-entity lemmas.

    Returns:
        Tuple ``(docs, entity_lemmas)`` where ``docs`` is one space-joined string
        of content lemmas per passage, and ``entity_lemmas`` is the set of lemmas
        that appear inside any named-entity span anywhere in the corpus. The
        latter is used to exclude entity terms (company / product / place names)
        from the domain vocabulary robustly -- POS-based exclusion alone is leaky
        because spaCy tags the same token inconsistently (e.g. "Northwind" as
        PROPN in most passages but NOUN in a few).
    """
    from .preprocessing import _load_nlp

    nlp = _load_nlp(spacy_model)
    docs_out: List[str] = []
    entity_lemmas: set[str] = set()
    # nlp.pipe is much faster than calling nlp() per passage.
    for doc in nlp.pipe(texts, batch_size=64):
        lemmas = []
        for t in doc:
            if t.ent_type_ and t.is_alpha and len(t.lemma_) >= _MIN_LEN:
                entity_lemmas.add(t.lemma_.lower())
            if (
                t.pos_ in _CONTENT_POS
                and t.is_alpha
                and not t.is_stop
                and len(t.lemma_) >= _MIN_LEN
            ):
                lemmas.append(t.lemma_.lower())
        docs_out.append(" ".join(lemmas))
    return docs_out, entity_lemmas


def _specificity_factor():
    """Return a function term -> specificity in ``[0, 1]`` (1 = rare in English).

    Uses ``wordfreq`` Zipf frequency (0 for unknown/rare words, ~7-8 for "the").
    A word at Zipf ``z`` gets specificity ``clip((MAX_ZIPF - z) / MAX_ZIPF)`` so
    common words approach 0 and unseen technical terms approach 1. If ``wordfreq``
    is unavailable, returns a constant 1.0 (no contrastive re-weighting).
    """
    try:
        from wordfreq import zipf_frequency
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("wordfreq not installed; using plain TF-IDF salience (no contrastive weighting).")
        return lambda term: 1.0

    MAX_ZIPF = 8.0

    def factor(term: str) -> float:
        z = zipf_frequency(term, "en")
        return max(0.0, min(1.0, (MAX_ZIPF - z) / MAX_ZIPF))

    return factor


def build_domain_vocab(
    texts: List[str], spacy_model: str, top_n: int = 80
) -> Dict[str, float]:
    """Build a contrastively-weighted domain vocabulary from corpus passages.

    Args:
        texts: Corpus passage texts.
        spacy_model: spaCy model name used for lemmatization.
        top_n: Number of top-scoring terms to keep.

    Returns:
        Mapping of ``lemma -> weight`` with weights in ``(0, 1]`` (highest-scoring
        term = 1.0). Empty if the corpus is too small to fit a vectorizer.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    lemma_docs, entity_lemmas = _lemmatize_passages(texts, spacy_model)
    non_empty = [d for d in lemma_docs if d.strip()]
    if len(non_empty) < 2:
        logger.warning("Corpus too small for TF-IDF vocab (%d docs); skipping.", len(non_empty))
        return {}

    # min_df=2 drops one-off terms (noise); English stop words as a safety net
    # on top of the spaCy stop-word filter already applied.
    vectorizer = TfidfVectorizer(
        min_df=2,
        stop_words="english",
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )
    matrix = vectorizer.fit_transform(lemma_docs)
    terms = vectorizer.get_feature_names_out()
    salience = matrix.sum(axis=0).A1  # summed TF-IDF mass per term

    # Contrastive re-weighting: salience * specificity(term). Exclude terms that
    # occur inside named-entity spans (company / product / place names) — those
    # are the corpus's *subject*, already measured by entity density, and would
    # otherwise inflate every query that names them.
    specificity = _specificity_factor()
    scored = [
        (term, float(sal) * specificity(term))
        for term, sal in zip(terms, salience)
        if term not in entity_lemmas
    ]
    ranked = sorted(scored, key=lambda kv: kv[1], reverse=True)
    ranked = [(t, s) for t, s in ranked if s > 0.0][:top_n]
    n_excluded = len(terms) - len(scored)
    logger.info("Excluded %d entity-span terms from domain vocab", n_excluded)

    if not ranked:
        return {}
    max_score = ranked[0][1] or 1.0
    vocab = {term: round(float(score) / float(max_score), 4) for term, score in ranked}
    logger.info(
        "Built domain vocab: %d terms from %d passages (top: %s)",
        len(vocab), len(lemma_docs), ", ".join(list(vocab)[:8]),
    )
    return vocab


def save_domain_vocab(path: Path, vocab: Dict[str, float]) -> None:
    """Persist the domain vocabulary to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2, sort_keys=False)
    logger.info("Saved domain vocab (%d terms) to %s", len(vocab), path)


def load_domain_vocab(path: Path) -> Dict[str, float]:
    """Load a domain vocabulary from JSON, or return {} if absent."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return {str(k).lower(): float(v) for k, v in json.load(fh).items()}
