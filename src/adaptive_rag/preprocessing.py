"""Module 1 — Query Preprocessing.

Convert a raw query string into a structured :class:`ParsedQuery` using spaCy's
``en_core_web_sm`` pipeline: tokenization, (optional) stop-word removal,
lemmatization, POS tagging, named entity recognition, and dependency parsing.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Token:
    """A single parsed token with its linguistic annotations."""

    text: str
    lemma: str
    pos: str          # coarse-grained POS tag (e.g. NOUN, VERB)
    tag: str          # fine-grained POS tag (e.g. NN, VBZ)
    dep: str          # dependency relation to its head
    head: str         # text of the syntactic head
    is_stop: bool
    is_alpha: bool
    is_punct: bool


@dataclass(frozen=True)
class Entity:
    """A named entity span."""

    text: str
    label: str        # entity type (e.g. ORG, GPE, LAW)
    start: int        # start token index
    end: int          # end token index (exclusive)


@dataclass(frozen=True)
class ParsedQuery:
    """Structured representation of a query (Module 1 output).

    Attributes:
        raw: The original query string.
        tokens: All tokens with linguistic annotations.
        content_tokens: Non-stop, non-punctuation alphabetic tokens.
        entities: Named entities detected in the query.
        dependency_edges: (child_index, head_index, relation) triples describing
            the dependency tree. A token that is its own head (the root) has
            ``head_index == child_index``.
        root_index: Index of the syntactic root token (-1 if empty).
        num_sentences: Number of sentences spaCy segmented the query into.
    """

    raw: str
    tokens: List[Token]
    content_tokens: List[Token]
    entities: List[Entity]
    dependency_edges: List[Tuple[int, int, str]]
    root_index: int
    num_sentences: int = 1

    @property
    def num_tokens(self) -> int:
        """Number of non-punctuation tokens (used as the query-length feature)."""
        return sum(1 for t in self.tokens if not t.is_punct)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ParsedQuery(raw={self.raw!r}, num_tokens={self.num_tokens}, "
            f"entities={[e.text for e in self.entities]}, "
            f"root={self.tokens[self.root_index].text if self.root_index >= 0 else None!r})"
        )


@functools.lru_cache(maxsize=4)
def _load_nlp(model_name: str):
    """Load and cache a spaCy pipeline.

    Args:
        model_name: spaCy model identifier (e.g. ``en_core_web_sm``).

    Returns:
        A loaded spaCy ``Language`` object.

    Raises:
        OSError: With an actionable message if the model is not installed.
    """
    import spacy

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - environment dependent
        raise OSError(
            f"spaCy model '{model_name}' is not installed. Run:\n"
            f"    python -m spacy download {model_name}"
        ) from exc
    logger.debug("Loaded spaCy model '%s'", model_name)
    return nlp


class QueryPreprocessor:
    """Wraps a spaCy pipeline to produce :class:`ParsedQuery` objects."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Initialize the preprocessor.

        Args:
            model_name: spaCy model to load.
        """
        self.model_name = model_name
        self._nlp = _load_nlp(model_name)

    def parse(self, query: str, remove_stopwords: bool = False) -> ParsedQuery:
        """Parse a raw query into a :class:`ParsedQuery`.

        Args:
            query: Raw query text.
            remove_stopwords: If True, exclude stop words from ``content_tokens``
                (they are always kept in ``tokens`` for structural analysis).

        Returns:
            The structured parse result.
        """
        doc = self._nlp(query)

        tokens: List[Token] = [
            Token(
                text=t.text,
                lemma=t.lemma_.lower(),
                pos=t.pos_,
                tag=t.tag_,
                dep=t.dep_,
                head=t.head.text,
                is_stop=t.is_stop,
                is_alpha=t.is_alpha,
                is_punct=t.is_punct,
            )
            for t in doc
        ]

        content_tokens = [
            tok
            for tok, t in zip(tokens, doc)
            if t.is_alpha and not t.is_punct and (not remove_stopwords or not t.is_stop)
        ]

        entities = [
            Entity(text=ent.text, label=ent.label_, start=ent.start, end=ent.end)
            for ent in doc.ents
        ]

        dependency_edges: List[Tuple[int, int, str]] = [
            (t.i, t.head.i, t.dep_) for t in doc
        ]

        root_index = -1
        for t in doc:
            if t.dep_ == "ROOT" or t.head.i == t.i:
                root_index = t.i
                break

        num_sentences = max(1, sum(1 for _ in doc.sents))

        parsed = ParsedQuery(
            raw=query,
            tokens=tokens,
            content_tokens=content_tokens,
            entities=entities,
            dependency_edges=dependency_edges,
            root_index=root_index,
            num_sentences=num_sentences,
        )
        logger.debug(
            "Parsed query %r -> %d tokens, %d content tokens, %d entities, root=%s",
            query,
            len(tokens),
            len(content_tokens),
            len(entities),
            tokens[root_index].text if root_index >= 0 else None,
        )
        return parsed


def dependency_depth(parsed: ParsedQuery) -> int:
    """Compute the maximum depth of the dependency tree.

    Depth is the longest path from the root to any leaf (root depth = 1). Works
    on the ``(child, head, relation)`` edge list and is robust to multiple
    sentences (returns the deepest tree). Cycles cannot occur in a well-formed
    dependency parse, but a visited-set guards against pathological input.

    Args:
        parsed: A parsed query.

    Returns:
        The maximum tree depth (0 for an empty query).
    """
    if not parsed.tokens:
        return 0

    # Build child adjacency: head -> [children], excluding self-loops (roots).
    children: dict[int, List[int]] = {}
    roots: List[int] = []
    for child, head, _rel in parsed.dependency_edges:
        if child == head:
            roots.append(child)
        else:
            children.setdefault(head, []).append(child)

    if not roots:  # fallback: treat token 0 as root
        roots = [0]

    def _depth(node: int, visited: set[int]) -> int:
        if node in visited:
            return 0
        visited.add(node)
        kids = children.get(node, [])
        if not kids:
            return 1
        return 1 + max(_depth(k, visited) for k in kids)

    return max(_depth(r, set()) for r in roots)
