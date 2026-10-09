"""Dataset ingestion — build the corpus and precompute indexes to disk.

Downloads a small, no-auth public QA-over-documents dataset (SQuAD by default)
via the ``datasets`` library, deduplicates contexts into a passage corpus,
records gold (question, answer, gold-passage) triples for evaluation, and builds
the dense + BM25 indexes. After one run everything works offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from .config import Config, load_config
from .logging_utils import get_logger
from .retrieval.indexer import CorpusIndex, Passage

logger = get_logger(__name__)


def _normalize_ws(text: str) -> str:
    """Collapse all runs of whitespace to single spaces (for robust matching)."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_document(text: str, title: str, max_chars: int, min_chars: int) -> List[str]:
    """Split a document into passage-sized chunks on paragraph boundaries.

    Paragraphs (separated by blank lines) are greedily merged up to ``max_chars``;
    a paragraph longer than ``max_chars`` on its own becomes its own chunk. This
    keeps chunks semantically coherent while bounding their size for retrieval.

    Args:
        text: Raw document text.
        title: Document title (used only for logging).
        max_chars: Target maximum chunk length in characters.
        min_chars: Chunks below this length are merged with the next paragraph
            when possible.

    Returns:
        A list of chunk strings.
    """
    # Strip Markdown heading markers ("## Foo" -> "Foo") so the answering model
    # sees clean prose instead of echoing "#" symbols or bare heading words.
    text = re.sub(r"(?m)^#+\s*", "", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for para in paragraphs:
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) <= max_chars or len(buf) < min_chars:
            buf = candidate
        else:
            chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def _load_folder_records(cfg: Config) -> tuple[List[Passage], List[Dict]]:
    """Build a passage corpus from local .md/.txt files and resolve eval Q&A.

    Each document is chunked; each chunk becomes a passage whose title is the
    source filename. Eval questions from ``eval_questions_file`` are matched to
    their gold passage by locating the chunk that contains the ``gold_hint``
    (whitespace-normalized substring match).

    Args:
        cfg: Global config (dataset.docs_dir / eval_questions_file / chunking).

    Returns:
        Tuple of (passages, questions). Questions missing a resolvable gold
        passage are dropped with a warning.
    """
    docs_dir = Path(cfg.dataset.docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"docs_dir not found: {docs_dir}")

    files = sorted(
        [p for p in docs_dir.iterdir() if p.suffix.lower() in (".md", ".txt")]
    )
    if not files:
        raise FileNotFoundError(f"No .md/.txt files found in {docs_dir}")

    passages: List[Passage] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_document(
            text, path.stem, cfg.dataset.chunk_max_chars, cfg.dataset.chunk_min_chars
        ):
            passages.append(Passage(doc_id=len(passages), text=chunk, title=path.name))
    logger.info("Built %d passages from %d documents in %s", len(passages), len(files), docs_dir)

    # Precompute normalized passage text for gold-hint resolution.
    norm_passages = [_normalize_ws(p.text).lower() for p in passages]

    questions: List[Dict] = []
    q_file = Path(cfg.dataset.eval_questions_file)
    if q_file.exists():
        for line in q_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            hint = _normalize_ws(obj.get("gold_hint", "")).lower()
            gold_id = -1
            if hint:
                for i, ptext in enumerate(norm_passages):
                    if hint in ptext:
                        gold_id = i
                        break
            if gold_id < 0:
                logger.warning("Could not resolve gold passage for question: %s", obj["question"])
                continue
            questions.append(
                {
                    "question": obj["question"].strip(),
                    "answers": obj.get("answers", [""]) or [""],
                    "gold_doc_id": gold_id,
                }
            )
        logger.info("Resolved %d eval questions from %s", len(questions), q_file)
    else:
        logger.warning("Eval questions file not found: %s (eval will be unavailable)", q_file)

    return passages, questions


def _load_squad_records(cfg: Config) -> tuple[List[Passage], List[Dict]]:
    """Load a SQuAD subset into a passage corpus + question set.

    Deduplicates contexts (many questions share a context) so each unique
    passage is a corpus document, and maps each kept question to its gold
    passage id.

    Args:
        cfg: Global config (dataset name/split/sizes).

    Returns:
        Tuple of (passages, questions). Each question dict has keys
        ``question``, ``answers`` (list[str]), ``gold_doc_id``.
    """
    from datasets import load_dataset

    logger.info("Downloading dataset '%s' [%s] ...", cfg.dataset.name, cfg.dataset.split)
    ds = load_dataset(cfg.dataset.name, split=cfg.dataset.split)

    context_to_id: Dict[str, int] = {}
    passages: List[Passage] = []
    questions: List[Dict] = []
    q_per_context: Dict[int, int] = {}
    cap = max(1, cfg.dataset.questions_per_context)

    for row in ds:
        context = row["context"].strip()
        if context not in context_to_id:
            if len(passages) >= cfg.dataset.num_passages:
                # Corpus is full; only keep questions whose context is already in.
                if len(questions) >= cfg.dataset.num_questions:
                    break
                continue
            doc_id = len(passages)
            context_to_id[context] = doc_id
            passages.append(Passage(doc_id=doc_id, text=context, title=row.get("title", "")))

        # Cap questions per context so the eval set spans many distinct passages
        # (SQuAD is ordered; many consecutive questions share one context).
        doc_id = context_to_id[context]
        if len(questions) < cfg.dataset.num_questions and q_per_context.get(doc_id, 0) < cap:
            answers = row.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
            questions.append(
                {
                    "question": row["question"].strip(),
                    "answers": list(answer_texts) or [""],
                    "gold_doc_id": doc_id,
                }
            )
            q_per_context[doc_id] = q_per_context.get(doc_id, 0) + 1

        if len(passages) >= cfg.dataset.num_passages and len(questions) >= cfg.dataset.num_questions:
            break

    logger.info("Built %d unique passages, %d questions", len(passages), len(questions))
    return passages, questions


def build(cfg: Config | None = None) -> None:
    """Run full ingestion: corpus + indexes + question set to disk.

    Args:
        cfg: Optional config; loaded from default location if omitted.
    """
    cfg = cfg or load_config()
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)

    source = cfg.dataset.source.lower()
    if source == "folder":
        passages, questions = _load_folder_records(cfg)
    elif source == "squad":
        passages, questions = _load_squad_records(cfg)
    else:
        raise ValueError(f"Unknown dataset.source: {cfg.dataset.source!r} (use 'folder' or 'squad')")

    # Persist the question/eval set.
    with cfg.paths.questions_path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q) + "\n")
    logger.info("Wrote %d questions to %s", len(questions), cfg.paths.questions_path)

    # Build + save dense and BM25 indexes.
    CorpusIndex.build_and_save(cfg, passages)

    # Build + save the corpus-derived domain vocabulary (Knowledge feature).
    if cfg.features.use_corpus_domain_vocab:
        from .domain_vocab import build_domain_vocab, save_domain_vocab

        vocab = build_domain_vocab(
            [p.text for p in passages],
            cfg.models.spacy_model,
            cfg.features.domain_vocab_top_n,
        )
        if vocab:
            save_domain_vocab(
                cfg.paths.domain_vocab_path(cfg.features.domain_vocab_file), vocab
            )

    logger.info("Ingestion complete. Indexes written to %s", cfg.paths.data_dir)


def load_questions(cfg: Config) -> List[Dict]:
    """Load the persisted evaluation question set.

    Args:
        cfg: Global config.

    Returns:
        List of question dicts (``question``, ``answers``, ``gold_doc_id``).

    Raises:
        FileNotFoundError: If the question set has not been built.
    """
    if not cfg.paths.questions_path.exists():
        raise FileNotFoundError(
            f"Question set not found at {cfg.paths.questions_path}. Run `rag ingest` first."
        )
    questions: List[Dict] = []
    with cfg.paths.questions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            questions.append(json.loads(line))
    return questions


if __name__ == "__main__":  # pragma: no cover
    from .logging_utils import configure_logging

    configure_logging(verbose=True)
    build()
