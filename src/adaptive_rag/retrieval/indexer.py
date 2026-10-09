"""Corpus + index construction, persistence, and loading (Module 7 support).

Builds and caches to disk:

* the corpus (``corpus.jsonl`` — one passage per line),
* dense embeddings (``embeddings.npy`` — L2-normalized float32 matrix),
* the BM25 index (``bm25.pkl`` — a pickled ``BM25Okapi`` + tokenization),
* index metadata (``index_meta.json`` — model names, sizes).

After a single ``ingest`` run everything loads from disk and runs offline.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import Config
from ..logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def bm25_tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenization used consistently for BM25."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Passage:
    """A corpus passage."""

    doc_id: int
    text: str
    title: str = ""


class CorpusIndex:
    """In-memory corpus with its dense and sparse indexes.

    Instances are created by :meth:`load` (reads prebuilt artifacts) or by the
    ingestion script via :meth:`build_and_save`.
    """

    def __init__(
        self,
        passages: List[Passage],
        embeddings: Optional[np.ndarray],
        bm25,
        embedding_model_name: str,
    ) -> None:
        self.passages = passages
        self.embeddings = embeddings  # (N, D) float32, L2-normalized
        self.bm25 = bm25
        self.embedding_model_name = embedding_model_name

    @property
    def texts(self) -> List[str]:
        return [p.text for p in self.passages]

    def __len__(self) -> int:
        return len(self.passages)

    # ---------------------------------------------------------------- building
    @classmethod
    def build_and_save(
        cls,
        cfg: Config,
        passages: List[Passage],
        embed_model=None,
    ) -> "CorpusIndex":
        """Build dense + BM25 indexes from passages and persist to disk.

        Args:
            cfg: Global config (for paths + model names).
            passages: Corpus passages to index.
            embed_model: Optional preloaded SentenceTransformer; loaded from
                config if not provided.

        Returns:
            The constructed :class:`CorpusIndex`.
        """
        from rank_bm25 import BM25Okapi

        data_dir = cfg.paths.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        # --- corpus ---------------------------------------------------------
        with cfg.paths.corpus_path.open("w", encoding="utf-8") as fh:
            for p in passages:
                fh.write(json.dumps({"doc_id": p.doc_id, "title": p.title, "text": p.text}) + "\n")
        logger.info("Wrote %d passages to %s", len(passages), cfg.paths.corpus_path)

        # --- dense embeddings ----------------------------------------------
        if embed_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s' ...", cfg.models.embedding_model)
            embed_model = SentenceTransformer(cfg.models.embedding_model, device="cpu")

        logger.info("Encoding %d passages (this runs once) ...", len(passages))
        embeddings = embed_model.encode(
            [p.text for p in passages],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,  # enables cosine == dot product
            show_progress_bar=False,
        ).astype(np.float32)
        np.save(cfg.paths.embeddings_path, embeddings)
        logger.info("Saved embeddings %s to %s", embeddings.shape, cfg.paths.embeddings_path)

        # --- BM25 -----------------------------------------------------------
        tokenized = [bm25_tokenize(p.text) for p in passages]
        bm25 = BM25Okapi(tokenized)
        with cfg.paths.bm25_path.open("wb") as fh:
            pickle.dump({"bm25": bm25}, fh)
        logger.info("Saved BM25 index to %s", cfg.paths.bm25_path)

        # --- metadata -------------------------------------------------------
        # Record the true source: for folder ingestion the corpus is the docs
        # directory, not SQuAD (whose name/split only apply to source == "squad").
        source = cfg.dataset.source.lower()
        if source == "folder":
            dataset_label = f"folder:{Path(cfg.dataset.docs_dir).name}"
            split_label = ""
        else:
            dataset_label = cfg.dataset.name
            split_label = cfg.dataset.split
        meta = {
            "num_passages": len(passages),
            "embedding_model": cfg.models.embedding_model,
            "embedding_dim": int(embeddings.shape[1]),
            "source": source,
            "dataset": dataset_label,
            "split": split_label,
        }
        with cfg.paths.meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        return cls(passages, embeddings, bm25, cfg.models.embedding_model)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, cfg: Config) -> "CorpusIndex":
        """Load prebuilt corpus + indexes from disk.

        Args:
            cfg: Global config with paths.

        Returns:
            The loaded :class:`CorpusIndex`.

        Raises:
            FileNotFoundError: If the indexes have not been built yet.
        """
        if not cfg.paths.corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found at {cfg.paths.corpus_path}. Run `rag ingest` first."
            )
        passages: List[Passage] = []
        with cfg.paths.corpus_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                passages.append(Passage(doc_id=obj["doc_id"], text=obj["text"], title=obj.get("title", "")))

        embeddings = None
        if cfg.paths.embeddings_path.exists():
            embeddings = np.load(cfg.paths.embeddings_path).astype(np.float32)

        bm25 = None
        if cfg.paths.bm25_path.exists():
            with cfg.paths.bm25_path.open("rb") as fh:
                bm25 = pickle.load(fh)["bm25"]

        model_name = cfg.models.embedding_model
        if cfg.paths.meta_path.exists():
            with cfg.paths.meta_path.open("r", encoding="utf-8") as fh:
                model_name = json.load(fh).get("embedding_model", model_name)

        logger.info(
            "Loaded corpus (%d passages), embeddings=%s, bm25=%s",
            len(passages),
            None if embeddings is None else embeddings.shape,
            bm25 is not None,
        )
        return cls(passages, embeddings, bm25, model_name)
