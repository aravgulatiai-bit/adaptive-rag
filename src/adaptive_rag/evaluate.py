"""Evaluation harness — adaptive vs. fixed baselines.

Compares three strategies on the same question set:

* **adaptive**       — this system (fuzzy complexity -> adaptive policy),
* **fixed_simple**   — always Top-2 dense, no re-ranking,
* **fixed_complex**  — always Top-10 hybrid + cross-encoder re-ranking.

Per strategy it reports: recall@k (gold passage retrieved), token-level F1 / EM
against gold answers, mean end-to-end latency, mean retrieval cost (docs
retrieved + re-rank usage). For the adaptive run it also reports the complexity
score distribution and query-class assignments. Output goes to the console as a
table and to CSV.
"""

from __future__ import annotations

import csv
import re
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .config import Config, load_config
from .ingest import load_questions
from .llm import BaseLLM
from .logging_utils import get_logger
from .pipeline import AdaptiveRAGPipeline
from .policy import RetrievalPolicy
from .retrieval.types import RetrievedDoc

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SQuAD-style answer normalization + metrics.
# ---------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation/articles/extra whitespace (SQuAD style)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between a prediction and a single ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_over_answers(prediction: str, answers: List[str], metric: Callable[[str, str], float]) -> float:
    """Return the max metric over all acceptable gold answers."""
    if not answers:
        return 0.0
    return max(metric(prediction, ans) for ans in answers)


# ---------------------------------------------------------------------------
# Strategy definitions.
# ---------------------------------------------------------------------------
@dataclass
class StrategyResult:
    """Accumulated per-strategy metrics."""

    name: str
    n: int = 0
    recall_hits: int = 0
    f1_sum: float = 0.0
    em_sum: float = 0.0
    latency_sum: float = 0.0
    docs_sum: int = 0
    rerank_count: int = 0
    # Adaptive-only diagnostics.
    class_counts: Counter = field(default_factory=Counter)
    scores: List[float] = field(default_factory=list)

    def add(
        self,
        recall_hit: bool,
        f1: float,
        em: float,
        latency: float,
        docs: int,
        reranked: bool,
        query_class: str = "",
        score: float | None = None,
    ) -> None:
        self.n += 1
        self.recall_hits += int(recall_hit)
        self.f1_sum += f1
        self.em_sum += em
        self.latency_sum += latency
        self.docs_sum += docs
        self.rerank_count += int(reranked)
        if query_class:
            self.class_counts[query_class] += 1
        if score is not None:
            self.scores.append(score)

    @property
    def recall(self) -> float:
        return self.recall_hits / self.n if self.n else 0.0

    @property
    def mean_f1(self) -> float:
        return self.f1_sum / self.n if self.n else 0.0

    @property
    def mean_em(self) -> float:
        return self.em_sum / self.n if self.n else 0.0

    @property
    def mean_latency(self) -> float:
        return self.latency_sum / self.n if self.n else 0.0

    @property
    def mean_docs(self) -> float:
        return self.docs_sum / self.n if self.n else 0.0

    @property
    def rerank_pct(self) -> float:
        return 100.0 * self.rerank_count / self.n if self.n else 0.0


def _fixed_simple_policy(cfg: Config) -> RetrievalPolicy:
    return RetrievalPolicy(
        query_class="FixedSimple",
        top_k=cfg.evaluation.fixed_simple_top_k,
        use_dense=True,
        use_bm25=False,
        rerank=False,
    )


def _fixed_complex_policy(cfg: Config) -> RetrievalPolicy:
    return RetrievalPolicy(
        query_class="FixedComplex",
        top_k=cfg.evaluation.fixed_complex_top_k,
        use_dense=True,
        use_bm25=True,
        rerank=True,
    )


def _recall_hit(docs: List[RetrievedDoc], gold_doc_id: int) -> bool:
    """True if the gold passage id appears among retrieved docs."""
    return any(d.doc_id == gold_doc_id for d in docs)


class Evaluator:
    """Runs the three strategies over the question set and aggregates metrics."""

    def __init__(self, cfg: Config | None = None, pipeline: AdaptiveRAGPipeline | None = None) -> None:
        self.cfg = cfg or load_config()
        self.pipeline = pipeline or AdaptiveRAGPipeline(self.cfg)

    def _run_fixed(
        self, strategy: StrategyResult, query: str, gold_doc_id: int, answers: List[str],
        policy: RetrievalPolicy, generate: bool,
    ) -> None:
        start = time.perf_counter()
        docs, reranked = self.pipeline.retrieve(query, policy)
        pred = ""
        if generate:
            pred = self.pipeline.answer_from_docs(query, docs)
        latency = time.perf_counter() - start
        strategy.add(
            recall_hit=_recall_hit(docs, gold_doc_id),
            f1=metric_over_answers(pred, answers, token_f1) if generate else 0.0,
            em=metric_over_answers(pred, answers, exact_match) if generate else 0.0,
            latency=latency, docs=len(docs), reranked=reranked,
        )

    def evaluate(self, limit: int | None = None, generate: bool = True) -> Dict[str, StrategyResult]:
        """Evaluate all strategies.

        Args:
            limit: Optional cap on the number of questions (for a quick run).
            generate: If False, skip LLM generation and report retrieval metrics
                only (F1/EM will be 0). Much faster.

        Returns:
            Mapping strategy name -> :class:`StrategyResult`.
        """
        questions = load_questions(self.cfg)
        if limit is not None:
            questions = questions[:limit]

        # Warm-up (untimed): exercise every lazily-loaded component (query
        # encoder, cross-encoder, LLM) once so that one-time model loading is
        # excluded from per-query latency. Without this, whichever strategy
        # happens to trigger a lazy load first absorbs it into its mean, which
        # badly skews small-n comparisons.
        logger.info("Warming up models (untimed) ...")
        warmup_policy = _fixed_complex_policy(self.cfg)
        warmup_docs, _ = self.pipeline.retrieve("warm-up query", warmup_policy)
        if generate:
            self.pipeline.answer_from_docs("warm-up query", warmup_docs)

        results = {
            "adaptive": StrategyResult("adaptive"),
            "fixed_simple": StrategyResult("fixed_simple"),
            "fixed_complex": StrategyResult("fixed_complex"),
        }
        simple_policy = _fixed_simple_policy(self.cfg)
        complex_policy = _fixed_complex_policy(self.cfg)

        for i, q in enumerate(questions, 1):
            query = q["question"]
            gold = q["gold_doc_id"]
            answers = q.get("answers", [""])

            # Adaptive.
            start = time.perf_counter()
            _, _, _, score, policy = self.pipeline.analyze(query)
            docs, reranked = self.pipeline.retrieve(query, policy)
            pred = self.pipeline.answer_from_docs(query, docs) if generate else ""
            latency = time.perf_counter() - start
            results["adaptive"].add(
                recall_hit=_recall_hit(docs, gold),
                f1=metric_over_answers(pred, answers, token_f1) if generate else 0.0,
                em=metric_over_answers(pred, answers, exact_match) if generate else 0.0,
                latency=latency, docs=len(docs), reranked=reranked,
                query_class=policy.query_class, score=score,
            )

            # Fixed baselines.
            self._run_fixed(results["fixed_simple"], query, gold, answers, simple_policy, generate)
            self._run_fixed(results["fixed_complex"], query, gold, answers, complex_policy, generate)

            if i % 10 == 0 or i == len(questions):
                logger.info("Evaluated %d/%d questions", i, len(questions))

        return results


def format_table(results: Dict[str, StrategyResult]) -> str:
    """Format the per-strategy metrics as a console table."""
    header = (
        f"{'Strategy':<15}{'Recall@k':>10}{'F1':>8}{'EM':>8}"
        f"{'Latency(s)':>12}{'MeanDocs':>10}{'Rerank%':>9}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for name in ("adaptive", "fixed_simple", "fixed_complex"):
        r = results[name]
        rows.append(
            f"{r.name:<15}{r.recall:>10.3f}{r.mean_f1:>8.3f}{r.mean_em:>8.3f}"
            f"{r.mean_latency:>12.3f}{r.mean_docs:>10.2f}{r.rerank_pct:>8.1f}%"
        )
    return "\n".join(rows)


def format_distribution(results: Dict[str, StrategyResult]) -> str:
    """Format the adaptive complexity/class distribution."""
    adaptive = results["adaptive"]
    lines = ["", "Adaptive query-class distribution:"]
    total = sum(adaptive.class_counts.values()) or 1
    for cls in ("Simple", "Medium", "Complex"):
        c = adaptive.class_counts.get(cls, 0)
        lines.append(f"  {cls:<8}: {c:>4}  ({100.0 * c / total:5.1f}%)")
    if adaptive.scores:
        scores = sorted(adaptive.scores)
        mean = sum(scores) / len(scores)
        median = scores[len(scores) // 2]
        lines.append(
            f"Complexity score: min={scores[0]:.3f} median={median:.3f} "
            f"mean={mean:.3f} max={scores[-1]:.3f}"
        )
    return "\n".join(lines)


def write_csv(results: Dict[str, StrategyResult], cfg: Config) -> str:
    """Write the summary metrics to ``results/eval_results.csv``.

    Returns:
        The output path as a string.
    """
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.paths.results_dir / "eval_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["strategy", "n", "recall_at_k", "mean_f1", "mean_em",
             "mean_latency_s", "mean_docs", "rerank_pct"]
        )
        for name in ("adaptive", "fixed_simple", "fixed_complex"):
            r = results[name]
            writer.writerow(
                [r.name, r.n, f"{r.recall:.4f}", f"{r.mean_f1:.4f}", f"{r.mean_em:.4f}",
                 f"{r.mean_latency:.4f}", f"{r.mean_docs:.2f}", f"{r.rerank_pct:.1f}"]
            )
        # Adaptive distribution rows.
        writer.writerow([])
        writer.writerow(["adaptive_class", "count"])
        for cls in ("Simple", "Medium", "Complex"):
            writer.writerow([cls, results["adaptive"].class_counts.get(cls, 0)])
    logger.info("Wrote evaluation results to %s", out_path)
    return str(out_path)


def run_evaluation(cfg: Config | None = None, limit: int | None = None, generate: bool = True) -> str:
    """Run the full evaluation and print + persist the results.

    Args:
        cfg: Optional config.
        limit: Optional question cap.
        generate: Whether to run LLM generation (F1/EM) or retrieval-only.

    Returns:
        The report text (table + distribution).
    """
    cfg = cfg or load_config()
    evaluator = Evaluator(cfg)
    results = evaluator.evaluate(limit=limit, generate=generate)
    table = format_table(results)
    dist = format_distribution(results)
    csv_path = write_csv(results, cfg)
    report = f"{table}\n{dist}\n\nResults CSV: {csv_path}"
    return report
