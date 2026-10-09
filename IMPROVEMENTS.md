# Adaptive RAG vs. Traditional Fixed-Strategy RAG

## Improvements and Measured Proof

This document summarizes what the Adaptive RAG system (fuzzy query-complexity
routing) improves over a **traditional RAG**, which applies one fixed retrieval
configuration to every query, and presents the evidence gathered from this
repository's own evaluation harness and runtime traces.

**Traditional baselines used for comparison** (both implemented in
`src/adaptive_rag/evaluate.py` and run on identical questions, corpus, models,
and hardware):

- **fixed_simple** — always Top-2 dense retrieval, no re-ranking (a cheap
  traditional setup).
- **fixed_complex** — always Top-10 hybrid (BM25 + dense) + cross-encoder
  re-ranking (an expensive traditional setup).

The adaptive system classifies each query with a hierarchical fuzzy engine
(score ∈ [0,1]) and routes it to Simple / Medium / Complex retrieval
accordingly. All results below are reproducible with the commands listed at the
end.

---

## Improvement 1 — Complex-tier answer quality at a fraction of the cost

**Claim.** The adaptive system matches the answer quality and recall of the
expensive always-complex configuration while spending far less compute, because
it only escalates the small fraction of queries that need it.

**Proof — enterprise corpus evaluation** (20 documents → 56 passages, 22
gold-labeled questions, CPU, flan-t5-base; warmed-up timing — one untimed
warm-up query precedes measurement so one-time model loading is excluded;
saved in `results/eval_results.csv`):

| Strategy       | Recall@k | F1     | EM     | Mean latency | Mean docs | Re-rank usage |
|----------------|---------:|-------:|-------:|-------------:|----------:|--------------:|
| **adaptive**   |    1.000 | 0.9167 | 0.7727 |  **0.466 s** |  **4.91** |      **9.1%** |
| fixed_simple   |    1.000 | 0.9167 | 0.7727 |      0.266 s |      2.00 |          0.0% |
| fixed_complex  |    1.000 | 0.9167 | 0.7727 |      0.752 s |     10.00 |        100.0% |

Identical quality across all three columns (Recall 1.000, F1 0.9167,
EM 0.7727) — but the adaptive system achieved it at:

- **~1.6× lower latency** than fixed_complex (0.466 s vs 0.752 s per query),
- **roughly half the retrieval volume** (4.91 vs 10 documents per query),
- **cross-encoder invoked on 9.1% of queries instead of 100%** (a ~11×
  reduction in re-ranking compute, the single most expensive retrieval step).

Query-class distribution chosen by the fuzzy engine: Simple 18.2%,
Medium 72.7%, Complex 9.1% (complexity scores: min 0.156, median 0.500,
mean 0.466, max 0.820). The engine correctly identified that most of this
workload did not need the expensive path — that recognition *is* the saving.

---

## Improvement 2 — Higher recall than the cheap traditional setup, paid for only where needed

**Claim.** Against the cheap traditional configuration, adaptive routing buys
substantially higher retrieval recall, and it does so without paying the
always-complex price.

**Proof — SQuAD evaluation** (300 Wikipedia passages, 200 questions each mapped
to a distinct gold passage, CPU, warmed-up timing; saved in
`results_squad/eval_results.csv`):

| Strategy       | Recall@k | Mean latency | Mean docs | Re-rank usage |
|----------------|---------:|-------------:|----------:|--------------:|
| **adaptive**   |**0.935** |      0.447 s |      4.93 |          5.0% |
| fixed_simple   |    0.755 |      0.310 s |      2.00 |          0.0% |
| fixed_complex  |    0.995 |      1.270 s |     10.00 |        100.0% |

On this larger, noisier corpus the cheap traditional setup **missed the gold
passage on 24.5% of questions** (recall 0.755). The adaptive system closed 75%
of that gap (recall 0.935, within 6 points of the always-complex ceiling)
while running at **roughly one third of fixed_complex's latency and about half
its retrieval volume**, re-ranking only 5% of queries.

(One honest note: on SQuAD, *reader-side* F1/EM actually favor fixed_simple —
the tiny flan-t5-base reader is distracted by extra context faster than it
benefits from recall, and SQuAD-style QA appears in flan-t5's training
mixture, so it sometimes answers from memory. The retrieval-side comparison
above is the meaningful one; see the paper's Results/Limitations sections for
the full analysis.)

This is the general pattern: as the corpus grows and gets noisier, shallow
fixed retrieval degrades, and the adaptive system's selective escalation
becomes a *quality* win, not only a cost win.

---

## Improvement 3 — Per-query cost proportional to query difficulty

**Claim.** A traditional RAG charges every query the same cost regardless of
difficulty. The adaptive system's cost scales with measured complexity.

**Proof — runtime traces from this repo** (verbose `rag ask -v` runs):

| Query | Complexity score | Class | Docs retrieved | Re-ranked? |
|---|---:|---|---:|---|
| "Who is the CEO of Northwind Analytics?" | 0.168 | Simple | 2 | No |
| "Define RTO." | 0.156 | Simple | 2 | No |
| "How long is personal data retained after an account is closed?" | 0.500 | Medium | 5 | No |
| "What is the 401(k) company match?" | 0.500 | Medium | 5 | No |
| "What are the differences between the Starter, Growth, and Enterprise subscription tiers?" | 0.844 | Complex | 10 | Yes |
| "Compare the data-retention obligations under GDPR and CCPA for cloud providers processing biometric data" | 0.844 | Complex | 10 | Yes |

A trivial lookup consumed a 2-document dense search; a multi-entity comparison
consumed a 10-document hybrid search plus cross-encoder re-ranking. Under a
traditional RAG, both would have cost exactly the same — either both cheap
(risking the complex query's recall) or both expensive (wasting ~5× compute on
the trivial one).

The routing is also *explainable*: every decision comes with a full trace
(six features → three subsystem scores → overall score → policy), e.g. the
tiers question scored Complex via Linguistic 0.828 + Reasoning 0.842 (deep
parse + comparison intent) even though Knowledge was low (0.156) — a different
justification than the GDPR query, which was driven by domain density. A
traditional RAG has no equivalent notion of *why* a query got its treatment.

---

## Improvement 4 — Correct escalation of the queries that actually need depth

**Claim.** The fuzzy engine sends genuinely hard queries to the deep path and
trivial ones to the shallow path — it is not a coin flip.

**Proof.**

- The two spec-designed sanity queries separate cleanly:
  "What is GDPR?" → **0.156 (Simple)**; "Compare the data-retention obligations
  under GDPR and CCPA for cloud providers processing biometric data" →
  **0.844 (Complex)**.
- On the Complex-routed tiers question, the cross-encoder placed the two
  genuinely relevant passages (the SLA/support-tiers chunk, score +1.95, and
  the feature-tiers chunk) in the top-3 fed to the LLM, out of a 50-candidate
  fused pool.
- The test suite (36 passing tests, `pytest`) locks this behavior in:
  monotonicity tests (a long, entity-dense comparison query must outscore
  "What is X?"), boundary tests for every fuzzy subsystem, policy threshold
  tests at 0.35/0.70, and rule-base completeness over all 27 subsystem-level
  combinations (no input can produce an empty output).

---

## Improvement 5 — Cost control without a quality cliff (the trade-off is tunable)

**Claim.** Traditional RAG forces a one-time, global choice between cheap and
good. The adaptive system turns that into a per-query decision with tunable
thresholds.

**Proof / mechanism.** The 0.35 / 0.70 complexity thresholds, per-class top-k,
and re-rank flags live in `config/config.yaml` (`policy.bands`), not in code.
Shifting the Complex threshold down escalates more queries (more recall
insurance, more cost); shifting it up saves more compute. The evaluation
harness (`rag eval`) re-measures the whole trade-off after any change —
the cost/quality frontier is observable and adjustable, which a fixed pipeline
simply does not expose.

---

## Improvement 6 — Training-free routing that adapts to the corpus (no hand-tuned priors)

**Claim.** The two feature components most likely to be criticized as
hand-engineered priors — the domain-keyword list and the keyword-based intent
rules — are now derived automatically or from the parse, so the router adapts to
a new corpus with no manual editing, while staying training-free and fully
interpretable.

**Proof / mechanism.**

- **Corpus-derived domain vocabulary.** At ingest, a weighted vocabulary is
  extracted from the corpus by TF-IDF keyword extraction, then *contrastively*
  re-weighted by each term's rarity in general English (`wordfreq`), suppressing
  generic words and promoting specialized ones. The enterprise corpus yields
  *uptime, incident, api, vendor, deployment, encryption*; the SQuAD/Wikipedia
  corpus yields *complexity, patent, computational, theory* — each corpus gets
  its own vocabulary, saved as a printable `data/domain_vocab.json`, with **zero
  hand-editing**. Weights let rare technical terms move the feature more than
  common ones (`data/domain_vocab.json`, `src/adaptive_rag/domain_vocab.py`).
  Terms occurring inside named-entity spans (the corpus's *subject* — e.g. the
  company/product name "Northwind"/"InsightHub") are excluded, since those are
  already captured by entity density; this prevents simple entity lookups like
  "Who is the CEO of Northwind Analytics?" from being spuriously escalated.
- **Syntax-aware soft intent.** Intent now reads the dependency parse, not just
  keywords: comparative morphology ("faster") and coordinated alternatives
  ("Postgres or MySQL") catch keyword-free comparisons, while "between 1990 and
  1995" is correctly rejected as non-comparison. The output is a *continuous*
  score (evidence-weighted mixture), so genuinely mixed queries land between
  anchors instead of snapping to one bucket. Regression tests lock in the
  previously-failing cases (`tests/test_features.py`).

**Effect on the benchmarks.** Quality held (enterprise F1/EM unchanged at
ceiling; SQuAD recall 0.930 → 0.935 with no loss), and the routing became
corpus-aware while staying well-calibrated: with the entity exclusion in place,
simple entity lookups route Simple again (enterprise Simple fraction 18.2%),
and the adaptive system re-ranks only 5–9% of queries — an order of magnitude
below the always-complex baseline's 100%. Both changes remain **training-free**;
they replace designer priors with corpus statistics and syntax, not learned
weights. What is still hand-set (normalization caps, membership breakpoints,
policy thresholds) is documented as remaining future work.

---

## Honest caveats (what the numbers do *not* claim)

- **On the small enterprise corpus, answer quality tied across strategies** —
  with 56 clean passages, even top-2 dense always found the gold passage, so
  adaptive's win there is purely cost/latency. The *quality* separation
  appears on the larger SQuAD corpus (Improvement 2), where shallow retrieval
  starts missing.
- **fixed_simple is still the fastest.** Adaptive pays a small classification
  overhead (~75 ms of spaCy/fuzzy work per query) plus deeper retrieval on the
  queries it escalates. The claim is *better placed* spending, not minimal
  spending.
- **Generation quality is bounded by the answering LLM** (flan-t5-base). It
  extracts short facts well but cannot synthesize multi-part comparisons, and
  end-to-end F1/EM reflect that ceiling for all strategies equally. The LLM is
  a swappable interface (`llm.py`); a stronger model changes those absolute
  numbers without touching the routing evidence above.
- Latencies are from one CPU-only Windows machine; absolute values will vary,
  but the *ratios* (docs retrieved, re-rank fraction) are hardware-independent.

---

## Reproduce every number in this document

```bash
cd "C:\Project RAG"
.\.venv\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1

# Enterprise evaluation table (Improvement 1) -> console + results/eval_results.csv
rag eval

# Per-query routing traces (Improvement 3, 4)
rag ask "Who is the CEO of Northwind Analytics?" -v
rag ask "What are the differences between the Starter, Growth, and Enterprise subscription tiers?" -v

# Behavioral guarantees (Improvement 4)
pytest

# SQuAD run (Improvement 2): uses the dedicated SQuAD config (separate
# data_squad/ + results_squad/ dirs, so the enterprise indexes are kept):
#   rag --config config/config_squad.yaml ingest
#   rag --config config/config_squad.yaml eval
```

Artifacts: `results/eval_results.csv` (latest run), `README.md` (methodology),
`sample_docs/` + `sample_docs_eval.jsonl` (corpus and gold labels),
`tests/` (36 tests).
