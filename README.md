# Adaptive RAG with a Hierarchical Fuzzy Query-Complexity Engine

A fully **free, local, CPU-only** Retrieval-Augmented Generation system that
**classifies each query's complexity with a two-stage fuzzy inference engine and
adapts its retrieval strategy accordingly** — instead of using one fixed
retrieval configuration for every query.

Simple questions get cheap dense retrieval; complex, multi-entity, reasoning-heavy
questions get hybrid retrieval plus cross-encoder re-ranking. The complexity
decision is made **before any documents are retrieved**, from lightweight
linguistic/knowledge/reasoning features of the query alone.

No paid APIs, no API keys, no GPU required.

---

## How it works

```
                       ┌─────────────────────────────────────────────────────────┐
   raw query  ─────▶   │  1. Preprocess (spaCy)   →  ParsedQuery                   │
                       │  2. Feature extraction   →  6-D normalized vector         │
                       │  3. Three fuzzy experts  →  Linguistic / Knowledge /      │
                       │       (Mamdani, 9 rules)     Reasoning complexity          │
                       │  4. Hierarchical fuzzy   →  Overall complexity (fuzzy)    │
                       │  5. Defuzzify (centroid) →  score ∈ [0,1]  e.g. 0.81      │
                       │  6. Policy selector      →  Simple / Medium / Complex     │
                       └─────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │  7. Adaptive retrieval (whichever the policy dictates):   │
                       │       Simple : Top-2 dense, no re-rank                     │
                       │       Medium : Top-5 hybrid (BM25 + dense RRF)            │
                       │       Complex: Top-10 hybrid + cross-encoder re-rank      │
                       │  8. LLM answer (local flan-t5-base) → structured response │
                       └─────────────────────────────────────────────────────────┘
```

### The six features (Module 2)

Extracted from the parsed query only — **no retrieval happens here**. Each is
normalized to `[0, 1]` by dividing by a documented cap (in `config/config.yaml`)
and clipping.

| Category   | Feature                  | Raw definition                                   | Cap |
|------------|--------------------------|--------------------------------------------------|-----|
| Linguistic | Query length             | non-punctuation token count                      | 30  |
| Linguistic | Dependency depth         | max depth of the dependency tree                 | 10  |
| Knowledge  | Entity density           | named entities / tokens                          | 0.5 |
| Knowledge  | Domain term density      | weighted domain mass / tokens                    | 0.2 |
| Reasoning  | Intent type              | soft mixture over {definition, procedure, reasoning, comparison} | — |
| Reasoning  | Semantic ambiguity       | mean WordNet polysemy of content words           | 10  |

**Domain term density (corpus-derived).** Rather than a hand-typed keyword list,
a weighted vocabulary is built from the corpus at ingest time: TF-IDF keyword
extraction, then a *contrastive* re-weighting by each term's rarity in general
English (via `wordfreq`), so corpus-frequent-but-generic words ("day", "company")
are suppressed and specialized terms ("encryption", "uptime", "biometric") rise.
It's saved to `data/domain_vocab.json` (a printable ranked list), adapts to any
corpus with no manual editing, and weights rare terms more than common ones. The
hand-authored `domain_keywords` in config are merged in as a supplement/fallback.
Toggle with `features.use_corpus_domain_vocab`.

**Intent (syntax-aware, soft).** Intent uses lexical cues *and* the dependency
parse — comparative morphology ("faster") and coordinated nouns ("Postgres or
MySQL") catch comparisons with no keyword, while "between 1990 and 1995" is
correctly *not* a comparison. The result is a **continuous** score (an
evidence-weighted mixture over `definition=0.10, procedure=0.40, reasoning=0.70,
comparison=0.95`), so a single-intent query hits its anchor exactly and a mixed
query lands between anchors. Both mechanisms stay rule-based and training-free.
Toggle with `features.intent_syntax_aware` (false = legacy regex-only).

### The fuzzy engine (Modules 3–5)

Rather than one monolithic system with hundreds of rules, the design is
**hierarchical**:

- **Stage 1 — three small expert subsystems** (scikit-fuzzy, Mamdani), each with
  two inputs and a complete **9-rule** base:
  - *Linguistic*: query length + dependency depth
  - *Knowledge*: entity density + domain keyword density
  - *Reasoning*: intent type + semantic ambiguity
- **Stage 2 — a hierarchical engine** combines the three subsystem outputs into
  **Overall Complexity**. Because it operates on three *summarized* inputs, it
  needs only a compact rule base. All 27 level-combinations are enumerated and
  mapped to an output level by a transparent, monotonic level-sum rule
  (Low=0, Medium=1, High=2): sum ≤ 1 → Low, sum 2–3 → Medium, sum ≥ 4 → High.
  This is **complete by construction** (no input combination yields an empty
  output) and is compressed into three disjunctive rules.

All fuzzy variables share Low/Medium/High membership functions on `[0, 1]`:

- Low: trapezoid `(0, 0, 0.2, 0.4)`
- Medium: triangle `(0.25, 0.5, 0.75)`
- High: trapezoid `(0.6, 0.8, 1, 1)`

Overlapping supports guarantee every crisp input has nonzero membership.
**Defuzzification** uses the centroid method to produce a continuous score.

### Retrieval policy (Module 6)

Thresholds and top-k values live in `config/config.yaml`, never hardcoded:

| Complexity score | Class   | Strategy                                            |
|------------------|---------|-----------------------------------------------------|
| 0.00 – 0.35      | Simple  | Top-2 dense, no re-ranking                          |
| 0.35 – 0.70      | Medium  | Top-5 hybrid (BM25 + dense)                         |
| 0.70 – 1.00      | Complex | Top-10 hybrid + cross-encoder re-ranking            |

### Hybrid fusion: why RRF

Dense cosine scores (`[-1, 1]`) and BM25 scores (unbounded, corpus-dependent)
are not directly comparable, and min-max normalization is brittle. **Reciprocal
Rank Fusion** fuses by rank instead of raw score:

```
rrf_score(d) = Σ_retrievers  1 / (k + rank_r(d))      # k = 60
```

It is scale-free, needs no per-retriever weight tuning, is robust to score
outliers, and is a strong, widely-used default for combining heterogeneous
rankers.

### Swappable LLM

`llm.py` defines a `BaseLLM` interface. The default `FlanT5LLM`
(`google/flan-t5-base`, CPU) can be swapped for `OllamaLLM` (stub included) or a
hosted API model by changing `models.llm_backend` in the config — no other code
changes required.

---

## Setup

Requires **Python 3.10+**. Everything runs on CPU.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Download the spaCy model and WordNet corpus (one time)
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# 4. Build the corpus + indexes (one time)
rag ingest
```

First run downloads models (~1.4 GB total: torch CPU, flan-t5-base,
two MiniLM models, spaCy, WordNet, SQuAD). Everything is cached and runs
**offline** afterwards.

> **Tip — faster repeated runs / restricted networks.** Once the models are
> cached, export `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before running.
> This skips Hugging Face's network reachability check, which can otherwise add a
> long stall (or hang on networks that block outbound HTTPS):
>
> ```bash
> export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # Windows: set them via $env:
> ```

---

## The corpus / knowledge base

By default the system builds its knowledge base from a **sample enterprise
document set** — 20 Markdown files under [`sample_docs/`](sample_docs) for a
fictional company, *Northwind Analytics*: company overview, data-retention and
privacy policies (GDPR/CCPA), information security, access control, incident
response, HR policies, product/API/engineering docs, and the support SLA.
`rag ingest` chunks these into passages and builds the dense + BM25 indexes in
`data/`.

This is controlled by the `dataset:` section of `config/config.yaml`:

```yaml
dataset:
  source: folder            # "folder" (local docs) or "squad" (HF download)
  docs_dir: sample_docs     # folder of .md/.txt files to index
  eval_questions_file: sample_docs_eval.jsonl
  chunk_max_chars: 700      # target passage size
```

### Use your own documents

1. Drop your `.md` or `.txt` files into `sample_docs/` (or point `docs_dir` at
   any folder).
2. (Optional) add eval questions to `sample_docs_eval.jsonl` — one JSON object
   per line with `question`, `answers`, and a `gold_hint` (a distinctive phrase
   from the passage that should be retrieved). Ingestion resolves each hint to
   its passage automatically, so `rag eval` keeps working across re-chunking.
3. Re-run `rag ingest`.

To fall back to the public SQuAD demo instead, set `source: squad`.

---

## Review demo (one click)

Double-click `run_demo.bat` (or the **Adaptive RAG Demo** shortcut on the Desktop).
A menu opens; choose **1** for the guided demo and press Enter to move between steps:

1. Project overview and the 20-document knowledge base.
2. Model loading and warm-up (about a minute on first run).
3. Three live queries (Simple, Medium, Complex) showing the six features, the three fuzzy
   subsystem scores, the complexity score, the chosen retrieval tier, the retrieved passages and the answer.
4. A side-by-side comparison of the three queries.
5. Saved evaluation results (Enterprise-22 and SQuAD-200).
6. Your own questions, then the web UI opens in the browser using the already-loaded models.

The menu also runs the 45-test suite (option 4). `python demo.py --auto` runs the guided demo without pauses.

## Usage

### Ask a question (with verbose trace)

```bash
rag ask "What is GDPR?" -v
```

`-v` prints the full trace: features → subsystem scores → overall score → policy
→ retrieved documents → answer. Without `-v` you get a concise answer plus its
complexity/class/strategy.

```bash
rag ask "Compare the data-retention obligations under GDPR and CCPA for cloud providers processing biometric data" -v
```

### Run the evaluation harness

```bash
rag eval                 # full question set, with answer generation
rag eval --limit 50      # quick run on 50 questions
rag eval --no-generate   # retrieval metrics only (fast; skips the LLM)
```

Compares **adaptive** vs. **fixed-simple** (always top-2 dense) vs.
**fixed-complex** (always top-10 hybrid + re-rank), reporting recall@k, token-F1,
EM, mean latency, mean docs retrieved, and re-rank usage, plus the adaptive
complexity-score distribution and query-class counts. Results are printed and
written to `results/eval_results.csv`.

### Web UI

A minimal browser frontend for entering a query and viewing its complexity
class, subsystem/overall scores, retrieval policy, cost metrics, retrieved
documents, and the answer:

```bash
rag serve                       # then open http://127.0.0.1:5000
rag serve --port 8000           # custom port
```

The server loads the models once at startup (first launch warms up in
~60–90 s on CPU) and then answers queries interactively. Requires Flask
(`pip install flask`, included in `requirements.txt`).

---

## Example results

A sample `rag eval` run on the enterprise corpus (20 documents → 56 passages),
22 questions, CPU, `flan-t5-base`:

| Strategy       | Recall@k | F1    | EM    | Latency (s) | Mean docs | Re-rank % |
|----------------|---------:|------:|------:|------------:|----------:|----------:|
| **adaptive**   |    1.000 | 0.917 | 0.773 |       0.466 |      4.91 |       9.1 |
| fixed_simple   |    1.000 | 0.917 | 0.773 |       0.266 |      2.00 |       0.0 |
| fixed_complex  |    1.000 | 0.917 | 0.773 |       0.752 |     10.00 |     100.0 |

Adaptive query-class distribution: Simple 18.2%, Medium 72.7%, Complex 9.1%
(complexity score min 0.156, median 0.500, mean 0.466, max 0.820).

**How to read this.** On this small factual-lookup workload every strategy retrieves
the gold passage (recall 1.000) and, thanks to retrieve-then-focus, reaches the same
answer quality (F1 0.917, EM 0.773). The meaningful difference is therefore **cost**:
the **adaptive** system matches the always-complex baseline's quality while running at
**~1.6× lower latency (0.47 s vs 0.75 s), retrieving 4.91 vs 10 documents, and
invoking the expensive cross-encoder on only 9.1% of queries instead of 100%**. It
gets complex-tier answers at near-simple-tier cost by escalating *only where the
fuzzy engine judges it necessary*. On a larger, noisier corpus the quality gap would
widen too, because always-simple's shallow top-2 retrieval starts missing gold
passages that the adaptive and complex strategies still recover.

Two design choices make this work well with a small model:

- **Retrieve-then-focus** — retrieval fetches the policy's full top-k for recall,
  but only the top `models.llm_context_top_n` (default 3) best-ranked passages are
  placed in the prompt. Small models degrade when fed many passages; earlier runs
  that dumped all 10 into the prompt collapsed answer quality to F1 ≈ 0.03. Set
  `llm_context_top_n` high to feed everything instead.
- **Clean chunks** — Markdown heading markers are stripped during ingestion so the
  model reads prose, not `#` symbols.

`flan-t5-base` is still a small model, so genuinely synthetic questions (e.g. an
open-ended GDPR-vs-CCPA *comparison*) produce weaker prose than short factual
lookups. Because the LLM is a swappable interface, dropping in a stronger local
model via Ollama improves those without touching the rest of the pipeline.

## Project layout

```
config/config.yaml            All tunables: thresholds, top-k, models, caps, paths
src/adaptive_rag/
  config.py                   Typed config loader
  logging_utils.py            DEBUG-level stage logging
  preprocessing.py            Module 1 — spaCy parse → ParsedQuery
  features.py                 Module 2 — 6-D normalized feature vector
  fuzzy/
    membership.py             Shared Low/Medium/High membership functions
    subsystems.py             Module 3 — three 9-rule Mamdani subsystems
    hierarchical.py           Modules 4+5 — stage-2 engine + centroid defuzz
  policy.py                   Module 6 — score → retrieval policy
  retrieval/
    indexer.py                Build/save/load dense + BM25 indexes
    dense.py, bm25.py         Dense (cosine) and sparse (BM25) retrieval
    hybrid.py                 Module 7 — RRF fusion
    rerank.py                 Cross-encoder re-ranking
  llm.py                      Module 8 — swappable local LLM
  pipeline.py                 Orchestration → RAGResponse
  ingest.py                   Dataset download + index build
  evaluate.py                 Evaluation harness
  cli.py                      `rag ask` / `rag eval` / `rag ingest`
tests/                        pytest suite
data/                         Built by `rag ingest` (corpus + indexes)
results/                      Evaluation CSV output
```

---

## Tests

```bash
pytest                       # full suite
pytest tests/test_fuzzy.py   # just the fuzzy engine
```

The suite covers feature extraction on hand-crafted queries, each fuzzy
subsystem's boundary behavior, hierarchical rule-base completeness, monotonicity
sanity checks (a long entity-dense comparison query scores higher than
"What is X?"), policy-selector thresholds, and an end-to-end smoke test on a
small in-memory corpus.

---

## Configuration

Everything policy-relevant is in `config/config.yaml`: complexity thresholds and
per-class top-k, model names, the LLM backend, feature normalization caps, the
domain-keyword list, the RRF constant, and dataset sizes. Point the CLI at a
different file with `--config path/to/config.yaml`, or set the
`ADAPTIVE_RAG_CONFIG` environment variable.

---

## Notes

- `google/flan-t5-base` is a small model — absolute answer-quality numbers are
  modest. The evaluation harness compares strategies **relatively**, which is
  what demonstrates the adaptive system's cost/quality trade-off.
- To use a stronger local model, run Ollama and set `llm_backend: ollama` and
  `llm_model: llama3` (or similar) in the config.
```
