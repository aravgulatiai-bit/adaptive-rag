"""Adaptive RAG - guided review demo.

Double-click ``run_demo.bat`` (or run ``python demo.py``) and press Enter to move
through each step.  Models are loaded once and shared by every step, including
the web UI, so nothing has to be reloaded during the review.

    python demo.py            # interactive menu
    python demo.py --auto     # run the guided demo without pauses
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.system("")  # enable ANSI colours in the Windows console

# ----------------------------------------------------------------- styling
B, DIM, R = "\033[1m", "\033[2m", "\033[0m"
CYAN, GREEN, YELLOW, RED, MAG = "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[35m"
CLASS_COL = {"Simple": GREEN, "Medium": YELLOW, "Complex": RED}
W = 78
AUTO = False

DEMO_QUERIES = [
    ("Who is the CEO of Northwind Analytics?",
     "A short factual look-up: few words, one entity, no reasoning."),
    ("What is the 401(k) company match and when are employees vested?",
     "Two linked facts from one policy: moderate length and domain terms."),
    ("What are the encryption, authentication, and access-control requirements "
     "in the information security policy?",
     "Several requirements to gather and list: long, term-dense, enumerative."),
]


def rule(title: str = "", ch: str = "─") -> None:
    if title:
        print(f"\n{CYAN}{B}{ch * 3} {title} {ch * max(3, W - len(title) - 5)}{R}")
    else:
        print(f"{DIM}{ch * W}{R}")


def banner(title: str, sub: str = "") -> None:
    print(f"\n{CYAN}{B}╔{'═' * (W - 2)}╗")
    print(f"║{title.center(W - 2)}║")
    if sub:
        print(f"║{sub.center(W - 2)}║")
    print(f"╚{'═' * (W - 2)}╝{R}")


def pause(msg: str = "Press Enter to continue") -> None:
    if AUTO:
        print()
        return
    try:
        input(f"\n{DIM}  ▶ {msg} ...{R}")
    except EOFError:
        pass


def bar(x: float, width: int = 24) -> str:
    n = int(round(max(0.0, min(1.0, x)) * width))
    return "█" * n + "░" * (width - n)


def level(x: float) -> str:
    return "Low" if x < 0.35 else ("High" if x >= 0.70 else "Medium")


# ----------------------------------------------------------------- pipeline
_PIPE = None


def pipeline():
    global _PIPE
    if _PIPE is None:
        rule("Loading the system")
        print("  Loading spaCy, the embedding model, the cross-encoder, the indexes")
        print("  and the Flan-T5 answering model. This takes about a minute on first run.")
        t0 = time.perf_counter()
        from adaptive_rag.logging_utils import configure_logging
        from adaptive_rag.pipeline import AdaptiveRAGPipeline

        configure_logging(verbose=False)
        import logging

        for name in ("", "adaptive_rag", "transformers", "sentence_transformers"):
            logging.getLogger(name).setLevel(logging.WARNING)
        _PIPE = AdaptiveRAGPipeline()
        print(f"  Indexes and retrievers loaded in {time.perf_counter() - t0:.1f}s. Warming up all three tiers ...")
        t1 = time.perf_counter()
        _PIPE.run("warm up query")                       # loads the LLM
        _PIPE.retrieve("warm up query", _PIPE.selector.select(0.9))  # loads the cross-encoder path
        print(f"  {GREEN}Ready{R} (warm-up {time.perf_counter() - t1:.1f}s). "
              f"Timings below exclude model loading.")
    return _PIPE


# ----------------------------------------------------------------- steps
def step_overview() -> None:
    banner("ADAPTIVE RAG WITH DYNAMIC RETRIEVAL STRATEGIES",
           "for Enterprise Knowledge Systems  -  Review Demo")
    print(f"""
  {B}Problem{R}   Normal RAG retrieves the same way for every question. Easy questions
            pay for expensive retrieval; hard ones may still get too little.

  {B}Idea{R}      Measure how complex a question is {B}before{R} retrieving, then choose
            how much retrieval effort to spend on it.

  {B}Pipeline{R}  Query → 6 features → 3 fuzzy subsystems (Mamdani, 9 rules each)
            → hierarchical fusion → complexity score → retrieval tier → answer

  {B}Tiers{R}     {GREEN}Simple {R} score < 0.35   top-2 dense retrieval
            {YELLOW}Medium {R} 0.35 – 0.70    top-5 hybrid (BM25 + dense, RRF fusion)
            {RED}Complex{R} score ≥ 0.70   top-10 hybrid + cross-encoder re-ranking""")
    meta = ROOT / "data" / "index_meta.json"
    docs = sorted((ROOT / "sample_docs").glob("*.md"))
    passages = sum(1 for _ in open(ROOT / "data" / "corpus.jsonl", encoding="utf-8"))
    vocab = len(json.load(open(ROOT / "data" / "domain_vocab.json", encoding="utf-8")))
    rule("Knowledge base: Northwind Analytics (fictional enterprise)")
    print(f"  {len(docs)} policy and product documents → {passages} passages indexed, "
          f"{vocab} corpus-derived domain terms")
    for i in range(0, len(docs), 2):
        a = docs[i].stem
        b = docs[i + 1].stem if i + 1 < len(docs) else ""
        print(f"   {DIM}•{R} {a:<36} {DIM}•{R} {b}")
    _ = meta


def show_query(q: str, note: str = "", answer: bool = True):
    p = pipeline()
    t0 = time.perf_counter()
    parsed, fv, sub, score, policy = p.analyze(q)
    t_route = time.perf_counter() - t0
    docs, reranked = p.retrieve(q, policy)
    t_ret = time.perf_counter() - t0 - t_route
    ans = p.answer_from_docs(q, docs) if answer else ""
    total = time.perf_counter() - t0
    col = CLASS_COL.get(policy.query_class, "")

    rule("Query")
    print(f"  {B}“{q}”{R}")
    if note:
        print(f"  {DIM}{note}{R}")

    rule("Step 1 · Six features (raw → normalised to 0–1)")
    rows = [
        ("Query length", f"{fv.query_length_raw} words", fv.query_length_norm),
        ("Dependency depth", f"{fv.dependency_depth_raw}", fv.dependency_depth_norm),
        ("Entity density", f"{fv.entity_density_raw:.3f}", fv.entity_density_norm),
        ("Domain-term density", f"{fv.domain_keyword_density_raw:.3f}", fv.domain_keyword_density_norm),
        ("Intent", f"{fv.intent_label}", fv.intent_type_norm),
        ("Semantic ambiguity", f"{fv.semantic_ambiguity_raw:.2f}", fv.semantic_ambiguity_norm),
    ]
    for name, raw, norm in rows:
        print(f"  {name:<20} {raw:>12}   {bar(norm)} {norm:.3f}  {DIM}{level(norm)}{R}")

    rule("Step 2 · Fuzzy subsystems (Mamdani, 9 rules each)")
    for name, val, ins in (("Linguistic", sub.linguistic, "length + depth"),
                           ("Knowledge", sub.knowledge, "entities + domain terms"),
                           ("Reasoning", sub.reasoning, "intent + ambiguity")):
        print(f"  {name:<11} {bar(val)} {val:.3f}  {level(val):<6} {DIM}from {ins}{R}")

    rule("Step 3 · Hierarchical fusion → complexity score → retrieval tier")
    print(f"  Complexity score  {bar(score, 40)} {B}{score:.3f}{R}")
    print(f"  {DIM}                  0 ─────── 0.35 ─────────── 0.70 ─────── 1{R}")
    print(f"  Query class       {col}{B}{policy.query_class.upper()}{R}")
    print(f"  Strategy          {policy.describe()}")
    print(f"  Re-ranking ran    {'yes' if reranked else 'no'}")

    rule(f"Step 4 · Retrieved passages ({len(docs)})")
    for i, d in enumerate(docs[:5], 1):
        src = (getattr(d, "title", "") or f"passage {d.doc_id}").replace(".md", "")
        snippet = " ".join(d.text.split())[:88]
        print(f"  [{i}] {DIM}{src[:34]:<34}{R} {snippet}…")
    if len(docs) > 5:
        print(f"  {DIM}… and {len(docs) - 5} more (only the top 3 are sent to the answering model){R}")

    if answer:
        rule("Step 5 · Answer")
        print(f"  {GREEN}{B}{ans}{R}")
    print(f"\n  {DIM}Routing {t_route * 1000:.0f} ms · retrieval {t_ret * 1000:.0f} ms · "
          f"total {total:.2f} s{R}")
    return {"q": q, "cls": policy.query_class, "score": score, "docs": len(docs),
            "rerank": reranked, "lat": total, "ans": ans}


def step_queries() -> list:
    out = []
    for i, (q, note) in enumerate(DEMO_QUERIES, 1):
        banner(f"LIVE QUERY {i} of {len(DEMO_QUERIES)}")
        out.append(show_query(q, note))
        pause("Press Enter for the next step")
    return out


def step_compare(res: list) -> None:
    banner("SAME SYSTEM, THREE DIFFERENT RETRIEVAL STRATEGIES")
    print(f"\n  {'Query':<44}{'Class':<9}{'Score':>6}{'Docs':>6}{'Rerank':>8}{'Time':>7}")
    rule()
    for r in res:
        q = r["q"] if len(r["q"]) <= 42 else r["q"][:41] + "…"
        col = CLASS_COL.get(r["cls"], "")
        print(f"  {q:<44}{col}{r['cls']:<9}{R}{r['score']:>6.3f}{r['docs']:>6}"
              f"{('yes' if r['rerank'] else 'no'):>8}{r['lat']:>6.2f}s")
    print(f"\n  The router itself costs only milliseconds; it decides how many seconds of")
    print(f"  retrieval and re-ranking each question is worth.")


def _read_csv(path: Path):
    rows, dist = [], []
    with open(path, encoding="utf-8") as f:
        part = rows
        for row in csv.reader(f):
            if not row:
                part = dist
                continue
            part.append(row)
    return rows, dist


def step_results() -> None:
    banner("EVALUATION RESULTS", "adaptive routing vs. always-cheap and always-expensive baselines")
    for title, path in (("Enterprise-22 (our Northwind corpus, 22 gold questions)", ROOT / "results" / "eval_results.csv"),
                        ("SQuAD-200 (public benchmark, 200 questions)", ROOT / "results_squad" / "eval_results.csv")):
        if not path.exists():
            continue
        rows, dist = _read_csv(path)
        rule(title)
        print(f"  {'Strategy':<16}{'Recall@k':>9}{'F1':>8}{'EM':>8}{'Latency':>10}{'Docs':>7}{'Rerank':>8}")
        for r in rows[1:]:
            name = r[0]
            hl = B if name == "adaptive" else ""
            print(f"  {hl}{name:<16}{float(r[2]):>9.3f}{float(r[3]):>8.3f}{float(r[4]):>8.3f}"
                  f"{float(r[5]):>9.2f}s{float(r[6]):>7.2f}{float(r[7]):>7.1f}%{R}")
        if len(dist) > 1:
            total = sum(int(c) for _, c in dist[1:])
            print("  Adaptive routing: " + "   ".join(
                f"{CLASS_COL.get(k, '')}{k}{R} {int(c) * 100 / total:.0f}%" for k, c in dist[1:]))
    print(f"""
  {B}Takeaways{R}
   • SQuAD: recall 0.755 → 0.935 (75% of the gap to the 0.995 ceiling) while
     re-ranking only 5% of queries, at about one third of the expensive latency.
   • Enterprise: same F1/EM as the expensive pipeline with half the documents
     and re-ranking on 9% instead of 100% of queries.""")


def step_tests() -> None:
    banner("AUTOMATED TEST SUITE")
    print("  Running pytest (features, fuzzy engine, policy, end-to-end) ...\n")
    sys.stdout.flush()
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=", "-v", "-p", "no:cacheprovider", "--tb=short", "--color=no"],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    passed = failed = 0
    for line in proc.stdout.splitlines():
        if " PASSED" in line or " FAILED" in line or " ERROR" in line:
            name = line.split(" ")[0].replace("tests/", "")
            ok = " PASSED" in line
            passed += ok
            failed += not ok
            print(f"  {GREEN + '✔' if ok else RED + '✘'}{R} {DIM}{name}{R}")
    col = GREEN if failed == 0 and passed else RED
    print(f"\n  {col}{B}{passed} passed, {failed} failed{R} in {time.perf_counter() - t0:.1f}s")
    if failed or not passed:
        print(proc.stdout[-3000:] + proc.stderr[-2000:])


def step_ask() -> None:
    banner("ASK YOUR OWN QUESTION", "type a question about Northwind Analytics; blank line to return")
    print(f"  {DIM}Ideas: What is the Recovery Time Objective for InsightHub?")
    print("         Within how many days must Northwind fulfill a GDPR data subject request?")
    print(f"         How often must employees rotate their passwords?{R}")
    while True:
        try:
            q = input(f"\n  {B}Question>{R} ").strip()
        except EOFError:
            return
        if not q:
            return
        try:
            show_query(q)
        except Exception as exc:  # keep the demo alive whatever happens
            print(f"  {RED}Error: {exc}{R}")


def step_web(block: bool = True) -> None:
    banner("WEB INTERFACE")
    p = pipeline()
    from adaptive_rag import webapp

    webapp._pipeline = p          # reuse the already-loaded models: no second load
    app = webapp.create_app()
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    url = "http://127.0.0.1:5000"
    import socket

    with socket.socket() as sock:
        busy = sock.connect_ex(("127.0.0.1", 5000)) == 0
    if busy:
        print(f"  A web UI is already running at {B}{url}{R} - opening it.")
        webbrowser.open(url)
        return
    th = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False),
                          daemon=True)
    th.start()
    time.sleep(1.5)
    print(f"  Serving at {B}{url}{R} - opening your browser.")
    print("  Click the example links (simple / medium / complex) or type any question.")
    webbrowser.open(url)
    if block:
        try:
            input(f"\n{DIM}  ▶ Press Enter to stop the web UI and exit ...{R}")
        except (EOFError, KeyboardInterrupt):
            pass


def guided() -> None:
    step_overview()
    pause("Press Enter to load the models")
    pipeline()
    pause("Press Enter to run the first live query")
    res = step_queries()
    step_compare(res)
    pause("Press Enter to see the evaluation results")
    step_results()
    if AUTO:
        return
    pause("Press Enter to ask your own questions (blank line skips)")
    step_ask()
    step_web()


def menu() -> None:
    step_overview()
    while True:
        print(f"""
{CYAN}{B}  DEMO MENU{R}
   {B}1{R}  Guided demo (recommended for the review)
   {B}2{R}  Ask your own questions (full trace for each)
   {B}3{R}  Show evaluation results
   {B}4{R}  Run the test suite
   {B}5{R}  Open the web interface
   {B}Q{R}  Quit""")
        try:
            c = input(f"  {B}Choose [1-5, Q]>{R} ").strip().lower()
        except EOFError:
            return
        if c in ("1", ""):
            guided()
            return
        if c == "2":
            step_ask()
        elif c == "3":
            step_results()
        elif c == "4":
            step_tests()
        elif c == "5":
            step_web()
            return
        elif c in ("q", "quit", "exit"):
            return


def main() -> None:
    global AUTO
    ap = argparse.ArgumentParser(description="Adaptive RAG review demo")
    ap.add_argument("--auto", action="store_true", help="run the guided demo without pauses")
    ap.add_argument("--guided", action="store_true", help="skip the menu and start the guided demo")
    a = ap.parse_args()
    AUTO = a.auto
    try:
        if a.auto or a.guided:
            guided()
        else:
            menu()
    except KeyboardInterrupt:
        pass
    print(f"\n{DIM}  Demo finished.{R}")


if __name__ == "__main__":
    main()
