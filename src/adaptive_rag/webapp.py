"""Minimal web frontend for the Adaptive RAG system.

A small Flask app that wraps :class:`AdaptiveRAGPipeline` and serves a single-page
UI where a user enters a query and sees the predicted complexity class, the
fuzzy subsystem/overall scores, the chosen retrieval policy, cost metrics, the
retrieved documents, and the generated answer.

Run with::

    rag serve                       # then open http://127.0.0.1:5000
    python -m adaptive_rag.webapp   # equivalent
"""

from __future__ import annotations

import argparse

from flask import Flask, jsonify, request

from .logging_utils import configure_logging, get_logger
from .pipeline import AdaptiveRAGPipeline

logger = get_logger(__name__)

# The pipeline is heavy (models + indexes); build it once and reuse.
_pipeline: AdaptiveRAGPipeline | None = None


def get_pipeline() -> AdaptiveRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AdaptiveRAGPipeline()
    return _pipeline


def _response_to_dict(r) -> dict:
    """Serialize a RAGResponse into the JSON payload the UI consumes."""
    fv = r.features
    return {
        "query": r.query,
        "answer": r.answer,
        "complexity_score": round(r.complexity_score, 4),
        "query_class": r.policy.query_class,
        "strategy": r.policy.describe(),
        "subsystems": {
            "linguistic": round(r.subsystem_scores.linguistic, 4),
            "knowledge": round(r.subsystem_scores.knowledge, 4),
            "reasoning": round(r.subsystem_scores.reasoning, 4),
        },
        "features": {
            "query_length": {"raw": fv.query_length_raw, "norm": round(fv.query_length_norm, 3)},
            "dependency_depth": {"raw": fv.dependency_depth_raw, "norm": round(fv.dependency_depth_norm, 3)},
            "entity_density": {"raw": round(fv.entity_density_raw, 3), "norm": round(fv.entity_density_norm, 3)},
            "domain_keyword_density": {"raw": round(fv.domain_keyword_density_raw, 3), "norm": round(fv.domain_keyword_density_norm, 3)},
            "intent_type": {"raw": fv.intent_label, "norm": round(fv.intent_type_norm, 3)},
            "semantic_ambiguity": {"raw": round(fv.semantic_ambiguity_raw, 2), "norm": round(fv.semantic_ambiguity_norm, 3)},
        },
        "metrics": {
            "latency_s": round(r.latency_s, 3),
            "docs_retrieved": r.retrieval_cost,
            "reranked": r.reranked,
            "top_k": r.policy.top_k,
        },
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "source": d.source,
                "score": round(d.score, 4),
                "snippet": (d.text[:300] + ("..." if len(d.text) > 300 else "")).replace("\n", " "),
            }
            for d in r.documents
        ],
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return INDEX_HTML

    @app.post("/api/ask")
    def ask():
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Empty query."}), 400
        try:
            resp = get_pipeline().run(query)
            return jsonify(_response_to_dict(resp))
        except Exception as exc:  # pragma: no cover - surface errors to the UI
            logger.exception("Query failed")
            return jsonify({"error": str(exc)}), 500

    return app


def main(host: str = "127.0.0.1", port: int = 5000, warm: bool = True) -> None:
    configure_logging(verbose=False)
    app = create_app()
    if warm:
        logger.info("Warming up models (first load can take ~60-90s on CPU) ...")
        try:
            get_pipeline().run("warm up query")
            logger.info("Ready.")
        except Exception:
            logger.warning("Warm-up failed; first request will be slow.")
    logger.info("Serving Adaptive RAG UI at http://%s:%d", host, port)
    app.run(host=host, port=port, debug=False)


def _cli() -> None:
    p = argparse.ArgumentParser(description="Serve the Adaptive RAG web UI.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-warm", action="store_true", help="Skip model warm-up at startup")
    a = p.parse_args()
    main(host=a.host, port=a.port, warm=not a.no_warm)


# --------------------------------------------------------------------------- UI
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Adaptive RAG</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#ffffff; --ink:#1f2430; --muted:#6b7280; --line:#e5e7eb;
    --blue:#2C6FB5; --green:#2E8B57; --amber:#E8A73C; --red:#C0392B;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:900px;margin:0 auto;padding:32px 20px 64px}
  h1{font-size:22px;margin:0 0 2px;font-weight:700}
  .sub{color:var(--muted);font-size:13px;margin-bottom:22px}
  .askbar{display:flex;gap:10px}
  #q{flex:1;padding:13px 15px;border:1px solid var(--line);border-radius:10px;
     font-size:15px;background:var(--card);outline:none}
  #q:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(44,111,181,.12)}
  button{padding:13px 22px;border:0;border-radius:10px;background:var(--blue);
         color:#fff;font-size:15px;font-weight:600;cursor:pointer}
  button:disabled{opacity:.55;cursor:default}
  .examples{margin:12px 0 0;font-size:13px;color:var(--muted)}
  .examples a{color:var(--blue);text-decoration:none;cursor:pointer;margin-right:14px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
        padding:20px;margin-top:18px}
  .row{display:flex;flex-wrap:wrap;gap:14px}
  .grow{flex:1;min-width:0}
  .badge{display:inline-block;padding:5px 14px;border-radius:999px;color:#fff;
         font-weight:700;font-size:14px;letter-spacing:.3px}
  .score{font-size:40px;font-weight:800;line-height:1}
  .lab{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
  .bar{height:9px;background:#eef1f4;border-radius:999px;overflow:hidden;margin-top:6px}
  .bar>i{display:block;height:100%;border-radius:999px}
  .sub3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:6px}
  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;text-align:center}
  .metrics .v{font-size:20px;font-weight:700}
  .answer{font-size:16px;line-height:1.6;white-space:pre-wrap}
  .doc{border-top:1px solid var(--line);padding:10px 0;font-size:13.5px}
  .doc:first-child{border-top:0}
  .doc .meta{color:var(--muted);font-size:12px;margin-bottom:2px}
  details summary{cursor:pointer;color:var(--muted);font-size:13px;margin-top:4px}
  .spin{display:inline-block;width:16px;height:16px;border:2px solid #fff;
        border-top-color:transparent;border-radius:50%;animation:sp .7s linear infinite;
        vertical-align:-3px;margin-right:8px}
  @keyframes sp{to{transform:rotate(360deg)}}
  .err{color:var(--red);font-weight:600}
  .hint{color:var(--muted);font-size:12px;margin-top:4px}
</style>
</head>
<body>
<div class="wrap">
  <h1>Adaptive RAG</h1>
  <div class="sub">Fuzzy query-complexity routing &middot; enter a question to see its class, metrics, and answer</div>

  <div class="askbar">
    <input id="q" placeholder="Ask a question about the knowledge base..." autofocus/>
    <button id="go" onclick="ask()">Ask</button>
  </div>
  <div class="examples">
    Try:
    <a onclick="ex('Who is the CEO of Northwind Analytics?')">a simple query</a>
    <a onclick="ex('What is the 401(k) company match and when are employees vested?')">a medium query</a>
    <a onclick="ex('What are the encryption, authentication, and access-control requirements in the information security policy?')">a complex query</a>
  </div>

  <div id="out"></div>
</div>

<script>
const CLASS_COLOR = {Simple:"var(--green)", Medium:"var(--amber)", Complex:"var(--red)"};
function ex(t){ document.getElementById('q').value = t; ask(); }
function pct(x){ return Math.round(x*100)+'%'; }
function bar(v, color){ return `<div class="bar"><i style="width:${pct(v)};background:${color}"></i></div>`; }

async function ask(){
  const q = document.getElementById('q').value.trim();
  const out = document.getElementById('out');
  const btn = document.getElementById('go');
  if(!q){ return; }
  btn.disabled = true;
  out.innerHTML = `<div class="card"><span class="spin"></span>Analyzing query and retrieving… <span class="hint">(first query loads the model, ~60–90s)</span></div>`;
  try{
    const r = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
                                       body: JSON.stringify({query:q})});
    const d = await r.json();
    if(d.error){ out.innerHTML = `<div class="card err">Error: ${d.error}</div>`; return; }
    render(d);
  }catch(e){ out.innerHTML = `<div class="card err">Request failed: ${e}</div>`; }
  finally{ btn.disabled = false; }
}

function render(d){
  const c = CLASS_COLOR[d.query_class] || "var(--blue)";
  const s = d.subsystems, m = d.metrics, f = d.features;
  const docs = d.documents.map(x =>
    `<div class="doc"><div class="meta">#${x.doc_id} &middot; ${x.title||'untitled'} &middot; ${x.source} &middot; score ${x.score}</div>${x.snippet}</div>`
  ).join('');
  document.getElementById('out').innerHTML = `
  <div class="card">
    <div class="row" style="align-items:center">
      <div style="min-width:150px">
        <div class="lab">Query class</div>
        <div style="margin-top:6px"><span class="badge" style="background:${c}">${d.query_class}</span></div>
      </div>
      <div style="min-width:150px">
        <div class="lab">Complexity score</div>
        <div class="score" style="color:${c}">${d.complexity_score.toFixed(3)}</div>
      </div>
      <div class="grow">
        <div class="lab">Retrieval strategy</div>
        <div style="margin-top:8px;font-weight:600">${d.strategy}</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="lab">Subsystem complexity</div>
    <div class="sub3">
      <div><b>Linguistic</b> — ${s.linguistic.toFixed(3)} ${bar(s.linguistic,"var(--blue)")}</div>
      <div><b>Knowledge</b> — ${s.knowledge.toFixed(3)} ${bar(s.knowledge,"var(--blue)")}</div>
      <div><b>Reasoning</b> — ${s.reasoning.toFixed(3)} ${bar(s.reasoning,"var(--blue)")}</div>
    </div>
    <details>
      <summary>Show the six input features</summary>
      <div class="sub3" style="grid-template-columns:1fr 1fr;margin-top:10px">
        <div>Query length: <b>${f.query_length.raw}</b> (${f.query_length.norm})</div>
        <div>Dependency depth: <b>${f.dependency_depth.raw}</b> (${f.dependency_depth.norm})</div>
        <div>Entity density: <b>${f.entity_density.raw}</b> (${f.entity_density.norm})</div>
        <div>Domain-term density: <b>${f.domain_keyword_density.raw}</b> (${f.domain_keyword_density.norm})</div>
        <div>Intent: <b>${f.intent_type.raw}</b> (${f.intent_type.norm})</div>
        <div>Semantic ambiguity: <b>${f.semantic_ambiguity.raw}</b> (${f.semantic_ambiguity.norm})</div>
      </div>
    </details>
  </div>

  <div class="card">
    <div class="metrics">
      <div><div class="lab">Latency</div><div class="v">${m.latency_s.toFixed(2)}s</div></div>
      <div><div class="lab">Docs retrieved</div><div class="v">${m.docs_retrieved}</div></div>
      <div><div class="lab">Re-ranked</div><div class="v">${m.reranked ? 'Yes' : 'No'}</div></div>
      <div><div class="lab">Top-k</div><div class="v">${m.top_k}</div></div>
    </div>
  </div>

  <div class="card">
    <div class="lab">Answer</div>
    <div class="answer" style="margin-top:8px">${escapeHtml(d.answer) || '<i>(no answer)</i>'}</div>
  </div>

  <div class="card">
    <div class="lab">Retrieved documents (${d.documents.length})</div>
    <div style="margin-top:8px">${docs || '<i>none</i>'}</div>
  </div>`;
}
function escapeHtml(t){ const e=document.createElement('div'); e.textContent=t; return e.innerHTML; }
document.getElementById('q').addEventListener('keydown', e=>{ if(e.key==='Enter') ask(); });
</script>
</body>
</html>
"""


if __name__ == "__main__":  # pragma: no cover
    _cli()
