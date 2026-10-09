"""Command-line interface: ``rag ask`` / ``rag eval`` / ``rag ingest``.

Examples:
    rag ingest
    rag ask "What is the capital of France?" -v
    rag eval --limit 50
    rag eval --no-generate      # retrieval metrics only (fast)
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import load_config
from .logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def _cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest import build

    cfg = load_config(args.config)
    build(cfg)
    print("Ingestion complete.")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from .pipeline import AdaptiveRAGPipeline

    cfg = load_config(args.config)
    pipeline = AdaptiveRAGPipeline(cfg)
    response = pipeline.run(args.question, generate=not args.no_generate)

    if args.verbose:
        print(response.trace_str())
    else:
        print(f"\nQ: {response.query}")
        print(f"Complexity: {response.complexity_score:.3f} ({response.policy.query_class})")
        print(f"Strategy: {response.policy.describe()}")
        print(f"\nA: {response.answer}\n")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import run_evaluation

    cfg = load_config(args.config)
    report = run_evaluation(cfg, limit=args.limit, generate=not args.no_generate)
    print("\n" + report + "\n")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .webapp import main as serve_main

    serve_main(host=args.host, port=args.port, warm=not args.no_warm)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    # Common flags usable either before OR after the subcommand
    # (e.g. `rag -v ask "..."` and `rag ask "..." -v` both work).
    # default=SUPPRESS is essential: the subparser re-parses these options, and
    # a plain default would overwrite a value given before the subcommand
    # (e.g. `rag --config X ask ...` would silently fall back to the default).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Path to config YAML (default: config/config.yaml)",
    )
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable DEBUG logging for all stages",
    )

    parser = argparse.ArgumentParser(
        prog="rag",
        description="Adaptive RAG with a Hierarchical Fuzzy Query-Complexity Engine.",
        parents=[common],
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", parents=[common], help="Download dataset and build corpus + indexes")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", parents=[common], help="Answer a question with a verbose complexity trace")
    p_ask.add_argument("question", help="The question to answer")
    p_ask.add_argument("--no-generate", action="store_true", help="Skip LLM generation (retrieval only)")
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("eval", parents=[common], help="Run the evaluation harness (adaptive vs. baselines)")
    p_eval.add_argument("--limit", type=int, default=None, help="Limit number of eval questions")
    p_eval.add_argument("--no-generate", action="store_true", help="Retrieval metrics only (fast)")
    p_eval.set_defaults(func=_cmd_eval)

    p_serve = sub.add_parser("serve", parents=[common], help="Launch the web UI (browser frontend)")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=5000, help="Port (default 5000)")
    p_serve.add_argument("--no-warm", action="store_true", help="Skip model warm-up at startup")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    # Ensure UTF-8 console output (the verbose trace uses box-drawing glyphs;
    # Windows consoles default to cp1252 and would raise UnicodeEncodeError).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - stream not reconfigurable
                pass

    parser = build_parser()
    args = parser.parse_args(argv)
    # Fill in the SUPPRESS-defaulted common flags if given at neither position.
    args.config = getattr(args, "config", None)
    args.verbose = getattr(args, "verbose", False)
    configure_logging(verbose=args.verbose)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
