"""Centralized logging setup.

Every pipeline stage logs its intermediate output at ``DEBUG`` level through a
logger obtained from :func:`get_logger`. Call :func:`configure_logging` once at
program start (the CLI does this) to control verbosity.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(verbose: bool = False, level: int | None = None) -> None:
    """Configure the root ``adaptive_rag`` logger.

    Args:
        verbose: If True, set level to ``DEBUG`` so every stage's intermediate
            output is shown. Otherwise use ``INFO``.
        level: Explicit logging level; overrides ``verbose`` when provided.
    """
    global _CONFIGURED
    resolved = level if level is not None else (logging.DEBUG if verbose else logging.INFO)

    root = logging.getLogger("adaptive_rag")
    root.setLevel(resolved)

    # Avoid duplicate handlers if called more than once.
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(resolved)

    root.setLevel(resolved)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``adaptive_rag`` namespace.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if not _CONFIGURED:
        configure_logging(verbose=False)
    # Normalize "adaptive_rag.foo.bar" so children roll up to the root logger.
    short = name.split(".")[-1]
    return logging.getLogger(f"adaptive_rag.{short}")
