"""Module 8 (part 1) — Local LLM interface.

A small, swappable answering-LLM abstraction. The default backend is a local
``google/flan-t5-base`` seq2seq model running on CPU. The :class:`BaseLLM`
interface makes it trivial to drop in Ollama or a hosted API model later without
touching the rest of the pipeline.
"""

from __future__ import annotations

import abc
import functools
from typing import List

from .logging_utils import get_logger

logger = get_logger(__name__)


# Prompt template: {query + retrieved context} -> answer.
_PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def build_prompt(question: str, contexts: List[str], max_context_chars: int = 3000) -> str:
    """Assemble the LLM prompt from a question and retrieved passages.

    Args:
        question: The user query.
        contexts: Retrieved passage texts, most relevant first.
        max_context_chars: Truncate the concatenated context to this many chars
            to fit small models' input windows.

    Returns:
        The formatted prompt string.
    """
    joined = ""
    for ctx in contexts:
        # No "[n]" markers — small models tend to echo them as the answer.
        piece = f"- {ctx.strip()}\n"
        if len(joined) + len(piece) > max_context_chars:
            break
        joined += piece
    if not joined:
        joined = "(no context retrieved)"
    return _PROMPT_TEMPLATE.format(context=joined.strip(), question=question.strip())


class BaseLLM(abc.ABC):
    """Abstract answering-LLM interface."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for a prompt.

        Args:
            prompt: The full prompt (question + context).

        Returns:
            The generated answer text.
        """
        raise NotImplementedError

    def answer(self, question: str, contexts: List[str]) -> str:
        """Convenience: build a prompt from question+contexts and generate."""
        prompt = build_prompt(question, contexts)
        logger.debug("LLM prompt (%d chars):\n%s", len(prompt), prompt)
        out = self.generate(prompt)
        logger.debug("LLM answer: %s", out)
        return out


class FlanT5LLM(BaseLLM):
    """Local ``flan-t5`` seq2seq answering model (CPU)."""

    def __init__(self, model_name: str = "google/flan-t5-base", max_new_tokens: int = 128) -> None:
        """Initialize the model lazily on first use.

        Args:
            model_name: Hugging Face model id.
            max_new_tokens: Maximum tokens to generate per answer.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        logger.debug("Loading LLM '%s' on CPU ...", self.model_name)
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tok,
            device=-1,  # CPU
            torch_dtype=torch.float32,
        )

    def generate(self, prompt: str) -> str:
        self._ensure()
        out = self._pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            truncation=True,
        )
        return out[0]["generated_text"].strip()


class OllamaLLM(BaseLLM):  # pragma: no cover - optional backend
    """Example swap-in backend for a locally-running Ollama server.

    Not exercised by the default demo (kept dependency-free), but shows exactly
    where a different LLM plugs in. Requires the ``requests`` package and a
    running Ollama daemon.
    """

    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host

    def generate(self, prompt: str) -> str:
        import requests  # local import so it's optional

        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


@functools.lru_cache(maxsize=2)
def get_llm(backend: str, model_name: str, max_new_tokens: int) -> BaseLLM:
    """Factory returning a cached LLM instance for the configured backend.

    Args:
        backend: Backend identifier (``"flan-t5"`` or ``"ollama"``).
        model_name: Model id passed to the backend.
        max_new_tokens: Generation length cap (used by flan-t5).

    Returns:
        A :class:`BaseLLM` implementation.

    Raises:
        ValueError: If the backend is unknown.
    """
    backend = backend.lower()
    if backend in ("flan-t5", "flant5", "flan"):
        return FlanT5LLM(model_name=model_name, max_new_tokens=max_new_tokens)
    if backend == "ollama":
        return OllamaLLM(model=model_name)
    raise ValueError(f"Unknown LLM backend: {backend!r}")
