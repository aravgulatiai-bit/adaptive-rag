"""Adaptive RAG with a Hierarchical Fuzzy Query-Complexity Engine.

A free, fully-local, CPU-only Retrieval-Augmented Generation system that
classifies each query's complexity with a two-stage fuzzy inference engine and
adapts its retrieval strategy accordingly.

Pipeline stages (one module each):
    1. preprocessing  — spaCy parse -> ParsedQuery
    2. features       — 6-D normalized feature vector
    3. fuzzy.subsystems — three small Mamdani expert systems
    4/5. fuzzy.hierarchical — stage-2 engine + centroid defuzzification
    6. policy         — score -> retrieval configuration
    7. retrieval.*    — dense / BM25 / hybrid / cross-encoder rerank
    8. llm            — local answering model (swappable)
    pipeline          — orchestration -> RAGResponse
"""

__version__ = "0.1.0"
