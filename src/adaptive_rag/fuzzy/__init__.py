"""Fuzzy inference engine (Modules 3-5).

Two-stage hierarchical Mamdani design:

* :mod:`adaptive_rag.fuzzy.subsystems` — three small expert subsystems mapping
  the six raw features to Linguistic / Knowledge / Reasoning complexity.
* :mod:`adaptive_rag.fuzzy.hierarchical` — a second-stage engine combining the
  three subsystem outputs into an Overall Complexity score via centroid
  defuzzification.
"""
