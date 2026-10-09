"""Shared membership-function definitions for the fuzzy engine.

Every fuzzy variable lives on the universe ``[0, 1]`` (step 0.01) and uses the
same three overlapping Low/Medium/High membership functions. The overlapping
supports guarantee that *any* crisp input in ``[0, 1]`` has nonzero membership
in at least one term, which — together with complete rule bases — ensures no
input combination produces an empty (undefined) output.

Breakpoints (documented, tunable):

* Low    : trapezoid (0.00, 0.00, 0.20, 0.40)
* Medium : triangle  (0.25, 0.50, 0.75)
* High   : trapezoid (0.60, 0.80, 1.00, 1.00)
"""

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Shared universe for all fuzzy variables.
UNIVERSE = np.arange(0.0, 1.001, 0.01)

LOW_PARAMS = (0.0, 0.0, 0.2, 0.4)
MEDIUM_PARAMS = (0.25, 0.5, 0.75)
HIGH_PARAMS = (0.6, 0.8, 1.0, 1.0)

# Canonical level names used everywhere in the rule bases.
LEVELS = ("low", "medium", "high")


def add_lmh_terms(var: "ctrl.Antecedent | ctrl.Consequent") -> None:
    """Attach standard Low/Medium/High terms to a fuzzy variable.

    Args:
        var: A scikit-fuzzy Antecedent or Consequent defined on :data:`UNIVERSE`.
    """
    var["low"] = fuzz.trapmf(var.universe, LOW_PARAMS)
    var["medium"] = fuzz.trimf(var.universe, MEDIUM_PARAMS)
    var["high"] = fuzz.trapmf(var.universe, HIGH_PARAMS)


def make_antecedent(name: str) -> "ctrl.Antecedent":
    """Create an Antecedent on the shared universe with L/M/H terms."""
    ant = ctrl.Antecedent(UNIVERSE, name)
    add_lmh_terms(ant)
    return ant


def make_consequent(name: str) -> "ctrl.Consequent":
    """Create a Consequent on the shared universe with L/M/H terms.

    Uses the centroid defuzzification method (Module 5).
    """
    con = ctrl.Consequent(UNIVERSE, name, defuzzify_method="centroid")
    add_lmh_terms(con)
    return con
