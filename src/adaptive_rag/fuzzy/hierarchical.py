"""Modules 4 & 5 — Hierarchical Fuzzy Engine + Defuzzification.

The second-stage Mamdani engine combines the three subsystem outputs
(Linguistic, Knowledge, Reasoning complexity) into a single **Overall
Complexity**, defuzzified with the centroid method into a continuous score in
``[0.00, 1.00]``.

Rule base
---------
Because this stage operates on three *summarized* inputs (not six raw features),
a compact rule base suffices. We enumerate all 27 level-combinations
(3 inputs x Low/Medium/High) and map each to an output level with a
transparent, monotonic aggregation based on the level-sum
(Low=0, Medium=1, High=2, so the sum ranges 0..6):

* **Low**    if level-sum <= 1  (all Low, or one Medium and the rest Low).
* **Medium** if level-sum is 2 or 3.
* **High**   if level-sum >= 4.

This makes the base *complete by construction*: every one of the 27
combinations maps to exactly one output level, so no input can produce an empty
output. The 27 combinations are then compressed into three disjunctive rules
(one per output level) — the executable equivalent of the canonical statements
documented in :data:`CANONICAL_RULES`.

This mapping is consistent with the spec's examples:

* IF Linguistic High AND Knowledge High THEN Overall High  (sum >= 4 -> High).
* IF Linguistic Medium AND Knowledge Medium AND Reasoning Low THEN Overall Medium
  (sum = 2 -> Medium).
* IF all Low THEN Overall Low  (sum = 0 -> Low).

It is also monotonic: raising any single input's level never lowers the overall
complexity, which underpins the monotonicity sanity checks in the test suite.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

from skfuzzy import control as ctrl

from ..logging_utils import get_logger
from .membership import LEVELS, make_antecedent, make_consequent
from .subsystems import SubsystemScores

logger = get_logger(__name__)

_LEVEL_IDX = {"low": 0, "medium": 1, "high": 2}

# Human-readable canonical rules this aggregation encodes (for documentation and
# the README). The executable system compresses these into 3 disjunctive rules.
CANONICAL_RULES: Tuple[str, ...] = (
    "IF all three are Low THEN Overall is Low",
    "IF one is Medium AND the other two are Low THEN Overall is Low",
    "IF two are Medium AND one is Low THEN Overall is Medium",
    "IF Linguistic is Medium AND Knowledge is Medium AND Reasoning is Low THEN Overall is Medium",
    "IF all three are Medium THEN Overall is Medium",
    "IF one is High AND the other two are Low THEN Overall is Medium",
    "IF one is High AND one Medium AND one Low THEN Overall is Medium",
    "IF two or more are High THEN Overall is High",
    "IF Linguistic is High AND Knowledge is High THEN Overall is High",
    "IF one is High AND two are Medium THEN Overall is High",
    "IF one is High AND one Medium AND (level-sum >= 4) THEN Overall is High",
    "IF the summed level (Low=0,Med=1,High=2) <= 1 THEN Overall is Low",
    "IF the summed level is 2 or 3 THEN Overall is Medium",
    "IF the summed level >= 4 THEN Overall is High",
)


def aggregate_level(linguistic: str, knowledge: str, reasoning: str) -> str:
    """Map three input levels to an overall level (see module docstring).

    Args:
        linguistic: One of ``{"low", "medium", "high"}``.
        knowledge: One of ``{"low", "medium", "high"}``.
        reasoning: One of ``{"low", "medium", "high"}``.

    Returns:
        The overall level, one of ``{"low", "medium", "high"}``.
    """
    levels = [linguistic, knowledge, reasoning]
    level_sum = sum(_LEVEL_IDX[x] for x in levels)

    if level_sum <= 1:
        return "low"
    if level_sum >= 4:
        return "high"
    return "medium"


def build_rule_map() -> Dict[Tuple[str, str, str], str]:
    """Enumerate all 27 combinations and their overall level."""
    rule_map: Dict[Tuple[str, str, str], str] = {}
    for ling, know, reas in itertools.product(LEVELS, repeat=3):
        rule_map[(ling, know, reas)] = aggregate_level(ling, know, reas)
    return rule_map


@dataclass(frozen=True)
class ComplexityResult:
    """Full output of the fuzzy engine (Modules 3-5)."""

    subsystem: SubsystemScores
    overall_score: float  # continuous complexity score in [0, 1]


class HierarchicalFuzzyEngine:
    """Second-stage Mamdani engine producing the Overall Complexity score."""

    def __init__(self) -> None:
        self._ling = make_antecedent("linguistic")
        self._know = make_antecedent("knowledge")
        self._reas = make_antecedent("reasoning")
        self._out = make_consequent("overall")

        self._rule_map = build_rule_map()

        # Compress the 27 combinations into one disjunctive rule per output level.
        grouped: Dict[str, List] = {"low": [], "medium": [], "high": []}
        for (ling, know, reas), out_level in self._rule_map.items():
            antecedent = self._ling[ling] & self._know[know] & self._reas[reas]
            grouped[out_level].append(antecedent)

        rules = []
        for out_level, antecedents in grouped.items():
            if not antecedents:
                continue
            combined = antecedents[0]
            for extra in antecedents[1:]:
                combined = combined | extra
            rules.append(ctrl.Rule(combined, self._out[out_level]))

        self._system = ctrl.ControlSystem(rules)
        self._sim = ctrl.ControlSystemSimulation(self._system)
        logger.debug(
            "Built hierarchical engine: %d combinations -> %d grouped rules "
            "(low=%d, medium=%d, high=%d)",
            len(self._rule_map), len(rules),
            len(grouped["low"]), len(grouped["medium"]), len(grouped["high"]),
        )

    def infer(self, scores: SubsystemScores) -> float:
        """Combine subsystem scores into the overall complexity score.

        Args:
            scores: Outputs of the three subsystems.

        Returns:
            Centroid-defuzzified overall complexity in ``[0, 1]``.
        """
        self._sim.input["linguistic"] = float(max(0.0, min(1.0, scores.linguistic)))
        self._sim.input["knowledge"] = float(max(0.0, min(1.0, scores.knowledge)))
        self._sim.input["reasoning"] = float(max(0.0, min(1.0, scores.reasoning)))
        self._sim.compute()
        score = float(self._sim.output["overall"])
        # Numerical guard: keep strictly within [0, 1].
        score = max(0.0, min(1.0, score))
        logger.debug(
            "Hierarchical infer(L=%.3f, K=%.3f, R=%.3f) -> overall=%.4f",
            scores.linguistic, scores.knowledge, scores.reasoning, score,
        )
        return score
