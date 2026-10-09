"""Module 3 — Three Fuzzy Expert Subsystems (Mamdani inference).

Rather than one large fuzzy system with hundreds of rules, three small ones each
take two features and emit a Low/Medium/High complexity, defuzzified to a
continuous score in ``[0, 1]``:

* **Linguistic**  : query_length + dependency_depth
* **Knowledge**   : entity_density + domain_keyword_density
* **Reasoning**   : intent_type + semantic_ambiguity

Each subsystem has a complete 3x3 = 9-rule base. The default output level is the
rounded average of the two input levels (Low=0, Medium=1, High=2, ties round
up). The Reasoning subsystem additionally biases a High intent (e.g. a
comparison query) toward High complexity, implementing the spec rule
"IF Intent is Comparison THEN Reasoning Complexity is High".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from skfuzzy import control as ctrl

from ..logging_utils import get_logger
from .membership import LEVELS, make_antecedent, make_consequent

logger = get_logger(__name__)

_LEVEL_IDX = {"low": 0, "medium": 1, "high": 2}
_IDX_LEVEL = {0: "low", 1: "medium", 2: "high"}


def _avg_level(a: str, b: str) -> str:
    """Rounded average of two levels (ties round up)."""
    avg = (_LEVEL_IDX[a] + _LEVEL_IDX[b]) / 2.0
    # round-half-up
    import math

    return _IDX_LEVEL[int(math.floor(avg + 0.5))]


def _build_grid(overrides: Dict[Tuple[str, str], str] | None = None) -> Dict[Tuple[str, str], str]:
    """Build a complete 3x3 (level_a, level_b) -> out_level mapping."""
    grid: Dict[Tuple[str, str], str] = {}
    for a in LEVELS:
        for b in LEVELS:
            grid[(a, b)] = _avg_level(a, b)
    if overrides:
        grid.update(overrides)
    return grid


@dataclass
class FuzzySubsystem:
    """A two-input, one-output Mamdani fuzzy subsystem.

    Attributes:
        name: Human-readable subsystem name.
        input_a_name: Name of the first input variable.
        input_b_name: Name of the second input variable.
        output_name: Name of the output variable.
    """

    name: str
    input_a_name: str
    input_b_name: str
    output_name: str
    _sim: ctrl.ControlSystemSimulation = None  # set in __post_init__

    def __init__(
        self,
        name: str,
        input_a_name: str,
        input_b_name: str,
        output_name: str,
        overrides: Dict[Tuple[str, str], str] | None = None,
    ) -> None:
        self.name = name
        self.input_a_name = input_a_name
        self.input_b_name = input_b_name
        self.output_name = output_name

        self._a = make_antecedent(input_a_name)
        self._b = make_antecedent(input_b_name)
        self._out = make_consequent(output_name)

        grid = _build_grid(overrides)
        rules = []
        for (la, lb), out_level in grid.items():
            rules.append(
                ctrl.Rule(self._a[la] & self._b[lb], self._out[out_level])
            )
        self._grid = grid
        self._system = ctrl.ControlSystem(rules)
        self._sim = ctrl.ControlSystemSimulation(self._system)
        logger.debug("Built subsystem '%s' with %d rules", name, len(rules))

    def infer(self, a: float, b: float) -> float:
        """Run inference and return the defuzzified output in ``[0, 1]``.

        Args:
            a: Crisp value of the first input (``[0, 1]``).
            b: Crisp value of the second input (``[0, 1]``).

        Returns:
            The centroid-defuzzified complexity score in ``[0, 1]``.
        """
        # Use a fresh simulation-free compute call by resetting inputs each time.
        self._sim.input[self.input_a_name] = float(max(0.0, min(1.0, a)))
        self._sim.input[self.input_b_name] = float(max(0.0, min(1.0, b)))
        self._sim.compute()
        score = float(self._sim.output[self.output_name])
        logger.debug(
            "%s.infer(%s=%.3f, %s=%.3f) -> %.4f",
            self.name, self.input_a_name, a, self.input_b_name, b, score,
        )
        return score


@dataclass(frozen=True)
class SubsystemScores:
    """Outputs of the three subsystems (Module 3)."""

    linguistic: float
    knowledge: float
    reasoning: float


class ComplexitySubsystems:
    """Bundles the three expert subsystems."""

    def __init__(self) -> None:
        self.linguistic = FuzzySubsystem(
            name="LinguisticComplexity",
            input_a_name="query_length",
            input_b_name="dependency_depth",
            output_name="linguistic_complexity",
        )
        self.knowledge = FuzzySubsystem(
            name="KnowledgeComplexity",
            input_a_name="entity_density",
            input_b_name="domain_density",
            output_name="knowledge_complexity",
        )
        # Reasoning: a High intent (comparison) alone forces High complexity.
        reasoning_overrides = {
            ("high", "low"): "high",
            ("high", "medium"): "high",
            ("high", "high"): "high",
        }
        self.reasoning = FuzzySubsystem(
            name="ReasoningComplexity",
            input_a_name="intent_type",
            input_b_name="semantic_ambiguity",
            output_name="reasoning_complexity",
            overrides=reasoning_overrides,
        )

    def infer(
        self,
        query_length: float,
        dependency_depth: float,
        entity_density: float,
        domain_density: float,
        intent_type: float,
        semantic_ambiguity: float,
    ) -> SubsystemScores:
        """Run all three subsystems on the six normalized features.

        Returns:
            A :class:`SubsystemScores` with linguistic/knowledge/reasoning
            complexity, each in ``[0, 1]``.
        """
        ling = self.linguistic.infer(query_length, dependency_depth)
        know = self.knowledge.infer(entity_density, domain_density)
        reas = self.reasoning.infer(intent_type, semantic_ambiguity)
        scores = SubsystemScores(linguistic=ling, knowledge=know, reasoning=reas)
        logger.debug("Subsystem scores: %s", scores)
        return scores
