"""Composite evaluator — combine multiple evaluators with AND/OR/threshold logic."""

from __future__ import annotations

import enum

from construct.evaluator import Evaluator, EvalScore
from construct.models import Scenario, ScenarioResult


class CombineMode(enum.Enum):
    AND = "and"
    OR = "or"
    THRESHOLD = "threshold"


class CompositeEvaluator:
    """Combines multiple evaluators using AND, OR, or a weighted score threshold."""

    def __init__(
        self,
        evaluators: list[Evaluator],
        mode: CombineMode = CombineMode.AND,
        threshold: float = 0.7,
    ) -> None:
        self._evaluators = evaluators
        self._mode = mode
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "composite"

    async def evaluate(self, scenario: Scenario, result: ScenarioResult) -> EvalScore:
        sub_scores: list[EvalScore] = []
        for ev in self._evaluators:
            sub_scores.append(await ev.evaluate(scenario, result))

        if not sub_scores:
            return EvalScore(name=self.name, score=1.0, passed=True, details={"sub_scores": []})

        avg_score = sum(s.score for s in sub_scores) / len(sub_scores)

        if self._mode == CombineMode.AND:
            passed = all(s.passed for s in sub_scores)
        elif self._mode == CombineMode.OR:
            passed = any(s.passed for s in sub_scores)
        else:  # THRESHOLD
            passed = avg_score >= self._threshold

        details = {
            "mode": self._mode.value,
            "sub_scores": [
                {"name": s.name, "score": s.score, "passed": s.passed}
                for s in sub_scores
            ],
        }

        return EvalScore(
            name=self.name,
            score=avg_score,
            passed=passed,
            details=details,
        )
