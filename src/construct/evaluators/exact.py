"""Exact action sequence match evaluator."""

from __future__ import annotations

from construct.evaluator import EvalScore
from construct.models import Scenario, ScenarioResult


class ExactPathEvaluator:
    """Compares the action sequence against expected_actions exactly."""

    def __init__(self, *, check_parameters: bool = True) -> None:
        self._check_parameters = check_parameters

    @property
    def name(self) -> str:
        return "exact_path"

    async def evaluate(self, scenario: Scenario, result: ScenarioResult) -> EvalScore:
        expected = scenario.expected_actions
        actual = result.actions

        if len(expected) != len(actual):
            return EvalScore(
                name=self.name,
                score=0.0,
                passed=False,
                details={
                    "reason": "length_mismatch",
                    "expected_count": len(expected),
                    "actual_count": len(actual),
                },
            )

        matches = 0
        mismatches: list[dict] = []
        for i, (exp, act) in enumerate(zip(expected, actual)):
            name_match = exp.name == act.name
            param_match = (not self._check_parameters) or (exp.parameters == act.parameters)
            if name_match and param_match:
                matches += 1
            else:
                mismatches.append({
                    "step": i,
                    "expected": {"name": exp.name, "parameters": exp.parameters},
                    "actual": {"name": act.name, "parameters": act.parameters},
                })

        score = matches / len(expected) if expected else 1.0
        return EvalScore(
            name=self.name,
            score=score,
            passed=score == 1.0,
            details={"mismatches": mismatches},
        )
