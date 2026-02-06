"""Evaluator protocol and score model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from construct.models import Scenario, ScenarioResult


@dataclass
class EvalScore:
    """Score produced by an evaluator."""

    name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for evaluators that score scenario results."""

    @property
    def name(self) -> str: ...

    async def evaluate(self, scenario: Scenario, result: ScenarioResult) -> EvalScore: ...
