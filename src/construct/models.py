"""Core data models for the construct testing framework."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class TerminationReason(enum.Enum):
    """Why a scenario run ended."""

    DONE = "done"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class Action:
    """A single robot action (tool call)."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def to_interact_prompt(self) -> str:
        """Format as a prompt string for Odyssey interact()."""
        if self.raw_text:
            return self.raw_text
        parts = [self.name]
        for k, v in self.parameters.items():
            parts.append(f"{k}={v}")
        return " ".join(parts)


@dataclass(frozen=True)
class Scenario:
    """A test scenario describing an expected robot behaviour."""

    name: str
    prompt: str
    expected_actions: list[Action] = field(default_factory=list)
    success_criteria: str = ""
    image: str | bytes | np.ndarray | Path | None = None
    portrait: bool = True
    max_steps: int = 20
    timeout_s: float = 120.0
    vlm_system_prompt: str | None = None
    tags: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class StepResult:
    """The outcome of a single simulation step."""

    step_index: int
    action: Action
    reasoning: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class ScenarioResult:
    """Complete result of running a scenario."""

    scenario: Scenario
    steps: list[StepResult] = field(default_factory=list)
    termination_reason: TerminationReason = TerminationReason.DONE
    error: str | None = None
    eval_scores: dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0

    @property
    def actions(self) -> list[Action]:
        return [s.action for s in self.steps]

    @property
    def passed(self) -> bool:
        if not self.eval_scores:
            return self.termination_reason == TerminationReason.DONE
        return all(
            score.get("passed", False) if isinstance(score, dict) else getattr(score, "passed", False)
            for score in self.eval_scores.values()
        )
