"""VLM backend protocol and response model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from construct.models import Action, Scenario


@dataclass
class VLMResponse:
    """Response from a VLM backend decision call."""

    action: Action
    done: bool = False
    reasoning: str = ""
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@runtime_checkable
class VLMBackend(Protocol):
    """Protocol for VLM backends that decide robot actions from frames."""

    async def decide(
        self,
        frame: np.ndarray,
        step_index: int,
        scenario: Scenario,
        action_history: list[Action],
    ) -> VLMResponse:
        """Given a frame and context, decide the next action."""
        ...

    async def close(self) -> None:
        """Release any resources held by the backend."""
        ...
