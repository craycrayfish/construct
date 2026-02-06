"""Outcome evaluator — judges whether the final state satisfies success criteria."""

from __future__ import annotations

import json

from google import genai
from google.genai import types

from construct.evaluator import EvalScore
from construct.models import Scenario, ScenarioResult

_OUTCOME_PROMPT = """\
You are evaluating whether a robot successfully completed a task based on the \
actions it took.

Task: {task}
Success criteria: {criteria}

Actions taken:
{actions}

Termination reason: {termination}

Did the robot successfully complete the task? Score from 0.0 to 1.0.

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}
"""


class OutcomeEvaluator:
    """Evaluates whether the final outcome satisfies the scenario's success criteria."""

    def __init__(self, *, api_key: str | None = None, threshold: float = 0.7) -> None:
        self._client = genai.Client(api_key=api_key)
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "outcome"

    async def evaluate(self, scenario: Scenario, result: ScenarioResult) -> EvalScore:
        actions_str = "\n".join(
            f"  {i+1}. {a.name}({a.parameters})" for i, a in enumerate(result.actions)
        )

        prompt = _OUTCOME_PROMPT.format(
            task=scenario.prompt,
            criteria=scenario.success_criteria or "(not specified)",
            actions=actions_str or "  (none)",
            termination=result.termination_reason.value,
        )

        response = self._client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        parsed = json.loads(text)
        score = float(parsed["score"])

        return EvalScore(
            name=self.name,
            score=score,
            passed=score >= self._threshold,
            details={"reasoning": parsed.get("reasoning", ""), "threshold": self._threshold},
        )
