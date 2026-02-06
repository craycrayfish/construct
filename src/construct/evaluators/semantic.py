"""Semantic similarity evaluator using an LLM as judge."""

from __future__ import annotations

from google import genai
from google.genai import types

from construct.evaluator import EvalScore
from construct.models import Scenario, ScenarioResult


_JUDGE_PROMPT = """\
You are evaluating whether a robot's action sequence is semantically equivalent \
to the expected sequence for a given task.

Task: {task}

Expected actions:
{expected}

Actual actions:
{actual}

Score from 0.0 to 1.0 how semantically similar the actual path is to the expected path. \
Consider action intent, ordering, and overall task completion.

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}
"""


class SemanticEvaluator:
    """Uses an LLM to judge semantic similarity between expected and actual action paths."""

    def __init__(self, *, api_key: str | None = None, threshold: float = 0.7) -> None:
        self._client = genai.Client(api_key=api_key)
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "semantic"

    async def evaluate(self, scenario: Scenario, result: ScenarioResult) -> EvalScore:
        expected_str = "\n".join(
            f"  {i+1}. {a.name}({a.parameters})" for i, a in enumerate(scenario.expected_actions)
        )
        actual_str = "\n".join(
            f"  {i+1}. {a.name}({a.parameters})" for i, a in enumerate(result.actions)
        )

        prompt = _JUDGE_PROMPT.format(
            task=scenario.prompt,
            expected=expected_str or "  (none specified)",
            actual=actual_str or "  (none)",
        )

        response = self._client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )

        import json
        text = response.text.strip()
        # Strip markdown code fences if present
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
