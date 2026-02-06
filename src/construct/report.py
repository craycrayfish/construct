"""Structured reporting for scenario results — console and JSON output."""

from __future__ import annotations

import json
from pathlib import Path

from construct.models import ScenarioResult


class StructuredReport:
    """Collects scenario results and produces console or JSON reports."""

    def __init__(self) -> None:
        self._results: list[ScenarioResult] = []

    def add(self, result: ScenarioResult) -> None:
        self._results.append(result)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self._results)

    def to_console(self) -> str:
        lines: list[str] = []
        lines.append(f"\n{'='*60}")
        lines.append("CONSTRUCT TEST REPORT")
        lines.append(f"{'='*60}")

        for r in self._results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.scenario.name}")
            lines.append(f"         termination: {r.termination_reason.value}")
            lines.append(f"         steps: {len(r.steps)}")
            if r.eval_scores:
                for name, score in r.eval_scores.items():
                    s = score if isinstance(score, dict) else {"score": score}
                    lines.append(f"         {name}: {s.get('score', '?')} ({'PASS' if s.get('passed') else 'FAIL'})")
            if r.error:
                lines.append(f"         error: {r.error}")

        passed = sum(1 for r in self._results if r.passed)
        total = len(self._results)
        lines.append(f"{'='*60}")
        lines.append(f"  {passed}/{total} scenarios passed")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        return [
            {
                "name": r.scenario.name,
                "passed": r.passed,
                "termination_reason": r.termination_reason.value,
                "steps": len(r.steps),
                "total_latency_ms": r.total_latency_ms,
                "total_cost_usd": r.total_cost_usd,
                "eval_scores": r.eval_scores,
                "error": r.error,
            }
            for r in self._results
        ]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())
