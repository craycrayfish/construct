"""pytest plugin for construct — fixtures, markers, and hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from construct.config import ConstructConfig
from construct.evaluators.exact import ExactPathEvaluator
from construct.loader import load_scenarios
from construct.models import Scenario, ScenarioResult
from construct.runner import ScenarioRunner
from construct.vlm import VLMBackend


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "construct: mark a test as a construct scenario test")


@pytest.fixture(scope="session")
def construct_config() -> ConstructConfig:
    """Session-scoped construct configuration from environment."""
    return ConstructConfig()


@pytest.fixture(scope="session")
def vlm_backend(construct_config: ConstructConfig) -> VLMBackend:
    """Session-scoped VLM backend. Override this fixture to use a custom backend."""
    from construct.gemini import GeminiRoboticsBackend

    return GeminiRoboticsBackend(construct_config)


@pytest.fixture()
def evaluators() -> list:
    """Default evaluators. Override this fixture to customize."""
    return [ExactPathEvaluator()]


@pytest.fixture()
def scenario_runner(
    construct_config: ConstructConfig,
    vlm_backend: VLMBackend,
    evaluators: list,
) -> ScenarioRunner:
    """A configured ScenarioRunner ready to execute scenarios."""
    return ScenarioRunner(config=construct_config, vlm=vlm_backend, evaluators=evaluators)


@pytest.fixture()
def run_scenario(scenario_runner: ScenarioRunner):
    """Convenience fixture — call with a Scenario, returns ScenarioResult.

    Automatically asserts that the result passed.
    """

    async def _run(scenario: Scenario) -> ScenarioResult:
        result = await scenario_runner.run(scenario)
        assert result.passed, (
            f"Scenario '{scenario.name}' failed: "
            f"termination={result.termination_reason.value}, "
            f"scores={result.eval_scores}, "
            f"error={result.error}"
        )
        return result

    return _run


@pytest.fixture(scope="session")
def all_scenarios(request: pytest.FixtureRequest) -> list[Scenario]:
    """Load all scenarios from the ``scenarios/`` directory at the project root."""
    root = Path(request.config.rootdir) / "scenarios"
    if not root.is_dir():
        return []
    return load_scenarios(root)


@pytest.fixture()
def run_all_scenarios(
    all_scenarios: list[Scenario],
    scenario_runner: ScenarioRunner,
):
    """Run every loaded scenario and return results.

    Automatically asserts that each scenario passed.
    """

    async def _run() -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        for scenario in all_scenarios:
            result = await scenario_runner.run(scenario)
            assert result.passed, (
                f"Scenario '{scenario.name}' failed: "
                f"termination={result.termination_reason.value}, "
                f"scores={result.eval_scores}, "
                f"error={result.error}"
            )
            results.append(result)
        return results

    return _run
