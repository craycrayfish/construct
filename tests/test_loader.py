"""Tests for construct.loader — scenario discovery and import."""

from __future__ import annotations

from pathlib import Path

import pytest

from construct.loader import load_scenarios
from construct.models import Action, Scenario


@pytest.fixture()
def scenario_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with scenario modules."""
    return tmp_path


def _write_module(directory: Path, name: str, content: str) -> Path:
    p = directory / name
    p.write_text(content)
    return p


class TestLoadScenarios:
    def test_loads_single_file(self, scenario_dir: Path) -> None:
        _write_module(
            scenario_dir,
            "basic.py",
            (
                "from construct.models import Scenario, Action\n"
                "SCENARIOS = [\n"
                "    Scenario(name='test1', prompt='do something',\n"
                "             expected_actions=[Action(name='act1')]),\n"
                "]\n"
            ),
        )
        result = load_scenarios(scenario_dir)
        assert len(result) == 1
        assert result[0].name == "test1"

    def test_loads_multiple_files(self, scenario_dir: Path) -> None:
        for i in range(3):
            _write_module(
                scenario_dir,
                f"scenario_{i}.py",
                (
                    "from construct.models import Scenario\n"
                    f"SCENARIOS = [Scenario(name='s{i}', prompt='p{i}')]\n"
                ),
            )
        result = load_scenarios(scenario_dir)
        assert len(result) == 3
        names = {s.name for s in result}
        assert names == {"s0", "s1", "s2"}

    def test_skips_underscore_files(self, scenario_dir: Path) -> None:
        _write_module(
            scenario_dir,
            "_helper.py",
            "SCENARIOS = []\n",
        )
        _write_module(
            scenario_dir,
            "__init__.py",
            "",
        )
        _write_module(
            scenario_dir,
            "valid.py",
            (
                "from construct.models import Scenario\n"
                "SCENARIOS = [Scenario(name='v', prompt='p')]\n"
            ),
        )
        result = load_scenarios(scenario_dir)
        assert len(result) == 1
        assert result[0].name == "v"

    def test_error_missing_directory(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_scenarios("/nonexistent/path/scenarios")

    def test_error_missing_scenarios_attr(self, scenario_dir: Path) -> None:
        _write_module(scenario_dir, "bad.py", "x = 1\n")
        with pytest.raises(ValueError, match="missing a SCENARIOS list"):
            load_scenarios(scenario_dir)

    def test_error_scenarios_not_list(self, scenario_dir: Path) -> None:
        _write_module(
            scenario_dir,
            "bad.py",
            "SCENARIOS = 'not a list'\n",
        )
        with pytest.raises(ValueError, match="must be a list"):
            load_scenarios(scenario_dir)

    def test_error_invalid_scenario_type(self, scenario_dir: Path) -> None:
        _write_module(
            scenario_dir,
            "bad.py",
            "SCENARIOS = [{'name': 'oops'}]\n",
        )
        with pytest.raises(ValueError, match="expected Scenario"):
            load_scenarios(scenario_dir)

    def test_empty_directory(self, scenario_dir: Path) -> None:
        result = load_scenarios(scenario_dir)
        assert result == []

    def test_preserves_scenario_fields(self, scenario_dir: Path) -> None:
        _write_module(
            scenario_dir,
            "full.py",
            (
                "from construct.models import Scenario, Action\n"
                "SCENARIOS = [\n"
                "    Scenario(\n"
                "        name='full_test',\n"
                "        prompt='do something complex',\n"
                "        expected_actions=[Action(name='a', parameters={'k': 'v'})],\n"
                "        success_criteria='it works',\n"
                "        tags=['tag1', 'tag2'],\n"
                "    ),\n"
                "]\n"
            ),
        )
        result = load_scenarios(scenario_dir)
        assert len(result) == 1
        s = result[0]
        assert s.name == "full_test"
        assert s.prompt == "do something complex"
        assert len(s.expected_actions) == 1
        assert s.expected_actions[0].parameters == {"k": "v"}
        assert s.success_criteria == "it works"
        assert s.tags == ["tag1", "tag2"]
