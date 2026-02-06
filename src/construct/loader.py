"""Scenario loader — discovers and imports scenario definitions from a directory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from construct.models import Scenario


def load_scenario_file(filepath: str | Path) -> list[Scenario]:
    """Load scenarios from a single Python module.

    The module must define a module-level ``SCENARIOS`` list of
    :class:`Scenario` instances.

    Parameters
    ----------
    filepath:
        Path to the ``.py`` file containing scenario definitions.

    Returns
    -------
    list[Scenario]
        The scenarios defined in the file.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the module is missing ``SCENARIOS`` or contains invalid entries.
    """
    py_file = Path(filepath)
    if not py_file.is_file():
        raise FileNotFoundError(f"Scenario file not found: {py_file}")

    module_name = f"_construct_scenarios.{py_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import scenario file: {py_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "SCENARIOS"):
        raise ValueError(
            f"Scenario file {py_file.name} is missing a SCENARIOS list"
        )

    module_scenarios = module.SCENARIOS
    if not isinstance(module_scenarios, list):
        raise ValueError(
            f"SCENARIOS in {py_file.name} must be a list, got {type(module_scenarios).__name__}"
        )

    for i, item in enumerate(module_scenarios):
        if not isinstance(item, Scenario):
            raise ValueError(
                f"SCENARIOS[{i}] in {py_file.name} is {type(item).__name__}, expected Scenario"
            )

    return module_scenarios


def load_scenarios(directory: str | Path) -> list[Scenario]:
    """Load all scenarios from Python modules in *directory*.

    Each ``.py`` file (excluding files starting with ``_``) must define a
    module-level ``SCENARIOS`` list of :class:`Scenario` instances.

    Parameters
    ----------
    directory:
        Path to the directory containing scenario modules.

    Returns
    -------
    list[Scenario]
        A flat list of every scenario found across all modules.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    ValueError
        If a module is missing ``SCENARIOS`` or contains invalid entries.
    """
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Scenario directory not found: {path}")

    scenarios: list[Scenario] = []

    for py_file in sorted(path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        scenarios.extend(load_scenario_file(py_file))

    return scenarios
