"""Tests for the construct CLI."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from construct.models import Action, Scenario, TerminationReason
from construct.runner import ScenarioRunner
from construct.vlm import VLMResponse


# ---------------------------------------------------------------------------
# on_step callback integration
# ---------------------------------------------------------------------------


async def test_on_step_callback_invoked(construct_config, mock_vlm, mock_simulate, dummy_frame):
    """The on_step callback receives (step_idx, step, frame) for each step."""
    responses = [
        VLMResponse(action=Action(name="move_forward", parameters={"distance": 1.0}), done=False),
        VLMResponse(action=Action(name="done"), done=True),
    ]
    vlm = mock_vlm(responses)
    patches = mock_simulate()
    scenario = Scenario(name="cb_test", prompt="test callback")

    captured = []

    def on_step(step_idx, step, frame):
        captured.append((step_idx, step.action.name, frame.shape))

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        runner = ScenarioRunner(config=construct_config, vlm=vlm)
        result = await runner.run(scenario, on_step=on_step)

    assert result.termination_reason == TerminationReason.DONE
    assert len(captured) == 2
    assert captured[0] == (0, "move_forward", (64, 64, 3))
    assert captured[1] == (1, "done", (64, 64, 3))


async def test_on_step_none_is_fine(construct_config, mock_vlm, mock_simulate):
    """Passing on_step=None (the default) should not break anything."""
    responses = [VLMResponse(action=Action(name="done"), done=True)]
    vlm = mock_vlm(responses)
    patches = mock_simulate()
    scenario = Scenario(name="no_cb", prompt="no callback")

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        runner = ScenarioRunner(config=construct_config, vlm=vlm)
        result = await runner.run(scenario)

    assert result.termination_reason == TerminationReason.DONE
    assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# Frame saving
# ---------------------------------------------------------------------------


def test_make_frame_saver_writes_pngs(tmp_path):
    """_make_frame_saver should write PNG files in step subdirectories."""
    from construct.cli import _make_frame_saver
    from construct.models import StepResult

    frames_dir = tmp_path / "frames"
    saver = _make_frame_saver(frames_dir)

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    step = StepResult(step_index=0, action=Action(name="test"))
    # on_step now receives a single frame, not a list
    saver(0, step, frame)
    saver(1, step, frame)

    assert (frames_dir / "step_0000").is_dir()
    assert (frames_dir / "step_0001").is_dir()
    assert (frames_dir / "step_0000" / "frame_0000.png").exists()
    assert (frames_dir / "step_0001" / "frame_0000.png").exists()
    # Verify it's valid PNG (starts with PNG signature)
    data = (frames_dir / "step_0000" / "frame_0000.png").read_bytes()
    assert data[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# result.json saving
# ---------------------------------------------------------------------------


def test_save_result_json(tmp_path):
    """_save_result_json should write a valid JSON file with expected fields."""
    from construct.cli import _save_result_json

    scenario = Scenario(name="json_test", prompt="save json", tags=["t1"])
    result_obj = __import__("construct.models", fromlist=["ScenarioResult"]).ScenarioResult(scenario=scenario)
    result_obj.steps.append(
        __import__("construct.models", fromlist=["StepResult"]).StepResult(
            step_index=0,
            action=Action(name="act", parameters={"k": "v"}),
            reasoning="because",
            latency_ms=42.0,
        )
    )

    _save_result_json(tmp_path, result_obj)

    data = json.loads((tmp_path / "result.json").read_text())
    assert data["scenario"]["name"] == "json_test"
    assert data["scenario"]["tags"] == ["t1"]
    assert len(data["steps"]) == 1
    assert data["steps"][0]["action"]["name"] == "act"
    assert data["steps"][0]["reasoning"] == "because"
    assert data["termination_reason"] == "done"


# ---------------------------------------------------------------------------
# CLI integration via main()
# ---------------------------------------------------------------------------


def test_run_output_dir_structure(tmp_path, construct_config, mock_vlm, mock_simulate, monkeypatch):
    """Full CLI run should create outputs/<name>_<ts>/ with result.json and frames/."""
    import asyncio
    from construct.cli import _run_command

    # Write a scenario file
    scenario_file = tmp_path / "scenarios" / "demo.py"
    scenario_file.parent.mkdir()
    scenario_file.write_text(
        "from construct.models import Scenario, Action\n"
        "SCENARIOS = [Scenario(name='demo_s', prompt='do it')]\n"
    )

    responses = [VLMResponse(action=Action(name="done"), done=True)]
    vlm = mock_vlm(responses)
    patches = mock_simulate()

    monkeypatch.chdir(tmp_path)

    class FakeArgs:
        scenario = "demo"
        no_frames = False
        json = False
        verbose = False
        max_steps = None

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(patch("construct.cli.ConstructConfig", return_value=construct_config))
        stack.enter_context(patch("construct.cli.GeminiRoboticsBackend", return_value=vlm))
        code = asyncio.run(_run_command(FakeArgs()))

    assert code == 0

    # Find the output directory
    outputs = list((tmp_path / "outputs").iterdir())
    assert len(outputs) == 1
    run_dir = outputs[0]
    assert run_dir.name.startswith("demo_s_")
    assert (run_dir / "result.json").exists()
    assert (run_dir / "frames").is_dir()


def test_no_frames_skips_directory(tmp_path, construct_config, mock_vlm, mock_simulate, monkeypatch):
    """--no-frames should skip creating the frames directory."""
    import asyncio
    from construct.cli import _run_command

    scenario_file = tmp_path / "scenarios" / "nf.py"
    scenario_file.parent.mkdir()
    scenario_file.write_text(
        "from construct.models import Scenario, Action\n"
        "SCENARIOS = [Scenario(name='nf_s', prompt='no frames')]\n"
    )

    responses = [VLMResponse(action=Action(name="done"), done=True)]
    vlm = mock_vlm(responses)
    patches = mock_simulate()

    monkeypatch.chdir(tmp_path)

    class FakeArgs:
        scenario = "nf"
        no_frames = True
        json = False
        verbose = False
        max_steps = None

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(patch("construct.cli.ConstructConfig", return_value=construct_config))
        stack.enter_context(patch("construct.cli.GeminiRoboticsBackend", return_value=vlm))
        code = asyncio.run(_run_command(FakeArgs()))

    assert code == 0
    outputs = list((tmp_path / "outputs").iterdir())
    run_dir = outputs[0]
    assert (run_dir / "result.json").exists()
    assert not (run_dir / "frames").exists()


def test_missing_scenario_returns_exit_code_1(tmp_path, monkeypatch):
    """A missing scenario file should return exit code 1."""
    import asyncio
    from construct.cli import _run_command

    monkeypatch.chdir(tmp_path)

    class FakeArgs:
        scenario = "nonexistent"
        no_frames = False
        json = False
        verbose = False
        max_steps = None

    code = asyncio.run(_run_command(FakeArgs()))
    assert code == 1


def test_json_output(tmp_path, construct_config, mock_vlm, mock_simulate, monkeypatch, capsys):
    """--json should print JSON to stdout."""
    import asyncio
    from construct.cli import _run_command

    scenario_file = tmp_path / "scenarios" / "js.py"
    scenario_file.parent.mkdir()
    scenario_file.write_text(
        "from construct.models import Scenario, Action\n"
        "SCENARIOS = [Scenario(name='js_s', prompt='json out')]\n"
    )

    responses = [VLMResponse(action=Action(name="done"), done=True)]
    vlm = mock_vlm(responses)
    patches = mock_simulate()

    monkeypatch.chdir(tmp_path)

    class FakeArgs:
        scenario = "js"
        no_frames = True
        json = True
        verbose = False
        max_steps = None

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(patch("construct.cli.ConstructConfig", return_value=construct_config))
        stack.enter_context(patch("construct.cli.GeminiRoboticsBackend", return_value=vlm))
        code = asyncio.run(_run_command(FakeArgs()))

    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert data[0]["name"] == "js_s"


def test_max_steps_limits_actions(tmp_path, construct_config, mock_vlm, mock_simulate, monkeypatch):
    """--max-steps should override the scenario's max_steps."""
    import asyncio
    from construct.cli import _run_command

    scenario_file = tmp_path / "scenarios" / "ms.py"
    scenario_file.parent.mkdir()
    scenario_file.write_text(
        "from construct.models import Scenario\n"
        "SCENARIOS = [Scenario(name='ms_s', prompt='max steps', max_steps=50)]\n"
    )

    # 10 non-done responses — with max_steps=2 the runner should stop at 2
    responses = [
        VLMResponse(action=Action(name="wander"), done=False)
        for _ in range(10)
    ]
    vlm = mock_vlm(responses)
    patches = mock_simulate()

    monkeypatch.chdir(tmp_path)

    class FakeArgs:
        scenario = "ms"
        no_frames = True
        json = True
        verbose = False
        max_steps = 2

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(patch("construct.cli.ConstructConfig", return_value=construct_config))
        stack.enter_context(patch("construct.cli.GeminiRoboticsBackend", return_value=vlm))
        code = asyncio.run(_run_command(FakeArgs()))

    import json as _json
    result_dirs = list((tmp_path / "outputs").iterdir())
    data = _json.loads((result_dirs[0] / "result.json").read_text())
    assert len(data["steps"]) == 2
    assert data["termination_reason"] == "max_steps"


# ---------------------------------------------------------------------------
# load_scenario_file
# ---------------------------------------------------------------------------


def test_load_scenario_file(tmp_path):
    """load_scenario_file should load scenarios from a single .py file."""
    from construct.loader import load_scenario_file

    f = tmp_path / "single.py"
    f.write_text(
        "from construct.models import Scenario\n"
        "SCENARIOS = [Scenario(name='s1', prompt='p1')]\n"
    )
    result = load_scenario_file(f)
    assert len(result) == 1
    assert result[0].name == "s1"


def test_load_scenario_file_not_found():
    """load_scenario_file should raise FileNotFoundError for missing files."""
    from construct.loader import load_scenario_file

    with pytest.raises(FileNotFoundError, match="not found"):
        load_scenario_file("/nonexistent/file.py")
