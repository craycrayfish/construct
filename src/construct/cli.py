"""construct CLI — run scenarios from the command line."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from construct.config import ConstructConfig
from construct.frame_utils import frame_to_png_bytes
from construct.gemini import GeminiRoboticsBackend
from construct.loader import load_scenario_file
from construct.models import ScenarioResult, StepResult
from construct.report import StructuredReport
from construct.runner import ScenarioRunner


logger = logging.getLogger("construct")


def _resolve_scenario_path(name: str) -> Path:
    """Resolve a scenario name to a file path.

    Looks for ``scenarios/<name>.py`` relative to cwd.  If *name* already
    ends with ``.py`` or is an absolute path, use it directly.
    """
    p = Path(name)
    if p.suffix == ".py" and p.exists():
        return p
    candidate = Path("scenarios") / f"{name}.py"
    if candidate.exists():
        return candidate
    # Fall back — maybe the user gave a full relative path without .py
    candidate = Path(f"{name}.py")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Scenario file not found: {name}")


def _make_frame_saver(frames_dir: Path):
    """Return an ``on_step`` callback that writes the VLM decision frame as a PNG.

    Each step gets a subdirectory ``step_XXXX/`` containing a single
    ``frame_0000.png`` — the last frame of the simulation clip that the
    VLM evaluated.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    def _save(step_idx: int, step: StepResult, frame: np.ndarray) -> None:
        step_dir = frames_dir / f"step_{step_idx:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        path = step_dir / "frame_0000.png"
        path.write_bytes(frame_to_png_bytes(frame))

    return _save


def _save_result_json(run_dir: Path, result: ScenarioResult) -> None:
    """Write ``result.json`` with scenario metadata, steps, and eval scores."""
    data = {
        "scenario": {
            "name": result.scenario.name,
            "prompt": result.scenario.prompt,
            "success_criteria": result.scenario.success_criteria,
            "max_steps": result.scenario.max_steps,
            "timeout_s": result.scenario.timeout_s,
            "tags": result.scenario.tags,
        },
        "termination_reason": result.termination_reason.value,
        "passed": result.passed,
        "steps": [
            {
                "step_index": s.step_index,
                "action": {"name": s.action.name, "parameters": s.action.parameters},
                "reasoning": s.reasoning,
                "latency_ms": s.latency_ms,
                "cost_usd": s.cost_usd,
            }
            for s in result.steps
        ],
        "eval_scores": result.eval_scores,
        "total_latency_ms": result.total_latency_ms,
        "total_cost_usd": result.total_cost_usd,
        "error": result.error,
    }
    out = run_dir / "result.json"
    out.write_text(json.dumps(data, indent=2))


async def _run_command(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand."""
    try:
        scenario_path = _resolve_scenario_path(args.scenario)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    scenarios = load_scenario_file(scenario_path)
    if not scenarios:
        print(f"No scenarios found in {scenario_path}", file=sys.stderr)
        return 1

    config = ConstructConfig()
    vlm = GeminiRoboticsBackend(config)
    runner = ScenarioRunner(config=config, vlm=vlm)
    report = StructuredReport()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for scenario in scenarios:
        if args.max_steps is not None:
            scenario = dataclasses.replace(scenario, max_steps=args.max_steps)

        run_dir = Path("outputs") / f"{scenario.name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        on_step = None
        if not args.no_frames:
            on_step = _make_frame_saver(run_dir / "frames")

        logger.info("Running scenario: %s", scenario.name)
        result = await runner.run(scenario, on_step=on_step)
        _save_result_json(run_dir, result)
        report.add(result)
        logger.info("Saved artifacts to %s", run_dir)

    if args.json:
        print(report.to_json())
    else:
        print(report.to_console())

    return 0 if report.all_passed else 1


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``construct`` CLI."""
    parser = argparse.ArgumentParser(prog="construct", description="construct — VLM robotic test runner")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run scenarios from a file")
    run_parser.add_argument("scenario", help="Scenario file name (e.g. 'example' for scenarios/example.py)")
    run_parser.add_argument("--no-frames", action="store_true", help="Skip saving video frames")
    run_parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    run_parser.add_argument("--max-steps", type=int, default=None, help="Override max actions per scenario (default: use scenario value)")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    view_parser = sub.add_parser("view", help="Browse run outputs in a web UI")
    view_parser.add_argument("--outputs-dir", default="outputs", help="Outputs directory")
    view_parser.add_argument("--port", type=int, default=8228, help="Server port")
    view_parser.add_argument("--no-open", action="store_true", help="Don't open browser")
    view_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command == "run":
        exit_code = asyncio.run(_run_command(args))
    elif args.command == "view":
        from construct.viewer import view_command

        exit_code = view_command(args)
    else:
        parser.print_help()
        exit_code = 0

    sys.exit(exit_code)
