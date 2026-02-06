# CLAUDE.md — construct

## What is this repo?

A testing framework for VLMs controlling robotic systems. Odyssey (world model) generates video frames, a VLM (Gemini Robotics ER) decides robot actions via tool calls, and evaluators score the action path.

## Tech stack

- Python 3.12+, `uv` for dependency management, `hatchling` build backend
- `odyssey` SDK (git dependency) — world model client
- `google-genai` — Gemini Robotics ER backend
- `pytest` + `pytest-asyncio` — test runner
- `numpy` + `pillow` — frame handling

## Project layout

```
src/construct/          # Package source
  models.py             # Scenario, Action, StepResult, ScenarioResult
  vlm.py                # VLMBackend Protocol, VLMResponse
  evaluator.py          # Evaluator Protocol, EvalScore
  config.py             # ConstructConfig (env vars)
  session.py            # OdysseySession (async ctx manager, frame sync)
  runner.py             # ScenarioRunner (core simulation loop)
  gemini.py             # Gemini Robotics ER backend
  frame_utils.py        # numpy → PNG/base64
  report.py             # Console/JSON reporting
  pytest_plugin.py      # Fixtures registered via entry-points.pytest11
  evaluators/           # exact, semantic, outcome, composite
tests/                  # Self-tests using mock VLM + mocked Odyssey
```

## Common commands

```bash
uv sync --extra test     # Install all deps including test extras
uv run pytest tests/ -v  # Run tests
```

## Key patterns

- All Odyssey and VLM calls are async. Tests use `asyncio_mode = "auto"`.
- `OdysseySession` uses `asyncio.Event` for frame sync: `interact()` clears the event, the frame callback sets it, `wait_for_frame()` awaits it.
- Frame callbacks from Odyssey are **sync** (not coroutines), called from an async background task.
- The VLM backend is a Protocol — swap `GeminiRoboticsBackend` for any implementation matching `VLMBackend`.
- Tests mock the Odyssey session with `AsyncMock` and use `MockVLMBackend` with pre-configured responses.

## Environment variables

- `ODYSSEY_API_KEY` — required for real Odyssey connections
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` — required for Gemini backend
