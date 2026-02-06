# CLAUDE.md — construct

## What is this repo?

A testing framework for VLMs controlling robotic systems. Odyssey (world model) generates video frames, a VLM (Gemini Robotics ER) decides robot actions via tool calls, and evaluators score the action path.

## Tech stack

- Python 3.12+, `uv` for dependency management, `hatchling` build backend
- `odyssey` SDK (git dependency) — world model client
- `google-genai` — Gemini Robotics ER backend
- `pytest` + `pytest-asyncio` — test runner
- `numpy` + `pillow` — frame handling
- `av` (PyAV) — video frame extraction from simulation clips

## Project layout

```
src/construct/          # Package source
  models.py             # Scenario, Action, StepResult, ScenarioResult
  vlm.py                # VLMBackend Protocol, VLMResponse
  evaluator.py          # Evaluator Protocol, EvalScore
  config.py             # ConstructConfig (env vars)
  runner.py             # ScenarioRunner (simulate-based clip-chaining loop)
  prompt.py             # PromptBuilder type, default_prompt_builder
  video_utils.py        # download_video, extract_last_frame (PyAV)
  gemini.py             # Gemini Robotics ER backend
  frame_utils.py        # numpy → PNG/base64
  report.py             # Console/JSON reporting
  viewer.py             # Local web UI for browsing run outputs
  pytest_plugin.py      # Fixtures registered via entry-points.pytest11
  evaluators/           # exact, semantic, outcome, composite
tests/                  # Self-tests using mock VLM + mocked Odyssey
```

## Common commands

```bash
uv sync --extra test     # Install all deps including test extras
uv run pytest tests/ -v  # Run tests
uv run construct view    # Browse run outputs in a local web UI
```

## Key patterns

- All Odyssey and VLM calls are async. Tests use `asyncio_mode = "auto"`.
- `ScenarioRunner` uses the Odyssey **simulate API** for clip-chaining: each VLM decision produces a new simulation clip. The last frame of clip N becomes the starting image of clip N+1, and prompts accumulate via a pluggable `PromptBuilder` function.
- The simulate loop: `simulate(script)` → poll `get_simulate_status()` → download video → `extract_last_frame()` (PyAV) → VLM decides → repeat.
- The VLM backend is a Protocol — swap `GeminiRoboticsBackend` for any implementation matching `VLMBackend`.
- Tests mock `Odyssey.simulate`, `download_video`, and `extract_last_frame` and use `MockVLMBackend` with pre-configured responses.

## Viewer (`construct view`)

A zero-dependency local web UI for browsing scenario run outputs. Serves a single-page app from Python's `http.server` with vanilla JS — no npm or extra packages.

```bash
construct view                          # serve outputs/ on port 8228, open browser
construct view --port 9000              # custom port
construct view --outputs-dir /path      # custom outputs directory
construct view --no-open                # don't auto-open browser
```

The UI has a sidebar listing all runs (newest-first, with pass/fail badges) and a main panel showing scenario metadata, frame images, step-by-step navigation (arrow keys or buttons), and action details (name, parameters, reasoning, latency, cost).

The server exposes a small JSON API (`/api/runs`, `/api/runs/<name>`) and serves frame PNGs from `/frames/<name>/<file>`. It binds to `127.0.0.1` only and rejects path-traversal attempts.

## Environment variables

- `ODYSSEY_API_KEY` — required for real Odyssey connections
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` — required for Gemini backend
