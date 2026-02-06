# construct

A testing framework for Vision-Language Models (VLMs) controlling robotic systems.

construct connects a **world model** ([Odyssey](https://github.com/odysseyml/odyssey-python)) to a **VLM under test** ([Gemini Robotics ER](https://deepmind.google/technologies/gemini/robotics/)) in a closed loop, then scores the resulting action sequence with pluggable evaluators:

```
Odyssey generates frames  -->  VLM decides action  -->  action sent to Odyssey  -->  new frames  -->  ...  -->  evaluate
```

Odyssey simulates the physical world: given a scene description and a robot action, it streams video frames showing what the robot would see. The VLM observes each frame and returns the next tool-call action. construct orchestrates this loop, records every step, and evaluates whether the VLM made the right decisions.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An [Odyssey](https://odysseyml.com) API key
- A [Gemini](https://ai.google.dev/) API key (for the default Gemini Robotics ER backend)

## Installation

```bash
git clone <repo-url> && cd construct
uv sync              # production deps
uv sync --extra test # include test deps
```

## Configuration

Set the following environment variables (or add them to a `.env` file in the project root):

| Variable | Required | Purpose |
|---|---|---|
| `ODYSSEY_API_KEY` | Yes | Odyssey world-model API key |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Yes | Gemini Robotics ER API key |

## Quick start

### 1. Define a scenario

Create a file in `scenarios/`, e.g. `scenarios/pick_up_cup.py`:

```python
from construct.models import Scenario, Action

SCENARIOS = [
    Scenario(
        name="pick_up_cup",
        prompt="A robot arm near a table with a cup",
        expected_actions=[
            Action(name="move_forward", parameters={"distance": 0.5}),
            Action(name="pick_up", parameters={"object": "cup"}),
        ],
        success_criteria="The robot picks up the cup",
    ),
]
```

### 2. Run with the CLI

```bash
# Run a scenario file and save outputs (frames + result.json)
construct run pick_up_cup

# Limit the number of actions
construct run pick_up_cup --max-steps 5

# Output JSON report to stdout
construct run pick_up_cup --json

# Skip saving video frames
construct run pick_up_cup --no-frames
```

Results are written to `outputs/<scenario_name>_<timestamp>/` with:

```
outputs/pick_up_cup_20260206T031642Z/
  result.json
  frames/
    step_0000/
      frame_0000.png ... frame_0023.png
    step_0001/
      frame_0000.png ... frame_0023.png
```

### 3. Browse results in the viewer

```bash
construct view                    # serve on port 8228, open browser
construct view --port 9000        # custom port
construct view --outputs-dir /p   # custom outputs directory
construct view --no-open          # don't auto-open browser
```

The viewer is a zero-dependency local web UI with a sidebar listing all runs (with pass/fail badges), frame-by-frame animation playback, step navigation (arrow keys), and action details per step.

### 4. Run with pytest

construct ships a pytest plugin (registered automatically via `entry-points`) that provides fixtures for writing test cases:

```python
from construct.models import Scenario, Action

async def test_pick_up_cup(run_scenario):
    result = await run_scenario(Scenario(
        name="pick_up_cup",
        prompt="A robot arm near a table with a cup",
        expected_actions=[
            Action(name="move_forward", parameters={"distance": 0.5}),
            Action(name="pick_up", parameters={"object": "cup"}),
        ],
        success_criteria="The robot picks up the cup",
    ))
    # run_scenario auto-asserts result.passed; inspect further if needed
    assert len(result.steps) == 2
```

```bash
uv run pytest tests/ -v
```

#### Provided fixtures

| Fixture | Scope | Description |
|---|---|---|
| `construct_config` | session | `ConstructConfig` loaded from environment |
| `vlm_backend` | session | Default Gemini Robotics ER backend (override for custom VLMs) |
| `evaluators` | function | Default evaluator list (`[ExactPathEvaluator()]`) |
| `scenario_runner` | function | Configured `ScenarioRunner` ready to execute scenarios |
| `run_scenario` | function | Call with a `Scenario`, returns `ScenarioResult`, auto-asserts pass |
| `all_scenarios` | session | All scenarios loaded from `scenarios/` directory |
| `run_all_scenarios` | function | Run every loaded scenario, auto-assert each passes |

## Evaluators

Evaluators score the action sequence produced by the VLM. They implement the `Evaluator` protocol and return an `EvalScore` with a numeric score and pass/fail flag.

| Evaluator | Description |
|---|---|
| `ExactPathEvaluator` | Exact match against an expected action sequence |
| `SemanticEvaluator` | LLM-as-judge semantic similarity scoring |
| `OutcomeEvaluator` | Final-state success check based on `success_criteria` |
| `CompositeEvaluator` | Combine multiple evaluators with AND / OR / threshold logic |

## Custom VLM backends

The VLM backend is a protocol -- swap `GeminiRoboticsBackend` for any class implementing `VLMBackend`:

```python
from construct.vlm import VLMBackend, VLMResponse
from construct.models import Action, Scenario
import numpy as np

class MyBackend:
    async def decide(
        self,
        frame: np.ndarray,
        step_index: int,
        scenario: Scenario,
        action_history: list[Action],
    ) -> VLMResponse:
        # Your logic here
        ...

    async def close(self) -> None:
        ...
```

Override the `vlm_backend` pytest fixture to use it in tests, or pass it directly to `ScenarioRunner` in CLI/script usage.

## Project layout

```
src/construct/
  models.py             Scenario, Action, StepResult, ScenarioResult
  vlm.py                VLMBackend protocol, VLMResponse
  evaluator.py          Evaluator protocol, EvalScore
  config.py             ConstructConfig (env vars / .env)
  session.py            OdysseySession (async context manager, frame sync)
  runner.py             ScenarioRunner (core simulation loop)
  gemini.py             GeminiRoboticsBackend (Gemini Robotics ER)
  frame_utils.py        numpy -> PNG / base64 conversion
  report.py             Console and JSON reporting
  viewer.py             Local web UI for browsing run outputs
  loader.py             Scenario file / directory loader
  cli.py                CLI entry point (run, view)
  pytest_plugin.py      pytest fixtures (auto-registered)
  evaluators/
    exact.py            ExactPathEvaluator
    semantic.py         SemanticEvaluator
    outcome.py          OutcomeEvaluator
    composite.py        CompositeEvaluator
tests/                  Self-tests using mock VLM + mocked Odyssey
scenarios/              Scenario definition files
```

## Development

```bash
uv sync --extra dev      # install dev + test deps
uv run pytest tests/ -v  # run tests
```
