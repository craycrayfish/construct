# construct

A testing framework for Vision-Language Models (VLMs) controlling robotic systems. It connects [Odyssey](https://github.com/odysseyml/odyssey-python) (world model) to [Gemini Robotics ER](https://deepmind.google/technologies/gemini/robotics/) (VLM under test) in a closed loop:

```
Odyssey generates frame → VLM decides action → action sent to Odyssey → new frame → repeat → evaluate
```

## Install

```bash
uv sync --extra test
```

## Usage

Write pytest tests that define scenarios and assert results:

```python
from construct import Scenario, Action

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
```

Run with:

```bash
uv run pytest tests/
```

## Built-in Evaluators

- **ExactPathEvaluator** — exact action sequence match
- **SemanticEvaluator** — LLM-as-judge semantic similarity
- **OutcomeEvaluator** — final-state success check
- **CompositeEvaluator** — combine evaluators with AND/OR/threshold

## Environment Variables

| Variable | Purpose |
|---|---|
| `ODYSSEY_API_KEY` | Odyssey API key |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini API key |
