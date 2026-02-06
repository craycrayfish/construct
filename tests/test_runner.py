"""Tests for ScenarioRunner with mock VLM and mocked Odyssey."""

from unittest.mock import patch

import pytest

from construct.models import Action, Scenario, TerminationReason
from construct.runner import ScenarioRunner
from construct.vlm import VLMResponse


async def test_runner_basic_loop(construct_config, mock_vlm, mock_odyssey_session):
    """Runner should execute the VLM loop and collect steps."""
    responses = [
        VLMResponse(action=Action(name="move_forward", parameters={"distance": 1.0}), done=False),
        VLMResponse(action=Action(name="pick_up", parameters={"object": "cup"}), done=False),
        VLMResponse(action=Action(name="done"), done=True),
    ]
    vlm = mock_vlm(responses)
    session = mock_odyssey_session()

    scenario = Scenario(name="basic", prompt="pick up the cup")

    with patch("construct.runner.OdysseySession", return_value=session):
        runner = ScenarioRunner(config=construct_config, vlm=vlm)
        result = await runner.run(scenario)

    assert result.termination_reason == TerminationReason.DONE
    assert len(result.steps) == 3
    assert result.steps[0].action.name == "move_forward"
    assert result.steps[2].action.name == "done"


async def test_runner_max_steps(construct_config, mock_vlm, mock_odyssey_session):
    """Runner should stop after max_steps if VLM never signals done."""
    responses = [
        VLMResponse(action=Action(name="wander"), done=False)
        for _ in range(10)
    ]
    vlm = mock_vlm(responses)
    session = mock_odyssey_session()

    scenario = Scenario(name="wanderer", prompt="wander around", max_steps=3)

    with patch("construct.runner.OdysseySession", return_value=session):
        runner = ScenarioRunner(config=construct_config, vlm=vlm)
        result = await runner.run(scenario)

    assert result.termination_reason == TerminationReason.MAX_STEPS
    assert len(result.steps) == 3


async def test_runner_with_evaluator(construct_config, mock_vlm, mock_odyssey_session):
    """Runner should run evaluators and populate eval_scores."""
    responses = [
        VLMResponse(action=Action(name="approach_bed", parameters={"side": "left"}), done=False),
        VLMResponse(action=Action(name="done"), done=True),
    ]
    vlm = mock_vlm(responses)
    session = mock_odyssey_session()

    scenario = Scenario(
        name="with_eval",
        prompt="approach the bed",
        expected_actions=[
            Action(name="approach_bed", parameters={"side": "left"}),
        ],
    )

    from construct.evaluators.exact import ExactPathEvaluator

    # Only the first action (non-done) should match since ExactPathEvaluator
    # compares against expected. We have 2 actual actions vs 1 expected.
    # Let's adjust to match exactly.
    responses_exact = [
        VLMResponse(action=Action(name="approach_bed", parameters={"side": "left"}), done=True),
    ]
    vlm = mock_vlm(responses_exact)

    with patch("construct.runner.OdysseySession", return_value=session):
        runner = ScenarioRunner(
            config=construct_config,
            vlm=vlm,
            evaluators=[ExactPathEvaluator()],
        )
        result = await runner.run(scenario)

    assert "exact_path" in result.eval_scores
    assert result.eval_scores["exact_path"]["passed"] is True
    assert result.passed is True
