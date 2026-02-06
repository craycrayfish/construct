"""Tests for ScenarioRunner with mock VLM and simulate-based Odyssey mocks."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from construct.models import Action, Scenario, TerminationReason
from construct.runner import ScenarioRunner
from construct.vlm import VLMResponse


# ---------------------------------------------------------------------------
# Existing tests (rewritten for simulate API)
# ---------------------------------------------------------------------------


async def test_runner_basic_loop(construct_config, mock_vlm, mock_simulate):
    """Runner should execute the VLM loop and collect steps."""
    responses = [
        VLMResponse(action=Action(name="move_forward", parameters={"distance": 1.0}), done=False),
        VLMResponse(action=Action(name="pick_up", parameters={"object": "cup"}), done=False),
        VLMResponse(action=Action(name="done"), done=True),
    ]
    vlm = mock_vlm(responses)
    patches = mock_simulate()

    scenario = Scenario(name="basic", prompt="pick up the cup")

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        runner = ScenarioRunner(config=construct_config, vlm=vlm)
        result = await runner.run(scenario)

    assert result.termination_reason == TerminationReason.DONE
    assert len(result.steps) == 3
    assert result.steps[0].action.name == "move_forward"
    assert result.steps[2].action.name == "done"


async def test_runner_max_steps(construct_config, mock_vlm, mock_simulate):
    """Runner should stop after max_steps if VLM never signals done."""
    responses = [
        VLMResponse(action=Action(name="wander"), done=False)
        for _ in range(10)
    ]
    vlm = mock_vlm(responses)
    patches = mock_simulate()

    scenario = Scenario(name="wanderer", prompt="wander around", max_steps=3)

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        runner = ScenarioRunner(config=construct_config, vlm=vlm)
        result = await runner.run(scenario)

    assert result.termination_reason == TerminationReason.MAX_STEPS
    assert len(result.steps) == 3


async def test_runner_with_evaluator(construct_config, mock_vlm, mock_simulate):
    """Runner should run evaluators and populate eval_scores."""
    responses = [
        VLMResponse(action=Action(name="approach_bed", parameters={"side": "left"}), done=True),
    ]
    vlm = mock_vlm(responses)
    patches = mock_simulate()

    scenario = Scenario(
        name="with_eval",
        prompt="approach the bed",
        expected_actions=[
            Action(name="approach_bed", parameters={"side": "left"}),
        ],
    )

    from construct.evaluators.exact import ExactPathEvaluator

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        runner = ScenarioRunner(
            config=construct_config,
            vlm=vlm,
            evaluators=[ExactPathEvaluator()],
        )
        result = await runner.run(scenario)

    assert "exact_path" in result.eval_scores
    assert result.eval_scores["exact_path"]["passed"] is True
    assert result.passed is True


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------


async def test_prompt_accumulation(construct_config, mock_vlm, mock_simulate):
    """Verify prompts build correctly across steps using the default builder."""
    responses = [
        VLMResponse(action=Action(name="move_forward", parameters={"distance": 1.0}), done=False),
        VLMResponse(action=Action(name="pick_up", parameters={"object": "cup"}), done=False),
        VLMResponse(action=Action(name="done"), done=True),
    ]
    vlm = mock_vlm(responses)
    patches = mock_simulate()

    scenario = Scenario(name="prompt_accum", prompt="pick up the cup")

    with ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in patches]
        odyssey_cls_mock = mocks[0]
        runner = ScenarioRunner(config=construct_config, vlm=vlm)
        result = await runner.run(scenario)

    # The Odyssey mock instance is what simulate is called on
    client_mock = odyssey_cls_mock.return_value
    calls = client_mock.simulate.call_args_list

    # Scene 0: original prompt
    assert calls[0].kwargs["script"][0]["start"]["prompt"] == "pick up the cup"

    # Scene 1: original + first action
    scene1_prompt = calls[1].kwargs["script"][0]["start"]["prompt"]
    assert "pick up the cup" in scene1_prompt
    assert "move_forward distance=1.0" in scene1_prompt

    # Scene 2: accumulated + second action
    scene2_prompt = calls[2].kwargs["script"][0]["start"]["prompt"]
    assert "pick_up object=cup" in scene2_prompt


async def test_custom_prompt_builder(construct_config, mock_vlm, mock_simulate):
    """Verify a custom prompt_builder function is called."""
    responses = [
        VLMResponse(action=Action(name="wave"), done=False),
        VLMResponse(action=Action(name="done"), done=True),
    ]
    vlm = mock_vlm(responses)
    patches = mock_simulate()

    def custom_builder(prev: str, action: Action) -> str:
        return f"CUSTOM: {action.name}"

    scenario = Scenario(name="custom_pb", prompt="original")

    with ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in patches]
        odyssey_cls_mock = mocks[0]
        runner = ScenarioRunner(config=construct_config, vlm=vlm, prompt_builder=custom_builder)
        result = await runner.run(scenario)

    client_mock = odyssey_cls_mock.return_value
    calls = client_mock.simulate.call_args_list

    # Scene 0: original prompt
    assert calls[0].kwargs["script"][0]["start"]["prompt"] == "original"
    # Scene 1: custom builder output
    assert calls[1].kwargs["script"][0]["start"]["prompt"] == "CUSTOM: wave"


async def test_simulate_job_failure(construct_config, mock_vlm):
    """A FAILED simulate job should result in TerminationReason.ERROR."""
    responses = [VLMResponse(action=Action(name="done"), done=True)]
    vlm = mock_vlm(responses)

    from odyssey import SimulationJobStatus

    failed_job = MagicMock()
    failed_job.job_id = "job-fail"
    failed_job.status = SimulationJobStatus.FAILED
    failed_job.streams = []
    failed_job.error_message = "GPU out of memory"

    scenario = Scenario(name="fail_test", prompt="do something")

    with (
        patch("construct.runner.Odyssey", return_value=MagicMock(
            simulate=AsyncMock(return_value=failed_job),
            get_simulate_status=AsyncMock(return_value=failed_job),
        )),
        patch("construct.runner.download_video"),
        patch("construct.runner.extract_last_frame"),
    ):
        runner = ScenarioRunner(config=construct_config, vlm=vlm)
        result = await runner.run(scenario)

    assert result.termination_reason == TerminationReason.ERROR
    assert "GPU out of memory" in result.error


async def test_simulate_timeout(construct_config, mock_vlm, mock_simulate, dummy_frame):
    """If the scenario timeout is exceeded, termination_reason should be TIMEOUT."""
    responses = [
        VLMResponse(action=Action(name="slow"), done=False, latency_ms=0)
        for _ in range(10)
    ]
    vlm = mock_vlm(responses)
    patches = mock_simulate()

    # Very short timeout — the loop should hit it
    scenario = Scenario(name="timeout_test", prompt="hurry", max_steps=10, timeout_s=0.001)

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        runner = ScenarioRunner(config=construct_config, vlm=vlm, poll_interval_s=0.0)
        result = await runner.run(scenario)

    assert result.termination_reason == TerminationReason.TIMEOUT
