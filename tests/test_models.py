"""Tests for core data models."""

from construct.models import (
    Action,
    Scenario,
    ScenarioResult,
    StepResult,
    TerminationReason,
)


class TestAction:
    def test_to_interact_prompt_with_raw_text(self):
        a = Action(name="move", parameters={"x": 1}, raw_text="move x=1")
        assert a.to_interact_prompt() == "move x=1"

    def test_to_interact_prompt_without_raw_text(self):
        a = Action(name="pick_up", parameters={"object": "cup"})
        assert a.to_interact_prompt() == "pick_up object=cup"

    def test_to_interact_prompt_no_params(self):
        a = Action(name="done")
        assert a.to_interact_prompt() == "done"

    def test_frozen(self):
        a = Action(name="test")
        try:
            a.name = "other"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestScenario:
    def test_defaults(self):
        s = Scenario(name="test", prompt="do something")
        assert s.max_steps == 20
        assert s.timeout_s == 120.0
        assert s.portrait is True
        assert s.expected_actions == []
        assert s.tags == []

    def test_with_actions(self):
        actions = [Action(name="a"), Action(name="b")]
        s = Scenario(name="test", prompt="p", expected_actions=actions)
        assert len(s.expected_actions) == 2


class TestScenarioResult:
    def test_passed_no_scores(self):
        s = Scenario(name="t", prompt="p")
        r = ScenarioResult(scenario=s, termination_reason=TerminationReason.DONE)
        assert r.passed is True

    def test_failed_on_error(self):
        s = Scenario(name="t", prompt="p")
        r = ScenarioResult(scenario=s, termination_reason=TerminationReason.ERROR)
        assert r.passed is False

    def test_passed_with_scores(self):
        s = Scenario(name="t", prompt="p")
        r = ScenarioResult(
            scenario=s,
            eval_scores={"exact": {"score": 1.0, "passed": True}},
        )
        assert r.passed is True

    def test_failed_with_scores(self):
        s = Scenario(name="t", prompt="p")
        r = ScenarioResult(
            scenario=s,
            eval_scores={
                "exact": {"score": 1.0, "passed": True},
                "semantic": {"score": 0.3, "passed": False},
            },
        )
        assert r.passed is False

    def test_actions_property(self):
        s = Scenario(name="t", prompt="p")
        r = ScenarioResult(
            scenario=s,
            steps=[
                StepResult(step_index=0, action=Action(name="a")),
                StepResult(step_index=1, action=Action(name="b")),
            ],
        )
        assert [a.name for a in r.actions] == ["a", "b"]
