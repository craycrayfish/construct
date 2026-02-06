"""ScenarioRunner — core simulation loop."""

from __future__ import annotations

import asyncio
import logging
import time

from construct.config import ConstructConfig
from construct.evaluator import Evaluator
from construct.models import (
    Scenario,
    ScenarioResult,
    StepResult,
    TerminationReason,
)
from construct.session import OdysseySession
from construct.vlm import VLMBackend

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """Runs a scenario through the Odyssey ↔ VLM loop and evaluates the result."""

    def __init__(
        self,
        config: ConstructConfig,
        vlm: VLMBackend,
        evaluators: list[Evaluator] | None = None,
    ) -> None:
        self._config = config
        self._vlm = vlm
        self._evaluators = evaluators or []

    async def run(self, scenario: Scenario, on_step=None) -> ScenarioResult:
        result = ScenarioResult(scenario=scenario)
        t0 = time.monotonic()

        try:
            async with OdysseySession(self._config) as session:
                await session.start_stream(
                    prompt=scenario.prompt,
                    portrait=scenario.portrait,
                    image=scenario.image,
                )

                # Wait for the first frame before entering the loop
                frame = await session.wait_for_frame(timeout=scenario.timeout_s)

                for step_idx in range(scenario.max_steps):
                    elapsed = time.monotonic() - t0
                    remaining = scenario.timeout_s - elapsed
                    if remaining <= 0:
                        result.termination_reason = TerminationReason.TIMEOUT
                        break

                    vlm_resp = await asyncio.wait_for(
                        self._vlm.decide(
                            frame=frame,
                            step_index=step_idx,
                            scenario=scenario,
                            action_history=result.actions,
                        ),
                        timeout=remaining,
                    )

                    step = StepResult(
                        step_index=step_idx,
                        action=vlm_resp.action,
                        reasoning=vlm_resp.reasoning,
                        latency_ms=vlm_resp.latency_ms,
                        cost_usd=vlm_resp.cost_usd,
                    )
                    result.steps.append(step)
                    result.total_latency_ms += vlm_resp.latency_ms
                    result.total_cost_usd += vlm_resp.cost_usd

                    if on_step is not None:
                        on_step(step_idx, step, frame)

                    if vlm_resp.done:
                        result.termination_reason = TerminationReason.DONE
                        break

                    # Send action to Odyssey and wait for the next frame
                    await session.interact(vlm_resp.action.to_interact_prompt())
                    frame = await session.wait_for_frame(timeout=remaining)
                else:
                    result.termination_reason = TerminationReason.MAX_STEPS

                await session.end_stream()

        except asyncio.TimeoutError:
            result.termination_reason = TerminationReason.TIMEOUT
        except Exception as exc:
            result.termination_reason = TerminationReason.ERROR
            result.error = str(exc)
            logger.exception("Scenario %s failed", scenario.name)

        # Run evaluators
        for evaluator in self._evaluators:
            try:
                score = await evaluator.evaluate(scenario, result)
                result.eval_scores[score.name] = {
                    "score": score.score,
                    "passed": score.passed,
                    "details": score.details,
                }
            except Exception:
                logger.exception("Evaluator %s failed for %s", evaluator.name, scenario.name)

        return result
