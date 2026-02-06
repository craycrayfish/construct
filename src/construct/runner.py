"""ScenarioRunner — core simulation loop using the Odyssey simulate API."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path

from odyssey import Odyssey, SimulationJobStatus

from construct.config import ConstructConfig
from construct.evaluator import Evaluator
from construct.models import (
    Scenario,
    ScenarioResult,
    StepResult,
    TerminationReason,
)
from construct.prompt import PromptBuilder, default_prompt_builder
from construct.video_utils import download_video, extract_last_frame
from construct.vlm import VLMBackend

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """Runs a scenario through simulate → VLM clip-chaining loop and evaluates the result."""

    def __init__(
        self,
        config: ConstructConfig,
        vlm: VLMBackend,
        evaluators: list[Evaluator] | None = None,
        prompt_builder: PromptBuilder | None = None,
        clip_duration_s: float = 5.0,
        poll_interval_s: float = 2.0,
    ) -> None:
        self._config = config
        self._vlm = vlm
        self._evaluators = evaluators or []
        self._prompt_builder = prompt_builder or default_prompt_builder
        self._clip_duration_s = clip_duration_s
        self._poll_interval_s = poll_interval_s

    async def _run_clip(
        self,
        client: Odyssey,
        prompt: str,
        image,
        portrait: bool,
    ):
        """Submit a simulate job, poll until completion, and return the job detail."""
        script = [
            {
                "timestamp_ms": 0,
                "start": {"prompt": prompt, "image": image},
            },
            {
                "timestamp_ms": int(self._clip_duration_s * 1000),
                "end": {},
            },
        ]
        job = await client.simulate(script=script, portrait=portrait)

        while job.status not in (SimulationJobStatus.COMPLETED, SimulationJobStatus.FAILED):
            await asyncio.sleep(self._poll_interval_s)
            job = await client.get_simulate_status(job.job_id)

        if job.status == SimulationJobStatus.FAILED:
            msg = job.error_message or "Simulation job failed"
            raise RuntimeError(msg)

        return job

    async def run(self, scenario: Scenario, on_step=None) -> ScenarioResult:
        result = ScenarioResult(scenario=scenario)
        t0 = time.monotonic()

        try:
            client = Odyssey(api_key=self._config.odyssey_api_key)

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)

                # Scene 0: initial clip from scenario prompt
                current_prompt = scenario.prompt
                job = await self._run_clip(client, current_prompt, scenario.image, scenario.portrait)
                video_path = await download_video(job.streams[0].video_url, tmp / "scene_0.mp4")
                frame = extract_last_frame(video_path)

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

                    # Build next prompt and run next clip
                    current_prompt = self._prompt_builder(current_prompt, vlm_resp.action)
                    job = await self._run_clip(client, current_prompt, frame, scenario.portrait)
                    video_path = await download_video(
                        job.streams[0].video_url, tmp / f"scene_{step_idx + 1}.mp4"
                    )
                    frame = extract_last_frame(video_path)
                else:
                    result.termination_reason = TerminationReason.MAX_STEPS

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
