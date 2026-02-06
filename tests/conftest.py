"""Shared test fixtures — mock VLM backend and simulate-based Odyssey mocks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from construct.config import ConstructConfig
from construct.models import Action
from construct.vlm import VLMResponse


class MockVLMBackend:
    """A deterministic VLM backend that returns pre-configured actions."""

    def __init__(self, responses: list[VLMResponse] | None = None) -> None:
        self._responses = responses or []
        self._call_count = 0

    async def decide(self, frame, step_index, scenario, action_history):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = VLMResponse(
                action=Action(name="done", parameters={}),
                done=True,
            )
        self._call_count += 1
        return resp

    async def close(self):
        pass


@pytest.fixture()
def mock_vlm():
    """Factory fixture for creating MockVLMBackend with specific responses."""

    def _factory(responses: list[VLMResponse] | None = None) -> MockVLMBackend:
        return MockVLMBackend(responses)

    return _factory


@pytest.fixture()
def construct_config():
    return ConstructConfig(
        odyssey_api_key="test-key",
        gemini_api_key="test-key",
    )


@pytest.fixture()
def dummy_frame():
    """A small dummy RGB frame for testing."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture()
def mock_simulate(dummy_frame):
    """Patch Odyssey.simulate, get_simulate_status, download_video, and extract_last_frame.

    Returns a factory that accepts an optional list of frames (one per clip).
    Each call to simulate returns a completed job with a mock video URL,
    and extract_last_frame returns the corresponding frame from the list.
    """

    def _factory(frames: list[np.ndarray] | None = None):
        frame_list = frames or [dummy_frame] * 10
        clip_idx = [0]

        # Build a fake SimulationJobDetail
        def _make_job(status="completed", error_message=None):
            stream = MagicMock()
            stream.video_url = f"https://mock.video/scene_{clip_idx[0]}.mp4"

            job = MagicMock()
            job.job_id = f"job-{clip_idx[0]}"
            job.status = _status_enum(status)
            job.streams = [stream]
            job.error_message = error_message
            return job

        def _status_enum(status_str):
            from odyssey import SimulationJobStatus
            return SimulationJobStatus(status_str)

        async def _simulate(*, script=None, scripts=None, script_url=None, portrait=True):
            job = _make_job()
            clip_idx[0] += 1
            return job

        async def _get_status(job_id):
            return _make_job()

        def _extract_last_frame(video_path):
            idx = min(clip_idx[0] - 1, len(frame_list) - 1)
            return frame_list[max(0, idx)]

        async def _download_video(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.touch()
            return dest

        patches = (
            patch("construct.runner.Odyssey", return_value=MagicMock(
                simulate=AsyncMock(side_effect=_simulate),
                get_simulate_status=AsyncMock(side_effect=_get_status),
            )),
            patch("construct.runner.download_video", side_effect=_download_video),
            patch("construct.runner.extract_last_frame", side_effect=_extract_last_frame),
        )

        return patches

    return _factory
