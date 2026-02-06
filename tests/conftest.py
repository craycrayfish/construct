"""Shared test fixtures — mock VLM backend and mocked Odyssey session."""

from __future__ import annotations

from collections.abc import AsyncGenerator
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
def mock_odyssey_session(dummy_frame):
    """Patches OdysseySession to avoid real network calls.

    Returns frames from a list, advancing on each interact() call.
    """

    def _factory(frames: list[np.ndarray] | None = None):
        frame_list = frames or [dummy_frame, dummy_frame, dummy_frame]
        frame_idx = [0]

        session = AsyncMock()
        session.connect = AsyncMock()
        session.disconnect = AsyncMock()
        session.start_stream = AsyncMock(return_value="mock-stream-id")
        session.end_stream = AsyncMock()

        async def _wait_for_frame(timeout=None):
            idx = min(frame_idx[0], len(frame_list) - 1)
            return frame_list[idx]

        session.wait_for_frame = AsyncMock(side_effect=_wait_for_frame)

        async def _interact(prompt):
            frame_idx[0] += 1

        session.interact = AsyncMock(side_effect=_interact)

        # Make it work as an async context manager
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        return session

    return _factory
