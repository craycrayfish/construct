"""OdysseySession — async context manager wrapping the Odyssey client lifecycle."""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType

import numpy as np
from odyssey import Odyssey, VideoFrame

from construct.config import ConstructConfig

logger = logging.getLogger(__name__)


class OdysseySession:
    """Manages Odyssey connect/stream/disconnect lifecycle with frame sync.

    Usage::

        async with OdysseySession(config) as session:
            await session.start_stream(prompt, portrait=True)
            frame = await session.wait_for_frame()
            await session.interact("pick_up object=cup")
            frame = await session.wait_for_frame()
            await session.end_stream()
    """

    def __init__(self, config: ConstructConfig) -> None:
        self._config = config
        self._client = Odyssey(api_key=config.odyssey_api_key)
        self._frame: np.ndarray | None = None
        self._frame_event = asyncio.Event()
        self._connected = False

    # -- async context manager --

    async def __aenter__(self) -> OdysseySession:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.disconnect()

    # -- lifecycle --

    async def connect(self) -> None:
        await self._client.connect(
            on_video_frame=self._on_video_frame,
            on_connected=lambda: logger.info("Odyssey connected"),
            on_disconnected=lambda: logger.info("Odyssey disconnected"),
        )
        self._connected = True

    async def disconnect(self) -> None:
        if self._connected:
            await self._client.disconnect()
            self._connected = False

    # -- streaming --

    async def start_stream(
        self,
        prompt: str,
        *,
        portrait: bool = True,
        image: str | bytes | np.ndarray | None = None,
    ) -> str:
        stream_id = await self._client.start_stream(
            prompt=prompt,
            portrait=portrait,
            image=image,
        )
        return stream_id

    async def end_stream(self) -> None:
        await self._client.end_stream()

    # -- interaction with frame sync --

    async def interact(self, prompt: str) -> None:
        """Send an interaction and clear the frame event so wait_for_frame()
        will block until a *new* post-interaction frame arrives."""
        self._frame_event.clear()
        await self._client.interact(prompt)

    async def wait_for_frame(self, timeout: float | None = None) -> np.ndarray:
        """Wait until a new frame arrives and return it as an RGB numpy array."""
        await asyncio.wait_for(self._frame_event.wait(), timeout=timeout)
        assert self._frame is not None
        return self._frame

    # -- internal callback --

    def _on_video_frame(self, vf: VideoFrame) -> None:
        self._frame = vf.data
        self._frame_event.set()
