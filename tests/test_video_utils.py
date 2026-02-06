"""Tests for video_utils — download and frame extraction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pytest

from construct.video_utils import download_video, extract_last_frame


@pytest.fixture()
def tiny_video(tmp_path) -> Path:
    """Create a minimal 3-frame video file using PyAV."""
    path = tmp_path / "test.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=10)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"

    for i in range(3):
        # Each frame has a different red channel value so we can identify the last
        arr = np.full((16, 16, 3), fill_value=i * 80, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return path


class TestExtractLastFrame:
    def test_returns_last_frame(self, tiny_video):
        frame = extract_last_frame(tiny_video)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (16, 16, 3)
        assert frame.dtype == np.uint8
        # The last frame had fill_value=160 — due to lossy compression
        # we check that it's closer to 160 than to 0 or 80
        mean = frame.mean()
        assert mean > 100, f"Expected mean > 100 (last frame), got {mean}"

    def test_nonexistent_video_raises(self, tmp_path):
        path = tmp_path / "missing.mp4"
        with pytest.raises(Exception):
            extract_last_frame(path)


class TestDownloadVideo:
    async def test_downloads_to_dest(self, tmp_path):
        dest = tmp_path / "downloaded.mp4"
        content = b"fake video content"

        def _fake_urlretrieve(url, filename):
            Path(filename).write_bytes(content)

        with patch("construct.video_utils.urllib.request.urlretrieve", side_effect=_fake_urlretrieve):
            result = await download_video("https://example.com/video.mp4", dest)

        assert result == dest
        assert dest.read_bytes() == content

    async def test_creates_parent_directories(self, tmp_path):
        dest = tmp_path / "a" / "b" / "video.mp4"

        def _fake_urlretrieve(url, filename):
            Path(filename).write_bytes(b"data")

        with patch("construct.video_utils.urllib.request.urlretrieve", side_effect=_fake_urlretrieve):
            result = await download_video("https://example.com/video.mp4", dest)

        assert dest.exists()
