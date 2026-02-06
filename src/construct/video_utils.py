"""Video utilities — download simulation clips and extract frames."""

from __future__ import annotations

import asyncio
import urllib.request
from pathlib import Path

import av
import numpy as np


async def download_video(url: str, dest: Path) -> Path:
    """Download video from a presigned URL to *dest* (async via thread pool)."""

    def _download() -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return dest

    return await asyncio.to_thread(_download)


def extract_last_frame(video_path: Path) -> np.ndarray:
    """Open *video_path* with PyAV and return the last frame as an RGB numpy array."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    last_frame: av.VideoFrame | None = None
    for frame in container.decode(stream):
        last_frame = frame
    container.close()
    if last_frame is None:
        msg = f"No video frames found in {video_path}"
        raise ValueError(msg)
    return last_frame.to_ndarray(format="rgb24")
