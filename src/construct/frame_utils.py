"""Utilities for converting numpy frames to PNG bytes and base64."""

from __future__ import annotations

import base64
import io

import numpy as np


def frame_to_png_bytes(frame: np.ndarray) -> bytes:
    """Convert an RGB numpy array (H, W, 3) to PNG bytes."""
    # Import PIL lazily so it's only needed when actually encoding
    from PIL import Image

    img = Image.fromarray(frame, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert an RGB numpy array to a base64-encoded PNG string."""
    return base64.b64encode(frame_to_png_bytes(frame)).decode("ascii")
