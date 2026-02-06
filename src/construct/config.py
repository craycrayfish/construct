"""Configuration for construct, sourced from .env file and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ConstructConfig:
    """Central configuration for the construct framework."""

    odyssey_api_key: str = field(default_factory=lambda: os.environ.get("ODYSSEY_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")))
    gemini_model: str = "gemini-robotics-er-1.5-preview"
    default_max_steps: int = 20
    default_timeout_s: float = 120.0
    default_portrait: bool = True
    report_json: bool = False
    report_path: str | None = None
