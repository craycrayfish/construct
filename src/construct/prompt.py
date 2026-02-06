"""Pluggable prompt-building strategies for clip chaining."""

from __future__ import annotations

from collections.abc import Callable

from construct.models import Action

PromptBuilder = Callable[[str, Action], str]


def default_prompt_builder(previous_prompt: str, action: Action) -> str:
    """Accumulate the previous prompt with the VLM action description."""
    return f"{previous_prompt}\n\n{action.to_interact_prompt()}"
