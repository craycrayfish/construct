"""Gemini Robotics ER backend implementation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from google import genai
from google.genai import types

from construct.config import ConstructConfig
from construct.frame_utils import frame_to_png_bytes
from construct.models import Action, Scenario
from construct.vlm import VLMResponse

# Default robot action vocabulary
DEFAULT_TOOLS = types.Tool(
    function_declarations=[
        {
            "name": "move_forward",
            "description": "Move the robot forward by a specified distance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance": {"type": "number", "description": "Distance in meters"},
                },
            },
        },
        {
            "name": "turn",
            "description": "Turn the robot by a specified angle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {"type": "number", "description": "Angle in degrees (positive=left, negative=right)"},
                },
            },
        },
        {
            "name": "pick_up",
            "description": "Pick up an object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object": {"type": "string", "description": "Name of the object to pick up"},
                },
            },
        },
        {
            "name": "place",
            "description": "Place the held object at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Where to place the object"},
                },
            },
        },
        {
            "name": "done",
            "description": "Signal that the task is complete.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
)


def _build_tools(scenario: Scenario) -> types.Tool:
    """Build Gemini tool definitions from scenario or fall back to defaults."""
    if scenario.tools:
        return types.Tool(function_declarations=scenario.tools)
    return DEFAULT_TOOLS


class GeminiRoboticsBackend:
    """VLMBackend implementation using Gemini Robotics ER."""

    def __init__(self, config: ConstructConfig) -> None:
        self._config = config
        self._client = genai.Client(api_key=config.gemini_api_key or None)
        self._model = config.gemini_model
        self._conversation_history: list[types.Content] = []

    async def decide(
        self,
        frame: np.ndarray,
        step_index: int,
        scenario: Scenario,
        action_history: list[Action],
    ) -> VLMResponse:
        tools = _build_tools(scenario)
        frame_bytes = frame_to_png_bytes(frame)

        # Build the current turn's content
        user_parts: list[Any] = [
            types.Part.from_bytes(data=frame_bytes, mime_type="image/png"),
        ]
        if step_index == 0:
            user_parts.append(f"Task: {scenario.prompt}\nObserve the scene and decide the next robot action.")
        else:
            user_parts.append("Observe the updated scene and decide the next robot action.")

        user_content = types.Content(role="user", parts=user_parts)

        # Build full conversation contents
        contents = list(self._conversation_history) + [user_content]

        config = types.GenerateContentConfig(
            tools=[tools],
            temperature=0.0,
            system_instruction=scenario.vlm_system_prompt,
        )

        t0 = time.monotonic()
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        # Parse function call from response
        candidate = response.candidates[0]
        part = candidate.content.parts[0]

        if part.function_call:
            fc = part.function_call
            action = Action(
                name=fc.name,
                parameters=dict(fc.args) if fc.args else {},
            )
            is_done = fc.name == "done"
            reasoning = ""
        else:
            # If the model returned text instead of a function call
            action = Action(name="unknown", parameters={}, raw_text=part.text or "")
            is_done = False
            reasoning = part.text or ""

        # Update conversation history for multi-turn
        self._conversation_history.append(user_content)
        self._conversation_history.append(candidate.content)

        return VLMResponse(
            action=action,
            done=is_done,
            reasoning=reasoning,
            latency_ms=latency_ms,
        )

    async def close(self) -> None:
        self._conversation_history.clear()
