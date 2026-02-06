"""Monitoring scenario — ego-centric hallway approach in a skilled nursing facility."""

from construct import Scenario

PROMPT = """\
Create a photorealistic, physics-accurate video simulation rendered strictly from the \
first-person, ego-centric sensor perspective of a mobile assistive robot inside a skilled \
nursing facility during daytime. Camera & embodiment constraints (hard requirements): \
The camera represents the robot's fixed forward-facing sensor. The robot's full body is \
never visible. Only two robotic forearms and hands are visible at the very bottom edge of \
the frame, partially cropped, persistent, and spatially stable across all frames. No \
third-person, trailing, side, or cinematic camera views.

Scene state (single continuous shot):
- The robot is already oriented forward in a skilled nursing facility hallway.
- The hallway extends ahead toward a closed patient room door on the right, which becomes \
increasingly prominent in the forward field of view.
- The physical hallway remains unchanged throughout the clip.

Environment:
- Handrails on both walls, room number placards, sanitizer dispensers, parked linen or \
medical carts.
- Worn vinyl or linoleum flooring with subtle reflections.
- Overhead fluorescent lighting with neutral institutional tones.

Motion & physics:
- Smooth, slow forward motion with mechanical stabilization.
- No sudden acceleration, no handheld shake.

Lens & tone:
- Wide-angle sensor lens, deep focus.
- Calm, observational, routine.

Negative prompts:
Third-person view, full robot visible, scene cuts, environment changes, cinematic framing\
"""

SCENARIOS = [
    Scenario(
        name="monitoring",
        prompt=PROMPT,
        success_criteria="Robot navigates the hallway smoothly toward the patient room door",
        tags=["monitoring", "navigation", "nursing-facility"],
    ),
]
