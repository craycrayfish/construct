"""Example scenario definition."""

from construct import Action, Scenario

SCENARIOS = [
    Scenario(
        name="pick_up_cup",
        prompt="A robot arm near a table with a red cup",
        expected_actions=[
            Action(name="move_to", parameters={"target": "red_cup"}),
            Action(name="grasp", parameters={"object": "red_cup"}),
            Action(name="lift", parameters={"height_cm": 10}),
        ],
        success_criteria="Robot picks up the red cup from the table",
        tags=["manipulation", "pick-and-place"],
    ),
]
