"""One versioned prompt file, preserving system/user message boundaries."""

from pathlib import Path

from northcua.agents.desktop.prompt.builders import ObservationPromptBuilder, SystemPromptBuilder
from northcua.core.enums import CoordinateSpace

PROMPT_SHA256 = "492d89f7f229cc2549ed81ed3068030a3c4726b94905f4527f39b5faf31e9ec1"


def prompt_path() -> Path:
    return Path(__file__).with_name("prompt.j2")


class ProfileSystemBuilder(SystemPromptBuilder):
    def __init__(self) -> None:
        super().__init__(prompt_path())
        self.environment.globals["section"] = "system"

    def render(self, screenshot_width, screenshot_height, *, coordinate_policy, **kwargs) -> str:
        # This is the historical training-aligned virtual coordinate canvas.
        # Images retain their original bytes; dispatch scales exactly once.
        if coordinate_policy.input_space == CoordinateSpace.NORMALIZED_1000:
            screenshot_width = screenshot_height = 1000
        return super().render(screenshot_width, screenshot_height, coordinate_policy=coordinate_policy, **kwargs)


class ProfileObservationBuilder(ObservationPromptBuilder):
    def __init__(self) -> None:
        super().__init__(prompt_path(), prompt_path())

    def render_persistent(self, task: str) -> str:
        return (
            self.environment.from_string(prompt_path().read_text(encoding="utf-8"))
            .render(section="persistent", task=task)
            .strip()
        )

    def render_observation(self, **kwargs) -> str:
        return (
            self.environment.from_string(prompt_path().read_text(encoding="utf-8"))
            .render(section="observation", **kwargs)
            .strip()
        )
