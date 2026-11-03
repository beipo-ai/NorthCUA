"""An optional Desktop system prompt; task/observation fields are runtime data."""

from pathlib import Path

from northcua.agents.desktop.prompt.builders import ObservationPromptBuilder

SYSTEM_TEMPLATE_SHA256 = "b2ff0d2ac8bbef4a472250b2fac3cf3edc170fba4514eddf54794b43da298149"


def system_template_path() -> Path:
    return Path(__file__).with_name("system.j2")


class ProfileObservationBuilder(ObservationPromptBuilder):
    """Avoid adding default infeasibility or pixel-coordinate instructions.

    The single system template owns all behavioral instructions. The original
    task is passed verbatim; observation text only labels the attached state.
    """

    def render_persistent(self, task: str) -> str:
        return task

    def render_observation(
        self,
        *,
        platform_name: str,
        total_image_count: int,
        history_image_count: int,
        has_current_image: bool,
        current_image_role: str,
        observation_metadata=None,
    ) -> str:
        return (
            f"Current observation: platform={platform_name}; retained_images={total_image_count}; "
            f"history_images={history_image_count}; current_image={has_current_image}; "
            f"current_image_role={current_image_role}."
        )
