from __future__ import annotations

import re

from northcua.agents.web.domain.panel_models import WebStepPanelRecord
from northcua.agents.web.domain.results import WebStepStatus
from northcua.app.terminal_renderer import WebAgentTerminalRenderer
from northcua.core.models import AssistantToolCallRecord

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return ANSI_RE.sub("", text)


def test_web_renderer_hides_missing_reasoning_but_shows_assistant_text() -> None:
    renderer = WebAgentTerminalRenderer()
    output = _plain(
        renderer.render_step(
            WebStepPanelRecord(
                step=1,
                status=WebStepStatus.OK,
                url="https://example.com",
                assistant_text="I will inspect the current URL before taking another action.",
                assistant_tool_calls=[AssistantToolCallRecord(id="call-1", name="web_get_url", arguments="{}")],
                observation_summary="Example page",
            )
        )
    )

    assert "Reasoning" not in output
    assert "Assistant" in output
    assert "I will inspect the current URL" in output
    assert "Model call: web_get_url {}" in output


def test_web_renderer_shows_provider_reasoning_when_present() -> None:
    renderer = WebAgentTerminalRenderer()
    output = _plain(
        renderer.render_step(
            WebStepPanelRecord(
                step=1,
                status=WebStepStatus.OK,
                reasoning="Need a fresh snapshot.",
                assistant_text="I will refresh the DOM snapshot.",
            )
        )
    )

    assert "Reasoning" in output
    assert "Need a fresh snapshot." in output
    assert "Assistant" in output
