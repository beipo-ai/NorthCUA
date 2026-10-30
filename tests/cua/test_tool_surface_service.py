"""Regression coverage for desktop tool surface construction from NexAU agents."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from northcua.agentkernel.tools.surface import (
    ToolSurfaceService,
    active_provider_tool_payload,
    normalize_allowed_tool_names,
)


@pytest.mark.parametrize("value", ["click", ["click", "click"], [" click"], [""]])
def test_allowed_tool_names_metadata_fails_closed(value: object) -> None:
    with pytest.raises(ValueError, match="allowed_tool_names"):
        normalize_allowed_tool_names(value)


def test_tool_surface_service_uses_live_agent_tool_payload() -> None:
    agent = SimpleNamespace(
        tool_call_payload=[
            {
                "name": "click",
                "description": "Click the current desktop.",
                "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}},
            }
        ],
        config=SimpleNamespace(llm_config=SimpleNamespace(api_type="openai_responses")),
    )

    surface = ToolSurfaceService().build_for_agent(agent)

    assert surface.tool_choice == "auto"
    assert surface.tools[0]["name"] == "click"


def test_tool_surface_service_uses_anthropic_tool_payload() -> None:
    agent = SimpleNamespace(
        tool_call_payload=[
            {
                "name": "click",
                "description": "Click the current desktop.",
                "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}},
            }
        ],
        config=SimpleNamespace(llm_config=SimpleNamespace(api_type="anthropic_chat_completion")),
    )

    surface = ToolSurfaceService().build_for_agent(agent)

    assert surface.tool_choice == {"type": "auto"}
    assert surface.tools[0]["name"] == "click"
    assert "input_schema" in surface.tools[0]


def test_tool_surface_service_filters_to_task_allowlist_before_required_choice() -> None:
    installed: list[list[dict[str, object]]] = []
    caller = SimpleNamespace()
    agent = SimpleNamespace(
        tool_call_payload=[
            {"name": name, "description": name, "input_schema": {"type": "object", "properties": {}}}
            for name in ("click", "screenshot", "wait")
        ],
        config=SimpleNamespace(llm_config=SimpleNamespace(api_type="openai_responses")),
        executor=SimpleNamespace(
            update_structured_tools=lambda tools: installed.append(tools),
            llm_caller=caller,
        ),
    )

    surface = ToolSurfaceService().activate_for_agent(
        agent,
        tool_choice_policy="required",
        allowed_tool_names=("click",),
    )

    assert surface.tool_choice == "required"
    assert [tool["name"] for tool in surface.tools] == ["click"]
    assert [[tool["name"] for tool in tools] for tools in installed] == [["click"]]
    assert [tool["name"] for tool in active_provider_tool_payload(agent) or []] == ["click"]
    assert [tool["name"] for tool in active_provider_tool_payload(caller) or []] == ["click"]


def test_tool_surface_service_rejects_allowlist_name_absent_from_runtime() -> None:
    agent = SimpleNamespace(
        tool_call_payload=[
            {"name": "click", "description": "click", "input_schema": {"type": "object", "properties": {}}}
        ],
        config=SimpleNamespace(llm_config=SimpleNamespace(api_type="openai_responses")),
    )

    with pytest.raises(ValueError, match="absent from the runtime surface"):
        ToolSurfaceService().build_for_agent(agent, allowed_tool_names=("browser_click",))
