from __future__ import annotations

import pytest

from northcua.agentkit import AgentCliCommandSpec, AgentCliSpec, AgentRuntimeSpec, AgentSpec
from northcua.agentkit.registry import get_agentkit_registry, load_all_agent_specs, reset_agentkit_registry_for_tests
from northcua.core.config import NorthCUAAppConfig
from northcua.core.protocol import GenericTaskRequestPayload, parse_task_request_payload
from northcua.submitter import DesktopSubmitSpec
from northcua_protocol.capabilities import Capability, PlatformKind
from northcua_protocol.payloads import AgentKind as ProtocolAgentKind
from northcua_protocol.payloads import TaskRequest
from tests.agentkit.test_agentkit_lowcode import AndroidProbes, AndroidSettings, AndroidTools


def test_decorators_collect_typed_tool_and_probe_specs() -> None:
    tools = AndroidTools.declared_tools()
    probes = AndroidProbes.declared_probes()

    assert tools[0].name == "android_tap"
    assert tools[0].parameters_schema["properties"]["x"]["type"] == "integer"
    assert probes[0].name == "android_screenshot"
    assert probes[0].persist is True


def test_agentkit_registry_accepts_custom_android_without_core_edits() -> None:
    reset_agentkit_registry_for_tests()
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
        cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
        tools=AndroidTools.declared_tools(),
        probes=AndroidProbes.declared_probes(),
    )

    get_agentkit_registry().register(spec)

    assert get_agentkit_registry().get("android") is spec


def test_agentkit_entry_point_discovery_accepts_custom_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_agentkit_registry_for_tests()

    android_spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
        cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
    )

    class _EntryPoint:
        def load(self) -> object:
            return lambda: android_spec

    monkeypatch.setattr("northcua.agentkit.registry.entry_points", lambda *, group: [_EntryPoint()])

    registry = load_all_agent_specs()

    assert registry.maybe_get("android") is android_spec


def test_agentkit_registry_rejects_duplicate_kind() -> None:
    reset_agentkit_registry_for_tests()
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
        cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
    )
    get_agentkit_registry().register(spec)

    with pytest.raises(ValueError, match="already registered"):
        get_agentkit_registry().register(
            AgentSpec(
                runtime=AgentRuntimeSpec(kind="android", display_name="Other Android", platform=PlatformKind.ANDROID),
                cli=AgentCliSpec(agent_name="android2", help="Android CUA commands"),
            )
        )


def test_generic_task_payload_uses_registered_custom_agent() -> None:
    reset_agentkit_registry_for_tests()
    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(
                agent_name="android",
                help="Android CUA commands",
                local_commands=(
                    AgentCliCommandSpec(
                        name="run",
                        help="Run Android CUA",
                        handler="tests.agentkit.test_agentkit_lowcode:_dummy_handler",
                    ),
                ),
            ),
        )
    )

    payload = parse_task_request_payload({"agent_kind": "android", "request": "open settings", "max_steps": 2})

    assert isinstance(payload, GenericTaskRequestPayload)
    assert payload.agent_kind == "android"
    assert payload.max_steps == 2


def test_unknown_task_payload_kind_fails_cleanly() -> None:
    reset_agentkit_registry_for_tests()

    with pytest.raises(ValueError, match="Unknown AgentKit task agent_kind"):
        parse_task_request_payload({"agent_kind": "missing", "request": "noop"})


def test_protocol_agent_kind_constants_stay_compatible_with_custom_strings() -> None:
    request = TaskRequest(
        task_id="task",
        trace_id="trace",
        agent_kind=ProtocolAgentKind.DESKTOP,
        request="hello",
    )
    custom = TaskRequest(
        task_id="task-custom",
        trace_id="trace-custom",
        agent_kind="android",
        request="hello",
    )

    assert request.agent_kind == "desktop"
    assert custom.agent_kind == "android"


def test_generic_app_config_agents_mapping_preserves_custom_settings() -> None:
    config = NorthCUAAppConfig.model_validate(
        {
            "agent_config_path": "config/agent.yaml",
            "agents": {
                "android": {
                    "agent_config_path": "config/agent.android.yaml",
                    "settings": {"adb_binary": "adb"},
                    "enabled": True,
                }
            },
        }
    )

    assert config.agents["android"].agent_config_path == "config/agent.android.yaml"
    assert config.agents["android"].settings == {"adb_binary": "adb"}
    assert config.agents["desktop"].agent_config_path == "config/agent.yaml"


def test_agent_spec_validates_typed_settings() -> None:
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
        cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
        settings_model=AndroidSettings,
    )

    settings = spec.validate_settings({"adb_binary": "adb-custom", "max_steps": 7})

    assert isinstance(settings, AndroidSettings)
    assert settings.adb_binary == "adb-custom"
    assert settings.max_steps == 7


def test_builtin_specs_publish_typed_tool_schemas_and_remote_capabilities() -> None:
    reset_agentkit_registry_for_tests()
    registry = load_all_agent_specs()

    desktop = registry.get("desktop")
    web = registry.get("web")
    desktop_submit = DesktopSubmitSpec(query="Observe the desktop")
    assert desktop_submit.max_steps == 100
    assert desktop_submit.submit_payload()["metadata"]["tool_choice_policy"] == "auto"
    assert (
        DesktopSubmitSpec(query="Act now", tool_choice_policy="required").submit_payload()["metadata"][
            "tool_choice_policy"
        ]
        == "required"
    )
    assert DesktopSubmitSpec(query="Act now", allowed_tool_names=("click",)).submit_payload()["metadata"][
        "allowed_tool_names"
    ] == ["click"]

    assert {
        Capability.DESKTOP_INPUT,
        Capability.DESKTOP_SCREENSHOT,
        Capability.DESKTOP_WINDOW,
    }.issubset(desktop.remote.required_capabilities)
    assert {
        Capability.DESKTOP_INPUT,
        Capability.DESKTOP_SCREENSHOT,
        Capability.DESKTOP_WINDOW,
    }.issubset(desktop_submit.required_capabilities)
    assert web.tools
    assert web.tools[0].parameters_schema["type"] == "object"
    assert any(tool.parameters_schema.get("properties") for tool in web.tools)


def test_builtin_specs_publish_presentation_and_session_adapters() -> None:
    reset_agentkit_registry_for_tests()
    registry = load_all_agent_specs()

    desktop = registry.get("desktop")
    web = registry.get("web")

    assert desktop.presentation.renderer_cls == "northcua.app.terminal_renderer:NorthCUATerminalRenderer"
    assert web.presentation.renderer_cls == "northcua.app.terminal_renderer:WebAgentTerminalRenderer"
    assert web.session.adapter_cls == "northcua.agents.web.session.agentkit_adapter:WebAgentDeviceSessionAdapter"
