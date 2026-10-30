"""Tests for runtime-shared and controller runner plugin discovery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from northcua.controller.execution.runner_plugins import (
    ControllerRunnerRegistrationContext,
    clear_controller_runner_plugins,
    populate_runner_registry,
)
from northcua.controller.execution.runner_registry import RunnerRegistry
from northcua_runtime_shared import desktop_wire_tool_descriptors
from northcua_runtime_shared.desktop_bridge import (
    build_registered_mcp_server,
    clear_desktop_mcp_bridge,
    is_registered_mcp_server,
    normalize_mcp_tool_result,
)
from northcua_runtime_shared.validation import validate_command_against_tool


class _FakeDesktopBridge:
    """Minimal bridge stub used to exercise runtime-shared plugin loading."""

    def ensure_registered(self) -> None:
        return None

    def build_server(self, *, server_name: str, process_name: str) -> object:
        return {"server_name": server_name, "process_name": process_name}

    def is_registered(self, server_name: str) -> bool:
        return server_name == "DesktopNativeActionExecutor"

    def normalize_tool_result(self, tool_result: object) -> dict[str, object] | str:
        return {"normalized": str(tool_result)}


class _FakeEntryPoint:
    """Tiny stand-in for importlib.metadata entry points."""

    def __init__(self, loaded: object) -> None:
        self._loaded = loaded

    def load(self) -> object:
        return self._loaded


def test_desktop_bridge_discovers_entry_point_plugins(monkeypatch) -> None:
    """Runtime-shared should load the desktop bridge from entry points on demand."""

    clear_desktop_mcp_bridge()
    monkeypatch.setattr(
        "northcua_runtime_shared.desktop_bridge.entry_points",
        lambda *, group: [_FakeEntryPoint(lambda: _FakeDesktopBridge())] if group else [],
    )
    assert is_registered_mcp_server("DesktopNativeActionExecutor") is True
    assert build_registered_mcp_server(server_name="DesktopNativeActionExecutor", process_name="test") == {
        "server_name": "DesktopNativeActionExecutor",
        "process_name": "test",
    }
    assert normalize_mcp_tool_result("ok") == {"normalized": "ok"}
    clear_desktop_mcp_bridge()


def test_controller_runner_plugins_populate_registry(monkeypatch) -> None:
    """Controller runner plugins should populate the execution registry via entry points."""

    clear_controller_runner_plugins()

    def _plugin(runner_registry: RunnerRegistry, sdk: object, context: ControllerRunnerRegistrationContext) -> None:
        del sdk

        async def _runner(execution_context):
            del execution_context
            return context.app_config_path

        runner_registry.register("desktop", _runner)

    monkeypatch.setattr(
        "northcua.controller.execution.runner_plugins.entry_points",
        lambda *, group: [_FakeEntryPoint(_plugin)] if group else [],
    )

    registry = RunnerRegistry()
    populate_runner_registry(
        runner_registry=registry,
        sdk=SimpleNamespace(),
        context=ControllerRunnerRegistrationContext(
            app_config_path="config/app.yaml",
            command_timeout_seconds=30.0,
        ),
    )

    assert registry.registered_agent_kinds() == ("desktop",)
    assert registry.registration_source("desktop").endswith(
        "test_controller_runner_plugins_populate_registry.<locals>._plugin"
    )
    clear_controller_runner_plugins()


def test_descriptor_validation_rejects_nested_desktop_payload_mismatches() -> None:
    """Controller admission should enforce nested schemas before runtime dispatch."""

    descriptors = {descriptor.tool_id: descriptor for descriptor in desktop_wire_tool_descriptors()}

    try:
        validate_command_against_tool(
            tool_id="drag",
            parameters={"path": [{"x": 1, "y": 2}, {"x": 3}]},
            descriptor=descriptors["drag"],
        )
    except ValueError as exc:
        assert "parameters.path[1]" in str(exc)
    else:
        raise AssertionError("drag payload without y coordinate should be rejected")

    try:
        validate_command_against_tool(
            tool_id="keypress",
            parameters={"keys": [1]},
            descriptor=descriptors["keypress"],
        )
    except ValueError as exc:
        assert "parameters.keys[0]" in str(exc)
    else:
        raise AssertionError("keypress keys array should require string items")


def test_keypress_descriptor_accepts_provider_null_default_for_omitted_branch() -> None:
    """Provider payloads may include null defaults for the omitted branch."""

    descriptors = {descriptor.tool_id: descriptor for descriptor in desktop_wire_tool_descriptors()}

    validate_command_against_tool(
        tool_id="keypress",
        parameters={"key": "enter", "keys": None},
        descriptor=descriptors["keypress"],
    )
    validate_command_against_tool(
        tool_id="keypress",
        parameters={"key": None, "keys": ["ctrl+l", "tab"]},
        descriptor=descriptors["keypress"],
    )


def test_keypress_descriptor_rejects_two_non_null_branches() -> None:
    """The keypress schema still enforces exactly one concrete input branch."""

    descriptors = {descriptor.tool_id: descriptor for descriptor in desktop_wire_tool_descriptors()}

    with pytest.raises(ValueError, match="must match exactly one schema"):
        validate_command_against_tool(
            tool_id="keypress",
            parameters={"key": "enter", "keys": ["tab"]},
            descriptor=descriptors["keypress"],
        )
