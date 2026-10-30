"""Tests for the standalone device worker data plane."""

from __future__ import annotations

import asyncio

import pytest

from northcua.core.coordinates import (
    CoordinatePolicy,
    CoordinateTarget,
    coordinate_policy_context_values,
    resolve_coordinate_policy,
)
from northcua.core.enums import CoordinateSpace
from northcua.runtime.mcp.desktop_native_server import DesktopNativeServerState
from northcua_device.config import DeviceClientConfig
from northcua_device.executor import CommandExecutor
from northcua_device.runtime.browser import BrowserRuntimeAdapter
from northcua_device.runtime.desktop import DesktopRuntimeAdapter
from northcua_device.runtime.shell import EchoRuntimeAdapter
from northcua_device.session_actor import SessionActor
from northcua_device.worker import DeviceWorker
from northcua_protocol.capabilities import Capability, PlatformKind, ToolDescriptor, ToolRiskLevel
from northcua_protocol.payloads import AGENTKIT_SETTINGS_PARAM, CommandRequest, CommandResult, SessionLeaseGranted
from northcua_protocol.state import CommandState, DeviceState
from northcua_runtime_shared import (
    RuntimeHealth,
    desktop_tool_contracts,
    desktop_tool_registrations,
    validate_command_against_tool,
)
from northcua_runtime_shared.tool_contracts import DirectHandlerBinding, define_tool


def _executor(allowed_tools: set[str] | None = None) -> CommandExecutor:
    return CommandExecutor(
        device_id="dev-1",
        platform=PlatformKind.MACOS,
        adapters=[EchoRuntimeAdapter()],
        allowed_tools=allowed_tools or {"device.echo"},
    )


def test_device_config_precedence_from_overrides(tmp_path) -> None:
    config_path = tmp_path / "device.yaml"
    config_path.write_text("device_id: yaml-device\nserver_url: ws://yaml\n", encoding="utf-8")

    config = DeviceClientConfig.from_sources(
        config_path=str(config_path),
        overrides={"device_id": "override-device", "server_url": "ws://override"},
    )

    assert config.device_id == "override-device"
    assert config.server_url == "ws://override"


def test_device_config_loads_telemetry_stream_options(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "device.yaml"
    config_path.write_text(
        "\n".join(
            [
                "telemetry_stream_mode: raw",
                "telemetry_sample_rate: 0.75",
                "telemetry_always_events:",
                "  - device_driver_call_finished",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NORTHCUA_DEVICE_TELEMETRY_STREAM_MODE", "sampled")
    monkeypatch.setenv("NORTHCUA_DEVICE_TELEMETRY_SAMPLE_RATE", "2")
    monkeypatch.setenv("NORTHCUA_DEVICE_TELEMETRY_ALWAYS_EVENTS", "command_started,command_result_ready")

    config = DeviceClientConfig.from_sources(config_path=str(config_path))

    assert config.telemetry_stream_mode == "terminal_sampled"
    assert config.telemetry_sample_rate == 1.0
    assert config.telemetry_always_events == {"command_started", "command_result_ready"}


def test_executor_enforces_tool_allowlist() -> None:
    async def _run() -> None:
        result = await _executor(allowed_tools={"other.tool"}).execute(
            CommandRequest(
                command_id="cmd-1",
                idempotency_key="idem",
                session_id="sess",
                lease_id="lease",
                device_id="dev-1",
                tool_id="device.echo",
            )
        )
        assert result.status == CommandState.FAILED
        assert "allowlisted" in (result.error or "")

    asyncio.run(_run())


def test_executor_preserves_runtime_side_channel_after_schema_validation() -> None:
    class _SideChannelRuntimeAdapter:
        def __init__(self) -> None:
            self.seen_parameters: dict[str, object] = {}
            self._contract = define_tool(
                tool_id="side.open",
                capability=Capability.BROWSER_PAGE,
                binding=DirectHandlerBinding(handler_path="tests.device.test_device_worker:_side_open"),
                json_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
            )

        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.BROWSER_PAGE}

        @property
        def tool_contracts(self):
            return [self._contract]

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [self._contract.to_descriptor()]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth()

        async def execute(self, command: CommandRequest) -> CommandResult:
            self.seen_parameters = dict(command.parameters)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                lease_generation=command.lease_generation,
                status=CommandState.SUCCEEDED,
                result={"ok": True},
            )

    adapter = _SideChannelRuntimeAdapter()
    executor = CommandExecutor(
        device_id="dev-1",
        platform=PlatformKind.BROWSER,
        adapters=[adapter],
        allowed_tools={"side.open"},
    )

    async def _run() -> None:
        result = await executor.execute(
            CommandRequest(
                command_id="cmd-side",
                idempotency_key="idem-side",
                session_id="sess-side",
                lease_id="lease-side",
                device_id="dev-1",
                tool_id="side.open",
                parameters={
                    "url": "https://example.com",
                    AGENTKIT_SETTINGS_PARAM: {"allow_private_network_urls": True, "headed": True},
                },
            )
        )
        assert result.status == CommandState.SUCCEEDED

    asyncio.run(_run())

    assert adapter.seen_parameters == {
        "url": "https://example.com",
        AGENTKIT_SETTINGS_PARAM: {"allow_private_network_urls": True, "headed": True},
    }


def test_executor_rejects_unsafe_runtime_side_channel_before_adapter_execute() -> None:
    class _SideChannelRuntimeAdapter:
        def __init__(self) -> None:
            self.called = False
            self._contract = define_tool(
                tool_id="side.open",
                capability=Capability.BROWSER_PAGE,
                binding=DirectHandlerBinding(handler_path="tests.device.test_device_worker:_side_open"),
                json_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
            )

        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.BROWSER_PAGE}

        @property
        def tool_contracts(self):
            return [self._contract]

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [self._contract.to_descriptor()]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth()

        async def execute(self, command: CommandRequest) -> CommandResult:
            del command
            self.called = True
            raise AssertionError("unsafe side channel must fail before adapter execution")

    adapter = _SideChannelRuntimeAdapter()
    executor = CommandExecutor(
        device_id="dev-1",
        platform=PlatformKind.BROWSER,
        adapters=[adapter],
        allowed_tools={"side.open"},
    )

    async def _run() -> CommandResult:
        return await executor.execute(
            CommandRequest(
                command_id="cmd-side-unsafe",
                idempotency_key="idem-side-unsafe",
                session_id="sess-side",
                lease_id="lease",
                device_id="dev-1",
                tool_id="side.open",
                parameters={
                    "url": "https://example.com",
                    AGENTKIT_SETTINGS_PARAM: {
                        "playwright_browser": "chromium",
                        "unsafe_backend_binary": "/tmp/browser",
                    },
                },
            )
        )

    result = asyncio.run(_run())

    assert result.status == CommandState.FAILED
    assert "unsafe_backend_binary" in (result.error or "")
    assert adapter.called is False


def test_executor_descriptor_only_advertises_allowlisted_tools() -> None:
    executor = CommandExecutor(
        device_id="dev-1",
        platform=PlatformKind.MACOS,
        adapters=[EchoRuntimeAdapter(), BrowserRuntimeAdapter()],
        allowed_tools={"device.echo"},
    )

    descriptor = executor.descriptor(
        display_name="dev-1",
        max_sessions=1,
        max_gui_sessions=1,
        gui_resource_id="local-test-gui",
        max_concurrent_commands=1,
        supports_parallel_sessions=False,
        requires_foreground_focus=True,
    )

    assert [tool.tool_id for tool in descriptor.tools] == ["device.echo"]
    assert descriptor.capabilities == {Capability.DESKTOP_SCREENSHOT}


def test_executor_descriptor_only_advertises_allowlisted_capabilities() -> None:
    executor = CommandExecutor(
        device_id="dev-1",
        platform=PlatformKind.MACOS,
        adapters=[EchoRuntimeAdapter(), BrowserRuntimeAdapter()],
        allowed_tools=set(),
        allowed_capabilities={Capability.BROWSER_PAGE},
    )

    descriptor = executor.descriptor(
        display_name="dev-1",
        max_sessions=1,
        max_gui_sessions=1,
        gui_resource_id="local-test-gui",
        max_concurrent_commands=1,
        supports_parallel_sessions=False,
        requires_foreground_focus=True,
    )

    assert "device.echo" not in {tool.tool_id for tool in descriptor.tools}
    assert descriptor.capabilities == {Capability.BROWSER_PAGE}


def test_executor_rejects_duplicate_tool_ids_across_adapters() -> None:
    with pytest.raises(ValueError, match="Duplicate device tool_id"):
        CommandExecutor(
            device_id="dev-1",
            platform=PlatformKind.MACOS,
            adapters=[EchoRuntimeAdapter(), EchoRuntimeAdapter()],
            allowed_tools=set(),
        )


def test_browser_platform_worker_advertises_agentkit_browser_tools() -> None:
    worker = DeviceWorker(
        DeviceClientConfig(
            server_url="ws://controller",
            device_id="browser-dev",
            platform=PlatformKind.BROWSER,
            agentkit_agent_kinds={"web"},
        )
    )

    descriptor = worker.descriptor()

    tool_ids = {tool.tool_id for tool in descriptor.tools}
    assert "web_snapshot" in tool_ids
    assert "web.web_scroll" in tool_ids
    assert "web_scroll" in tool_ids
    assert descriptor.capabilities == {Capability.BROWSER_PAGE}


def test_browser_platform_worker_requires_agentkit_agent_kind_opt_in() -> None:
    worker = DeviceWorker(
        DeviceClientConfig(
            server_url="ws://controller",
            device_id="browser-dev",
            platform=PlatformKind.BROWSER,
        )
    )

    descriptor = worker.descriptor()

    assert {tool.tool_id for tool in descriptor.tools} == set()
    assert descriptor.capabilities == set()


def test_linux_web_worker_advertises_browser_page_tools_when_agentkit_web_enabled() -> None:
    worker = DeviceWorker(
        DeviceClientConfig(
            server_url="ws://controller",
            device_id="linux-web-dev",
            platform=PlatformKind.LINUX,
            agentkit_agent_kinds={"web"},
            allowed_capabilities={Capability.BROWSER_PAGE},
        )
    )

    descriptor = worker.descriptor()
    tool_ids = {tool.tool_id for tool in descriptor.tools}

    assert "web_snapshot" in tool_ids
    assert "web.web_scroll" in tool_ids
    assert "click" not in tool_ids
    assert descriptor.capabilities == {Capability.BROWSER_PAGE}


def test_non_macos_desktop_worker_does_not_advertise_cua_tools() -> None:
    worker = DeviceWorker(
        DeviceClientConfig(
            server_url="ws://controller",
            device_id="linux-desktop",
            platform=PlatformKind.LINUX,
        )
    )

    tool_ids = {tool.tool_id for tool in worker.descriptor().tools}

    assert "click" in tool_ids
    assert not {tool_id for tool_id in tool_ids if tool_id.startswith("cua_")}


def test_desktop_worker_advertises_readonly_shell_when_readonly_tool_is_allowlisted() -> None:
    worker = DeviceWorker(
        DeviceClientConfig(
            server_url="ws://controller",
            device_id="linux-shell",
            platform=PlatformKind.LINUX,
            allowed_tools={"shell.exec_readonly"},
        )
    )

    tool_ids = {tool.tool_id for tool in worker.descriptor().tools}

    assert "shell.exec_readonly" in tool_ids
    assert "shell.exec" not in tool_ids


def test_desktop_runtime_adapter_applies_remote_coordinate_metadata() -> None:
    click = next(contract for contract in desktop_tool_contracts() if contract.tool_id == "click")

    class _FakeServer:
        async def call_tool(self, tool_name: str, parameters: dict[str, object]) -> dict[str, object]:
            policy = DesktopNativeServerState().coordinate_policy
            return {
                "tool_name": tool_name,
                "parameters": parameters,
                "coordinate_space": policy.input_space.value,
            }

    async def _run() -> None:
        DesktopNativeServerState().reset()
        adapter = DesktopRuntimeAdapter(
            servers_by_name={"DesktopNativeActionExecutor": _FakeServer()},
            tool_contracts=[click],
            tool_server_names={"click": "DesktopNativeActionExecutor"},
            apply_command_metadata=_apply_desktop_metadata_for_test,
        )
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-click",
                idempotency_key="idem-click",
                session_id="sess-click",
                lease_id="lease-click",
                device_id="dev-1",
                tool_id="click",
                parameters={"x": 500, "y": 500},
                metadata=coordinate_policy_context_values(
                    CoordinateTarget.DESKTOP,
                    CoordinatePolicy(input_space=CoordinateSpace.NORMALIZED_1000),
                ),
            )
        )

        assert result.status == CommandState.SUCCEEDED
        assert isinstance(result.result, dict)
        assert result.result["coordinate_space"] == "normalized_1000"
        DesktopNativeServerState().reset()

    asyncio.run(_run())


def test_desktop_runtime_adapter_resets_coordinate_policy_without_metadata(monkeypatch) -> None:
    monkeypatch.delenv("NORTHCUA_DESKTOP_COORDINATE_SPACE", raising=False)
    monkeypatch.delenv("NORTHCUA_DESKTOP_COORDINATE_CLAMP_TO_BOUNDS", raising=False)
    click = next(contract for contract in desktop_tool_contracts() if contract.tool_id == "click")

    class _FakeServer:
        async def call_tool(self, tool_name: str, parameters: dict[str, object]) -> dict[str, object]:
            policy = DesktopNativeServerState().coordinate_policy
            return {
                "tool_name": tool_name,
                "parameters": parameters,
                "coordinate_space": policy.input_space.value,
            }

    async def _run() -> None:
        DesktopNativeServerState().set_coordinate_policy(CoordinatePolicy(input_space=CoordinateSpace.NORMALIZED_1000))
        adapter = DesktopRuntimeAdapter(
            servers_by_name={"DesktopNativeActionExecutor": _FakeServer()},
            tool_contracts=[click],
            tool_server_names={"click": "DesktopNativeActionExecutor"},
            apply_command_metadata=_apply_desktop_metadata_for_test,
        )
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-click",
                idempotency_key="idem-click",
                session_id="sess-click",
                lease_id="lease-click",
                device_id="dev-1",
                tool_id="click",
                parameters={"x": 10, "y": 20},
            )
        )

        assert result.status == CommandState.SUCCEEDED
        assert isinstance(result.result, dict)
        assert result.result["coordinate_space"] == "image_pixels"
        DesktopNativeServerState().reset()

    asyncio.run(_run())


def _apply_desktop_metadata_for_test(metadata: dict[str, object]) -> None:
    policy = resolve_coordinate_policy(
        CoordinateTarget.DESKTOP,
        context_values={str(key): str(value) for key, value in metadata.items()},
        fallback_policy=CoordinatePolicy(input_space=CoordinateSpace.IMAGE_PIXELS),
    )
    DesktopNativeServerState().set_coordinate_policy(policy)


def test_session_actor_serializes_and_deduplicates_commands() -> None:
    async def _run() -> None:
        lease = SessionLeaseGranted(
            lease_id="lease-1",
            session_id="sess-1",
            task_id="task-1",
            device_id="dev-1",
            trace_id="trace-1",
            lease_generation=2,
        )
        actor = SessionActor(lease, _executor())
        command = CommandRequest(
            command_id="cmd-1",
            idempotency_key="idem",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            parameters={"hello": "world"},
            lease_generation=2,
        )

        first = await actor.execute(command)
        second = await actor.execute(command)

        assert first.status == CommandState.SUCCEEDED
        assert first.lease_generation == 2
        assert second == first

        conflict = await actor.execute(command.model_copy(update={"parameters": {"hello": "other"}}))
        assert conflict.status == CommandState.FAILED
        assert conflict.error == "COMMAND_ID_CONFLICT"
        assert conflict.lease_generation == 2

    asyncio.run(_run())


def test_device_actor_drain_rejects_new_leases() -> None:
    from northcua_device.device_actor import DeviceActor

    actor = DeviceActor(device_id="dev-1", executor=_executor(), max_sessions=1, max_concurrent_commands=1)
    lease = SessionLeaseGranted(
        lease_id="lease-1",
        session_id="sess-1",
        task_id="task-1",
        device_id="dev-1",
        trace_id="trace-1",
    )
    actor.start_draining()

    try:
        actor.accept_lease(lease)
    except RuntimeError as exc:
        assert "draining" in str(exc)
    else:
        raise AssertionError("draining worker should reject new leases")

    assert actor.state == DeviceState.DRAINING


def test_session_actor_rejects_stale_lease_generation() -> None:
    async def _run() -> None:
        lease = SessionLeaseGranted(
            lease_id="lease-1",
            session_id="sess-1",
            task_id="task-1",
            device_id="dev-1",
            trace_id="trace-1",
            lease_generation=2,
        )
        actor = SessionActor(lease, _executor())
        result = await actor.execute(
            CommandRequest(
                command_id="cmd-stale",
                idempotency_key="idem",
                session_id="sess-1",
                lease_id="lease-1",
                device_id="dev-1",
                tool_id="device.echo",
                lease_generation=1,
            )
        )
        assert result.status == CommandState.FAILED
        assert result.error == "LEASE_EXPIRED"
        assert result.lease_generation == 1

    asyncio.run(_run())


def test_desktop_adapter_rejects_invalid_inline_artifact() -> None:
    with pytest.raises(ValueError, match="Invalid base64|Only base64|Excess data|Discontinuous padding"):
        DesktopRuntimeAdapter._decode_inline_artifact("@@@")
