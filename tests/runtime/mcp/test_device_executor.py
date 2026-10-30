from __future__ import annotations

import asyncio

import pytest

from northcua.core.coordinates import CoordinatePolicy, CoordinateTarget, coordinate_policy_context_values
from northcua.core.enums import CoordinateSpace, ResultStatus
from northcua.core.protocol import CommandRequestPayload
from northcua.runtime.mcp.builtin_servers import ensure_builtin_mcp_servers_registered
from northcua.runtime.mcp.desktop_native_server import DesktopNativeServerState
from northcua.runtime.mcp.device_executor import NorthCUADeviceCommandExecutor
from northcua.runtime.mcp.device_shell_server import DEVICE_SHELL_EXECUTOR_NAME
from northcua.runtime.mcp.registry import NorthCUAMCPRegistry, RegisteredMCPServer
from northcua_protocol.capabilities import ToolRiskLevel
from northcua_runtime_shared.tool_contracts import McpBinding, define_tool


class _FakeServer:
    def __init__(self, result, *tool_names: str):
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.tool_names = tool_names

    async def list_tools(self):
        return [type("_Tool", (), {"name": tool_name})() for tool_name in self.tool_names]

    async def call_tool(self, tool_name: str, parameters: dict[str, object]):
        self.calls.append((tool_name, parameters))
        return self.result


@pytest.fixture
def rejected_action_server(monkeypatch: pytest.MonkeyPatch) -> _FakeServer:
    """Keep pre-dispatch rejection checks independent of a physical display."""
    server = _FakeServer({"unexpected": "dispatch"}, "scroll", "type")

    def build(server_name: str, *, context=None):
        assert server_name == "DesktopNativeActionExecutor"
        return server

    monkeypatch.setattr(NorthCUAMCPRegistry, "build", staticmethod(build))
    return server


def test_device_executor_routes_requests_through_mcp_servers() -> None:
    ensure_builtin_mcp_servers_registered()
    executor = NorthCUADeviceCommandExecutor.__new__(NorthCUADeviceCommandExecutor)
    executor.data_server = _FakeServer({"screenshot_url": "data:image/png;base64,AAAA"}, "capture_desktop_screenshot")
    executor.action_server = _FakeServer({"text": "Clicked"}, "click", "screenshot")
    executor.cli_server = _FakeServer("Executed command: open -a Calculator", "exec")
    executor.browser_server = _FakeServer({"text": "Hovered"}, "browser_hover")
    executor._servers_by_registry_name = {
        "DesktopNativeDataExecutor": executor.data_server,
        "DesktopNativeActionExecutor": executor.action_server,
        "CommandLineExecutor": executor.cli_server,
        "PlaywrightBrowserExecutor": executor.browser_server,
    }
    executor._tool_server_by_name = None

    payload = CommandRequestPayload(
        session_id="sess-1",
        round_num=1,
        round_step=0,
        commands=[
            {
                "call_id": "call-data",
                "server_name": "DesktopNativeDataExecutor",
                "tool_name": "capture_desktop_screenshot",
                "tool_type": "data",
                "parameters": {"all_screens": False},
            },
            {
                "call_id": "call-action",
                "tool_name": "click",
                "tool_type": "action",
                "parameters": {"x": 1, "y": 2},
            },
            {
                "call_id": "call-screenshot",
                "tool_name": "screenshot",
                "tool_type": "action",
                "parameters": {"all_screens": False},
            },
            {
                "call_id": "call-cli",
                "server_name": "CommandLineExecutor",
                "tool_name": "exec",
                "tool_type": "action",
                "parameters": {"bash_command": "open -a Calculator"},
            },
            {
                "call_id": "call-browser",
                "tool_name": "browser_hover",
                "tool_type": "action",
                "parameters": {"ref": "e2"},
            },
        ],
    )

    result = asyncio.run(executor.execute_command_request(payload))

    assert result.status == ResultStatus.SUCCESS
    assert executor.data_server.calls == [("capture_desktop_screenshot", {"all_screens": False})]
    assert executor.action_server.calls == [
        ("click", {"x": 1.0, "y": 2.0, "button": "left"}),
        ("screenshot", {"all_screens": False}),
    ]
    assert executor.cli_server.calls == [("exec", {"bash_command": "open -a Calculator"})]
    assert executor.browser_server.calls == [("browser_hover", {"ref": "e2"})]
    assert result.results[0].result == {"screenshot_url": "data:image/png;base64,AAAA"}
    assert result.results[1].result == {"text": "Clicked"}
    assert result.results[2].result == {"text": "Clicked"}
    assert result.results[3].result == "Executed command: open -a Calculator"
    assert result.results[4].result == {"text": "Hovered"}


def test_device_executor_applies_command_payload_coordinate_policy() -> None:
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    executor = NorthCUADeviceCommandExecutor.__new__(NorthCUADeviceCommandExecutor)
    executor.data_server = _FakeServer({"text": "Captured"}, "capture_desktop_screenshot")
    executor.action_server = _FakeServer({"text": "Clicked"}, "click")
    executor.cli_server = _FakeServer("", "exec")
    executor.browser_server = _FakeServer({}, "browser_hover")
    executor._servers_by_registry_name = {
        "DesktopNativeDataExecutor": executor.data_server,
        "DesktopNativeActionExecutor": executor.action_server,
    }
    executor._tool_server_by_name = None

    payload = CommandRequestPayload(
        session_id="sess-1",
        round_num=1,
        round_step=1,
        metadata=coordinate_policy_context_values(
            CoordinateTarget.DESKTOP,
            CoordinatePolicy(input_space=CoordinateSpace.NORMALIZED_1000),
        ),
        commands=[
            {
                "call_id": "call-click",
                "tool_name": "click",
                "tool_type": "action",
                "parameters": {"x": 500, "y": 500},
            }
        ],
    )

    result = asyncio.run(executor.execute_command_request(payload))

    assert result.status == ResultStatus.SUCCESS
    assert DesktopNativeServerState().coordinate_policy.input_space == CoordinateSpace.NORMALIZED_1000
    DesktopNativeServerState().reset()


def test_device_executor_validates_schema_before_mcp_call(rejected_action_server: _FakeServer) -> None:
    executor = NorthCUADeviceCommandExecutor(allowed_server_names={"DesktopNativeActionExecutor"})

    result = asyncio.run(
        executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess-invalid",
                round_num=1,
                round_step=0,
                commands=[
                    {
                        "call_id": "call-invalid",
                        "server_name": "DesktopNativeActionExecutor",
                        "tool_name": "scroll",
                        "tool_type": "action",
                        "parameters": {"direction": "diagonal", "amount": 3},
                    }
                ],
            )
        )
    )

    assert result.status == ResultStatus.FAILURE
    assert "Extra inputs are not permitted" in (result.results[0].error or "")
    assert rejected_action_server.calls == []


def test_device_executor_routes_public_name_to_contract_and_mcp_tool_name() -> None:
    server_name = "PublicNameDeviceExecutorTest"
    contract = define_tool(
        tool_id="canonical_echo",
        public_name="public_echo",
        description="Echo through a public provider name.",
        json_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        side_effect=False,
        readonly=True,
        binding=McpBinding(server_name=server_name),
    )
    try:
        NorthCUAMCPRegistry.register(
            server_name,
            lambda _context: _FakeServer({"ok": True}, "public_echo"),
            tool_contracts=[contract],
        )
        fake_server = _FakeServer({"ok": True}, "public_echo")
        executor = NorthCUADeviceCommandExecutor.__new__(NorthCUADeviceCommandExecutor)
        executor._allowed_server_names = {server_name}
        executor._max_allowed_risk = ToolRiskLevel.HIGH
        executor._servers_by_registry_name = {server_name: fake_server}
        executor._tool_server_by_name = None
        executor._contract_by_server_tool = {}
        executor._mcp_tool_name_by_server_tool = {}
        executor._server_name_by_tool_name = {}
        executor._ambiguous_tool_names = set()
        executor._index_registered_tool_contracts()

        result = asyncio.run(
            executor.execute_command_request(
                CommandRequestPayload(
                    session_id="sess-public-name",
                    round_num=1,
                    round_step=0,
                    commands=[
                        {
                            "call_id": "call-public-name",
                            "server_name": server_name,
                            "tool_name": "public_echo",
                            "tool_type": "action",
                            "parameters": {"message": "hello"},
                        }
                    ],
                )
            )
        )

        assert result.status == ResultStatus.SUCCESS
        assert fake_server.calls == [("public_echo", {"message": "hello"})]
    finally:
        NorthCUAMCPRegistry._registry.pop(server_name, None)


def test_device_executor_rejects_same_server_runtime_name_collision() -> None:
    server_name = "RuntimeNameCollisionDeviceExecutorTest"
    first = define_tool(
        tool_id="first_runtime_name",
        public_name="first_public_name",
        aliases=("shared_runtime_name",),
        json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        binding=McpBinding(server_name=server_name),
    )
    second = define_tool(
        tool_id="second_runtime_name",
        public_name="second_public_name",
        aliases=("shared_runtime_name",),
        json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        binding=McpBinding(server_name=server_name),
    )
    try:
        NorthCUAMCPRegistry.register(
            server_name,
            lambda _context: _FakeServer({"ok": True}, "first_public_name", "second_public_name"),
            tool_contracts=[first, second],
        )
        executor = NorthCUADeviceCommandExecutor.__new__(NorthCUADeviceCommandExecutor)
        executor._servers_by_registry_name = {server_name: _FakeServer({"ok": True})}
        executor._allowed_server_names = {server_name}
        executor._max_allowed_risk = ToolRiskLevel.HIGH
        executor._contract_by_server_tool = {}
        executor._mcp_tool_name_by_server_tool = {}
        executor._server_name_by_tool_name = {}
        executor._ambiguous_tool_names = set()

        with pytest.raises(ValueError, match="Duplicate device-side runtime tool name"):
            executor._index_registered_tool_contracts()
    finally:
        NorthCUAMCPRegistry._registry.pop(server_name, None)


def test_device_executor_enforces_max_allowed_risk_before_mcp_call(rejected_action_server: _FakeServer) -> None:
    executor = NorthCUADeviceCommandExecutor(
        allowed_server_names={"DesktopNativeActionExecutor"},
        max_allowed_risk=ToolRiskLevel.MEDIUM,
    )

    result = asyncio.run(
        executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess-risk",
                round_num=1,
                round_step=0,
                commands=[
                    {
                        "call_id": "call-risk",
                        "server_name": "DesktopNativeActionExecutor",
                        "tool_name": "type",
                        "tool_type": "action",
                        "parameters": {"text": "danger"},
                    }
                ],
            )
        )
    )

    assert result.status == ResultStatus.FAILURE
    assert "risk level 'high' exceeds allowed maximum 'medium'" in (result.results[0].error or "")
    assert rejected_action_server.calls == []


def test_device_executor_routes_readonly_shell_contract_for_evaluator_probe() -> None:
    executor = NorthCUADeviceCommandExecutor(allowed_server_names={DEVICE_SHELL_EXECUTOR_NAME})

    result = asyncio.run(
        executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess-shell",
                round_num=1,
                round_step=0,
                commands=[
                    {
                        "call_id": "call-shell",
                        "server_name": DEVICE_SHELL_EXECUTOR_NAME,
                        "tool_name": "shell.exec_readonly",
                        "tool_type": "client_probe",
                        "parameters": {"cmd": "pwd"},
                    }
                ],
            )
        )
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.results[0].status == ResultStatus.SUCCESS
    assert result.results[0].result["returncode"] == 0
    assert result.results[0].result["stdout"]


def test_device_executor_readonly_shell_rejects_mutating_command() -> None:
    executor = NorthCUADeviceCommandExecutor(allowed_server_names={DEVICE_SHELL_EXECUTOR_NAME})

    result = asyncio.run(
        executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess-shell-reject",
                round_num=1,
                round_step=0,
                commands=[
                    {
                        "call_id": "call-shell-reject",
                        "server_name": DEVICE_SHELL_EXECUTOR_NAME,
                        "tool_name": "shell.exec_readonly",
                        "tool_type": "client_probe",
                        "parameters": {"cmd": "touch /tmp/northcua-should-not-write"},
                    }
                ],
            )
        )
    )

    assert result.status == ResultStatus.FAILURE
    assert "not allowlisted" in (result.results[0].error or "")


def test_device_executor_rejects_registered_tools_without_contract_before_mcp_call() -> None:
    server_name = "NoContractDeviceExecutorTest"
    tool_name = "no_contract_probe"
    factory_calls: list[str] = []

    def _factory(_context):
        factory_calls.append("built")
        return _FakeServer({"ok": True}, tool_name)

    try:
        NorthCUAMCPRegistry._registry[server_name] = RegisteredMCPServer(
            name=server_name,
            factory=_factory,
        )

        executor = NorthCUADeviceCommandExecutor(allowed_server_names={server_name})

        result = asyncio.run(
            executor.execute_command_request(
                CommandRequestPayload(
                    session_id="sess-no-contract",
                    round_num=1,
                    round_step=0,
                    commands=[
                        {
                            "call_id": "call-no-contract",
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "tool_type": "data",
                            "parameters": {},
                        }
                    ],
                )
            )
        )

        assert result.status == ResultStatus.FAILURE
        assert factory_calls == []
        assert "has no ToolContract metadata" in (result.results[0].error or "")
    finally:
        NorthCUAMCPRegistry._registry.pop(server_name, None)


def test_device_executor_does_not_preload_registered_server_without_contract() -> None:
    server_name = "NoContractDeviceExecutorPreloadTest"
    tool_name = "no_contract_preload_probe"
    factory_calls: list[str] = []

    def _factory(_context):
        factory_calls.append("built")
        return _FakeServer({"ok": True}, tool_name)

    try:
        NorthCUAMCPRegistry._registry[server_name] = RegisteredMCPServer(
            name=server_name,
            factory=_factory,
        )

        executor = NorthCUADeviceCommandExecutor(allowed_server_names={server_name})
        result = asyncio.run(
            executor.execute_command_request(
                CommandRequestPayload(
                    session_id="sess-no-contract-preload",
                    round_num=1,
                    round_step=0,
                    commands=[
                        {
                            "call_id": "call-no-contract-preload",
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "tool_type": "data",
                            "parameters": {},
                        }
                    ],
                )
            )
        )

        assert factory_calls == []
        assert result.status == ResultStatus.FAILURE
        assert "has no ToolContract metadata" in (result.results[0].error or "")
    finally:
        NorthCUAMCPRegistry._registry.pop(server_name, None)


def test_device_executor_resets_coordinate_policy_when_payload_metadata_is_absent(monkeypatch) -> None:
    ensure_builtin_mcp_servers_registered()
    monkeypatch.delenv("NORTHCUA_DESKTOP_COORDINATE_SPACE", raising=False)
    monkeypatch.delenv("NORTHCUA_DESKTOP_COORDINATE_CLAMP_TO_BOUNDS", raising=False)
    DesktopNativeServerState().set_coordinate_policy(CoordinatePolicy(input_space=CoordinateSpace.NORMALIZED_1000))
    executor = NorthCUADeviceCommandExecutor.__new__(NorthCUADeviceCommandExecutor)
    executor.data_server = _FakeServer({"text": "Captured"}, "capture_desktop_screenshot")
    executor.action_server = _FakeServer({"text": "Clicked"}, "click")
    executor.cli_server = _FakeServer("", "exec")
    executor.browser_server = _FakeServer({}, "browser_hover")
    executor._servers_by_registry_name = {
        "DesktopNativeActionExecutor": executor.action_server,
    }
    executor._tool_server_by_name = None

    result = asyncio.run(
        executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess-1",
                round_num=1,
                round_step=2,
                commands=[
                    {
                        "call_id": "call-click",
                        "tool_name": "click",
                        "tool_type": "action",
                        "parameters": {"x": 10, "y": 20},
                    }
                ],
            )
        )
    )

    assert result.status == ResultStatus.SUCCESS
    assert DesktopNativeServerState().coordinate_policy.input_space == CoordinateSpace.IMAGE_PIXELS
    DesktopNativeServerState().reset()
