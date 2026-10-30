from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest

from northcua.agentkit import AgentCliSpec, AgentRemoteSpec, AgentRuntimeSpec, AgentSessionSpec, AgentSpec, ToolUseRef
from northcua.agentkit.controller_dispatch import command_result_record
from northcua.agentkit.controller_runner import AgentKitRemoteRunnerConfig, register_controller_runner
from northcua.agentkit.device_adapter import AgentKitMCPRuntimeAdapter
from northcua.agentkit.device_artifacts import AgentKitCommandArtifactExtractor
from northcua.agentkit.external_tools import (
    command_records_from_pending_tool_calls,
    tool_execution_records_from_command_results,
)
from northcua.agentkit.registry import get_agentkit_registry, reset_agentkit_registry_for_tests
from northcua.agentkit.session_context import AGENTKIT_SETTINGS_PARAM
from northcua.agents.web.backend import WebActionCommandResult
from northcua.agents.web.session import RegisteredWebAgentSession, WebAgentSessionRegistry
from northcua.agents.web.session.agentkit_adapter import WebAgentDeviceSessionAdapter
from northcua.controller.execution.executor_protocol import ControllerExecutionContext
from northcua.controller.execution.runner_plugins import ControllerRunnerRegistrationContext
from northcua.controller.execution.runner_registry import RunnerRegistry
from northcua.core.enums import ResultStatus
from northcua.core.models import CommandExecutionResultRecord
from northcua_controller_plugins.agentkit_web_remote_dispatch import RemoteWebActionBackend
from northcua_device.runtime.desktop import DesktopRuntimeAdapter
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef
from northcua_protocol.capabilities import Capability, PlatformKind, ToolDescriptor, ToolRiskLevel
from northcua_protocol.payloads import CommandRequest, CommandResult, SessionLeaseGranted, TaskRequest
from northcua_protocol.state import CommandState
from northcua_runtime_shared.desktop_surface import desktop_tool_contracts
from northcua_runtime_shared.tool_contracts import McpBinding, ToolPlacement, define_tool
from tests.agentkit.test_agentkit_lowcode import AndroidProbes, AndroidSessionAdapter, AndroidTools


@pytest.mark.parametrize(
    "status,error",
    [
        (CommandState.TIMED_OUT, "COMMAND_TIMED_OUT"),
        (CommandState.CANCELLED, "COMMAND_DISPATCH_CANCELLED"),
        (CommandState.FAILED, "COMMAND_TRANSPORT_FAILED"),
    ],
)
def test_shared_and_web_error_conversion_keep_existing_return_contract(status, error):
    async def run():
        result = CommandResult(
            command_id="call", session_id="session", lease_id="lease", device_id="device", status=status, error=error
        )
        record = command_result_record("call", result)
        assert record.status == ResultStatus.FAILURE
        assert record.error == error

        class Dispatcher:
            async def dispatch(self, **kwargs):
                return result

        output = await RemoteWebActionBackend(Dispatcher()).click_ref("1")
        assert output.success is False
        assert output.error == error

    asyncio.run(run())


def test_agentkit_device_adapter_executes_lowcode_handler() -> None:
    reset_agentkit_registry_for_tests()
    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
            tools=AndroidTools.declared_tools(),
            probes=AndroidProbes.declared_probes(),
        )
    )

    async def _run() -> None:
        adapter = AgentKitMCPRuntimeAdapter(agent_kinds={"android"})
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-android",
                idempotency_key="idem",
                session_id="sess-android",
                lease_id="lease",
                device_id="dev-android",
                tool_id="android_tap",
                parameters={"x": 10, "y": 20},
            )
        )

        assert result.status == CommandState.SUCCEEDED
        assert result.result == {"ok": True, "x": 10, "y": 20}

    asyncio.run(_run())


def test_agentkit_external_command_records_preserve_per_call_server_route() -> None:
    commands = command_records_from_pending_tool_calls(
        pending_tool_calls=[
            {
                "id": "call-one",
                "name": "first_tool",
                "server_name": "FirstServer",
                "input": {"value": 1},
            },
            {
                "id": "call-two",
                "name": "second_tool",
                "server_name": "SecondServer",
                "input": {"value": 2},
            },
        ],
        server_name="FallbackServer",
        round_num=3,
        round_step=4,
    )

    assert [(command.server_name, command.tool_name, command.parameters) for command in commands] == [
        ("FirstServer", "first_tool", {"value": 1}),
        ("SecondServer", "second_tool", {"value": 2}),
    ]


def test_agentkit_device_adapter_enforces_contract_risk_gate_before_handler_call() -> None:
    reset_agentkit_registry_for_tests()
    calls: list[str] = []

    def _dangerous_handler() -> dict[str, object]:
        calls.append("called")
        return {"ok": True}

    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android-risk", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(agent_name="android-risk", help="Android CUA commands"),
            tools=(
                ToolUseRef.from_contract(
                    define_tool(
                        tool_id="android_dangerous",
                        json_schema={"type": "object", "properties": {}, "additionalProperties": False},
                        placement=ToolPlacement.LOCAL_HANDLER,
                        capability=Capability.DESKTOP_INPUT,
                        risk_level=ToolRiskLevel.HIGH,
                    ),
                    handler=_dangerous_handler,
                ),
            ),
        )
    )

    async def _run() -> None:
        adapter = AgentKitMCPRuntimeAdapter(agent_kinds={"android-risk"}, max_allowed_risk=ToolRiskLevel.MEDIUM)
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-android-risk",
                idempotency_key="idem-risk",
                session_id="sess-android-risk",
                lease_id="lease",
                device_id="dev-android",
                tool_id="android_dangerous",
                parameters={},
            )
        )

        assert result.status == CommandState.FAILED
        assert "risk level 'high' exceeds allowed maximum 'medium'" in (result.error or "")

    asyncio.run(_run())
    assert calls == []


def test_agentkit_device_adapter_runs_session_adapter_for_handler_tools(tmp_path: Path) -> None:
    reset_agentkit_registry_for_tests()
    AndroidSessionAdapter.calls = []
    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
            session=AgentSessionSpec(adapter_cls=AndroidSessionAdapter),
            tools=AndroidTools.declared_tools(),
        )
    )

    async def _run() -> None:
        adapter = AgentKitMCPRuntimeAdapter(agent_kinds={"android"}, artifact_root=tmp_path)
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-android-session",
                idempotency_key="idem-session",
                session_id="sess-android-session",
                lease_id="lease",
                device_id="dev-android",
                tool_id="android_tap",
                parameters={"x": 1, "y": 2},
            )
        )

        assert result.status == CommandState.SUCCEEDED

    asyncio.run(_run())

    assert AndroidSessionAdapter.calls == ["sess-android-session"]


def test_agentkit_device_adapter_executes_model_visible_web_action_on_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_agentkit_registry_for_tests()
    captured: dict[str, object] = {}

    class _FakeWebBackend:
        async def open(self, url: str) -> WebActionCommandResult:
            captured["url"] = url
            return WebActionCommandResult(success=True, data={"url": url})

        async def get_url(self) -> WebActionCommandResult:
            return WebActionCommandResult(success=True, data={"url": str(captured.get("url") or "")})

    def _fake_build_playwright_web_backend(*, config: object, session_id: str) -> _FakeWebBackend:
        captured["config"] = config
        captured["session_id"] = session_id
        return _FakeWebBackend()

    monkeypatch.setattr(
        "northcua.agents.web.session.agentkit_adapter.build_playwright_web_backend",
        _fake_build_playwright_web_backend,
    )
    registry = WebAgentSessionRegistry()
    registry.clear()

    async def _run() -> None:
        adapter = AgentKitMCPRuntimeAdapter(agent_kinds={"web"}, artifact_root=tmp_path)
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-web-open",
                idempotency_key="idem-web-open",
                session_id="sess-web-open",
                lease_id="lease",
                device_id="dev",
                tool_id="web.web_open",
                parameters={
                    "url": "http://127.0.0.1:8123",
                    AGENTKIT_SETTINGS_PARAM: {
                        "headed": True,
                        "snapshot_eval_timeout_seconds": 11.0,
                        "post_action_observation_mode": "none",
                        "persist_post_action_observation": False,
                    },
                },
            )
        )
        assert result.status == CommandState.SUCCEEDED
        assert isinstance(result.result, dict)
        assert result.result["status"] == "ok"
        assert result.result["action"] == "open"

    try:
        asyncio.run(_run())
        assert captured["url"] == "http://127.0.0.1:8123"
        assert captured["session_id"] == "sess-web-open"
        config = captured["config"]
        assert getattr(config, "headed") is True
        assert getattr(config, "snapshot_eval_timeout_seconds") == 11.0
    finally:
        registry.clear()


def test_agentkit_device_adapter_returns_retryable_web_validation_tool_error(tmp_path: Path) -> None:
    reset_agentkit_registry_for_tests()
    registry = WebAgentSessionRegistry()
    registry.clear()

    async def _run() -> None:
        adapter = AgentKitMCPRuntimeAdapter(agent_kinds={"web"}, artifact_root=tmp_path)
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-web-bad-wait",
                idempotency_key="idem-web-bad-wait",
                session_id="sess-web-bad-wait",
                lease_id="lease",
                device_id="dev",
                tool_id="web.web_wait",
                parameters={
                    "duration_ms": 1000,
                    "load_state": "load",
                },
            )
        )
        assert result.status == CommandState.SUCCEEDED
        assert isinstance(result.result, dict)
        assert result.result["status"] == "error"
        assert result.result["action"] == "wait"
        assert result.result["code"] == "invalid_wait_target_combination"
        assert result.result["reason_code"] == "invalid_wait_target_combination"
        assert result.result["retryable"] is True
        assert result.result["received_targets"] == ["duration_ms", "load_state"]
        assert result.result["feedback_source"] == "contract_validator"
        assert "duration_ms, load_state" in str(result.result["hint"])
        with pytest.raises(LookupError):
            registry.get_registered_session("sess-web-bad-wait")

    try:
        asyncio.run(_run())
    finally:
        registry.clear()


def test_agentkit_device_adapter_strips_session_settings_before_server_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_agentkit_registry_for_tests()
    AndroidSessionAdapter.calls = []
    fake_server_calls: list[dict[str, object]] = []

    class _FakeServer:
        async def call_tool(self, tool_name: str, parameters: dict[str, object]) -> dict[str, object]:
            fake_server_calls.append({"tool_name": tool_name, "parameters": parameters})
            return {"ok": True, "parameters": parameters}

    monkeypatch.setattr(
        "northcua.agentkit.device_adapter.NorthCUAMCPRegistry.build",
        lambda *args, **kwargs: _FakeServer(),
    )
    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android-server", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(agent_name="android-server", help="Android CUA commands"),
            session=AgentSessionSpec(adapter_cls=AndroidSessionAdapter),
            tools=(
                ToolUseRef.from_contract(
                    define_tool(
                        tool_id="android_server_tap",
                        json_schema={
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                            "required": ["x"],
                            "additionalProperties": False,
                        },
                        capability=Capability.DESKTOP_INPUT,
                        binding=McpBinding(server_name="AndroidServerExecutor"),
                    )
                ),
            ),
        )
    )

    async def _run() -> None:
        adapter = AgentKitMCPRuntimeAdapter(agent_kinds={"android-server"}, artifact_root=tmp_path)
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-android-server",
                idempotency_key="idem-server",
                session_id="sess-android-server",
                lease_id="lease",
                device_id="dev-android",
                tool_id="android-server.android_server_tap",
                parameters={"x": 3, AGENTKIT_SETTINGS_PARAM: {"adb_binary": "adb-custom"}},
            )
        )

        assert result.status == CommandState.SUCCEEDED

    asyncio.run(_run())

    assert AndroidSessionAdapter.calls == ["sess-android-server"]
    assert fake_server_calls == [{"tool_name": "android_server_tap", "parameters": {"x": 3}}]


def test_agentkit_device_adapter_healthcheck_accepts_lowcode_handler() -> None:
    reset_agentkit_registry_for_tests()
    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
            tools=AndroidTools.declared_tools(),
        )
    )

    async def _run() -> None:
        adapter = AgentKitMCPRuntimeAdapter(agent_kinds={"android"})
        health = await adapter.healthcheck()
        assert health.healthy is True

    asyncio.run(_run())


def test_command_result_record_rehydrates_inline_image_artifacts() -> None:
    artifact = ArtifactRef.from_inline_bytes(
        artifact_id="cmd-result-data_url",
        kind=ArtifactKind.SCREENSHOT,
        data=b"fake-png",
        mime_type="image/png",
    )

    record = command_result_record(
        "call-1",
        CommandResult(
            command_id="cmd",
            session_id="sess",
            lease_id="lease",
            device_id="dev",
            status=CommandState.SUCCEEDED,
            result={"result": {"data_url": "", "data_url_artifact_id": artifact.artifact_id}},
            artifacts=[artifact],
        ),
    )

    assert isinstance(record.result, dict)
    result = record.result["result"]
    assert isinstance(result, dict)
    assert result["data_url"].startswith("data:image/png;base64,")
    assert result["data_url_artifact_sha256"] == artifact.sha256


def test_command_result_record_rehydrates_canonical_artifact_ids_by_original_id() -> None:
    artifact = ArtifactRef.from_inline_bytes(
        artifact_id="cmd-result-screenshot_url",
        kind=ArtifactKind.SCREENSHOT,
        data=b"fake-png",
        mime_type="image/png",
    )
    canonical = artifact.model_copy(update={"artifact_id": "controller-catalog-shot"})

    record = command_result_record(
        "call-1",
        CommandResult(
            command_id="cmd",
            session_id="sess",
            lease_id="lease",
            device_id="dev",
            status=CommandState.SUCCEEDED,
            result={
                "screenshot_url": "",
                "screenshot_url_artifact_id": "controller-catalog-shot",
                "screenshot_url_artifact_id_original": "cmd-result-screenshot_url",
            },
            artifacts=[canonical],
        ),
    )

    assert isinstance(record.result, dict)
    assert record.result["screenshot_url"].startswith("data:image/png;base64,")
    assert record.result["screenshot_url_artifact_sha256"] == artifact.sha256


def test_command_result_record_rehydrates_artifact_only_screenshot_payload() -> None:
    """Remote commands that return only an image artifact still match local screenshot payloads."""

    artifact = ArtifactRef.from_inline_bytes(
        artifact_id="cmd-result-screenshot",
        kind=ArtifactKind.SCREENSHOT,
        data=b"fake-png",
        mime_type="image/png",
    )

    record = command_result_record(
        "call-screenshot",
        CommandResult(
            command_id="cmd",
            session_id="sess",
            lease_id="lease",
            device_id="dev",
            status=CommandState.SUCCEEDED,
            result=None,
            artifacts=[artifact],
        ),
    )

    assert isinstance(record.result, dict)
    assert record.result["screenshot_url"].startswith("data:image/png;base64,")


def test_desktop_runtime_adapter_externalizes_worker_local_screenshot_paths(tmp_path: Path) -> None:
    """Remote desktop tools should preserve local action-result screenshot semantics."""

    screenshot = tmp_path / "action.png"
    screenshot.write_bytes(b"fake-png")

    class _FakeServer:
        async def call_tool(self, tool_name: str, parameters: dict[str, object]) -> dict[str, object]:
            assert tool_name == "click"
            assert parameters == {"x": 1.0, "y": 2.0, "button": "left"}
            return {
                "text": "Clicked",
                "screenshot_path": str(screenshot),
                "screenshot_format": "local_path",
            }

    click_contract = next(contract for contract in desktop_tool_contracts() if contract.tool_id == "click")
    adapter = DesktopRuntimeAdapter(
        servers_by_name={"DesktopNativeActionExecutor": _FakeServer()},
        tool_contracts=[click_contract],
        normalize_result=lambda result: dict(result) if isinstance(result, dict) else str(result),
    )

    async def _run() -> CommandResult:
        return await adapter.execute(
            CommandRequest(
                command_id="cmd-click",
                idempotency_key="idem",
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                tool_id="click",
                parameters={"x": 1, "y": 2},
            )
        )

    result = asyncio.run(_run())

    assert result.status == CommandState.SUCCEEDED
    assert isinstance(result.result, dict)
    assert result.result["text"] == "Clicked"
    assert result.result["screenshot_path"] == str(screenshot)
    assert result.result["screenshot_url"] == ""
    assert result.result["screenshot_url_artifact_id"] == "cmd-click-screenshot_url"
    assert len(result.artifacts) == 1

    record = command_result_record("call-click", result)

    assert isinstance(record.result, dict)
    assert record.result["screenshot_path"] == str(screenshot)
    assert record.result["screenshot_url"].startswith("data:image/png;base64,")


def test_agentkit_artifact_extractor_externalizes_worker_local_screenshot_paths(tmp_path: Path) -> None:
    """Generic AgentKit remote tools should not degrade screenshots to text-only payloads."""

    screenshot = tmp_path / "agentkit-shot.png"
    screenshot.write_bytes(b"fake-png")

    normalized, artifacts = AgentKitCommandArtifactExtractor().extract(
        "cmd-agentkit-shot",
        {"text": "captured", "screenshot_path": str(screenshot)},
    )

    assert isinstance(normalized, dict)
    assert normalized["text"] == "captured"
    assert normalized["screenshot_path"] == str(screenshot)
    assert normalized["screenshot_url"] == ""
    assert normalized["screenshot_url_artifact_id"] == "cmd-agentkit-shot-screenshot_url"
    assert len(artifacts) == 1

    record = command_result_record(
        "call-agentkit-shot",
        CommandResult(
            command_id="cmd-agentkit-shot",
            session_id="sess",
            lease_id="lease",
            device_id="dev",
            status=CommandState.SUCCEEDED,
            result=normalized,
            artifacts=artifacts,
        ),
    )

    assert isinstance(record.result, dict)
    assert record.result["screenshot_path"] == str(screenshot)
    assert record.result["screenshot_url"].startswith("data:image/png;base64,")


def test_agentkit_artifact_extractor_accepts_web_sized_inline_screenshots(tmp_path: Path) -> None:
    """High-resolution Web screenshots may exceed the old controller task-inline cap."""

    screenshot = tmp_path / "large-web-viewport.png"
    screenshot.write_bytes(b"x" * 3_100_000)

    normalized, artifacts = AgentKitCommandArtifactExtractor().extract(
        "cmd-large-web-shot",
        {"text": "captured", "screenshot_path": str(screenshot)},
    )

    assert isinstance(normalized, dict)
    assert normalized["text"] == "captured"
    assert normalized["screenshot_url"] == ""
    assert normalized["screenshot_url_artifact_id"] == "cmd-large-web-shot-screenshot_url"
    assert len(artifacts) == 1
    assert artifacts[0].size_bytes == 3_100_000

    record = command_result_record(
        "call-large-web-shot",
        CommandResult(
            command_id="cmd-large-web-shot",
            session_id="sess",
            lease_id="lease",
            device_id="dev",
            status=CommandState.SUCCEEDED,
            result=normalized,
            artifacts=artifacts,
        ),
    )

    assert isinstance(record.result, dict)
    assert record.result["screenshot_path"] == str(screenshot)
    assert record.result["screenshot_url"].startswith("data:image/png;base64,")


def test_agentkit_artifact_extractor_accepts_web_sized_data_url_screenshots() -> None:
    data = b"x" * 3_100_000
    data_url = f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"

    normalized, artifacts = AgentKitCommandArtifactExtractor().extract(
        "cmd-large-inline-web-shot",
        {"text": "captured", "screenshot_url": data_url},
    )

    assert isinstance(normalized, dict)
    assert normalized["screenshot_url"] == ""
    assert normalized["screenshot_url_artifact_id"] == "cmd-large-inline-web-shot-screenshot_url"
    assert len(artifacts) == 1
    assert artifacts[0].size_bytes == 3_100_000


def test_agentkit_artifact_extractor_accepts_padded_data_url_at_configured_cap() -> None:
    data = b"x" * 10
    data_url = f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"

    normalized, artifacts = AgentKitCommandArtifactExtractor(max_inline_image_artifact_bytes=10).extract(
        "cmd-cap-inline-web-shot",
        {"screenshot_url": data_url},
    )

    assert isinstance(normalized, dict)
    assert normalized["screenshot_url"] == ""
    assert normalized["screenshot_url_artifact_id"] == "cmd-cap-inline-web-shot-screenshot_url"
    assert len(artifacts) == 1
    assert artifacts[0].size_bytes == 10


def test_agentkit_artifact_extractor_rejects_images_above_configured_cap(tmp_path: Path) -> None:
    screenshot = tmp_path / "too-large.png"
    screenshot.write_bytes(b"x" * 11)

    extractor = AgentKitCommandArtifactExtractor(max_inline_image_artifact_bytes=10)

    with pytest.raises(ValueError, match="exceeds AgentKit max inline artifact bytes"):
        extractor.extract("cmd-too-large", {"screenshot_path": str(screenshot)})
