from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from northcua.agentkernel.context.domain.enums import ContextManagementMode
from northcua.agentkit import AgentCliSpec, AgentRemoteSpec, AgentRuntimeSpec, AgentSpec
from northcua.agentkit.controller_dispatch import command_result_record
from northcua.agentkit.controller_runner import AgentKitRemoteRunnerConfig, register_controller_runner, trace_artifacts
from northcua.agentkit.device_adapter import AgentKitMCPRuntimeAdapter
from northcua.agentkit.external_tools import tool_execution_records_from_command_results
from northcua.agentkit.registry import get_agentkit_registry, reset_agentkit_registry_for_tests
from northcua.agentkit.runtime_sidecars import RUNTIME_SIDECAR_KEY
from northcua.agentkit.session_context import AGENTKIT_SETTINGS_PARAM
from northcua.agents.web.backend import WebActionCommandResult
from northcua.agents.web.domain import WebMouseButton, WebWaitParams
from northcua.agents.web.evaluation.tools import web_browser_client_tool_refs
from northcua.agents.web.observation.ref_alias import WebObservationRefAliases
from northcua.agents.web.session import RegisteredWebAgentSession, WebAgentSessionRegistry
from northcua.agents.web.session.agentkit_adapter import WebAgentDeviceSessionAdapter
from northcua.batch.context_snapshot import snapshot_controller_app_config
from northcua.cli.controller_command_runner import _process_runner_app_config_path
from northcua.controller.execution.executor_protocol import ControllerExecutionContext
from northcua.controller.execution.runner_plugins import ControllerRunnerRegistrationContext
from northcua.controller.execution.runner_registry import RunnerRegistry
from northcua.core.config import NorthCUAAppConfig, WebAgentConfig
from northcua.core.enums import ResultStatus
from northcua.core.models import CommandExecutionResultRecord, CommandRequestRecord
from northcua.core.protocol import CommandRequestPayload
from northcua.core.terminal import AgentTerminalDecision
from northcua_controller_plugins import web_runner_evaluation_helpers as web_eval_helpers
from northcua_controller_plugins.agentkit_remote_device_executor import RemoteAgentKitDeviceCommandExecutor
from northcua_controller_plugins.agentkit_web_remote_dispatch import RemoteWebActionBackend
from northcua_controller_plugins.agentkit_web_runner import (
    WebAgentKitRemoteTaskRunner,
    _web_ref_alias_side_channel,
    _web_runtime_side_channel_settings,
)
from northcua_controller_plugins.web_runner_evaluation_helpers import (
    WebEvaluatorOutcome,
    collect_web_evaluation_artifacts,
)
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef
from northcua_protocol.capabilities import PlatformKind
from northcua_protocol.payloads import (
    CommandRequest,
    CommandResult,
    SessionLeaseGranted,
    TaskFinished,
    TaskRequest,
    sanitize_agentkit_settings_side_channel,
)
from northcua_protocol.state import CommandState
from northcua_runtime_shared.validation import ToolParameterValidationError


def test_remote_web_playwright_backend_treats_tool_error_payload_as_failure() -> None:
    class _FakeDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            return CommandResult(
                command_id="cmd-web",
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                status=CommandState.SUCCEEDED,
                result={"status": "error", "error": "stale ref"},
            )

    async def _run() -> None:
        client = RemoteWebActionBackend(_FakeDispatcher())  # type: ignore[arg-type]
        result = await client.click_ref("@stale")
        assert result.success is False
        assert result.error == "stale ref"

    asyncio.run(_run())


def test_remote_agentkit_device_executor_dispatches_model_visible_web_action() -> None:
    captured: list[dict[str, object]] = []

    class _FakeDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            captured.append(dict(kwargs))
            return CommandResult(
                command_id=str(kwargs["command_id"]),
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                status=CommandState.SUCCEEDED,
                result={"status": "ok", "action": "open", "text": "opened"},
            )

    async def _run() -> None:
        executor = RemoteAgentKitDeviceCommandExecutor(
            _FakeDispatcher(),  # type: ignore[arg-type]
            namespace="web",
            timeout_seconds=37.0,
            side_channel_settings={"headed": True},
            dynamic_side_channel_settings=lambda: {
                "ref_alias_observation_id": "obs-1",
                "ref_alias_epoch": 7,
                "ref_aliases": [{"ref": "6", "source_ref": "@e42"}],
            },
        )
        result = await executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess",
                round_num=1,
                round_step=2,
                commands=[
                    CommandRequestRecord(
                        call_id="call-web-open",
                        server_name="WebAgentActionExecutor",
                        tool_name="web_open",
                        parameters={"url": "https://example.test"},
                        tool_type="web_action",
                    )
                ],
            )
        )
        assert result.status == ResultStatus.SUCCESS
        assert result.results[0].result == {"status": "ok", "action": "open", "text": "opened"}

    asyncio.run(_run())

    assert captured == [
        {
            "tool_id": "web.web_open",
            "parameters": {
                "url": "https://example.test",
                AGENTKIT_SETTINGS_PARAM: {
                    "headed": True,
                    "ref_alias_observation_id": "obs-1",
                    "ref_alias_epoch": 7,
                    "ref_aliases": [{"ref": "6", "source_ref": "@e42"}],
                },
            },
            "command_id": "call-web-open",
            "timeout_seconds": 37.0,
        }
    ]


def test_remote_agentkit_device_executor_returns_retryable_web_parameter_error() -> None:
    class _RejectingDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            del kwargs
            raise ToolParameterValidationError(
                "command parameter 'parameters' must match exactly one schema; "
                "unexpected parameters at parameters: load_state, text, url_pattern, selector"
            )

    async def _run() -> None:
        from northcua_controller_plugins.agentkit_web_runner import _web_parameter_validation_error_payload

        executor = RemoteAgentKitDeviceCommandExecutor(
            _RejectingDispatcher(),  # type: ignore[arg-type]
            namespace="web",
            timeout_seconds=37.0,
            parameter_validation_error_projector=_web_parameter_validation_error_payload,
        )
        result = await executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess",
                round_num=1,
                round_step=2,
                commands=[
                    CommandRequestRecord(
                        call_id="call-web-wait-invalid",
                        server_name="WebAgentActionExecutor",
                        tool_name="web_wait",
                        parameters={
                            "duration_ms": 1000,
                            "load_state": None,
                            "text": None,
                            "url_pattern": None,
                            "selector": None,
                        },
                        tool_type="web_action",
                    )
                ],
            )
        )

        assert result.status == ResultStatus.SUCCESS
        command_result = result.results[0]
        assert command_result.call_id == "call-web-wait-invalid"
        assert command_result.status == ResultStatus.SUCCESS
        assert isinstance(command_result.result, dict)
        assert command_result.result["status"] == "error"
        assert command_result.result["action"] == "wait"
        assert command_result.result["code"] == "invalid_params"
        assert command_result.result["retryable"] is True
        assert command_result.result["error"] == "web_wait arguments must contain exactly one wait target."
        assert "exactly one" in str(command_result.result["hint"])
        assert "unexpected parameters" not in str(command_result.result)

        executions = tool_execution_records_from_command_results(
            pending_tool_calls=[
                {
                    "id": "call-web-wait-invalid",
                    "name": "web_wait",
                    "input": {
                        "duration_ms": 1000,
                        "load_state": None,
                        "text": None,
                        "url_pattern": None,
                        "selector": None,
                    },
                }
            ],
            command_results=result.results,
        )
        assert len(executions) == 1
        assert executions[0].tool_call_id == "call-web-wait-invalid"
        assert executions[0].is_error is True
        assert isinstance(executions[0].tool_output, dict)
        assert executions[0].tool_output["retryable"] is True

    asyncio.run(_run())


def test_remote_agentkit_device_executor_does_not_swallow_transport_failure() -> None:
    class _DisconnectedDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            del kwargs
            raise ConnectionError("controller connection lost")

    async def _run() -> None:
        executor = RemoteAgentKitDeviceCommandExecutor(
            _DisconnectedDispatcher(),  # type: ignore[arg-type]
            namespace="web",
            timeout_seconds=37.0,
        )
        with pytest.raises(ConnectionError, match="controller connection lost"):
            await executor.execute_command_request(
                CommandRequestPayload(
                    session_id="sess",
                    round_num=1,
                    round_step=2,
                    commands=[
                        CommandRequestRecord(
                            call_id="call-web-wait-disconnected",
                            server_name="WebAgentActionExecutor",
                            tool_name="web_wait",
                            parameters={"duration_ms": 1000},
                            tool_type="web_action",
                        )
                    ],
                )
            )

    asyncio.run(_run())


def test_remote_agentkit_device_executor_consumes_runtime_sidecar_without_leaking_to_tool_output() -> None:
    consumed: list[dict[str, object]] = []

    class _FakeDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            return CommandResult(
                command_id=str(kwargs["command_id"]),
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                status=CommandState.SUCCEEDED,
                result={
                    "status": "ok",
                    "action": "snapshot",
                    "text": "Captured current snapshot.",
                    RUNTIME_SIDECAR_KEY: {
                        "web_ref_aliases": {
                            "observation_id": "snapshot-obs",
                            "alias_epoch": 8,
                            "aliases": [{"ref": "43", "source_ref": "@e97"}],
                        }
                    },
                },
            )

    async def _run() -> None:
        executor = RemoteAgentKitDeviceCommandExecutor(
            _FakeDispatcher(),  # type: ignore[arg-type]
            namespace="web",
            timeout_seconds=37.0,
            runtime_sidecar_handler=lambda sidecar: consumed.append(dict(sidecar)),
        )
        result = await executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess",
                round_num=1,
                round_step=2,
                commands=[
                    CommandRequestRecord(
                        call_id="call-web-snapshot",
                        server_name="WebAgentActionExecutor",
                        tool_name="web_snapshot",
                        parameters={},
                        tool_type="web_action",
                    )
                ],
            )
        )
        assert result.status == ResultStatus.SUCCESS
        assert result.results[0].result == {
            "status": "ok",
            "action": "snapshot",
            "text": "Captured current snapshot.",
        }

    asyncio.run(_run())

    assert consumed == [
        {
            "web_ref_aliases": {
                "observation_id": "snapshot-obs",
                "alias_epoch": 8,
                "aliases": [{"ref": "43", "source_ref": "@e97"}],
            }
        }
    ]


def test_remote_agentkit_device_executor_merges_existing_agentkit_settings() -> None:
    captured: list[dict[str, object]] = []

    class _FakeDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            captured.append(dict(kwargs))
            return CommandResult(
                command_id=str(kwargs["command_id"]),
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                status=CommandState.SUCCEEDED,
                result={"status": "ok"},
            )

    async def _run() -> None:
        executor = RemoteAgentKitDeviceCommandExecutor(
            _FakeDispatcher(),  # type: ignore[arg-type]
            namespace="web",
            timeout_seconds=10.0,
            side_channel_settings={"headed": True},
            dynamic_side_channel_settings=lambda: {
                "ref_alias_observation_id": "obs-2",
                "ref_alias_epoch": 2,
                "ref_aliases": [{"ref": "9", "source_ref": "@e9"}],
            },
        )
        await executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess",
                round_num=1,
                round_step=2,
                commands=[
                    CommandRequestRecord(
                        call_id="call-web-click",
                        server_name="WebAgentActionExecutor",
                        tool_name="web_click_ref",
                        parameters={AGENTKIT_SETTINGS_PARAM: {"allow_localhost_urls": True}},
                        tool_type="web_action",
                    )
                ],
            )
        )

    asyncio.run(_run())

    settings = captured[0]["parameters"][AGENTKIT_SETTINGS_PARAM]
    assert settings == {
        "allow_localhost_urls": True,
        "headed": True,
        "ref_alias_observation_id": "obs-2",
        "ref_alias_epoch": 2,
        "ref_aliases": [{"ref": "9", "source_ref": "@e9"}],
    }


def test_command_result_record_strips_runtime_sidecar_by_default() -> None:
    record = command_result_record(
        "call-web",
        CommandResult(
            command_id="cmd-web",
            session_id="sess",
            lease_id="lease",
            device_id="dev",
            status=CommandState.SUCCEEDED,
            result={
                "status": "ok",
                "text": "done",
                RUNTIME_SIDECAR_KEY: {"web_ref_aliases": {"observation_id": "obs", "aliases": []}},
            },
        ),
    )

    assert record.result == {"status": "ok", "text": "done"}


def test_web_runtime_sidecar_refreshes_controller_ref_aliases() -> None:
    from northcua_controller_plugins.agentkit_web_runner import _consume_web_runtime_sidecar

    class _FakeClient:
        pass

    registry = WebAgentSessionRegistry()
    registry.clear()
    session = RegisteredWebAgentSession(
        session_id="sess-sidecar",
        client=_FakeClient(),  # type: ignore[arg-type]
        artifact_store=None,  # type: ignore[arg-type]
        settings=WebAgentConfig(),
    )
    registry.register(session)
    try:
        session.update_ref_aliases(
            WebObservationRefAliases.model_validate(
                {"observation_id": "initial", "aliases": [{"ref": "6", "source_ref": "@e42"}]}
            )
        )
        _consume_web_runtime_sidecar(
            "sess-sidecar",
            {
                "web_ref_aliases": {
                    "observation_id": "snapshot-obs",
                    "alias_epoch": session.current_ref_aliases.alias_epoch + 1,
                    "aliases": [{"ref": "43", "source_ref": "@e97"}],
                }
            },
        )

        assert session.current_ref_aliases.observation_id == "snapshot-obs"
        assert session.resolve_model_ref("43") == "@e97"
        with pytest.raises(ValueError, match="Unknown or stale web ref"):
            session.resolve_model_ref("6")
    finally:
        registry.clear()


def test_web_runtime_sidecar_ignores_stale_controller_ref_aliases() -> None:
    from northcua_controller_plugins.agentkit_web_runner import _consume_web_runtime_sidecar

    class _FakeClient:
        pass

    registry = WebAgentSessionRegistry()
    registry.clear()
    session = RegisteredWebAgentSession(
        session_id="sess-stale-sidecar",
        client=_FakeClient(),  # type: ignore[arg-type]
        artifact_store=None,  # type: ignore[arg-type]
        settings=WebAgentConfig(),
    )
    registry.register(session)
    try:
        session.update_ref_aliases(
            WebObservationRefAliases.model_validate(
                {"observation_id": "fresh", "aliases": [{"ref": "39", "source_ref": "@e99"}]}
            )
        )
        _consume_web_runtime_sidecar(
            "sess-stale-sidecar",
            {
                "web_ref_aliases": {
                    "observation_id": "old",
                    "alias_epoch": session.current_ref_aliases.alias_epoch - 1,
                    "aliases": [{"ref": "39", "source_ref": "@e42"}],
                }
            },
        )

        assert session.current_ref_aliases.observation_id == "fresh"
        assert session.resolve_model_ref("39") == "@e99"
    finally:
        registry.clear()


def test_web_ref_alias_side_channel_preserves_empty_alias_clears() -> None:
    class _FakeClient:
        pass

    registry = WebAgentSessionRegistry()
    registry.clear()
    session = RegisteredWebAgentSession(
        session_id="sess-empty-aliases",
        client=_FakeClient(),  # type: ignore[arg-type]
        artifact_store=None,  # type: ignore[arg-type]
        settings=WebAgentConfig(),
    )
    registry.register(session)
    try:
        session.update_ref_aliases(
            WebObservationRefAliases.model_validate(
                {"observation_id": "initial", "aliases": [{"ref": "6", "source_ref": "@e42"}]}
            )
        )
        session.clear_ref_aliases(observation_id="post-click-none")

        assert _web_ref_alias_side_channel("sess-empty-aliases") == {
            "ref_alias_observation_id": "post-click-none",
            "ref_alias_epoch": 2,
            "ref_aliases": [],
        }
    finally:
        registry.clear()


def test_remote_web_playwright_backend_snapshot_uses_raw_probe_not_model_action() -> None:
    captured_tools: list[str] = []

    class _FakeDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            tool_id = str(kwargs.get("tool_id") or "")
            captured_tools.append(tool_id)
            assert tool_id == "web.web_agent_snapshot"
            return CommandResult(
                command_id="cmd-web-snapshot",
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                status=CommandState.SUCCEEDED,
                result={
                    "url": "https://example.test",
                    "title": "Example",
                    "snapshot": 'button "Submit" [ref=@e1]',
                    "refs": {"e1": {"role": "button", "name": "Submit", "text": "", "tag": "button"}},
                },
            )

    async def _run() -> None:
        client = RemoteWebActionBackend(_FakeDispatcher())  # type: ignore[arg-type]
        snapshot = await client.snapshot()
        assert [element.model_ref for element in snapshot.elements] == ["@e1"]
        assert snapshot.snapshot == 'button "Submit" [ref=@e1]'

    asyncio.run(_run())
    assert captured_tools == ["web.web_agent_snapshot"]


def test_trace_artifacts_use_explicit_web_trace_label(tmp_path: Path) -> None:
    artifact_root = tmp_path / "web-session"
    artifact_root.mkdir()
    web_trace = artifact_root / "web_agent_trace.jsonl"
    web_trace.write_text('{"session_step":1}\n', encoding="utf-8")

    class _Store:
        root = artifact_root
        primary_trace_path = web_trace

    context = ControllerExecutionContext(
        request=TaskRequest(task_id="task-web", trace_id="trace-web", agent_kind="web", request="Check the page"),
        session_id="sess-web",
        lease=SessionLeaseGranted(
            lease_id="lease",
            session_id="sess-web",
            task_id="task-web",
            device_id="browser-1",
            trace_id="trace-web",
        ),
    )

    refs = trace_artifacts(context, _Store())

    assert [ref.artifact_id for ref in refs] == ["task-web-web-trace"]


def test_trace_artifacts_keep_desktop_steps_label(tmp_path: Path) -> None:
    artifact_root = tmp_path / "desktop-session"
    artifact_root.mkdir()
    steps = artifact_root / "steps.jsonl"
    steps.write_text('{"session_step":1}\n', encoding="utf-8")

    class _Store:
        root = artifact_root
        primary_trace_path = steps

    context = ControllerExecutionContext(
        request=TaskRequest(
            task_id="task-desktop",
            trace_id="trace-desktop",
            agent_kind="desktop",
            request="Open Calculator",
        ),
        session_id="sess-desktop",
        lease=SessionLeaseGranted(
            lease_id="lease",
            session_id="sess-desktop",
            task_id="task-desktop",
            device_id="desktop-1",
            trace_id="trace-desktop",
        ),
    )

    refs = trace_artifacts(context, _Store())

    assert [ref.artifact_id for ref in refs] == ["task-desktop-steps"]


def test_remote_web_playwright_backend_sends_web_settings_side_channel() -> None:
    captured_parameters: list[dict[str, object]] = []

    class _FakeDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            parameters = kwargs.get("parameters")
            captured_parameters.append(dict(parameters) if isinstance(parameters, dict) else {})
            return CommandResult(
                command_id="cmd-web",
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                status=CommandState.SUCCEEDED,
                result={"status": "ok", "result": {"url": "https://example.com"}},
            )

    async def _run() -> None:
        client = RemoteWebActionBackend(
            _FakeDispatcher(),  # type: ignore[arg-type]
            web_settings={"allow_localhost_urls": False, "headed": True, "playwright_browser": "chromium"},
        )
        result = await client.open("https://example.com")
        assert result.success is True

    asyncio.run(_run())

    assert captured_parameters[0][AGENTKIT_SETTINGS_PARAM] == {
        "allow_localhost_urls": False,
        "headed": True,
        "playwright_browser": "chromium",
    }


def test_remote_web_playwright_backend_rejects_unsafe_web_settings_side_channel() -> None:
    with pytest.raises(ValueError, match="unsafe_backend_binary"):
        RemoteWebActionBackend(  # type: ignore[arg-type]
            object(),
            web_settings={"allow_localhost_urls": False, "unsafe_backend_binary": "custom-browser"},
        )


def test_web_remote_runner_projects_full_web_config_to_safe_side_channel() -> None:
    settings = _web_runtime_side_channel_settings(
        WebAgentConfig(
            headed=True,
            allow_localhost_urls=False,
            snapshot_eval_timeout_seconds=11.0,
            snapshot_fallback_max_refs=321,
        )
    )

    assert settings["headed"] is True
    assert settings["allow_localhost_urls"] is False
    assert settings["playwright_browser"] == "chromium"
    assert "snapshot_max_refs" not in settings
    assert "snapshot_max_visited" not in settings
    assert settings["snapshot_eval_timeout_seconds"] == 11.0
    assert settings["snapshot_fallback_max_refs"] == 321
    assert "unsafe_backend_binary" not in settings
    RemoteWebActionBackend(object(), web_settings=settings)  # type: ignore[arg-type]
    sanitized = sanitize_agentkit_settings_side_channel(settings)
    assert sanitized["snapshot_eval_timeout_seconds"] == 11.0
    assert sanitized["snapshot_fallback_max_refs"] == 321


def test_remote_web_playwright_backend_unwraps_ok_result_payload_for_parity() -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    class _FakeDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            captured.append((str(kwargs.get("tool_id")), dict(kwargs.get("parameters") or {})))
            return CommandResult(
                command_id="cmd-web",
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                status=CommandState.SUCCEEDED,
                result={"status": "ok", "result": {"clicked": "@e2"}},
            )

    async def _run() -> None:
        client = RemoteWebActionBackend(_FakeDispatcher())  # type: ignore[arg-type]
        result = await client.click_ref("e2")
        assert result.success is True
        assert result.data == {"clicked": "@e2"}

    asyncio.run(_run())
    assert captured == [("web.web_agent_backend_click_ref", {"ref": "@e2"})]


def test_remote_web_playwright_backend_dispatches_extended_web_actions() -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    class _FakeDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            captured.append((str(kwargs.get("tool_id")), dict(kwargs.get("parameters") or {})))
            return CommandResult(
                command_id="cmd-web",
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                status=CommandState.SUCCEEDED,
                result={"status": "ok", "result": {"ok": True}},
            )

    async def _run() -> None:
        client = RemoteWebActionBackend(_FakeDispatcher())  # type: ignore[arg-type]
        await client.hover_ref("e2")
        await client.select_option_ref("country", ["US"])
        await client.check_ref("terms")
        await client.uncheck_ref("@terms")
        await client.drag_ref("src", "dst")
        await client.get_box("e4")
        await client.get_boxes(["e4", "@e5", "e4"])
        await client.get_ref_states(["e4", "@e5", "e4"])
        await client.get_viewport_size()
        await client.click_point(10, 20, WebMouseButton.RIGHT)
        await client.double_click_point(11, 21, WebMouseButton.LEFT)
        await client.drag_point(1, 2, 3, 4)
        await client.list_tabs()
        await client.new_tab(url="https://example.com", label="docs")
        await client.close_tab("t2")
        await client.focus_tab("docs")

    asyncio.run(_run())

    assert captured == [
        ("web.web_agent_backend_hover_ref", {"ref": "@e2"}),
        ("web.web_agent_backend_select_option_ref", {"ref": "@country", "values": ["US"]}),
        ("web.web_agent_backend_check_ref", {"ref": "@terms"}),
        ("web.web_agent_backend_uncheck_ref", {"ref": "@terms"}),
        ("web.web_agent_backend_drag_ref", {"from_ref": "@src", "to_ref": "@dst"}),
        ("web.web_agent_get_box", {"ref": "@e4"}),
        ("web.web_agent_get_boxes", {"refs": ["@e4", "@e5"]}),
        ("web.web_agent_get_ref_states", {"refs": ["@e4", "@e5"]}),
        ("web.web_agent_viewport", {}),
        ("web.web_agent_backend_click_point", {"x": 10, "y": 20, "button": "right"}),
        ("web.web_agent_backend_double_click_point", {"x": 11, "y": 21, "button": "left"}),
        ("web.web_agent_backend_drag_point", {"from_x": 1, "from_y": 2, "to_x": 3, "to_y": 4}),
        ("web.web_agent_backend_list_tabs", {}),
        ("web.web_agent_backend_new_tab", {"url": "https://example.com", "label": "docs"}),
        ("web.web_agent_backend_close_tab", {"tab_ref": "t2"}),
        ("web.web_agent_backend_focus_tab", {"tab_ref": "docs"}),
    ]


def test_remote_web_playwright_backend_wait_uses_timeout_override() -> None:
    captured: list[float | None] = []

    class _FakeDispatcher:
        async def dispatch(self, **kwargs: object) -> CommandResult:
            timeout = kwargs.get("timeout_seconds")
            captured.append(float(timeout) if isinstance(timeout, (int, float)) else None)
            return CommandResult(
                command_id="cmd-web",
                session_id="sess",
                lease_id="lease",
                device_id="dev",
                status=CommandState.SUCCEEDED,
                result={"status": "ok", "result": {"waited": True}},
            )

    async def _run() -> None:
        client = RemoteWebActionBackend(_FakeDispatcher(), timeout_seconds=30.0)  # type: ignore[arg-type]
        await client.wait(WebWaitParams(load_state="domcontentloaded"), timeout_seconds=1.25)
        await client.wait(WebWaitParams(duration_ms=25))

    asyncio.run(_run())

    assert captured == [1.25, 30.0]


def test_web_agent_device_session_adapter_uses_web_settings_side_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeWebActionBackend:
        pass

    def _fake_build_playwright_web_backend(*, config: object, session_id: str) -> _FakeWebActionBackend:
        captured["config"] = config
        captured["session_id"] = session_id
        return _FakeWebActionBackend()

    monkeypatch.setattr(
        "northcua.agents.web.session.agentkit_adapter.build_playwright_web_backend",
        _fake_build_playwright_web_backend,
    )
    registry = WebAgentSessionRegistry()
    registry.clear()
    adapter = WebAgentDeviceSessionAdapter(agent_kind="web")
    try:
        adapter.ensure_device_session(
            command=CommandRequest(
                command_id="cmd-web-settings",
                idempotency_key="idem-web-settings",
                session_id="sess-web-settings",
                lease_id="lease",
                device_id="dev",
                tool_id="web.web_open",
                parameters={AGENTKIT_SETTINGS_PARAM: {"allow_localhost_urls": False}},
            ),
            artifact_root=tmp_path,
        )
        registered = registry.get_registered_session("sess-web-settings")
        assert registered.settings.allow_localhost_urls is False
        assert registered.settings.playwright_browser == "chromium"
        assert captured["session_id"] == "sess-web-settings"
    finally:
        registry.clear()


def test_web_agent_device_session_adapter_updates_existing_ref_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeWebActionBackend:
        pass

    monkeypatch.setattr(
        "northcua.agents.web.session.agentkit_adapter.build_playwright_web_backend",
        lambda *args, **kwargs: _FakeWebActionBackend(),
    )
    registry = WebAgentSessionRegistry()
    registry.clear()
    adapter = WebAgentDeviceSessionAdapter(agent_kind="web")
    try:
        base_command = CommandRequest(
            command_id="cmd-web-settings",
            idempotency_key="idem-web-settings",
            session_id="sess-web-alias",
            lease_id="lease",
            device_id="dev",
            tool_id="web.web_open",
            parameters={AGENTKIT_SETTINGS_PARAM: {"allow_localhost_urls": True}},
        )
        adapter.ensure_device_session(command=base_command, artifact_root=tmp_path)
        registered = registry.get_registered_session("sess-web-alias")
        assert registered.current_ref_aliases.aliases == []

        alias_command = base_command.model_copy(
            update={
                "command_id": "cmd-web-alias",
                "idempotency_key": "idem-web-alias",
                "parameters": {
                    AGENTKIT_SETTINGS_PARAM: {
                        "allow_localhost_urls": True,
                        "ref_alias_observation_id": "initial-obs",
                        "ref_aliases": [{"ref": "6", "source_ref": "@e42"}],
                    }
                },
            }
        )
        adapter.ensure_device_session(command=alias_command, artifact_root=tmp_path)

        assert registered.current_ref_aliases.observation_id == "initial-obs"
        assert registered.resolve_model_ref("6") == "@e42"

        clear_alias_command = base_command.model_copy(
            update={
                "command_id": "cmd-web-clear-alias",
                "idempotency_key": "idem-web-clear-alias",
                "parameters": {
                    AGENTKIT_SETTINGS_PARAM: {
                        "allow_localhost_urls": True,
                        "ref_alias_observation_id": "post-empty",
                        "ref_alias_epoch": 1,
                        "ref_aliases": [],
                    }
                },
            }
        )
        adapter.ensure_device_session(command=clear_alias_command, artifact_root=tmp_path)

        assert registered.current_ref_aliases.observation_id == "post-empty"
        assert registered.current_ref_aliases.aliases == []
        with pytest.raises(ValueError, match="Unknown or stale web ref"):
            registered.resolve_model_ref("6")
    finally:
        registry.clear()


def test_web_agent_device_session_adapter_ignores_stale_ref_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeWebActionBackend:
        pass

    monkeypatch.setattr(
        "northcua.agents.web.session.agentkit_adapter.build_playwright_web_backend",
        lambda *args, **kwargs: _FakeWebActionBackend(),
    )
    registry = WebAgentSessionRegistry()
    registry.clear()
    adapter = WebAgentDeviceSessionAdapter(agent_kind="web")
    try:
        base_command = CommandRequest(
            command_id="cmd-web-settings",
            idempotency_key="idem-web-settings",
            session_id="sess-web-stale-alias",
            lease_id="lease",
            device_id="dev",
            tool_id="web.web_open",
            parameters={AGENTKIT_SETTINGS_PARAM: {"allow_localhost_urls": True}},
        )
        adapter.ensure_device_session(command=base_command, artifact_root=tmp_path)
        registered = registry.get_registered_session("sess-web-stale-alias")
        registered.update_ref_aliases(
            WebObservationRefAliases.model_validate(
                {"observation_id": "fresh", "aliases": [{"ref": "39", "source_ref": "@e99"}]}
            )
        )

        stale_command = base_command.model_copy(
            update={
                "command_id": "cmd-web-stale-alias",
                "idempotency_key": "idem-web-stale-alias",
                "parameters": {
                    AGENTKIT_SETTINGS_PARAM: {
                        "allow_localhost_urls": True,
                        "ref_alias_observation_id": "old",
                        "ref_alias_epoch": registered.current_ref_aliases.alias_epoch - 1,
                        "ref_aliases": [{"ref": "39", "source_ref": "@e42"}],
                    }
                },
            }
        )
        adapter.ensure_device_session(command=stale_command, artifact_root=tmp_path)

        assert registered.current_ref_aliases.observation_id == "fresh"
        assert registered.resolve_model_ref("39") == "@e99"
    finally:
        registry.clear()


def test_web_agent_device_session_adapter_ignores_equal_epoch_divergent_ref_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeWebActionBackend:
        pass

    monkeypatch.setattr(
        "northcua.agents.web.session.agentkit_adapter.build_playwright_web_backend",
        lambda *args, **kwargs: _FakeWebActionBackend(),
    )
    registry = WebAgentSessionRegistry()
    registry.clear()
    adapter = WebAgentDeviceSessionAdapter(agent_kind="web")
    try:
        base_command = CommandRequest(
            command_id="cmd-web-settings",
            idempotency_key="idem-web-settings",
            session_id="sess-web-equal-epoch-alias",
            lease_id="lease",
            device_id="dev",
            tool_id="web.web_open",
            parameters={AGENTKIT_SETTINGS_PARAM: {"allow_localhost_urls": True}},
        )
        adapter.ensure_device_session(command=base_command, artifact_root=tmp_path)
        registered = registry.get_registered_session("sess-web-equal-epoch-alias")
        registered.update_ref_aliases(
            WebObservationRefAliases.model_validate(
                {"observation_id": "fresh", "aliases": [{"ref": "39", "source_ref": "@e99"}]}
            )
        )

        conflicting_command = base_command.model_copy(
            update={
                "command_id": "cmd-web-equal-conflict",
                "idempotency_key": "idem-web-equal-conflict",
                "parameters": {
                    AGENTKIT_SETTINGS_PARAM: {
                        "allow_localhost_urls": True,
                        "ref_alias_observation_id": "conflict",
                        "ref_alias_epoch": registered.current_ref_aliases.alias_epoch,
                        "ref_aliases": [{"ref": "39", "source_ref": "@e42"}],
                    }
                },
            }
        )
        adapter.ensure_device_session(command=conflicting_command, artifact_root=tmp_path)

        assert registered.current_ref_aliases.observation_id == "fresh"
        assert registered.resolve_model_ref("39") == "@e99"
    finally:
        registry.clear()


def test_agentkit_web_close_session_tool_closes_and_unregisters() -> None:
    class _FakeClient:
        closed = False

        async def close(self) -> WebActionCommandResult:
            self.closed = True
            return WebActionCommandResult(success=True, data={"status": "ok"})

    fake_client = _FakeClient()
    registry = WebAgentSessionRegistry()
    registry.clear()
    registry.register(
        RegisteredWebAgentSession(
            session_id="sess-web-close",
            client=fake_client,  # type: ignore[arg-type]
            artifact_store=object(),  # type: ignore[arg-type]
            settings=object(),  # type: ignore[arg-type]
        )
    )

    async def _run() -> None:
        adapter = AgentKitMCPRuntimeAdapter(agent_kinds={"web"})
        result = await adapter.execute(
            CommandRequest(
                command_id="cmd-close",
                idempotency_key="idem-close",
                session_id="sess-web-close",
                lease_id="lease",
                device_id="dev",
                tool_id="web.web_close_session",
            )
        )
        assert result.status == CommandState.SUCCEEDED

    try:
        asyncio.run(_run())
        assert fake_client.closed is True
        with pytest.raises(LookupError):
            registry.get_registered_session("sess-web-close")
    finally:
        registry.clear()


def test_controller_runner_uses_spec_runner_factory() -> None:
    reset_agentkit_registry_for_tests()
    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
            remote=AgentRemoteSpec(
                enabled=True,
                runner_factory="tests.agentkit.test_agentkit_lowcode:_dummy_runner_factory",
            ),
        )
    )
    registry = RunnerRegistry()

    register_controller_runner(
        registry,
        sdk=object(),  # type: ignore[arg-type]
        context=ControllerRunnerRegistrationContext(app_config_path="config/app.local.yaml", command_timeout_seconds=3),
    )

    assert registry.resolve("android") is not None


def test_controller_runner_registers_one_immutable_config_for_process_reuse(tmp_path: Path) -> None:
    reset_agentkit_registry_for_tests()
    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
            remote=AgentRemoteSpec(
                enabled=True,
                runner_factory="tests.agentkit.test_agentkit_lowcode:_dummy_runner_factory",
            ),
        )
    )
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "name: android\nllm_config:\n  model: fixture\n  base_url: https://example.invalid\n  api_key: key\n",
        encoding="utf-8",
    )
    app = tmp_path / "app.yaml"
    app.write_text(
        "agent_config_path: agent.yaml\ncontext_management:\n  mode: enforce\n",
        encoding="utf-8",
    )
    registry = RunnerRegistry()

    register_controller_runner(
        registry,
        sdk=object(),  # type: ignore[arg-type]
        context=ControllerRunnerRegistrationContext(
            app_config_path=str(app),
            command_timeout_seconds=3,
            artifact_root_dir_override=str(tmp_path / "artifacts"),
        ),
    )
    frozen = registry.registered_app_config_path("android")
    assert frozen is not None
    frozen_text = Path(frozen).read_text(encoding="utf-8")
    app.write_text("context_management:\n  mode: off\n", encoding="utf-8")

    assert Path(frozen).read_text(encoding="utf-8") == frozen_text
    assert Path(frozen).is_relative_to(tmp_path / "artifacts" / ".controller_config_snapshot")


def test_controller_runner_reuses_a_published_parent_snapshot_for_inline_and_process(tmp_path: Path) -> None:
    reset_agentkit_registry_for_tests()
    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
            remote=AgentRemoteSpec(
                enabled=True,
                runner_factory="tests.agentkit.test_agentkit_lowcode:_dummy_runner_factory",
            ),
        )
    )
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "name: android\nllm_config:\n  model: fixture\n  base_url: https://example.invalid\n  api_key: key\n",
        encoding="utf-8",
    )
    app = tmp_path / "app.yaml"
    app.write_text(
        "agent_config_path: agent.yaml\ncontext_management:\n  mode: enforce\n",
        encoding="utf-8",
    )
    parent_snapshot = snapshot_controller_app_config(app, snapshot_root=tmp_path / "parent-snapshot")
    registry = RunnerRegistry()

    register_controller_runner(
        registry,
        sdk=object(),  # type: ignore[arg-type]
        context=ControllerRunnerRegistrationContext(
            app_config_path=str(parent_snapshot),
            command_timeout_seconds=3,
            artifact_root_dir_override=str(tmp_path / "artifacts"),
        ),
    )

    assert registry.registered_app_config_path("android") == str(parent_snapshot)
    assert _process_runner_app_config_path(registry, {"android"}, fallback="unused") == str(parent_snapshot)
    assert not (tmp_path / "artifacts" / ".controller_config_snapshot").exists()


def test_controller_runner_off_mode_keeps_original_config_path(tmp_path: Path) -> None:
    reset_agentkit_registry_for_tests()
    get_agentkit_registry().register(
        AgentSpec(
            runtime=AgentRuntimeSpec(kind="android", display_name="Android CUA", platform=PlatformKind.ANDROID),
            cli=AgentCliSpec(agent_name="android", help="Android CUA commands"),
            remote=AgentRemoteSpec(
                enabled=True,
                runner_factory="tests.agentkit.test_agentkit_lowcode:_dummy_runner_factory",
            ),
        )
    )
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "name: android\nllm_config:\n  model: fixture\n  base_url: https://example.invalid\n  api_key: key\n",
        encoding="utf-8",
    )
    app = tmp_path / "app.yaml"
    app.write_text("agent_config_path: agent.yaml\ncontext_management:\n  mode: off\n", encoding="utf-8")
    registry = RunnerRegistry()

    register_controller_runner(
        registry,
        sdk=object(),  # type: ignore[arg-type]
        context=ControllerRunnerRegistrationContext(app_config_path=str(app), command_timeout_seconds=3),
    )

    assert registry.registered_app_config_path("android") == str(app)
    assert not (tmp_path / ".controller_config_snapshot").exists()


def test_controller_runner_mixed_profiles_share_snapshot_without_policy_drift(tmp_path: Path) -> None:
    reset_agentkit_registry_for_tests()
    for kind in ("android", "ios"):
        get_agentkit_registry().register(
            AgentSpec(
                runtime=AgentRuntimeSpec(kind=kind, display_name=kind.title(), platform=PlatformKind.ANDROID),
                cli=AgentCliSpec(agent_name=kind, help=f"{kind} CUA commands"),
                remote=AgentRemoteSpec(
                    enabled=True,
                    runner_factory="tests.agentkit.test_agentkit_lowcode:_dummy_runner_factory",
                ),
            )
        )
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "name: mobile\nllm_config:\n  model: fixture\n  base_url: https://example.invalid\n  api_key: key\n",
        encoding="utf-8",
    )
    app = tmp_path / "app.yaml"
    app.write_text(
        """agent_config_path: agent.yaml
context_management:
  mode: off
  agent_overrides:
    android:
      mode: enforce
    ios:
      mode: off
""",
        encoding="utf-8",
    )
    registry = RunnerRegistry()

    register_controller_runner(
        registry,
        sdk=object(),  # type: ignore[arg-type]
        context=ControllerRunnerRegistrationContext(
            app_config_path=str(app),
            command_timeout_seconds=3,
            artifact_root_dir_override=str(tmp_path / "artifacts"),
        ),
    )

    android_path = registry.registered_app_config_path("android")
    ios_path = registry.registered_app_config_path("ios")
    assert android_path is not None
    assert android_path == ios_path
    assert android_path != str(app)
    frozen = NorthCUAAppConfig.from_yaml(Path(android_path))
    assert frozen.context_management.resolve("android").mode == ContextManagementMode.ENFORCE
    assert frozen.context_management.resolve("ios").mode == ContextManagementMode.OFF


def test_web_remote_runner_forwards_round_and_session_step_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    transport_factory_calls: list[dict[str, object]] = []
    result_path = tmp_path / "result.json"
    result_path.write_text("{}", encoding="utf-8")

    class _FakeWebSocketConfig:
        runtime_mode = "local"

        def model_copy(self, *, update: dict[str, object]) -> object:
            copy = _FakeWebSocketConfig()
            copy.runtime_mode = update.get("runtime_mode", self.runtime_mode)
            return copy

    class _FakeWebConfig:
        max_steps = 5
        headed = False

        def model_copy(self, *, update: dict[str, object]) -> object:
            copy = _FakeWebConfig()
            copy.max_steps = int(update.get("max_steps", self.max_steps))
            copy.headed = bool(update.get("headed", self.headed))
            return copy

        def model_dump(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {"max_steps": self.max_steps, "headed": self.headed}

    class _FakeAppConfig:
        websocket = _FakeWebSocketConfig()
        web = _FakeWebConfig()

    class _FakeArtifactStore:
        pass

    _FakeArtifactStore.runtime_log_path = result_path
    _FakeArtifactStore.llm_trace_log_path = result_path
    _FakeArtifactStore.primary_trace_path = result_path
    _FakeArtifactStore.trace_path = result_path
    _FakeArtifactStore.result_path = result_path

    class _FakeStatus:
        value = "success"

    class _FakeFacade:
        async def run_web(self, **kwargs: object) -> object:
            calls.append(kwargs)
            result = type(
                "_Result",
                (),
                {
                    "status": _FakeStatus(),
                    "hit_step_limit": False,
                    "final_answer": "done",
                    "tool_errors": [],
                    "round_num": 1,
                    "evaluation_report": {},
                    "evaluation_report_path": "",
                    "evaluation_error": "",
                    "to_terminal_decision": lambda self: AgentTerminalDecision.finish(final_answer="done"),
                },
            )()
            await kwargs["on_finalize"](result)
            if result.evaluation_report:
                payload = json.loads(result_path.read_text())
                payload.update(
                    evaluation_report=result.evaluation_report, evaluation_report_path=result.evaluation_report_path
                )
                result_path.write_text(json.dumps(payload))
            return result

    class _FakeApplication:
        app_config = _FakeAppConfig()
        artifact_store = _FakeArtifactStore()
        facade = _FakeFacade()

        @classmethod
        def from_app_config(cls, *args: object, **kwargs: object) -> "_FakeApplication":
            del args, kwargs
            return cls()

        async def stop(self) -> None:
            return None

    monkeypatch.setattr("northcua.agents.web.WebAgentApplication", _FakeApplication)
    context = ControllerExecutionContext(
        request=TaskRequest(
            task_id="task-web",
            trace_id="trace-web",
            agent_kind="web",
            request="Check the page",
            metadata={
                "max_steps": 8,
                "max_session_steps": 17,
                "headed": "true",
                "tool_choice_policy": "required",
                "allowed_tool_names": ["web_snapshot", "web_click_ref"],
            },
        ),
        session_id="sess-web",
        lease=SessionLeaseGranted(
            lease_id="lease",
            session_id="sess-web",
            task_id="task-web",
            device_id="browser-1",
            trace_id="trace-web",
        ),
    )

    class _FakeTransport:
        lease = context.lease
        metadata_overrides: dict[str, object] = {}

        async def send_command(self, command: CommandRequest, *, trace_id: str) -> CommandResult:
            raise AssertionError(f"unexpected command dispatch: {command.command_id} {trace_id}")

        async def emit_progress(self, message_type: object, payload: object) -> None:
            del message_type, payload

    transport = _FakeTransport()

    def _transport_factory(
        runner_context: ControllerExecutionContext,
        metadata_overrides: dict[str, object],
    ) -> _FakeTransport:
        assert runner_context is context
        transport_factory_calls.append(dict(metadata_overrides))
        return transport

    async def _run() -> None:
        runner = WebAgentKitRemoteTaskRunner(
            config=AgentKitRemoteRunnerConfig(app_config_path="config/app.yaml", max_steps_per_task=4),
            command_transport_factory=_transport_factory,  # type: ignore[arg-type]
        )
        finished = await runner.run(context)

        assert finished.state.value == "finished"
        assert finished.error is None

    asyncio.run(_run())

    assert calls
    assert calls[0]["max_steps"] == 4
    assert calls[0]["max_session_steps"] == 17
    assert calls[0]["headed"] is True
    assert calls[0]["tool_choice_policy"] == "required"
    assert calls[0]["allowed_tool_names"] == ("web_snapshot", "web_click_ref")
    assert isinstance(calls[0]["external_tool_device_executor"], RemoteAgentKitDeviceCommandExecutor)
    executor = calls[0]["external_tool_device_executor"]
    assert executor.dispatcher.transport is transport
    assert transport_factory_calls == [{}]
    client_factory = calls[0]["client_factory"]
    client = client_factory("sess-web")  # type: ignore[operator]
    assert client.web_settings["headed"] is True
    assert client.dispatcher.transport is transport


def test_run_web_round_evaluator_wires_configured_remote_live_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeToolsConfig:
        enabled = True
        allow_client_live_probe = True
        allow_client_mutating_tools = True

    class _FakeEvaluationConfig:
        tools = _FakeToolsConfig()

    class _FakeAppConfig:
        evaluation = _FakeEvaluationConfig()

    class _FakeArtifactStore:
        root = tmp_path
        session_id = "sess-web"

    class _FakeApp:
        app_config = _FakeAppConfig()
        artifact_store = _FakeArtifactStore()

    captured: dict[str, object] = {}

    class _FakeEvaluatorService:
        def __init__(self, *, model_step_executor: object) -> None:
            captured["model_step_executor"] = model_step_executor

        async def evaluate_artifact_root(self, artifact_root: Path, **kwargs: object) -> object:
            captured.update(kwargs)
            dispatcher = kwargs["client_tool_dispatcher"]
            tool = {tool.name: tool for tool in web_browser_client_tool_refs(allow_mutating=True)}[
                "client_browser_click_ref"
            ]
            captured["live_probe_payload"] = await dispatcher.dispatch(  # type: ignore[attr-defined]
                tool=tool,
                parameters={"ref": "@e1"},
                remote=True,
            )
            report_dir = artifact_root / "evaluations" / "web_round_evaluator" / "round_0001"
            report_dir.mkdir(parents=True)
            (report_dir / "final_report.json").write_text('{"overall_verdict":"success"}\n', encoding="utf-8")
            return object()

    class _FakeRemoteDispatcher:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def dispatch(self, **kwargs: object) -> CommandResult:
            self.calls.append(dict(kwargs))
            return CommandResult(
                command_id=str(kwargs.get("command_id") or "cmd-web"),
                session_id="sess",
                lease_id="lease",
                device_id="browser-1",
                status=CommandState.SUCCEEDED,
                result={"status": "ok", "result": {"clicked": "@e1"}},
            )

    remote_dispatcher = _FakeRemoteDispatcher()
    monkeypatch.setattr(web_eval_helpers, "_build_model_step_executor", lambda _app: object())
    monkeypatch.setattr(
        "northcua.agents.web.evaluation.service.WebRoundEvaluatorService",
        _FakeEvaluatorService,
    )

    outcome = asyncio.run(
        web_eval_helpers.run_web_round_evaluator(
            _FakeApp(),
            remote=True,
            task_id="task-web",
            remote_dispatcher=remote_dispatcher,  # type: ignore[arg-type]
            command_timeout_seconds=37.0,
        )
    )

    assert captured["remote"] is True
    assert captured["enable_live_probe"] is True
    assert captured["allow_client_mutating_tools"] is True
    assert captured["client_tool_dispatcher"] is not None
    assert remote_dispatcher.calls[0]["tool_id"] == "web.web_click_ref"
    assert remote_dispatcher.calls[0]["parameters"] == {"ref": "@e1"}
    assert remote_dispatcher.calls[0]["timeout_seconds"] == 37.0
    assert captured["live_probe_payload"]["status"] == "success"  # type: ignore[index]
    assert outcome.report == {"overall_verdict": "success"}


def test_web_remote_runner_attaches_web_evaluator_report_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "evaluation_report": {"overall_verdict": "stale"},
                "evaluation_report_path": "/tmp/stale-final-report.json",
            }
        ),
        encoding="utf-8",
    )

    class _FakeWebSocketConfig:
        runtime_mode = "local"

        def model_copy(self, *, update: dict[str, object]) -> object:
            copy = _FakeWebSocketConfig()
            copy.runtime_mode = update.get("runtime_mode", self.runtime_mode)
            return copy

    class _FakeWebConfig:
        max_steps = 5
        headed = False
        command_timeout_seconds = 91.0

        def model_copy(self, *, update: dict[str, object]) -> object:
            copy = _FakeWebConfig()
            copy.headed = bool(update.get("headed", self.headed))
            return copy

        def model_dump(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "max_steps": self.max_steps,
                "headed": self.headed,
                "command_timeout_seconds": self.command_timeout_seconds,
            }

    class _FakeEvaluationConfig:
        enabled = True
        fail_open = False

    class _FakeAppConfig:
        websocket = _FakeWebSocketConfig()
        web = _FakeWebConfig()
        evaluation = _FakeEvaluationConfig()

    class _FakeArtifactStore:
        pass

    _FakeArtifactStore.runtime_log_path = result_path
    _FakeArtifactStore.llm_trace_log_path = result_path
    _FakeArtifactStore.primary_trace_path = result_path
    _FakeArtifactStore.trace_path = result_path
    _FakeArtifactStore.result_path = result_path

    class _FakeStatus:
        value = "success"

    class _FakeFacade:
        async def run_web(self, **kwargs: object) -> object:
            executor = kwargs["external_tool_device_executor"]
            assert getattr(executor, "timeout_seconds") == 91.0
            client = kwargs["client_factory"]("sess-web")
            assert getattr(client, "timeout_seconds") == 91.0
            result = type(
                "_Result",
                (),
                {
                    "status": _FakeStatus(),
                    "hit_step_limit": False,
                    "final_answer": "done",
                    "tool_errors": [],
                    "round_num": 1,
                    "evaluation_report": {},
                    "evaluation_report_path": "",
                    "evaluation_error": "",
                    "to_terminal_decision": lambda self: AgentTerminalDecision.finish(final_answer="done"),
                },
            )()
            await kwargs["on_finalize"](result)
            if result.evaluation_report:
                payload = json.loads(result_path.read_text())
                payload.update(
                    evaluation_report=result.evaluation_report, evaluation_report_path=result.evaluation_report_path
                )
                result_path.write_text(json.dumps(payload))
            return result

    class _FakeApplication:
        app_config = _FakeAppConfig()
        artifact_store = _FakeArtifactStore()
        facade = _FakeFacade()

        @classmethod
        def from_app_config(cls, *args: object, **kwargs: object) -> "_FakeApplication":
            del args, kwargs
            return cls()

        async def stop(self) -> None:
            return None

    async def _fake_evaluator(
        app: object,
        *,
        remote: bool,
        round_num: int = 1,
        task_id: str = "",
        remote_dispatcher: object | None = None,
        command_timeout_seconds: float = 0.0,
    ) -> WebEvaluatorOutcome:
        del app
        assert remote is True
        assert task_id == "task-web"
        assert remote_dispatcher is not None
        assert command_timeout_seconds == 91.0
        return WebEvaluatorOutcome(
            report={"overall_verdict": "success", "web_blocker": "none"},
            report_path=str(tmp_path / "evaluations" / "web_round_evaluator" / "round_0001" / "final_report.json"),
            artifacts=[
                ArtifactRef.from_inline_bytes(
                    artifact_id="task-web-evaluation-web_round_evaluator-round_0001-final_report",
                    kind=ArtifactKind.JSON,
                    data=b"{}",
                    mime_type="application/json",
                )
            ],
        )

    monkeypatch.setattr("northcua.agents.web.WebAgentApplication", _FakeApplication)
    monkeypatch.setattr(
        "northcua_controller_plugins.agentkit_web_runner.run_web_round_evaluator",
        _fake_evaluator,
    )
    context = ControllerExecutionContext(
        request=TaskRequest(
            task_id="task-web",
            trace_id="trace-web",
            agent_kind="web",
            request="Check the page",
            metadata={},
        ),
        session_id="sess-web",
        lease=SessionLeaseGranted(
            lease_id="lease",
            session_id="sess-web",
            task_id="task-web",
            device_id="browser-1",
            trace_id="trace-web",
        ),
    )

    async def _run() -> TaskFinished:
        runner = WebAgentKitRemoteTaskRunner(
            config=AgentKitRemoteRunnerConfig(
                app_config_path="config/app.yaml",
                command_timeout_seconds=11.0,
                max_steps_per_task=4,
            ),
            sdk=object(),  # type: ignore[arg-type]
        )
        return await runner.run(context)

    finished = asyncio.run(_run())

    assert finished.state.value == "finished"
    assert isinstance(finished.result, dict)
    assert finished.result["evaluation_report"] == {"overall_verdict": "success", "web_blocker": "none"}
    assert finished.artifacts[-1].artifact_id == "task-web-evaluation-web_round_evaluator-round_0001-final_report"
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["evaluation_report"] == {"overall_verdict": "success", "web_blocker": "none"}
    assert persisted["evaluation_report_path"].endswith("final_report.json")
    assert persisted["evaluation_report_path"] != "/tmp/stale-final-report.json"
    result_artifact = next(artifact for artifact in finished.artifacts if artifact.artifact_id == "task-web-result")
    result_bytes = result_path.read_bytes()
    assert result_artifact.sha256 == hashlib.sha256(result_bytes).hexdigest()
    assert result_artifact.size_bytes == len(result_bytes)


def test_web_round_evaluator_helper_honors_round_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEvaluationConfig:
        round_timeout_seconds = 0.01

    class _FakeAppConfig:
        evaluation = _FakeEvaluationConfig()

    class _FakeArtifactStore:
        root = tmp_path
        session_id = "sess-web"

    class _FakeApp:
        app_config = _FakeAppConfig()
        artifact_store = _FakeArtifactStore()

    class _HangingEvaluatorService:
        def __init__(self, *, model_step_executor: object) -> None:
            del model_step_executor

        async def evaluate_artifact_root(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            await asyncio.sleep(10)
            return object()

    monkeypatch.setattr(web_eval_helpers, "_build_model_step_executor", lambda _app: object())
    monkeypatch.setattr(
        "northcua.agents.web.evaluation.service.WebRoundEvaluatorService",
        _HangingEvaluatorService,
    )

    with pytest.raises(TimeoutError, match="Web evaluator timed out after 0.01s"):
        asyncio.run(web_eval_helpers.run_web_round_evaluator(_FakeApp(), remote=True))


def test_web_round_evaluator_retries_transient_llm_capacity_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeEvaluationConfig:
        retry_delays_seconds = (0.0,)

    class _FakeAppConfig:
        evaluation = _FakeEvaluationConfig()

    class _FakeArtifactStore:
        root = tmp_path
        session_id = "sess-web"

    class _FakeApp:
        app_config = _FakeAppConfig()
        artifact_store = _FakeArtifactStore()

    attempts: list[int] = []

    class _FlakyEvaluatorService:
        def __init__(self, *, model_step_executor: object) -> None:
            del model_step_executor

        async def evaluate_artifact_root(self, artifact_root: Path, **kwargs: object) -> object:
            del kwargs
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("provider returned 500: 没有可用token")
            report_dir = artifact_root / "evaluations" / "web_round_evaluator" / "round_0001"
            report_dir.mkdir(parents=True)
            (report_dir / "final_report.json").write_text('{"overall_verdict":"success"}\n', encoding="utf-8")
            return object()

    monkeypatch.setattr(web_eval_helpers, "_build_model_step_executor", lambda _app: object())
    monkeypatch.setattr(
        "northcua.agents.web.evaluation.service.WebRoundEvaluatorService",
        _FlakyEvaluatorService,
    )

    outcome = asyncio.run(web_eval_helpers.run_web_round_evaluator(_FakeApp(), remote=True))

    assert len(attempts) == 2
    assert outcome.report["overall_verdict"] == "success"


def test_web_evaluation_artifact_ids_include_evaluator_kind(tmp_path: Path) -> None:
    artifact_root = tmp_path / "session"
    round_root = artifact_root / "evaluations" / "web_round_evaluator" / "round_0001"
    round_root.mkdir(parents=True)
    (round_root / "final_report.json").write_text('{"overall_verdict":"success"}\n', encoding="utf-8")

    refs = collect_web_evaluation_artifacts(artifact_root=artifact_root, task_id="task-web")

    assert [ref.artifact_id for ref in refs] == ["task-web-evaluation-web_round_evaluator-round_0001-final_report"]
