"""Evaluator remote/client tool and configuration safety tests."""

from __future__ import annotations

import asyncio

import pytest
from nexau.core.messages import ImageBlock, TextBlock, ToolResultBlock
from pydantic import ValidationError

from northcua.agentkit.evaluator import (
    EvaluatorArtifactLayout,
    EvaluatorSpec,
    EvaluatorToolRef,
    RemoteEvaluatorClientToolDispatcher,
    UnavailableEvaluatorClientToolDispatcher,
    standard_client_gui_tools,
    standard_client_probe_tools,
)
from northcua.agents.desktop.evaluation.spec import desktop_round_evaluator_spec
from northcua.core.config import NorthCUAAppConfig
from northcua.core.enums import EvaluationToolMode, EvaluationToolPlacement, ResultStatus
from northcua.core.models import CommandExecutionResultRecord
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult
from northcua_runtime_shared.tool_contracts import ToolPlacement, define_tool


def test_evaluator_spec_rejects_duplicate_or_reserved_tool_names() -> None:
    with pytest.raises(ValueError, match="Duplicate evaluator tool names"):
        EvaluatorSpec.with_default_tools(
            kind="fake_eval",
            display_name="Fake Evaluator",
            tools=[
                EvaluatorToolRef.from_contract(
                    define_tool(
                        tool_id="duplicate_read_step",
                        public_name="read_step",
                        description="duplicate",
                        placement=ToolPlacement.EVALUATOR_ARTIFACT,
                        json_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    ),
                    placement=EvaluationToolPlacement.SERVER_ARTIFACT,
                )
            ],
        )
    with pytest.raises(ValueError, match="reserved"):
        EvaluatorToolRef.from_contract(
            define_tool(
                tool_id="bad_final_report",
                public_name="final_report",
                description="bad",
                placement=ToolPlacement.EVALUATOR_ARTIFACT,
                json_schema={"type": "object", "properties": {}, "additionalProperties": False},
            ),
            placement=EvaluationToolPlacement.SERVER_ARTIFACT,
        )


def test_evaluator_config_rejects_invalid_kind_and_screenshot_limit() -> None:
    with pytest.raises(ValidationError):
        NorthCUAAppConfig(
            agent_config_path="agent.yaml",
            evaluation={"default_evaluator_kind": ""},
        )
    with pytest.raises(ValidationError):
        NorthCUAAppConfig(
            agent_config_path="agent.yaml",
            evaluation={"evidence": {"screenshot_max": -1}},
        )
    with pytest.raises(ValidationError):
        NorthCUAAppConfig(
            agent_config_path="agent.yaml",
            evaluation={"default_evaluator_kind": "../../escape"},
        )


def test_standard_client_probe_bundle_dispatches_through_remote_path() -> None:
    dispatched: list[dict[str, object]] = []

    async def _dispatch(pending):
        dispatched.extend(pending)
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            tool_blocks=[
                ToolResultBlock(
                    tool_use_id=call_id,
                    content=[ImageBlock(url="data:image/png;base64,AAAA"), TextBlock(text="captured")],
                    is_error=False,
                )
            ],
            command_results=[
                CommandExecutionResultRecord(
                    call_id=f"session::lease::{call_id}",
                    status=ResultStatus.SUCCESS,
                    result={"screenshot_url": "data:image/png;base64,AAAA", "text": "captured"},
                )
            ],
        )

    dispatcher = RemoteEvaluatorClientToolDispatcher(_dispatch)
    screenshot_tool = standard_client_probe_tools()[0]

    result = asyncio.run(dispatcher.dispatch(tool=screenshot_tool, parameters={"all_screens": False}, remote=True))

    assert dispatched[0]["name"] == "capture_desktop_screenshot"
    assert dispatched[0]["server_name"] == "DesktopNativeDataExecutor"
    assert dispatched[0]["input"] == {"all_screens": False}
    assert result["status"] == "success"
    assert result["result"] == {"screenshot_url": "[image data omitted; chars=26]", "text": "captured"}


def test_unavailable_client_tool_dispatcher_never_executes_live_tool() -> None:
    dispatcher = UnavailableEvaluatorClientToolDispatcher()
    result = dispatcher.dispatch(
        tool=standard_client_probe_tools()[0],
        parameters={"all_screens": False},
        remote=True,
    )

    assert result == {
        "status": "failure",
        "tool": "client_screenshot",
        "error": "Client tools are unavailable during artifact-only replay.",
        "error_type": "ClientToolUnavailable",
    }


def test_remote_client_tool_dispatcher_strips_mixed_case_inline_images() -> None:
    dispatched: list[dict[str, object]] = []

    async def _dispatch(pending):
        dispatched.extend(pending)
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={"screenshot_url": "data:IMAGE/PNG;base64,AAAA", "text": "captured"},
                )
            ],
        )

    dispatcher = RemoteEvaluatorClientToolDispatcher(_dispatch)
    screenshot_tool = standard_client_probe_tools()[0]

    result = asyncio.run(dispatcher.dispatch(tool=screenshot_tool, parameters={"all_screens": False}, remote=True))

    assert result["status"] == "success"
    assert result["result"] == {"screenshot_url": "[image data omitted; chars=26]", "text": "captured"}


def test_remote_client_tool_dispatcher_redacts_result_path_fields_without_corrupting_urls() -> None:
    async def _dispatch(pending):
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={
                        "url": "https://example.test/path",
                        "screenshot_path": "/tmp/private-shot.png",
                        "capture_error": "failed at /tmp/private-shot.png",
                        "image_omitted_reason": "failed at /tmp/private-shot.png",
                        "nested": {"image_path": "C:\\secret\\shot.png"},
                        "text": "captured at /tmp/private-shot.png with data:IMAGE/PNG;base64,AAAA",
                    },
                )
            ],
        )

    dispatcher = RemoteEvaluatorClientToolDispatcher(_dispatch)
    screenshot_tool = standard_client_probe_tools()[0]

    result = asyncio.run(dispatcher.dispatch(tool=screenshot_tool, parameters={"all_screens": False}, remote=True))

    assert result["status"] == "success"
    assert result["result"] == {
        "url": "https://example.test/path",
        "screenshot_path": "[path omitted]",
        "capture_error": "[redacted capture_error]",
        "capture_error_present": True,
        "image_omitted_reason": "[redacted image_omitted_reason]",
        "image_omitted_reason_present": True,
        "nested": {"image_path": "[path omitted]"},
        "text": "captured at [path omitted] with [image data omitted; chars=26]",
    }


def test_remote_client_tool_dispatcher_redacts_failure_error_media_and_paths() -> None:
    dispatched: list[dict[str, object]] = []

    async def _dispatch(pending):
        dispatched.extend(pending)
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.FAILURE,
                    result={"ok": False},
                    error="failed data:IMAGE/PNG;base64,AAAA at /tmp/private.png",
                )
            ],
        )

    dispatcher = RemoteEvaluatorClientToolDispatcher(_dispatch)
    screenshot_tool = standard_client_probe_tools()[0]

    result = asyncio.run(dispatcher.dispatch(tool=screenshot_tool, parameters={"all_screens": False}, remote=True))

    assert result["status"] == "failure"
    assert result["error"] == "failed [image data omitted; chars=26] at [path omitted]"


def test_remote_client_tool_dispatcher_redacts_fallback_tool_block_text() -> None:
    dispatched: list[dict[str, object]] = []

    async def _dispatch(pending):
        dispatched.extend(pending)
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            tool_blocks=[
                ToolResultBlock(
                    tool_use_id=call_id,
                    content=[TextBlock(text="fallback data:IMAGE/PNG;base64,AAAA at /tmp/private.png")],
                    is_error=True,
                )
            ],
        )

    dispatcher = RemoteEvaluatorClientToolDispatcher(_dispatch)
    screenshot_tool = standard_client_probe_tools()[0]

    result = asyncio.run(dispatcher.dispatch(tool=screenshot_tool, parameters={"all_screens": False}, remote=True))

    assert result["status"] == "failure"
    assert result["text"] == "fallback [image data omitted; chars=26] at [path omitted]"


def test_standard_client_probe_shell_tools_dispatch_to_device_shell() -> None:
    dispatched: list[dict[str, object]] = []

    async def _dispatch(pending):
        dispatched.extend(pending)
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={"stdout": "ok\n", "stderr": "", "returncode": 0},
                )
            ],
        )

    tools_by_name = {tool.name: tool for tool in standard_client_probe_tools()}
    shell_contract = tools_by_name["client_bash_readonly"].contract
    assert set(shell_contract.parameters_schema.get("properties", {})) == {"cmd"}
    assert "bash_command" not in shell_contract.parameters_schema.get("properties", {})
    dispatcher = RemoteEvaluatorClientToolDispatcher(_dispatch)

    bash_result = asyncio.run(
        dispatcher.dispatch(
            tool=tools_by_name["client_bash_readonly"],
            parameters={"cmd": "pwd"},
            remote=True,
        )
    )

    assert dispatched[0]["name"] == "shell.exec_readonly"
    assert dispatched[0]["server_name"] == "DeviceShellExecutor"
    assert dispatched[0]["input"] == {"cmd": "pwd"}
    assert bash_result["status"] == "success"
    assert "client_python_readonly" not in tools_by_name


@pytest.mark.parametrize(
    "bash_command",
    [
        "touch /tmp/evaluator-write",
        "env sh -c 'touch /tmp/evaluator-write'",
        "date -s '2020-01-01'",
    ],
)
def test_standard_client_probe_shell_tools_reject_mutating_commands_before_dispatch(bash_command: str) -> None:
    dispatched: list[dict[str, object]] = []

    async def _dispatch(pending):
        dispatched.extend(pending)
        return RemoteToolDispatchResult()

    tools_by_name = {tool.name: tool for tool in standard_client_probe_tools()}
    dispatcher = RemoteEvaluatorClientToolDispatcher(_dispatch)

    with pytest.raises(ValueError, match="readonly shell command"):
        asyncio.run(
            dispatcher.dispatch(
                tool=tools_by_name["client_bash_readonly"],
                parameters={"cmd": bash_command},
                remote=True,
            )
        )

    assert dispatched == []


def test_standard_client_probe_shell_tools_reject_python_write_bypass_before_dispatch() -> None:
    dispatched: list[dict[str, object]] = []

    async def _dispatch(pending):
        dispatched.extend(pending)
        return RemoteToolDispatchResult()

    tools_by_name = {tool.name: tool for tool in standard_client_probe_tools()}
    dispatcher = RemoteEvaluatorClientToolDispatcher(_dispatch)

    with pytest.raises(ValueError, match="executable 'python3' is not allowlisted"):
        asyncio.run(
            dispatcher.dispatch(
                tool=tools_by_name["client_bash_readonly"],
                parameters={
                    "cmd": (
                        'python3 -c \'import pathlib; p=pathlib.Path("/tmp/northcua_pwn"); '
                        'getattr(p,"write_text")("x")\''
                    )
                },
                remote=True,
            )
        )

    assert dispatched == []


def test_standard_client_gui_tools_dispatch_to_desktop_device_tools() -> None:
    dispatched: list[dict[str, object]] = []

    async def _dispatch(pending):
        dispatched.extend(pending)
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={"ok": True},
                )
            ],
        )

    tools_by_name = {tool.name: tool for tool in standard_client_gui_tools()}
    dispatcher = RemoteEvaluatorClientToolDispatcher(_dispatch)

    result = asyncio.run(
        dispatcher.dispatch(
            tool=tools_by_name["client_gui_click"],
            parameters={"x": 12, "y": 34, "button": "left"},
            remote=True,
        )
    )

    assert dispatched[0]["name"] == "click"
    assert dispatched[0]["server_name"] == "DesktopNativeActionExecutor"
    assert dispatched[0]["input"] == {"x": 12, "y": 34, "button": "left"}
    assert result["status"] == "success"


def test_desktop_evaluator_exposes_configured_probe_and_gui_tools() -> None:
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.yaml",
        evaluation={
            "tools": {
                "allow_client_live_probe": True,
                "allow_client_mutating_tools": True,
                "allowed_placements": ["server_artifact", "client_probe", "client_mutating", "finalizer"],
                "client_probe_tools": ["client_bash_readonly"],
                "client_gui_tools": ["client_gui_click"],
            }
        },
    )

    spec = desktop_round_evaluator_spec(app_config)
    visible_names = {tool.name for tool in spec.visible_tools()}

    assert "client_bash_readonly" in visible_names
    assert "client_gui_click" in visible_names
    assert "client_gui_type" not in visible_names


def test_desktop_evaluator_does_not_expose_ocr_tool() -> None:
    spec = desktop_round_evaluator_spec(NorthCUAAppConfig(agent_config_path="agent.yaml"))
    visible_names = {tool.name for tool in spec.visible_tools()}

    assert "compare_images" in visible_names
    assert "extract_image_text_summary" not in visible_names


def test_desktop_evaluator_llm_selected_mode_defaults_to_standard_live_tools() -> None:
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.yaml",
        evaluation={
            "tools": {
                "mode": EvaluationToolMode.ALLOW_LLM_SELECTED_TOOLS,
                "allow_client_live_probe": True,
                "allow_client_mutating_tools": True,
                "allowed_placements": ["server_artifact", "client_probe", "client_mutating", "finalizer"],
            }
        },
    )

    spec = desktop_round_evaluator_spec(app_config)
    visible_names = {tool.name for tool in spec.visible_tools()}

    assert {"client_bash_readonly", "client_gui_screenshot", "client_gui_click"}.issubset(visible_names)


def test_evaluator_client_mutating_tools_require_separate_gate() -> None:
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.yaml",
        evaluation={
            "tools": {
                "allow_client_live_probe": True,
                "allowed_placements": ["server_artifact", "client_probe", "client_mutating", "finalizer"],
                "client_gui_tools": ["client_gui_click"],
            }
        },
    )

    spec = desktop_round_evaluator_spec(app_config)
    visible_names = {tool.name for tool in spec.visible_tools()}

    assert "client_gui_click" not in visible_names


def test_evaluator_artifact_layout_rejects_escaping_kind(tmp_path) -> None:
    with pytest.raises(ValueError, match="Invalid evaluator kind"):
        EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="../bad")
