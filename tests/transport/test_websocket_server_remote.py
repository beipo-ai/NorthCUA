from __future__ import annotations

import asyncio
import base64

from northcua.agents.desktop.runtime import DesktopTaskRunResult
from northcua.agents.desktop.server import NorthCUAWebSocketServer
from northcua.core.enums import ResultStatus, RuntimeMode, TaskStatus
from northcua.core.models import (
    CommandExecutionResultRecord,
    DesktopStateSnapshot,
    LLMTraceRecord,
    ObservationContext,
    StepLogRecord,
    WindowState,
)
from northcua.runtime.desktop import DesktopObservationService
from northcua.transport.ws.result_blocks import command_results_to_tool_execution_records
from tests.transport.websocket_server_test_utils import _DummyFacade, _DummyTransport, _DummyWebSocket


def test_remote_server_injects_fresh_observation_provider_into_shared_runtime(monkeypatch) -> None:
    server = NorthCUAWebSocketServer(
        facade=_DummyFacade(statuses=[TaskStatus.CONTINUE, TaskStatus.FINISH]),
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8765,
        runtime_mode=RuntimeMode.REMOTE,
    )
    websocket = _DummyWebSocket()
    facade = server.facade
    snapshots = [
        DesktopStateSnapshot(platform_name="remote", clean_screenshot_url="data:image/png;base64,AAAA"),
        DesktopStateSnapshot(platform_name="remote", clean_screenshot_url="data:image/png;base64,BBBB"),
    ]
    snapshot_calls = 0

    async def _collect_remote_observation_context(_websocket, _facade):
        nonlocal snapshot_calls
        snapshot = snapshots[snapshot_calls]
        snapshot_calls += 1
        return DesktopObservationService.context_from_snapshot(snapshot)

    server._collect_remote_observation_context = _collect_remote_observation_context  # type: ignore[method-assign]

    observed_contexts: list[ObservationContext] = []

    async def _run_round_input(self, round_input, *, options, **kwargs):
        del self, kwargs
        assert options.observation_provider is not None
        observed_contexts.append(await options.observation_provider.collect(session_step=1, previous_steps=[]))
        observed_contexts.append(await options.observation_provider.collect(session_step=2, previous_steps=[]))
        step = StepLogRecord(
            session_id=facade.rounds.state.session_id,
            session_step=0,
            round_step=0,
            round_num=1,
            status=TaskStatus.FINISH,
            request=round_input.query,
            result="done",
            llm_trace=LLMTraceRecord(),
            assistant_text="done",
            current_window=WindowState(),
        )
        return DesktopTaskRunResult(steps=[step], terminal_status=TaskStatus.FINISH)

    monkeypatch.setattr(
        "northcua.agents.desktop.server.DesktopTaskRuntime.run_round_input",
        _run_round_input,
    )
    facade.rounds.enqueue_initial_request("Open Calculator")
    asyncio.run(server._run_pending_rounds(websocket, facade))

    assert snapshot_calls == 2
    assert [context.current_clean_screenshot_url for context in observed_contexts] == [
        "data:image/png;base64,AAAA",
        "data:image/png;base64,BBBB",
    ]


def test_command_result_to_tool_result_block_supports_image_paths_and_data_urls(tmp_path) -> None:
    png_path = tmp_path / "shot.png"
    png_path.write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aT8kAAAAASUVORK5CYII=")
    )

    path_result = CommandExecutionResultRecord(
        call_id="call-1",
        status=ResultStatus.SUCCESS,
        result={"text": "Clicked", "screenshot_path": str(png_path)},
    )
    block = NorthCUAWebSocketServer._command_result_to_tool_result_block(path_result)
    assert isinstance(block.content, list)
    assert any(getattr(item, "url", "").startswith("data:image/png;base64,") for item in block.content)

    url_result = CommandExecutionResultRecord(
        call_id="call-2",
        status=ResultStatus.SUCCESS,
        result="data:image/png;base64,AAAA",
    )
    url_block = NorthCUAWebSocketServer._command_result_to_tool_result_block(url_result)
    assert isinstance(url_block.content, list)
    assert getattr(url_block.content[0], "url", "") == "data:image/png;base64,AAAA"


def test_command_result_to_tool_result_block_preserves_error_call_id_and_text() -> None:
    result = CommandExecutionResultRecord(
        call_id="call-bad-click",
        status=ResultStatus.FAILURE,
        error="controller-side tool failure",
        result={"status": "error", "error": "Tool 'click' has no implementation."},
    )

    block = NorthCUAWebSocketServer._command_result_to_tool_result_block(result)

    assert block.tool_use_id == "call-bad-click"
    assert block.is_error is True
    assert "Tool 'click' has no implementation" in str(block.content)


def test_command_result_to_tool_result_block_uses_error_field_when_payload_is_empty() -> None:
    result = CommandExecutionResultRecord(
        call_id="call-timeout",
        status=ResultStatus.FAILURE,
        error="COMMAND_TIMED_OUT while executing capture_desktop_screenshot",
        result=None,
    )

    block = NorthCUAWebSocketServer._command_result_to_tool_result_block(result)

    assert block.tool_use_id == "call-timeout"
    assert block.is_error is True
    assert block.content == "COMMAND_TIMED_OUT while executing capture_desktop_screenshot"


def test_command_result_to_tool_result_block_drops_images_from_failed_remote_commands(tmp_path) -> None:
    png_path = tmp_path / "failed-shot.png"
    png_path.write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aT8kAAAAASUVORK5CYII=")
    )
    result = CommandExecutionResultRecord(
        call_id="call-failed-image",
        status=ResultStatus.FAILURE,
        result={
            "status": "error",
            "error": "Click target was outside the screenshot bounds.",
            "screenshot_path": str(png_path),
        },
    )

    block = NorthCUAWebSocketServer._command_result_to_tool_result_block(result)

    assert block.is_error is True
    assert block.content == "Click target was outside the screenshot bounds."


def test_remote_command_result_execution_record_uses_error_field_when_payload_is_empty() -> None:
    records = command_results_to_tool_execution_records(
        pending_tool_calls=[{"id": "call-error", "name": "click", "input": {"x": 999, "y": 2}}],
        results=[
            CommandExecutionResultRecord(
                call_id="call-error",
                status=ResultStatus.FAILURE,
                error="Click target was outside the screenshot bounds.",
                result=None,
            )
        ],
    )

    assert len(records) == 1
    assert records[0].is_error is True
    assert records[0].tool_output == {
        "status": "error",
        "error": "Click target was outside the screenshot bounds.",
    }
    assert records[0].llm_tool_output == records[0].tool_output
