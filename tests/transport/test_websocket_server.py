from __future__ import annotations

import asyncio

from northcua.agents.desktop.server import NorthCUAWebSocketServer
from northcua.core.enums import ResultStatus, TaskStatus, WebSocketMessageType
from northcua.core.models import (
    CommandExecutionResultRecord,
)
from northcua.core.protocol import CommandResultPayload, DesktopTaskRequestPayload, NorthCUAWebSocketEnvelope
from tests.transport.websocket_server_test_utils import _DummyFacade, _DummyTransport, _DummyWebSocket


def test_websocket_server_accepts_task_and_emits_completion() -> None:
    server = NorthCUAWebSocketServer(
        facade=_DummyFacade(),
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8765,
    )
    websocket = _DummyWebSocket()

    envelope = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.TASK_REQUEST,
        DesktopTaskRequestPayload(request="Open Calculator"),
    )

    async def _run() -> None:
        await server._dispatch_envelope(websocket, envelope)
        await asyncio.gather(*server._background_tasks)

    asyncio.run(_run())

    parsed = [NorthCUAWebSocketEnvelope.model_validate_json(message) for message in websocket.messages]
    assert [message.type for message in parsed] == [
        WebSocketMessageType.TASK_ACCEPTED,
        WebSocketMessageType.TASK_FINISHED,
    ]
    assert parsed[-1].payload.result == "done"


def test_websocket_server_preserves_cua_target_metadata_without_prompt_injection() -> None:
    facade = _DummyFacade()
    server = NorthCUAWebSocketServer(
        facade=facade,
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8765,
    )
    websocket = _DummyWebSocket()

    envelope = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.TASK_REQUEST,
        DesktopTaskRequestPayload(
            request="Summarize this window",
            metadata={
                "cua_target": {
                    "pid": 123,
                    "window_id": 456,
                    "app_name": "Notes",
                    "window_title": "Scratch",
                }
            },
        ),
    )

    async def _run() -> None:
        await server._dispatch_envelope(websocket, envelope)
        await asyncio.gather(*server._background_tasks)

    asyncio.run(_run())

    assert facade.run_inputs[0].query == "Summarize this window"
    assert "cua_get_window_state" not in facade.run_inputs[0].query
    assert "post-action window state" not in facade.run_inputs[0].query
    assert "refresh after every action" not in facade.run_inputs[0].query
    assert facade.run_inputs[0].metadata == envelope.payload.metadata


def test_websocket_server_keeps_running_until_terminal_status() -> None:
    server = NorthCUAWebSocketServer(
        facade=_DummyFacade(statuses=[TaskStatus.CONTINUE, TaskStatus.FINISH]),
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8765,
    )
    websocket = _DummyWebSocket()

    envelope = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.TASK_REQUEST,
        DesktopTaskRequestPayload(request="Open Calculator"),
    )

    async def _run() -> None:
        await server._dispatch_envelope(websocket, envelope)
        await asyncio.gather(*server._background_tasks)

    asyncio.run(_run())

    parsed = [NorthCUAWebSocketEnvelope.model_validate_json(message) for message in websocket.messages]
    assert [message.type for message in parsed] == [
        WebSocketMessageType.TASK_ACCEPTED,
        WebSocketMessageType.TASK_FINISHED,
    ]
    assert parsed[-1].payload.result == "done"


def test_websocket_server_records_command_results() -> None:
    facade = _DummyFacade()
    server = NorthCUAWebSocketServer(
        facade=facade,
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8765,
    )
    websocket = _DummyWebSocket()

    envelope = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.COMMAND_RESULT,
        CommandResultPayload(
            session_id="sess-1",
            status=ResultStatus.SUCCESS,
            results=[
                CommandExecutionResultRecord(
                    call_id="call-1",
                    status=ResultStatus.SUCCESS,
                    result={"text": "Clicked"},
                )
            ],
        ),
    )
    asyncio.run(server._dispatch_envelope(websocket, envelope))

    assert "call-1" in facade.rounds.state.pending_command_results
    assert websocket.messages == []


def test_websocket_server_rejects_same_session_concurrent_request() -> None:
    server = NorthCUAWebSocketServer(
        facade=_DummyFacade(),
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8765,
    )
    websocket = _DummyWebSocket()
    server._active_sessions.add("sess-1")
    envelope = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.TASK_REQUEST,
        DesktopTaskRequestPayload(session_id="sess-1", request="Open Calculator"),
    )

    asyncio.run(server._dispatch_envelope(websocket, envelope))

    parsed = [NorthCUAWebSocketEnvelope.model_validate_json(message) for message in websocket.messages]
    assert parsed[0].type == WebSocketMessageType.ERROR
    assert "already running" in parsed[0].payload.error


def test_websocket_server_ignores_cross_connection_command_results() -> None:
    facade = _DummyFacade()
    server = NorthCUAWebSocketServer(
        facade=facade,
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8765,
    )
    websocket_owner = _DummyWebSocket()
    websocket_other = _DummyWebSocket()
    server._session_connections["sess-1"] = websocket_owner
    envelope = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.COMMAND_RESULT,
        CommandResultPayload(
            session_id="sess-1",
            status=ResultStatus.SUCCESS,
            results=[
                CommandExecutionResultRecord(
                    call_id="call-1",
                    status=ResultStatus.SUCCESS,
                    result={"text": "Clicked"},
                )
            ],
        ),
    )

    asyncio.run(server._dispatch_envelope(websocket_other, envelope))

    assert "call-1" not in facade.rounds.state.pending_command_results


def test_websocket_server_awaits_command_results_via_facade_consumption() -> None:
    facade = _DummyFacade()
    server = NorthCUAWebSocketServer(
        facade=facade,
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8765,
    )

    async def _run():
        await facade.record_command_results(
            [
                CommandExecutionResultRecord(
                    call_id="call-1",
                    status=ResultStatus.SUCCESS,
                    result={"text": "Clicked"},
                )
            ]
        )
        return await server._await_command_results(facade, ["call-1"])

    results = asyncio.run(_run())

    assert results[0].call_id == "call-1"
    assert "call-1" not in facade.rounds.state.pending_command_results


def test_websocket_server_creates_session_scoped_facade() -> None:
    server = NorthCUAWebSocketServer(
        facade=_DummyFacade(),
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8765,
    )
    websocket = _DummyWebSocket()

    envelope = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.TASK_REQUEST,
        DesktopTaskRequestPayload(session_id="sess-2", request="Open Calculator"),
    )

    async def _run() -> None:
        await server._dispatch_envelope(websocket, envelope)
        await asyncio.gather(*server._background_tasks)

    asyncio.run(_run())

    assert "sess-2" in server._facades
    parsed = [NorthCUAWebSocketEnvelope.model_validate_json(message) for message in websocket.messages]
    assert parsed[0].payload.session_id == "sess-2"
    assert parsed[-1].payload.session_id == "sess-2"


def test_websocket_server_stop_drains_spawned_facades() -> None:
    root_facade = _DummyFacade(session_id="sess-root")
    spawned_facade = _DummyFacade(session_id="sess-other")
    server = NorthCUAWebSocketServer(
        facade=root_facade,
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        host="127.0.0.1",
        port=8765,
    )
    server._facades["sess-other"] = spawned_facade

    asyncio.run(server.stop())

    assert root_facade.drain_calls == 1
    assert spawned_facade.drain_calls == 1
    assert server._facades == {}
