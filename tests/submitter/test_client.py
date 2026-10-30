from __future__ import annotations

import asyncio
import time

from northcua.submitter.client import ControllerSubmitterClient
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope
from northcua_protocol.payloads import CommandRequest, TaskAccepted, TaskFinished, TelemetryUpdate
from northcua_protocol.payloads.task import TaskStatusFilter
from northcua_protocol.state import TaskState


def test_submitter_client_disables_system_proxy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _connect(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("northcua.submitter.client.websockets.connect", _connect)

    client = ControllerSubmitterClient("ws://127.0.0.1:8765", max_message_bytes=123)
    asyncio.run(client.connect())

    assert captured["args"] == ("ws://127.0.0.1:8765",)
    assert captured["kwargs"] == {
        "proxy": None,
        "open_timeout": 30.0,
        "max_size": 123,
        "ping_interval": None,
        "ping_timeout": None,
    }


def test_submitter_client_ping_keepalive_can_be_disabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _connect(*args, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("northcua.submitter.client.websockets.connect", _connect)
    monkeypatch.setenv("NORTHCUA_WEBSOCKET_CLIENT_PING_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("NORTHCUA_WEBSOCKET_CLIENT_PING_TIMEOUT_SECONDS", "0")

    client = ControllerSubmitterClient("ws://127.0.0.1:8765")
    asyncio.run(client.connect())

    assert captured["kwargs"]["ping_interval"] is None
    assert captured["kwargs"]["ping_timeout"] is None


def test_submitter_client_open_timeout_can_be_overridden(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _connect(*args, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("northcua.submitter.client.websockets.connect", _connect)
    monkeypatch.setenv("NORTHCUA_CONTROLLER_CLIENT_OPEN_TIMEOUT_SECONDS", "0")

    client = ControllerSubmitterClient("ws://127.0.0.1:8765")
    asyncio.run(client.connect())

    assert captured["kwargs"]["open_timeout"] is None


def test_submitter_client_response_timeout_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_CONTROLLER_CLIENT_RESPONSE_TIMEOUT_SECONDS", "0")

    client = ControllerSubmitterClient("ws://127.0.0.1:8765")

    assert client.response_timeout_seconds is None


def test_submitter_client_zero_timeouts_disable_explicitly() -> None:
    client = ControllerSubmitterClient(
        "ws://127.0.0.1:8765",
        open_timeout_seconds=0,
        response_timeout_seconds=0,
    )

    assert client.open_timeout_seconds is None
    assert client.response_timeout_seconds is None


def test_submit_task_request_sent_callback_runs_after_websocket_send() -> None:
    events: list[str] = []
    accepted = TaskAccepted(task_id="task-1", trace_id="trace-1", session_id="session-1")
    client = ControllerSubmitterClient("ws://127.0.0.1:8765")
    client._connection = _SubmitConnection(  # type: ignore[attr-defined]
        events,
        Envelope.wrap(MessageType.TASK_ACCEPTED, accepted, trace_id="trace-1").model_dump_json(),
    )

    async def _run() -> TaskAccepted:
        return await client.submit_task(
            agent_kind="desktop",
            request="do it",
            task_id="task-1",
            trace_id="trace-1",
            on_request_sent=lambda: events.append("request_sent"),
        )

    result = asyncio.run(_run())

    assert result == accepted
    assert events == ["send_start", "send_done", "request_sent", "recv"]


def test_task_statuses_timeout_is_bounded() -> None:
    client = ControllerSubmitterClient("ws://127.0.0.1:8765", response_timeout_seconds=0.01)
    client._connection = _HangingConnection()  # type: ignore[attr-defined]

    async def _run() -> None:
        await client.task_statuses(["task-1"])

    try:
        asyncio.run(_run())
    except TimeoutError:
        return
    raise AssertionError("task_statuses should time out when the controller does not respond")


def test_list_tasks_timeout_is_bounded() -> None:
    client = ControllerSubmitterClient("ws://127.0.0.1:8765", response_timeout_seconds=0.01)
    client._connection = _HangingConnection()  # type: ignore[attr-defined]

    async def _run() -> None:
        await client.list_tasks(TaskStatusFilter(batch_id="batch"))

    try:
        asyncio.run(_run())
    except TimeoutError:
        return
    raise AssertionError("list_tasks should time out when the controller does not respond")


def test_cancel_send_timeout_is_bounded() -> None:
    client = ControllerSubmitterClient("ws://127.0.0.1:8765", response_timeout_seconds=0.01)
    client._connection = _HangingSendConnection()  # type: ignore[attr-defined]

    async def _run() -> None:
        from northcua_protocol.payloads import TaskCancel

        await client.cancel_task(TaskCancel(task_id="task-1", trace_id="trace-1"))

    try:
        asyncio.run(_run())
    except TimeoutError:
        return
    raise AssertionError("cancel_task should time out when websocket send does not return")


def test_wait_for_finished_timeout_is_wall_clock_across_progress_messages() -> None:
    command = CommandRequest(
        command_id="cmd-progress",
        idempotency_key="idem-progress",
        session_id="sess-1",
        lease_id="lease-1",
        device_id="dev-1",
        tool_id="cua_click",
        parameters={"element_index": 15},
    )
    client = ControllerSubmitterClient("ws://127.0.0.1:8765")
    client._connection = _RepeatingConnection(  # type: ignore[attr-defined]
        Envelope.wrap(MessageType.COMMAND_REQUEST, command, trace_id="trace-1").model_dump_json()
    )

    async def _run() -> float:
        started = time.monotonic()
        try:
            await asyncio.wait_for(client.wait_for_finished("task-1", timeout_seconds=0.03), timeout=0.2)
        except TimeoutError:
            return time.monotonic() - started
        raise AssertionError("wait_for_finished should time out without TASK_FINISHED")

    assert asyncio.run(_run()) < 0.12


def test_watch_task_timeout_is_wall_clock_across_progress_messages() -> None:
    command = CommandRequest(
        command_id="cmd-progress",
        idempotency_key="idem-progress",
        session_id="sess-1",
        lease_id="lease-1",
        device_id="dev-1",
        tool_id="cua_click",
        parameters={"element_index": 15},
    )
    client = ControllerSubmitterClient("ws://127.0.0.1:8765")
    client._connection = _RepeatingConnection(  # type: ignore[attr-defined]
        Envelope.wrap(MessageType.COMMAND_REQUEST, command, trace_id="trace-1").model_dump_json()
    )

    async def _run() -> tuple[float, int]:
        started = time.monotonic()
        seen = 0
        try:
            async for _envelope in client.watch_task("task-1", session_id="sess-1", timeout_seconds=0.03):
                seen += 1
        except TimeoutError:
            return time.monotonic() - started, seen
        raise AssertionError("watch_task should time out without TASK_FINISHED")

    elapsed, seen = asyncio.run(_run())
    assert elapsed < 0.12
    assert seen > 0


def test_watch_task_expired_deadline_does_not_create_recv_coroutine() -> None:
    client = ControllerSubmitterClient("ws://127.0.0.1:8765")
    connection = _CountingConnection()
    client._connection = connection  # type: ignore[attr-defined]

    async def _run() -> None:
        async for _envelope in client.watch_task("task-1", timeout_seconds=0):
            raise AssertionError("watch_task should time out before receiving")

    try:
        asyncio.run(_run())
    except TimeoutError:
        pass
    else:
        raise AssertionError("watch_task should time out before receiving")
    assert connection.recv_calls == 0


def test_watch_task_includes_session_scoped_command_progress() -> None:
    command = CommandRequest(
        command_id="cmd-1",
        idempotency_key="idem-1",
        session_id="sess-1",
        lease_id="lease-1",
        device_id="dev-1",
        tool_id="cua_click",
        parameters={"element_index": 15},
    )
    finished = TaskFinished(
        task_id="task-1",
        trace_id="trace-1",
        session_id="sess-1",
        state=TaskState.FINISHED,
        result="done",
    )
    client = ControllerSubmitterClient("ws://127.0.0.1:8765")
    client._connection = _FakeConnection(  # type: ignore[attr-defined]
        [
            Envelope.wrap(MessageType.COMMAND_REQUEST, command, trace_id="trace-1").model_dump_json(),
            Envelope.wrap(MessageType.TASK_FINISHED, finished, trace_id="trace-1").model_dump_json(),
        ]
    )

    async def _run() -> list[MessageType]:
        envelopes = []
        async for envelope in client.watch_task("task-1", session_id="sess-1"):
            envelopes.append(envelope)
        return [envelope.type for envelope in envelopes]

    assert asyncio.run(_run()) == [MessageType.COMMAND_REQUEST, MessageType.TASK_FINISHED]


def test_watch_task_includes_nested_task_telemetry_progress() -> None:
    telemetry = TelemetryUpdate(
        events=[
            {
                "event_type": "desktop_step",
                "task_id": "task-1",
                "session_id": "sess-1",
                "step": {"assistant_text": "I will click the Equals button."},
            }
        ]
    )
    finished = TaskFinished(
        task_id="task-1",
        trace_id="trace-1",
        session_id="sess-1",
        state=TaskState.FINISHED,
        result="done",
    )
    client = ControllerSubmitterClient("ws://127.0.0.1:8765")
    client._connection = _FakeConnection(  # type: ignore[attr-defined]
        [
            Envelope.wrap(MessageType.TELEMETRY_UPDATE, telemetry, trace_id="trace-1").model_dump_json(),
            Envelope.wrap(MessageType.TASK_FINISHED, finished, trace_id="trace-1").model_dump_json(),
        ]
    )

    async def _run() -> list[MessageType]:
        envelopes = []
        async for envelope in client.watch_task("task-1", session_id="sess-1"):
            envelopes.append(envelope)
        return [envelope.type for envelope in envelopes]

    assert asyncio.run(_run()) == [MessageType.TELEMETRY_UPDATE, MessageType.TASK_FINISHED]


def test_watch_task_does_not_finish_on_foreign_task_in_same_session() -> None:
    foreign_finished = TaskFinished(
        task_id="task-other",
        trace_id="trace-other",
        session_id="sess-1",
        state=TaskState.FINISHED,
        result="wrong task",
    )
    finished = TaskFinished(
        task_id="task-1",
        trace_id="trace-1",
        session_id="sess-1",
        state=TaskState.FINISHED,
        result="done",
    )
    client = ControllerSubmitterClient("ws://127.0.0.1:8765")
    client._connection = _FakeConnection(  # type: ignore[attr-defined]
        [
            Envelope.wrap(MessageType.TASK_FINISHED, foreign_finished, trace_id="trace-other").model_dump_json(),
            Envelope.wrap(MessageType.TASK_FINISHED, finished, trace_id="trace-1").model_dump_json(),
        ]
    )

    async def _run() -> list[str]:
        task_ids = []
        async for envelope in client.watch_task("task-1", session_id="sess-1"):
            task_ids.append(envelope.payload.task_id)
        return task_ids

    assert asyncio.run(_run()) == ["task-1"]


class _FakeConnection:
    def __init__(self, messages: list[str]) -> None:
        self._messages = messages

    async def recv(self) -> str:
        return self._messages.pop(0)


class _SubmitConnection:
    def __init__(self, events: list[str], accepted_message: str) -> None:
        self.events = events
        self.accepted_message = accepted_message

    async def send(self, _raw: str) -> None:
        self.events.append("send_start")
        await asyncio.sleep(0)
        self.events.append("send_done")

    async def recv(self) -> str:
        self.events.append("recv")
        return self.accepted_message


class _RepeatingConnection:
    def __init__(self, message: str, *, delay_seconds: float = 0.005) -> None:
        self._message = message
        self._delay_seconds = delay_seconds

    async def recv(self) -> str:
        await asyncio.sleep(self._delay_seconds)
        return self._message


class _CountingConnection:
    recv_calls = 0

    async def recv(self) -> str:
        self.recv_calls += 1
        await asyncio.Future()
        return ""


class _HangingConnection:
    async def send(self, _raw: str) -> None:
        return None

    async def recv(self) -> str:
        await asyncio.Future()
        return ""


class _HangingSendConnection:
    async def send(self, _raw: str) -> None:
        await asyncio.Future()
