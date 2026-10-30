"""Connection-level tests for standalone device workers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from northcua_device.config import DeviceClientConfig
from northcua_device.connection import DeviceConnection, DeviceSendStats
from northcua_device.runtime.desktop import DesktopRuntimeAdapter
from northcua_device.runtime.shell import EchoRuntimeAdapter
from northcua_device.worker import DeviceWorker
from northcua_protocol.capabilities import Capability, PlatformKind, ToolDescriptor, ToolRiskLevel
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope, parse_envelope
from northcua_protocol.payloads import (
    CommandRequest,
    CommandResult,
    DeviceDrain,
    PingPayload,
    SessionLeaseGranted,
    SessionLeaseReleased,
)
from northcua_protocol.state import CommandState
from northcua_runtime_shared import RuntimeHealth
from northcua_runtime_shared.tool_contracts import (
    DeviceCommandBinding,
    ToolAudience,
    ToolContract,
    ToolPlacement,
    ToolVisibility,
    define_tool,
)


def test_device_connection_disables_proxy_and_transport_ping_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _connect(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("northcua_device.connection.websockets.connect", _connect)

    connection = DeviceConnection("wss://controller.example/ws", max_message_bytes=456)
    asyncio.run(connection.connect())

    assert captured["args"] == ("wss://controller.example/ws",)
    assert captured["kwargs"] == {"proxy": None, "max_size": 456, "ping_interval": None, "ping_timeout": None}


def test_device_connection_transport_ping_can_be_disabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _connect(*args, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("northcua_device.connection.websockets.connect", _connect)
    monkeypatch.setenv("NORTHCUA_WEBSOCKET_CLIENT_PING_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("NORTHCUA_WEBSOCKET_CLIENT_PING_TIMEOUT_SECONDS", "0")

    connection = DeviceConnection("wss://controller.example/ws", max_message_bytes=456)
    asyncio.run(connection.connect())

    assert captured["kwargs"]["ping_interval"] is None
    assert captured["kwargs"]["ping_timeout"] is None


def test_device_connection_send_reports_envelope_bytes() -> None:
    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, raw: str) -> None:
            self.sent.append(raw)

    async def _run() -> None:
        websocket = _FakeWebSocket()
        connection = DeviceConnection("wss://controller.example/ws")
        connection._connection = websocket  # type: ignore[assignment]

        stats = await connection.send(Envelope.wrap(MessageType.PING, PingPayload(nonce="n-1"), trace_id="trace-1"))

        assert websocket.sent
        assert stats.envelope_bytes == len(websocket.sent[0].encode("utf-8"))
        assert stats.elapsed_ms >= 0

    asyncio.run(_run())


def test_device_connection_close_waits_for_inflight_send() -> None:
    class _SlowWebSocket:
        def __init__(self) -> None:
            self.send_started = asyncio.Event()
            self.release_send = asyncio.Event()
            self.order: list[str] = []

        async def send(self, raw: str) -> None:
            del raw
            self.order.append("send_started")
            self.send_started.set()
            await self.release_send.wait()
            self.order.append("send_finished")

        async def close(self) -> None:
            self.order.append("close")

    async def _run() -> None:
        websocket = _SlowWebSocket()
        connection = DeviceConnection("wss://controller.example/ws")
        connection._connection = websocket  # type: ignore[assignment]
        send_task = asyncio.create_task(
            connection.send(Envelope.wrap(MessageType.PING, PingPayload(nonce="n-1"), trace_id="trace-1"))
        )
        await websocket.send_started.wait()
        close_task = asyncio.create_task(connection.close())
        await asyncio.sleep(0)
        websocket.release_send.set()
        await asyncio.gather(send_task, close_task)

        assert websocket.order == ["send_started", "send_finished", "close"]
        assert connection._connection is None

    asyncio.run(_run())


class _ScriptedConnection:
    """In-memory controller connection used by worker connection tests."""

    def __init__(self, envelopes: list[Envelope]) -> None:
        self.envelopes = envelopes
        self.sent: list[Envelope] = []
        self.connect_calls = 0
        self.closed = False

    async def connect(self) -> None:
        """Record one connection attempt."""

        self.connect_calls += 1

    async def close(self) -> None:
        """Mark the connection closed."""

        self.closed = True

    async def send(self, envelope: Envelope) -> None:
        """Capture envelopes sent by the worker."""

        self.sent.append(envelope)

    async def messages(self) -> AsyncIterator[Envelope]:
        """Yield scripted controller envelopes once."""

        for envelope in self.envelopes:
            yield envelope


class _ClickServer:
    """Fake MCP action server that records click commands per sandbox."""

    def __init__(self, sandbox_id: str) -> None:
        self.sandbox_id = sandbox_id
        self.calls: list[dict[str, object]] = []

    async def call_tool(self, tool_name: str, parameters: dict[str, object]) -> dict[str, object]:
        """Return a deterministic fake MCP action result."""

        self.calls.append({"tool_name": tool_name, "parameters": dict(parameters)})
        return {"sandbox_id": self.sandbox_id, "tool_name": tool_name, "parameters": dict(parameters)}


def _lease(*, device_id: str, session_id: str, lease_id: str, task_id: str) -> SessionLeaseGranted:
    """Build a lease envelope payload for a device worker."""

    return SessionLeaseGranted(
        lease_id=lease_id,
        session_id=session_id,
        task_id=task_id,
        device_id=device_id,
        trace_id=f"trace-{device_id}",
    )


def _command(*, device_id: str, session_id: str, lease_id: str, x: int) -> CommandRequest:
    """Build a click command payload for a device worker."""

    return CommandRequest(
        command_id=f"cmd-{device_id}",
        idempotency_key=f"idem-{device_id}",
        session_id=session_id,
        lease_id=lease_id,
        device_id=device_id,
        tool_id="click",
        parameters={"x": x, "y": 20},
    )


def _click_adapter(server: _ClickServer) -> DesktopRuntimeAdapter:
    """Build a desktop adapter backed by a fake MCP action server."""

    click_descriptor = ToolDescriptor(
        tool_id="click",
        capability=Capability.DESKTOP_INPUT,
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
        side_effect=True,
        requires_focus=True,
        risk_level=ToolRiskLevel.MEDIUM,
    )
    return DesktopRuntimeAdapter(
        servers_by_name={"DesktopNativeActionExecutor": server},
        tool_contracts=[_contract_from_descriptor(click_descriptor)],
        tool_server_names={"click": "DesktopNativeActionExecutor"},
        normalize_result=lambda result: result if isinstance(result, dict) else {"result": str(result)},
    )


class _ContractBackedTestAdapter:
    @property
    def tools(self) -> list[ToolDescriptor]:
        raise NotImplementedError

    @property
    def tool_contracts(self) -> list[ToolContract]:
        return [_contract_from_descriptor(tool) for tool in self.tools]


def _contract_from_descriptor(descriptor: ToolDescriptor) -> ToolContract:
    return define_tool(
        tool_id=descriptor.tool_id,
        public_name=descriptor.tool_id,
        namespace="tests.device",
        audience=ToolAudience.RUNTIME,
        placement=ToolPlacement.DEVICE_COMMAND,
        visibility=ToolVisibility.RUNTIME_ONLY,
        capability=descriptor.capability,
        json_schema=descriptor.parameters_schema,
        timeout_seconds=descriptor.timeout_seconds,
        side_effect=descriptor.side_effect,
        requires_focus=descriptor.requires_focus,
        risk_level=descriptor.risk_level,
        max_payload_bytes=descriptor.max_payload_bytes,
        binding=DeviceCommandBinding(server_name="tests.device"),
        source="tests.device.test_device_worker_connections",
    )


def _worker(
    *,
    device_id: str,
    connection: _ScriptedConnection,
    server: _ClickServer,
) -> DeviceWorker:
    """Build a worker that exposes the real desktop MCP ``click`` tool id."""

    return DeviceWorker(
        DeviceClientConfig(
            server_url="ws://controller",
            device_id=device_id,
            platform=PlatformKind.MACOS,
        ),
        adapters=[_click_adapter(server)],
        connection=connection,
    )


def test_two_sandbox_workers_execute_only_their_own_mcp_click_commands() -> None:
    """Two device clients should not mix command routing or MCP action results."""

    async def _run() -> None:
        lease_a = _lease(device_id="sandbox-a", session_id="session-a", lease_id="lease-a", task_id="task-a")
        lease_b = _lease(device_id="sandbox-b", session_id="session-b", lease_id="lease-b", task_id="task-b")
        command_a = _command(device_id="sandbox-a", session_id="session-a", lease_id="lease-a", x=10)
        command_b = _command(device_id="sandbox-b", session_id="session-b", lease_id="lease-b", x=99)
        connection_a = _ScriptedConnection(
            [
                Envelope.wrap(MessageType.SESSION_LEASE_GRANTED, lease_a, trace_id=lease_a.trace_id),
                Envelope.wrap(MessageType.COMMAND_REQUEST, command_a, trace_id=lease_a.trace_id),
            ]
        )
        connection_b = _ScriptedConnection(
            [
                Envelope.wrap(MessageType.SESSION_LEASE_GRANTED, lease_b, trace_id=lease_b.trace_id),
                Envelope.wrap(MessageType.COMMAND_REQUEST, command_b, trace_id=lease_b.trace_id),
            ]
        )
        server_a = _ClickServer("sandbox-a")
        server_b = _ClickServer("sandbox-b")

        await asyncio.gather(
            _worker(device_id="sandbox-a", connection=connection_a, server=server_a).connect_once(),
            _worker(device_id="sandbox-b", connection=connection_b, server=server_b).connect_once(),
        )

        result_a = next(item.payload for item in connection_a.sent if item.type == MessageType.COMMAND_RESULT)
        result_b = next(item.payload for item in connection_b.sent if item.type == MessageType.COMMAND_RESULT)
        assert isinstance(result_a, CommandResult)
        assert isinstance(result_b, CommandResult)
        assert result_a.status == CommandState.SUCCEEDED
        assert result_b.status == CommandState.SUCCEEDED
        assert result_a.device_id == "sandbox-a"
        assert result_b.device_id == "sandbox-b"
        assert result_a.result == {"sandbox_id": "sandbox-a", "tool_name": "click", "parameters": {"x": 10, "y": 20}}
        assert result_b.result == {"sandbox_id": "sandbox-b", "tool_name": "click", "parameters": {"x": 99, "y": 20}}
        assert server_a.calls == [{"tool_name": "click", "parameters": {"x": 10, "y": 20}}]
        assert server_b.calls == [{"tool_name": "click", "parameters": {"x": 99, "y": 20}}]

    asyncio.run(_run())


def test_worker_keeps_reading_control_messages_while_command_runs(tmp_path) -> None:
    class _SlowAdapter(_ContractBackedTestAdapter):
        def __init__(self, *, started: asyncio.Event, release: asyncio.Event) -> None:
            self.started = started
            self.release = release

        @property
        def capabilities(self) -> set[Capability]:
            return {Capability.SHELL_EXEC}

        @property
        def tools(self) -> list[ToolDescriptor]:
            return [
                ToolDescriptor(
                    tool_id="slow.tool",
                    capability=Capability.SHELL_EXEC,
                    timeout_seconds=1.0,
                    side_effect=False,
                    requires_focus=False,
                )
            ]

        async def healthcheck(self) -> RuntimeHealth:
            return RuntimeHealth(healthy=True, capability_health={Capability.DESKTOP_WINDOW: True})

        async def execute(self, command: CommandRequest) -> CommandResult:
            self.started.set()
            await self.release.wait()
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

    class _ResponsiveConnection(_ScriptedConnection):
        def __init__(
            self, *, lease: SessionLeaseGranted, command: CommandRequest, started: asyncio.Event, release: asyncio.Event
        ) -> None:
            super().__init__([])
            self.lease = lease
            self.command = command
            self.started = started
            self.release = release
            self.raw_sent: list[str] = []

        async def send(self, envelope: Envelope) -> None:
            await super().send(envelope)
            if envelope.type == MessageType.PONG:
                self.release.set()

        async def send_raw(self, raw: str) -> DeviceSendStats:
            self.raw_sent.append(raw)
            self.sent.append(parse_envelope(raw))
            return DeviceSendStats(envelope_bytes=len(raw.encode("utf-8")), elapsed_ms=0)

        async def messages(self) -> AsyncIterator[Envelope]:
            yield Envelope.wrap(MessageType.SESSION_LEASE_GRANTED, self.lease, trace_id=self.lease.trace_id)
            yield Envelope.wrap(MessageType.COMMAND_REQUEST, self.command, trace_id=self.lease.trace_id)
            await self.started.wait()
            yield Envelope.wrap(MessageType.PING, PingPayload(nonce="still-responsive"), trace_id=self.lease.trace_id)

    async def _run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        lease = _lease(device_id="dev-queue", session_id="session-queue", lease_id="lease-queue", task_id="task-queue")
        command = CommandRequest(
            command_id="cmd-slow",
            idempotency_key="idem-slow",
            session_id=lease.session_id,
            lease_id=lease.lease_id,
            device_id=lease.device_id,
            tool_id="slow.tool",
            timeout_seconds=1.0,
        )
        connection = _ResponsiveConnection(lease=lease, command=command, started=started, release=release)
        log_dir = tmp_path / "logs"
        worker = DeviceWorker(
            DeviceClientConfig(
                server_url="ws://controller",
                device_id=lease.device_id,
                platform=PlatformKind.MACOS,
                log_dir=str(log_dir),
                telemetry_sample_rate=1.0,
            ),
            adapters=[_SlowAdapter(started=started, release=release)],
            connection=connection,
        )

        await worker.connect_once()

        sent_types = [envelope.type for envelope in connection.sent]
        pong_index = sent_types.index(MessageType.PONG)
        result_index = sent_types.index(MessageType.COMMAND_RESULT)
        assert pong_index < result_index
        assert connection.raw_sent
        stage_events = [
            envelope.payload.event_type
            for envelope in connection.sent
            if envelope.type == MessageType.DEVICE_TELEMETRY and hasattr(envelope.payload, "event_type")
        ]
        assert "device_command_received" in stage_events
        assert "device_driver_call_started" in stage_events
        assert "device_driver_call_finished" in stage_events

        events = [
            json.loads(line)["event"]
            for line in (log_dir / "device_runtime.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        expected_order = [
            "device_command_received",
            "device_command_dequeued",
            "command_started",
            "device_command_execution_started",
            "device_driver_call_started",
            "device_driver_call_finished",
            "command_result_ready",
            "device_result_json_serialize_started",
            "device_result_json_serialize_finished",
            "device_websocket_send_started",
            "device_websocket_send_finished",
            "command_result_sent",
        ]
        positions = [events.index(event) for event in expected_order]
        assert positions == sorted(positions)

    asyncio.run(_run())


def test_worker_preserves_release_then_grant_ordering_for_session_lane(tmp_path) -> None:
    async def _run() -> None:
        first_lease = _lease(device_id="dev-order", session_id="session-old", lease_id="lease-old", task_id="task-old")
        second_lease = _lease(device_id="dev-order", session_id="session-new", lease_id="lease-new", task_id="task-new")
        release_first = SessionLeaseReleased(
            lease_id=first_lease.lease_id,
            session_id=first_lease.session_id,
            device_id=first_lease.device_id,
            reason="finished",
        )
        command = CommandRequest(
            command_id="cmd-new",
            idempotency_key="idem-new",
            session_id=second_lease.session_id,
            lease_id=second_lease.lease_id,
            device_id=second_lease.device_id,
            tool_id="device.echo",
            parameters={"value": "ok"},
        )
        connection = _ScriptedConnection(
            [
                Envelope.wrap(MessageType.SESSION_LEASE_GRANTED, first_lease, trace_id=first_lease.trace_id),
                Envelope.wrap(MessageType.SESSION_LEASE_RELEASED, release_first, trace_id=first_lease.trace_id),
                Envelope.wrap(MessageType.SESSION_LEASE_GRANTED, second_lease, trace_id=second_lease.trace_id),
                Envelope.wrap(MessageType.COMMAND_REQUEST, command, trace_id=second_lease.trace_id),
            ]
        )
        worker = DeviceWorker(
            DeviceClientConfig(
                server_url="ws://controller",
                device_id=second_lease.device_id,
                platform=PlatformKind.MACOS,
                max_sessions=1,
                log_dir=str(tmp_path / "logs-order"),
            ),
            adapters=[EchoRuntimeAdapter()],
            connection=connection,
        )

        await worker.connect_once()

        result = next(item.payload for item in connection.sent if item.type == MessageType.COMMAND_RESULT)
        assert isinstance(result, CommandResult)
        assert result.status == CommandState.SUCCEEDED
        assert worker.actor.active_sessions == 1
        assert worker.actor.session(second_lease.session_id) is not None

    asyncio.run(_run())


def test_worker_preserves_grant_then_drain_ordering_for_session_lane(tmp_path) -> None:
    async def _run() -> None:
        lease = _lease(
            device_id="dev-drain-order", session_id="session-drain", lease_id="lease-drain", task_id="task-drain"
        )
        command = CommandRequest(
            command_id="cmd-drain-order",
            idempotency_key="idem-drain-order",
            session_id=lease.session_id,
            lease_id=lease.lease_id,
            device_id=lease.device_id,
            tool_id="device.echo",
            parameters={"value": "ok"},
        )
        connection = _ScriptedConnection(
            [
                Envelope.wrap(MessageType.SESSION_LEASE_GRANTED, lease, trace_id=lease.trace_id),
                Envelope.wrap(
                    MessageType.DEVICE_DRAIN,
                    DeviceDrain(device_id=lease.device_id, reason="after-grant"),
                    trace_id=lease.trace_id,
                ),
                Envelope.wrap(MessageType.COMMAND_REQUEST, command, trace_id=lease.trace_id),
            ]
        )
        worker = DeviceWorker(
            DeviceClientConfig(
                server_url="ws://controller",
                device_id=lease.device_id,
                platform=PlatformKind.MACOS,
                log_dir=str(tmp_path / "logs-drain-order"),
            ),
            adapters=[EchoRuntimeAdapter()],
            connection=connection,
        )

        await worker.connect_once()

        result = next(item.payload for item in connection.sent if item.type == MessageType.COMMAND_RESULT)
        assert isinstance(result, CommandResult)
        assert result.status == CommandState.SUCCEEDED
        assert worker.actor.session(lease.session_id) is not None

    asyncio.run(_run())


def test_ordered_enqueue_unblocks_when_worker_fails_with_full_queue(tmp_path) -> None:
    async def _run() -> None:
        worker = DeviceWorker(
            DeviceClientConfig(
                server_url="ws://controller",
                device_id="dev-full-queue",
                platform=PlatformKind.MACOS,
                log_dir=str(tmp_path / "logs-full-queue"),
            ),
            adapters=[EchoRuntimeAdapter()],
            connection=_ScriptedConnection([]),
        )
        queue: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=1)
        lease = _lease(
            device_id="dev-full-queue", session_id="session-full", lease_id="lease-full", task_id="task-full"
        )
        await queue.put(Envelope.wrap(MessageType.SESSION_LEASE_GRANTED, lease, trace_id=lease.trace_id))

        async def _failed_worker() -> None:
            raise RuntimeError("command worker died")

        failed_worker = asyncio.create_task(_failed_worker())
        await asyncio.sleep(0)

        try:
            await asyncio.wait_for(
                worker._enqueue_ordered_envelope(
                    queue,
                    Envelope.wrap(
                        MessageType.COMMAND_REQUEST,
                        _command(device_id="dev-full-queue", session_id=lease.session_id, lease_id=lease.lease_id, x=1),
                        trace_id=lease.trace_id,
                    ),
                    failed_worker,
                ),
                timeout=0.2,
            )
        except RuntimeError as exc:
            assert "command worker died" in str(exc)
        else:
            raise AssertionError("enqueue should surface the failed worker instead of hanging on a full queue")

    asyncio.run(_run())


def test_worker_event_log_failures_do_not_break_connection(tmp_path) -> None:
    async def _run() -> None:
        connection = _ScriptedConnection([])
        worker = DeviceWorker(
            DeviceClientConfig(
                server_url="ws://controller",
                device_id="dev-log-failure",
                platform=PlatformKind.MACOS,
                log_dir=str(tmp_path / "logs-failure"),
            ),
            adapters=[EchoRuntimeAdapter()],
            connection=connection,
        )

        def _raise(*args, **kwargs) -> None:
            del args, kwargs
            raise OSError("log sink unavailable")

        worker.events.write = _raise  # type: ignore[method-assign]

        await worker.connect_once()

        assert connection.closed is True
        assert any(envelope.type == MessageType.DEVICE_HELLO for envelope in connection.sent)

    asyncio.run(_run())


def test_worker_command_send_failure_with_backlog_does_not_hang(tmp_path) -> None:
    class _FailingResultConnection(_ScriptedConnection):
        async def send(self, envelope: Envelope) -> None:
            if envelope.type == MessageType.COMMAND_RESULT:
                raise RuntimeError("send failed")
            await super().send(envelope)

    async def _run() -> None:
        lease = _lease(device_id="dev-fail", session_id="session-fail", lease_id="lease-fail", task_id="task-fail")
        first = CommandRequest(
            command_id="cmd-fail-1",
            idempotency_key="idem-fail-1",
            session_id=lease.session_id,
            lease_id=lease.lease_id,
            device_id=lease.device_id,
            tool_id="device.echo",
        )
        second = first.model_copy(update={"command_id": "cmd-fail-2", "idempotency_key": "idem-fail-2"})
        connection = _FailingResultConnection(
            [
                Envelope.wrap(MessageType.SESSION_LEASE_GRANTED, lease, trace_id=lease.trace_id),
                Envelope.wrap(MessageType.COMMAND_REQUEST, first, trace_id=lease.trace_id),
                Envelope.wrap(MessageType.COMMAND_REQUEST, second, trace_id=lease.trace_id),
            ]
        )
        worker = DeviceWorker(
            DeviceClientConfig(
                server_url="ws://controller",
                device_id=lease.device_id,
                platform=PlatformKind.MACOS,
                log_dir=str(tmp_path / "logs-send-failure"),
            ),
            adapters=[EchoRuntimeAdapter()],
            connection=connection,
        )

        try:
            await asyncio.wait_for(worker.connect_once(), timeout=1.0)
        except RuntimeError as exc:
            assert "send failed" in str(exc)
        else:
            raise AssertionError("result send failure should propagate without hanging")

        assert connection.closed is True

    asyncio.run(_run())


def test_connect_mode_retries_after_disconnect() -> None:
    """Long-running workers should reconnect after a clean controller disconnect."""

    class _ReconnectConnection:
        def __init__(self) -> None:
            self.connect_calls = 0

        async def connect(self) -> None:
            self.connect_calls += 1

        async def close(self) -> None:
            return None

        async def send(self, envelope: Envelope) -> None:
            del envelope
            return None

        async def messages(self) -> AsyncIterator[Envelope]:
            if self.connect_calls == 1:
                if False:
                    yield Envelope.wrap(
                        MessageType.PING, PingPayload(nonce="unreachable"), trace_id="unreachable"
                    )  # pragma: no cover
                return
            await asyncio.sleep(10)
            if False:
                yield Envelope.wrap(
                    MessageType.PING, PingPayload(nonce="unreachable"), trace_id="unreachable"
                )  # pragma: no cover

    async def _run() -> None:
        connection = _ReconnectConnection()
        worker = DeviceWorker(
            DeviceClientConfig(
                server_url="ws://controller",
                device_id="dev-reconnect",
                reconnect_initial_delay_seconds=0.01,
                reconnect_max_delay_seconds=0.01,
            ),
            adapters=[EchoRuntimeAdapter()],
            connection=connection,
        )
        task = asyncio.create_task(worker.run_forever())
        try:
            for _ in range(50):
                if connection.connect_calls >= 2:
                    break
                await asyncio.sleep(0.01)
            assert connection.connect_calls >= 2
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())
