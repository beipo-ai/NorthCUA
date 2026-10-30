"""Tests for controller scheduling, lease, and command-bus behavior."""

from __future__ import annotations

import asyncio
import logging

import websockets
from websockets.exceptions import ConnectionClosedError, InvalidMessage

from northcua.controller.api.server import ControllerServer, _WebsocketServerNoiseFilter
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua.controller.control_plane.leases import LeaseManager
from northcua.controller.control_plane.registry import DeviceRegistry
from northcua.controller.control_plane.scheduler import Scheduler
from northcua.controller.execution.command_bus import command_idempotency_key
from northcua.controller.observability.event_log import ControllerEventLog
from northcua.controller.policy.gui_resource import GuiResourcePolicy
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua_protocol.capabilities import Capability, DeviceDescriptor, PlatformKind, ToolDescriptor
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope, parse_envelope
from northcua_protocol.payloads import (
    CommandRequest,
    CommandResult,
    DeviceHello,
    TaskCancel,
    TaskFinished,
    TaskRequest,
)
from northcua_protocol.state import CommandState, SessionState, TaskState


def _store(tmp_path) -> SQLiteControllerStore:
    return SQLiteControllerStore(tmp_path / "controller.sqlite")


def _descriptor(
    device_id: str,
    *,
    active_focus: bool = False,
    capabilities: set[Capability] | None = None,
    gui_resource_id: str = "",
    max_sessions: int = 2,
    max_gui_sessions: int = 1,
    supports_parallel_sessions: bool | None = None,
    tool_requires_focus: bool = True,
) -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id=device_id,
        platform=PlatformKind.MACOS,
        capabilities=capabilities or {Capability.DESKTOP_INPUT},
        tools=[
            ToolDescriptor(
                tool_id="device.echo",
                capability=Capability.DESKTOP_INPUT,
                requires_focus=tool_requires_focus,
            )
        ],
        max_sessions=max_sessions,
        max_gui_sessions=max_gui_sessions,
        gui_resource_id=gui_resource_id,
        requires_foreground_focus=active_focus,
        supports_parallel_sessions=(
            not active_focus if supports_parallel_sessions is None else supports_parallel_sessions
        ),
    )


class _FakeWebSocket:
    """Minimal websocket stand-in for direct controller dispatch tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)


class _ClosingWebSocket(_FakeWebSocket):
    """Fake websocket that disconnects before sending protocol envelopes."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise ConnectionClosedError(None, None)


class _FailingLeaseWebSocket(_FakeWebSocket):
    """Fake websocket that accepts TASK_ACCEPTED but fails lease delivery."""

    async def send(self, raw: str) -> None:
        if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED:
            raise RuntimeError("lease send failed")
        await super().send(raw)


class _RecordingExecutor:
    """Controller executor test double that records scheduled task contexts."""

    def __init__(self) -> None:
        self.scheduled = []

    async def schedule(self, context) -> None:
        self.scheduled.append(context)

    async def drain(self) -> None:
        return None

    async def cancel_task(self, task_id: str, reason: str) -> None:
        del task_id, reason


async def _recv_until(websocket, message_type: MessageType):
    for _ in range(20):
        envelope = parse_envelope(await asyncio.wait_for(websocket.recv(), timeout=1.0))
        if envelope.type == message_type:
            return envelope
    raise AssertionError(f"did not receive {message_type}")


def test_scheduler_respects_capabilities_and_focus_capacity(tmp_path) -> None:
    store = _store(tmp_path)
    events = ControllerEventLog(store)
    registry = DeviceRegistry(store=store, events=events)
    registry.register(_descriptor("busy-focus", active_focus=True), trace_id="trace")
    registry.register(_descriptor("free"), trace_id="trace")
    registry.get("busy-focus").active_sessions = 1  # type: ignore[union-attr]

    selected = Scheduler(registry).select_device({Capability.DESKTOP_INPUT})

    assert selected.descriptor.device_id == "free"


def test_controller_treats_abrupt_websocket_disconnect_as_debug_log(tmp_path, caplog) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )

        with caplog.at_level(logging.WARNING):
            await server._handle_connection(_ClosingWebSocket())  # type: ignore[arg-type]

        assert "connection handler failed" not in caplog.text
        assert "controller failed to dispatch envelope" not in caplog.text
        server.store.close()

    asyncio.run(_run())


def test_controller_delivers_task_metadata_with_session_lease(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("dev-metadata")),
                trace_id="trace-device",
            ),
        )
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-metadata",
                    trace_id="trace-task",
                    agent_kind="desktop",
                    request="click once",
                    target_device_id="dev-metadata",
                    required_capabilities={Capability.DESKTOP_INPUT},
                    metadata={"allowed_tool_names": ["click"], "tool_choice_policy": "required"},
                ),
                trace_id="trace-task",
            ),
        )

        lease = next(
            parse_envelope(raw).payload
            for raw in device.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        assert lease.metadata == {  # type: ignore[union-attr]
            "allowed_tool_names": ["click"],
            "tool_choice_policy": "required",
        }
        server.store.close()

    asyncio.run(_run())


def test_websocket_noise_filter_drops_empty_probe_handshake_tracebacks() -> None:
    record = logging.LogRecord(
        name="northcua.controller.api.server.websockets",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="opening handshake failed",
        args=(),
        exc_info=(InvalidMessage, InvalidMessage("did not receive a valid HTTP request"), None),
    )

    assert _WebsocketServerNoiseFilter().filter(record) is False


def test_device_disconnect_grace_allows_reconnect_before_interrupt(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                device_disconnect_grace_seconds=0.05,
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-a",
                    trace_id="trace-task",
                    agent_kind="desktop",
                    request="task A",
                    target_device_id="dev-a",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-task",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in device.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )

        server._schedule_device_disconnect_cleanup(
            device_id="dev-a",
            trace_id="trace-disconnect",
            reason="websocket disconnected",
        )
        server.connections.unregister(device)  # type: ignore[arg-type]
        await server._dispatch(
            _FakeWebSocket(),  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-b"),
        )
        await asyncio.sleep(0.08)

        assert server.store.task_row("task-a")["status"] == TaskState.RUNNING.value
        assert server.store.session_row(lease.session_id)["status"] in {  # type: ignore[attr-defined]
            SessionState.LEASED.value,
            SessionState.RUNNING.value,
        }
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert "device_disconnect_grace_started" in event_types
        assert "device_disconnect_grace_recovered" in event_types
        assert "task_interrupted" not in event_types
        server.store.close()

    asyncio.run(_run())


def test_device_disconnect_cleanup_started_is_not_cancelled_by_reconnect(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                device_disconnect_grace_seconds=0.01,
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()

        async def slow_cleanup(*, device_id: str, trace_id: str, reason: str) -> None:
            del device_id, trace_id, reason
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_finished.set()

        monkeypatch.setattr(server, "_cleanup_device_disconnect", slow_cleanup)
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        server.connections.unregister(device)  # type: ignore[arg-type]
        server._schedule_device_disconnect_cleanup(
            device_id="dev-a",
            trace_id="trace-disconnect",
            reason="websocket disconnected",
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)

        reconnect = _FakeWebSocket()
        reconnect_task = asyncio.create_task(
            server._dispatch(  # type: ignore[arg-type]
                reconnect,
                Envelope.wrap(
                    MessageType.DEVICE_HELLO,
                    DeviceHello(descriptor=_descriptor("dev-a")),
                    trace_id="trace-b",
                ),
            )
        )
        await asyncio.sleep(0)
        assert not reconnect_task.done()
        assert server.connections.for_device("dev-a") is None

        allow_cleanup.set()
        await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)
        await asyncio.wait_for(reconnect_task, timeout=1.0)
        active_connection = server.connections.for_device("dev-a")
        assert active_connection is not None
        assert active_connection.websocket is reconnect
        assert "dev-a" not in server._device_disconnect_tasks
        assert "dev-a" not in server._device_disconnect_cleanup_tasks
        server.store.close()

    asyncio.run(_run())


def test_device_disconnect_cleanup_failure_blocks_reconnect_until_retry_succeeds(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                device_disconnect_grace_seconds=0.0,
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        attempts = 0

        async def failing_cleanup(*, device_id: str, trace_id: str, reason: str) -> None:
            del device_id, trace_id, reason
            nonlocal attempts
            attempts += 1
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(server, "_cleanup_device_disconnect", failing_cleanup)
        server._schedule_device_disconnect_cleanup(
            device_id="dev-a",
            trace_id="trace-disconnect",
            reason="websocket disconnected",
        )
        for _ in range(20):
            if server._device_disconnect_cleanup_failures.get("dev-a") == "cleanup failed":  # noqa: SLF001
                break
            await asyncio.sleep(0.01)
        assert server._device_disconnect_cleanup_failures["dev-a"] == "cleanup failed"  # noqa: SLF001

        try:
            await server._dispatch(
                _FakeWebSocket(),  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.DEVICE_HELLO,
                    DeviceHello(descriptor=_descriptor("dev-a")),
                    trace_id="trace-b",
                ),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("reconnect should be rejected while cleanup retry still fails")
        assert server.connections.for_device("dev-a") is None
        assert attempts == 2

        async def successful_cleanup(*, device_id: str, trace_id: str, reason: str) -> None:
            del device_id, trace_id, reason

        monkeypatch.setattr(server, "_cleanup_device_disconnect", successful_cleanup)
        reconnect = _FakeWebSocket()
        await server._dispatch(
            reconnect,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-c"),
        )
        active_connection = server.connections.for_device("dev-a")
        assert active_connection is not None
        assert active_connection.websocket is reconnect
        assert "dev-a" not in server._device_disconnect_cleanup_failures  # noqa: SLF001
        server.store.close()

    asyncio.run(_run())


def test_stop_waits_for_started_device_disconnect_cleanup_before_closing_store(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                device_disconnect_grace_seconds=0.0,
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()
        cleanup_observed_open_store = False

        async def slow_cleanup(*, device_id: str, trace_id: str, reason: str) -> None:
            del device_id, trace_id, reason
            nonlocal cleanup_observed_open_store
            cleanup_started.set()
            await allow_cleanup.wait()
            server.store.connection.execute("SELECT 1").fetchone()
            server.events.write(
                "device_disconnect_cleanup_test_finished",
                trace_id="trace-stop",
                entity_type="device",
                entity_id="dev-a",
            )
            cleanup_observed_open_store = True

        monkeypatch.setattr(server, "_cleanup_device_disconnect", slow_cleanup)
        server._schedule_device_disconnect_cleanup(
            device_id="dev-a",
            trace_id="trace-disconnect",
            reason="websocket disconnected",
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
        stop_task = asyncio.create_task(server.stop())
        await asyncio.sleep(0)
        assert not stop_task.done()
        assert "dev-a" in server._device_disconnect_cleanup_tasks  # noqa: SLF001

        allow_cleanup.set()
        await asyncio.wait_for(stop_task, timeout=1.0)
        assert cleanup_observed_open_store
        assert not server._device_disconnect_cleanup_tasks  # noqa: SLF001

    asyncio.run(_run())


def test_lease_manager_persists_task_session_and_event(tmp_path) -> None:
    store = _store(tmp_path)
    events = ControllerEventLog(store)
    registry = DeviceRegistry(store=store, events=events)
    registry.register(_descriptor("dev-1"), trace_id="trace-1")
    manager = LeaseManager(store=store, scheduler=Scheduler(registry), events=events)
    request = TaskRequest(
        task_id="task-1",
        trace_id="trace-1",
        agent_kind="desktop",
        request="open calculator",
        required_capabilities={Capability.DESKTOP_INPUT},
    )

    lease = manager.acquire_for_task(request, session_id="sess-1")

    assert lease.device_id == "dev-1"
    assert store.session_row("sess-1")["lease_id"] == lease.lease_id
    assert any(row["event_type"] == "lease_granted" for row in store.event_rows())


def test_scheduler_allows_sessions_sharing_one_gui_resource(tmp_path) -> None:
    store = _store(tmp_path)
    events = ControllerEventLog(store)
    registry = DeviceRegistry(store=store, events=events)
    registry.register(_descriptor("dev-a", active_focus=True, gui_resource_id="local-gui"), trace_id="trace-a")
    registry.register(_descriptor("dev-b", active_focus=True, gui_resource_id="local-gui"), trace_id="trace-b")
    scheduler = Scheduler(registry)

    selected = scheduler.select_device({Capability.DESKTOP_INPUT})
    registry.reserve_session(selected.descriptor.device_id)
    second = scheduler.select_device({Capability.DESKTOP_INPUT})

    assert second.descriptor.device_id != selected.descriptor.device_id


def test_scheduler_prefers_gui_resource_owner_when_parallel_sessions_supported(tmp_path) -> None:
    store = _store(tmp_path)
    events = ControllerEventLog(store)
    registry = DeviceRegistry(store=store, events=events)
    registry.register(
        _descriptor(
            "dev-a",
            active_focus=True,
            gui_resource_id="local-gui",
            max_sessions=3,
            max_gui_sessions=3,
            supports_parallel_sessions=True,
        ),
        trace_id="trace-a",
    )
    registry.register(
        _descriptor(
            "dev-b",
            active_focus=True,
            gui_resource_id="local-gui",
            max_sessions=3,
            max_gui_sessions=3,
            supports_parallel_sessions=True,
        ),
        trace_id="trace-b",
    )
    scheduler = Scheduler(registry)

    first = scheduler.select_device({Capability.DESKTOP_INPUT})
    registry.reserve_session(first.descriptor.device_id)
    second = scheduler.select_device({Capability.DESKTOP_INPUT})

    assert first.descriptor.device_id == "dev-a"
    assert second.descriptor.device_id == "dev-a"


def test_scheduler_allows_parallel_distinct_gui_resources(tmp_path) -> None:
    store = _store(tmp_path)
    events = ControllerEventLog(store)
    registry = DeviceRegistry(store=store, events=events)
    registry.register(_descriptor("dev-a", active_focus=True, gui_resource_id="sandbox-a"), trace_id="trace-a")
    registry.register(_descriptor("dev-b", active_focus=True, gui_resource_id="sandbox-b"), trace_id="trace-b")
    scheduler = Scheduler(registry)

    first = scheduler.select_device({Capability.DESKTOP_INPUT})
    registry.reserve_session(first.descriptor.device_id)
    second = scheduler.select_device({Capability.DESKTOP_INPUT})

    assert second.descriptor.device_id != first.descriptor.device_id


def test_gui_policy_locks_only_focus_required_commands(tmp_path) -> None:
    store = _store(tmp_path)
    registry = DeviceRegistry(store=store, events=ControllerEventLog(store))
    registry.register(
        _descriptor("dev-focus", active_focus=True, gui_resource_id="local-gui", tool_requires_focus=True),
        trace_id="trace-focus",
    )
    registry.register(
        _descriptor("dev-probe", active_focus=True, gui_resource_id="local-gui", tool_requires_focus=False),
        trace_id="trace-probe",
    )
    policy = GuiResourcePolicy()
    focus_device = registry.get("dev-focus")
    probe_device = registry.get("dev-probe")

    assert focus_device is not None
    assert probe_device is not None
    assert (
        policy.command_gui_resource_id(
            device=focus_device,
            descriptor=registry.tool_descriptor("dev-focus", "device.echo"),
        )
        == "local-gui"
    )
    assert (
        policy.command_gui_resource_id(
            device=probe_device,
            descriptor=registry.tool_descriptor("dev-probe", "device.echo"),
        )
        == ""
    )


def test_controller_runs_sessions_sharing_gui_resource(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws_a = _FakeWebSocket()
        ws_b = _FakeWebSocket()

        await server._dispatch(
            ws_a,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("dev-a", active_focus=True, gui_resource_id="local-gui")),
                trace_id="trace-a",
            ),
        )
        await server._dispatch(
            ws_b,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("dev-b", active_focus=True, gui_resource_id="local-gui")),
                trace_id="trace-b",
            ),
        )
        await server._dispatch(
            ws_a,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-a",
                    trace_id="trace-task-a",
                    agent_kind="desktop",
                    request="task A",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-task-a",
            ),
        )
        await server._dispatch(
            ws_b,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-b",
                    trace_id="trace-task-b",
                    agent_kind="desktop",
                    request="task B",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-task-b",
            ),
        )

        assert server.store.task_row("task-a")["status"] == TaskState.RUNNING.value
        assert server.store.task_row("task-b")["status"] == TaskState.RUNNING.value
        assert sum(parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED for raw in ws_a.sent) == 1
        assert sum(parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED for raw in ws_b.sent) == 1
        server.store.close()

    asyncio.run(_run())


def test_controller_envelope_default_allows_base64_artifact_overhead() -> None:
    config = ControllerSecurityConfig()

    assert config.max_envelope_bytes == 32_000_000
    assert config.max_inline_artifact_bytes == 4_000_000
    assert config.max_envelope_bytes > config.max_inline_artifact_bytes


def test_controller_default_command_timeout_allows_slow_gui_probes() -> None:
    config = ControllerConfig()

    assert config.command_timeout_seconds == 60.0


def test_controller_default_websocket_accepts_burstier_remote_runs() -> None:
    config = ControllerConfig()

    assert config.websocket_open_timeout_seconds == 120.0
    assert config.websocket_backlog == 4096
    assert config.offline_after_seconds == 240.0
