"""Tests for controller scheduling, lease, and command-bus behavior."""

from __future__ import annotations

import asyncio

import websockets

from northcua.controller.api.server import ControllerServer
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
from northcua_protocol.state import CommandState, TaskState


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


def test_command_bus_serializes_gui_commands_for_shared_resource(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
                command_stage_event_mode="raw",
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
        session_a = server.store.session_row_for_task("task-a")
        session_b = server.store.session_row_for_task("task-b")
        assert session_a is not None and session_b is not None
        lease_a = server.store.lease_row(str(session_a["lease_id"]))
        lease_b = server.store.lease_row(str(session_b["lease_id"]))
        assert lease_a is not None and lease_b is not None
        active = 0
        max_active = 0

        async def sender(request: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal active, max_active
            del trace_id
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            return CommandResult(
                command_id=request.command_id,
                session_id=request.session_id,
                lease_id=request.lease_id,
                device_id=request.device_id,
                lease_generation=request.lease_generation,
                status=CommandState.SUCCEEDED,
            )

        server.command_bus.register_sender("dev-a", sender)
        server.command_bus.register_sender("dev-b", sender)
        command_a = CommandRequest(
            command_id="cmd-a",
            idempotency_key=command_idempotency_key(
                session_id=str(session_a["session_id"]),
                lease_id=str(session_a["lease_id"]),
                command_id="cmd-a",
                payload={"tool_id": "device.echo", "parameters": {}},
            ),
            session_id=str(session_a["session_id"]),
            lease_id=str(session_a["lease_id"]),
            device_id="dev-a",
            tool_id="device.echo",
            parameters={},
            lease_generation=int(lease_a["generation"]),
            timeout_seconds=2.0,
        )
        command_b = CommandRequest(
            command_id="cmd-b",
            idempotency_key=command_idempotency_key(
                session_id=str(session_b["session_id"]),
                lease_id=str(session_b["lease_id"]),
                command_id="cmd-b",
                payload={"tool_id": "device.echo", "parameters": {}},
            ),
            session_id=str(session_b["session_id"]),
            lease_id=str(session_b["lease_id"]),
            device_id="dev-b",
            tool_id="device.echo",
            parameters={},
            lease_generation=int(lease_b["generation"]),
            timeout_seconds=2.0,
        )

        result_a, result_b = await asyncio.gather(
            server.command_bus.send(command_a, trace_id="trace-task-a"),
            server.command_bus.send(command_b, trace_id="trace-task-b"),
        )

        assert result_a.status == CommandState.SUCCEEDED
        assert result_b.status == CommandState.SUCCEEDED
        assert max_active == 1
        server.events.flush()
        assert any(row["event_type"] == "command_gui_resource_acquired" for row in server.store.event_rows())
        server.store.close()

    asyncio.run(_run())


def test_device_submitter_connection_gets_own_lease_and_commands(tmp_path) -> None:
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
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            ws_b,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-b")), trace_id="trace-b"),
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

        lease_a = next(
            parse_envelope(raw).payload
            for raw in ws_a.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        lease_b = next(
            parse_envelope(raw).payload
            for raw in ws_b.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        assert lease_a.device_id == "dev-a"  # type: ignore[attr-defined]
        assert lease_b.device_id == "dev-b"  # type: ignore[attr-defined]

        command = CommandRequest(
            command_id="cmd-a",
            idempotency_key=command_idempotency_key(
                session_id=lease_a.session_id,  # type: ignore[attr-defined]
                lease_id=lease_a.lease_id,  # type: ignore[attr-defined]
                command_id="cmd-a",
                payload={"tool_id": "device.echo"},
            ),
            session_id=lease_a.session_id,  # type: ignore[attr-defined]
            lease_id=lease_a.lease_id,  # type: ignore[attr-defined]
            device_id="dev-a",
            tool_id="device.echo",
        )
        send_task = asyncio.create_task(server.command_bus.send(command, trace_id="trace-task-a"))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if any(parse_envelope(raw).type == MessageType.COMMAND_REQUEST for raw in ws_a.sent):
                break

        assert any(parse_envelope(raw).type == MessageType.COMMAND_REQUEST for raw in ws_a.sent)
        assert not any(parse_envelope(raw).type == MessageType.COMMAND_REQUEST for raw in ws_b.sent)
        await server._dispatch(
            ws_a,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.COMMAND_RESULT,
                CommandResult(
                    command_id="cmd-a",
                    session_id=command.session_id,
                    lease_id=command.lease_id,
                    device_id="dev-a",
                    status=CommandState.SUCCEEDED,
                    result={"ok": True},
                ),
                trace_id="trace-task-a",
            ),
        )
        result = await send_task
        assert result.status == CommandState.SUCCEEDED
        server.store.close()

    asyncio.run(_run())
