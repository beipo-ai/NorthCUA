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


def test_task_service_set_executor_schedules_runner_after_lease(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        executor = _RecordingExecutor()
        server.task_service.set_executor(executor)
        ws_device = _FakeWebSocket()

        await server._dispatch(
            ws_device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(
                    descriptor=_descriptor(
                        "dev-a",
                        capabilities={
                            Capability.DESKTOP_INPUT,
                            Capability.DESKTOP_SCREENSHOT,
                            Capability.DESKTOP_WINDOW,
                            Capability.A11Y_TREE,
                        },
                    )
                ),
                trace_id="trace-a",
            ),
        )
        await server._dispatch(
            ws_device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-a",
                    trace_id="trace-task-a",
                    agent_kind="desktop",
                    request="task A",
                    target_device_id="dev-a",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-task-a",
            ),
        )

        assert len(executor.scheduled) == 1
        assert executor.scheduled[0].request.task_id == "task-a"
        server.events.flush()
        assert any(row["event_type"] == "task_runner_scheduled" for row in server.store.event_rows())
        server.store.close()

    asyncio.run(_run())


def test_submitter_can_cancel_own_task(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        submitter = _FakeWebSocket()
        device = _FakeWebSocket()

        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-1")), trace_id="trace-dev"),
        )
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-cancel",
                    trace_id="trace-cancel",
                    agent_kind="desktop",
                    request="cancel me",
                    token="",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-cancel",
            ),
        )

        accepted = next(
            parse_envelope(raw).payload
            for raw in submitter.sent
            if parse_envelope(raw).type == MessageType.TASK_ACCEPTED
        )
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_CANCEL,
                TaskCancel(
                    task_id="task-cancel",
                    trace_id="trace-cancel",
                    session_id=accepted.session_id,
                    reason="submitter cancelled",
                ),
                trace_id="trace-cancel",
            ),
        )

        finished = next(
            parse_envelope(raw).payload
            for raw in submitter.sent
            if parse_envelope(raw).type == MessageType.TASK_FINISHED
        )

        assert finished.task_id == "task-cancel"
        assert finished.state == TaskState.INTERRUPTED
        assert finished.error == "submitter cancelled"
        server.store.close()

    asyncio.run(_run())
