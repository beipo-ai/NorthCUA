"""Controller task status/list API tests."""

from __future__ import annotations

import asyncio

import pytest

from northcua.controller.api.server import ControllerServer
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua_protocol.capabilities import Capability, DeviceDescriptor, PlatformKind, ToolDescriptor
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope, parse_envelope
from northcua_protocol.payloads import (
    DeviceHello,
    TaskFinished,
    TaskListRequest,
    TaskRequest,
    TaskStatusFilter,
    TaskStatusRequest,
)
from northcua_protocol.state import TaskState


def _descriptor(device_id: str) -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id=device_id,
        platform=PlatformKind.MACOS,
        capabilities={Capability.DESKTOP_INPUT},
        tools=[ToolDescriptor(tool_id="device.echo", capability=Capability.DESKTOP_INPUT)],
        max_sessions=2,
    )


class _FakeWebSocket:
    """Minimal websocket stand-in for direct controller dispatch tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)


def test_submitter_can_query_task_status_and_list_by_batch(tmp_path) -> None:
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
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("dev-status")),
                trace_id="trace-dev",
            ),
        )
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-status",
                    trace_id="trace-status",
                    agent_kind="desktop",
                    request="status me",
                    required_capabilities={Capability.DESKTOP_INPUT},
                    metadata={"batch_id": "batch-a"},
                ),
                trace_id="trace-status",
            ),
        )
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_STATUS_REQUEST,
                TaskStatusRequest(task_ids=["task-status"]),
                trace_id="trace-status",
            ),
        )
        status_payload = parse_envelope(submitter.sent[-1]).payload
        assert parse_envelope(submitter.sent[-1]).type == MessageType.TASK_STATUS_RESPONSE
        assert status_payload.statuses[0].task_id == "task-status"  # type: ignore[attr-defined]
        assert status_payload.statuses[0].device_id == "dev-status"  # type: ignore[attr-defined]
        assert status_payload.statuses[0].metadata == {"batch_id": "batch-a"}  # type: ignore[attr-defined]

        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_LIST_REQUEST,
                TaskListRequest(filter=TaskStatusFilter(batch_id="batch-a")),
                trace_id="trace-status",
            ),
        )
        list_payload = parse_envelope(submitter.sent[-1]).payload
        assert parse_envelope(submitter.sent[-1]).type == MessageType.TASK_LIST_RESPONSE
        assert [task.task_id for task in list_payload.tasks] == ["task-status"]  # type: ignore[attr-defined]
        assert list_payload.tasks[0].metadata == {"batch_id": "batch-a"}  # type: ignore[attr-defined]
        server.store.close()

    asyncio.run(_run())


def test_task_status_snapshot_includes_terminal_decision(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        peer = _FakeWebSocket()
        await server._dispatch(
            peer,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("dev-terminal")),
                trace_id="trace-dev",
            ),
        )
        await server._dispatch(
            peer,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-terminal",
                    trace_id="trace-terminal",
                    agent_kind="desktop",
                    request="finish me",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-terminal",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in peer.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        await server._dispatch(
            peer,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_FINISHED,
                TaskFinished(
                    task_id="task-terminal",
                    trace_id="trace-terminal",
                    session_id=lease.session_id,  # type: ignore[attr-defined]
                    device_id="dev-terminal",
                    lease_id=lease.lease_id,  # type: ignore[attr-defined]
                    state=TaskState.FINISHED,
                    terminal_decision={"action": "finish", "reason": "completed"},
                ),
                trace_id="trace-terminal",
            ),
        )
        await server._dispatch(
            peer,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_STATUS_REQUEST,
                TaskStatusRequest(task_ids=["task-terminal"]),
                trace_id="trace-terminal",
            ),
        )
        status_payload = parse_envelope(peer.sent[-1]).payload
        assert status_payload.statuses[0].terminal_decision == {"action": "finish", "reason": "completed"}  # type: ignore[attr-defined]
        server.store.close()

    asyncio.run(_run())


def test_task_list_applies_explicit_limit(tmp_path) -> None:
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
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("dev-status")),
                trace_id="trace-dev",
            ),
        )
        for task_id in ("task-1", "task-2"):
            await server._dispatch(
                submitter,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_REQUEST,
                    TaskRequest(
                        task_id=task_id,
                        trace_id=f"trace-{task_id}",
                        agent_kind="desktop",
                        request="status me",
                        required_capabilities={Capability.DESKTOP_INPUT},
                        metadata={"batch_id": "batch-limit"},
                    ),
                    trace_id=f"trace-{task_id}",
                ),
            )

        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_LIST_REQUEST,
                TaskListRequest(filter=TaskStatusFilter(batch_id="batch-limit", limit=1)),
                trace_id="trace-status",
            ),
        )

        list_payload = parse_envelope(submitter.sent[-1]).payload
        assert len(list_payload.tasks) == 1  # type: ignore[attr-defined]
        server.store.close()

    asyncio.run(_run())


def test_task_status_read_api_requires_submitter_token(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(submitter_token="read-secret"),
            )
        )
        submitter = _FakeWebSocket()

        with pytest.raises(PermissionError, match="submitter token rejected"):
            await server._dispatch(
                submitter,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_STATUS_REQUEST,
                    TaskStatusRequest(task_ids=["task-status"], token="wrong"),
                    trace_id="trace-status",
                ),
            )

        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_STATUS_REQUEST,
                TaskStatusRequest(task_ids=["task-status"], token="read-secret"),
                trace_id="trace-status",
            ),
        )
        assert parse_envelope(submitter.sent[-1]).type == MessageType.TASK_STATUS_RESPONSE
        server.store.close()

    asyncio.run(_run())
