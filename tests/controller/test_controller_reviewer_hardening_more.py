"""Regression tests for reviewer-found controller hardening issues."""

from __future__ import annotations

import asyncio
import base64
import hashlib

from northcua.controller.api.server import ControllerServer
from northcua.controller.auth import ControllerAuth
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua.controller.execution.command_bus import CommandBus
from northcua.controller.observability.event_log import ControllerEventLog
from northcua.controller.storage.blob_store import LocalControllerBlobStore
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef
from northcua_protocol.capabilities import Capability, DeviceDescriptor, PlatformKind, ToolDescriptor
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope, parse_envelope
from northcua_protocol.payloads import (
    CommandRequest,
    CommandResult,
    DeviceHeartbeat,
    DeviceHello,
    SessionLeaseReleased,
    TaskCancel,
    TaskRequest,
)
from northcua_protocol.state import CommandState, SessionState, TaskState


class _FakeWebSocket:
    """Minimal websocket stand-in for direct controller dispatch tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)


def _descriptor(device_id: str, *, max_sessions: int = 2, focus: bool = False) -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id=device_id,
        platform=PlatformKind.MACOS,
        capabilities={Capability.DESKTOP_INPUT},
        tools=[ToolDescriptor(tool_id="device.echo", capability=Capability.DESKTOP_INPUT)],
        max_sessions=max_sessions,
        max_gui_sessions=1,
        max_concurrent_commands=1,
        requires_foreground_focus=focus,
        supports_parallel_sessions=not focus,
    )


def _task(task_id: str, trace_id: str, *, target_device_id: str = "") -> TaskRequest:
    return TaskRequest(
        task_id=task_id,
        trace_id=trace_id,
        agent_kind="desktop",
        request=f"run {task_id}",
        target_device_id=target_device_id,
        required_capabilities={Capability.DESKTOP_INPUT},
    )


def _lease_from(websocket: _FakeWebSocket):
    return next(
        parse_envelope(raw).payload
        for raw in websocket.sent
        if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
    )


def test_device_originated_lease_release_is_rejected(tmp_path) -> None:
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
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.TASK_REQUEST, _task("task-a", "trace-task-a"), trace_id="trace-task-a"),
        )
        lease = _lease_from(device)

        try:
            await server._dispatch(
                device,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.SESSION_LEASE_RELEASED,
                    SessionLeaseReleased(
                        lease_id=lease.lease_id,
                        session_id=lease.session_id,
                        device_id="dev-a",
                        reason="device asked release",
                    ),
                    trace_id="trace-task-a",
                ),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("device-originated lease release should be rejected")

        assert server.store.session_row(lease.session_id)["status"] != SessionState.RELEASED.value
        assert server.registry.get("dev-a").active_sessions == 1  # type: ignore[union-attr]
        server.store.close()

    asyncio.run(_run())


def test_inbound_controller_blob_must_be_owned_by_command_scope(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    blob_store = LocalControllerBlobStore(tmp_path / "blobs")
    bus = CommandBus(
        store=store,
        auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
        events=ControllerEventLog(store),
        blob_store=blob_store,
        max_inline_artifact_bytes=1,
    )
    request = _task("task-a", "trace-a")
    store.create_task(request, session_id="sess-a", submitter_id="submitter-a")
    store.create_lease(lease_id="lease-a", session_id="sess-a", task_id="task-a", device_id="dev-a")
    raw = b"known blob"
    inline = ArtifactRef.from_inline_bytes(
        artifact_id="artifact-a",
        kind=ArtifactKind.TEXT,
        data=raw,
        mime_type="text/plain",
    )
    owned = blob_store.materialize(inline)

    try:
        bus.record_result(
            CommandResult(
                command_id="cmd-a",
                session_id="sess-a",
                lease_id="lease-a",
                device_id="dev-a",
                status=CommandState.SUCCEEDED,
                artifacts=[owned.model_copy(update={"artifact_id": "relabelled"})],
            ),
            trace_id="trace-a",
        )
    except ValueError as exc:
        assert "not owned" in str(exc)
    else:
        raise AssertionError("device should not relabel an arbitrary controller blob into a command result")

    assert hashlib.sha256(raw).hexdigest() == owned.sha256
    assert base64.b64decode(inline.inline_data or "") == raw
    store.close()
