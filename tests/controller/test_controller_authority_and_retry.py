"""Tests for controller authority checks, retry, and executor scheduling."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from northcua.controller.api.server import ControllerServer
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua.controller.execution.command_bus import command_idempotency_key
from northcua.controller.storage.blob_store import LocalControllerBlobStore
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
    TaskFinished,
    TaskRequest,
)
from northcua_protocol.state import CommandState, TaskState


def _descriptor(device_id: str, *, active_focus: bool = False) -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id=device_id,
        platform=PlatformKind.MACOS,
        capabilities={Capability.DESKTOP_INPUT},
        tools=[ToolDescriptor(tool_id="device.echo", capability=Capability.DESKTOP_INPUT)],
        max_sessions=2,
        max_gui_sessions=1,
        requires_foreground_focus=active_focus,
        supports_parallel_sessions=not active_focus,
    )


class _FakeWebSocket:
    """Minimal websocket stand-in for direct controller dispatch tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)


async def _wait_for_message_type(websocket: _FakeWebSocket, message_type: MessageType) -> None:
    async def _wait() -> None:
        while not any(parse_envelope(raw).type == message_type for raw in websocket.sent):
            await asyncio.sleep(0.001)

    await asyncio.wait_for(_wait(), timeout=1.0)


def test_command_result_sender_must_own_device_and_lease(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        owner = _FakeWebSocket()
        intruder = _FakeWebSocket()
        await server._dispatch(
            owner,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            intruder,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-b")), trace_id="trace-b"),
        )
        await server._dispatch(
            owner,  # type: ignore[arg-type]
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
        lease = next(
            parse_envelope(raw).payload
            for raw in owner.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        command = CommandRequest(
            command_id="cmd-a",
            idempotency_key=command_idempotency_key(
                session_id=lease.session_id,  # type: ignore[attr-defined]
                lease_id=lease.lease_id,  # type: ignore[attr-defined]
                command_id="cmd-a",
                payload={"tool_id": "device.echo"},
            ),
            session_id=lease.session_id,  # type: ignore[attr-defined]
            lease_id=lease.lease_id,  # type: ignore[attr-defined]
            device_id="dev-a",
            tool_id="device.echo",
        )
        send_task = asyncio.create_task(server.command_bus.send(command, trace_id="trace-task-a"))
        await _wait_for_message_type(owner, MessageType.COMMAND_REQUEST)

        try:
            await server._dispatch(
                intruder,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.COMMAND_RESULT,
                    CommandResult(
                        command_id="cmd-a",
                        session_id=command.session_id,
                        lease_id=command.lease_id,
                        device_id="dev-a",
                        lease_generation=command.lease_generation,
                        status=CommandState.SUCCEEDED,
                        result={"ok": False},
                    ),
                    trace_id="trace-task-a",
                ),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("intruder should not be able to complete another device command")

        await server._dispatch(
            owner,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.COMMAND_RESULT,
                CommandResult(
                    command_id="cmd-a",
                    session_id=command.session_id,
                    lease_id=command.lease_id,
                    device_id="dev-a",
                    lease_generation=command.lease_generation,
                    status=CommandState.SUCCEEDED,
                    result={"ok": True},
                ),
                trace_id="trace-task-a",
            ),
        )
        assert (await send_task).status == CommandState.SUCCEEDED
        server.store.close()

    asyncio.run(_run())


def test_command_result_sender_must_match_session_identity(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        owner = _FakeWebSocket()
        await server._dispatch(
            owner,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            owner,  # type: ignore[arg-type]
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
        lease = next(
            parse_envelope(raw).payload
            for raw in owner.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        command = CommandRequest(
            command_id="cmd-a",
            idempotency_key=command_idempotency_key(
                session_id=lease.session_id,  # type: ignore[attr-defined]
                lease_id=lease.lease_id,  # type: ignore[attr-defined]
                command_id="cmd-a",
                payload={"tool_id": "device.echo"},
            ),
            session_id=lease.session_id,  # type: ignore[attr-defined]
            lease_id=lease.lease_id,  # type: ignore[attr-defined]
            device_id="dev-a",
            tool_id="device.echo",
        )
        send_task = asyncio.create_task(server.command_bus.send(command, trace_id="trace-task-a"))
        await _wait_for_message_type(owner, MessageType.COMMAND_REQUEST)

        try:
            await server._dispatch(
                owner,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.COMMAND_RESULT,
                    CommandResult(
                        command_id="cmd-a",
                        session_id="wrong-session",
                        lease_id=command.lease_id,
                        device_id="dev-a",
                        lease_generation=command.lease_generation,
                        status=CommandState.SUCCEEDED,
                    ),
                    trace_id="trace-task-a",
                ),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("command result with a mismatched session should be rejected")
        send_task.cancel()
        with suppress(asyncio.CancelledError):
            await send_task
        server.store.close()

    asyncio.run(_run())


def test_task_metadata_size_limit_rejects_oversized_submitter_payload(tmp_path) -> None:
    server = ControllerServer(
        ControllerConfig(
            db_path=str(tmp_path / "controller.sqlite"),
            security=ControllerSecurityConfig(insecure_dev_no_auth=True, max_task_metadata_bytes=4),
        )
    )
    ws = _FakeWebSocket()

    async def _run() -> None:
        try:
            await server._dispatch(
                ws,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_REQUEST,
                    TaskRequest(
                        task_id="task-large-meta",
                        trace_id="trace-meta",
                        agent_kind="desktop",
                        request="task",
                        metadata={"html_content": "too large"},
                    ),
                    trace_id="trace-meta",
                ),
            )
        except ValueError as exc:
            assert "max_task_metadata_bytes" in str(exc)
        else:
            raise AssertionError("oversized task metadata should be rejected")
        finally:
            server.store.close()

    asyncio.run(_run())


def test_task_request_token_is_redacted_before_persistence(tmp_path) -> None:
    server = ControllerServer(
        ControllerConfig(
            db_path=str(tmp_path / "controller.sqlite"),
            security=ControllerSecurityConfig(token="submit-token"),
        )
    )
    ws = _FakeWebSocket()

    async def _run() -> None:
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-token",
                    trace_id="trace-token",
                    agent_kind="desktop",
                    request="task",
                    token="submit-token",
                ),
                trace_id="trace-token",
            ),
        )
        row = server.store.task_row("task-token")
        assert row is not None
        assert "submit-token" not in row["request_json"]
        assert TaskRequest.model_validate_json(row["request_json"]).token == ""
        server.store.close()

    asyncio.run(_run())


def test_lease_release_must_match_active_persisted_owner(tmp_path) -> None:
    server = ControllerServer(
        ControllerConfig(
            db_path=str(tmp_path / "controller.sqlite"),
            security=ControllerSecurityConfig(insecure_dev_no_auth=True),
        )
    )
    ws = _FakeWebSocket()

    async def _run() -> None:
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-release",
                    trace_id="trace-release",
                    agent_kind="desktop",
                    request="task",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-release",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in ws.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )

        try:
            server.leases.release(
                SessionLeaseReleased(
                    lease_id=lease.lease_id,  # type: ignore[attr-defined]
                    session_id=lease.session_id,  # type: ignore[attr-defined]
                    device_id="dev-b",
                    reason="spoofed",
                ),
                trace_id="trace-release",
            )
        except PermissionError as exc:
            assert "lease release does not match" in str(exc)
        else:
            raise AssertionError("foreign device should not be able to release a lease")
        finally:
            server.store.close()

    asyncio.run(_run())
