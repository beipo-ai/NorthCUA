"""Tests for remote dispatcher payloads and controller executor scheduling."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from northcua.controller.api.server import ControllerServer
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua.controller.execution.remote_dispatcher import RemoteCommandDispatcher
from northcua.controller.storage.identifiers import command_storage_key
from northcua_protocol.capabilities import Capability, DeviceDescriptor, PlatformKind, ToolDescriptor
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope, parse_envelope
from northcua_protocol.payloads import CommandRequest, CommandResult, DeviceHello, SessionLeaseGranted, TaskRequest
from northcua_protocol.runtime_sidecars import RUNTIME_SIDECAR_KEY
from northcua_protocol.state import CommandState


def _descriptor(device_id: str) -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id=device_id,
        platform=PlatformKind.MACOS,
        capabilities={
            Capability.DESKTOP_INPUT,
            Capability.DESKTOP_SCREENSHOT,
            Capability.DESKTOP_WINDOW,
            Capability.A11Y_TREE,
        },
        tools=[ToolDescriptor(tool_id="device.echo", capability=Capability.DESKTOP_INPUT)],
        max_sessions=2,
        max_gui_sessions=1,
    )


class _FakeWebSocket:
    """Minimal websocket stand-in for controller dispatch tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)


class _FakeTaskExecutor:
    """Collect controller execution contexts scheduled after lease delivery."""

    def __init__(self) -> None:
        self.contexts = []

    async def schedule(self, context) -> None:
        self.contexts.append(context)

    async def drain(self) -> None:
        return None

    async def cancel_task(self, task_id: str, reason: str) -> None:
        del task_id, reason


def test_remote_command_dispatcher_builds_generation_aware_command() -> None:
    async def _run() -> None:
        captured: list[CommandRequest] = []
        progress: list[tuple[MessageType, object]] = []

        async def send_command(command: CommandRequest, *, trace_id: str) -> CommandResult:
            captured.append(command)
            assert trace_id == "trace-1"
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        dispatcher = RemoteCommandDispatcher(
            sdk=SimpleNamespace(send_command=send_command),
            lease=SessionLeaseGranted(
                lease_id="lease-1",
                session_id="sess-1",
                task_id="task-1",
                device_id="dev-1",
                trace_id="trace-1",
                lease_generation=3,
                metadata={
                    "northcua.desktop.coordinate_space": "image_pixels",
                    "lease_metadata": "kept",
                },
            ),
            metadata_overrides={
                "northcua.desktop.coordinate_space": "normalized_1000",
                "northcua.desktop.coordinate_clamp_to_bounds": "true",
            },
            progress_sink=lambda message_type, payload: _append_progress(progress, message_type, payload),
        )

        result = await dispatcher.dispatch(tool_id="device.echo", parameters={"hello": "world"}, command_id="cmd-1")

        assert result.status == CommandState.SUCCEEDED
        assert captured[0].lease_generation == 3
        assert captured[0].metadata == {
            "northcua.desktop.coordinate_space": "normalized_1000",
            "northcua.desktop.coordinate_clamp_to_bounds": "true",
            "lease_metadata": "kept",
        }
        assert captured[0].idempotency_key
        assert [item[0] for item in progress] == [MessageType.COMMAND_REQUEST, MessageType.COMMAND_RESULT]
        assert isinstance(progress[0][1], CommandRequest)
        assert isinstance(progress[1][1], CommandResult)

    asyncio.run(_run())


def test_controller_schedules_executor_after_lease_delivery(tmp_path) -> None:
    async def _run() -> None:
        executor = _FakeTaskExecutor()
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            ),
            executor=executor,
        )
        ws = _FakeWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-exec",
                    trace_id="trace-exec",
                    agent_kind="desktop",
                    request="do work",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-exec",
            ),
        )

        assert len(executor.contexts) == 1
        assert executor.contexts[0].request.task_id == "task-exec"
        assert executor.contexts[0].lease.device_id == "dev-a"
        assert executor.contexts[0].progress_sink is not None
        await executor.contexts[0].progress_sink(
            MessageType.COMMAND_REQUEST,
            CommandRequest(
                command_id="cmd-progress",
                idempotency_key="idem-progress",
                session_id=executor.contexts[0].session_id,
                lease_id=executor.contexts[0].lease.lease_id,
                device_id="dev-a",
                tool_id="device.echo",
                parameters={"value": 1},
            ),
        )
        assert parse_envelope(ws.sent[-1]).type == MessageType.COMMAND_REQUEST
        server.store.close()

    asyncio.run(_run())


def test_controller_strips_runtime_sidecar_before_persisting_command_result(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _FakeWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-sidecar")), trace_id="t"),
        )
        request = CommandRequest(
            command_id="cmd-sidecar",
            idempotency_key="idem-sidecar",
            session_id="sess-sidecar",
            lease_id="lease-sidecar",
            device_id="dev-sidecar",
            tool_id="device.echo",
            parameters={"value": 1},
        )
        server.store.claim_command(request)
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.COMMAND_RESULT,
                CommandResult(
                    command_id=request.command_id,
                    session_id=request.session_id,
                    lease_id=request.lease_id,
                    device_id=request.device_id,
                    status=CommandState.SUCCEEDED,
                    result={
                        "status": "ok",
                        RUNTIME_SIDECAR_KEY: {"web_ref_aliases": {"aliases": [{"ref": "1", "source_ref": "@e1"}]}},
                    },
                ),
                trace_id="t",
            ),
        )
        row = server.store.connection.execute(
            "SELECT result_json FROM commands WHERE command_id=?",
            (
                command_storage_key(
                    session_id=request.session_id,
                    lease_id=request.lease_id,
                    command_id=request.command_id,
                ),
            ),
        ).fetchone()
        assert row is not None
        result_json = str(row["result_json"])
        assert RUNTIME_SIDECAR_KEY not in result_json
        assert "web_ref_aliases" not in result_json
        assert '"status":"ok"' in result_json
        server.store.close()

    asyncio.run(_run())


async def _append_progress(
    progress: list[tuple[MessageType, object]],
    message_type: MessageType,
    payload: object,
) -> None:
    progress.append((message_type, payload))
