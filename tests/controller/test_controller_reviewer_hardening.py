"""Regression tests for reviewer-found controller hardening issues."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json

import pytest

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
    AGENTKIT_SETTINGS_PARAM,
    CommandRequest,
    CommandResult,
    DeviceHeartbeat,
    DeviceHello,
    SessionLeaseReleased,
    TaskCancel,
    TaskRequest,
)
from northcua_protocol.state import CommandState, SessionState, TaskState
from northcua_runtime_shared.validation import ToolParameterValidationError


class _FakeWebSocket:
    """Minimal websocket stand-in for direct controller dispatch tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)


class _FailTaskAcceptedWebSocket(_FakeWebSocket):
    async def send(self, raw: str) -> None:
        if parse_envelope(raw).type == MessageType.TASK_ACCEPTED:
            raise RuntimeError("accepted send failed")
        await super().send(raw)


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


def _task(
    task_id: str,
    trace_id: str,
    *,
    target_device_id: str = "",
    metadata: dict[str, object] | None = None,
) -> TaskRequest:
    return TaskRequest(
        task_id=task_id,
        trace_id=trace_id,
        agent_kind="desktop",
        request=f"run {task_id}",
        target_device_id=target_device_id,
        required_capabilities={Capability.DESKTOP_INPUT},
        metadata=metadata or {},
    )


def _lease_from(websocket: _FakeWebSocket):
    return next(
        parse_envelope(raw).payload
        for raw in websocket.sent
        if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
    )


def test_duplicate_task_id_is_rejected_before_owner_replace(tmp_path) -> None:
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
            Envelope.wrap(MessageType.TASK_REQUEST, _task("task-shared", "trace-original"), trace_id="trace-original"),
        )

        try:
            await server._dispatch(
                intruder,  # type: ignore[arg-type]
                Envelope.wrap(MessageType.TASK_REQUEST, _task("task-shared", "trace-hijack"), trace_id="trace-hijack"),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("duplicate task_id should be rejected")

        row = server.store.task_row("task-shared")
        assert row is not None
        assert row["trace_id"] == "trace-original"
        assert row["submitter_id"].startswith("submitter-")
        assert server.task_service._submitter_by_task_id["task-shared"] is owner
        server.store.close()

    asyncio.run(_run())


def test_duplicate_task_id_does_not_temporarily_replace_owner_mapping(tmp_path) -> None:
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
            Envelope.wrap(MessageType.TASK_REQUEST, _task("task-shared", "trace-original"), trace_id="trace-original"),
        )
        original_accept_async = server.queue.accept_async
        observed_owner_during_duplicate = None

        async def duplicate_accept_async(*args, **kwargs):
            nonlocal observed_owner_during_duplicate
            observed_owner_during_duplicate = server.task_service._submitter_by_task_id.get("task-shared")  # noqa: SLF001
            await original_accept_async(*args, **kwargs)

        server.queue.accept_async = duplicate_accept_async  # type: ignore[method-assign]
        try:
            await server._dispatch(
                intruder,  # type: ignore[arg-type]
                Envelope.wrap(MessageType.TASK_REQUEST, _task("task-shared", "trace-hijack"), trace_id="trace-hijack"),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("duplicate task_id should be rejected")

        assert observed_owner_during_duplicate is owner
        assert server.task_service._submitter_by_task_id["task-shared"] is owner  # noqa: SLF001
        server.store.close()

    asyncio.run(_run())


def test_duplicate_task_id_retry_window_uses_original_owner_mapping(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        owner = _FakeWebSocket()
        intruder = _FakeWebSocket()

        shared_submitter_id = "shared-submitter"
        await server._dispatch(
            owner,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                _task("task-shared", "trace-original", metadata={"submitter_id": shared_submitter_id}),
                trace_id="trace-original",
            ),
        )
        assert server.connections.for_submitter(shared_submitter_id).websocket is owner
        original_accept_async = server.queue.accept_async

        async def duplicate_accept_async(*args, **kwargs):
            await server.task_service.retry_queued_tasks(trace_id="trace-duplicate-window")
            await original_accept_async(*args, **kwargs)

        server.queue.accept_async = duplicate_accept_async  # type: ignore[method-assign]
        try:
            await server._dispatch(
                intruder,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_REQUEST,
                    _task("task-shared", "trace-hijack", metadata={"submitter_id": shared_submitter_id}),
                    trace_id="trace-hijack",
                ),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("duplicate task_id should be rejected")

        assert server.connections.for_submitter(shared_submitter_id).websocket is owner
        server.task_service._submitter_by_task_id.pop("task-shared", None)  # noqa: SLF001
        await server.task_service.retry_queued_tasks(trace_id="trace-recover-after-duplicate")
        assert server.task_service._submitter_by_task_id["task-shared"] is owner  # noqa: SLF001
        assert server.store.task_row("task-shared")["trace_id"] == "trace-original"
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert "task_submitter_mapping_recovered" in event_types
        assert "task_orphaned_missing_submitter" not in event_types
        server.store.close()

    asyncio.run(_run())


def test_retry_queued_burst_is_single_flight_and_coalesced(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        scheduler = server.task_service._scheduler  # noqa: SLF001
        started = asyncio.Event()
        release = asyncio.Event()
        sweep_count = 0

        async def _blocked_sweep(*, trace_id: str) -> None:
            nonlocal sweep_count
            del trace_id
            sweep_count += 1
            started.set()
            await release.wait()

        scheduler._retry_queued_once = _blocked_sweep  # type: ignore[method-assign]  # noqa: SLF001
        first = asyncio.create_task(scheduler.retry_queued(trace_id="trace-0"))
        await started.wait()
        burst = [asyncio.create_task(scheduler.retry_queued(trace_id=f"trace-{index}")) for index in range(1, 501)]
        await asyncio.sleep(0)

        assert sweep_count == 1
        assert scheduler._retry_triggers_coalesced == 500  # noqa: SLF001
        release.set()
        await asyncio.gather(first, *burst)
        assert sweep_count == 2
        assert scheduler._retry_sweeps_started == 2  # noqa: SLF001
        server.events.close()
        server.store.close()

    asyncio.run(_run())


def test_retry_queued_coalesced_trigger_survives_failed_sweep(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        scheduler = server.task_service._scheduler  # noqa: SLF001
        started = asyncio.Event()
        release = asyncio.Event()
        sweep_count = 0

        async def _flaky_sweep(*, trace_id: str) -> None:
            nonlocal sweep_count
            del trace_id
            sweep_count += 1
            if sweep_count == 1:
                started.set()
                await release.wait()
                raise RuntimeError("injected lease delivery failure")

        scheduler._retry_queued_once = _flaky_sweep  # type: ignore[method-assign]  # noqa: SLF001
        first = asyncio.create_task(scheduler.retry_queued(trace_id="trace-0"))
        await started.wait()
        burst = [asyncio.create_task(scheduler.retry_queued(trace_id=f"trace-{index}")) for index in range(1, 501)]
        await asyncio.sleep(0)
        release.set()

        await asyncio.gather(first, *burst)
        snapshot = scheduler.retry_snapshot()
        assert sweep_count == 2
        assert snapshot["sweep_failures"] == 1
        assert snapshot["requested"] is False
        assert snapshot["active"] is False
        server.events.close()
        server.store.close()

    asyncio.run(_run())


def test_retry_queued_continues_after_one_task_delivery_fails(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        submitter = _FakeWebSocket()
        for task_id in ("task-a", "task-b"):
            await server._dispatch(
                submitter,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_REQUEST,
                    _task(task_id, f"trace-{task_id}"),
                    trace_id=f"trace-{task_id}",
                ),
            )

        scheduler = server.task_service._scheduler  # noqa: SLF001
        attempted: list[str] = []

        async def _flaky_schedule(**kwargs) -> None:
            attempted.append(kwargs["request"].task_id)
            if len(attempted) == 1:
                raise RuntimeError("injected device send failure")

        scheduler.try_schedule = _flaky_schedule  # type: ignore[method-assign]
        await scheduler._retry_queued_once(trace_id="trace-retry")  # noqa: SLF001

        assert attempted == ["task-a", "task-b"]
        assert scheduler.retry_snapshot()["item_failures"] == 1
        server.events.close()
        server.store.close()

    asyncio.run(_run())


def test_controller_stop_waits_for_shared_retry_sweep(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        scheduler = server.task_service._scheduler  # noqa: SLF001
        started = asyncio.Event()
        release = asyncio.Event()

        async def _blocked_sweep(*, trace_id: str) -> None:
            del trace_id
            started.set()
            await release.wait()

        scheduler._retry_queued_once = _blocked_sweep  # type: ignore[method-assign]  # noqa: SLF001
        retry = asyncio.create_task(scheduler.retry_queued(trace_id="trace-stop"))
        await started.wait()
        stopping = asyncio.create_task(server.stop())
        await asyncio.sleep(0.02)

        assert not stopping.done()
        release.set()
        await retry
        await asyncio.wait_for(stopping, timeout=1)

    asyncio.run(_run())


def test_task_accepted_send_failure_terminalizes_pending_accept(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        submitter = _FailTaskAcceptedWebSocket()

        try:
            await server._dispatch(
                submitter,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_REQUEST,
                    _task("task-accepted-send-fails", "trace-accepted-send-fails"),
                    trace_id="trace-accepted-send-fails",
                ),
            )
        except RuntimeError as exc:
            assert str(exc) == "accepted send failed"
        else:
            raise AssertionError("TASK_ACCEPTED send failure should propagate")

        row = server.store.task_row("task-accepted-send-fails")
        assert row["status"] == TaskState.INTERRUPTED.value
        assert "task acceptance failed before scheduling" in row["result_json"]
        assert server.task_service._submitter_by_task_id.get("task-accepted-send-fails") is None  # noqa: SLF001
        assert server.store.schedulable_task_rows() == []
        server.store.close()

    asyncio.run(_run())


def test_agentkit_remote_capability_floor_is_authorized(tmp_path) -> None:
    class _FakeExecutor:
        async def schedule(self, context: object) -> None:
            del context
            return None

        async def drain(self) -> None:
            return None

        async def cancel_task(self, task_id: str, reason: str) -> None:
            del task_id, reason

    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(
                    insecure_dev_no_auth=True,
                    allowed_capabilities={Capability.DESKTOP_INPUT},
                ),
            ),
            executor=_FakeExecutor(),
        )
        submitter = _FakeWebSocket()
        try:
            await server._dispatch(
                submitter,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_REQUEST,
                    TaskRequest(
                        task_id="task-understated",
                        trace_id="trace-understated",
                        agent_kind="desktop",
                        request="observe desktop",
                        required_capabilities={Capability.DESKTOP_INPUT},
                    ),
                    trace_id="trace-understated",
                ),
            )
        except PermissionError as exc:
            assert "capability is not allowlisted" in str(exc)
            assert "desktop.screenshot" in str(exc)
        else:
            raise AssertionError("AgentKit remote capability floor should be enforced before scheduling")
        server.store.close()

    asyncio.run(_run())


def test_command_validation_enforces_descriptor_capability_allowlist() -> None:
    auth = ControllerAuth(
        ControllerSecurityConfig(
            insecure_dev_no_auth=True,
            allowed_capabilities={Capability.DESKTOP_SCREENSHOT},
        )
    )
    command = CommandRequest(
        command_id="cmd-input",
        idempotency_key="cmd-input",
        session_id="sess",
        lease_id="lease",
        device_id="dev-a",
        tool_id="device.echo",
    )
    descriptor = ToolDescriptor(tool_id="device.echo", capability=Capability.DESKTOP_INPUT)

    try:
        auth.validate_command(command, descriptor=descriptor)
    except PermissionError as exc:
        assert "tool capability is not allowlisted" in str(exc)
    else:
        raise AssertionError("command descriptor capability should be checked against controller policy")


def test_command_validation_preserves_model_parameter_error_type() -> None:
    auth = ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True))
    command = CommandRequest(
        command_id="cmd-invalid-wait",
        idempotency_key="cmd-invalid-wait",
        session_id="sess",
        lease_id="lease",
        device_id="dev-a",
        tool_id="web.web_wait",
        parameters={"duration_ms": 1000, "load_state": None},
    )
    descriptor = ToolDescriptor(
        tool_id="web.web_wait",
        capability=Capability.BROWSER_PAGE,
        schema={
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"duration_ms": {"type": "integer"}},
                    "required": ["duration_ms"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"load_state": {"type": "string"}},
                    "required": ["load_state"],
                    "additionalProperties": False,
                },
            ]
        },
    )

    with pytest.raises(ToolParameterValidationError, match="must match exactly one schema"):
        auth.validate_command(command, descriptor=descriptor)


def test_command_validation_ignores_flat_runtime_side_channel_for_schema() -> None:
    auth = ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True))
    command = CommandRequest(
        command_id="cmd-web-open",
        idempotency_key="cmd-web-open",
        session_id="sess",
        lease_id="lease",
        device_id="dev-a",
        tool_id="web.web_open",
        parameters={
            "url": "https://example.test",
            AGENTKIT_SETTINGS_PARAM: {"allow_private_network_urls": True},
        },
    )
    descriptor = ToolDescriptor(
        tool_id="web.web_open",
        capability=Capability.BROWSER_PAGE,
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["url"],
            "properties": {"url": {"type": "string"}},
        },
    )

    auth.validate_command(command, descriptor=descriptor)


def test_command_validation_rejects_wrapped_runtime_side_channel() -> None:
    auth = ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True))
    command = CommandRequest(
        command_id="cmd-web-open",
        idempotency_key="cmd-web-open",
        session_id="sess",
        lease_id="lease",
        device_id="dev-a",
        tool_id="web.web_open",
        parameters={
            "url": "https://example.test",
            AGENTKIT_SETTINGS_PARAM: {"web_agent": {"allow_private_network_urls": True}},
        },
    )
    descriptor = ToolDescriptor(
        tool_id="web.web_open",
        capability=Capability.BROWSER_PAGE,
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["url"],
            "properties": {"url": {"type": "string"}},
        },
    )

    with pytest.raises(ValueError, match="web_agent"):
        auth.validate_command(command, descriptor=descriptor)


def test_command_validation_rejects_unsafe_runtime_side_channel() -> None:
    auth = ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True))
    command = CommandRequest(
        command_id="cmd-web-open",
        idempotency_key="cmd-web-open",
        session_id="sess",
        lease_id="lease",
        device_id="dev-a",
        tool_id="web.web_open",
        parameters={
            "url": "https://example.test",
            AGENTKIT_SETTINGS_PARAM: {"action_backend": "playwright"},
        },
    )
    descriptor = ToolDescriptor(
        tool_id="web.web_open",
        capability=Capability.BROWSER_PAGE,
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["url"],
            "properties": {"url": {"type": "string"}},
        },
    )

    with pytest.raises(ValueError, match="action_backend"):
        auth.validate_command(command, descriptor=descriptor)


def test_task_cancel_rejects_foreign_session_id_before_mutation(tmp_path) -> None:
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
            Envelope.wrap(MessageType.TASK_REQUEST, _task("task-a", "trace-a"), trace_id="trace-a"),
        )
        lease_a = _lease_from(owner)
        await server._dispatch(
            owner,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.TASK_REQUEST, _task("task-b", "trace-b"), trace_id="trace-b"),
        )
        lease_b = [
            parse_envelope(raw).payload
            for raw in owner.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        ][1]

        try:
            await server._dispatch(
                owner,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_CANCEL,
                    TaskCancel(task_id="task-a", trace_id="trace-a", session_id=lease_b.session_id),
                    trace_id="trace-a",
                ),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("cancel with a foreign session_id should be rejected")

        assert server.store.task_row("task-a")["status"] != TaskState.INTERRUPTED.value
        assert server.store.session_row(lease_a.session_id)["status"] != TaskState.INTERRUPTED.value
        server.store.close()

    asyncio.run(_run())


def test_heartbeat_cannot_lower_controller_reserved_capacity(tmp_path) -> None:
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
                DeviceHello(descriptor=_descriptor("dev-a", max_sessions=1)),
                trace_id="trace-a",
            ),
        )
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                _task("task-a", "trace-task-a", target_device_id="dev-a"),
                trace_id="trace-task-a",
            ),
        )
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HEARTBEAT,
                DeviceHeartbeat(device_id="dev-a", active_sessions=0, active_commands=0),
                trace_id="trace-heartbeat",
            ),
        )
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                _task("task-b", "trace-task-b", target_device_id="dev-a"),
                trace_id="trace-task-b",
            ),
        )

        lease_count = sum(1 for raw in device.sent if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED)
        assert lease_count == 1
        assert server.store.task_row("task-b")["status"] == TaskState.ACCEPTED.value
        server.store.close()

    asyncio.run(_run())


def test_detach_submitter_terminalizes_accepted_pending_orphan_once(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        submitter = _FakeWebSocket()
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.TASK_REQUEST, _task("task-orphan", "trace-orphan"), trace_id="trace-orphan"),
        )
        accepted = next(
            parse_envelope(raw).payload
            for raw in submitter.sent
            if parse_envelope(raw).type == MessageType.TASK_ACCEPTED
        )

        await server.task_service.detach_submitter(
            websocket=submitter,  # type: ignore[arg-type]
            trace_id="trace-detach",
            reason="submitter websocket disconnected",
        )
        await server.task_service.retry_queued_tasks(trace_id="trace-retry-a")
        await server.task_service.retry_queued_tasks(trace_id="trace-retry-b")

        assert server.store.task_row("task-orphan")["status"] == TaskState.INTERRUPTED.value
        assert server.store.session_row(accepted.session_id)["status"] == SessionState.INTERRUPTED.value
        assert server.store.schedulable_task_rows() == []
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert event_types.count("task_orphaned_missing_submitter") == 1
        result = json.loads(server.store.task_row("task-orphan")["result_json"])
        assert result["state"] == TaskState.INTERRUPTED.value
        server.store.close()

    asyncio.run(_run())


def test_detach_submitter_does_not_terminalize_running_task(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        device = _FakeWebSocket()
        submitter = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.TASK_REQUEST, _task("task-running", "trace-running"), trace_id="trace-running"),
        )
        accepted = next(
            parse_envelope(raw).payload
            for raw in submitter.sent
            if parse_envelope(raw).type == MessageType.TASK_ACCEPTED
        )

        await server.task_service.detach_submitter(
            websocket=submitter,  # type: ignore[arg-type]
            trace_id="trace-detach",
            reason="submitter websocket disconnected",
        )

        assert server.store.task_row("task-running")["status"] == TaskState.RUNNING.value
        assert server.store.session_row(accepted.session_id)["status"] == SessionState.RUNNING.value
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert "task_orphaned_missing_submitter" not in event_types
        server.store.close()

    asyncio.run(_run())


def test_retry_queued_missing_submitter_terminalizes_without_event_storm(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        submitter = _FakeWebSocket()
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.TASK_REQUEST, _task("task-missing", "trace-missing"), trace_id="trace-missing"),
        )
        accepted = next(
            parse_envelope(raw).payload
            for raw in submitter.sent
            if parse_envelope(raw).type == MessageType.TASK_ACCEPTED
        )
        server.task_service._submitter_by_task_id.pop("task-missing", None)  # noqa: SLF001
        server.connections.unregister(submitter)  # type: ignore[arg-type]
        server.store.connection.execute(
            "UPDATE tasks SET created_at=? WHERE task_id=?",
            ("2020-01-01T00:00:00+00:00", "task-missing"),
        )
        server.store.connection.commit()

        await server.task_service.retry_queued_tasks(trace_id="trace-retry-a")
        await server.task_service.retry_queued_tasks(trace_id="trace-retry-b")

        assert server.store.task_row("task-missing")["status"] == TaskState.INTERRUPTED.value
        assert server.store.session_row(accepted.session_id)["status"] == SessionState.INTERRUPTED.value
        assert server.store.schedulable_task_rows() == []
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert event_types.count("task_orphaned_missing_submitter") == 1
        assert "task_retry_missing_submitter_mapping_grace" not in event_types
        server.store.close()

    asyncio.run(_run())


def test_retry_queued_missing_submitter_waits_during_grace(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        submitter = _FakeWebSocket()
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                _task("task-missing-grace", "trace-missing-grace"),
                trace_id="trace-missing-grace",
            ),
        )
        accepted = next(
            parse_envelope(raw).payload
            for raw in submitter.sent
            if parse_envelope(raw).type == MessageType.TASK_ACCEPTED
        )
        server.task_service._submitter_by_task_id.pop("task-missing-grace", None)  # noqa: SLF001
        server.connections.unregister(submitter)  # type: ignore[arg-type]

        await server.task_service.retry_queued_tasks(trace_id="trace-retry-a")
        await server.task_service.retry_queued_tasks(trace_id="trace-retry-b")

        assert server.store.task_row("task-missing-grace")["status"] == TaskState.ACCEPTED.value
        assert server.store.session_row(accepted.session_id)["status"] == SessionState.PENDING.value
        assert len(server.store.schedulable_task_rows()) == 1
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert event_types.count("task_retry_missing_submitter_mapping_grace") == 1
        assert "task_orphaned_missing_submitter" not in event_types
        server.store.close()

    asyncio.run(_run())


def test_retry_queued_recovers_submitter_mapping_from_registry(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        submitter = _FakeWebSocket()
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                _task("task-recover-mapping", "trace-recover-mapping"),
                trace_id="trace-recover-mapping",
            ),
        )
        accepted = next(
            parse_envelope(raw).payload
            for raw in submitter.sent
            if parse_envelope(raw).type == MessageType.TASK_ACCEPTED
        )
        server.task_service._submitter_by_task_id.pop("task-recover-mapping", None)  # noqa: SLF001

        await server.task_service.retry_queued_tasks(trace_id="trace-retry")

        assert server.task_service._submitter_by_task_id["task-recover-mapping"] is submitter  # noqa: SLF001
        assert server.store.task_row("task-recover-mapping")["status"] == TaskState.ACCEPTED.value
        assert server.store.session_row(accepted.session_id)["status"] == SessionState.PENDING.value
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert "task_submitter_mapping_recovered" in event_types
        assert "task_orphaned_missing_submitter" not in event_types
        server.store.close()

    asyncio.run(_run())


def test_accept_task_mapping_is_visible_before_persisted_task_can_retry(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        original_accept_async = server.queue.accept_async

        async def racing_accept_async(*args, **kwargs):
            await original_accept_async(*args, **kwargs)
            await server.task_service.retry_queued_tasks(trace_id="trace-race")

        server.queue.accept_async = racing_accept_async  # type: ignore[method-assign]
        submitter = _FakeWebSocket()
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                _task("task-accept-race", "trace-accept-race"),
                trace_id="trace-accept-race",
            ),
        )
        accepted = next(
            parse_envelope(raw).payload
            for raw in submitter.sent
            if parse_envelope(raw).type == MessageType.TASK_ACCEPTED
        )

        assert server.task_service._submitter_by_task_id["task-accept-race"] is submitter  # noqa: SLF001
        assert server.store.task_row("task-accept-race")["status"] == TaskState.ACCEPTED.value
        assert server.store.session_row(accepted.session_id)["status"] == SessionState.PENDING.value
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert "task_orphaned_missing_submitter" not in event_types
        server.store.close()

    asyncio.run(_run())


def test_retry_queued_disconnected_submitter_terminalizes_without_event_storm(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        submitter = _FakeWebSocket()
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                _task("task-disconnected", "trace-disconnected"),
                trace_id="trace-disconnected",
            ),
        )
        accepted = next(
            parse_envelope(raw).payload
            for raw in submitter.sent
            if parse_envelope(raw).type == MessageType.TASK_ACCEPTED
        )
        server.connections.unregister(submitter)  # type: ignore[arg-type]

        await server.task_service.retry_queued_tasks(trace_id="trace-retry-a")
        await server.task_service.retry_queued_tasks(trace_id="trace-retry-b")

        assert server.store.task_row("task-disconnected")["status"] == TaskState.INTERRUPTED.value
        assert server.store.session_row(accepted.session_id)["status"] == SessionState.INTERRUPTED.value
        assert server.store.schedulable_task_rows() == []
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert event_types.count("task_orphaned_missing_submitter") == 1
        assert "task_retry_skipped_disconnected_submitter" not in event_types
        server.store.close()

    asyncio.run(_run())


def test_strict_target_without_active_websocket_waits_for_rehello_without_rollback_storm(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        stale_device = _FakeWebSocket()
        await server._dispatch(
            stale_device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        server.connections.unregister(stale_device)  # type: ignore[arg-type]

        submitter = _FakeWebSocket()
        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                _task("task-target", "trace-target", target_device_id="dev-a"),
                trace_id="trace-target",
            ),
        )
        accepted = next(
            parse_envelope(raw).payload
            for raw in submitter.sent
            if parse_envelope(raw).type == MessageType.TASK_ACCEPTED
        )
        await server.task_service.retry_queued_tasks(trace_id="trace-retry-a")
        await server.task_service.retry_queued_tasks(trace_id="trace-retry-b")

        assert server.store.task_row("task-target")["status"] == TaskState.ACCEPTED.value
        assert server.store.session_row(accepted.session_id)["status"] == SessionState.PENDING.value
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert event_types.count("task_queued") == 1
        assert "lease_granted" not in event_types
        assert "lease_delivery_rolled_back" not in event_types

        rehello_device = _FakeWebSocket()
        await server._dispatch(
            rehello_device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-b"),
        )

        assert server.store.task_row("task-target")["status"] == TaskState.RUNNING.value
        assert server.store.session_row(accepted.session_id)["status"] == SessionState.RUNNING.value
        assert any(parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED for raw in rehello_device.sent)
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert event_types.count("lease_granted") == 1
        assert "lease_delivery_rolled_back" not in event_types
        server.store.close()

    asyncio.run(_run())
