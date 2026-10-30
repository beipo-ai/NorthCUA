"""Tests for controller task scheduling, cancellation, and finish ownership."""

from __future__ import annotations

import asyncio

from northcua.controller.api.server import ControllerServer
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua_protocol.capabilities import Capability
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope, parse_envelope
from northcua_protocol.payloads import DeviceHello, TaskCancel, TaskFinished, TaskRequest
from northcua_protocol.state import TaskState
from tests.controller.test_controller_infra import _descriptor, _FakeWebSocket, _RecordingExecutor


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

    asyncio.run(_run())


def test_task_finish_sender_must_own_task_or_lease(tmp_path) -> None:
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
                    trace_id="trace-task",
                    agent_kind="desktop",
                    request="task A",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-task",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in owner.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )

        try:
            await server._dispatch(
                intruder,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_FINISHED,
                    TaskFinished(
                        task_id="task-a",
                        trace_id="trace-task",
                        session_id=lease.session_id,  # type: ignore[attr-defined]
                        device_id="dev-a",
                        lease_id=lease.lease_id,  # type: ignore[attr-defined]
                        state=TaskState.FINISHED,
                    ),
                    trace_id="trace-task",
                ),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("intruder should not finish another device task")

        server.store.close()

    asyncio.run(_run())
