"""Tests for controller task-finish authorization boundaries."""

from __future__ import annotations

import asyncio

from northcua.controller.api.server import ControllerServer
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua_protocol.capabilities import Capability
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope, parse_envelope
from northcua_protocol.payloads import DeviceHello, TaskFinished, TaskRequest
from northcua_protocol.state import TaskState
from tests.controller.test_controller_infra import _descriptor, _FakeWebSocket


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
