"""Controller websocket affinity and queued target-device tests."""

from __future__ import annotations

import asyncio

import websockets

from tests.controller.test_controller_infra import (
    Capability,
    CommandRequest,
    CommandResult,
    CommandState,
    ControllerConfig,
    ControllerSecurityConfig,
    ControllerServer,
    DeviceHello,
    Envelope,
    MessageType,
    TaskRequest,
    _descriptor,
    _FakeWebSocket,
    _recv_until,
    command_idempotency_key,
    parse_envelope,
)


def test_real_websocket_two_sandbox_self_submit_affinity(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                host="127.0.0.1",
                port=0,
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        await server.start()
        assert server._server is not None
        port = server._server.sockets[0].getsockname()[1]
        uri = f"ws://127.0.0.1:{port}"
        async with websockets.connect(uri, proxy=None) as ws_a, websockets.connect(uri, proxy=None) as ws_b:
            await ws_a.send(
                Envelope.wrap(
                    MessageType.DEVICE_HELLO,
                    DeviceHello(descriptor=_descriptor("dev-a")),
                    trace_id="trace-a",
                ).model_dump_json()
            )
            await ws_b.send(
                Envelope.wrap(
                    MessageType.DEVICE_HELLO,
                    DeviceHello(descriptor=_descriptor("dev-b")),
                    trace_id="trace-b",
                ).model_dump_json()
            )
            await ws_a.send(
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
                ).model_dump_json()
            )
            await ws_b.send(
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
                ).model_dump_json()
            )
            await _recv_until(ws_a, MessageType.TASK_ACCEPTED)
            lease_a = (await _recv_until(ws_a, MessageType.SESSION_LEASE_GRANTED)).payload
            await _recv_until(ws_b, MessageType.TASK_ACCEPTED)
            lease_b = (await _recv_until(ws_b, MessageType.SESSION_LEASE_GRANTED)).payload
            assert lease_a.device_id == "dev-a"  # type: ignore[attr-defined]
            assert lease_b.device_id == "dev-b"  # type: ignore[attr-defined]

            command = CommandRequest(
                command_id="cmd-real-a",
                idempotency_key=command_idempotency_key(
                    session_id=lease_a.session_id,  # type: ignore[attr-defined]
                    lease_id=lease_a.lease_id,  # type: ignore[attr-defined]
                    command_id="cmd-real-a",
                    payload={"tool_id": "device.echo"},
                ),
                session_id=lease_a.session_id,  # type: ignore[attr-defined]
                lease_id=lease_a.lease_id,  # type: ignore[attr-defined]
                device_id="dev-a",
                tool_id="device.echo",
            )
            send_task = asyncio.create_task(server.command_bus.send(command, trace_id="trace-task-a"))
            command_envelope = await _recv_until(ws_a, MessageType.COMMAND_REQUEST)
            assert command_envelope.payload.device_id == "dev-a"  # type: ignore[attr-defined]
            try:
                await asyncio.wait_for(ws_b.recv(), timeout=0.05)
            except TimeoutError:
                pass
            else:
                raise AssertionError("dev-b should not receive dev-a command")
            await ws_a.send(
                Envelope.wrap(
                    MessageType.COMMAND_RESULT,
                    CommandResult(
                        command_id="cmd-real-a",
                        session_id=command.session_id,
                        lease_id=command.lease_id,
                        device_id="dev-a",
                        status=CommandState.SUCCEEDED,
                        result={"ok": True},
                    ),
                    trace_id="trace-task-a",
                ).model_dump_json()
            )
            assert (await send_task).status == CommandState.SUCCEEDED
        await server.stop()

    asyncio.run(_run())


def test_task_request_stays_accepted_when_matching_device_is_at_capacity(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _FakeWebSocket()
        server.registry.register(_descriptor("busy-focus", active_focus=True), trace_id="trace")
        server.registry.get("busy-focus").active_sessions = 1  # type: ignore[union-attr]

        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="queued-task",
                    trace_id="trace-queued",
                    agent_kind="desktop",
                    request="task waits",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-queued",
            ),
        )

        assert any(parse_envelope(raw).type == MessageType.TASK_ACCEPTED for raw in ws.sent)
        assert not any(parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED for raw in ws.sent)
        server.events.flush()
        assert any(row["event_type"] == "task_queued" for row in server.store.event_rows())
        server.store.close()

    asyncio.run(_run())


def test_target_device_id_is_strict_and_does_not_spill_to_other_device(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        submitter = _FakeWebSocket()
        await server._dispatch(
            _FakeWebSocket(),  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("busy-focus", active_focus=True)),
                trace_id="trace-busy",
            ),
        )
        await server._dispatch(
            _FakeWebSocket(),  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("free-dev")),
                trace_id="trace-free",
            ),
        )
        server.registry.get("busy-focus").active_sessions = 1  # type: ignore[union-attr]

        await server._dispatch(
            submitter,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="strict-task",
                    trace_id="trace-strict",
                    agent_kind="desktop",
                    request="stay on busy device",
                    target_device_id="busy-focus",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-strict",
            ),
        )

        assert any(parse_envelope(raw).type == MessageType.TASK_ACCEPTED for raw in submitter.sent)
        assert not any(parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED for raw in submitter.sent)
        assert server.store.session_row_for_task("strict-task")["status"] == "pending"
        server.store.close()

    asyncio.run(_run())
