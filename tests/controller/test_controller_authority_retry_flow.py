"""Controller authority retry and terminal flow regression tests."""

from __future__ import annotations

import asyncio

from tests.controller.test_controller_authority_and_retry import (
    ArtifactKind,
    ArtifactRef,
    Capability,
    CommandRequest,
    CommandResult,
    CommandState,
    ControllerConfig,
    ControllerSecurityConfig,
    ControllerServer,
    DeviceHeartbeat,
    DeviceHello,
    Envelope,
    LocalControllerBlobStore,
    MessageType,
    TaskCancel,
    TaskFinished,
    TaskRequest,
    TaskState,
    _descriptor,
    _FakeWebSocket,
    command_idempotency_key,
    parse_envelope,
)


def test_task_request_rejects_external_controller_blob_inputs(tmp_path) -> None:
    sha = "a" * 64
    server = ControllerServer(
        ControllerConfig(
            db_path=str(tmp_path / "controller.sqlite"),
            security=ControllerSecurityConfig(insecure_dev_no_auth=True),
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
                        task_id="task-blob",
                        trace_id="trace-blob",
                        agent_kind="desktop",
                        request="task",
                        input_artifacts=[
                            ArtifactRef.from_controller_blob(
                                artifact_id="blob-1",
                                kind=ArtifactKind.TEXT,
                                uri=f"controller-blob://{sha}",
                                mime_type="text/plain",
                                sha256=sha,
                                size_bytes=1,
                            )
                        ],
                    ),
                    trace_id="trace-blob",
                ),
            )
        except ValueError as exc:
            assert "controller_blob task input" in str(exc)
        else:
            raise AssertionError("external controller_blob task inputs should be rejected")
        finally:
            server.store.close()

    asyncio.run(_run())


def test_controller_owned_completion_allows_verified_controller_blob_artifacts(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                blob_dir=str(tmp_path / "blobs"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        ws = _FakeWebSocket()
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("dev-a")),
                trace_id="trace-device",
            ),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-controller-blob",
                    trace_id="trace-task",
                    agent_kind="desktop",
                    request="finish me",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-task",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in ws.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        artifact = server.blob_store.materialize(
            ArtifactRef.from_inline_bytes(
                artifact_id="task-controller-blob-screenshot",
                kind=ArtifactKind.SCREENSHOT,
                data=b"png",
                mime_type="image/png",
            )
        )

        await server.task_service.complete_task(
            payload=TaskFinished(
                task_id="task-controller-blob",
                trace_id="trace-task",
                session_id=lease.session_id,  # type: ignore[attr-defined]
                device_id="dev-a",
                lease_id=lease.lease_id,  # type: ignore[attr-defined]
                state=TaskState.FINISHED,
                result={"ok": True},
                artifacts=[artifact],
            )
        )

        finished = [parse_envelope(raw) for raw in ws.sent if parse_envelope(raw).type == MessageType.TASK_FINISHED]
        assert finished
        assert finished[-1].payload.artifacts[0].storage_kind.value == "controller_blob"
        assert server.store.task_rows()[0]["status"] == TaskState.FINISHED.value
        server.store.close()

    asyncio.run(_run())


def test_controller_blob_store_verifies_existing_hash_file(tmp_path) -> None:
    artifact = ArtifactRef.from_inline_bytes(
        artifact_id="artifact-1",
        kind=ArtifactKind.TEXT,
        data=b"trusted",
        mime_type="text/plain",
    )
    store = LocalControllerBlobStore(tmp_path / "blobs")
    path = store.path_for_sha256(artifact.sha256)
    path.write_bytes(b"tampered")

    try:
        store.materialize(artifact)
    except ValueError as exc:
        assert "integrity mismatch" in str(exc)
    else:
        raise AssertionError("existing hash-named blob with wrong bytes should be rejected")


def test_queued_task_retries_when_device_capacity_frees(tmp_path) -> None:
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
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("focus-dev", active_focus=True)),
                trace_id="trace-device",
            ),
        )
        server.registry.get("focus-dev").active_sessions = 1  # type: ignore[union-attr]
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
        assert not any(parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED for raw in ws.sent)

        server.registry.release_session("focus-dev")
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HEARTBEAT,
                DeviceHeartbeat(device_id="focus-dev"),
                trace_id="trace-heartbeat",
            ),
        )

        assert any(parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED for raw in ws.sent)
        server.store.close()

    asyncio.run(_run())


def test_controller_forwards_task_finished_to_submitter(tmp_path) -> None:
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
            Envelope.wrap(
                MessageType.DEVICE_HELLO,
                DeviceHello(descriptor=_descriptor("dev-a")),
                trace_id="trace-device",
            ),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-a",
                    trace_id="trace-task",
                    agent_kind="desktop",
                    request="finish me",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-task",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in ws.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_FINISHED,
                TaskFinished(
                    task_id="task-a",
                    trace_id="trace-task",
                    session_id=lease.session_id,  # type: ignore[attr-defined]
                    device_id="dev-a",
                    lease_id=lease.lease_id,  # type: ignore[attr-defined]
                    state=TaskState.FINISHED,
                    result={"ok": True},
                    terminal_decision={"action": "finish", "reason": "completed"},
                ),
                trace_id="trace-task",
            ),
        )

        finished = [parse_envelope(raw) for raw in ws.sent if parse_envelope(raw).type == MessageType.TASK_FINISHED]
        assert finished
        assert finished[-1].payload.terminal_decision == {"action": "finish", "reason": "completed"}
        assert server.store.task_rows()[0]["status"] == TaskState.FINISHED.value
        server.store.close()

    asyncio.run(_run())


def test_task_finished_cannot_finish_another_task_with_same_device_lease(tmp_path) -> None:
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
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        for task_id in ("task-owned", "task-victim"):
            await server._dispatch(
                ws,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_REQUEST,
                    TaskRequest(
                        task_id=task_id,
                        trace_id=f"trace-{task_id}",
                        agent_kind="desktop",
                        request=task_id,
                        required_capabilities={Capability.DESKTOP_INPUT},
                    ),
                    trace_id=f"trace-{task_id}",
                ),
            )
        leases = [
            parse_envelope(raw).payload
            for raw in ws.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        ]
        owned_lease = next(lease for lease in leases if lease.task_id == "task-owned")  # type: ignore[attr-defined]
        try:
            await server._dispatch(
                ws,  # type: ignore[arg-type]
                Envelope.wrap(
                    MessageType.TASK_FINISHED,
                    TaskFinished(
                        task_id="task-victim",
                        trace_id="trace-victim",
                        session_id=owned_lease.session_id,  # type: ignore[attr-defined]
                        device_id="dev-a",
                        lease_id=owned_lease.lease_id,  # type: ignore[attr-defined]
                        state=TaskState.FINISHED,
                    ),
                    trace_id="trace-victim",
                ),
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("device should not finish a different task through its active lease")
        assert server.store.task_row("task-victim")["status"] == TaskState.RUNNING.value
        server.store.close()

    asyncio.run(_run())


def test_task_cancel_makes_late_command_result_terminal_cancelled(tmp_path) -> None:
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
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor("dev-a")), trace_id="trace-a"),
        )
        await server._dispatch(
            ws,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.TASK_REQUEST,
                TaskRequest(
                    task_id="task-cancel",
                    trace_id="trace-cancel",
                    agent_kind="desktop",
                    request="cancel me",
                    required_capabilities={Capability.DESKTOP_INPUT},
                ),
                trace_id="trace-cancel",
            ),
        )
        lease = next(
            parse_envelope(raw).payload
            for raw in ws.sent
            if parse_envelope(raw).type == MessageType.SESSION_LEASE_GRANTED
        )
        command = CommandRequest(
            command_id="cmd-cancel",
            idempotency_key=command_idempotency_key(
                session_id=lease.session_id,  # type: ignore[attr-defined]
                lease_id=lease.lease_id,  # type: ignore[attr-defined]
                command_id="cmd-cancel",
                payload={"tool_id": "device.echo"},
            ),
            session_id=lease.session_id,  # type: ignore[attr-defined]
            lease_id=lease.lease_id,  # type: ignore[attr-defined]
            device_id="dev-a",
            tool_id="device.echo",
        )
        send_task = asyncio.create_task(server.command_bus.send(command, trace_id="trace-cancel"))
        for _ in range(10):
            await asyncio.sleep(0)
            if any(parse_envelope(raw).type == MessageType.COMMAND_REQUEST for raw in ws.sent):
                break
        await server.task_service.cancel_task(
            websocket=ws,  # type: ignore[arg-type]
            payload=TaskCancel(
                task_id="task-cancel",
                trace_id="trace-cancel",
                session_id=lease.session_id,  # type: ignore[attr-defined]
                reason="test cancel",
            ),
        )
        assert (await send_task).status == CommandState.CANCELLED
        cached = server.command_bus.record_result(
            CommandResult(
                command_id="cmd-cancel",
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id="dev-a",
                status=CommandState.SUCCEEDED,
            ),
            trace_id="trace-cancel",
        )
        assert cached is not None
        assert cached.status == CommandState.CANCELLED
        server.store.close()

    asyncio.run(_run())
