"""Command-bus, idempotency, validation, and artifact materialization tests."""

from __future__ import annotations

import asyncio
import json
import time

from northcua.controller.auth import ControllerAuth
from northcua.controller.config import ControllerSecurityConfig
from northcua.controller.control_plane.registry import DeviceRegistry
from northcua.controller.execution import command_dispatcher
from northcua.controller.execution.command_bus import CommandBus, command_idempotency_key
from northcua.controller.observability.event_log import ControllerEventLog
from northcua.controller.storage.blob_store import LocalControllerBlobStore
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef, ArtifactScope, RetentionPolicy, StorageKind
from northcua_protocol.capabilities import Capability, DeviceDescriptor, PlatformKind, ToolDescriptor
from northcua_protocol.payloads import AGENTKIT_SETTINGS_PARAM, CommandRequest, CommandResult, TaskRequest
from northcua_protocol.runtime_sidecars import RUNTIME_SIDECAR_KEY
from northcua_protocol.state import CommandState, TaskState


def _store(tmp_path) -> SQLiteControllerStore:
    return SQLiteControllerStore(tmp_path / "controller.sqlite")


def _activate_session(
    store: SQLiteControllerStore,
    *,
    device_id: str = "dev-1",
    task_id: str = "task-1",
    session_id: str = "sess-1",
    lease_id: str = "lease-1",
    trace_id: str = "trace-1",
) -> None:
    request = TaskRequest(
        task_id=task_id,
        trace_id=trace_id,
        agent_kind="desktop",
        request="open calculator",
    )
    store.create_task(request, session_id=session_id, submitter_id="submitter-1", status=TaskState.ACCEPTED)
    store.create_lease(lease_id=lease_id, session_id=session_id, task_id=task_id, device_id=device_id)


def test_gui_lock_timeout_does_not_dispatch_late_after_lock_release(tmp_path, monkeypatch) -> None:
    async def run():
        store = _store(tmp_path)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=ControllerEventLog(store),
        )
        _activate_session(store)
        calls = []

        async def sender(command, trace_id):
            calls.append(command)
            raise AssertionError("queued command must never reach the driver")

        bus.register_sender("dev-1", sender)
        monkeypatch.setattr(bus._dispatcher, "_gui_resource_id_for", lambda request: "gui")
        lock = asyncio.Lock()
        await lock.acquire()
        bus._dispatcher._gui_resource_locks["gui"] = lock
        command = CommandRequest(
            command_id="queue-timeout",
            idempotency_key="queue-timeout",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=0.01,
        )
        try:
            result = await bus.send(command, trace_id="trace-1")
            assert result.status == CommandState.TIMED_OUT
        finally:
            lock.release()
        await asyncio.sleep(0)
        assert calls == []
        replay = await bus.send(command, trace_id="trace-1")
        assert replay.status == CommandState.TIMED_OUT
        assert calls == []

    asyncio.run(run())


def test_gui_lock_wait_is_deducted_from_sent_budget(tmp_path, monkeypatch) -> None:
    async def run():
        store = _store(tmp_path)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=ControllerEventLog(store),
        )
        _activate_session(store)
        budgets = []

        async def sender(command, trace_id):
            budgets.append(command.timeout_seconds)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        monkeypatch.setattr(bus._dispatcher, "_gui_resource_id_for", lambda request: "gui")
        lock = asyncio.Lock()
        await lock.acquire()
        bus._dispatcher._gui_resource_locks["gui"] = lock
        asyncio.get_running_loop().call_later(0.03, lock.release)
        command = CommandRequest(
            command_id="queue-budget",
            idempotency_key="queue-budget",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=1,
        )
        result = await bus.send(command, trace_id="trace-1")
        assert result.status == CommandState.SUCCEEDED
        assert len(budgets) == 1 and 0 < budgets[0] < 0.98

    asyncio.run(run())


def test_command_bus_deduplicates_results(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store)
        calls = 0

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal calls
            calls += 1
            assert trace_id == "trace-1"
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
                result={"ok": True},
            )

        bus.register_sender("dev-1", sender)
        idempotency_key = command_idempotency_key(
            session_id="sess-1",
            lease_id="lease-1",
            command_id="cmd-1",
            payload={"tool_id": "device.echo"},
        )
        command = CommandRequest(
            command_id="cmd-1",
            idempotency_key=idempotency_key,
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
        )

        first = await bus.send(command, trace_id="trace-1")
        second = await bus.send(command, trace_id="trace-1")
        event_rows = store.event_rows()
        event_types = [row["event_type"] for row in event_rows]

        assert first.status == CommandState.SUCCEEDED
        assert second.status == CommandState.SUCCEEDED
        assert calls == 1
        assert "command_result_received" in event_types
        assert "command_result_persist_done" in event_types
        assert event_types.index("command_result_received") < event_types.index("command_result_persist_done")
        for event_type in ("command_dispatch_claimed", "command_running", "command_result_persist_done"):
            payload = json.loads(next(row for row in event_rows if row["event_type"] == event_type)["payload_json"])
            assert {
                "store_queue_ms",
                "store_admission_ms",
                "store_executor_queue_ms",
                "store_queue_depth",
                "store_admitted_queue_depth",
            } <= set(payload)

    asyncio.run(_run())


def test_command_bus_returns_runtime_sidecar_without_persisting_it(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=ControllerEventLog(store),
        )
        _activate_session(store)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del trace_id
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                lease_generation=command.lease_generation,
                status=CommandState.SUCCEEDED,
                result={"ok": True, RUNTIME_SIDECAR_KEY: {"generation": 1}},
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-runtime-sidecar",
            idempotency_key="idem-runtime-sidecar",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
        )

        returned = await bus.send(command, trace_id="trace-1")
        row = store.command_row(command.command_id, session_id=command.session_id, lease_id=command.lease_id)

        assert isinstance(returned.result, dict)
        assert RUNTIME_SIDECAR_KEY in returned.result
        assert row is not None
        assert RUNTIME_SIDECAR_KEY not in str(row["result_json"])
        store.close()

    asyncio.run(_run())


def test_command_bus_does_not_attach_losing_sidecar_to_cached_result(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=ControllerEventLog(store),
        )
        _activate_session(store)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del trace_id
            assert (
                store.record_command_result(
                    CommandResult(
                        command_id=command.command_id,
                        session_id=command.session_id,
                        lease_id=command.lease_id,
                        device_id=command.device_id,
                        status=CommandState.SUCCEEDED,
                        result={"winner": "second"},
                    )
                )
                is None
            )
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
                result={"winner": "first", RUNTIME_SIDECAR_KEY: {"observation": "first"}},
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-sidecar-race",
            idempotency_key="idem-sidecar-race",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
        )

        returned = await bus.send(command, trace_id="trace-1")

        assert returned.result == {"winner": "second"}
        store.close()

    asyncio.run(_run())


def test_command_bus_allows_same_protocol_command_id_across_sessions(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store, device_id="dev-a")
        _activate_session(
            store,
            device_id="dev-b",
            task_id="task-2",
            session_id="sess-2",
            lease_id="lease-2",
            trace_id="trace-2",
        )
        seen: list[tuple[str, str]] = []

        def _sender_for(device_id: str):
            async def _sender(command: CommandRequest, trace_id: str) -> CommandResult:
                del trace_id
                seen.append((device_id, command.session_id))
                return CommandResult(
                    command_id=command.command_id,
                    session_id=command.session_id,
                    lease_id=command.lease_id,
                    device_id=device_id,
                    status=CommandState.SUCCEEDED,
                    result={"device_id": device_id, "session_id": command.session_id},
                )

            return _sender

        bus.register_sender("dev-a", _sender_for("dev-a"))
        bus.register_sender("dev-b", _sender_for("dev-b"))
        command_a = CommandRequest(
            command_id="cmd-shared",
            idempotency_key=command_idempotency_key(
                session_id="sess-1",
                lease_id="lease-1",
                command_id="cmd-shared",
                payload={"tool_id": "device.echo"},
            ),
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-a",
            tool_id="device.echo",
        )
        command_b = command_a.model_copy(
            update={
                "idempotency_key": command_idempotency_key(
                    session_id="sess-2",
                    lease_id="lease-2",
                    command_id="cmd-shared",
                    payload={"tool_id": "device.echo"},
                ),
                "session_id": "sess-2",
                "lease_id": "lease-2",
                "device_id": "dev-b",
            }
        )

        result_a, result_b = await asyncio.gather(
            bus.send(command_a, trace_id="trace-1"),
            bus.send(command_b, trace_id="trace-2"),
        )

        assert result_a.result == {"device_id": "dev-a", "session_id": "sess-1"}
        assert result_b.result == {"device_id": "dev-b", "session_id": "sess-2"}
        assert seen == [("dev-a", "sess-1"), ("dev-b", "sess-2")]
        assert store.command_row("cmd-shared", session_id="sess-1", lease_id="lease-1") is not None
        assert store.command_row("cmd-shared", session_id="sess-2", lease_id="lease-2") is not None

    asyncio.run(_run())


def test_controller_events_redact_secrets_and_inline_data(tmp_path) -> None:
    store = _store(tmp_path)
    events = ControllerEventLog(store)

    events.write(
        "sensitive_event",
        trace_id="trace-redact",
        entity_type="task",
        entity_id="task-redact",
        token="submit-secret",
        nested={"Authorization": "Bearer device-token", "image": "data:image/png;base64,ZmFrZQ=="},
    )

    payload = json.loads(store.event_rows()[0]["payload_json"])
    assert payload["token"] == "<redacted>"
    assert payload["nested"]["Authorization"] == "<redacted>"
    assert payload["nested"]["image"] == "<data-url:image/png:4 bytes>"


def test_command_bus_rejects_idempotency_conflicts(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del trace_id
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                lease_generation=command.lease_generation,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        first = CommandRequest(
            command_id="cmd-conflict",
            idempotency_key="idem-conflict",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            parameters={"value": 1},
        )
        conflicting = first.model_copy(update={"parameters": {"value": 2}})

        assert (await bus.send(first, trace_id="trace-1")).status == CommandState.SUCCEEDED
        try:
            await bus.send(conflicting, trace_id="trace-1")
        except ValueError as exc:
            assert "idempotency conflict" in str(exc)
        else:
            raise AssertionError("conflicting duplicate command should be rejected")

    asyncio.run(_run())


def test_command_bus_waits_for_inflight_duplicate_without_double_dispatch(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store)
        release_sender = asyncio.Event()
        calls = 0

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal calls
            del trace_id
            calls += 1
            await release_sender.wait()
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-inflight",
            idempotency_key="idem-inflight",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
        )
        first = asyncio.create_task(bus.send(command, trace_id="trace-1"))
        second = asyncio.create_task(bus.send(command, trace_id="trace-1"))
        await asyncio.sleep(0)
        release_sender.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result.status == CommandState.SUCCEEDED
        assert second_result.status == CommandState.SUCCEEDED
        assert calls == 1

    asyncio.run(_run())


def test_command_bus_slow_claim_does_not_block_event_loop(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store)
        original_claim = SQLiteControllerStore.claim_command_in_transaction

        def slow_claim(self: SQLiteControllerStore, request: CommandRequest):
            time.sleep(0.05)
            return original_claim(self, request)

        monkeypatch.setattr(SQLiteControllerStore, "claim_command_in_transaction", slow_claim)
        ticks = 0
        stop = False

        async def ticker() -> None:
            nonlocal ticks
            while not stop:
                ticks += 1
                await asyncio.sleep(0.005)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del trace_id
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        ticker_task = asyncio.create_task(ticker())
        try:
            send_task = asyncio.create_task(
                bus.send(
                    CommandRequest(
                        command_id="cmd-slow-claim",
                        idempotency_key="idem-slow-claim",
                        session_id="sess-1",
                        lease_id="lease-1",
                        device_id="dev-1",
                        tool_id="device.echo",
                    ),
                    trace_id="trace-1",
                )
            )
            await asyncio.sleep(0.03)
        finally:
            stop = True
            await ticker_task
        result = await send_task

        assert result.status == CommandState.SUCCEEDED
        assert ticks >= 3

    asyncio.run(_run())


def test_command_bus_does_not_leak_claim_lock_on_invalid_lease(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        command = CommandRequest(
            command_id="cmd-invalid-lease",
            idempotency_key="idem-invalid-lease",
            session_id="missing-session",
            lease_id="missing-lease",
            device_id="dev-1",
            tool_id="device.echo",
        )

        try:
            await bus.send(command, trace_id="trace-1")
        except PermissionError:
            pass
        else:
            raise AssertionError("invalid lease should fail validation")

        assert bus._claim_locks_by_idempotency_key == {}

    asyncio.run(_run())


def test_command_bus_cancelled_during_claim_does_not_leave_queued_command(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store)
        original_claim = SQLiteControllerStore.claim_command_in_transaction

        def slow_after_commit(self: SQLiteControllerStore, request: CommandRequest):
            result = original_claim(self, request)
            time.sleep(0.05)
            return result

        monkeypatch.setattr(SQLiteControllerStore, "claim_command_in_transaction", slow_after_commit)
        command = CommandRequest(
            command_id="cmd-cancel-during-claim",
            idempotency_key="idem-cancel-during-claim",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
        )
        send_task = asyncio.create_task(bus.send(command, trace_id="trace-1"))
        await asyncio.sleep(0.01)
        send_task.cancel()

        try:
            await send_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("send task should observe cancellation")

        row = store.command_row(command.command_id, session_id=command.session_id, lease_id=command.lease_id)
        assert row is not None
        assert row["status"] == CommandState.CANCELLED.value
        assert row["result_json"]
        assert bus._claim_locks_by_idempotency_key == {}

    asyncio.run(_run())


def test_command_dispatch_cancelled_after_running_records_cancelled_result(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store)
        sender_started = asyncio.Event()
        allow_sender = asyncio.Event()

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del trace_id
            sender_started.set()
            await allow_sender.wait()
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-cancel-during-dispatch",
            idempotency_key="idem-cancel-during-dispatch",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
        )
        store.claim_command(command)
        dispatch_task = asyncio.create_task(bus._dispatcher.dispatch(command, trace_id="trace-1"))
        await asyncio.wait_for(sender_started.wait(), timeout=1.0)
        dispatch_task.cancel()

        try:
            await dispatch_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("dispatcher should preserve cancellation")

        row = store.command_row(command.command_id, session_id=command.session_id, lease_id=command.lease_id)
        assert row is not None
        assert row["status"] == CommandState.CANCELLED.value
        assert "COMMAND_DISPATCH_CANCELLED" in str(row["result_json"])

    asyncio.run(_run())


def test_command_bus_times_out_and_keeps_late_result_terminal(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del trace_id
            await asyncio.sleep(0.05)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-timeout",
            idempotency_key="idem-timeout",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=0.001,
        )

        result = await bus.send(command, trace_id="trace-1")
        cached = bus.record_result(
            CommandResult(
                command_id="cmd-timeout",
                session_id="sess-1",
                lease_id="lease-1",
                device_id="dev-1",
                status=CommandState.SUCCEEDED,
            ),
            trace_id="trace-1",
        )

        assert result.status == CommandState.TIMED_OUT
        assert cached is not None
        assert cached.status == CommandState.TIMED_OUT

    asyncio.run(_run())


def test_command_bus_skips_artifact_materialization_for_late_terminal_result(tmp_path) -> None:
    class ExplodingBlobStore(LocalControllerBlobStore):
        def materialize_large_inline(self, artifact: ArtifactRef, *, max_inline_bytes: int) -> ArtifactRef:
            del artifact, max_inline_bytes
            raise AssertionError("late terminal results must not materialize artifacts")

    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            blob_store=ExplodingBlobStore(tmp_path / "blobs"),
            max_inline_artifact_bytes=1,
        )
        _activate_session(store)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del trace_id
            await asyncio.sleep(0.05)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-late-artifact",
            idempotency_key="idem-late-artifact",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=0.001,
        )

        result = await bus.send(command, trace_id="trace-1")
        cached = bus.record_result(
            CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
                artifacts=[
                    ArtifactRef.from_inline_bytes(
                        artifact_id="late-screenshot",
                        kind=ArtifactKind.SCREENSHOT,
                        data=b"late image bytes",
                        mime_type="image/png",
                    )
                ],
            ),
            trace_id="trace-1",
        )
        event_payloads = [
            json.loads(row["payload_json"]) for row in store.event_rows() if row["event_type"] == "late_command_result"
        ]

        assert result.status == CommandState.TIMED_OUT
        assert cached is not None
        assert cached.status == CommandState.TIMED_OUT
        assert event_payloads[-1]["skipped_materialization"] is True
        assert store.artifact_rows() == []

    asyncio.run(_run())


def test_async_command_bus_skips_unowned_artifact_for_terminal_result(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        blob_store = LocalControllerBlobStore(tmp_path / "blobs")
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=ControllerEventLog(store),
            blob_store=blob_store,
            max_inline_artifact_bytes=1,
        )
        _activate_session(store)
        command = CommandRequest(
            command_id="cmd-async-late-artifact",
            idempotency_key="idem-async-late-artifact",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
        )
        assert store.claim_command(command)[0] == "new"
        assert store.mark_command_dispatched_for_request(command) is None
        assert (
            store.record_command_result(
                CommandResult(
                    command_id=command.command_id,
                    session_id=command.session_id,
                    lease_id=command.lease_id,
                    device_id=command.device_id,
                    status=CommandState.TIMED_OUT,
                )
            )
            is None
        )
        owned = blob_store.materialize(
            ArtifactRef.from_inline_bytes(
                artifact_id="late-owned",
                kind=ArtifactKind.SCREENSHOT,
                data=b"late image",
                mime_type="image/png",
            )
        )

        cached = await bus.record_result_async(
            CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
                artifacts=[owned.model_copy(update={"artifact_id": "late-unowned"})],
            ),
            trace_id="trace-1",
        )

        assert cached.status == CommandState.TIMED_OUT
        assert store.artifact_rows() == []
        store.close()

    asyncio.run(_run())


def test_command_bus_accepts_device_timeout_result_during_grace(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setattr(command_dispatcher, "_CONTROLLER_DEVICE_RESULT_GRACE_FRACTION", 10.0)
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
        )
        _activate_session(store)

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            del trace_id
            await asyncio.sleep(0.03)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.TIMED_OUT,
                error="DEVICE_SIDE_TIMEOUT",
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-device-timeout",
            idempotency_key="idem-device-timeout",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=0.01,
        )

        result = await bus.send(command, trace_id="trace-1")
        event_types = [row["event_type"] for row in store.event_rows()]

        assert result.status == CommandState.TIMED_OUT
        assert result.error == "DEVICE_SIDE_TIMEOUT"
        assert "command_result_arrived_during_controller_grace" in event_types

    asyncio.run(_run())


def test_command_dispatcher_result_grace_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_RESULT_GRACE_FRACTION", "0.5")
    monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_RESULT_GRACE_MAX_SECONDS", "60")

    assert command_dispatcher._device_result_wait_timeout(10) == 15
    assert command_dispatcher._device_result_wait_timeout(180) == 240


def test_command_bus_caps_controller_timeout_to_registered_tool_descriptor(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.LINUX,
                capabilities={Capability.DESKTOP_INPUT},
                tools=[
                    ToolDescriptor(
                        tool_id="device.echo",
                        capability=Capability.DESKTOP_INPUT,
                        timeout_seconds=0.01,
                    )
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        observed_timeout = None

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal observed_timeout
            del trace_id
            observed_timeout = command.timeout_seconds
            await asyncio.sleep(0.05)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-capped-timeout",
            idempotency_key="idem-capped-timeout",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=60.0,
        )

        result = await bus.send(command, trace_id="trace-1")
        event_types = [row["event_type"] for row in store.event_rows()]

        assert observed_timeout == 0.01
        assert result.status == CommandState.TIMED_OUT
        assert "command_timeout_capped_by_descriptor" in event_types

    asyncio.run(_run())


def test_command_bus_uses_legacy_action_timeout_as_floor(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.delenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_FLOOR_SECONDS", raising=False)
        monkeypatch.setenv("NORTHCUA_ACTION_COMMAND_TIMEOUT_SECONDS", "0.04")
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.LINUX,
                capabilities={Capability.DESKTOP_INPUT},
                tools=[
                    ToolDescriptor(
                        tool_id="device.echo",
                        capability=Capability.DESKTOP_INPUT,
                        timeout_seconds=0.01,
                    )
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        observed_timeout = None

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal observed_timeout
            del trace_id
            observed_timeout = command.timeout_seconds
            await asyncio.sleep(0.02)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-legacy-floor-timeout",
            idempotency_key="idem-legacy-floor-timeout",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=60.0,
        )

        result = await bus.send(command, trace_id="trace-1")
        cap_event = next(
            row for row in store.event_rows() if row["event_type"] == "command_timeout_capped_by_descriptor"
        )
        payload = json.loads(cap_event["payload_json"])

        assert observed_timeout == 0.04
        assert result.status == CommandState.SUCCEEDED
        assert payload["descriptor_timeout_seconds"] == 0.01
        assert payload["effective_timeout_seconds"] == 0.04
        assert payload["configured_floor_seconds"] == 0.04

    asyncio.run(_run())


def test_command_bus_uses_configured_timeout_floor_when_descriptor_is_too_short(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_FLOOR_SECONDS", "0.04")
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.LINUX,
                capabilities={Capability.DESKTOP_INPUT},
                tools=[
                    ToolDescriptor(
                        tool_id="device.echo",
                        capability=Capability.DESKTOP_INPUT,
                        timeout_seconds=0.01,
                    )
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        observed_timeout = None

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal observed_timeout
            del trace_id
            observed_timeout = command.timeout_seconds
            await asyncio.sleep(0.02)
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-floor-timeout",
            idempotency_key="idem-floor-timeout",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=60.0,
        )

        result = await bus.send(command, trace_id="trace-1")
        cap_event = next(
            row for row in store.event_rows() if row["event_type"] == "command_timeout_capped_by_descriptor"
        )
        payload = json.loads(cap_event["payload_json"])

        assert observed_timeout == 0.04
        assert result.status == CommandState.SUCCEEDED
        assert payload["descriptor_timeout_seconds"] == 0.01
        assert payload["effective_timeout_seconds"] == 0.04

    asyncio.run(_run())


def test_command_bus_short_descriptor_floor_is_bounded_below_global_floor(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_FLOOR_SECONDS", "180")
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.LINUX,
                capabilities={Capability.DESKTOP_INPUT},
                tools=[
                    ToolDescriptor(
                        tool_id="device.echo",
                        capability=Capability.DESKTOP_INPUT,
                        timeout_seconds=10.0,
                    )
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        observed_timeout = None
        observed_metadata = None

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal observed_timeout, observed_metadata
            del trace_id
            observed_timeout = command.timeout_seconds
            observed_metadata = command.metadata
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-short-floor-timeout",
            idempotency_key="idem-short-floor-timeout",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=2400.0,
        )

        result = await bus.send(command, trace_id="trace-1")
        cap_event = next(
            row for row in store.event_rows() if row["event_type"] == "command_timeout_capped_by_descriptor"
        )
        payload = json.loads(cap_event["payload_json"])

        assert result.status == CommandState.SUCCEEDED
        assert observed_timeout == 90.0
        assert observed_metadata["controller_timeout_policy"]["effective_timeout_seconds"] == 90.0
        assert payload["effective_timeout_seconds"] == 90.0
        assert payload["configured_floor_seconds"] == 180.0
        assert payload["floor_class"] == "short"

    asyncio.run(_run())


def test_command_bus_applies_floor_metadata_without_capping_request(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_FLOOR_SECONDS", "180")
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.LINUX,
                capabilities={Capability.DESKTOP_INPUT},
                tools=[
                    ToolDescriptor(
                        tool_id="device.echo",
                        capability=Capability.DESKTOP_INPUT,
                        timeout_seconds=10.0,
                    )
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        observed_timeout = None
        observed_metadata = None

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal observed_timeout, observed_metadata
            del trace_id
            observed_timeout = command.timeout_seconds
            observed_metadata = command.metadata
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-floor-applied",
            idempotency_key="idem-floor-applied",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=60.0,
        )

        result = await bus.send(command, trace_id="trace-1")
        floor_event = next(row for row in store.event_rows() if row["event_type"] == "command_timeout_floor_applied")
        payload = json.loads(floor_event["payload_json"])

        assert result.status == CommandState.SUCCEEDED
        assert observed_timeout == 60.0
        assert observed_metadata["controller_timeout_policy"]["effective_timeout_seconds"] == 90.0
        assert payload["effective_timeout_seconds"] == 90.0
        assert payload["floor_class"] == "short"

    asyncio.run(_run())


def test_command_bus_honors_web_agent_side_channel_timeout_over_short_descriptor(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.delenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_FLOOR_SECONDS", raising=False)
        monkeypatch.delenv("NORTHCUA_ACTION_COMMAND_TIMEOUT_SECONDS", raising=False)
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.LINUX,
                capabilities={Capability.BROWSER_PAGE},
                tools=[
                    ToolDescriptor(
                        tool_id="web.web_agent_snapshot",
                        capability=Capability.BROWSER_PAGE,
                        timeout_seconds=10.0,
                    )
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        observed_timeout = None
        observed_metadata = None

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal observed_timeout, observed_metadata
            del trace_id
            observed_timeout = command.timeout_seconds
            observed_metadata = command.metadata
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-web-side-channel-timeout",
            idempotency_key="idem-web-side-channel-timeout",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="web.web_agent_snapshot",
            timeout_seconds=90.0,
            parameters={AGENTKIT_SETTINGS_PARAM: {"command_timeout_seconds": 90.0}},
        )

        result = await bus.send(command, trace_id="trace-1")
        floor_event = next(row for row in store.event_rows() if row["event_type"] == "command_timeout_floor_applied")
        payload = json.loads(floor_event["payload_json"])

        assert result.status == CommandState.SUCCEEDED
        assert observed_timeout == 90.0
        assert observed_metadata["controller_timeout_policy"]["descriptor_timeout_seconds"] == 10.0
        assert observed_metadata["controller_timeout_policy"]["runtime_timeout_seconds"] == 90.0
        assert observed_metadata["controller_timeout_policy"]["effective_timeout_seconds"] == 90.0
        assert payload["floor_class"] == "runtime_side_channel"

    asyncio.run(_run())


def test_command_bus_raises_web_agent_request_timeout_from_side_channel(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.delenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_FLOOR_SECONDS", raising=False)
        monkeypatch.delenv("NORTHCUA_ACTION_COMMAND_TIMEOUT_SECONDS", raising=False)
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.LINUX,
                capabilities={Capability.BROWSER_PAGE},
                tools=[
                    ToolDescriptor(
                        tool_id="web.web_agent_snapshot",
                        capability=Capability.BROWSER_PAGE,
                        timeout_seconds=10.0,
                    )
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        observed_timeout = None

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal observed_timeout
            del trace_id
            observed_timeout = command.timeout_seconds
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-web-side-channel-raise-timeout",
            idempotency_key="idem-web-side-channel-raise-timeout",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="web.web_agent_snapshot",
            timeout_seconds=10.0,
            parameters={AGENTKIT_SETTINGS_PARAM: {"command_timeout_seconds": 90.0}},
        )

        result = await bus.send(command, trace_id="trace-1")
        raise_event = next(
            row for row in store.event_rows() if row["event_type"] == "command_timeout_raised_by_runtime_side_channel"
        )
        payload = json.loads(raise_event["payload_json"])

        assert result.status == CommandState.SUCCEEDED
        assert observed_timeout == 90.0
        assert payload["descriptor_timeout_seconds"] == 10.0
        assert payload["runtime_timeout_seconds"] == 90.0
        assert payload["effective_timeout_seconds"] == 90.0

    asyncio.run(_run())


def test_command_bus_long_descriptor_keeps_global_timeout_floor(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_FLOOR_SECONDS", "180")
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.LINUX,
                capabilities={Capability.DESKTOP_INPUT},
                tools=[
                    ToolDescriptor(
                        tool_id="long.operation",
                        capability=Capability.DESKTOP_INPUT,
                        timeout_seconds=120.0,
                    )
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        observed_timeout = None

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal observed_timeout
            del trace_id
            observed_timeout = command.timeout_seconds
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-long-floor-timeout",
            idempotency_key="idem-long-floor-timeout",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="long.operation",
            timeout_seconds=2400.0,
        )

        await bus.send(command, trace_id="trace-1")
        cap_event = next(
            row for row in store.event_rows() if row["event_type"] == "command_timeout_capped_by_descriptor"
        )
        payload = json.loads(cap_event["payload_json"])

        assert observed_timeout == 180.0
        assert payload["effective_timeout_seconds"] == 180.0
        assert payload["floor_class"] == "long"

    asyncio.run(_run())


def test_command_bus_short_timeout_floor_can_be_overridden(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_TIMEOUT_FLOOR_SECONDS", "180")
        monkeypatch.setenv("NORTHCUA_CONTROLLER_COMMAND_SHORT_TIMEOUT_FLOOR_SECONDS", "45")
        store = _store(tmp_path)
        events = ControllerEventLog(store)
        registry = DeviceRegistry(store=store, events=events)
        registry.register(
            DeviceDescriptor(
                device_id="dev-1",
                platform=PlatformKind.LINUX,
                capabilities={Capability.DESKTOP_INPUT},
                tools=[
                    ToolDescriptor(tool_id="device.echo", capability=Capability.DESKTOP_INPUT, timeout_seconds=10.0)
                ],
            ),
            trace_id="trace-1",
        )
        bus = CommandBus(
            store=store,
            auth=ControllerAuth(ControllerSecurityConfig(insecure_dev_no_auth=True)),
            events=events,
            registry=registry,
        )
        _activate_session(store)

        observed_timeout = None

        async def sender(command: CommandRequest, trace_id: str) -> CommandResult:
            nonlocal observed_timeout
            del trace_id
            observed_timeout = command.timeout_seconds
            return CommandResult(
                command_id=command.command_id,
                session_id=command.session_id,
                lease_id=command.lease_id,
                device_id=command.device_id,
                status=CommandState.SUCCEEDED,
            )

        bus.register_sender("dev-1", sender)
        command = CommandRequest(
            command_id="cmd-short-override",
            idempotency_key="idem-short-override",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            timeout_seconds=2400.0,
        )

        await bus.send(command, trace_id="trace-1")

        assert observed_timeout == 45.0

    asyncio.run(_run())
