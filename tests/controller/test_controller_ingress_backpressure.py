"""Controller ingress backpressure safeguards."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone

from northcua.controller.api.command_waits import PendingCommandResult
from northcua.controller.api.dispatch import _should_persist_device_telemetry
from northcua.controller.api.server import ControllerServer, _DeviceTelemetryIngressLagTracker
from northcua.controller.config import ControllerConfig, ControllerSecurityConfig
from northcua.controller.control_plane.registry import DeviceRegistry
from northcua.controller.observability.event_log import ControllerEventLog
from northcua.controller.storage.async_store import controller_store_call_profiled
from northcua.controller.storage.blob_store import LocalControllerBlobStore
from northcua.controller.storage.persistence import SQLiteControllerStore, command_storage_key
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef
from northcua_protocol.capabilities import Capability, DeviceDescriptor, PlatformKind, ToolDescriptor
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope, parse_envelope
from northcua_protocol.payloads import (
    CommandRequest,
    CommandResult,
    DeviceHeartbeat,
    DeviceHello,
    DeviceTelemetryEvent,
    TaskRequest,
)
from northcua_protocol.runtime_sidecars import RUNTIME_SIDECAR_KEY
from northcua_protocol.state import CommandState, TaskState


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)


class _SequenceWebSocket(_FakeWebSocket):
    def __init__(self, messages: list[str]) -> None:
        super().__init__()
        self._messages = iter(messages)
        self.remote_address = ("127.0.0.1", 12345)
        self.close_code = 1000
        self.close_reason = "test complete"

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _descriptor(device_id: str = "dev-1") -> DeviceDescriptor:
    return DeviceDescriptor(
        device_id=device_id,
        platform=PlatformKind.LINUX,
        capabilities={Capability.DESKTOP_INPUT},
        tools=[ToolDescriptor(tool_id="device.echo", capability=Capability.DESKTOP_INPUT)],
    )


def test_device_heartbeat_persistence_is_sampled(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    events = ControllerEventLog(store)
    registry = DeviceRegistry(store=store, events=events, heartbeat_persist_interval_seconds=60.0)
    registry.register(_descriptor(), trace_id="trace-dev")

    registry.heartbeat(DeviceHeartbeat(device_id="dev-1", active_sessions=1), trace_id="trace-hb-1")
    registry.heartbeat(DeviceHeartbeat(device_id="dev-1", active_sessions=1), trace_id="trace-hb-2")

    assert [row["event_type"] for row in store.event_rows()].count("device_heartbeat") == 0

    eager = DeviceRegistry(store=store, events=events, heartbeat_persist_interval_seconds=0)
    eager.register(_descriptor("dev-2"), trace_id="trace-dev-2")
    eager.heartbeat(DeviceHeartbeat(device_id="dev-2", active_sessions=1), trace_id="trace-hb-3")
    eager.heartbeat(DeviceHeartbeat(device_id="dev-2", active_sessions=1), trace_id="trace-hb-4")

    assert [row["event_type"] for row in store.event_rows()].count("device_heartbeat") == 2
    store.close()


def test_command_result_ingress_wakes_waiter_before_persistence(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
                device_telemetry_persist_mode="raw",
                command_stage_event_mode="raw",
            )
        )
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )
        task = TaskRequest(
            task_id="task-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="echo",
            required_capabilities={Capability.DESKTOP_INPUT},
        )
        server.store.create_task(task, session_id="sess-1", submitter_id="submitter-1", status=TaskState.ACCEPTED)
        server.store.create_lease(lease_id="lease-1", session_id="sess-1", task_id="task-1", device_id="dev-1")
        command = CommandRequest(
            command_id="cmd-1",
            idempotency_key="idem-1",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
        )
        server.store.claim_command(command)
        server.store.update_command_status_for_request(command, CommandState.RUNNING)
        future: asyncio.Future[CommandResult] = asyncio.get_running_loop().create_future()
        server.command_results.pending_results[
            command_storage_key(session_id="sess-1", lease_id="lease-1", command_id="cmd-1")
        ] = PendingCommandResult.from_request(command, future=future)
        record_called = False

        async def _unexpected_sqlite_lookup(*_args, **_kwargs):
            raise AssertionError("matched command results must use in-memory authority")

        monkeypatch.setattr(
            "northcua.controller.api.dispatch.controller_store_read_profiled",
            _unexpected_sqlite_lookup,
        )

        def _record_result(_result: CommandResult, *, trace_id: str) -> CommandResult:
            nonlocal record_called
            record_called = True
            raise AssertionError("websocket ingress must not persist before waking command waiters")

        server.command_bus.record_result = _record_result  # type: ignore[method-assign]

        payload = CommandResult(
            command_id="cmd-1",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            status=CommandState.SUCCEEDED,
            result={"ok": True},
        )
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.COMMAND_RESULT, payload, trace_id="trace-result"),
        )

        assert future.done()
        assert future.result().result == {"ok": True}
        assert record_called is False
        assert server.store.command_row("cmd-1", session_id="sess-1", lease_id="lease-1")["status"] == "running"
        server.store.close()

    asyncio.run(_run())


def test_blocked_heartbeat_persistence_does_not_delay_next_result_on_same_websocket(
    tmp_path,
    monkeypatch,
) -> None:
    async def _run() -> None:
        monkeypatch.setenv("NORTHCUA_CONTROLLER_DEVICE_HEARTBEAT_FLUSH_INTERVAL_SECONDS", "0")
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
                heartbeat_persist_interval_seconds=0,
            )
        )
        command = CommandRequest(
            command_id="cmd-heartbeat-hol",
            idempotency_key="idem-heartbeat-hol",
            session_id="sess-heartbeat-hol",
            lease_id="lease-heartbeat-hol",
            device_id="dev-1",
            tool_id="device.echo",
        )
        result = CommandResult(
            command_id=command.command_id,
            session_id=command.session_id,
            lease_id=command.lease_id,
            device_id=command.device_id,
            status=CommandState.SUCCEEDED,
            result={"ok": True},
        )
        websocket = _SequenceWebSocket(
            [
                Envelope.wrap(
                    MessageType.DEVICE_HEARTBEAT,
                    DeviceHeartbeat(device_id="dev-1", active_sessions=1),
                    trace_id="trace-heartbeat",
                ).model_dump_json(),
                Envelope.wrap(
                    MessageType.COMMAND_RESULT,
                    result,
                    trace_id="trace-result",
                ).model_dump_json(),
            ]
        )
        server.registry.register(_descriptor(), trace_id="trace-device")
        server.connections.mark_device(websocket, "dev-1")
        task = TaskRequest(
            task_id="task-heartbeat-hol",
            trace_id="trace-task",
            agent_kind="desktop",
            request="echo",
            required_capabilities={Capability.DESKTOP_INPUT},
        )
        server.store.create_task(
            task,
            session_id=command.session_id,
            submitter_id="submitter-1",
            status=TaskState.ACCEPTED,
        )
        server.store.create_lease(
            lease_id=command.lease_id,
            session_id=command.session_id,
            task_id=task.task_id,
            device_id=command.device_id,
        )
        server.store.claim_command(command)
        server.store.update_command_status_for_request(command, CommandState.RUNNING)
        future: asyncio.Future[CommandResult] = asyncio.get_running_loop().create_future()
        server.command_results.pending_results[
            command_storage_key(
                session_id=command.session_id,
                lease_id=command.lease_id,
                command_id=command.command_id,
            )
        ] = PendingCommandResult.from_request(command, future=future)
        heartbeat_store_started = asyncio.Event()
        release_heartbeat_store = asyncio.Event()

        async def _blocked_heartbeat_store(_path, _operation, **_kwargs):
            heartbeat_store_started.set()
            await release_heartbeat_store.wait()
            return 1

        monkeypatch.setattr(
            "northcua.controller.storage.device_heartbeat_batcher.controller_store_call",
            _blocked_heartbeat_store,
        )

        handler = asyncio.create_task(server._handle_connection(websocket))  # type: ignore[arg-type]
        try:
            resolved = await asyncio.wait_for(asyncio.shield(future), timeout=0.25)

            assert resolved.result == {"ok": True}
            await asyncio.wait_for(heartbeat_store_started.wait(), timeout=0.25)
            assert not release_heartbeat_store.is_set()
        finally:
            release_heartbeat_store.set()
            await asyncio.wait_for(handler, timeout=2)
            await server.registry.aclose()
            server.events.close()
            server.store.close()

    asyncio.run(_run())


def test_command_result_waiter_receives_runtime_sidecar_before_persist_strip(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
                device_telemetry_persist_mode="raw",
                command_stage_event_mode="raw",
            )
        )
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )
        command = CommandRequest(
            command_id="cmd-sidecar",
            idempotency_key="idem-sidecar",
            session_id="sess-sidecar",
            lease_id="lease-sidecar",
            device_id="dev-1",
            tool_id="device.echo",
        )
        server.store.claim_command(command)
        server.store.update_command_status_for_request(command, CommandState.RUNNING)
        future: asyncio.Future[CommandResult] = asyncio.get_running_loop().create_future()
        server.command_results.pending_results[
            command_storage_key(session_id=command.session_id, lease_id=command.lease_id, command_id=command.command_id)
        ] = PendingCommandResult.from_request(command, future=future)

        payload = CommandResult(
            command_id=command.command_id,
            session_id=command.session_id,
            lease_id=command.lease_id,
            device_id=command.device_id,
            status=CommandState.SUCCEEDED,
            result={
                "status": "ok",
                RUNTIME_SIDECAR_KEY: {
                    "web_ref_aliases": {
                        "observation_id": "obs-new",
                        "alias_epoch": 7,
                        "aliases": [{"ref": "1", "source_ref": "@e1"}],
                    }
                },
            },
        )
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.COMMAND_RESULT, payload, trace_id="trace-result"),
        )

        assert future.done()
        assert future.result().result == payload.result
        assert isinstance(future.result().result, dict)
        assert RUNTIME_SIDECAR_KEY in future.result().result
        row = server.store.command_row(command.command_id, session_id=command.session_id, lease_id=command.lease_id)
        assert row["status"] == "running"
        server.store.close()

    asyncio.run(_run())


def test_late_async_terminal_result_skips_unowned_artifact_validation(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                blob_dir=str(tmp_path / "blobs"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )
        task = TaskRequest(
            task_id="task-late-artifact",
            trace_id="trace-late-artifact",
            agent_kind="desktop",
            request="late artifact",
        )
        server.store.create_task(
            task,
            session_id="sess-late-artifact",
            submitter_id="submitter-late-artifact",
            status=TaskState.ACCEPTED,
        )
        server.store.create_lease(
            lease_id="lease-late-artifact",
            session_id="sess-late-artifact",
            task_id=task.task_id,
            device_id="dev-1",
        )
        command = CommandRequest(
            command_id="cmd-late-artifact",
            idempotency_key="idem-late-artifact",
            session_id="sess-late-artifact",
            lease_id="lease-late-artifact",
            device_id="dev-1",
            tool_id="device.echo",
        )
        assert server.store.claim_command(command)[0] == "new"
        assert server.store.mark_command_dispatched_for_request(command) is None
        assert (
            server.store.record_command_result(
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
        owned = LocalControllerBlobStore(tmp_path / "blobs").materialize(
            ArtifactRef.from_inline_bytes(
                artifact_id="late-artifact",
                kind=ArtifactKind.SCREENSHOT,
                data=b"late image",
                mime_type="image/png",
            )
        )

        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.COMMAND_RESULT,
                CommandResult(
                    command_id=command.command_id,
                    session_id=command.session_id,
                    lease_id=command.lease_id,
                    device_id=command.device_id,
                    status=CommandState.SUCCEEDED,
                    artifacts=[owned.model_copy(update={"artifact_id": "not-owned-by-command"})],
                ),
                trace_id="trace-late-artifact",
            ),
        )

        row = server.store.command_row(command.command_id, session_id=command.session_id, lease_id=command.lease_id)
        assert row is not None
        assert row["status"] == CommandState.TIMED_OUT.value
        server.events.flush()
        late_events = [row for row in server.store.event_rows() if row["event_type"] == "late_command_result"]
        assert late_events
        assert json.loads(late_events[-1]["payload_json"])["skipped_materialization"] is True
        server.store.close()

    asyncio.run(_run())


def test_device_stage_telemetry_persists_controller_event(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
                device_telemetry_persist_mode="raw",
                command_stage_event_mode="raw",
            )
        )
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )

        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_TELEMETRY,
                DeviceTelemetryEvent(
                    device_id="dev-1",
                    event_type="device_driver_call_finished",
                    entity_type="command",
                    entity_id="cmd-1",
                    command_id="cmd-1",
                    session_id="sess-1",
                    tool_id="click",
                    stage="driver",
                    payload={"elapsed_ms": 123, "Authorization": "Bearer secret"},
                ),
                trace_id="trace-stage",
            ),
        )

        server.events.flush()
        row = next(row for row in server.store.event_rows() if row["event_type"] == "device_driver_call_finished")
        assert row["entity_type"] == "command"
        assert row["entity_id"] == "cmd-1"
        payload = json.loads(row["payload_json"])
        assert payload["source"] == "device"
        assert payload["stage"] == "driver"
        assert payload["elapsed_ms"] == 123
        assert payload["controller_received_timestamp"]
        assert isinstance(payload["controller_ingress_lag_ms"], int)
        assert payload["Authorization"] == "<redacted>"
        server.store.close()

    asyncio.run(_run())


def test_device_readiness_stage_events_cover_handshake_heartbeat_and_capabilities(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
                device_telemetry_persist_mode="raw",
            )
        )
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_HEARTBEAT,
                DeviceHeartbeat(
                    device_id="dev-1",
                    active_sessions=0,
                    runtime_health={"healthy": True},
                    capability_health={Capability.DESKTOP_INPUT: True},
                ),
                trace_id="trace-heartbeat",
            ),
        )

        server.events.flush()
        stage_payloads = [
            json.loads(row["payload_json"])
            for row in server.store.event_rows()
            if row["event_type"] == "device_readiness_stage"
        ]
        stages = {payload["stage"] for payload in stage_payloads}
        assert {"device_hello", "websocket_open", "first_heartbeat", "capability_ready"} <= stages
        hello = next(payload for payload in stage_payloads if payload["stage"] == "device_hello")
        assert isinstance(hello["db_commit_ms"], int)
        heartbeat = next(payload for payload in stage_payloads if payload["stage"] == "first_heartbeat")
        assert heartbeat["capability_health_count"] == 1
        server.store.close()

    asyncio.run(_run())


def test_device_stage_telemetry_tracks_ingress_lag_from_device_timestamp(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
                device_telemetry_persist_mode="raw",
                command_stage_event_mode="raw",
            )
        )
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )

        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_TELEMETRY,
                DeviceTelemetryEvent(
                    device_id="dev-1",
                    event_type="device_websocket_send_finished",
                    entity_type="command",
                    entity_id="cmd-1",
                    command_id="cmd-1",
                    session_id="sess-1",
                    tool_id="type",
                    stage="websocket_send",
                    timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    payload={"device_event_timestamp": "2020-01-01T00:00:00+00:00"},
                ),
                trace_id="trace-stage",
            ),
        )

        server.events.flush()
        row = next(row for row in server.store.event_rows() if row["event_type"] == "device_websocket_send_finished")
        payload = json.loads(row["payload_json"])
        assert payload["controller_ingress_lag_ms"] > 0
        snapshot = server._device_telemetry_ingress_lag.drain_snapshot()
        assert snapshot["overall"]["count"] == 1
        assert snapshot["by_event_type"]["device_websocket_send_finished"]["count"] == 1
        assert snapshot["by_tool_stage"]["type:websocket_send"]["count"] == 1
        server.store.close()

    asyncio.run(_run())


def test_device_stage_telemetry_aggregate_mode_skips_raw_event_rows(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
                device_telemetry_persist_mode="aggregate",
            )
        )
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )

        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(
                MessageType.DEVICE_TELEMETRY,
                DeviceTelemetryEvent(
                    device_id="dev-1",
                    event_type="device_websocket_send_finished",
                    entity_type="command",
                    entity_id="cmd-1",
                    command_id="cmd-1",
                    session_id="sess-1",
                    tool_id="type",
                    stage="websocket_send",
                    timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    payload={"device_event_timestamp": "2020-01-01T00:00:00+00:00"},
                ),
                trace_id="trace-stage",
            ),
        )

        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert "device_websocket_send_finished" not in event_types
        snapshot = server._device_telemetry_ingress_lag.drain_snapshot()
        assert snapshot["overall"]["count"] == 1
        assert snapshot["by_tool_stage"]["type:websocket_send"]["count"] == 1
        server.store.close()

    asyncio.run(_run())


def test_device_telemetry_ingress_lag_tracker_summarizes_by_tool_stage() -> None:
    tracker = _DeviceTelemetryIngressLagTracker()

    for lag_ms in [100, 200, 300]:
        tracker.observe(
            {
                "event_type": "device_driver_call_finished",
                "tool_id": "type",
                "stage": "driver",
                "controller_ingress_lag_ms": lag_ms,
            }
        )

    snapshot = tracker.drain_snapshot()

    assert snapshot["overall"] == {"count": 3, "p50_ms": 200, "p90_ms": 280, "p95_ms": 290, "max_ms": 300}
    assert snapshot["by_event_type"]["device_driver_call_finished"]["p50_ms"] == 200
    assert snapshot["by_tool_stage"]["type:driver"]["p95_ms"] == 290
    assert tracker.drain_snapshot()["overall"]["count"] == 0


def test_device_telemetry_sampling_uses_per_event_entropy() -> None:
    base = {
        "device_id": "dev-1",
        "event_type": "device_ordered_queue_full",
        "device_timestamp": "2026-01-01T00:00:00+00:00",
    }

    first = _should_persist_device_telemetry(
        base,
        trace_id="trace",
        message_id="msg-1",
        mode="sample",
        sample_rate=0.5,
    )
    assert (
        _should_persist_device_telemetry(
            base,
            trace_id="trace",
            message_id="msg-1",
            mode="sample",
            sample_rate=0.5,
        )
        is first
    )

    decisions = {
        _should_persist_device_telemetry(
            {**base, "device_timestamp": f"2026-01-01T00:00:{second:02d}+00:00"},
            trace_id="trace",
            message_id=f"msg-{second}",
            mode="sample",
            sample_rate=0.5,
        )
        for second in range(20)
    }
    assert decisions == {False, True}


def test_command_stage_events_aggregate_instead_of_one_row_per_stage(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    events = ControllerEventLog(
        store,
        command_stage_event_mode="aggregate",
        command_stage_event_sample_rate=0.0,
        event_write_batch_size=1000,
    )

    for index in range(500):
        events.write(
            "command_sent",
            trace_id=f"trace-{index}",
            entity_type="command",
            entity_id=f"cmd-{index}",
            command_id=f"cmd-{index}",
            tool_id="device.echo",
        )

    assert store.event_rows() == []
    events.flush()
    rows = store.event_rows()
    assert [row["event_type"] for row in rows] == ["command_stage_events_aggregated"]
    payload = json.loads(rows[0]["payload_json"])
    assert payload["total"] == 500
    assert payload["counts_by_event_type"] == {"command_sent": 500}
    assert payload["counts_by_tool_id"] == {"device.echo": 500}
    store.close()


def test_all_device_command_phases_follow_command_stage_aggregation_policy(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    events = ControllerEventLog(
        store,
        command_stage_event_mode="aggregate",
        command_stage_event_sample_rate=0.0,
        event_write_batch_size=100,
    )
    stages = [
        "device_command_received",
        "device_command_dequeued",
        "device_command_prepare_started",
        "device_command_prepare_finished",
        "device_driver_call_started",
        "device_driver_call_finished",
        "device_result_json_serialize_started",
        "device_result_json_serialize_finished",
        "device_websocket_send_started",
        "device_websocket_send_finished",
    ]

    for stage in stages:
        events.write(
            stage,
            trace_id="trace-command",
            entity_type="command",
            entity_id="cmd-1",
            command_id="cmd-1",
            tool_id="click",
            source="device",
        )

    assert store.event_rows() == []
    events.flush()
    rows = store.event_rows()
    assert [row["event_type"] for row in rows] == ["command_stage_events_aggregated"]
    payload = json.loads(rows[0]["payload_json"])
    assert payload["total"] == len(stages)
    assert payload["counts_by_event_type"] == {stage: 1 for stage in sorted(stages)}
    assert payload["counts_by_tool_id"] == {"click": len(stages)}
    assert payload["counts_by_source"] == {"device": len(stages)}
    store.close()


def test_command_stage_event_off_drops_hot_events(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    events = ControllerEventLog(
        store,
        command_stage_event_mode="off",
        event_write_batch_size=10,
    )

    for index in range(25):
        events.write(
            "command_sent",
            trace_id=f"trace-{index}",
            entity_type="command",
            entity_id=f"cmd-{index}",
            command_id=f"cmd-{index}",
            tool_id="device.echo",
        )

    events.flush()
    assert store.event_rows() == []
    store.close()


def test_command_stage_sample_mode_does_not_write_aggregate(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    events = ControllerEventLog(
        store,
        command_stage_event_mode="sample",
        command_stage_event_sample_rate=1.0,
        event_write_batch_size=100,
    )

    for index in range(5):
        events.write(
            "command_sent",
            trace_id=f"trace-{index}",
            entity_type="command",
            entity_id=f"cmd-{index}",
            command_id=f"cmd-{index}",
            tool_id="device.echo",
        )

    events.flush()
    event_types = [row["event_type"] for row in store.event_rows()]
    assert event_types == ["command_sent"] * 5
    assert "command_stage_events_aggregated" not in event_types
    store.close()


def test_event_log_batches_non_hot_events_until_flush() -> None:
    class _CountingStore:
        def __init__(self) -> None:
            self.single_appends = 0
            self.batch_appends = 0
            self.rows = []

        def append_event(self, **kwargs: object) -> None:
            self.single_appends += 1
            self.rows.append(kwargs)

        def append_events(self, rows: object) -> None:
            self.batch_appends += 1
            self.rows.extend(list(rows))  # type: ignore[arg-type]

    store = _CountingStore()
    events = ControllerEventLog(
        store,  # type: ignore[arg-type]
        event_write_batch_size=100,
        event_write_flush_interval_seconds=0.0,
    )

    for index in range(10):
        events.write("device_registered", trace_id=f"trace-{index}", entity_type="device", entity_id=f"dev-{index}")

    assert store.single_appends == 0
    assert store.batch_appends == 0
    events.flush()
    assert store.single_appends == 0
    assert store.batch_appends == 1
    assert len(store.rows) == 10


def test_event_log_flush_failure_retains_buffered_events() -> None:
    class _FlakyStore:
        def __init__(self) -> None:
            self.fail = True
            self.rows = []

        def append_event(self, **kwargs: object) -> None:
            self.append_events([kwargs])

        def append_events(self, rows: object) -> None:
            if self.fail:
                raise RuntimeError("temporary sqlite failure")
            self.rows.extend(list(rows))  # type: ignore[arg-type]

    store = _FlakyStore()
    events = ControllerEventLog(
        store,  # type: ignore[arg-type]
        event_write_batch_size=100,
        event_write_flush_interval_seconds=0.0,
    )
    events.write("device_registered", trace_id="trace", entity_type="device", entity_id="dev")

    events.flush()
    assert store.rows == []
    store.fail = False
    events.flush()
    assert len(store.rows) == 1


def test_command_send_to_device_omits_metadata_on_wire_for_template_compatibility(tmp_path) -> None:
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
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )
        task = TaskRequest(
            task_id="task-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="echo",
            required_capabilities={Capability.DESKTOP_INPUT},
        )
        server.store.create_task(task, session_id="sess-1", submitter_id="submitter-1", status=TaskState.ACCEPTED)
        server.store.create_lease(lease_id="lease-1", session_id="sess-1", task_id="task-1", device_id="dev-1")
        command = CommandRequest(
            command_id="cmd-1",
            idempotency_key="idem-1",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
            metadata={"controller_timeout_policy": {"effective_timeout_seconds": 90.0}},
        )
        server.store.claim_command(command)
        send_task = asyncio.create_task(server.command_results.send_to_device("dev-1", command, "trace-1"))

        for _ in range(50):
            if device.sent:
                break
            await asyncio.sleep(0.01)
        sent = parse_envelope(device.sent[-1])
        assert isinstance(sent.payload, CommandRequest)
        assert sent.payload.metadata == {"controller_timeout_policy": {"effective_timeout_seconds": 90.0}}

        future = server.command_results.pending_results[
            command_storage_key(session_id="sess-1", lease_id="lease-1", command_id="cmd-1")
        ]
        future.set_result(
            CommandResult(
                command_id="cmd-1",
                session_id="sess-1",
                lease_id="lease-1",
                device_id="dev-1",
                status=CommandState.SUCCEEDED,
            )
        )
        assert (await send_task).status == CommandState.SUCCEEDED
        server.store.close()

    asyncio.run(_run())


def test_command_send_to_device_waits_for_transient_device_reconnect(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        server.command_results.device_connection_wait_seconds = 0.2
        server.command_results.device_connection_poll_seconds = 0.01
        command = CommandRequest(
            command_id="cmd-reconnect",
            idempotency_key="idem-reconnect",
            session_id="sess-reconnect",
            lease_id="lease-reconnect",
            device_id="dev-1",
            tool_id="device.echo",
        )
        send_task = asyncio.create_task(server.command_results.send_to_device("dev-1", command, "trace-reconnect"))
        await asyncio.sleep(0.03)
        device = _FakeWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )
        for _ in range(50):
            if device.sent:
                break
            await asyncio.sleep(0.01)
        future = server.command_results.pending_results[
            command_storage_key(session_id="sess-reconnect", lease_id="lease-reconnect", command_id="cmd-reconnect")
        ]
        future.set_result(
            CommandResult(
                command_id="cmd-reconnect",
                session_id="sess-reconnect",
                lease_id="lease-reconnect",
                device_id="dev-1",
                status=CommandState.SUCCEEDED,
            )
        )

        assert (await send_task).status == CommandState.SUCCEEDED
        assert device.sent
        server.events.flush()
        event_types = [row["event_type"] for row in server.store.event_rows()]
        assert "command_device_connection_wait_started" in event_types
        assert "command_device_connection_wait_resolved" in event_types
        server.store.close()

    asyncio.run(_run())


def test_command_send_failure_clears_pending_waiter(tmp_path) -> None:
    async def _run() -> None:
        class _FailingCommandWebSocket(_FakeWebSocket):
            async def send(self, raw: str) -> None:
                if parse_envelope(raw).type == MessageType.COMMAND_REQUEST:
                    raise RuntimeError("command send failed")
                await super().send(raw)

        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        device = _FailingCommandWebSocket()
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )
        command = CommandRequest(
            command_id="cmd-send-fails",
            idempotency_key="idem-send-fails",
            session_id="sess-send-fails",
            lease_id="lease-send-fails",
            device_id="dev-1",
            tool_id="device.echo",
        )

        try:
            await server.command_results.send_to_device("dev-1", command, "trace-send-fails")
        except RuntimeError:
            pass
        else:
            raise AssertionError("command websocket send should fail")

        assert server.command_results.pending_results == {}
        server.store.close()

    asyncio.run(_run())


def test_cancel_for_device_async_persists_cancelled_result_json(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        task = TaskRequest(
            task_id="task-device-cancel-result",
            trace_id="trace-device-cancel-result",
            agent_kind="desktop",
            request="echo",
            required_capabilities={Capability.DESKTOP_INPUT},
        )
        server.store.create_task(
            task,
            session_id="sess-device-cancel-result",
            submitter_id="submitter-1",
            status=TaskState.ACCEPTED,
        )
        server.store.create_lease(
            lease_id="lease-device-cancel-result",
            session_id="sess-device-cancel-result",
            task_id=task.task_id,
            device_id="dev-1",
        )
        command = CommandRequest(
            command_id="cmd-device-cancel-result",
            idempotency_key="idem-device-cancel-result",
            session_id="sess-device-cancel-result",
            lease_id="lease-device-cancel-result",
            device_id="dev-1",
            tool_id="device.echo",
        )
        server.store.claim_command(command)

        await server.command_results.cancel_for_device_async("dev-1")

        row = server.store.command_row(
            "cmd-device-cancel-result",
            session_id="sess-device-cancel-result",
            lease_id="lease-device-cancel-result",
        )
        assert row is not None
        assert row["status"] == CommandState.CANCELLED.value
        assert "device disconnected" in str(row["result_json"])
        server.store.close()

    asyncio.run(_run())


def test_cancel_for_device_async_overtakes_normal_writer_backlog(tmp_path) -> None:
    async def _run() -> None:
        server = ControllerServer(
            ControllerConfig(
                db_path=str(tmp_path / "controller.sqlite"),
                security=ControllerSecurityConfig(insecure_dev_no_auth=True),
            )
        )
        task = TaskRequest(
            task_id="task-device-cancel-priority",
            trace_id="trace-device-cancel-priority",
            agent_kind="desktop",
            request="echo",
        )
        server.store.create_task(
            task,
            session_id="sess-device-cancel-priority",
            submitter_id="submitter-1",
            status=TaskState.ACCEPTED,
        )
        server.store.create_lease(
            lease_id="lease-device-cancel-priority",
            session_id="sess-device-cancel-priority",
            task_id=task.task_id,
            device_id="dev-priority",
        )
        server.store.claim_command(
            CommandRequest(
                command_id="cmd-device-cancel-priority",
                idempotency_key="idem-device-cancel-priority",
                session_id="sess-device-cancel-priority",
                lease_id="lease-device-cancel-priority",
                device_id="dev-priority",
                tool_id="device.echo",
            )
        )
        gate = threading.Event()
        started = threading.Event()

        def _block_writer(_store) -> None:
            started.set()
            gate.wait(timeout=2)

        blocker = asyncio.create_task(
            controller_store_call_profiled(
                server.store.path,
                _block_writer,
                operation_name="disconnect-priority-blocker",
            )
        )
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.005)
        assert started.is_set()
        normal = [
            asyncio.create_task(
                controller_store_call_profiled(
                    server.store.path,
                    lambda _store: time.sleep(0.01),
                    operation_name=f"normal-backlog-{index}",
                )
            )
            for index in range(160)
        ]
        await asyncio.sleep(0.02)
        cancel = asyncio.create_task(server.command_results.cancel_for_device_async("dev-priority"))
        gate.set()
        try:
            await asyncio.wait_for(cancel, timeout=0.5)
            assert any(not pending.done() for pending in normal)
            row = server.store.command_row(
                "cmd-device-cancel-priority",
                session_id="sess-device-cancel-priority",
                lease_id="lease-device-cancel-priority",
            )
            assert row is not None
            assert row["status"] == CommandState.CANCELLED.value
            await asyncio.gather(blocker, *normal)
        finally:
            gate.set()
            await asyncio.gather(blocker, *normal, return_exceptions=True)
            server.store.close()

    asyncio.run(_run())


def test_out_of_band_command_result_emits_receive_and_persist_telemetry(tmp_path) -> None:
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
            Envelope.wrap(MessageType.DEVICE_HELLO, DeviceHello(descriptor=_descriptor()), trace_id="trace-dev"),
        )
        task = TaskRequest(
            task_id="task-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="echo",
            required_capabilities={Capability.DESKTOP_INPUT},
        )
        server.store.create_task(task, session_id="sess-1", submitter_id="submitter-1", status=TaskState.ACCEPTED)
        server.store.create_lease(lease_id="lease-1", session_id="sess-1", task_id="task-1", device_id="dev-1")
        command = CommandRequest(
            command_id="cmd-1",
            idempotency_key="idem-1",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            tool_id="device.echo",
        )
        server.store.claim_command(command)
        server.store.update_command_status_for_request(command, CommandState.RUNNING)

        payload = CommandResult(
            command_id="cmd-1",
            session_id="sess-1",
            lease_id="lease-1",
            device_id="dev-1",
            status=CommandState.SUCCEEDED,
            result={"ok": True},
        )
        await server._dispatch(
            device,  # type: ignore[arg-type]
            Envelope.wrap(MessageType.COMMAND_RESULT, payload, trace_id="trace-result"),
        )

        row = server.store.command_row("cmd-1", session_id="sess-1", lease_id="lease-1")
        assert row["status"] == "succeeded"
        server.events.flush()
        event_rows = server.store.event_rows()
        event_types = [row["event_type"] for row in event_rows]
        assert event_types.index("command_result_received") < event_types.index("command_result_persist_done")
        received = next(row for row in event_rows if row["event_type"] == "command_result_received")
        received_payload = json.loads(received["payload_json"])
        assert received_payload["out_of_band"] is True
        assert received_payload["status"] == "succeeded"
        persisted = next(row for row in event_rows if row["event_type"] == "command_result_persist_done")
        persisted_payload = json.loads(persisted["payload_json"])
        assert persisted_payload["status"] == "succeeded"
        assert {
            "store_queue_ms",
            "store_admission_ms",
            "store_executor_queue_ms",
            "store_queue_depth",
            "store_admitted_queue_depth",
        } <= set(persisted_payload)
        server.store.close()

    asyncio.run(_run())
