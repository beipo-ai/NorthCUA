"""Controller telemetry persistence and websocket monitor tests."""

from __future__ import annotations

import asyncio
import json
import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from northcua.controller.api.command_waits import PendingCommandResultRegistry
from northcua.controller.api.dispatch import ControllerEnvelopeDispatcher
from northcua.controller.api.server import ControllerServer, _tcp_socket_queue_metrics
from northcua.controller.auth import ControllerAuth
from northcua.controller.config import ControllerConfig
from northcua.controller.control_plane.connections import ConnectionRegistry
from northcua.controller.observability import ControllerTelemetrySink
from northcua.controller.observability.event_log import ControllerEventLog
from northcua.controller.storage import SQLiteControllerStore
from northcua.telemetry import (
    CasePhase,
    CaseStatus,
    CaseTelemetrySnapshot,
    PhaseTiming,
    TelemetryEntityKind,
    TelemetryEvent,
    TelemetryEventType,
)
from northcua_protocol.enums import MessageType
from northcua_protocol.envelope import Envelope, parse_envelope
from northcua_protocol.payloads import CommandRequest, CommandResult, TelemetrySubscribeRequest, TelemetryUpdate
from northcua_protocol.state import CommandState


def test_controller_telemetry_sink_persists_events_and_snapshots(tmp_path: Path) -> None:
    db_path = tmp_path / "controller.sqlite"
    sink = ControllerTelemetrySink(db_path)
    sink.append_event(
        TelemetryEvent(
            experiment_id="experiment",
            batch_id="batch",
            case_id="case-1",
            entity_kind=TelemetryEntityKind.CASE,
            entity_id="case-1",
            event_type=TelemetryEventType.STATUS_CHANGED,
            phase=CasePhase.MAIN_AGENT,
            status=CaseStatus.RUNNING_AGENT,
        )
    )
    sink.upsert_snapshot(
        CaseTelemetrySnapshot(
            experiment_id="experiment",
            batch_id="batch",
            case_id="case-1",
            status=CaseStatus.RUNNING_AGENT,
            phase=CasePhase.MAIN_AGENT,
        )
    )
    sink.flush()

    store = SQLiteControllerStore(db_path)
    try:
        assert len(store.telemetry_event_rows(experiment_id="experiment")) == 1
        assert len(store.case_snapshot_rows(experiment_id="experiment")) == 1
    finally:
        store.close()


def test_controller_telemetry_sink_sync_fallback_keeps_short_busy_timeout(tmp_path: Path) -> None:
    db_path = tmp_path / "controller.sqlite"
    sink = ControllerTelemetrySink(db_path, async_writes=False)
    try:
        timeout_ms = sink._store.connection.execute("PRAGMA busy_timeout").fetchone()[0]  # noqa: SLF001
    finally:
        sink.flush()

    assert timeout_ms == 1000


def test_controller_telemetry_sink_async_writer_uses_long_busy_timeout(tmp_path: Path) -> None:
    sink = ControllerTelemetrySink(tmp_path / "controller.sqlite")
    try:
        assert sink._busy_timeout_ms == 30000  # noqa: SLF001
    finally:
        sink.flush()


def test_controller_telemetry_sink_busy_timeout_can_be_overridden(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_CONTROLLER_TELEMETRY_BUSY_TIMEOUT_MS", "250")
    sink = ControllerTelemetrySink(tmp_path / "controller.sqlite", async_writes=False)
    try:
        timeout_ms = sink._store.connection.execute("PRAGMA busy_timeout").fetchone()[0]  # noqa: SLF001
    finally:
        sink.flush()

    assert timeout_ms == 250


def test_controller_telemetry_sink_does_not_block_callers_while_sqlite_is_locked(tmp_path: Path) -> None:
    db_path = tmp_path / "controller.sqlite"
    SQLiteControllerStore(db_path).close()
    blocker = sqlite3.connect(db_path)
    blocker.execute("BEGIN IMMEDIATE")
    sink = ControllerTelemetrySink(
        db_path,
        busy_timeout_ms=2_000,
        batch_interval_seconds=0.01,
        batch_size=16,
    )
    try:
        start = time.perf_counter()
        for index in range(100):
            sink.append_event(_telemetry_event(f"case-{index}"))
        elapsed = time.perf_counter() - start

        assert elapsed < 0.25
        assert sink.stats()["closed"] is False
    finally:
        blocker.rollback()
        blocker.close()
        sink.flush()

    store = SQLiteControllerStore(db_path)
    try:
        assert len(store.telemetry_event_rows(experiment_id="experiment", limit=200)) == 100
    finally:
        store.close()


def test_controller_telemetry_sink_handles_concurrent_producers(tmp_path: Path) -> None:
    db_path = tmp_path / "controller.sqlite"
    sink = ControllerTelemetrySink(db_path, batch_interval_seconds=0.01, batch_size=32)

    def write_batch(worker_index: int) -> None:
        for index in range(50):
            case_id = f"case-{worker_index}-{index}"
            sink.append_event(_telemetry_event(case_id))
            sink.upsert_snapshot(
                CaseTelemetrySnapshot(
                    experiment_id="experiment",
                    batch_id="batch",
                    case_id=case_id,
                    status=CaseStatus.RUNNING_AGENT,
                    phase=CasePhase.MAIN_AGENT,
                )
            )
            sink.record_phase(
                PhaseTiming(
                    experiment_id="experiment",
                    batch_id="batch",
                    case_id=case_id,
                    phase=CasePhase.MAIN_AGENT,
                    status=CaseStatus.RUNNING_AGENT,
                    started_at="2026-05-19T00:00:00+00:00",
                    finished_at="2026-05-19T00:00:01+00:00",
                    duration_seconds=1.0,
                )
            )

    threads = [threading.Thread(target=write_batch, args=(worker_index,)) for worker_index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    sink.flush()

    store = SQLiteControllerStore(db_path)
    try:
        assert len(store.telemetry_event_rows(experiment_id="experiment", limit=500)) == 400
        assert len(store.case_snapshot_rows(experiment_id="experiment")) == 400
        assert len(store.case_phase_timing_rows(experiment_id="experiment")) == 400
    finally:
        store.close()


def test_controller_telemetry_sink_drops_when_async_queue_is_full(tmp_path: Path) -> None:
    db_path = tmp_path / "controller.sqlite"
    SQLiteControllerStore(db_path).close()
    blocker = sqlite3.connect(db_path)
    blocker.execute("BEGIN IMMEDIATE")
    sink = ControllerTelemetrySink(
        db_path,
        busy_timeout_ms=2_000,
        batch_interval_seconds=1.0,
        batch_size=128,
        queue_max_size=1,
    )
    try:
        start = time.perf_counter()
        for index in range(200):
            sink.append_event(_telemetry_event(f"case-{index}"))
        elapsed = time.perf_counter() - start

        assert elapsed < 0.25
        assert sink.stats()["dropped_writes"] > 0
    finally:
        blocker.rollback()
        blocker.close()
        sink.flush()


def test_controller_telemetry_sink_flush_can_retry_after_shutdown_enqueue_timeout(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "controller.sqlite"
    sink = ControllerTelemetrySink(db_path, shutdown_timeout_seconds=0.0)
    telemetry_queue = sink._queue  # noqa: SLF001
    assert telemetry_queue is not None
    original_put = telemetry_queue.put
    calls = 0

    def flaky_put(item, timeout=None) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise queue.Full
        original_put(item, timeout=timeout)

    monkeypatch.setattr(telemetry_queue, "put", flaky_put)

    sink.flush()
    assert calls == 1
    assert sink.stats()["closed"] is True
    assert sink._stop_enqueued is False  # noqa: SLF001

    sink.flush()
    assert calls == 2
    assert sink._stop_enqueued is True  # noqa: SLF001
    stop_done = sink._stop_done  # noqa: SLF001
    assert stop_done is not None
    assert stop_done.wait(timeout=1.0)


def test_controller_snapshots_are_scoped_by_experiment(tmp_path: Path) -> None:
    db_path = tmp_path / "controller.sqlite"
    sink = ControllerTelemetrySink(db_path)
    for experiment_id in ("experiment-a", "experiment-b"):
        sink.upsert_snapshot(
            CaseTelemetrySnapshot(
                experiment_id=experiment_id,
                batch_id="shared-batch",
                case_id="case-1",
                status=CaseStatus.SUCCEEDED,
                phase=CasePhase.COMPLETED,
            )
        )
    sink.flush()

    store = SQLiteControllerStore(db_path)
    try:
        assert len(store.case_snapshot_rows(batch_id="shared-batch")) == 2
        assert len(store.case_snapshot_rows(experiment_id="experiment-a")) == 1
    finally:
        store.close()


def _telemetry_event(case_id: str) -> TelemetryEvent:
    return TelemetryEvent(
        experiment_id="experiment",
        batch_id="batch",
        case_id=case_id,
        entity_kind=TelemetryEntityKind.CASE,
        entity_id=case_id,
        event_type=TelemetryEventType.STATUS_CHANGED,
        phase=CasePhase.MAIN_AGENT,
        status=CaseStatus.RUNNING_AGENT,
    )


def test_connection_registry_snapshot_counts_roles() -> None:
    registry = ConnectionRegistry()
    device = _RecordingWebsocket()
    submitter = _RecordingWebsocket()
    hybrid = _RecordingWebsocket()

    registry.mark_device(device, "dev-1")
    registry.mark_submitter(submitter, "sub-1")
    registry.mark_device(hybrid, "dev-2")
    registry.mark_submitter(hybrid, "sub-2")

    snapshot = registry.snapshot()

    assert snapshot["active_connections"] == 3
    assert snapshot["device_connections"] == 2
    assert snapshot["submitter_connections"] == 2
    assert snapshot["role_counts"] == {
        "unknown": 0,
        "device": 1,
        "submitter": 1,
        "device_submitter": 1,
    }


def test_tcp_socket_queue_metrics_parses_ss_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "northcua.controller.api.server.subprocess.run",
        lambda *_, **__: SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    "ESTAB 0 12 127.0.0.1:9192 127.0.0.1:50000",
                    "ESTAB 4 0 127.0.0.1:9192 127.0.0.1:50001",
                    "LISTEN 0 512 127.0.0.1:9192 *:*",
                ]
            ),
            stderr="",
        ),
    )

    metrics = _tcp_socket_queue_metrics(9192)

    assert metrics["socket_metrics_available"] is True
    assert metrics["socket_count"] == 3
    assert metrics["recv_q_max"] == 4
    assert metrics["send_q_max"] == 512
    assert metrics["socket_state_counts"] == {"ESTAB": 2, "LISTEN": 1}


@pytest.mark.asyncio
async def test_command_result_wait_emits_websocket_send_timing(tmp_path: Path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    events = ControllerEventLog(store)
    connections = ConnectionRegistry()
    pending = PendingCommandResultRegistry(store=store, connections=connections, events=events)
    websocket = _RecordingWebsocket()
    connections.mark_device(websocket, "dev-1")
    command = CommandRequest(
        command_id="cmd-1",
        idempotency_key="idem-1",
        session_id="sess-1",
        lease_id="lease-1",
        device_id="dev-1",
        tool_id="device.echo",
    )

    task = asyncio.create_task(pending.send_to_device("dev-1", command, "trace-1"))
    for _ in range(20):
        if websocket.sent:
            break
        await asyncio.sleep(0.01)
    assert websocket.sent
    result = CommandResult(
        command_id="cmd-1",
        session_id="sess-1",
        lease_id="lease-1",
        device_id="dev-1",
        status=CommandState.SUCCEEDED,
    )
    waiter = pending.pop_future_for_result(result)
    assert waiter is not None
    waiter.set_result(result)

    assert (await task).status == CommandState.SUCCEEDED
    payloads = {
        row["event_type"]: json.loads(row["payload_json"])
        for row in store.event_rows()
        if row["event_type"] in {"command_device_send_done", "command_device_result_wait_resolved"}
    }
    assert payloads["command_device_send_done"]["connection_id"]
    assert payloads["command_device_send_done"]["pending_command_waiters"] == 1
    assert payloads["command_device_send_done"]["envelope_bytes"] > 0
    assert payloads["command_device_result_wait_resolved"]["result_wait_ms"] >= 0
    store.close()


@pytest.mark.asyncio
async def test_controller_dispatcher_returns_telemetry_update(tmp_path: Path) -> None:
    server = ControllerServer(
        ControllerConfig(
            db_path=str(tmp_path / "controller.sqlite"),
            blob_dir=str(tmp_path / "blobs"),
            security={"insecure_dev_no_auth": True},
        )
    )
    sink = ControllerTelemetrySink(server.store.path)
    sink.upsert_snapshot(
        CaseTelemetrySnapshot(
            experiment_id="experiment",
            batch_id="batch",
            case_id="case-1",
            status=CaseStatus.SUCCEEDED,
            phase=CasePhase.COMPLETED,
        )
    )
    sink.flush()
    websocket = _RecordingWebsocket()
    dispatcher = ControllerEnvelopeDispatcher(
        auth=ControllerAuth(server.config.security),
        connections=server.connections,
        registry=server.registry,
        command_bus=server.command_bus,
        task_service=server.task_service,
        store=server.store,
        events=server.events,
        command_results=server.command_results,
    )

    await dispatcher.dispatch(
        websocket,
        Envelope.wrap(
            MessageType.TELEMETRY_SUBSCRIBE,
            TelemetrySubscribeRequest(experiment_id="experiment"),
            trace_id=uuid.uuid4().hex,
        ),
    )

    envelope = parse_envelope(websocket.sent[0])
    assert envelope.type == MessageType.TELEMETRY_UPDATE
    assert isinstance(envelope.payload, TelemetryUpdate)
    assert envelope.payload.summary["succeeded"] == 1
    assert envelope.payload.case_snapshots[0]["case_id"] == "case-1"
    server.store.close()


@pytest.mark.asyncio
async def test_controller_telemetry_read_does_not_block_event_loop(tmp_path: Path, monkeypatch) -> None:
    server = ControllerServer(
        ControllerConfig(
            db_path=str(tmp_path / "controller.sqlite"),
            blob_dir=str(tmp_path / "blobs"),
            security={"insecure_dev_no_auth": True},
        )
    )
    from northcua.controller.observability import telemetry_reader

    original = telemetry_reader._update_from_db  # noqa: SLF001

    def _slow_update(db_path, request):
        time.sleep(0.05)
        return original(db_path, request)

    monkeypatch.setattr(telemetry_reader, "_update_from_db", _slow_update)
    update = asyncio.create_task(
        server._dispatcher.telemetry_reader.update_async(TelemetrySubscribeRequest(experiment_id="experiment"))
    )
    await asyncio.sleep(0.01)

    assert not update.done()
    assert (await update).summary["total_cases"] == 0
    server.store.close()


@pytest.mark.asyncio
async def test_controller_telemetry_summary_does_not_require_snapshot_payload(tmp_path: Path) -> None:
    server = ControllerServer(
        ControllerConfig(
            db_path=str(tmp_path / "controller.sqlite"),
            blob_dir=str(tmp_path / "blobs"),
            security={"insecure_dev_no_auth": True},
        )
    )
    sink = ControllerTelemetrySink(server.store.path)
    sink.upsert_snapshot(
        CaseTelemetrySnapshot(
            experiment_id="experiment",
            batch_id="batch",
            case_id="case-1",
            status=CaseStatus.SUCCEEDED,
            phase=CasePhase.COMPLETED,
        )
    )
    sink.flush()
    update = server._dispatcher.telemetry_reader.update(
        TelemetrySubscribeRequest(experiment_id="experiment", include_snapshot=False)
    )

    assert update.case_snapshots == []
    assert update.summary["succeeded"] == 1
    server.store.close()


class _RecordingWebsocket:
    """Minimal websocket double for dispatcher tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        """Record one serialized envelope."""

        self.sent.append(raw)
