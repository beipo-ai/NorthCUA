"""Batch event writer diagnostics tests."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import cast

from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState
from northcua.batch.runner import events as batch_events
from northcua.batch.runner.events import BatchAttemptEventMessage, BatchEventWriter
from northcua.batch.state import BatchTaskLease, SQLiteBatchStateRepository
from northcua.telemetry.background_jsonl import BackgroundJsonlWriter
from northcua.telemetry.contracts import CasePhase, CaseStatus, FailureCategory, TelemetryEntityKind
from northcua.telemetry.sinks import LocalTelemetrySink


def test_attempt_event_writes_jsonl_without_telemetry_sink(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="retry")
    manifest = BatchManifest(batch_id="batch-1", tasks=[task])
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)

    events.attempt_event(
        case_id=task.task_id,
        message=BatchAttemptEventMessage.CONTROLLER_SUBMIT_FAILED_BEFORE_TERMINAL,
        phase=CasePhase.CONTROLLER_SUBMIT,
        status=CaseStatus.FAILED,
        attempt=3,
        entity_kind=TelemetryEntityKind.TASK,
        entity_id="controller-1",
        details={"failure_category": FailureCategory.CONTROLLER_SUBMIT, "elapsed_seconds": 1.25},
        controller_task_id="controller-1",
        trace_id="trace-1",
        sandbox_id="sandbox-1",
        device_id="device-1",
    )

    rows = [
        json.loads(line) for line in (tmp_path / "batch_attempt_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0] == {
        "timestamp": rows[0]["timestamp"],
        "batch_id": "batch-1",
        "case_id": "task-1",
        "attempt": 3,
        "message": "controller_submit_failed_before_terminal",
        "phase": "controller_submit",
        "status": "failed",
        "entity_kind": "task",
        "entity_id": "controller-1",
        "controller_task_id": "controller-1",
        "trace_id": "trace-1",
        "sandbox_id": "sandbox-1",
        "device_id": "device-1",
        "details": rows[0]["details"],
    }
    assert rows[0]["details"]["failure_category"] == "controller_submit"
    assert rows[0]["details"]["elapsed_seconds"] == 1.25
    assert rows[0]["details"]["concurrency_snapshot"]["task_status_counts"] == {"pending": 1}
    assert "resource_snapshot" in rows[0]["details"]


def test_event_writer_flushes_owned_background_jsonl_writer(tmp_path) -> None:
    manifest = BatchManifest(
        batch_id="batch-1",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="retry")],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    jsonl_writer = BackgroundJsonlWriter(queue_max_size=8, shutdown_timeout_seconds=1.0)
    events = BatchEventWriter(
        repository=repository,
        output_dir=tmp_path,
        batch_id=manifest.batch_id,
        jsonl_writer=jsonl_writer,
        owns_jsonl_writer=True,
    )

    events.attempt_event(
        case_id="task-1",
        message=BatchAttemptEventMessage.ATTEMPT_STARTED,
        phase=CasePhase.MAIN_AGENT,
        status=CaseStatus.RUNNING_AGENT,
    )
    events.prepared_event(
        case_id="task-1",
        message="prepared_sandbox_ready",
        status=CaseStatus.PROVISIONING,
        attempt=1,
    )
    events.flush()

    assert jsonl_writer.stats().closed is True
    assert json.loads((tmp_path / "batch_attempt_events.jsonl").read_text(encoding="utf-8"))["message"] == (
        "attempt_started"
    )
    assert json.loads((tmp_path / "prepared_sandbox_events.jsonl").read_text(encoding="utf-8"))["message"] == (
        "prepared_sandbox_ready"
    )


def test_event_writer_drops_jsonl_when_background_writer_rejects_without_sync_fallback(tmp_path) -> None:
    class RejectingWriter:
        def append_json(self, path, payload):
            _ = (path, payload)
            return False

        def append_text(self, path, text):
            _ = (path, text)
            return False

        def flush(self):
            return None

    manifest = BatchManifest(
        batch_id="batch-1",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="retry")],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(
        repository=repository,
        output_dir=tmp_path,
        batch_id=manifest.batch_id,
        jsonl_writer=cast(BackgroundJsonlWriter, RejectingWriter()),
    )

    events.attempt_event(
        case_id="task-1",
        message=BatchAttemptEventMessage.ATTEMPT_STARTED,
        phase=CasePhase.MAIN_AGENT,
        status=CaseStatus.RUNNING_AGENT,
    )

    assert not (tmp_path / "batch_attempt_events.jsonl").exists()


def test_for_lease_reuses_telemetry_recorder_without_rescanning_repository(tmp_path) -> None:
    class CountingRepository:
        path = None

        def __init__(self) -> None:
            self.all_calls = 0

        def all(self):
            self.all_calls += 1
            return [BatchTaskState(task_id="task-1")]

    repository = CountingRepository()
    events = BatchEventWriter(
        repository=repository,
        output_dir=tmp_path,
        batch_id="batch-1",
        telemetry_sink=LocalTelemetrySink(tmp_path),
    )
    assert repository.all_calls == 1

    lease = BatchTaskLease(
        task_id="task-1",
        lease_owner="runner-1",
        lease_token="token-1",
        generation=1,
        state=BatchTaskState(task_id="task-1"),
    )
    scoped = events.for_lease(lease)

    assert repository.all_calls == 1
    assert scoped.telemetry is events.telemetry


def test_attempt_event_does_not_scan_repository_when_sqlite_snapshot_fails(tmp_path, monkeypatch) -> None:
    class PathBackedRepository:
        path = tmp_path / "batch_state.sqlite"

        def all(self):
            raise AssertionError("attempt event should not scan repository rows after SQLite snapshot failure")

    def fail_snapshot(path):
        _ = path
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(batch_events, "_sqlite_concurrency_snapshot", fail_snapshot)
    events = BatchEventWriter(repository=PathBackedRepository(), output_dir=tmp_path, batch_id="batch-1")

    events.attempt_event(
        case_id="task-1",
        message=BatchAttemptEventMessage.ATTEMPT_STARTED,
        phase=CasePhase.MAIN_AGENT,
        status=CaseStatus.RUNNING_AGENT,
    )

    row = json.loads((tmp_path / "batch_attempt_events.jsonl").read_text(encoding="utf-8").strip())
    assert row["details"]["concurrency_snapshot"] == {}


def test_prepared_event_writes_jsonl_without_telemetry_sink(tmp_path) -> None:
    manifest = BatchManifest(
        batch_id="batch-1", tasks=[BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="x")]
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)

    events.prepared_event(
        case_id="task-1",
        message="prepared_sandbox_ready",
        status=CaseStatus.PROVISIONING,
        attempt=1,
        details={"prepare_seconds": 12.5},
        sandbox_id="sandbox-1",
        device_id="device-1",
    )

    row = json.loads((tmp_path / "prepared_sandbox_events.jsonl").read_text(encoding="utf-8").strip())
    assert row["message"] == "prepared_sandbox_ready"
    assert row["details"] == {"prepare_seconds": 12.5}


def test_slow_event_write_warning_is_throttled_across_lease_writers(
    tmp_path,
    caplog,
    monkeypatch,
) -> None:
    manifest = BatchManifest(
        batch_id="batch-1", tasks=[BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="x")]
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    lease = BatchTaskLease(
        task_id="task-1",
        lease_owner="runner-1",
        lease_token="token-1",
        generation=1,
        state=BatchTaskState(task_id="task-1"),
    )
    monkeypatch.setattr(batch_events, "_SLOW_WRITE_SECONDS", 0.0)
    monkeypatch.setattr(batch_events, "_SLOW_WRITE_LOG_INTERVAL_SECONDS", 60.0)
    caplog.set_level(logging.WARNING, logger=batch_events.__name__)

    events.attempt_event(
        case_id="task-1",
        message=BatchAttemptEventMessage.ATTEMPT_STARTED,
        phase=CasePhase.MAIN_AGENT,
        status=CaseStatus.RUNNING_AGENT,
    )
    events.for_lease(lease).attempt_event(
        case_id="task-1",
        message=BatchAttemptEventMessage.ATTEMPT_CLEANUP_FINISHED,
        phase=CasePhase.CLEANUP,
        status=CaseStatus.FINALIZING,
    )

    slow_records = [record for record in caplog.records if record.message == "batch event writer slow sync operation"]
    assert len(slow_records) == 1
    assert getattr(slow_records[0], "operation", "") == "attempt_event"
    assert getattr(slow_records[0], "case_id", "") == "task-1"
