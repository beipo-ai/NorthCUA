"""Regression tests for controller SQLite hot paths."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import Future

from northcua.controller.api.server import _ControllerOperationTimingTracker
from northcua.controller.execution.task_status import TaskStatusReader
from northcua.controller.observability import event_log as event_log_module
from northcua.controller.observability.event_log import ControllerEventLog
from northcua.controller.storage.async_store import ControllerStorePriority
from northcua.controller.storage.events import ControllerEventRow
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua.controller.storage.queue import TaskQueue
from northcua_protocol.capabilities import Capability
from northcua_protocol.payloads import CommandRequest, CommandResult, TaskFinished, TaskRequest, TaskStatusFilter
from northcua_protocol.state import CommandState, SessionState, TaskState


def _store(tmp_path) -> SQLiteControllerStore:
    return SQLiteControllerStore(tmp_path / "controller.sqlite")


def _plan_details(store: SQLiteControllerStore, sql: str, params: tuple[object, ...]) -> list[str]:
    return [str(row[3]) for row in store.connection.execute("EXPLAIN QUERY PLAN " + sql, params)]


def _insert_command(
    store: SQLiteControllerStore,
    *,
    command_id: str,
    device_id: str,
    session_id: str,
    status: CommandState,
) -> None:
    request = {
        "command_id": command_id,
        "idempotency_key": f"idem-{command_id}",
        "session_id": session_id,
        "lease_id": "lease-1",
        "device_id": device_id,
        "tool_id": "device.echo",
    }
    store.connection.execute(
        """
        INSERT INTO commands(
            command_id, protocol_command_id, idempotency_key, session_id, lease_id, device_id,
            status, request_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            command_id,
            command_id,
            f"idem-{command_id}",
            session_id,
            "lease-1",
            device_id,
            status.value,
            json.dumps(request),
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    store.connection.commit()


def _task_request(task_id: str) -> TaskRequest:
    return TaskRequest(
        task_id=task_id,
        trace_id=f"trace-{task_id}",
        agent_kind="desktop",
        request="run",
        required_capabilities={Capability.DESKTOP_INPUT},
    )


def test_controller_store_indexes_disconnect_cleanup_queries(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        indexes = {
            row["name"]
            for table in ("commands", "leases", "sessions", "tasks")
            for row in store.connection.execute(f"PRAGMA index_list({table})")
        }

        assert "commands_device_status_idx" in indexes
        assert "commands_session_status_idx" in indexes
        assert "leases_session_generation_idx" in indexes
        assert "sessions_device_status_idx" in indexes
        assert "sessions_task_status_idx" in indexes
        assert "sessions_status_task_idx" in indexes
        assert "tasks_created_idx" in indexes
        assert "tasks_created_task_idx" in indexes

        assert not any(
            detail == "SCAN commands"
            for detail in _plan_details(
                store,
                "SELECT * FROM commands WHERE device_id=? AND status IN (?, ?, ?)",
                ("dev-1", "queued", "sent", "running"),
            )
        )
        assert not any(
            detail == "SCAN commands"
            for detail in _plan_details(
                store,
                "UPDATE commands SET status=?, updated_at=? WHERE device_id=? AND status IN (?, ?, ?)",
                ("cancelled", "now", "dev-1", "queued", "sent", "running"),
            )
        )
        assert not any(
            detail == "SCAN commands"
            for detail in _plan_details(
                store,
                "SELECT * FROM commands WHERE session_id=? AND status IN (?, ?, ?)",
                ("sess-1", "queued", "sent", "running"),
            )
        )
        assert not any(
            detail == "SCAN sessions"
            for detail in _plan_details(
                store,
                "SELECT session_id, task_id FROM sessions WHERE device_id=? AND status IN (?, ?)",
                ("dev-1", "leased", "running"),
            )
        )
        assert not any(
            detail == "SCAN sessions"
            for detail in _plan_details(
                store,
                """
                SELECT tasks.*, sessions.session_id
                FROM tasks
                JOIN sessions ON sessions.task_id=tasks.task_id
                WHERE tasks.status=? AND sessions.status=?
                ORDER BY tasks.created_at ASC
                """,
                ("accepted", "pending"),
            )
        )
        assert not any(
            detail == "SCAN leases"
            for detail in _plan_details(
                store,
                "SELECT MAX(generation) AS generation FROM leases WHERE session_id=?",
                ("sess-1",),
            )
        )
        assert any(
            "USING INDEX tasks_created_idx" in detail
            for detail in _plan_details(store, "SELECT * FROM tasks ORDER BY created_at DESC", ())
        )
        assert any(
            "USING INDEX tasks_created_task_idx" in detail
            for detail in _plan_details(
                store, "SELECT * FROM tasks ORDER BY created_at DESC, task_id ASC LIMIT 100", ()
            )
        )
    finally:
        store.close()


def test_create_lease_records_storage_operation_timings(tmp_path) -> None:
    store = _store(tmp_path)
    observed: list[tuple[str, float]] = []
    try:
        request = _task_request("task-lease-timing")
        store.create_task(request, session_id="sess-lease-timing", submitter_id="submitter")

        generation = store.create_lease(
            lease_id="lease-timing",
            session_id="sess-lease-timing",
            task_id=request.task_id,
            device_id="dev-timing",
            timing_observer=lambda operation, elapsed_ms: observed.append((operation, elapsed_ms)),
        )

        assert generation == 1
        operations = {operation for operation, _ in observed}
        assert {
            "storage.lease_generation_query",
            "storage.lease_insert",
            "storage.lease_update_session",
            "storage.lease_update_task",
            "storage.lease_commit",
        }.issubset(operations)
        assert all(elapsed_ms >= 0 for _, elapsed_ms in observed)
    finally:
        store.close()


def test_mark_running_async_does_not_reopen_terminal_task(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        try:
            events = ControllerEventLog(store)
            queue = TaskQueue(store=store, events=events)
            request = _task_request("task-mark-running-cas")
            store.create_task(request, session_id="sess-mark-running-cas", submitter_id="submitter")
            store.create_lease(
                lease_id="lease-mark-running-cas",
                session_id="sess-mark-running-cas",
                task_id=request.task_id,
                device_id="dev-mark-running-cas",
            )

            finished = TaskFinished(
                task_id=request.task_id,
                trace_id=request.trace_id,
                session_id="sess-mark-running-cas",
                device_id="dev-mark-running-cas",
                lease_id="lease-mark-running-cas",
                state=TaskState.INTERRUPTED,
                error="disconnect won",
            )
            assert await queue.finish_async(finished)
            await queue.mark_running_async(
                task_id=request.task_id,
                trace_id=request.trace_id,
                session_id="sess-mark-running-cas",
                device_id="dev-mark-running-cas",
            )

            row = store.task_row(request.task_id)
            assert row is not None
            assert row["status"] == TaskState.INTERRUPTED.value
            assert "disconnect won" in str(row["result_json"])
        finally:
            store.close()

    asyncio.run(_run())


def test_mark_running_async_requires_leased_session(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        try:
            events = ControllerEventLog(store)
            queue = TaskQueue(store=store, events=events)
            request = _task_request("task-session-interrupted-cas")
            store.create_task(request, session_id="sess-session-interrupted-cas", submitter_id="submitter")
            store.create_lease(
                lease_id="lease-session-interrupted-cas",
                session_id="sess-session-interrupted-cas",
                task_id=request.task_id,
                device_id="dev-session-interrupted-cas",
            )
            store.update_session_status("sess-session-interrupted-cas", SessionState.INTERRUPTED)

            await queue.mark_running_async(
                task_id=request.task_id,
                trace_id=request.trace_id,
                session_id="sess-session-interrupted-cas",
                device_id="dev-session-interrupted-cas",
            )

            row = store.task_row(request.task_id)
            session = store.session_row("sess-session-interrupted-cas")
            assert row is not None
            assert session is not None
            assert row["status"] == TaskState.SCHEDULED.value
            assert session["status"] == SessionState.INTERRUPTED.value
        finally:
            store.close()

    asyncio.run(_run())


def test_create_lease_if_schedulable_does_not_reopen_terminal_task(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        request = _task_request("task-create-lease-stale")
        store.create_task(request, session_id="sess-create-lease-stale", submitter_id="submitter")
        finished = TaskFinished(
            task_id=request.task_id,
            trace_id=request.trace_id,
            session_id="sess-create-lease-stale",
            device_id="",
            lease_id="",
            state=TaskState.INTERRUPTED,
            error="cancel won",
        )
        assert store.finish_task_if_nonterminal(
            request.task_id,
            TaskState.INTERRUPTED,
            result_json=finished.model_dump_json(),
        )

        try:
            store.create_lease_if_schedulable(
                lease_id="lease-create-lease-stale",
                session_id="sess-create-lease-stale",
                task_id=request.task_id,
                device_id="dev-create-lease-stale",
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("stale lease creation should fail")

        row = store.task_row(request.task_id)
        session = store.session_row("sess-create-lease-stale")
        assert row is not None
        assert session is not None
        assert row["status"] == TaskState.INTERRUPTED.value
        assert session["status"] == SessionState.PENDING.value
        assert session["lease_id"] is None
    finally:
        store.close()


def test_rollback_undelivered_lease_does_not_reopen_terminal_task(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        request = _task_request("task-rollback-stale")
        store.create_task(request, session_id="sess-rollback-stale", submitter_id="submitter")
        store.create_lease_if_schedulable(
            lease_id="lease-rollback-stale",
            session_id="sess-rollback-stale",
            task_id=request.task_id,
            device_id="dev-rollback-stale",
        )
        finished = TaskFinished(
            task_id=request.task_id,
            trace_id=request.trace_id,
            session_id="sess-rollback-stale",
            device_id="dev-rollback-stale",
            lease_id="lease-rollback-stale",
            state=TaskState.INTERRUPTED,
            error="terminal won before rollback",
        )
        assert store.finish_task_if_nonterminal(
            request.task_id,
            TaskState.INTERRUPTED,
            result_json=finished.model_dump_json(),
        )

        lease_released, task_requeued = store.rollback_undelivered_lease_if_active(
            lease_id="lease-rollback-stale",
            session_id="sess-rollback-stale",
            task_id=request.task_id,
        )

        row = store.task_row(request.task_id)
        session = store.session_row("sess-rollback-stale")
        lease = store.lease_row("lease-rollback-stale")
        assert lease_released is True
        assert task_requeued is False
        assert row is not None
        assert session is not None
        assert lease is not None
        assert row["status"] == TaskState.INTERRUPTED.value
        assert session["status"] == SessionState.RELEASED.value
        assert lease["status"] == SessionState.RELEASED.value
    finally:
        store.close()


def test_mark_queued_async_does_not_reopen_terminal_task(tmp_path) -> None:
    async def _run() -> None:
        store = _store(tmp_path)
        try:
            events = ControllerEventLog(store)
            queue = TaskQueue(store=store, events=events)
            request = _task_request("task-mark-queued-stale")
            store.create_task(request, session_id="sess-mark-queued-stale", submitter_id="submitter")
            finished = TaskFinished(
                task_id=request.task_id,
                trace_id=request.trace_id,
                session_id="sess-mark-queued-stale",
                device_id="",
                lease_id="",
                state=TaskState.INTERRUPTED,
                error="terminal won before queue",
            )
            assert await queue.finish_async(finished)

            queued = await queue.mark_queued_async(
                request,
                session_id="sess-mark-queued-stale",
                reason="stale retry",
            )

            row = store.task_row(request.task_id)
            assert queued is False
            assert row is not None
            assert row["status"] == TaskState.INTERRUPTED.value
        finally:
            store.close()

    asyncio.run(_run())


def test_mark_command_dispatched_does_not_reopen_cancelled_command(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        request = CommandRequest(
            command_id="cmd-cancelled-before-dispatch",
            idempotency_key="idem-cancelled-before-dispatch",
            session_id="sess-cancelled-before-dispatch",
            lease_id="lease-cancelled-before-dispatch",
            device_id="dev-cancelled-before-dispatch",
            tool_id="device.echo",
        )
        claim, cached = store.claim_command(request)
        assert claim == "new"
        assert cached is None
        cancelled = CommandResult(
            command_id=request.command_id,
            session_id=request.session_id,
            lease_id=request.lease_id,
            device_id=request.device_id,
            status=CommandState.CANCELLED,
            error="session terminal",
        )
        assert store.record_command_result(cancelled) is None

        preempted = store.mark_command_dispatched_for_request(request)

        row = store.command_row(request.command_id, session_id=request.session_id, lease_id=request.lease_id)
        assert preempted is not None
        assert preempted.status == CommandState.CANCELLED
        assert row is not None
        assert row["status"] == CommandState.CANCELLED.value
        assert "session terminal" in str(row["result_json"])
    finally:
        store.close()


def test_task_status_reader_records_query_operation_timings(tmp_path) -> None:
    store = _store(tmp_path)
    observed: list[tuple[str, float]] = []
    try:
        request = _task_request("task-status-timing")
        store.create_task(request, session_id="sess-status-timing", submitter_id="submitter")
        reader = TaskStatusReader(
            store,
            timing_observer=lambda operation, elapsed_ms: observed.append((operation, elapsed_ms)),
        )

        assert len(reader.statuses_for_ids([request.task_id])) == 1
        assert len(reader.list_tasks(TaskStatusFilter(limit=1))) == 1

        operations = {operation for operation, _ in observed}
        assert "task_status.task_rows_by_ids" in operations
        assert "task_status.statuses_for_ids_total" in operations
        assert "task_status.task_rows_latest" in operations
        assert "task_status.list_tasks_total" in operations
        assert all(elapsed_ms >= 0 for _, elapsed_ms in observed)
    finally:
        store.close()


def test_task_status_reader_chunks_large_queries_and_preserves_order(tmp_path) -> None:
    store = _store(tmp_path)
    observed: list[str] = []
    try:
        for task_id in ("task-status-a", "task-status-b"):
            store.create_task(
                _task_request(task_id),
                session_id=f"sess-{task_id}",
                submitter_id="submitter",
            )
        reader = TaskStatusReader(store, timing_observer=lambda operation, _elapsed_ms: observed.append(operation))
        task_ids = [
            "task-status-a",
            *(f"missing-{index}" for index in range(1_001)),
            "task-status-b",
            "task-status-a",
        ]

        snapshots = reader.statuses_for_ids(task_ids)

        assert [snapshot.task_id for snapshot in snapshots] == ["task-status-a", "task-status-b", "task-status-a"]
        assert observed.count("task_status.task_rows_by_ids") == 3
        observed.clear()
        assert reader.statuses_for_ids([]) == []
        assert "task_status.task_rows_by_ids" not in observed
        assert observed == ["task_status.statuses_for_ids_total"]
    finally:
        store.close()


def test_task_status_reader_uses_bounded_latest_pages(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    try:
        for idx in range(3):
            task = _task_request(f"task-bounded-{idx}")
            store.create_task(task, session_id=f"sess-bounded-{idx}", submitter_id="submitter")

        def _fail_unbounded_task_rows():
            raise AssertionError("list_tasks must not load all controller task rows")

        monkeypatch.setattr(store, "task_rows", _fail_unbounded_task_rows)
        reader = TaskStatusReader(store)

        snapshots = reader.list_tasks(TaskStatusFilter(limit=2))

        assert len(snapshots) == 2
        assert {snapshot.task_id for snapshot in snapshots}.issubset(
            {"task-bounded-0", "task-bounded-1", "task-bounded-2"}
        )
    finally:
        store.close()


def test_task_status_reader_orders_explicit_task_ids_by_created_at(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        for idx in range(3):
            task = _task_request(f"task-explicit-{idx}")
            store.create_task(task, session_id=f"sess-explicit-{idx}", submitter_id="submitter")
            store.connection.execute(
                "UPDATE tasks SET created_at=? WHERE task_id=?",
                (f"2026-01-01T00:00:0{idx}+00:00", task.task_id),
            )
        store.connection.commit()
        reader = TaskStatusReader(store)

        snapshots = reader.list_tasks(
            TaskStatusFilter(
                task_ids=["task-explicit-0", "task-explicit-2", "task-explicit-1"],
                limit=1,
            )
        )

        assert [snapshot.task_id for snapshot in snapshots] == ["task-explicit-2"]
    finally:
        store.close()


def test_task_queue_finish_keeps_first_terminal_result(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        task = _task_request("task-terminal-cas")
        store.create_task(task, session_id="sess-terminal-cas", submitter_id="submitter")
        events = ControllerEventLog(store, event_write_batch_size=100, event_write_flush_interval_seconds=0.0)
        queue = TaskQueue(store=store, events=events)
        first = TaskFinished(
            task_id=task.task_id,
            trace_id=task.trace_id,
            session_id="sess-terminal-cas",
            state=TaskState.FINISHED,
            result={"winner": "first"},
        )
        second = TaskFinished(
            task_id=task.task_id,
            trace_id=task.trace_id,
            session_id="sess-terminal-cas",
            state=TaskState.FAILED,
            error="late replay",
            result={"winner": "second"},
        )

        assert queue.finish(first)
        assert not queue.finish(second)

        row = store.task_row(task.task_id)
        assert row is not None
        assert row["status"] == TaskState.FINISHED.value
        assert "first" in str(row["result_json"])
        assert "second" not in str(row["result_json"])
        events.flush()
        assert [row["event_type"] for row in store.event_rows()] == ["task_finished"]
        events.close()
    finally:
        store.close()


def test_controller_operation_timing_tracker_summarizes_and_drains() -> None:
    tracker = _ControllerOperationTimingTracker()

    tracker.observe("storage.lease_generation_query", 10.0)
    tracker.observe("storage.lease_generation_query", 20.0)
    tracker.observe("task_status.task_rows_latest", 30.0)

    snapshot = tracker.drain_snapshot()

    assert snapshot["storage.lease_generation_query"]["count"] == 2
    assert snapshot["storage.lease_generation_query"]["p50_ms"] == 15.0
    assert snapshot["storage.lease_generation_query"]["max_ms"] == 20.0
    assert snapshot["task_status.task_rows_latest"]["count"] == 1
    assert tracker.drain_snapshot() == {}


def test_update_commands_for_device_statuses_cancels_active_commands_only(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        _insert_command(store, command_id="queued", device_id="dev-1", session_id="sess-1", status=CommandState.QUEUED)
        _insert_command(store, command_id="sent", device_id="dev-1", session_id="sess-1", status=CommandState.SENT)
        _insert_command(
            store, command_id="running", device_id="dev-1", session_id="sess-1", status=CommandState.RUNNING
        )
        _insert_command(
            store, command_id="succeeded", device_id="dev-1", session_id="sess-1", status=CommandState.SUCCEEDED
        )
        _insert_command(store, command_id="other", device_id="dev-2", session_id="sess-2", status=CommandState.RUNNING)

        store.update_commands_for_device_statuses(
            "dev-1",
            {CommandState.QUEUED, CommandState.SENT, CommandState.RUNNING},
            CommandState.CANCELLED,
        )

        rows = {
            row["command_id"]: row["status"]
            for row in store.connection.execute("SELECT command_id, status FROM commands ORDER BY command_id")
        }
        assert rows == {
            "other": CommandState.RUNNING.value,
            "queued": CommandState.CANCELLED.value,
            "running": CommandState.CANCELLED.value,
            "sent": CommandState.CANCELLED.value,
            "succeeded": CommandState.SUCCEEDED.value,
        }
    finally:
        store.close()


def test_batched_event_log_flush_uses_background_writer(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    try:
        events = ControllerEventLog(store, event_write_batch_size=100, event_write_flush_interval_seconds=0.0)

        def _fail_sync_append(rows: object) -> None:
            del rows
            raise AssertionError("batched event writes must not use the controller loop connection")

        monkeypatch.setattr(store, "append_events", _fail_sync_append)
        for index in range(5):
            events.write("device_registered", trace_id=f"trace-{index}", entity_type="device", entity_id=f"dev-{index}")

        events.flush()

        assert [row["event_type"] for row in store.event_rows()] == ["device_registered"] * 5
        events.close()
    finally:
        store.close()


def test_batched_event_log_uses_shared_background_writer(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    submissions: list[ControllerStorePriority] = []
    original_submit = event_log_module.controller_store_submit_profiled

    def _submit(*args, **kwargs):
        submissions.append(kwargs["priority"])
        return original_submit(*args, **kwargs)

    monkeypatch.setattr(event_log_module, "controller_store_submit_profiled", _submit)
    try:
        events = ControllerEventLog(store, event_write_batch_size=2, event_write_flush_interval_seconds=0.0)
        events.write("device_registered", trace_id="trace-1", entity_type="device", entity_id="dev-1")
        events.write("device_registered", trace_id="trace-2", entity_type="device", entity_id="dev-2")
        events.flush()

        assert submissions == [ControllerStorePriority.BACKGROUND]
        assert events.snapshot()["persisted_rows"] == 2
        assert events.snapshot()["failures"] == 0
        events.close()
    finally:
        store.close()


def test_batched_event_log_can_flush_without_waiting_inside_running_loop(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    waits: list[bool] = []

    def _record_append(self: object, rows: list[object], *, wait: bool) -> None:
        del self, rows
        waits.append(wait)

    monkeypatch.setattr(event_log_module._AsyncEventWriter, "append", _record_append)
    try:
        events = ControllerEventLog(store, event_write_batch_size=100, event_write_flush_interval_seconds=0.0)

        async def _run() -> None:
            events.write("device_registered", trace_id="trace", entity_type="device", entity_id="dev")
            events.flush(wait=False)

        import asyncio

        asyncio.run(_run())

        assert waits == [False]
        events.close()
    finally:
        store.close()


def test_batched_event_log_public_flush_waits_for_already_queued_rows(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    try:
        events = ControllerEventLog(store, event_write_batch_size=100, event_write_flush_interval_seconds=0.0)
        original_append_rows = event_log_module._append_event_rows

        def _slow_append_rows(connection: object, rows: list[object]) -> None:
            import time

            time.sleep(0.05)
            original_append_rows(connection, rows)  # type: ignore[arg-type]

        monkeypatch.setattr(event_log_module, "_append_event_rows", _slow_append_rows)

        async def _enqueue_without_waiting() -> None:
            events.write("device_registered", trace_id="trace", entity_type="device", entity_id="dev")
            events.flush(wait=False)

        import asyncio

        asyncio.run(_enqueue_without_waiting())
        events.flush()

        assert [row["event_type"] for row in store.event_rows()] == ["device_registered"]
        events.close()
    finally:
        store.close()


def test_batched_event_log_retries_failed_non_waiting_write(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    try:
        events = ControllerEventLog(store, event_write_batch_size=100, event_write_flush_interval_seconds=0.0)
        original_append_rows = event_log_module._append_event_rows
        attempts = {"count": 0}

        def _flaky_append_rows(connection: object, rows: list[object]) -> None:
            if not rows:
                original_append_rows(connection, rows)  # type: ignore[arg-type]
                return
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise sqlite3.OperationalError("temporary busy")
            original_append_rows(connection, rows)  # type: ignore[arg-type]

        import sqlite3

        monkeypatch.setattr(event_log_module, "_append_event_rows", _flaky_append_rows)

        async def _enqueue_without_waiting() -> None:
            events.write("device_registered", trace_id="trace", entity_type="device", entity_id="dev")
            events.flush(wait=False)

        import asyncio

        asyncio.run(_enqueue_without_waiting())
        events.flush()

        assert attempts["count"] == 2
        assert [row["event_type"] for row in store.event_rows()] == ["device_registered"]
        events.close()
    finally:
        store.close()


def test_async_event_writer_close_drains_full_queue_and_reports_in_flight(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(event_log_module, "_EVENT_WRITER_QUEUE_MAX", 1)
    write_future: Future[object] = Future()
    write_started = threading.Event()

    def _blocked_submit(*_args, **_kwargs):
        write_started.set()
        return write_future

    monkeypatch.setattr(event_log_module, "controller_store_submit_profiled", _blocked_submit)
    writer = event_log_module._AsyncEventWriter(tmp_path / "controller.sqlite")

    def _row(index: int) -> ControllerEventRow:
        return ControllerEventRow(
            trace_id=f"trace-{index}",
            entity_type="benchmark",
            entity_id=f"event-{index}",
            event_type="event_writer_close_test",
            payload={},
        )

    closer = threading.Thread(target=writer.close)
    try:
        writer.append([_row(1)], wait=False)
        assert write_started.wait(timeout=1)
        writer.append([_row(2)], wait=False)
        writer.append([_row(3)], wait=False)

        closer.start()
        deadline = time.monotonic() + 1
        while writer.snapshot()["accepting"] and time.monotonic() < deadline:
            time.sleep(0.001)

        snapshot = writer.snapshot()
        assert snapshot["accepting"] == 0
        assert snapshot["pending_batches"] == 1
        assert snapshot["in_flight_batches"] == 1
        assert snapshot["in_flight_rows"] == 1
        assert snapshot["thread_alive"] == 1
        assert snapshot["dropped_rows"] == 1
        writer.append([_row(4)], wait=False)
        assert writer.snapshot()["dropped_rows"] == 2
        assert closer.is_alive()

        write_future.set_result(None)
        closer.join(timeout=1)

        final = writer.snapshot()
        assert not closer.is_alive()
        assert final["thread_alive"] == 0
        assert final["in_flight_batches"] == 0
        assert final["in_flight_rows"] == 0
        assert final["persisted_rows"] == 2
        assert final["last_success_age_ms"] >= 0
    finally:
        if not write_future.done():
            write_future.set_result(None)
        closer.join(timeout=1)
        writer.close()


def test_async_event_writer_full_queue_snapshot_wait_append_and_close_do_not_deadlock(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(event_log_module, "_EVENT_WRITER_QUEUE_MAX", 1)
    first_future: Future[object] = Future()
    write_started = threading.Event()

    def _blocked_submit(*_args, **_kwargs):
        write_started.set()
        return first_future

    monkeypatch.setattr(event_log_module, "controller_store_submit_profiled", _blocked_submit)
    writer = event_log_module._AsyncEventWriter(tmp_path / "controller.sqlite")

    def _row(index: int) -> ControllerEventRow:
        return ControllerEventRow(
            trace_id=f"trace-deadlock-{index}",
            entity_type="benchmark",
            entity_id=f"event-deadlock-{index}",
            event_type="event_writer_deadlock_test",
            payload={},
        )

    producer = threading.Thread(target=lambda: writer.append([_row(3)], wait=True))
    observer = threading.Thread(target=writer.snapshot)
    closer = threading.Thread(target=writer.close)
    try:
        writer.append([_row(1)], wait=False)
        assert write_started.wait(timeout=1)
        writer.append([_row(2)], wait=False)

        producer.start()
        deadline = time.monotonic() + 1
        while writer.snapshot()["active_producers"] == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert writer.snapshot()["active_producers"] == 1

        observer.start()
        observer.join(timeout=1)
        assert not observer.is_alive()

        closer.start()
        assert closer.is_alive()
        first_future.set_result(None)

        producer.join(timeout=1)
        closer.join(timeout=1)
        assert not producer.is_alive()
        assert not closer.is_alive()
        assert writer.snapshot()["persisted_rows"] == 3
    finally:
        if not first_future.done():
            first_future.set_result(None)
        producer.join(timeout=1)
        closer.join(timeout=1)
        writer.close()


def test_async_event_writer_full_queue_snapshot_and_drain_do_not_deadlock(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(event_log_module, "_EVENT_WRITER_QUEUE_MAX", 1)
    first_future: Future[object] = Future()
    write_started = threading.Event()

    def _blocked_submit(*_args, **_kwargs):
        write_started.set()
        return first_future

    monkeypatch.setattr(event_log_module, "controller_store_submit_profiled", _blocked_submit)
    writer = event_log_module._AsyncEventWriter(tmp_path / "controller.sqlite")

    def _row(index: int) -> ControllerEventRow:
        return ControllerEventRow(
            trace_id=f"trace-drain-{index}",
            entity_type="benchmark",
            entity_id=f"event-drain-{index}",
            event_type="event_writer_drain_test",
            payload={},
        )

    drainer = threading.Thread(target=writer.drain)
    observer = threading.Thread(target=writer.snapshot)
    try:
        writer.append([_row(1)], wait=False)
        assert write_started.wait(timeout=1)
        writer.append([_row(2)], wait=False)

        drainer.start()
        deadline = time.monotonic() + 1
        while writer.snapshot()["active_producers"] == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert writer.snapshot()["active_producers"] == 1

        observer.start()
        observer.join(timeout=1)
        assert not observer.is_alive()

        first_future.set_result(None)
        drainer.join(timeout=1)
        assert not drainer.is_alive()
        assert writer.snapshot()["persisted_rows"] == 2
    finally:
        if not first_future.done():
            first_future.set_result(None)
        drainer.join(timeout=1)
        writer.close()


def test_async_event_writer_fatal_current_item_releases_waiting_append(tmp_path, monkeypatch) -> None:
    failure = RuntimeError("injected writer crash")

    def _crash_current_item(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(event_log_module._AsyncEventWriter, "_write_item", _crash_current_item)
    writer = event_log_module._AsyncEventWriter(tmp_path / "controller.sqlite")
    row = ControllerEventRow(
        trace_id="trace-fatal-wait",
        entity_type="benchmark",
        entity_id="event-fatal-wait",
        event_type="event_writer_fatal_wait_test",
        payload={},
    )
    errors: list[BaseException] = []

    def _append_waiting() -> None:
        try:
            writer.append([row], wait=True)
        except BaseException as exc:
            errors.append(exc)

    producer = threading.Thread(target=_append_waiting)
    try:
        producer.start()
        producer.join(timeout=1)

        assert not producer.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert errors[0].__cause__ is failure
        snapshot = writer.snapshot()
        assert snapshot["thread_alive"] == 0
        assert snapshot["in_flight_rows"] == 0
        assert snapshot["failures"] == 1
        assert snapshot["dropped_rows"] == 0
    finally:
        producer.join(timeout=1)
        writer.close()


def test_async_event_writer_fatal_current_item_counts_non_waiting_drop(tmp_path, monkeypatch) -> None:
    failure = RuntimeError("injected writer crash")

    def _crash_current_item(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(event_log_module._AsyncEventWriter, "_write_item", _crash_current_item)
    writer = event_log_module._AsyncEventWriter(tmp_path / "controller.sqlite")
    row = ControllerEventRow(
        trace_id="trace-fatal-nonwait",
        entity_type="benchmark",
        entity_id="event-fatal-nonwait",
        event_type="event_writer_fatal_nonwait_test",
        payload={},
    )
    try:
        writer.append([row], wait=False)
        writer._thread.join(timeout=1)  # noqa: SLF001 - deterministic fatal writer injection

        snapshot = writer.snapshot()
        assert snapshot["thread_alive"] == 0
        assert snapshot["in_flight_rows"] == 0
        assert snapshot["failures"] == 1
        assert snapshot["dropped_rows"] == 1
    finally:
        writer.close()
