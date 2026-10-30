from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from northcua.batch.backpressure import (
    _RUNTIME_TELEMETRY_QUERY,
    ControllerBackpressureGate,
    _read_backpressure_snapshot,
)
from northcua.batch.contracts import BatchManifest, BatchRunConfig, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.metrics import BatchMetrics
from northcua.batch.planner import PlannedTaskSubmission
from northcua.batch.preparation import PreparedSandboxPolicy, PreparedSandboxRecord, PreparedSandboxStatus
from northcua.batch.preparation.assignment import PreparedSandboxAssignment
from northcua.batch.runner import BatchRunner
from northcua.batch.runner import attempt_helpers as attempt_helpers_module
from northcua.batch.runner import attempt_reconcile as attempt_reconcile_module
from northcua.batch.runner import attempt_submit as attempt_submit_module
from northcua.batch.runner import scheduler as scheduler_module
from northcua.batch.runner.attempt_helpers import PreparedSandboxBackpressureDeferred, _execute_attempt
from northcua.batch.runner.attempt_submit import _local_duplicate_statuses, submit_to_controller
from northcua.batch.runner.events import BatchAttemptEventMessage
from northcua.batch.runner.leases import BatchTaskLeaseContext
from northcua.batch.runner.resume import BatchStatusReconciler, ControllerStatusUnavailable
from northcua.batch.runner.scheduler import BatchScheduler, _batch_has_no_remaining_work
from northcua.batch.runner.telemetry_helpers import CleanupResult
from northcua.batch.state import LeasePolicy, SQLiteBatchStateRepository
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua.sandboxes.contracts import SandboxHandle, SandboxLease
from northcua.sandboxes.provider.fake import FakeSandboxProvider
from northcua.sandboxes.provider.registry import SandboxProviderRegistry
from northcua.telemetry.contracts import FailureCategory
from northcua_protocol.payloads import TaskAccepted, TaskFinished, TaskRequest, TaskStatusSnapshot
from northcua_protocol.state import SessionState, TaskState


@pytest.mark.asyncio
async def test_backpressure_gate_requires_sustained_controller_pressure(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        sample_interval_seconds=0.1,
    )

    _insert_runtime_event(controller_db, event_loop_lag_ms=11_000, recv_q_max=0)
    _insert_runtime_event(controller_db, event_loop_lag_ms=12_000, recv_q_max=0)
    assert not (await gate.refresh_now()).active

    _insert_runtime_event(controller_db, event_loop_lag_ms=13_000, recv_q_max=0)
    snapshot = await gate.refresh_now()

    assert snapshot.active
    assert "event_loop_lag_ms" in snapshot.reason
    assert snapshot.samples == 3
    assert snapshot.latest_event_loop_lag_ms == 13_000


@pytest.mark.asyncio
async def test_backpressure_gate_clears_when_latest_samples_recover(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        sample_interval_seconds=0.1,
    )
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    assert (await gate.refresh_now()).active

    for value in (100, 200, 300):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    snapshot = await gate.refresh_now()

    assert not snapshot.active
    assert snapshot.latest_event_loop_lag_ms == 300


@pytest.mark.asyncio
async def test_backpressure_gate_ignores_stale_controller_telemetry(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    old = time.time() - 600
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0, created_at=old)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        max_sample_age_seconds=120,
        sample_interval_seconds=0.1,
    )

    snapshot = await gate.refresh_now()

    assert not snapshot.active
    assert snapshot.reason == "stale_controller_telemetry"


def test_backpressure_runtime_query_uses_existing_created_at_index(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in range(100):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)

    with sqlite3.connect(controller_db) as connection:
        plan = "\n".join(
            str(row) for row in connection.execute("EXPLAIN QUERY PLAN " + _RUNTIME_TELEMETRY_QUERY, (3,)).fetchall()
        )

    assert "events_type_created_idx" in plan
    assert "USE TEMP B-TREE" not in plan


def test_backpressure_snapshot_timeout_is_non_active_and_telemetered(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)

    snapshot = _read_backpressure_snapshot(
        controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        max_sample_age_seconds=120,
        query_timeout_seconds=0,
    )

    assert not snapshot.active
    assert snapshot.reason == "sample_timeout"
    assert snapshot.sample_timed_out is True


@pytest.mark.asyncio
async def test_backpressure_snapshot_returns_cached_state_while_background_sample_runs(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        sample_interval_seconds=0.1,
    )

    first = await gate.snapshot()
    second = gate.cached_snapshot()
    for _ in range(20):
        if second.active:
            break
        await asyncio.sleep(0.025)
        second = gate.cached_snapshot()

    assert not first.active
    assert second.active
    await gate.close()


@pytest.mark.asyncio
async def test_backpressure_sample_timeout_preserves_recent_active_snapshot(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        max_sample_age_seconds=120,
    )
    assert (await gate.refresh_now()).active
    active_cached_at = gate._cached_at  # noqa: SLF001 - targeted cache expiry regression.
    active_last_seen_at = gate._last_active_at  # noqa: SLF001 - targeted cache expiry regression.

    gate.sample_timeout_seconds = 0
    snapshot = await gate.refresh_now()

    assert snapshot.active
    assert snapshot.sample_timed_out is True
    assert "event_loop_lag_ms" in snapshot.reason
    assert gate._cached_at == active_cached_at  # noqa: SLF001 - timeout must not extend active cache age.
    assert gate._last_active_at == active_last_seen_at  # noqa: SLF001 - timeout must not extend jitter recovery.

    gate.max_sample_age_seconds = 1.0
    gate._cached_at = time.monotonic() - 2.0  # noqa: SLF001 - simulate repeated timeouts beyond cache age.
    expired = await gate.refresh_now()

    assert not expired.active
    assert expired.reason == "sample_timeout"
    assert expired.sample_timed_out is True


@pytest.mark.asyncio
async def test_local_duplicate_status_lookup_timeout_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    monkeypatch.setenv("NORTHCUA_BATCH_DUPLICATE_STATUS_DB_TIMEOUT_SECONDS", "0.01")

    def slow_statuses(*args: object, **kwargs: object) -> list[TaskStatusSnapshot]:
        _ = args, kwargs
        time.sleep(0.2)
        return []

    monkeypatch.setattr("northcua.batch.controller_db_status.LocalControllerDBStatusReader.statuses", slow_statuses)
    started = time.monotonic()

    snapshots = await _local_duplicate_statuses(controller_db_path=controller_db, task_id="controller-task")

    assert snapshots is None
    assert time.monotonic() - started < 0.1


@pytest.mark.asyncio
async def test_duplicate_controller_task_id_db_timeout_does_not_use_websocket_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    monkeypatch.setenv("NORTHCUA_BATCH_DUPLICATE_STATUS_DB_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("NORTHCUA_BATCH_RECONCILE_STATUS_WEBSOCKET_FALLBACK_ON_DB_TIMEOUT", "0")

    def slow_statuses(*args: object, **kwargs: object) -> list[TaskStatusSnapshot]:
        _ = args, kwargs
        time.sleep(0.2)
        return []

    monkeypatch.setattr("northcua.batch.controller_db_status.LocalControllerDBStatusReader.statuses", slow_statuses)
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    watcher = _DuplicateTaskWatcher(
        TaskStatusSnapshot(
            task_id="controller-task",
            trace_id="trace",
            session_id="session-controller-task",
            agent_kind="desktop",
            state=TaskState.FINISHED,
            metadata={
                "batch_id": "batch",
                "batch_task_id": "case-1",
                "batch_attempt": 1,
            },
        )
    )

    started = time.monotonic()
    result = await submit_to_controller(
        planner=_Planner(),
        watcher=watcher,  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest, telemetry_controller_db_path=controller_db),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=_NullEvents(),
    )

    assert time.monotonic() - started < 0.1
    assert result.status == BatchTaskStatus.FAILED
    assert result.controller_task_id == "controller-task"
    assert result.trace_id == "trace"
    assert result.last_reconciled_status == "unknown"
    assert result.next_attempt_after > time.time()
    assert watcher.submit_calls == 1
    assert watcher.status_calls == 0


@pytest.mark.asyncio
async def test_resume_status_lookup_timeout_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    monkeypatch.setenv("NORTHCUA_BATCH_RECONCILE_STATUS_DB_TIMEOUT_SECONDS", "0.01")

    def slow_statuses(*args: object, **kwargs: object) -> list[TaskStatusSnapshot]:
        _ = args, kwargs
        time.sleep(0.2)
        return []

    monkeypatch.setattr("northcua.batch.controller_db_status.LocalControllerDBStatusReader.statuses", slow_statuses)
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    watcher = _StatusWatcher([])
    reconciler = BatchStatusReconciler(watcher)
    started = time.monotonic()

    with pytest.raises(ControllerStatusUnavailable):
        await reconciler._statuses(  # noqa: SLF001 - targeted timeout regression.
            config=BatchRunConfig(manifest=manifest, telemetry_controller_db_path=controller_db),
            controller_url="ws://controller",
            token="token",
            task_ids=["controller-task"],
            status_semaphore=None,
        )

    assert watcher.calls == 0
    assert time.monotonic() - started < 0.1


@pytest.mark.asyncio
async def test_resume_status_lookup_timeout_websocket_fallback_is_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    monkeypatch.setenv("NORTHCUA_BATCH_RECONCILE_STATUS_DB_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("NORTHCUA_BATCH_RECONCILE_STATUS_WEBSOCKET_FALLBACK_ON_DB_TIMEOUT", "1")

    def slow_statuses(*args: object, **kwargs: object) -> list[TaskStatusSnapshot]:
        _ = args, kwargs
        time.sleep(0.2)
        return []

    monkeypatch.setattr("northcua.batch.controller_db_status.LocalControllerDBStatusReader.statuses", slow_statuses)
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        state=TaskState.FINISHED,
    )
    watcher = _StatusWatcher([snapshot])
    reconciler = BatchStatusReconciler(watcher)
    started = time.monotonic()

    snapshots = await reconciler._statuses(  # noqa: SLF001 - targeted timeout regression.
        config=BatchRunConfig(manifest=manifest, telemetry_controller_db_path=controller_db),
        controller_url="ws://controller",
        token="token",
        task_ids=["controller-task"],
        status_semaphore=None,
    )

    assert snapshots == [snapshot]
    assert watcher.calls == 1
    assert time.monotonic() - started < 0.1


@pytest.mark.asyncio
async def test_pressure_reconciliation_defers_fresh_retry_until_pressure_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = BatchManifest(
        batch_id="pressure-reconcile-defers-retry",
        tasks=[
            BatchTaskSpec(
                task_id="case-1",
                agent_kind="desktop",
                query="retry after pressure",
                max_retries=1,
            )
        ],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    old_claim = repository.claim_next(
        lease_owner="old-runner",
        policy=LeasePolicy(lease_timeout_seconds=0.01, heartbeat_interval_seconds=0.01),
        max_attempts_by_task={"case-1": 2},
    )
    assert old_claim is not None
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-old",
            trace_id="trace-old",
            session_id="session-old",
        ),
        lease_token=old_claim.lease_token,
    )
    time.sleep(0.03)
    gate = _OneShotPressureGate()
    executor = _FreshRetryExecutor(gate)
    monkeypatch.setattr(
        scheduler_module.ControllerBackpressureGate,
        "from_config",
        classmethod(lambda cls, config: gate),
    )
    scheduler = BatchScheduler(
        provider_registry=SandboxProviderRegistry(),
        executor=executor,  # type: ignore[arg-type]
        resume_reconciler=_RetryableTerminalReconciler(),
    )

    started_at = time.monotonic()
    states = await asyncio.wait_for(
        scheduler.run(
            config=BatchRunConfig(
                manifest=manifest,
                output_dir=tmp_path,
                concurrency=1,
                lease_timeout_seconds=30,
                infra_retry_backoff_seconds=[0.2],
            ),
            repository=repository,
            metrics=BatchMetrics(),
            events=_NullEvents(),  # type: ignore[arg-type]
            output_dir=tmp_path,
        ),
        timeout=3,
    )

    assert executor.calls == 1
    assert executor.snapshot_calls_at_submit[0] >= 2
    assert time.monotonic() - started_at >= 0.15
    assert states[0].status == BatchTaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_backpressure_wait_stops_when_control_flag_fires(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        sleep_seconds=30,
        sample_interval_seconds=0.1,
    )

    await gate.refresh_now()
    proceed = await gate.wait_if_needed(scope="claim", stop_requested=lambda: True)

    assert not proceed


@pytest.mark.asyncio
async def test_backpressure_wait_accepts_async_stop_predicate(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        sleep_seconds=30,
        sample_interval_seconds=0.1,
    )

    async def stop_requested() -> bool:
        await asyncio.sleep(0)
        return True

    await gate.refresh_now()
    proceed = await gate.wait_if_needed(scope="claim", stop_requested=stop_requested)

    assert not proceed


@pytest.mark.asyncio
async def test_backpressure_async_stop_predicate_does_not_block_event_loop(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        sleep_seconds=30,
        sample_interval_seconds=0.1,
    )
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    async def stop_requested() -> bool:
        await asyncio.to_thread(time.sleep, 0.05)
        return True

    await gate.refresh_now()
    ticker_task = asyncio.create_task(ticker())
    proceed = await asyncio.wait_for(gate.wait_if_needed(scope="claim", stop_requested=stop_requested), timeout=1)
    ticker_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ticker_task

    assert not proceed
    assert ticks > 3


@pytest.mark.asyncio
async def test_submit_gate_cancels_before_opening_controller_task(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        sleep_seconds=30,
        sample_interval_seconds=0.1,
    )
    watcher = _NoSubmitWatcher()
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )

    await gate.refresh_now()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=watcher,
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        lease=SandboxLease(
            lease_id="lease",
            handle=SandboxHandle(sandbox_id="sandbox", provider="fake", template="template"),
            device_id="device",
        ),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=_NullEvents(),  # type: ignore[arg-type]
        backpressure_gate=gate,
        backpressure_stop_requested=lambda: True,
    )

    assert result.status == BatchTaskStatus.CANCELED
    assert watcher.called == 0


@pytest.mark.asyncio
async def test_submit_gate_observes_async_cancel_while_waiting_for_pressure(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        sleep_seconds=0.01,
        sample_interval_seconds=0.1,
    )
    watcher = _NoSubmitWatcher()
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    cancel_requested = False

    async def stop_requested() -> bool:
        await asyncio.sleep(0)
        return cancel_requested

    async def request_cancel() -> None:
        nonlocal cancel_requested
        await asyncio.sleep(0.02)
        cancel_requested = True

    await gate.refresh_now()
    cancel_task = asyncio.create_task(request_cancel())
    result = await asyncio.wait_for(
        submit_to_controller(
            planner=_Planner(),
            watcher=watcher,
            config=BatchRunConfig(manifest=manifest),
            task=manifest.tasks[0],
            lease=SandboxLease(
                lease_id="lease",
                handle=SandboxHandle(sandbox_id="sandbox", provider="fake", template="template"),
                device_id="device",
            ),
            controller_url="ws://controller",
            token="token",
            metrics=BatchMetrics(),
            state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
            events=_NullEvents(),  # type: ignore[arg-type]
            backpressure_gate=gate,
            backpressure_stop_requested=stop_requested,
        ),
        timeout=1,
    )
    await cancel_task

    assert result.status == BatchTaskStatus.CANCELED
    assert watcher.called == 0


@pytest.mark.asyncio
async def test_submit_skips_controller_when_local_submitted_write_is_rejected() -> None:
    watcher = _NoSubmitWatcher()
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=watcher,
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        lease=SandboxLease(
            lease_id="lease",
            handle=SandboxHandle(sandbox_id="sandbox", provider="fake", template="template"),
            device_id="device",
        ),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=_RejectingEvents(),  # type: ignore[arg-type]
    )

    assert result.status == BatchTaskStatus.CANCELED
    assert result.controller_task_id == ""
    assert result.trace_id == ""
    assert watcher.called == 0


@pytest.mark.asyncio
async def test_submit_records_preflight_timing_events(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_JITTER_RECOVERY_WINDOW_SECONDS", "300")
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_JITTER_NORMAL_CAP_SECONDS", "30")
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_JITTER_RECOVERY_CAP_SECONDS", "60")
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    events = _NullEvents()
    gate = _DelayingSubmitBackpressureGate(recently_active=True)
    ready_for_submit_at = time.monotonic() - 1.25

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=_SuccessfulWatcher(),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest, controller_submit_jitter_seconds=120),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,  # type: ignore[arg-type]
        backpressure_gate=gate,  # type: ignore[arg-type]
        ready_for_controller_submit_at=ready_for_submit_at,
    )

    messages = [event["message"] for event in events.events]
    assert result.status == BatchTaskStatus.SUCCEEDED
    assert gate.scopes == ["submit"]
    assert BatchAttemptEventMessage.CONTROLLER_SUBMIT_PREFLIGHT_STARTED in messages
    assert BatchAttemptEventMessage.CONTROLLER_SUBMIT_PREFLIGHT_FINISHED in messages
    assert messages.index(BatchAttemptEventMessage.CONTROLLER_SUBMIT_PREFLIGHT_STARTED) < messages.index(
        BatchAttemptEventMessage.CONTROLLER_SUBMIT_PREFLIGHT_FINISHED
    )
    assert messages.index(BatchAttemptEventMessage.CONTROLLER_SUBMIT_PREFLIGHT_FINISHED) < messages.index(
        BatchAttemptEventMessage.CONTROLLER_SUBMIT_STARTED
    )
    assert messages.index(BatchAttemptEventMessage.CONTROLLER_SUBMIT_STARTED) < messages.index(
        BatchAttemptEventMessage.CONTROLLER_SUBMIT_OPEN_STARTED
    )
    assert messages.index(BatchAttemptEventMessage.CONTROLLER_SUBMIT_OPEN_STARTED) < messages.index(
        BatchAttemptEventMessage.CONTROLLER_SUBMIT_REQUEST_SENT
    )
    assert messages.index(BatchAttemptEventMessage.CONTROLLER_SUBMIT_REQUEST_SENT) < messages.index(
        BatchAttemptEventMessage.CONTROLLER_SUBMIT_ACCEPTED
    )
    started = next(
        event
        for event in events.events
        if event["message"] == BatchAttemptEventMessage.CONTROLLER_SUBMIT_PREFLIGHT_STARTED
    )
    finished = next(
        event
        for event in events.events
        if event["message"] == BatchAttemptEventMessage.CONTROLLER_SUBMIT_PREFLIGHT_FINISHED
    )
    started_details = started["details"]
    finished_details = finished["details"]
    open_started = next(
        event for event in events.events if event["message"] == BatchAttemptEventMessage.CONTROLLER_SUBMIT_OPEN_STARTED
    )
    request_sent = next(
        event for event in events.events if event["message"] == BatchAttemptEventMessage.CONTROLLER_SUBMIT_REQUEST_SENT
    )
    open_details = open_started["details"]
    request_details = request_sent["details"]
    assert started_details["ready_to_backpressure_gate_seconds"] >= 1.0
    assert started_details["backpressure_gate_present"] is True
    assert (
        finished_details["ready_to_after_backpressure_seconds"] >= started_details["ready_to_backpressure_gate_seconds"]
    )
    assert finished_details["backpressure_wait_seconds"] >= 0
    assert finished_details["upsert_task_submitted_seconds"] >= 0
    assert finished_details["requested_submit_jitter_seconds"] == 120
    assert finished_details["effective_submit_jitter_seconds"] == 60
    assert finished_details["submit_jitter_mode"] == "recovery_cap"
    assert "planned_submit_jitter_seconds" in open_details
    assert "actual_submit_jitter_sleep_seconds" in open_details
    assert "submit_jitter_overshoot_seconds" in open_details
    assert request_details["connection_open_elapsed_seconds"] >= 0
    assert events.states[0].status == BatchTaskStatus.SUBMITTED


@pytest.mark.asyncio
async def test_prepared_backpressure_returns_when_no_claimable_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_BACKPRESSURE_SLEEP_SECONDS", "0.01")
    monkeypatch.setenv("NORTHCUA_BATCH_BACKPRESSURE_SAMPLE_INTERVAL_SECONDS", "0.01")
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    manifest = BatchManifest(
        batch_id="prepared-pressure-done",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="done")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: FakeSandboxProvider())
    config = BatchRunConfig(
        manifest=manifest,
        output_dir=tmp_path,
        concurrency=1,
        prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1),
        telemetry_controller_db_path=controller_db,
    )
    runner = BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: _FinishingSubmitter(url),
    )

    first = await runner.run(config)
    assert first.succeeded == 1
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)

    second = await asyncio.wait_for(
        runner.run(config),
        timeout=1,
    )

    assert second.succeeded == 1


def test_backpressure_no_remaining_work_treats_non_retryable_failure_as_done() -> None:
    states = [
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            terminal_decision={"retryable": False, "success": False},
        )
    ]

    assert _batch_has_no_remaining_work(states, {"case-1": 3})


def test_backpressure_no_remaining_work_treats_exhausted_pending_as_done() -> None:
    states = [
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.PENDING,
            attempts=3,
            retry_budget_remaining=0,
        )
    ]

    assert _batch_has_no_remaining_work(states, {"case-1": 3})


def test_backpressure_no_remaining_work_treats_fresh_pending_as_work() -> None:
    states = [
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.PENDING,
            attempts=0,
            retry_budget_remaining=3,
        )
    ]

    assert not _batch_has_no_remaining_work(states, {"case-1": 3})


@pytest.mark.asyncio
async def test_prepared_miss_defers_without_direct_create_under_active_backpressure(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    gate = ControllerBackpressureGate(
        controller_db_path=controller_db,
        loop_lag_threshold_ms=10_000,
        recv_q_max_threshold=120_000,
        sustained_samples=3,
        sample_interval_seconds=0.1,
    )
    manifest = BatchManifest(
        batch_id="prepared-pressure-miss",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="done")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    coordinator = _NoAcquireCoordinator()

    with pytest.raises(PreparedSandboxBackpressureDeferred):
        await gate.refresh_now()
        await _execute_attempt(
            coordinator=coordinator,
            binding_factory=object(),  # type: ignore[arg-type]
            planner=object(),  # type: ignore[arg-type]
            watcher=object(),  # type: ignore[arg-type]
            config=BatchRunConfig(manifest=manifest, telemetry_controller_db_path=controller_db),
            task=manifest.tasks[0],
            events=_NullEvents(),
            metrics=BatchMetrics(),
            output_dir=tmp_path,
            attempt=1,
            create_semaphore=None,
            submit_semaphore=None,
            prepared_manager=_PreparedMissManager(),  # type: ignore[arg-type]
            backpressure_gate=gate,
        )
    assert coordinator.acquire_calls == 0


@pytest.mark.asyncio
async def test_execute_attempt_defers_cleanup_for_reconcile_blocking_submitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = BatchManifest(
        batch_id="submit-reconcile",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="done")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    coordinator = _SingleLeaseCoordinator()
    events = _NullEvents()

    async def fake_submit_to_controller(**kwargs: object) -> BatchTaskState:
        task = kwargs["task"]
        assert isinstance(task, BatchTaskSpec)
        return BatchTaskState(
            task_id=task.task_id,
            attempts=1,
            status=BatchTaskStatus.SUBMITTED,
            controller_task_id="controller-task",
            trace_id="trace",
        )

    async def forbidden_release_with_telemetry(**kwargs: object) -> object:
        _ = kwargs
        raise AssertionError("reconcile-blocking SUBMITTED must not release sandbox")

    monkeypatch.setattr(attempt_helpers_module, "submit_to_controller", fake_submit_to_controller)
    monkeypatch.setattr(attempt_helpers_module, "release_with_telemetry", forbidden_release_with_telemetry)

    result = await _execute_attempt(
        coordinator=coordinator,  # type: ignore[arg-type]
        binding_factory=object(),  # type: ignore[arg-type]
        planner=object(),  # type: ignore[arg-type]
        watcher=object(),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        events=events,  # type: ignore[arg-type]
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        attempt=1,
        create_semaphore=None,
        submit_semaphore=None,
    )

    assert result.status == BatchTaskStatus.SUBMITTED
    assert coordinator.release_calls == 0
    assert events.states[-1].cleanup_status == "deferred_controller_reconcile"
    assert any(
        event.get("message") == BatchAttemptEventMessage.ATTEMPT_CLEANUP_FINISHED
        and event.get("details", {}).get("cleanup_status") == "deferred_controller_reconcile"
        for event in events.events
    )


@pytest.mark.asyncio
async def test_execute_attempt_does_not_refresh_prepared_lease_before_submit_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = BatchManifest(
        batch_id="prepared-submit-gate",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="done")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    coordinator = _SingleLeaseCoordinator()
    prepared_manager = _AssignedPreparedManager()

    async def fake_submit_to_controller(**kwargs: object) -> BatchTaskState:
        assert kwargs.get("before_submit") is None
        assert prepared_manager.refresh_calls == 0
        task = kwargs["task"]
        assert isinstance(task, BatchTaskSpec)
        return BatchTaskState(
            task_id=task.task_id,
            attempts=1,
            status=BatchTaskStatus.SUBMITTED,
            controller_task_id="controller-task",
            trace_id="trace",
        )

    async def forbidden_release_with_telemetry(**kwargs: object) -> object:
        _ = kwargs
        raise AssertionError("submitted controller task must keep prepared lease until reconcile")

    monkeypatch.setattr(attempt_helpers_module, "submit_to_controller", fake_submit_to_controller)
    monkeypatch.setattr(attempt_helpers_module, "release_with_telemetry", forbidden_release_with_telemetry)

    result = await _execute_attempt(
        coordinator=coordinator,  # type: ignore[arg-type]
        binding_factory=object(),  # type: ignore[arg-type]
        planner=object(),  # type: ignore[arg-type]
        watcher=object(),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        events=_NullEvents(),  # type: ignore[arg-type]
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        attempt=1,
        create_semaphore=None,
        submit_semaphore=asyncio.Semaphore(1),
        prepared_manager=prepared_manager,  # type: ignore[arg-type]
    )

    assert result.status == BatchTaskStatus.SUBMITTED
    assert prepared_manager.refresh_calls == 0
    assert prepared_manager.deferred_startup_calls == 1
    assert prepared_manager.cleanup_deferred_calls == 1
    assert coordinator.acquire_calls == 0


@pytest.mark.asyncio
async def test_execute_attempt_defers_cleanup_for_unknown_controller_reconcile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = BatchManifest(
        batch_id="submit-reconcile-unknown",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="done")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    coordinator = _SingleLeaseCoordinator()

    async def fake_submit_to_controller(**kwargs: object) -> BatchTaskState:
        task = kwargs["task"]
        assert isinstance(task, BatchTaskSpec)
        return BatchTaskState(
            task_id=task.task_id,
            attempts=1,
            status=BatchTaskStatus.FAILED,
            controller_task_id="controller-task",
            trace_id="trace",
            last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
            last_reconciled_status="unknown",
        )

    async def forbidden_release_with_telemetry(**kwargs: object) -> object:
        _ = kwargs
        raise AssertionError("unknown controller reconcile must not release sandbox")

    events = _NullEvents()
    monkeypatch.setattr(attempt_helpers_module, "submit_to_controller", fake_submit_to_controller)
    monkeypatch.setattr(attempt_helpers_module, "release_with_telemetry", forbidden_release_with_telemetry)

    result = await _execute_attempt(
        coordinator=coordinator,  # type: ignore[arg-type]
        binding_factory=object(),  # type: ignore[arg-type]
        planner=object(),  # type: ignore[arg-type]
        watcher=object(),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        events=events,  # type: ignore[arg-type]
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        attempt=1,
        create_semaphore=None,
        submit_semaphore=None,
    )

    assert result.last_reconciled_status == "unknown"
    assert coordinator.release_calls == 0
    assert events.states[-1].cleanup_status == "deferred_controller_reconcile"
    assert any(
        event.get("message") == BatchAttemptEventMessage.ATTEMPT_CLEANUP_FINISHED
        and event.get("details", {}).get("cleanup_status") == "deferred_controller_reconcile"
        for event in events.events
    )


@pytest.mark.asyncio
async def test_execute_attempt_releases_current_lease_for_detached_duplicate_reconcile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = BatchManifest(
        batch_id="detached-duplicate",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="done")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    coordinator = _SingleLeaseCoordinator()

    async def fake_submit_to_controller(**kwargs: object) -> BatchTaskState:
        task = kwargs["task"]
        assert isinstance(task, BatchTaskSpec)
        return BatchTaskState(
            task_id=task.task_id,
            attempts=1,
            status=BatchTaskStatus.SUBMITTED,
            controller_task_id="controller-task",
            trace_id="trace",
            cleanup_status="release_current_lease_after_duplicate_reconcile",
        )

    events = _NullEvents()
    monkeypatch.setattr(attempt_helpers_module, "submit_to_controller", fake_submit_to_controller)

    result = await _execute_attempt(
        coordinator=coordinator,  # type: ignore[arg-type]
        binding_factory=object(),  # type: ignore[arg-type]
        planner=object(),  # type: ignore[arg-type]
        watcher=object(),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        events=events,  # type: ignore[arg-type]
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        attempt=1,
        create_semaphore=None,
        submit_semaphore=None,
    )

    assert result.cleanup_status == "release_current_lease_after_duplicate_reconcile"
    assert coordinator.release_calls == 1
    assert events.states[-1].cleanup_status == "ok"


@pytest.mark.asyncio
async def test_execute_attempt_finishes_prepared_release_mark_when_cancelled_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = BatchManifest(
        batch_id="prepared-release-cancel",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="done")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    coordinator = _SingleLeaseCoordinator()
    prepared_manager = _BlockingPreparedReleaseManager()

    async def fake_submit_to_controller(**kwargs: object) -> BatchTaskState:
        task = kwargs["task"]
        assert isinstance(task, BatchTaskSpec)
        return BatchTaskState(task_id=task.task_id, attempts=1, status=BatchTaskStatus.SUCCEEDED)

    async def fake_release_with_telemetry(**kwargs: object) -> CleanupResult:
        _ = kwargs
        return CleanupResult(ok=True, cleanup_status="ok")

    monkeypatch.setattr(attempt_helpers_module, "submit_to_controller", fake_submit_to_controller)
    monkeypatch.setattr(attempt_helpers_module, "release_with_telemetry", fake_release_with_telemetry)

    attempt_task = asyncio.create_task(
        _execute_attempt(
            coordinator=coordinator,  # type: ignore[arg-type]
            binding_factory=object(),  # type: ignore[arg-type]
            planner=object(),  # type: ignore[arg-type]
            watcher=object(),  # type: ignore[arg-type]
            config=BatchRunConfig(manifest=manifest),
            task=manifest.tasks[0],
            events=_NullEvents(),  # type: ignore[arg-type]
            metrics=BatchMetrics(),
            output_dir=tmp_path,
            attempt=1,
            create_semaphore=None,
            submit_semaphore=None,
            prepared_manager=prepared_manager,  # type: ignore[arg-type]
        )
    )
    await asyncio.wait_for(prepared_manager.mark_started.wait(), timeout=1.0)
    attempt_task.cancel()
    await asyncio.sleep(0)
    attempt_task.cancel()
    await asyncio.sleep(0)

    assert prepared_manager.released_calls == 1
    assert not prepared_manager.mark_finished.is_set()
    prepared_manager.mark_can_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await attempt_task

    assert prepared_manager.mark_finished.is_set()


@pytest.mark.asyncio
async def test_backpressure_defer_releases_claim_without_attempt_failure(tmp_path: Path) -> None:
    manifest = BatchManifest(
        batch_id="prepared-pressure-defer",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="done")],
    )
    config = BatchRunConfig(manifest=manifest, output_dir=tmp_path, concurrency=1)
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    claim = repository.claim_next(
        lease_owner=config.runner_id,
        policy=LeasePolicy.from_timeout(60),
        max_attempts_by_task={"case-1": 1},
    )
    assert claim is not None
    scheduler = BatchScheduler(
        provider_registry=SandboxProviderRegistry(),
        executor=_DeferringExecutor(),  # type: ignore[arg-type]
    )
    metrics = BatchMetrics()
    events = _NullEvents()

    state = await scheduler._run_claimed(
        config=config,
        task=manifest.tasks[0],
        claim=claim,
        repository=repository,
        metrics=metrics,
        events=events,  # type: ignore[arg-type]
        output_dir=tmp_path,
        policy=LeasePolicy.from_timeout(60),
        create_semaphore=None,
        submit_semaphore=None,
        status_semaphore=None,
        prepared_manager=None,
        backpressure_gate=None,
    )

    current = repository.get("case-1")
    assert state.status == BatchTaskStatus.PENDING
    assert state.attempts == 0
    assert current is not None
    assert current.status == BatchTaskStatus.PENDING
    assert current.attempts == 0
    assert metrics.attempt_failure_categories == {}
    assert not any(event.get("message") == "attempt_failed" for event in events.events)


@pytest.mark.asyncio
async def test_backpressure_stop_callback_does_not_read_repository_flags_on_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = BatchManifest(
        batch_id="backpressure-stop-cache",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="done")],
    )
    config = BatchRunConfig(manifest=manifest, output_dir=tmp_path, concurrency=1)
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    claim = repository.claim_next(
        lease_owner=config.runner_id,
        policy=LeasePolicy.from_timeout(60),
        max_attempts_by_task={"case-1": 1},
    )
    assert claim is not None

    loop_thread_id = threading.get_ident()
    control_flag_thread_ids: list[int] = []

    def record_control_flags() -> dict[str, bool]:
        control_flag_thread_ids.append(threading.get_ident())
        return {"cancel_requested": False}

    monkeypatch.setattr(repository, "control_flags", record_control_flags)
    executor = _StopCallbackExecutor()
    scheduler = BatchScheduler(
        provider_registry=SandboxProviderRegistry(),
        executor=executor,  # type: ignore[arg-type]
    )

    state = await scheduler._run_claimed(
        config=config,
        task=manifest.tasks[0],
        claim=claim,
        repository=repository,
        metrics=BatchMetrics(),
        events=_NullEvents(),  # type: ignore[arg-type]
        output_dir=tmp_path,
        policy=LeasePolicy.from_timeout(60),
        create_semaphore=None,
        submit_semaphore=None,
        status_semaphore=None,
    )

    assert executor.stop_callback_checked
    assert state.status == BatchTaskStatus.SUCCEEDED
    assert control_flag_thread_ids
    assert all(thread_id != loop_thread_id for thread_id in control_flag_thread_ids)


@pytest.mark.asyncio
async def test_scheduler_claims_ready_prepared_while_backpressured(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_BACKPRESSURE_SLEEP_SECONDS", "0.1")
    monkeypatch.setenv("NORTHCUA_BATCH_BACKPRESSURE_SAMPLE_INTERVAL_SECONDS", "0.1")
    monkeypatch.setattr(scheduler_module, "_CLAIM_LOOP_STATUS_LOG_INTERVAL_SECONDS", 0.0)
    caplog.set_level(logging.INFO, logger=scheduler_module.__name__)
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    manifest = BatchManifest(
        batch_id="prepared-ready-only-pressure",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="one"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="two"),
            BatchTaskSpec(task_id="case-3", agent_kind="desktop", query="three"),
        ],
    )
    config = BatchRunConfig(
        manifest=manifest,
        output_dir=tmp_path,
        concurrency=1,
        prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=3),
        telemetry_controller_db_path=controller_db,
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    executor = _RecordingExecutor(controller_db)
    scheduler = BatchScheduler(
        provider_registry=SandboxProviderRegistry(),
        executor=executor,  # type: ignore[arg-type]
    )

    async def fake_prepared_manager(**kwargs: object) -> "_ReadyOnlyPreparedManager":
        return _ReadyOnlyPreparedManager()

    monkeypatch.setattr(scheduler, "_maybe_start_prepared_manager", fake_prepared_manager)

    async def recover_controller() -> None:
        await asyncio.sleep(0.2)
        for value in (100, 200, 300):
            _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)

    recovery = asyncio.create_task(recover_controller())
    states = await asyncio.wait_for(
        scheduler.run(
            config=config,
            repository=repository,
            events=_NullEvents(),  # type: ignore[arg-type]
            metrics=BatchMetrics(),
            output_dir=tmp_path,
        ),
        timeout=2,
    )
    await recovery

    assert executor.task_ids[0] == "case-2"
    assert "case-2" in executor.task_ids
    assert {state.task_id for state in states if state.status == BatchTaskStatus.SUCCEEDED} == {
        "case-1",
        "case-2",
        "case-3",
    }


@pytest.mark.asyncio
async def test_scheduler_releases_stale_ready_prepared_snapshot_while_backpressured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_BACKPRESSURE_SLEEP_SECONDS", "0.1")
    monkeypatch.setenv("NORTHCUA_BATCH_BACKPRESSURE_SAMPLE_INTERVAL_SECONDS", "0.1")
    controller_db = tmp_path / "controller.sqlite"
    _controller_events_table(controller_db)
    for value in (13_000, 14_000, 15_000):
        _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)
    manifest = BatchManifest(
        batch_id="prepared-stale-ready-pressure",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="one"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="two"),
        ],
    )
    config = BatchRunConfig(
        manifest=manifest,
        output_dir=tmp_path,
        concurrency=1,
        prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=2),
        telemetry_controller_db_path=controller_db,
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    executor = _RecordingExecutor(controller_db)
    scheduler = BatchScheduler(
        provider_registry=SandboxProviderRegistry(),
        executor=executor,  # type: ignore[arg-type]
    )
    manager = _StaleReadyPreparedManager()

    async def fake_prepared_manager(**kwargs: object) -> "_StaleReadyPreparedManager":
        _ = kwargs
        return manager

    monkeypatch.setattr(scheduler, "_maybe_start_prepared_manager", fake_prepared_manager)

    async def recover_controller() -> None:
        await asyncio.sleep(0.2)
        for value in (100, 200, 300):
            _insert_runtime_event(controller_db, event_loop_lag_ms=value, recv_q_max=0)

    recovery = asyncio.create_task(recover_controller())
    states = await asyncio.wait_for(
        scheduler.run(
            config=config,
            repository=repository,
            events=_NullEvents(),  # type: ignore[arg-type]
            metrics=BatchMetrics(),
            output_dir=tmp_path,
        ),
        timeout=2,
    )
    await recovery

    assert manager.is_ready_checks >= 1
    assert executor.task_ids[0] == "case-1"
    assert {state.task_id for state in states if state.status == BatchTaskStatus.SUCCEEDED} == {"case-1", "case-2"}


@pytest.mark.asyncio
async def test_duplicate_controller_task_id_reconciles_matching_terminal_snapshot() -> None:
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-task",
        trace_id="trace",
        session_id="session-controller-task",
        agent_kind="desktop",
        state=TaskState.FINISHED,
        device_id="device",
        metadata={
            "batch_id": "batch",
            "batch_task_id": "case-1",
            "batch_attempt": 1,
        },
        result={"ok": True},
    )
    watcher = _DuplicateTaskWatcher(snapshot)
    events = _NullEvents()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=watcher,  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,
    )

    assert result.status == BatchTaskStatus.SUCCEEDED
    assert result.controller_task_id == "controller-task"
    assert result.trace_id == "trace"
    assert watcher.submit_calls == 1
    assert watcher.status_calls == 1
    assert not any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_FAILED_BEFORE_TERMINAL
        for event in events.events
    )
    assert any(event.get("details", {}).get("source") == "duplicate_task_id_reconcile" for event in events.events)


@pytest.mark.asyncio
async def test_duplicate_controller_task_id_uses_readable_local_db_miss_without_websocket(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    watcher = _DuplicateTaskWatcher(
        TaskStatusSnapshot(
            task_id="controller-task",
            trace_id="trace",
            session_id="session-controller-task",
            agent_kind="desktop",
            state=TaskState.FINISHED,
            metadata={
                "batch_id": "batch",
                "batch_task_id": "case-1",
                "batch_attempt": 1,
            },
        )
    )

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=watcher,  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest, telemetry_controller_db_path=controller_db),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=_NullEvents(),
    )

    assert result.status == BatchTaskStatus.FAILED
    assert "task_id already exists" in result.error
    assert result.controller_task_id == "controller-task"
    assert result.trace_id == "trace"
    assert result.last_reconciled_status == "unknown"
    assert result.next_attempt_after > time.time()
    assert watcher.submit_calls == 1
    assert watcher.status_calls == 0

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(result)
    assert (
        repository.claim_next(
            lease_owner="runner",
            policy=LeasePolicy.from_timeout(60),
            max_attempts_by_task={"case-1": 2},
        )
        is None
    )
    assert repository.upsert_state(result.model_copy(update={"next_attempt_after": time.time() - 1}))
    claim = repository.claim_next(
        lease_owner="runner",
        policy=LeasePolicy.from_timeout(60),
        max_attempts_by_task={"case-1": 2},
    )
    assert claim is not None
    assert claim.state.controller_task_id == "controller-task"
    assert claim.state.last_reconciled_status == "unknown"


@pytest.mark.asyncio
async def test_controller_submit_connection_refused_is_retryable_failed_not_active() -> None:
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    events = _NullEvents()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=_ConnectionRefusedWatcher(),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,
    )

    assert result.status == BatchTaskStatus.FAILED
    assert result.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert result.controller_task_id == "controller-task"
    assert any(state.status == BatchTaskStatus.SUBMITTED for state in events.states)
    assert any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_FAILED_BEFORE_TERMINAL
        and event.get("details", {}).get("failure_category") == FailureCategory.CONTROLLER_SUBMIT.value
        for event in events.events
    )


@pytest.mark.asyncio
async def test_duplicate_controller_task_id_requires_matching_batch_metadata() -> None:
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-task",
        trace_id="trace",
        session_id="session-controller-task",
        agent_kind="desktop",
        state=TaskState.FINISHED,
        metadata={
            "batch_id": "batch",
            "batch_task_id": "other-case",
            "batch_attempt": 1,
        },
    )
    watcher = _DuplicateTaskWatcher(snapshot)
    events = _NullEvents()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=watcher,  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,
    )

    assert result.status == BatchTaskStatus.FAILED
    assert "task_id already exists" in result.error
    assert watcher.submit_calls == 1
    assert watcher.status_calls == 1
    assert any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_FAILED_BEFORE_TERMINAL
        for event in events.events
    )


@pytest.mark.asyncio
async def test_duplicate_controller_task_id_rejects_missing_batch_metadata() -> None:
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-task",
        trace_id="trace",
        session_id="session-controller-task",
        agent_kind="desktop",
        state=TaskState.FINISHED,
        metadata={"batch_id": "batch"},
    )
    watcher = _DuplicateTaskWatcher(snapshot)
    events = _NullEvents()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=watcher,  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,
    )

    assert result.status == BatchTaskStatus.FAILED
    assert "task_id already exists" in result.error
    assert watcher.submit_calls == 1
    assert watcher.status_calls == 1
    assert any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_FAILED_BEFORE_TERMINAL
        for event in events.events
    )


@pytest.mark.asyncio
async def test_duplicate_controller_task_id_rejects_missing_trace_or_agent_kind() -> None:
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    for trace_id, agent_kind in (("", "desktop"), ("trace", "")):
        watcher = _DuplicateTaskWatcher(
            TaskStatusSnapshot(
                task_id="controller-task",
                trace_id=trace_id,
                session_id="session-controller-task",
                agent_kind=agent_kind,
                state=TaskState.FINISHED,
                metadata={
                    "batch_id": "batch",
                    "batch_task_id": "case-1",
                    "batch_attempt": 1,
                },
            )
        )

        result = await submit_to_controller(
            planner=_Planner(),
            watcher=watcher,  # type: ignore[arg-type]
            config=BatchRunConfig(manifest=manifest),
            task=manifest.tasks[0],
            lease=_lease(),
            controller_url="ws://controller",
            token="token",
            metrics=BatchMetrics(),
            state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
            events=_NullEvents(),
        )

        assert result.status == BatchTaskStatus.FAILED
        assert "task_id already exists" in result.error
        assert watcher.submit_calls == 1
        assert watcher.status_calls == 1


@pytest.mark.asyncio
async def test_duplicate_controller_task_id_preserves_non_terminal_for_later_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_ACCEPT_RECONCILE_POLL_SECONDS", "0.01")
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", timeout_seconds=0.02)],
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-task",
        trace_id="trace",
        session_id="session-controller-task",
        agent_kind="desktop",
        state=TaskState.RUNNING,
        device_id="device",
        metadata={
            "batch_id": "batch",
            "batch_task_id": "case-1",
            "batch_attempt": 1,
            "sandbox_id": "sandbox",
        },
    )
    events = _NullEvents()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=_DuplicateTaskWatcher(snapshot),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest, infra_retry_backoff_seconds=[0]),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,  # type: ignore[arg-type]
    )

    assert result.status == BatchTaskStatus.SUBMITTED
    assert result.controller_task_id == "controller-task"
    assert result.session_id == "session-controller-task"
    assert result.error == ""
    assert result.last_reconciled_status == "running"
    assert result.next_attempt_after == 0
    assert any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_EXISTING_NON_TERMINAL_RECONCILED
        for event in events.events
    )


@pytest.mark.asyncio
async def test_duplicate_controller_task_id_marks_detached_current_lease_releasable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_ACCEPT_RECONCILE_POLL_SECONDS", "0.01")
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", timeout_seconds=0.02)],
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-task",
        trace_id="trace",
        session_id="session-controller-task",
        agent_kind="desktop",
        state=TaskState.RUNNING,
        device_id="old-device",
        metadata={
            "batch_id": "batch",
            "batch_task_id": "case-1",
            "batch_attempt": 1,
            "sandbox_id": "old-sandbox",
        },
    )
    events = _NullEvents()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=_DuplicateTaskWatcher(snapshot),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest, infra_retry_backoff_seconds=[0]),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,  # type: ignore[arg-type]
    )

    assert result.status == BatchTaskStatus.SUBMITTED
    assert result.controller_task_id == "controller-task"
    assert result.cleanup_status == "release_current_lease_after_duplicate_reconcile"
    assert any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_EXISTING_NON_TERMINAL_RECONCILED
        and event.get("details", {}).get("defer_cleanup") is False
        for event in events.events
    )


@pytest.mark.asyncio
async def test_duplicate_controller_task_id_without_lease_identity_releases_current_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_ACCEPT_RECONCILE_POLL_SECONDS", "0.01")
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", timeout_seconds=0.02)],
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-task",
        trace_id="trace",
        session_id="session-controller-task",
        agent_kind="desktop",
        state=TaskState.RUNNING,
        metadata={
            "batch_id": "batch",
            "batch_task_id": "case-1",
            "batch_attempt": 1,
        },
    )

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=_DuplicateTaskWatcher(snapshot),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest, infra_retry_backoff_seconds=[0]),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=_NullEvents(),  # type: ignore[arg-type]
    )

    assert result.status == BatchTaskStatus.SUBMITTED
    assert result.cleanup_status == "release_current_lease_after_duplicate_reconcile"


@pytest.mark.asyncio
async def test_submit_accept_timeout_waits_for_existing_running_controller_task_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_ACCEPT_RECONCILE_POLL_SECONDS", "0.01")
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_ACCEPT_NONTERMINAL_RECONCILE_WAIT_SECONDS", "1")
    calls = 0

    async def local_statuses(*, controller_db_path: object, task_id: str) -> list[TaskStatusSnapshot]:
        nonlocal calls
        _ = controller_db_path
        assert task_id == "controller-task"
        calls += 1
        if calls == 1:
            return [_controller_snapshot(TaskState.RUNNING)]
        if calls == 2:
            raise TimeoutError("transient local controller DB timeout")
        return [_controller_snapshot(TaskState.FINISHED, result={"ok": True})]

    monkeypatch.setattr(attempt_reconcile_module, "_local_duplicate_statuses", local_statuses)
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    events = _NullEvents()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=_SubmitAcceptTimeoutWatcher(),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,  # type: ignore[arg-type]
    )

    assert result.status == BatchTaskStatus.SUCCEEDED
    assert result.controller_task_id == "controller-task"
    assert result.session_id == "session-controller-task"
    assert result.device_id == "device"
    assert result.error == ""
    assert result.last_reconciled_status == "finished"
    assert calls == 3
    assert not any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_FAILED_BEFORE_TERMINAL
        for event in events.events
    )
    assert any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_FINISHED
        and event.get("details", {}).get("source") == "submit_accept_timeout_reconcile"
        for event in events.events
    )


@pytest.mark.asyncio
async def test_submit_accept_timeout_reconciles_existing_terminal_controller_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def local_statuses(*, controller_db_path: object, task_id: str) -> list[TaskStatusSnapshot]:
        _ = controller_db_path
        assert task_id == "controller-task"
        return [_controller_snapshot(TaskState.FINISHED, result={"ok": True})]

    monkeypatch.setattr(attempt_reconcile_module, "_local_duplicate_statuses", local_statuses)
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    events = _NullEvents()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=_SubmitAcceptTimeoutWatcher(),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,  # type: ignore[arg-type]
    )

    assert result.status == BatchTaskStatus.SUCCEEDED
    assert result.last_reconciled_status == "finished"
    assert not any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_FAILED_BEFORE_TERMINAL
        for event in events.events
    )


@pytest.mark.asyncio
async def test_submit_accept_timeout_unknown_local_status_delays_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def local_statuses(*, controller_db_path: object, task_id: str) -> None:
        _ = controller_db_path
        assert task_id == "controller-task"
        return None

    monkeypatch.setattr(attempt_reconcile_module, "_local_duplicate_statuses", local_statuses)
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it")],
    )
    events = _NullEvents()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=_SubmitAcceptTimeoutWatcher(),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest, infra_retry_backoff_seconds=[1]),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,  # type: ignore[arg-type]
    )

    assert result.status == BatchTaskStatus.FAILED
    assert result.controller_task_id == "controller-task"
    assert result.trace_id == "trace"
    assert result.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert result.last_reconciled_status == "unknown"
    assert result.next_attempt_after > time.time()
    assert "will reconcile before fresh submit" in result.error
    assert any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_STATUS_DB_LOOKUP_FINISHED
        and event.get("details", {}).get("source") == "submit_accept_timeout_reconcile"
        and event.get("details", {}).get("timed_out_or_unknown") is True
        for event in events.events
    )
    assert not any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_FAILED_BEFORE_TERMINAL
        for event in events.events
    )


@pytest.mark.asyncio
async def test_submit_accept_timeout_keeps_existing_running_task_reconcile_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_ACCEPT_RECONCILE_POLL_SECONDS", "0.01")

    async def local_statuses(*, controller_db_path: object, task_id: str) -> list[TaskStatusSnapshot]:
        _ = controller_db_path
        assert task_id == "controller-task"
        return [_controller_snapshot(TaskState.RUNNING)]

    monkeypatch.setattr(attempt_reconcile_module, "_local_duplicate_statuses", local_statuses)
    manifest = BatchManifest(
        batch_id="batch",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", timeout_seconds=0.02)],
    )
    events = _NullEvents()

    result = await submit_to_controller(
        planner=_Planner(),
        watcher=_SubmitAcceptTimeoutWatcher(),  # type: ignore[arg-type]
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        lease=_lease(),
        controller_url="ws://controller",
        token="token",
        metrics=BatchMetrics(),
        state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
        events=events,  # type: ignore[arg-type]
    )

    assert result.status == BatchTaskStatus.SUBMITTED
    assert result.controller_task_id == "controller-task"
    assert result.last_reconciled_status == "running"
    assert result.next_attempt_after == 0
    assert not any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_FAILED_BEFORE_TERMINAL
        for event in events.events
    )
    assert any(
        event.get("message") == BatchAttemptEventMessage.CONTROLLER_SUBMIT_EXISTING_NON_TERMINAL_RECONCILED
        and event.get("details", {}).get("source") == "submit_accept_timeout_reconcile"
        for event in events.events
    )


def _controller_snapshot(
    state: TaskState,
    *,
    result: dict[str, object] | None = None,
    error: str = "",
) -> TaskStatusSnapshot:
    return TaskStatusSnapshot(
        task_id="controller-task",
        trace_id="trace",
        session_id="session-controller-task",
        agent_kind="desktop",
        state=state,
        device_id="device",
        metadata={
            "batch_id": "batch",
            "batch_task_id": "case-1",
            "batch_attempt": 1,
        },
        result=result,
        error=error,
    )


class _Planner:
    def plan(self, *args: object, **kwargs: object) -> PlannedTaskSubmission:
        return PlannedTaskSubmission(
            agent_kind="desktop",
            request="do it",
            target_device_id="device",
            required_capabilities=set(),
            input_artifacts=[],
            metadata={
                "batch_id": "batch",
                "batch_task_id": "case-1",
                "batch_attempt": 1,
            },
            controller_task_id="controller-task",
            trace_id="trace",
        )


class _StatusWatcher:
    def __init__(self, snapshots: list[TaskStatusSnapshot]) -> None:
        self.snapshots = snapshots
        self.calls = 0

    async def statuses(self, **kwargs: object) -> list[TaskStatusSnapshot]:
        self.calls += 1
        return list(self.snapshots)


class _RetryableTerminalReconciler:
    async def reconcile(self, *, existing: BatchTaskState, **kwargs: object) -> BatchTaskState | None:
        _ = kwargs
        if not existing.controller_task_id:
            return None
        return existing.model_copy(
            update={
                "status": BatchTaskStatus.FAILED,
                "error": "old controller task interrupted",
                "last_failure_category": FailureCategory.CONTROLLER_SUBMIT,
                "last_reconciled_status": "interrupted",
                "terminal_state": TaskState.INTERRUPTED,
            }
        )


class _OneShotPressureGate:
    sleep_seconds = 0.01

    def __init__(self) -> None:
        self.snapshot_calls = 0

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def active_snapshot(self, *, scope: str) -> object:
        _ = scope
        self.snapshot_calls += 1
        return type("BackpressureSnapshot", (), {"active": self.snapshot_calls == 1})()

    async def wait_if_needed(self, *, scope: str, stop_requested: object = None) -> bool:
        _ = scope, stop_requested
        return True


class _FreshRetryExecutor:
    def __init__(self, gate: _OneShotPressureGate) -> None:
        self.gate = gate
        self.calls = 0
        self.snapshot_calls_at_submit: list[int] = []

    async def run_task(self, **kwargs: object) -> BatchTaskState:
        self.calls += 1
        self.snapshot_calls_at_submit.append(self.gate.snapshot_calls)
        claim = kwargs["claim"]
        assert getattr(claim, "task_id") == "case-1"
        state = getattr(claim, "state")
        assert state.status == BatchTaskStatus.FAILED
        assert not state.controller_task_id
        return state.model_copy(update={"status": BatchTaskStatus.SUCCEEDED})


class _NoSubmitWatcher:
    def __init__(self) -> None:
        self.called = 0

    async def submit_and_wait(self, *args: object, **kwargs: object) -> object:
        self.called += 1
        raise AssertionError("submit_and_wait should not be called")


class _DelayingSubmitBackpressureGate:
    def __init__(self, *, recently_active: bool = False) -> None:
        self.scopes: list[str] = []
        self._recently_active = recently_active

    async def wait_if_needed(self, *, scope: str, stop_requested: object = None) -> bool:
        self.scopes.append(scope)
        await asyncio.sleep(0)
        return True

    def cached_snapshot(self) -> object:
        return type(
            "Snapshot",
            (),
            {
                "active": False,
                "reason": "",
                "samples": 0,
                "latest_event_loop_lag_ms": 0.0,
                "latest_recv_q_max": 0.0,
                "sample_duration_seconds": 0.0,
                "sample_timed_out": False,
                "sample_error": "",
            },
        )()

    def recently_active(self, *, window_seconds: float) -> bool:
        return self._recently_active


class _SuccessfulWatcher:
    async def submit_and_wait(self, *args: object, **kwargs: object) -> tuple[TaskAccepted, TaskFinished]:
        planned = kwargs["planned"]
        accepted = TaskAccepted(
            task_id=str(getattr(planned, "controller_task_id")),
            trace_id=str(getattr(planned, "trace_id")),
            session_id="session-controller-task",
        )
        before_submit = kwargs.get("before_submit")
        if before_submit is not None:
            await before_submit()
        on_submit_open_ready = kwargs.get("on_submit_open_ready")
        if on_submit_open_ready is not None:
            await on_submit_open_ready(
                {
                    "planned_submit_jitter_seconds": 0.0,
                    "actual_submit_jitter_sleep_seconds": 0.0,
                    "submit_jitter_overshoot_seconds": 0.0,
                }
            )
        on_request_sent = kwargs.get("on_request_sent")
        if on_request_sent is not None:
            on_request_sent({"connection_open_elapsed_seconds": 0.0})
        on_accepted = kwargs.get("on_accepted")
        if on_accepted is not None:
            on_accepted(accepted, 0.123)
        return accepted, TaskFinished(
            task_id=accepted.task_id,
            trace_id=accepted.trace_id,
            session_id=accepted.session_id,
            state=TaskState.FINISHED,
            device_id="device",
            lease_id="lease",
            result={"ok": True},
        )


class _DuplicateTaskWatcher:
    def __init__(self, snapshot: TaskStatusSnapshot) -> None:
        self.snapshot = snapshot
        self.submit_calls = 0
        self.status_calls = 0

    async def submit_and_wait(self, *args: object, **kwargs: object) -> object:
        self.submit_calls += 1
        raise RuntimeError("controller rejected task: task_id already exists")

    async def statuses(self, **kwargs: object) -> list[TaskStatusSnapshot]:
        self.status_calls += 1
        assert kwargs["task_ids"] == ["controller-task"]
        return [self.snapshot]


class _ConnectionRefusedWatcher:
    async def submit_and_wait(self, *args: object, **kwargs: object) -> object:
        _ = args, kwargs
        raise ConnectionRefusedError("ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 9192)")


class _SubmitAcceptTimeoutWatcher:
    async def submit_and_wait(self, *args: object, **kwargs: object) -> object:
        before_submit = kwargs.get("before_submit")
        if before_submit is not None:
            await before_submit()
        on_submit_open_ready = kwargs.get("on_submit_open_ready")
        if on_submit_open_ready is not None:
            await on_submit_open_ready({})
        on_request_sent = kwargs.get("on_request_sent")
        if on_request_sent is not None:
            on_request_sent({})
        raise TimeoutError

    async def statuses(self, **kwargs: object) -> list[TaskStatusSnapshot]:
        raise AssertionError("submit ACK timeout should reconcile from local controller DB")


class _FinishingSubmitter:
    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def __aenter__(self) -> "_FinishingSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        task_id = str(kwargs["task_id"])
        trace_id = str(kwargs["trace_id"])
        return TaskAccepted(task_id=task_id, trace_id=trace_id, session_id=f"session-{task_id}")

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        return TaskFinished(
            task_id=task_id,
            trace_id=f"trace-{task_id}",
            session_id=f"session-{task_id}",
            state=TaskState.FINISHED,
            device_id="device",
            lease_id="lease",
            result={"ok": True},
        )


class _NullTimer:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _NullEvents:
    def __init__(self) -> None:
        self.states: list[BatchTaskState] = []
        self.events: list[dict[str, object]] = []
        self.lease = None

    def upsert_task(self, state: BatchTaskState) -> bool:
        self.states.append(state)
        return True

    async def upsert_task_async(self, state: BatchTaskState) -> bool:
        return self.upsert_task(state)

    def for_lease(self, lease: object) -> "_NullEvents":
        self.lease = lease
        return self

    def attempt_event(self, **kwargs: object) -> None:
        self.events.append(kwargs)

    def phase_timer(self, **kwargs: object) -> _NullTimer:
        return _NullTimer()


class _RejectingEvents(_NullEvents):
    async def upsert_task_async(self, state: BatchTaskState) -> bool:
        self.states.append(state)
        return False


def _lease() -> SandboxLease:
    return SandboxLease(
        lease_id="lease",
        handle=SandboxHandle(sandbox_id="sandbox", provider="fake", template="template"),
        device_id="device",
    )


class _PreparedMissManager:
    async def claim(self, **kwargs: object) -> None:
        return None


class _AssignedPreparedManager:
    def __init__(self) -> None:
        now = time.time()
        lease = _lease()
        self.assignment = PreparedSandboxAssignment(
            record=PreparedSandboxRecord(
                task_id="case-1",
                attempt=1,
                status=PreparedSandboxStatus.ASSIGNED,
                runner_id="runner",
                provider=lease.handle.provider,
                template=lease.handle.template,
                sandbox_id=lease.handle.sandbox_id,
                device_id=lease.device_id,
                lease_id=lease.lease_id,
                reserved_at=now - 10,
                prepared_at=now - 5,
                assigned_at=now,
                expires_at=now + 600,
            ),
            lease=lease,
            lease_context=BatchTaskLeaseContext(lease=lease, manager=object()),  # type: ignore[arg-type]
        )
        self.deferred_startup_calls = 0
        self.refresh_calls = 0
        self.cleanup_deferred_calls = 0

    async def claim(self, **kwargs: object) -> PreparedSandboxAssignment:
        _ = kwargs
        return self.assignment

    async def run_deferred_startup(self, **kwargs: object) -> None:
        _ = kwargs
        self.deferred_startup_calls += 1

    async def refresh_assigned_lease_timeout(self, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        self.refresh_calls += 1
        return {"assignment_ttl_guard_passed": True}

    async def mark_cleanup_deferred_async(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs
        self.cleanup_deferred_calls += 1


class _BlockingPreparedReleaseManager:
    def __init__(self) -> None:
        now = time.time()
        lease = _lease()
        self.assignment = PreparedSandboxAssignment(
            record=PreparedSandboxRecord(
                task_id="case-1",
                attempt=1,
                status=PreparedSandboxStatus.ASSIGNED,
                runner_id="runner",
                provider=lease.handle.provider,
                template=lease.handle.template,
                sandbox_id=lease.handle.sandbox_id,
                device_id=lease.device_id,
                lease_id=lease.lease_id,
                reserved_at=now - 10,
                prepared_at=now - 5,
                assigned_at=now,
                expires_at=now + 600,
            ),
            lease=lease,
            lease_context=BatchTaskLeaseContext(lease=lease, manager=object()),  # type: ignore[arg-type]
        )
        self.mark_started = asyncio.Event()
        self.mark_can_finish = asyncio.Event()
        self.mark_finished = asyncio.Event()
        self.released_calls = 0

    async def claim(self, **kwargs: object) -> PreparedSandboxAssignment:
        _ = kwargs
        return self.assignment

    async def run_deferred_startup(self, **kwargs: object) -> None:
        _ = kwargs

    async def refresh_assigned_lease_timeout(self, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        return {"assignment_ttl_guard_passed": True}

    async def mark_released_async(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs
        self.released_calls += 1
        self.mark_started.set()
        await self.mark_can_finish.wait()
        self.mark_finished.set()


class _NoAcquireCoordinator:
    def __init__(self) -> None:
        self.acquire_calls = 0

    async def acquire(self, **kwargs: object) -> object:
        self.acquire_calls += 1
        raise AssertionError("direct create should not run under active backpressure")


class _SingleLeaseCoordinator:
    def __init__(self) -> None:
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire(self, **kwargs: object) -> BatchTaskLeaseContext:
        _ = kwargs
        self.acquire_calls += 1
        return BatchTaskLeaseContext(lease=_lease(), manager=object())  # type: ignore[arg-type]

    async def release(self, **kwargs: object) -> None:
        _ = kwargs
        self.release_calls += 1


class _DeferringExecutor:
    async def run_task(self, **kwargs: object) -> object:
        raise PreparedSandboxBackpressureDeferred("pressure")


class _StopCallbackExecutor:
    def __init__(self) -> None:
        self.stop_callback_checked = False

    async def run_task(self, **kwargs: object) -> BatchTaskState:
        stop_requested = kwargs["backpressure_stop_requested"]
        assert callable(stop_requested)
        result = stop_requested()
        if inspect.isawaitable(result):
            result = await result
        assert result is False
        self.stop_callback_checked = True
        task = kwargs["task"]
        assert isinstance(task, BatchTaskSpec)
        return BatchTaskState(task_id=task.task_id, status=BatchTaskStatus.SUCCEEDED, attempts=1)


class _RecordingExecutor:
    def __init__(self, controller_db: Path) -> None:
        self.controller_db = controller_db
        self.task_ids: list[str] = []

    async def run_task(self, **kwargs: object) -> BatchTaskState:
        task = kwargs["task"]
        assert isinstance(task, BatchTaskSpec)
        claim = kwargs["claim"]
        self.task_ids.append(task.task_id)
        if len(self.task_ids) == 1:
            for value in (100, 200, 300):
                _insert_runtime_event(self.controller_db, event_loop_lag_ms=value, recv_q_max=0)
        return BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.SUCCEEDED,
            attempts=getattr(getattr(claim, "state", None), "attempts", 0) + 1,
        )


class _ReadyOnlyPreparedManager:
    def ready_unassigned_task_attempts(self) -> set[tuple[str, int]]:
        return {("case-2", 1)}

    def is_ready_unassigned(self, *, task_id: str, attempt: int) -> bool:
        return (task_id, attempt) == ("case-2", 1)

    async def is_ready_unassigned_async(self, *, task_id: str, attempt: int) -> bool:
        return self.is_ready_unassigned(task_id=task_id, attempt=attempt)

    def has_foreign_claim_blocker(self, **kwargs: object) -> bool:
        _ = kwargs
        return False

    async def has_foreign_claim_blocker_async(self, **kwargs: object) -> bool:
        return self.has_foreign_claim_blocker(**kwargs)

    async def close(self) -> None:
        return None


class _StaleReadyPreparedManager(_ReadyOnlyPreparedManager):
    def __init__(self) -> None:
        self.is_ready_checks = 0

    async def is_ready_unassigned_async(self, *, task_id: str, attempt: int) -> bool:
        _ = task_id, attempt
        self.is_ready_checks += 1
        return False


def _controller_events_table(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
        connection.execute("CREATE INDEX events_type_created_idx ON events(event_type, created_at)")


def _insert_runtime_event(
    path: Path,
    *,
    event_loop_lag_ms: float,
    recv_q_max: float,
    created_at: float | None = None,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO events(trace_id, entity_type, entity_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "trace",
                "controller",
                "controller",
                "controller_runtime_telemetry",
                json.dumps({"event_loop_lag_ms": event_loop_lag_ms, "recv_q_max": recv_q_max}),
                str(created_at if created_at is not None else time.time()),
            ),
        )
