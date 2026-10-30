"""Worker heartbeat persistence is coalesced and terminal-state safe."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from northcua.controller.execution.processes.contracts import WorkerJobStatus
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua.controller.storage.worker_heartbeat_batcher import WorkerHeartbeatBatcher


def _create_job(store: SQLiteControllerStore, job_id: str) -> None:
    store.create_worker_job(
        job_id=job_id,
        task_id=f"task-{job_id}",
        attempt_id=f"attempt-{job_id}",
        session_id=f"session-{job_id}",
        lease_id=f"lease-{job_id}",
        device_id=f"device-{job_id}",
        agent_kind="desktop",
    )


@pytest.mark.asyncio
async def test_heartbeat_batcher_coalesces_latest_value_per_job(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(db_path)
    _create_job(store, "job-1")
    calls = 0

    async def _call(path: str | Path, operation, **_kwargs):
        nonlocal calls
        calls += 1
        worker_store = SQLiteControllerStore(path)
        try:
            return operation(worker_store)
        finally:
            worker_store.close()

    monkeypatch.setattr(
        "northcua.controller.storage.worker_heartbeat_batcher.controller_store_call",
        _call,
    )
    batcher = WorkerHeartbeatBatcher(db_path, flush_interval_seconds=60)
    for step in range(500):
        batcher.publish(job_id="job-1", worker_pid=42, payload={"step_index": step})
    await batcher.flush()

    row = store.worker_job_row("job-1")
    assert row is not None
    assert calls == 1
    assert row["status"] == WorkerJobStatus.RUNNING.value
    assert json.loads(row["heartbeat_payload_json"])["step_index"] == 499
    store.close()


@pytest.mark.asyncio
async def test_late_heartbeat_cannot_resurrect_terminal_job(tmp_path: Path) -> None:
    db_path = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(db_path)
    _create_job(store, "job-terminal")
    batcher = WorkerHeartbeatBatcher(db_path, flush_interval_seconds=60)
    batcher.publish(job_id="job-terminal", worker_pid=42, payload={"phase": "running"})
    store.mark_worker_job_status(job_id="job-terminal", status=WorkerJobStatus.CANCELED)

    await batcher.flush()

    row = store.worker_job_row("job-terminal")
    assert row is not None
    assert row["status"] == WorkerJobStatus.CANCELED.value
    store.close()


def test_late_worker_status_cannot_replace_terminal_state(tmp_path: Path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    _create_job(store, "job-status-race")
    store.mark_worker_job_status(job_id="job-status-race", status=WorkerJobStatus.RUNNING)
    store.mark_worker_job_status(job_id="job-status-race", status=WorkerJobStatus.CANCELED)

    store.mark_worker_job_status(job_id="job-status-race", status=WorkerJobStatus.RUNNING)
    store.mark_worker_job_status(job_id="job-status-race", status=WorkerJobStatus.FINISHING)
    store.mark_worker_job_status(job_id="job-status-race", status=WorkerJobStatus.FINISHED)

    row = store.worker_job_row("job-status-race")
    assert row is not None
    assert row["status"] == WorkerJobStatus.CANCELED.value
    store.close()


@pytest.mark.asyncio
async def test_discard_waits_for_inflight_batch_and_drops_newer_pending_value(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(db_path)
    _create_job(store, "job-race")
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_call(path: str | Path, operation, **_kwargs):
        started.set()
        await release.wait()
        worker_store = SQLiteControllerStore(path)
        try:
            return operation(worker_store)
        finally:
            worker_store.close()

    monkeypatch.setattr(
        "northcua.controller.storage.worker_heartbeat_batcher.controller_store_call",
        _blocked_call,
    )
    batcher = WorkerHeartbeatBatcher(db_path, flush_interval_seconds=0)
    batcher.publish(job_id="job-race", worker_pid=42, payload={"step_index": 1})
    await started.wait()
    batcher.publish(job_id="job-race", worker_pid=42, payload={"step_index": 2})
    discard = asyncio.create_task(batcher.discard("job-race"))
    await asyncio.sleep(0)
    assert not discard.done()
    release.set()
    await discard
    store.mark_worker_job_status(job_id="job-race", status=WorkerJobStatus.FINISHING)
    await batcher.flush()

    row = store.worker_job_row("job-race")
    assert row is not None
    assert row["status"] == WorkerJobStatus.FINISHING.value
    assert json.loads(row["heartbeat_payload_json"])["step_index"] == 1
    store.close()


@pytest.mark.asyncio
async def test_permanent_store_failure_stops_after_retry_budget(tmp_path: Path, monkeypatch) -> None:
    calls = 0

    async def _failed_call(_path: str | Path, _operation, **_kwargs):
        nonlocal calls
        calls += 1
        raise OSError("persistent sqlite failure")

    monkeypatch.setattr(
        "northcua.controller.storage.worker_heartbeat_batcher.controller_store_call",
        _failed_call,
    )
    batcher = WorkerHeartbeatBatcher(
        tmp_path / "controller.sqlite",
        flush_interval_seconds=0,
        max_retry_attempts=2,
        retry_backoff_seconds=0,
    )
    batcher.publish(job_id="job-failed", worker_pid=42, payload={"phase": "running"})
    task = batcher._flush_task  # noqa: SLF001 - focused retry lifecycle assertion
    assert task is not None
    await task

    assert calls == 2
    assert batcher._flush_task is None  # noqa: SLF001
    assert batcher._pending == {}  # noqa: SLF001
    await batcher.aclose()


@pytest.mark.asyncio
async def test_close_is_shared_and_rejects_new_heartbeats(tmp_path: Path, monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_call(_path: str | Path, _operation, **_kwargs):
        started.set()
        await release.wait()
        return 0

    monkeypatch.setattr(
        "northcua.controller.storage.worker_heartbeat_batcher.controller_store_call",
        _blocked_call,
    )
    batcher = WorkerHeartbeatBatcher(
        tmp_path / "controller.sqlite",
        flush_interval_seconds=0,
        flush_timeout_seconds=0.05,
    )
    batcher.publish(job_id="job-closing", worker_pid=42, payload={"phase": "running"})
    await started.wait()

    first = asyncio.create_task(batcher.aclose())
    await asyncio.sleep(0)
    batcher.publish(job_id="job-too-late", worker_pid=42, payload={"phase": "late"})
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(batcher.aclose())
    release.set()
    await second

    assert batcher._pending == {}  # noqa: SLF001
    assert batcher._closed  # noqa: SLF001


@pytest.mark.asyncio
async def test_discard_deadline_does_not_block_terminal_cleanup(tmp_path: Path, monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_call(_path: str | Path, _operation, **_kwargs):
        started.set()
        await release.wait()
        return 0

    monkeypatch.setattr(
        "northcua.controller.storage.worker_heartbeat_batcher.controller_store_call",
        _blocked_call,
    )
    batcher = WorkerHeartbeatBatcher(
        tmp_path / "controller.sqlite",
        flush_interval_seconds=0,
        discard_timeout_seconds=0.01,
        flush_timeout_seconds=0.01,
    )
    batcher.publish(job_id="job-blocked", worker_pid=42, payload={"phase": "running"})
    await started.wait()

    await asyncio.wait_for(batcher.discard("job-blocked"), timeout=0.1)
    release.set()
    task = batcher._flush_task  # noqa: SLF001
    if task is not None:
        await task
    await batcher.aclose()

    assert batcher._pending == {}  # noqa: SLF001
    assert batcher._flush_task is None  # noqa: SLF001
