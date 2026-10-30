"""Batch resume reconciliation tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time

import pytest

from northcua.batch import operations as batch_operations_module
from northcua.batch.contracts import (
    BatchDefaults,
    BatchManifest,
    BatchRunConfig,
    BatchTaskSpec,
    BatchTaskState,
    BatchTaskStatus,
)
from northcua.batch.metrics import BatchMetrics
from northcua.batch.operations import BatchOperations
from northcua.batch.runner.attempt_submit import _local_status_from_controller_terminal
from northcua.batch.runner.events import BatchEventWriter
from northcua.batch.runner.resume import BatchStatusReconciler, ControllerStatusUnavailable
from northcua.batch.runner.scheduler import BatchScheduler
from northcua.batch.runner.scheduler_helpers import BatchSchedulerHelperMixin
from northcua.batch.runner.stale_active_repair import _cas_update, repair_stale_active_rows_once
from northcua.batch.state import SQLiteBatchStateRepository
from northcua.batch.state.contracts import LeasePolicy
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua.reliability import FailureAction
from northcua.sandboxes.provider.registry import SandboxProviderRegistry
from northcua.telemetry.contracts import FailureCategory
from northcua_protocol.payloads import TaskFinished, TaskRequest, TaskStatusSnapshot
from northcua_protocol.state import TaskState


class _StatusWatcher:
    def __init__(self, *snapshots: TaskStatusSnapshot) -> None:
        self.snapshots = list(snapshots)
        self.calls: list[dict[str, object]] = []

    async def statuses(self, **kwargs: object) -> list[TaskStatusSnapshot]:
        self.calls.append(kwargs)
        if len(self.snapshots) > 1:
            return [self.snapshots.pop(0)]
        return [self.snapshots[0]]


class _FailingStatusWatcher:
    async def statuses(self, **kwargs: object) -> list[TaskStatusSnapshot]:
        raise RuntimeError("controller unavailable")


class _EmptyStatusWatcher:
    async def statuses(self, **kwargs: object) -> list[TaskStatusSnapshot]:
        return []


class _ReturningReconciler:
    def __init__(self, state: BatchTaskState) -> None:
        self.state = state

    async def reconcile(
        self,
        *,
        config: BatchRunConfig,
        task: BatchTaskSpec,
        existing: BatchTaskState,
        status_semaphore: object | None = None,
    ) -> BatchTaskState:
        _ = config, task, existing, status_semaphore
        return self.state


class _CancelingReconciler:
    async def reconcile(
        self,
        *,
        config: BatchRunConfig,
        task: BatchTaskSpec,
        existing: BatchTaskState,
        status_semaphore: object | None = None,
    ) -> BatchTaskState:
        _ = config, task, existing, status_semaphore
        raise asyncio.CancelledError


class _SchedulerHelper(BatchSchedulerHelperMixin):
    def __init__(self, reconciler: _ReturningReconciler) -> None:
        self._resume_reconciler = reconciler

    @staticmethod
    def _total_attempt_budget(*, config: BatchRunConfig, task: BatchTaskSpec) -> int:
        _ = config, task
        return 8


class _NoopRunner:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def run(self, config: BatchRunConfig):
        self.events.append("runner")
        from northcua.batch.contracts import BatchRunResult

        return BatchRunResult(
            batch_id=config.manifest.batch_id,
            total=len(config.manifest.tasks),
            succeeded=0,
            failed=0,
            canceled=0,
            incomplete=0,
            output_dir=str(config.output_dir),
            task_states=[],
        )


class _CancelAfterWaitingDeviceExecutor:
    async def run_task(self, *, task: BatchTaskSpec, events: BatchEventWriter, **kwargs: object) -> BatchTaskState:
        _ = kwargs
        state = BatchTaskState(task_id=task.task_id, attempts=1, status=BatchTaskStatus.WAITING_DEVICE)
        events.upsert_task(state)
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_scheduler_releases_pre_controller_lease_on_cancellation(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="resume me", max_retries=0)
    manifest = BatchManifest(batch_id="cancel-before-submit", tasks=[task])
    config = BatchRunConfig(manifest=manifest, output_dir=tmp_path, lease_timeout_seconds=3600)
    batch_dir = tmp_path / manifest.batch_id
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    policy = LeasePolicy.from_timeout(config.lease_timeout_seconds)
    claim = repository.claim_next(lease_owner=config.runner_id, policy=policy)

    assert claim is not None

    events = BatchEventWriter(
        repository=repository,
        output_dir=batch_dir,
        batch_id=manifest.batch_id,
    ).for_lease(claim)
    scheduler = BatchScheduler(
        provider_registry=SandboxProviderRegistry(),
        executor=_CancelAfterWaitingDeviceExecutor(),
    )

    with pytest.raises(asyncio.CancelledError):
        await scheduler._run_claimed(
            config=config,
            task=task,
            claim=claim,
            repository=repository,
            metrics=BatchMetrics(),
            events=events,
            output_dir=batch_dir,
            policy=policy,
            create_semaphore=None,
            submit_semaphore=None,
            status_semaphore=None,
        )

    with sqlite3.connect(batch_dir / "batch_state.sqlite") as connection:
        row = connection.execute(
            "SELECT status, attempts, retry_budget_remaining, lease_owner, lease_token, lease_expires_at "
            "FROM batch_tasks WHERE task_id=?",
            (task.task_id,),
        ).fetchone()

    assert row == (BatchTaskStatus.WAITING_DEVICE.value, 1, 0, "", "", 0)
    assert repository.claim_next(lease_owner="resume-runner", policy=policy) is not None


@pytest.mark.asyncio
async def test_scheduler_releases_reconcile_lease_on_cancellation(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="reconcile me", max_retries=0)
    manifest = BatchManifest(batch_id="cancel-during-reconcile", tasks=[task])
    config = BatchRunConfig(manifest=manifest, output_dir=tmp_path, lease_timeout_seconds=3600)
    batch_dir = tmp_path / manifest.batch_id
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    existing = BatchTaskState(
        task_id=task.task_id,
        attempts=1,
        status=BatchTaskStatus.SUBMITTED,
        controller_task_id="controller-1",
    )
    assert repository.upsert_state(existing)
    policy = LeasePolicy.from_timeout(config.lease_timeout_seconds)
    claim = repository.claim_next(lease_owner=config.runner_id, policy=policy)

    assert claim is not None

    events = BatchEventWriter(
        repository=repository,
        output_dir=batch_dir,
        batch_id=manifest.batch_id,
    ).for_lease(claim)
    scheduler = BatchScheduler(
        provider_registry=SandboxProviderRegistry(),
        executor=_CancelAfterWaitingDeviceExecutor(),
        resume_reconciler=_CancelingReconciler(),
    )

    with pytest.raises(asyncio.CancelledError):
        await scheduler._run_claimed(
            config=config,
            task=task,
            claim=claim,
            repository=repository,
            metrics=BatchMetrics(),
            events=events,
            output_dir=batch_dir,
            policy=policy,
            create_semaphore=None,
            submit_semaphore=None,
            status_semaphore=None,
        )

    with sqlite3.connect(batch_dir / "batch_state.sqlite") as connection:
        row = connection.execute(
            "SELECT status, controller_task_id, retry_budget_remaining, lease_owner, lease_token, lease_expires_at "
            "FROM batch_tasks WHERE task_id=?",
            (task.task_id,),
        ).fetchone()

    assert row == (BatchTaskStatus.SUBMITTED.value, "controller-1", 0, "", "", 0)
    assert repository.claim_next(lease_owner="resume-runner", policy=policy) is not None


def _force_batch_lease(
    path,
    *,
    task_id: str,
    owner: str,
    token: str = "lease-token",
    expires_at: float = 0.0,
    updated_at: float | None = None,
) -> None:
    timestamp = time.time() if updated_at is None else updated_at
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE batch_tasks
            SET lease_owner=?, lease_token=?, lease_expires_at=?, updated_at=?, generation=generation+1
            WHERE task_id=?
            """,
            (owner, token, expires_at, timestamp, task_id),
        )


def _batch_lease_row(path, task_id: str) -> sqlite3.Row:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT lease_owner, lease_token, lease_expires_at FROM batch_tasks WHERE task_id=?",
            (task_id,),
        ).fetchone()
        assert row is not None
        return row
    finally:
        connection.close()


def _finished_controller_db(path, *, task_id: str = "controller-1", batch_id: str = "resume-repair") -> None:
    store = SQLiteControllerStore(path)
    request = TaskRequest(
        task_id=task_id,
        trace_id="trace-1",
        agent_kind="web",
        request="done",
        metadata={"batch_id": batch_id},
    )
    store.create_task(request, session_id="session-1", submitter_id="submitter")
    store.update_task_status(
        task_id,
        TaskState.FINISHED,
        result_json=TaskFinished(
            task_id=task_id,
            trace_id="trace-1",
            session_id="session-1",
            state=TaskState.FINISHED,
            device_id="device-1",
        ).model_dump_json(),
        finished=True,
    )
    store.close()


@pytest.mark.asyncio
async def test_resume_runs_cleanup_before_runner_when_enabled(tmp_path, monkeypatch) -> None:
    batch_id = "resume-cleanup-order"
    batch_dir = tmp_path / batch_id
    batch_dir.mkdir()
    manifest = BatchManifest(batch_id=batch_id, tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="go")])
    (batch_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite").initialize_manifest(manifest)
    events: list[str] = []

    async def fake_cleanup(**kwargs):
        events.append("cleanup")
        assert kwargs["batch_id"] == batch_id

    monkeypatch.setattr(batch_operations_module, "_run_pre_resume_cleanup", fake_cleanup)
    operations = BatchOperations(runner_factory=lambda: _NoopRunner(events))

    await operations.resume(batch_id=batch_id, output_dir=tmp_path, pre_resume_cleanup=True)

    assert events == ["cleanup", "runner"]


@pytest.mark.asyncio
async def test_resume_cleanup_uses_manifest_default_concurrency_for_preflight_budget(tmp_path, monkeypatch) -> None:
    batch_id = "resume-cleanup-manifest-concurrency"
    batch_dir = tmp_path / batch_id
    batch_dir.mkdir()
    manifest = BatchManifest(
        batch_id=batch_id,
        defaults=BatchDefaults(concurrency=240),
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="go")],
    )
    (batch_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite").initialize_manifest(manifest)
    seen: dict[str, object] = {}

    async def fake_cleanup(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(batch_operations_module, "_run_pre_resume_cleanup", fake_cleanup)
    operations = BatchOperations(runner_factory=lambda: _NoopRunner([]))

    await operations.resume(batch_id=batch_id, output_dir=tmp_path, pre_resume_cleanup=True)

    assert seen["run_concurrency"] == 240


@pytest.mark.asyncio
async def test_resume_cleanup_explicit_concurrency_overrides_manifest_default(tmp_path, monkeypatch) -> None:
    batch_id = "resume-cleanup-explicit-concurrency"
    batch_dir = tmp_path / batch_id
    batch_dir.mkdir()
    manifest = BatchManifest(
        batch_id=batch_id,
        defaults=BatchDefaults(concurrency=240),
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="go")],
    )
    (batch_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite").initialize_manifest(manifest)
    seen: dict[str, object] = {}

    async def fake_cleanup(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(batch_operations_module, "_run_pre_resume_cleanup", fake_cleanup)
    operations = BatchOperations(runner_factory=lambda: _NoopRunner([]))

    await operations.resume(batch_id=batch_id, output_dir=tmp_path, concurrency=123, pre_resume_cleanup=True)

    assert seen["run_concurrency"] == 123


@pytest.mark.asyncio
async def test_resume_can_skip_pre_resume_cleanup(tmp_path, monkeypatch) -> None:
    batch_id = "resume-cleanup-skip"
    batch_dir = tmp_path / batch_id
    batch_dir.mkdir()
    manifest = BatchManifest(batch_id=batch_id, tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="go")])
    (batch_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite").initialize_manifest(manifest)
    events: list[str] = []

    async def fake_cleanup(**kwargs):
        events.append("cleanup")

    monkeypatch.setattr(batch_operations_module, "_run_pre_resume_cleanup", fake_cleanup)
    operations = BatchOperations(runner_factory=lambda: _NoopRunner(events))

    await operations.resume(batch_id=batch_id, output_dir=tmp_path, pre_resume_cleanup=False)

    assert events == ["runner"]


@pytest.mark.asyncio
async def test_resume_applies_explicit_selected_retry_without_mutating_manifest(tmp_path) -> None:
    batch_id = "resume-selected-retry"
    batch_dir = tmp_path / batch_id
    batch_dir.mkdir()
    manifest = BatchManifest(
        batch_id=batch_id,
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=0)],
    )
    manifest_path = batch_dir / "manifest.json"
    manifest_bytes = manifest.model_dump_json(indent=2).encode()
    manifest_path.write_bytes(manifest_bytes)
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="task-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            retry_budget_remaining=0,
            last_failure_category=FailureCategory.MAIN_AGENT,
            error="model provider unavailable",
        )
    )
    repository.set_control_flag("drain_requested", True)
    plan_path = tmp_path / "selected-retry.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plan_id": "retry-provider-failures-v1",
                "batch_id": batch_id,
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "extra_attempts": 2,
                "targets": [
                    {
                        "task_id": "task-1",
                        "expected_status": "failed",
                        "expected_attempts": 1,
                        "expected_retry_budget_remaining": 0,
                        "expected_failure_category": FailureCategory.MAIN_AGENT.value,
                        "expected_error_sha256": hashlib.sha256(b"model provider unavailable").hexdigest(),
                        "max_attempts": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    captured: list[BatchRunConfig] = []

    class _CapturingRunner:
        async def run(self, config: BatchRunConfig):
            captured.append(config)
            return await _NoopRunner([]).run(config)

    await BatchOperations(runner_factory=_CapturingRunner).resume(
        batch_id=batch_id,
        output_dir=tmp_path,
        pre_resume_cleanup=False,
        selected_retry_plan_paths=[plan_path],
    )

    state = repository.get("task-1")
    assert state is not None
    assert state.status == BatchTaskStatus.PENDING
    assert state.attempts == 1
    assert state.retry_budget_remaining == 2
    assert manifest_path.read_bytes() == manifest_bytes
    assert captured[0].manifest.tasks[0].max_retries == 2
    assert captured[0].identity_manifest == manifest
    assert captured[0].identity_manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()


def test_sqlite_manifest_persists_retry_budget_for_claiming(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=2)
    manifest = BatchManifest(batch_id="retry-budget", tasks=[task])
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    repository.upsert_state(BatchTaskState(task_id="task-1", status=BatchTaskStatus.FAILED, attempts=3))

    claim = repository.claim_next(lease_owner="runner", policy=LeasePolicy.from_timeout(30))

    assert claim is None


def test_sqlite_manifest_updates_retry_budget_on_resume(tmp_path) -> None:
    original = BatchManifest(
        batch_id="retry-budget",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=0)],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(original)
    repository.upsert_state(BatchTaskState(task_id="task-1", status=BatchTaskStatus.FAILED, attempts=1))

    updated = BatchManifest(
        batch_id="retry-budget",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=2)],
    )
    repository.initialize_manifest(updated)
    state = repository.get("task-1")
    assert state is not None
    assert state.retry_budget_remaining == 2
    claim = repository.claim_next(lease_owner="runner", policy=LeasePolicy.from_timeout(30))

    assert claim is not None
    assert claim.task_id == "task-1"


@pytest.mark.asyncio
async def test_scheduler_releases_claim_after_delayed_unknown_reconcile(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="resume")
    manifest = BatchManifest(batch_id="resume-delayed-reconcile", tasks=[task])
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    existing = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
    )
    delayed = existing.model_copy(
        update={
            "last_reconciled_status": "unknown",
            "next_attempt_after": time.time() - 1,
        }
    )
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    helper = _SchedulerHelper(_ReturningReconciler(delayed))

    result = await helper._reconcile_existing(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        events=events,
        existing=existing,
    )

    assert result == delayed
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.controller_task_id == "controller-1"
    assert stored.next_attempt_after == delayed.next_attempt_after


@pytest.mark.asyncio
async def test_scheduler_keeps_failed_unknown_controller_id_in_reconcile_loop(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="resume", max_retries=2)
    manifest = BatchManifest(batch_id="resume-failed-unknown-reconcile", tasks=[task])
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    existing = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
    )
    expired_retry_at = time.time() - 1
    delayed = existing.model_copy(
        update={
            "status": BatchTaskStatus.FAILED,
            "last_reconciled_status": "unknown",
            "last_failure_category": FailureCategory.CONTROLLER_SUBMIT,
            "next_attempt_after": expired_retry_at,
        }
    )
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    helper = _SchedulerHelper(_ReturningReconciler(delayed))

    result = await helper._reconcile_existing(
        config=BatchRunConfig(manifest=manifest, infra_retry_backoff_seconds=[60]),
        task=task,
        events=events,
        existing=existing,
    )

    assert result == delayed
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.FAILED
    assert stored.controller_task_id == "controller-1"
    assert stored.last_reconciled_status == "unknown"
    assert stored.next_attempt_after == expired_retry_at


@pytest.mark.asyncio
async def test_scheduler_preserves_nonterminal_reconciled_state_without_retry(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="resume")
    manifest = BatchManifest(batch_id="resume-nonterminal-reconcile", tasks=[task])
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    existing = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
    )
    reconciled = existing.model_copy(
        update={
            "last_reconciled_status": "running",
            "error": "controller task remained non-terminal during resume; will reconcile later",
        }
    )
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    helper = _SchedulerHelper(_ReturningReconciler(reconciled))

    result = await helper._reconcile_existing(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        events=events,
        existing=existing,
    )

    assert result == reconciled
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.SUBMITTED
    assert stored.controller_task_id == "controller-1"
    assert stored.last_reconciled_status == "running"


def test_scheduler_release_claim_preserves_real_cleanup_over_duplicate_reconcile_sentinel(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="resume")
    manifest = BatchManifest(batch_id="cleanup-sentinel", tasks=[task])
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    claim = repository.claim_next(lease_owner="runner", policy=LeasePolicy.from_timeout(30))
    assert claim is not None
    assert repository.upsert_state(
        BatchTaskState(task_id=task.task_id, status=BatchTaskStatus.SUBMITTED, cleanup_status="ok"),
        lease_token=claim.lease_token,
    )
    returned = BatchTaskState(
        task_id=task.task_id,
        status=BatchTaskStatus.SUBMITTED,
        controller_task_id="controller-1",
        cleanup_status="release_current_lease_after_duplicate_reconcile",
    )

    _SchedulerHelper._release_claim(
        repository=repository,
        claim=claim,
        state=returned,
        runner_id="runner",
    )

    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.cleanup_status == "ok"
    assert stored.controller_task_id == "controller-1"


@pytest.mark.asyncio
async def test_scheduler_releases_claim_after_failed_retry_backoff(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="resume")
    manifest = BatchManifest(batch_id="resume-failed-backoff", tasks=[task])
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    existing = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
    )
    delayed = existing.model_copy(
        update={
            "controller_task_id": "",
            "status": BatchTaskStatus.FAILED,
            "last_failure_category": FailureCategory.CONTROLLER_SUBMIT,
            "next_attempt_after": time.time() + 60,
        }
    )
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    helper = _SchedulerHelper(_ReturningReconciler(delayed))

    result = await helper._reconcile_existing(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        events=events,
        existing=existing,
    )

    assert result == delayed
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.FAILED
    assert stored.next_attempt_after == delayed.next_attempt_after


@pytest.mark.asyncio
async def test_scheduler_clears_terminal_controller_id_before_retry_backoff(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="resume")
    manifest = BatchManifest(batch_id="resume-terminal-backoff", tasks=[task])
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    existing = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        status=BatchTaskStatus.TIMED_OUT,
        terminal_state=TaskState.INTERRUPTED,
        terminal_decision={"retryable": True},
        attempts=1,
        retry_budget_remaining=1,
    )
    reconciled = existing.model_copy(
        update={
            "last_reconciled_status": TaskState.INTERRUPTED.value,
            "last_failure_category": FailureCategory.TIMEOUT,
            "next_attempt_after": 0,
        }
    )
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    helper = _SchedulerHelper(_ReturningReconciler(reconciled))
    started = time.time()

    result = await helper._reconcile_existing(
        config=BatchRunConfig(manifest=manifest, infra_retry_backoff_seconds=[30]),
        task=task,
        events=events,
        existing=existing,
    )

    assert result is not None
    assert result.controller_task_id == ""
    assert result.trace_id == ""
    assert result.session_id == ""
    assert result.terminal_state is None
    assert result.terminal_decision == {}
    assert result.next_attempt_after >= started + 30
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.controller_task_id == ""


@pytest.mark.asyncio
async def test_resume_reconciler_promotes_finished_controller_task() -> None:
    manifest = BatchManifest(
        batch_id="resume-api",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="done")],
    )
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="web",
        state=TaskState.FINISHED,
        device_id="device-1",
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.SUCCEEDED
    assert reconciled.trace_id == "trace-1"
    assert reconciled.device_id == "device-1"


@pytest.mark.asyncio
async def test_resume_reconciler_reads_local_controller_db_before_websocket(tmp_path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(controller_db)
    request = TaskRequest(
        task_id="controller-1",
        trace_id="trace-1",
        agent_kind="web",
        request="done",
        metadata={"batch_id": "resume-db"},
    )
    store.create_task(request, session_id="session-1", submitter_id="submitter")
    store.update_task_status(
        "controller-1",
        TaskState.FINISHED,
        result_json=TaskFinished(
            task_id="controller-1",
            trace_id="trace-1",
            session_id="session-1",
            state=TaskState.FINISHED,
            device_id="device-1",
        ).model_dump_json(),
        finished=True,
    )
    store.close()
    manifest = BatchManifest(
        batch_id="resume-db",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="done")],
    )
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
    )

    reconciled = await BatchStatusReconciler(_FailingStatusWatcher()).reconcile(
        config=BatchRunConfig(manifest=manifest, telemetry_controller_db_path=controller_db),
        task=manifest.tasks[0],
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.SUCCEEDED
    assert reconciled.device_id == "device-1"
    assert reconciled.last_reconciled_status == "finished"


@pytest.mark.asyncio
async def test_stale_active_repair_reconciles_terminal_controller_rows_from_local_db(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="done", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-terminal", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            controller_task_id="controller-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
        )
    )
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=time.time() - 1,
    )
    controller_db = tmp_path / "controller.sqlite"
    _finished_controller_db(controller_db, batch_id=manifest.batch_id)

    result = await repair_stale_active_rows_once(
        config=BatchRunConfig(
            manifest=manifest,
            runner_id="new-runner",
            telemetry_controller_db_path=controller_db,
        ),
        repository=repository,
        max_attempts_by_task={task.task_id: 1},
    )

    assert result.controller_terminal == 1
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.SUCCEEDED
    assert stored.last_reconciled_status == TaskState.FINISHED.value
    row = _batch_lease_row(batch_db, task.task_id)
    assert row["lease_token"] == ""


@pytest.mark.asyncio
async def test_stale_active_repair_skips_foreign_live_terminal_without_releasing_lease(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="done", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-terminal-live-foreign", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            controller_task_id="controller-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
        )
    )
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=time.time() + 3600,
    )
    controller_db = tmp_path / "controller.sqlite"
    _finished_controller_db(controller_db, batch_id=manifest.batch_id)

    result = await repair_stale_active_rows_once(
        config=BatchRunConfig(
            manifest=manifest,
            runner_id="new-runner",
            telemetry_controller_db_path=controller_db,
        ),
        repository=repository,
        max_attempts_by_task={task.task_id: 1},
    )

    assert result.repaired == 0
    assert result.skipped_live_lease == 1
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.SUBMITTED
    assert stored.last_reconciled_status == ""
    row = _batch_lease_row(batch_db, task.task_id)
    assert row["lease_token"] == "lease-token"


@pytest.mark.asyncio
async def test_stale_active_repair_skips_nonterminal_controller_snapshot(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="running", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-running-controller", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            controller_task_id="controller-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            reconcile_unknown_count=2,
        )
    )
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=time.time() - 1,
    )
    controller_db = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(controller_db)
    store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="web",
            request="running",
            metadata={"batch_id": manifest.batch_id},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    store.update_task_status("controller-1", TaskState.RUNNING)
    store.close()

    result = await repair_stale_active_rows_once(
        config=BatchRunConfig(
            manifest=manifest,
            runner_id="new-runner",
            telemetry_controller_db_path=controller_db,
        ),
        repository=repository,
        max_attempts_by_task={task.task_id: 1},
    )

    assert result.repaired == 0
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.SUBMITTED
    assert stored.controller_task_id == "controller-1"
    assert stored.reconcile_unknown_count == 2


@pytest.mark.asyncio
async def test_stale_active_repair_releases_expired_no_controller_id_without_retry_spin(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="stale", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-no-controller", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.WAITING_DEVICE,
            attempts=1,
        )
    )
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=time.time() - 1,
    )

    result = await repair_stale_active_rows_once(
        config=BatchRunConfig(manifest=manifest, runner_id="new-runner"),
        repository=repository,
        max_attempts_by_task={task.task_id: 1},
    )

    assert result.no_controller_id_expired == 1
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.FAILED
    assert stored.controller_task_id == ""
    assert stored.last_reconciled_status == "local_stale_reset"
    row = _batch_lease_row(batch_db, task.task_id)
    assert row["lease_token"] == ""
    assert (
        repository.claim_next(
            lease_owner="new-runner",
            policy=LeasePolicy.from_timeout(30),
            max_attempts_by_task={task.task_id: 1},
        )
        is None
    )


@pytest.mark.asyncio
async def test_stale_active_repair_marks_expired_missing_controller_task_without_clearing_id(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="missing", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-missing", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            controller_task_id="missing-controller",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
        )
    )
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=time.time() - 1,
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()

    result = await repair_stale_active_rows_once(
        config=BatchRunConfig(
            manifest=manifest,
            runner_id="new-runner",
            telemetry_controller_db_path=controller_db,
        ),
        repository=repository,
        max_attempts_by_task={task.task_id: 1},
    )

    assert result.missing_controller_task == 1
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.FAILED
    assert stored.controller_task_id == "missing-controller"
    assert stored.recovery_action == FailureAction.FAIL_FAST
    assert stored.last_reconciled_status == "unknown"
    row = _batch_lease_row(batch_db, task.task_id)
    assert row["lease_token"] == ""
    assert (
        repository.claim_next(
            lease_owner="new-runner",
            policy=LeasePolicy.from_timeout(30),
            max_attempts_by_task={task.task_id: 1},
        )
        is None
    )


@pytest.mark.asyncio
async def test_stale_active_repair_preserves_missing_controller_id_after_unknown_threshold(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="missing", max_retries=3)
    manifest = BatchManifest(batch_id="resume-repair-missing-after-threshold", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            controller_task_id="missing-controller",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            reconcile_unknown_count=2,
        )
    )
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=time.time() - 1,
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()

    result = await repair_stale_active_rows_once(
        config=BatchRunConfig(
            manifest=manifest,
            runner_id="new-runner",
            telemetry_controller_db_path=controller_db,
            infra_retry_backoff_seconds=[0],
        ),
        repository=repository,
        max_attempts_by_task={task.task_id: 4},
    )

    assert result.missing_controller_task == 1
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.FAILED
    assert stored.controller_task_id == "missing-controller"
    assert stored.last_reconciled_status == "unknown"
    assert stored.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert stored.reconcile_unknown_count == 3
    assert "will retry controller status reconcile before any fresh submit" in stored.error
    assert "cleared stale controller task id" not in stored.error


@pytest.mark.asyncio
async def test_stale_active_repair_skips_current_runner_live_no_controller_row(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="live", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-live", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.WAITING_DEVICE,
            attempts=1,
        )
    )
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="current-runner",
        expires_at=time.time() + 3600,
    )

    result = await repair_stale_active_rows_once(
        config=BatchRunConfig(manifest=manifest, runner_id="current-runner"),
        repository=repository,
        max_attempts_by_task={task.task_id: 1},
    )

    assert result.repaired == 0
    assert result.skipped_current_owner == 1
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.WAITING_DEVICE
    row = _batch_lease_row(batch_db, task.task_id)
    assert row["lease_token"] == "lease-token"


@pytest.mark.asyncio
async def test_stale_active_repair_skips_foreign_live_no_controller_row(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="live", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-live-foreign", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.WAITING_DEVICE,
            attempts=1,
        )
    )
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=time.time() + 3600,
    )

    result = await repair_stale_active_rows_once(
        config=BatchRunConfig(manifest=manifest, runner_id="new-runner"),
        repository=repository,
        max_attempts_by_task={task.task_id: 1},
    )

    assert result.repaired == 0
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.WAITING_DEVICE
    row = _batch_lease_row(batch_db, task.task_id)
    assert row["lease_token"] == "lease-token"


@pytest.mark.asyncio
async def test_stale_active_repair_skips_foreign_live_no_controller_with_fresh_heartbeat(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="live", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-live-foreign-heartbeat", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.WAITING_DEVICE,
            attempts=1,
        )
    )
    now = time.time()
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=now + 300,
        updated_at=now - 3600,
    )

    result = await repair_stale_active_rows_once(
        config=BatchRunConfig(manifest=manifest, runner_id="new-runner", lease_timeout_seconds=300),
        repository=repository,
        max_attempts_by_task={task.task_id: 1},
    )

    assert result.repaired == 0
    assert result.skipped_live_lease == 1
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.WAITING_DEVICE
    row = _batch_lease_row(batch_db, task.task_id)
    assert row["lease_token"] == "lease-token"


@pytest.mark.asyncio
async def test_stale_active_repair_releases_stale_foreign_live_no_controller_row(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="stale foreign", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-stale-foreign-live", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.WAITING_DEVICE,
            attempts=1,
        )
    )
    now = time.time()
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=now + 100,
        updated_at=now,
    )

    result = await repair_stale_active_rows_once(
        config=BatchRunConfig(manifest=manifest, runner_id="new-runner", lease_timeout_seconds=300),
        repository=repository,
        max_attempts_by_task={task.task_id: 1},
    )

    assert result.foreign_live_no_controller_id_stale == 1
    stored = repository.get(task.task_id)
    assert stored is not None
    assert stored.status == BatchTaskStatus.FAILED
    assert stored.last_failure_category == FailureCategory.READINESS_TIMEOUT
    assert stored.last_reconciled_status == "local_stale_reset"
    row = _batch_lease_row(batch_db, task.task_id)
    assert row["lease_token"] == ""
    assert (
        repository.claim_next(
            lease_owner="new-runner",
            policy=LeasePolicy.from_timeout(300),
            max_attempts_by_task={task.task_id: 1},
        )
        is None
    )


def test_stale_active_repair_clear_lease_cas_rechecks_lease_expiry(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="race", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-renewed-lease", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.WAITING_DEVICE,
            attempts=1,
        )
    )
    now = time.time()
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=now - 1,
    )
    connection = sqlite3.connect(batch_db)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT * FROM batch_tasks WHERE task_id=?", (task.task_id,)).fetchone()
        assert row is not None
        connection.execute(
            "UPDATE batch_tasks SET lease_expires_at=? WHERE task_id=?",
            (now + 3600, task.task_id),
        )
        state = BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.FAILED,
            attempts=1,
            last_failure_category=FailureCategory.READINESS_TIMEOUT,
        )

        assert not _cas_update(connection, row=row, state=state, now=now, clear_lease=True)
        current = connection.execute("SELECT status, lease_token FROM batch_tasks WHERE task_id=?", (task.task_id,))
        current_row = current.fetchone()
        assert current_row["status"] == BatchTaskStatus.WAITING_DEVICE.value
        assert current_row["lease_token"] == "lease-token"
    finally:
        connection.close()


def test_stale_active_repair_foreign_live_cas_rechecks_heartbeat(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="race", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-foreign-live-renewed", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.WAITING_DEVICE,
            attempts=1,
        )
    )
    now = time.time()
    stale_expires_before = now + 100
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=stale_expires_before,
        updated_at=now,
    )
    connection = sqlite3.connect(batch_db)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT * FROM batch_tasks WHERE task_id=?", (task.task_id,)).fetchone()
        assert row is not None
        connection.execute(
            "UPDATE batch_tasks SET updated_at=?, lease_expires_at=? WHERE task_id=?",
            (now, now + 3600, task.task_id),
        )
        state = BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.FAILED,
            attempts=1,
            last_failure_category=FailureCategory.READINESS_TIMEOUT,
        )

        assert not _cas_update(
            connection,
            row=row,
            state=state,
            now=now,
            clear_lease=True,
            foreign_live_owner_to_clear="old-runner",
            foreign_live_lease_expires_at_or_before=stale_expires_before,
        )
        current = connection.execute("SELECT status, lease_token FROM batch_tasks WHERE task_id=?", (task.task_id,))
        current_row = current.fetchone()
        assert current_row["status"] == BatchTaskStatus.WAITING_DEVICE.value
        assert current_row["lease_token"] == "lease-token"
    finally:
        connection.close()


def test_stale_active_repair_foreign_live_cas_rechecks_controller_submission(tmp_path) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="race", max_retries=0)
    manifest = BatchManifest(batch_id="resume-repair-foreign-live-submitted", tasks=[task])
    batch_db = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(batch_db)
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.WAITING_DEVICE,
            attempts=1,
        )
    )
    now = time.time()
    stale_expires_before = now + 100
    _force_batch_lease(
        batch_db,
        task_id=task.task_id,
        owner="old-runner",
        expires_at=stale_expires_before,
        updated_at=now,
    )
    connection = sqlite3.connect(batch_db)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT * FROM batch_tasks WHERE task_id=?", (task.task_id,)).fetchone()
        assert row is not None
        connection.execute(
            "UPDATE batch_tasks SET controller_task_id=?, updated_at=? WHERE task_id=?",
            ("controller-1", now, task.task_id),
        )
        state = BatchTaskState(
            task_id=task.task_id,
            status=BatchTaskStatus.FAILED,
            attempts=1,
            last_failure_category=FailureCategory.READINESS_TIMEOUT,
        )

        assert not _cas_update(
            connection,
            row=row,
            state=state,
            now=now,
            clear_lease=True,
            foreign_live_owner_to_clear="old-runner",
            foreign_live_lease_expires_at_or_before=stale_expires_before,
        )
        current = connection.execute(
            "SELECT status, controller_task_id, lease_token FROM batch_tasks WHERE task_id=?",
            (task.task_id,),
        )
        current_row = current.fetchone()
        assert current_row["status"] == BatchTaskStatus.WAITING_DEVICE.value
        assert current_row["controller_task_id"] == "controller-1"
        assert current_row["lease_token"] == "lease-token"
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_resume_reconciler_falls_back_to_websocket_on_local_controller_db_miss(tmp_path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(controller_db)
    store.close()
    manifest = BatchManifest(
        batch_id="resume-db-miss",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="done")],
    )
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="web",
        state=TaskState.FINISHED,
        device_id="device-1",
    )
    watcher = _StatusWatcher(snapshot)

    reconciled = await BatchStatusReconciler(watcher).reconcile(
        config=BatchRunConfig(
            manifest=manifest,
            controller_url="ws://controller",
            infra_retry_backoff_seconds=[0],
            telemetry_controller_db_path=controller_db,
        ),
        task=manifest.tasks[0],
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.SUCCEEDED
    assert reconciled.controller_task_id == "controller-1"
    assert reconciled.last_reconciled_status == TaskState.FINISHED.value
    assert watcher.calls


@pytest.mark.asyncio
async def test_resume_reconciler_delays_local_controller_db_miss_when_fallback_disabled(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller_db = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(controller_db)
    store.close()
    monkeypatch.setenv("NORTHCUA_BATCH_RECONCILE_STATUS_WEBSOCKET_FALLBACK_ON_DB_MISS", "0")
    manifest = BatchManifest(
        batch_id="resume-db-miss-no-fallback",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="done")],
    )
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="web",
        state=TaskState.FINISHED,
        device_id="device-1",
    )
    watcher = _StatusWatcher(snapshot)

    reconciled = await BatchStatusReconciler(watcher).reconcile(
        config=BatchRunConfig(
            manifest=manifest,
            controller_url="ws://controller",
            infra_retry_backoff_seconds=[0],
            telemetry_controller_db_path=controller_db,
        ),
        task=manifest.tasks[0],
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.controller_task_id == "controller-1"
    assert reconciled.last_reconciled_status == "unknown"
    assert "local controller DB status lookup returned no row" in reconciled.error
    assert "will retry controller status reconcile" in reconciled.error
    assert watcher.calls == []


@pytest.mark.asyncio
async def test_resume_reconciler_delays_db_timeout_without_websocket_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller_db = tmp_path / "controller.sqlite"
    store = SQLiteControllerStore(controller_db)
    store.close()
    monkeypatch.setenv("NORTHCUA_BATCH_RECONCILE_STATUS_DB_TIMEOUT_SECONDS", "0.01")
    monkeypatch.delenv("NORTHCUA_BATCH_RECONCILE_STATUS_WEBSOCKET_FALLBACK_ON_DB_TIMEOUT", raising=False)

    def slow_statuses(*args: object, **kwargs: object) -> list[TaskStatusSnapshot]:
        _ = args, kwargs
        time.sleep(0.2)
        return []

    monkeypatch.setattr("northcua.batch.controller_db_status.LocalControllerDBStatusReader.statuses", slow_statuses)
    manifest = BatchManifest(
        batch_id="resume-db-timeout",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="done", max_retries=1)],
    )
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="web",
        state=TaskState.FINISHED,
        device_id="device-1",
    )
    watcher = _StatusWatcher(snapshot)

    reconciled = await BatchStatusReconciler(watcher).reconcile(
        config=BatchRunConfig(
            manifest=manifest,
            controller_url="ws://controller",
            infra_retry_backoff_seconds=[0],
            telemetry_controller_db_path=controller_db,
        ),
        task=manifest.tasks[0],
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.controller_task_id == "controller-1"
    assert reconciled.last_reconciled_status == "unknown"
    assert reconciled.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert reconciled.reconcile_unknown_count == 1
    assert "local controller DB status lookup timed out" in reconciled.error
    assert "will retry controller status reconcile" in reconciled.error
    assert watcher.calls == []


@pytest.mark.asyncio
async def test_resume_reconciler_falls_back_to_websocket_when_local_db_unavailable(tmp_path) -> None:
    manifest = BatchManifest(
        batch_id="resume-db-unavailable",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="done")],
    )
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="web",
        state=TaskState.FINISHED,
        device_id="device-1",
    )
    watcher = _StatusWatcher(snapshot)

    reconciled = await BatchStatusReconciler(watcher).reconcile(
        config=BatchRunConfig(
            manifest=manifest,
            controller_url="ws://controller",
            telemetry_controller_db_path=tmp_path / "missing-controller.sqlite",
        ),
        task=manifest.tasks[0],
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.SUCCEEDED
    assert watcher.calls


@pytest.mark.asyncio
async def test_resume_reconciler_checks_failed_local_state_with_controller_id() -> None:
    manifest = BatchManifest(
        batch_id="resume-failed-local",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="done")],
    )
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.FAILED,
        error="local runner crashed after submit",
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="web",
        state=TaskState.FINISHED,
        device_id="device-1",
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.SUCCEEDED
    assert reconciled.error == ""
    assert reconciled.last_reconciled_status == "finished"


@pytest.mark.asyncio
async def test_resume_reconciler_uses_successful_uncontrollable_evaluator_verdict() -> None:
    manifest = BatchManifest(
        batch_id="resume-eval-accepted-infeasible",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="use unavailable hardware")],
    )
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
        error="hardware unavailable",
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.FAILED,
        error="hardware unavailable",
        result={
            "evaluation_report": {
                "overall_verdict": "success",
                "outcome_label": "success",
                "controllability": "uncontrollable",
            }
        },
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=manifest.tasks[0],
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.SUCCEEDED
    assert reconciled.terminal_state == TaskState.FAILED
    assert reconciled.error == ""
    assert reconciled.last_failure_category == FailureCategory.ACCEPTED_INFEASIBLE
    assert reconciled.last_reconciled_status == TaskState.FAILED.value


@pytest.mark.asyncio
async def test_resume_reconciler_waits_for_running_controller_task() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="wait")
    manifest = BatchManifest(batch_id="resume-running", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
    )
    watcher = _StatusWatcher(
        TaskStatusSnapshot(
            task_id="controller-1",
            trace_id="trace-1",
            session_id="session-1",
            agent_kind="web",
            state=TaskState.RUNNING,
        ),
        TaskStatusSnapshot(
            task_id="controller-1",
            trace_id="trace-1",
            session_id="session-1",
            agent_kind="web",
            state=TaskState.FINISHED,
            device_id="device-1",
        ),
    )

    reconciled = await BatchStatusReconciler(watcher, poll_interval_seconds=0).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.SUCCEEDED
    assert len(watcher.calls) == 2


@pytest.mark.asyncio
async def test_resume_reconciler_preserves_nonterminal_controller_task_after_wait_deadline() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="wait", timeout_seconds=0.01)
    manifest = BatchManifest(batch_id="resume-timeout", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
    )
    running = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="web",
        state=TaskState.RUNNING,
    )

    started = time.time()
    reconciled = await BatchStatusReconciler(_StatusWatcher(running), poll_interval_seconds=0).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.RUNNING
    assert reconciled.last_reconciled_status == "running"
    assert reconciled.next_attempt_after >= started
    assert "will reconcile later" in reconciled.error


@pytest.mark.asyncio
async def test_resume_reconciler_delays_when_status_becomes_unavailable_during_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="wait", timeout_seconds=10)
    manifest = BatchManifest(batch_id="resume-status-unavailable", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
    )
    running = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="web",
        state=TaskState.RUNNING,
    )
    reconciler = BatchStatusReconciler(_StatusWatcher(running), poll_interval_seconds=0)

    async def unavailable_statuses(**kwargs: object) -> list[TaskStatusSnapshot]:
        _ = kwargs
        raise ControllerStatusUnavailable("local controller DB status lookup timed out")

    monkeypatch.setattr(reconciler, "_statuses", unavailable_statuses)
    started = time.time()

    reconciled = await reconciler._wait_for_terminal_snapshot(  # noqa: SLF001 - targeted timeout regression.
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
        initial=running,
        status_semaphore=None,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.SUBMITTED
    assert reconciled.last_reconciled_status == "unknown"
    assert reconciled.next_attempt_after >= started
    assert "will reconcile later" in reconciled.error


@pytest.mark.asyncio
async def test_failed_controller_snapshot_can_still_enter_new_attempt_when_budget_remains() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=1)
    manifest = BatchManifest(batch_id="resume-retry", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="web",
        state=TaskState.FAILED,
        error="agent failed",
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.attempts == 1


@pytest.mark.asyncio
async def test_resume_reconciler_preserves_non_retryable_terminal_decision() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=3)
    manifest = BatchManifest(batch_id="resume-non-retryable", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
    )
    decision = {
        "action": "fail",
        "reason": "max_steps",
        "success": False,
        "retryable": False,
        "metadata": {"scope": "session", "limit": 4, "current": 4},
    }
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="web",
        state=TaskState.FAILED,
        error="session step budget exhausted",
        terminal_decision=decision,
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.recovery_action == FailureAction.FAIL_FAST
    assert reconciled.terminal_decision == decision


@pytest.mark.asyncio
async def test_resume_reconciler_projects_terminal_evaluation_pending() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="retry", max_retries=3)
    manifest = BatchManifest(batch_id="resume-evaluation-pending", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
    )
    decision = {
        "action": "finish",
        "success": True,
        "metadata": {
            "evaluation": {
                "status": "pending",
                "attempts": 2,
                "projection_level": "tight",
                "error_type": "EvaluatorPromptBudgetExceeded",
            }
        },
    }
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.FINISHED,
        result={"agent": "done"},
        terminal_decision=decision,
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.EVALUATION_PENDING
    assert reconciled.error == ""
    assert reconciled.last_failure_category == FailureCategory.EVALUATOR
    assert reconciled.recovery_action == FailureAction.FAIL_FAST
    assert reconciled.agent_terminal["state"] == TaskState.FINISHED.value
    assert reconciled.evaluation_status == "pending"
    assert reconciled.evaluation_attempts == 2
    assert reconciled.evaluation_projection_level == "tight"


@pytest.mark.asyncio
async def test_resume_reconciler_uses_non_success_evaluator_report_as_failed() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="retry", max_retries=3)
    manifest = BatchManifest(batch_id="resume-evaluator-failure-report", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.FINISHED,
        result={"evaluation_report": {"overall_verdict": "failure", "outcome_label": "failure"}},
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.terminal_state == TaskState.FINISHED


@pytest.mark.asyncio
async def test_resume_reconciler_does_not_fail_fast_device_transport() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="retry", max_retries=3)
    manifest = BatchManifest(batch_id="resume-device-transport", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
    )
    decision = {
        "action": "error",
        "reason": "runtime_error",
        "success": False,
        "retryable": False,
        "metadata": {"failure_category": FailureCategory.DEVICE_TRANSPORT.value},
    }
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.FAILED,
        error="device websocket disconnected",
        terminal_decision=decision,
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.last_failure_category == FailureCategory.DEVICE_TRANSPORT
    assert reconciled.recovery_action is None


@pytest.mark.asyncio
async def test_resume_reconciler_clean_interrupted_ignores_stale_transport_category() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="cancel", max_retries=1)
    manifest = BatchManifest(batch_id="resume-clean-interrupted", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
        last_failure_category=FailureCategory.DEVICE_TRANSPORT,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.INTERRUPTED,
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.CANCELED
    assert reconciled.last_failure_category == FailureCategory.DEVICE_TRANSPORT


@pytest.mark.asyncio
async def test_resume_reconciler_retries_residue_cleanup_interruption() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="retry", max_retries=1)
    manifest = BatchManifest(batch_id="resume-cleanup-interrupted", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.INTERRUPTED,
        error="controller residue cleanup interrupted stale task",
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert reconciled.recovery_action is None


@pytest.mark.asyncio
async def test_resume_reconciler_uses_terminal_failure_category() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="retry", max_retries=1)
    manifest = BatchManifest(batch_id="resume-evaluator-category", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
    )
    decision = {
        "action": "error",
        "reason": "runtime_error",
        "success": False,
        "retryable": True,
        "metadata": {"failure_category": FailureCategory.EVALUATOR.value},
    }
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.FAILED,
        error="evaluator failed",
        terminal_decision=decision,
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.last_failure_category == FailureCategory.EVALUATOR


@pytest.mark.asyncio
async def test_resume_reconciler_maps_interrupted_timeout_to_timed_out() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="desktop", query="retry", max_retries=1)
    manifest = BatchManifest(batch_id="resume-timeout-retry", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
        last_failure_category=FailureCategory.TIMEOUT,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.INTERRUPTED,
        error="batch task attempt timed out after 1200 seconds",
    )

    reconciled = await BatchStatusReconciler(_StatusWatcher(snapshot)).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.TIMED_OUT
    assert reconciled.last_failure_category == FailureCategory.TIMEOUT


def test_submitter_maps_interrupted_timeout_to_timed_out() -> None:
    status = _local_status_from_controller_terminal(
        TaskState.INTERRUPTED,
        last_failure_category=FailureCategory.TIMEOUT,
    )

    assert status == BatchTaskStatus.TIMED_OUT


def test_submitter_maps_interrupted_controller_submit_to_retryable_failed() -> None:
    status = _local_status_from_controller_terminal(
        TaskState.INTERRUPTED,
        last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
    )

    assert status == BatchTaskStatus.FAILED


def test_submitter_maps_non_success_evaluator_report_to_failed() -> None:
    status = _local_status_from_controller_terminal(
        TaskState.FINISHED,
        last_failure_category=None,
        evaluator_report_present=True,
    )

    assert status == BatchTaskStatus.FAILED


@pytest.mark.asyncio
async def test_resume_reconciler_retries_when_controller_status_fails_with_budget() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=1)
    manifest = BatchManifest(batch_id="resume-status-error", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
    )

    started = time.time()
    reconciled = await BatchStatusReconciler(_FailingStatusWatcher()).reconcile(
        config=BatchRunConfig(manifest=manifest, infra_retry_backoff_seconds=[0]),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.recovery_action is None
    assert reconciled.controller_task_id == "controller-1"
    assert "will retry controller status reconcile" in reconciled.error
    assert reconciled.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert reconciled.last_reconciled_status == "unknown"
    assert reconciled.next_attempt_after >= started + 1
    assert reconciled.reconcile_unknown_count == 1


@pytest.mark.asyncio
async def test_resume_reconciler_retries_when_controller_status_missing_with_budget() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=1)
    manifest = BatchManifest(batch_id="resume-status-missing", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
    )

    reconciled = await BatchStatusReconciler(_EmptyStatusWatcher()).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.recovery_action is None
    assert reconciled.controller_task_id == "controller-1"
    assert "will retry controller status reconcile" in reconciled.error
    assert reconciled.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert reconciled.last_reconciled_status == "unknown"
    assert reconciled.next_attempt_after > 0
    assert reconciled.reconcile_unknown_count == 1


@pytest.mark.asyncio
async def test_resume_reconciler_clears_missing_controller_status_after_reconcile_backoff() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=3)
    manifest = BatchManifest(batch_id="resume-status-missing-after-backoff", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
        reconcile_unknown_count=2,
    )

    started = time.time()
    reconciled = await BatchStatusReconciler(_EmptyStatusWatcher()).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.recovery_action is None
    assert reconciled.controller_task_id == ""
    assert "cleared stale controller task id for retry after 3 unknown reconcile attempts" in reconciled.error
    assert reconciled.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert reconciled.last_reconciled_status == "unknown"
    assert reconciled.next_attempt_after >= started + 30
    assert reconciled.next_attempt_after < started + 90
    assert reconciled.reconcile_unknown_count == 3


@pytest.mark.asyncio
async def test_resume_reconciler_keeps_status_endpoint_errors_delayed_after_reconcile_backoff() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=3)
    manifest = BatchManifest(batch_id="resume-status-error-after-backoff", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
        reconcile_unknown_count=2,
    )

    started = time.time()
    reconciled = await BatchStatusReconciler(_FailingStatusWatcher()).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.recovery_action is None
    assert reconciled.controller_task_id == "controller-1"
    assert "will retry controller status reconcile before any fresh submit" in reconciled.error
    assert reconciled.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert reconciled.last_reconciled_status == "unknown"
    assert reconciled.next_attempt_after >= started + 180
    assert reconciled.reconcile_unknown_count == 3


@pytest.mark.asyncio
async def test_resume_reconciler_blocks_unknown_controller_status_without_budget() -> None:
    task = BatchTaskSpec(task_id="task-1", agent_kind="web", query="retry", max_retries=1)
    manifest = BatchManifest(batch_id="resume-status-no-budget", tasks=[task])
    existing = BatchTaskState(
        task_id="task-1",
        controller_task_id="controller-1",
        status=BatchTaskStatus.RUNNING,
        attempts=2,
    )

    reconciled = await BatchStatusReconciler(_EmptyStatusWatcher()).reconcile(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        existing=existing,
    )

    assert reconciled is not None
    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.recovery_action == FailureAction.FAIL_FAST
    assert "retry budget exhausted" in reconciled.error
    assert reconciled.controller_task_id == "controller-1"
    assert reconciled.last_reconciled_status == "unknown"
