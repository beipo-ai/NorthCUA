"""Tests for the single-use prepared sandbox queue."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from northcua.batch.contracts import BatchManifest, BatchRunConfig, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.metrics import BatchMetrics
from northcua.batch.preparation import (
    PreparedSandboxDepth,
    PreparedSandboxFailureKind,
    PreparedSandboxFallbackReason,
    PreparedSandboxPolicy,
    PreparedSandboxRecord,
    PreparedSandboxStatus,
)
from northcua.batch.preparation import manager as preparation_manager_module
from northcua.batch.preparation import release as preparation_release_module
from northcua.batch.preparation import sqlite_store as sqlite_store_module
from northcua.batch.preparation.assignment import PreparedSandboxAssignment
from northcua.batch.preparation.manager import PreparedSandboxManager
from northcua.batch.preparation.planner import PreparedSandboxPlanner
from northcua.batch.preparation.preparer import _recipe_failure_details
from northcua.batch.preparation.sqlite_store import SQLitePreparedSandboxStore
from northcua.batch.runner import BatchRunner
from northcua.batch.runner import attempt_helpers as attempt_helpers_module
from northcua.batch.runner import attempt_submit as attempt_submit_module
from northcua.batch.runner import scheduler as scheduler_module
from northcua.batch.runner.bindings import SandboxDeviceBindingFactory
from northcua.batch.runner.events import BatchEventWriter
from northcua.batch.runner.leases import BatchTaskLeaseContext, BatchTaskLeaseCoordinator
from northcua.batch.runner.scheduler import BatchScheduler
from northcua.batch.state import LeasePolicy, SQLiteBatchStateRepository
from northcua.reliability import FailureAction
from northcua.sandboxes.contracts import (
    SandboxAllocationStrategy,
    SandboxCommandSpec,
    SandboxHandle,
    SandboxLease,
    SandboxProviderCapabilities,
    SandboxProviderFeature,
    SandboxProviderSpec,
    SandboxReadinessState,
)
from northcua.sandboxes.provider.fake import FakeSandboxProvider
from northcua.sandboxes.provider.registry import SandboxProviderRegistry
from northcua.telemetry.contracts import FailureCategory
from northcua_protocol.payloads import TaskAccepted, TaskFinished
from northcua_protocol.state import TaskState


def test_prepared_recipe_failure_details_are_bucketable() -> None:
    details = _recipe_failure_details(
        "sandbox startup command failed: label='catalog-open-file' exit_code=1 "
        "timed_out=False failure_kind=window_readiness_timeout_before_process_appeared "
        "stderr_tail='window readiness timed out before process appeared'"
    )

    assert details == {
        "recipe_failure_kind": "window_readiness_timeout_before_process_appeared",
        "recipe_label": "catalog-open-file",
    }


def test_terminal_evaluator_pending_preserves_agent_result_without_attempt_failure(tmp_path: Path) -> None:
    manifest = BatchManifest(
        batch_id="evaluator-pending",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    metrics = BatchMetrics()
    lease = SandboxLease(
        lease_id="lease-1",
        handle=SandboxHandle(sandbox_id="sandbox-1", provider="fake", template="desktop"),
        device_id="device-1",
    )
    state = BatchTaskState(
        task_id="case-1",
        attempts=1,
        status=BatchTaskStatus.RUNNING,
        sandbox_id="sandbox-1",
        device_id="device-1",
    )

    updated = attempt_submit_module._state_from_terminal_controller_result(  # noqa: SLF001
        accepted=TaskAccepted(task_id="controller-task-1", trace_id="trace-1", session_id="session-1"),
        finished=TaskFinished(
            task_id="controller-task-1",
            trace_id="trace-1",
            session_id="session-1",
            state=TaskState.FINISHED,
            device_id="device-1",
            result={"record": {"overall_verdict": "success"}},
            terminal_decision={
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
            },
        ),
        task=manifest.tasks[0],
        lease=lease,
        state=state,
        metrics=metrics,
        events=events,
        task_start=0.0,
    )

    attempt_events = [
        json.loads(line) for line in (tmp_path / "batch_attempt_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    details = next(event for event in attempt_events if event["message"] == "controller_submit_finished")["details"]
    assert updated.status == BatchTaskStatus.EVALUATION_PENDING
    assert updated.error == ""
    assert updated.last_failure_category == FailureCategory.EVALUATOR
    assert updated.agent_terminal["state"] == TaskState.FINISHED.value
    assert updated.evaluation_status == "pending"
    assert updated.evaluation_attempts == 2
    assert updated.evaluation_projection_level == "tight"
    assert metrics.attempt_failure_categories == {}
    assert details["terminal_status"] == BatchTaskStatus.EVALUATION_PENDING.value
    assert details["evaluation_status"] == "pending"


def _prepared_fallback_events(output_dir: Path, *, case_id: str = "case-1") -> list[dict[str, object]]:
    path = output_dir / "prepared_sandbox_events.jsonl"
    if not path.exists():
        return []
    return [
        event
        for event in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        if event["message"] == "prepared_sandbox_fallback" and event["case_id"] == case_id
    ]


def _prepared_events(output_dir: Path, *, message: str = "") -> list[dict[str, object]]:
    path = output_dir / "prepared_sandbox_events.jsonl"
    if not path.exists():
        return []
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not message:
        return events
    return [event for event in events if event["message"] == message]


class NoTimeoutExtensionFakeProvider(FakeSandboxProvider):
    """Fake provider variant that cannot extend provider-side lifetime."""

    def capabilities(self) -> SandboxProviderCapabilities:
        capabilities = super().capabilities()
        return SandboxProviderCapabilities(
            features={feature for feature in capabilities.features if feature != SandboxProviderFeature.EXTEND_TIMEOUT}
        )

    async def extend_timeout(self, handle: SandboxHandle, timeout_seconds: int) -> None:
        raise AssertionError("timeout extension should not be called when unsupported")


class RecordingSubmitter:
    """Submitter double that records startup commands visible at submit time."""

    command_counts_at_submit: list[int] = []
    command_labels_at_submit: list[list[str]] = []
    submit_times: list[float] = []
    provider: FakeSandboxProvider | None = None

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def __aenter__(self) -> "RecordingSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        assert self.provider is not None
        self.command_counts_at_submit.append(len(self.provider.commands))
        self.command_labels_at_submit.append([command.label for _, command in self.provider.commands])
        self.submit_times.append(time.monotonic())
        task_id = str(kwargs["task_id"])
        return TaskAccepted(task_id=task_id, trace_id=f"trace-{task_id}", session_id=f"session-{task_id}")

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


class SlowRecordingSubmitter(RecordingSubmitter):
    """Submitter double that lets a late prepared future finish before close."""

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        await asyncio.sleep(0.35)
        return await super().wait_for_finished(task_id, timeout_seconds=timeout_seconds)


class PreparedCreateFailsOnceProvider(FakeSandboxProvider):
    """Fake provider that fails one prepared create and then recovers."""

    prepared_failures_remaining = 1

    async def create(self, spec: SandboxProviderSpec):
        if spec.metadata.get("prepared_sandbox") == "true" and self.prepared_failures_remaining > 0:
            self.__class__.prepared_failures_remaining -= 1
            raise RuntimeError("prepared create failed once")
        return await super().create(spec)


class PreparedCreateSlowProvider(FakeSandboxProvider):
    """Fake provider whose prepared create outlives the claim wait timeout."""

    async def create(self, spec: SandboxProviderSpec):
        if spec.metadata.get("prepared_sandbox") == "true":
            await asyncio.sleep(0.3)
        return await super().create(spec)


class SequencedPreparedCreateProvider(FakeSandboxProvider):
    """Fake provider that pauses the first prepared create until released."""

    def __init__(self) -> None:
        super().__init__()
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.prepared_create_calls = 0

    async def create(self, spec: SandboxProviderSpec):
        if spec.metadata.get("prepared_sandbox") == "true":
            self.prepared_create_calls += 1
            if self.prepared_create_calls == 1:
                self.first_started.set()
                await self.release_first.wait()
        return await super().create(spec)


class SecondPreparedCreatePausesProvider(FakeSandboxProvider):
    """Fake provider that leaves the second prepared create in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()
        self.prepared_create_calls = 0

    async def create(self, spec: SandboxProviderSpec):
        if spec.metadata.get("prepared_sandbox") == "true":
            self.prepared_create_calls += 1
            if self.prepared_create_calls == 2:
                self.second_started.set()
                await self.release_second.wait()
        return await super().create(spec)


class PreparedSlowTerminateFailsProvider(FakeSandboxProvider):
    """Fake provider whose prepared late-timeout cleanup fails."""

    async def create(self, spec: SandboxProviderSpec):
        if spec.metadata.get("prepared_sandbox") == "true":
            await asyncio.sleep(0.3)
        return await super().create(spec)

    async def terminate(self, handle) -> None:
        await super().terminate(handle)
        if handle.metadata.get("prepared_sandbox") == "true":
            raise RuntimeError("prepared terminate failed")


class PreparedSlowTerminateWaitsProvider(FakeSandboxProvider):
    """Fake provider whose prepared termination waits for a test signal."""

    def __init__(self) -> None:
        super().__init__()
        self.release_termination = asyncio.Event()

    async def create(self, spec: SandboxProviderSpec):
        if spec.metadata.get("prepared_sandbox") == "true":
            await asyncio.sleep(0.03)
        return await super().create(spec)

    async def terminate(self, handle) -> None:
        if handle.metadata.get("prepared_sandbox") == "true":
            await self.release_termination.wait()
        await super().terminate(handle)


class PreparedReadyAfterFallbackProvider(PreparedSlowTerminateWaitsProvider):
    """Slow-terminate provider whose prepared sandbox exists only after the task fell back.

    Holding the prepared create until the task's own create starts means the
    prepared sandbox can never be assigned to the task. Its blocked terminate is
    then reachable only through the bounded close cleanup, never through task
    teardown, which would wait for ``release_termination`` and deadlock the run.
    """

    def __init__(self) -> None:
        super().__init__()
        self.task_create_started = asyncio.Event()

    async def create(self, spec: SandboxProviderSpec):
        if spec.metadata.get("prepared_sandbox") == "true":
            await self.task_create_started.wait()
        else:
            self.task_create_started.set()
        return await super().create(spec)


class PreparedBrokenPipeThenSucceedsProvider(FakeSandboxProvider):
    """Fake provider whose first prepared terminate raises BrokenPipeError."""

    def __init__(self) -> None:
        super().__init__()
        self.terminate_attempts: dict[str, int] = {}

    async def terminate(self, handle) -> None:
        attempts = self.terminate_attempts.get(handle.sandbox_id, 0) + 1
        self.terminate_attempts[handle.sandbox_id] = attempts
        if handle.metadata.get("prepared_sandbox") == "true" and attempts == 1:
            raise BrokenPipeError(32, "Broken pipe")
        await super().terminate(handle)


class TerminateFailsProvider(FakeSandboxProvider):
    """Fake provider that fails sandbox termination."""

    async def terminate(self, handle) -> None:
        await super().terminate(handle)
        raise RuntimeError("terminate failed")


class SequenceReadinessProbe:
    """Readiness probe that returns configured states in order."""

    def __init__(self, states: list[SandboxReadinessState]) -> None:
        self._states = list(states)
        self.timeout_seconds_seen: list[float] = []

    async def wait_for_ready(self, device_id: str, *, timeout_seconds: float) -> SandboxReadinessState:
        _ = device_id
        self.timeout_seconds_seen.append(timeout_seconds)
        if self._states:
            return self._states.pop(0)
        return SandboxReadinessState.READY


class BlockSecondReadinessProbe:
    """Readiness probe that lets prepare pass and pauses claim recheck."""

    def __init__(self) -> None:
        self.calls = 0
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()

    async def wait_for_ready(self, device_id: str, *, timeout_seconds: float) -> SandboxReadinessState:
        _ = device_id, timeout_seconds
        self.calls += 1
        if self.calls == 1:
            return SandboxReadinessState.READY
        self.second_started.set()
        await self.release_second.wait()
        return SandboxReadinessState.READY


class BlockingPrepareBackpressureGate:
    """Backpressure gate double that holds prepare scheduling until released."""

    sleep_seconds = 0.01

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.scopes: list[str] = []
        self.active = True

    async def wait_if_needed(self, *, scope: str, stop_requested=None) -> bool:
        self.scopes.append(scope)
        self.entered.set()
        if not self.active:
            return True
        while not self.release.is_set():
            if stop_requested is not None and stop_requested():
                return False
            await asyncio.sleep(0.01)
        return True

    async def active_snapshot(self, *, scope: str):
        self.scopes.append(f"snapshot:{scope}")
        return type("BackpressureSnapshot", (), {"active": self.active})()


class AlwaysActiveClaimBackpressureGate:
    sleep_seconds = 0.01

    def __init__(self, *, active_snapshots: int | None = None) -> None:
        self.scopes: list[str] = []
        self.active_snapshots = active_snapshots
        self.snapshot_calls = 0

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def wait_if_needed(self, *, scope: str, stop_requested=None) -> bool:
        self.scopes.append(scope)
        return True

    async def active_snapshot(self, *, scope: str):
        self.scopes.append(f"snapshot:{scope}")
        self.snapshot_calls += 1
        active = self.active_snapshots is None or self.snapshot_calls <= self.active_snapshots
        return type("BackpressureSnapshot", (), {"active": active})()


class ReadySetPreparedManager:
    def __init__(self, ready_attempts: set[tuple[str, int]]) -> None:
        self.ready_attempts = ready_attempts
        self.closed = False

    def ready_unassigned_task_attempts(self) -> set[tuple[str, int]]:
        return set(self.ready_attempts)

    def is_ready_unassigned(self, *, task_id: str, attempt: int) -> bool:
        return (task_id, attempt) in self.ready_attempts

    async def is_ready_unassigned_async(self, *, task_id: str, attempt: int) -> bool:
        return self.is_ready_unassigned(task_id=task_id, attempt=attempt)

    def has_foreign_claim_blocker(self, *, max_attempts_by_task, stale_after_seconds: float) -> bool:
        _ = max_attempts_by_task, stale_after_seconds
        return False

    async def has_foreign_claim_blocker_async(self, *, max_attempts_by_task, stale_after_seconds: float) -> bool:
        return self.has_foreign_claim_blocker(
            max_attempts_by_task=max_attempts_by_task,
            stale_after_seconds=stale_after_seconds,
        )

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_scheduler_reconciles_controller_rows_when_pressure_has_no_ready_sandboxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = BatchManifest(
        batch_id="pressure-reconcile-no-ready",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Reconcile old submit")],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch.sqlite")
    repository.initialize_manifest(manifest)
    old_claim = repository.claim_next(
        lease_owner="old-runner",
        policy=LeasePolicy(lease_timeout_seconds=0.01, heartbeat_interval_seconds=0.01),
    )
    assert old_claim is not None
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
        ),
        lease_token=old_claim.lease_token,
    )
    time.sleep(0.03)
    manager = ReadySetPreparedManager(set())
    gate = AlwaysActiveClaimBackpressureGate(active_snapshots=1)
    claimed: list[str] = []
    scheduler = BatchScheduler(provider_registry=SandboxProviderRegistry(), executor=object())  # type: ignore[arg-type]
    _patch_scheduler_for_claim_test(monkeypatch, scheduler=scheduler, manager=manager, gate=gate, claimed=claimed)

    states = await scheduler.run(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            concurrency=1,
            lease_timeout_seconds=30,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True),
        ),
        repository=repository,
        metrics=BatchMetrics(),
        events=BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id),
        output_dir=tmp_path,
    )

    assert claimed == ["case-1"]
    assert states[0].status == BatchTaskStatus.SUCCEEDED
    assert manager.closed


@pytest.mark.asyncio
async def test_scheduler_does_not_starve_ready_sandboxes_behind_reconcile_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = BatchManifest(
        batch_id="pressure-reconcile-before-ready",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Reconcile old submit"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Run ready prepared"),
        ],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch.sqlite")
    repository.initialize_manifest(manifest)
    old_claim = repository.claim_next(
        lease_owner="old-runner",
        policy=LeasePolicy(lease_timeout_seconds=0.01, heartbeat_interval_seconds=0.01),
    )
    assert old_claim is not None
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
        ),
        lease_token=old_claim.lease_token,
    )
    time.sleep(0.03)
    manager = ReadySetPreparedManager({("case-2", 1)})
    gate = AlwaysActiveClaimBackpressureGate(active_snapshots=1)
    claimed: list[str] = []
    scheduler = BatchScheduler(provider_registry=SandboxProviderRegistry(), executor=object())  # type: ignore[arg-type]
    _patch_scheduler_for_claim_test(monkeypatch, scheduler=scheduler, manager=manager, gate=gate, claimed=claimed)

    states = await scheduler.run(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            concurrency=1,
            lease_timeout_seconds=30,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True),
        ),
        repository=repository,
        metrics=BatchMetrics(),
        events=BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id),
        output_dir=tmp_path,
    )

    assert claimed == ["case-1", "case-2"]
    assert [state.status for state in states] == [BatchTaskStatus.SUCCEEDED, BatchTaskStatus.SUCCEEDED]
    assert manager.closed


@pytest.mark.asyncio
async def test_scheduler_reclaims_unknown_controller_rows_after_backoff_under_pressure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = BatchManifest(
        batch_id="pressure-reconcile-unknown-failed",
        tasks=[
            BatchTaskSpec(
                task_id="case-1",
                agent_kind="desktop",
                query="Reconcile unknown submit",
                max_retries=1,
            )
        ],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch.sqlite")
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id="controller-1",
            last_reconciled_status="unknown",
            next_attempt_after=time.time() - 1,
        )
    )
    manager = ReadySetPreparedManager(set())
    gate = AlwaysActiveClaimBackpressureGate()
    claimed: list[str] = []
    scheduler = BatchScheduler(provider_registry=SandboxProviderRegistry(), executor=object())  # type: ignore[arg-type]
    _patch_scheduler_for_claim_test(monkeypatch, scheduler=scheduler, manager=manager, gate=gate, claimed=claimed)

    states = await scheduler.run(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            concurrency=1,
            lease_timeout_seconds=30,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True),
        ),
        repository=repository,
        metrics=BatchMetrics(),
        events=BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id),
        output_dir=tmp_path,
    )

    assert claimed == ["case-1"]
    assert states[0].status == BatchTaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_scheduler_holds_fail_fast_unknown_rows_without_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = BatchManifest(
        batch_id="pressure-reconcile-unknown-fail-fast",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Reconcile unknown submit")],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch.sqlite")
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id="controller-1",
            recovery_action=FailureAction.FAIL_FAST,
            last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
            last_reconciled_status="unknown",
            next_attempt_after=time.time() - 1,
        )
    )
    manager = ReadySetPreparedManager(set())
    gate = AlwaysActiveClaimBackpressureGate()
    claimed: list[str] = []
    scheduler = BatchScheduler(provider_registry=SandboxProviderRegistry(), executor=object())  # type: ignore[arg-type]
    _patch_scheduler_for_claim_test(monkeypatch, scheduler=scheduler, manager=manager, gate=gate, claimed=claimed)

    states = await scheduler.run(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            concurrency=1,
            lease_timeout_seconds=30,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True),
        ),
        repository=repository,
        metrics=BatchMetrics(),
        events=BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id),
        output_dir=tmp_path,
    )

    assert claimed == []
    assert states[0].status == BatchTaskStatus.FAILED
    assert manager.closed


@pytest.mark.asyncio
async def test_scheduler_holds_unknown_rows_without_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = BatchManifest(
        batch_id="pressure-reconcile-unknown-budget-spent",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Reconcile unknown submit")],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch.sqlite")
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id="controller-1",
            last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
            last_reconciled_status="unknown",
            next_attempt_after=time.time() - 1,
        )
    )
    manager = ReadySetPreparedManager(set())
    gate = AlwaysActiveClaimBackpressureGate()
    claimed: list[str] = []
    scheduler = BatchScheduler(provider_registry=SandboxProviderRegistry(), executor=object())  # type: ignore[arg-type]
    _patch_scheduler_for_claim_test(monkeypatch, scheduler=scheduler, manager=manager, gate=gate, claimed=claimed)

    states = await scheduler.run(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            concurrency=1,
            lease_timeout_seconds=30,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True),
        ),
        repository=repository,
        metrics=BatchMetrics(),
        events=BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id),
        output_dir=tmp_path,
    )

    assert claimed == []
    assert states[0].status == BatchTaskStatus.FAILED
    assert manager.closed


def _patch_scheduler_for_claim_test(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scheduler: BatchScheduler,
    manager: ReadySetPreparedManager,
    gate: AlwaysActiveClaimBackpressureGate,
    claimed: list[str],
) -> None:
    monkeypatch.setattr(
        scheduler_module.ControllerBackpressureGate, "from_config", classmethod(lambda cls, config: gate)
    )

    async def _manager(
        *, config, repository, events, metrics, output_dir, concurrency, create_concurrency, backpressure_gate
    ):
        _ = config, repository, events, metrics, output_dir, concurrency, create_concurrency, backpressure_gate
        return manager

    async def _run_claimed(
        *,
        config,
        task,
        claim,
        repository,
        metrics,
        events,
        output_dir,
        policy,
        create_semaphore,
        submit_semaphore,
        status_semaphore,
        prepared_manager=None,
        backpressure_gate=None,
        control_flags=None,
        defer_fresh_retry_after_reconcile: bool = False,
    ):
        _ = config, task, metrics, events, output_dir, policy, create_semaphore
        _ = submit_semaphore, status_semaphore, prepared_manager, backpressure_gate
        _ = control_flags, defer_fresh_retry_after_reconcile
        claimed.append(claim.task_id)
        state = claim.state.model_copy(update={"status": BatchTaskStatus.SUCCEEDED})
        assert repository.release_claim(claim, terminal_state=state)
        return state

    monkeypatch.setattr(scheduler, "_maybe_start_prepared_manager", _manager)
    monkeypatch.setattr(scheduler, "_run_claimed", _run_claimed)


def test_prepared_sandbox_store_reservation_lifecycle(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(
        BatchManifest(
            batch_id="prepared-store",
            tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )

    assert record is not None
    assert record.status == PreparedSandboxStatus.RESERVED
    assert (
        store.reserve(
            task_id="case-1",
            attempt=1,
            runner_id="runner-b",
            depth=PreparedSandboxDepth.FULL_INIT,
            provider="fake",
            template="desktop",
            ttl_seconds=60,
        )
        is None
    )
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.25,
    )
    assert prepared is not None
    assigned = store.assign(task_id="case-1", attempt=1, runner_id="runner-a")
    assert assigned is not None
    assert assigned.status == PreparedSandboxStatus.ASSIGNED
    released = store.mark_released(assigned)
    assert released is not None
    assert released.status == PreparedSandboxStatus.RELEASED


def test_prepared_sandbox_store_tracks_deferred_cleanup_as_live_capacity(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(
        BatchManifest(
            batch_id="prepared-deferred-cleanup",
            tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assigned = store.assign(task_id="case-1", attempt=1, runner_id="runner-a")
    assert assigned is not None

    deferred = store.mark_cleanup_deferred(assigned, error="deferred_controller_reconcile")

    assert deferred is not None
    assert deferred.status == PreparedSandboxStatus.CLEANUP_DEFERRED
    assert store.active_record(task_id="case-1", attempt=1) is not None
    assert store.live_count() == 1
    assert store.live_budget_counts() == {
        "active_unassigned": 0,
        "assigned_total": 1,
        "assigned_live": 1,
        "assigned_terminal": 0,
        "cleanup_deferred": 1,
        "live_total": 1,
    }


def test_prepared_sandbox_store_assignable_requires_ready_owned_live_row(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(
        BatchManifest(
            batch_id="prepared-store-assignable",
            tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert record is not None
    assert not store.assignable(task_id="case-1", attempt=1, runner_id="runner-a")
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None

    assert store.assignable(task_id="case-1", attempt=1, runner_id="runner-a")
    assert not store.assignable(task_id="case-1", attempt=1, runner_id="runner-b")

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE prepared_sandboxes SET expires_at=1 WHERE task_id='case-1'")
    assert not store.assignable(task_id="case-1", attempt=1, runner_id="runner-a")


def test_batch_claim_skips_other_runner_active_prepared_reservation(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-claim",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    assert store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )

    assert (
        repository.claim_next(
            lease_owner="runner-b",
            policy=LeasePolicy.from_timeout(30),
            respect_prepared_reservations=True,
        )
        is None
    )
    assert repository.claim_next(lease_owner="runner-a", policy=LeasePolicy.from_timeout(30)) is not None


def test_batch_claim_prefers_current_runner_prepared_reservation(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-current-runner-priority",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Do two"),
        ],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    assert store.reserve(
        task_id="case-2",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy.from_timeout(30),
        respect_prepared_reservations=True,
    )

    assert claim is not None
    assert claim.task_id == "case-2"


def test_batch_claim_ready_allowlist_skips_generic_and_inflight_prepared_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-ready-pressure-claim",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Do two"),
            BatchTaskSpec(task_id="case-3", agent_kind="desktop", query="Do three"),
        ],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    reserved = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert reserved is not None
    ready_base = store.reserve(
        task_id="case-2",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert ready_base is not None
    preparing = store.mark_preparing(ready_base)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-2",
        device_id="device-2",
        lease_id="lease-2",
        prepare_seconds=1.0,
    )
    assert prepared is not None

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy.from_timeout(30),
        respect_prepared_reservations=True,
        prepared_ready_task_attempts={("case-2", 1)},
    )

    assert claim is not None
    assert claim.task_id == "case-2"


def test_batch_claim_ready_allowlist_empty_holds_claims_under_pressure(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-ready-pressure-empty",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Do two"),
        ],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    assert store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )

    assert (
        repository.claim_next(
            lease_owner="runner-a",
            policy=LeasePolicy.from_timeout(30),
            respect_prepared_reservations=True,
            prepared_ready_task_attempts=set(),
        )
        is None
    )


def test_batch_claim_ready_allowlist_preserves_controller_reconcile_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-ready-pressure-reconcile",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Reconcile old submit"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Do two"),
        ],
    )
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
        )
    )

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy.from_timeout(30),
        respect_prepared_reservations=True,
        prepared_ready_task_attempts=set(),
    )

    assert claim is not None
    assert claim.task_id == "case-1"


def test_batch_claim_ready_allowlist_keeps_real_ready_rows_after_reconcile(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-ready-pressure-reconcile-real-ready",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Reconcile old submit"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Run ready prepared"),
        ],
    )
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    reserved = store.reserve(
        task_id="case-2",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert reserved is not None
    preparing = store.mark_preparing(reserved)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-2",
        device_id="device-2",
        lease_id="lease-2",
        prepare_seconds=1.0,
    )
    assert prepared is not None

    first = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy.from_timeout(30),
        respect_prepared_reservations=True,
        prepared_ready_task_attempts={("case-2", 1)},
    )
    assert first is not None
    assert first.task_id == "case-2"
    assert repository.release_claim(
        first,
        terminal_state=first.state.model_copy(update={"status": BatchTaskStatus.SUCCEEDED}),
    )

    second = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy.from_timeout(30),
        respect_prepared_reservations=True,
        prepared_ready_task_attempts={("case-2", 1)},
    )
    assert second is not None
    assert second.task_id == "case-1"


def test_batch_claim_ready_allowlist_preserves_unknown_failed_reconcile_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-ready-pressure-unknown-failed",
        tasks=[
            BatchTaskSpec(
                task_id="case-1",
                agent_kind="desktop",
                query="Reconcile unknown submit",
                max_retries=1,
            )
        ],
    )
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id="controller-1",
            last_reconciled_status="unknown",
            next_attempt_after=time.time() - 1,
        )
    )

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy.from_timeout(30),
        respect_prepared_reservations=True,
        prepared_ready_task_attempts=set(),
    )

    assert claim is not None
    assert claim.task_id == "case-1"


def test_batch_claim_ready_allowlist_holds_unknown_failed_rows_during_backoff(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-ready-pressure-unknown-backoff",
        tasks=[
            BatchTaskSpec(
                task_id="case-1",
                agent_kind="desktop",
                query="Reconcile unknown submit",
                max_retries=1,
            )
        ],
    )
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id="controller-1",
            last_reconciled_status="unknown",
            next_attempt_after=time.time() + 60,
        )
    )

    assert (
        repository.claim_next(
            lease_owner="runner-a",
            policy=LeasePolicy.from_timeout(30),
            respect_prepared_reservations=True,
            prepared_ready_task_attempts=set(),
        )
        is None
    )


def test_batch_claim_ready_allowlist_blocks_fail_fast_unknown_rows_without_retry_budget(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-ready-pressure-unknown-fail-fast",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Reconcile unknown submit")],
    )
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id="controller-1",
            recovery_action=FailureAction.FAIL_FAST,
            last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
            last_reconciled_status="unknown",
            next_attempt_after=time.time() - 1,
        )
    )

    assert (
        repository.claim_next(
            lease_owner="runner-a",
            policy=LeasePolicy.from_timeout(30),
            max_attempts_by_task={"case-1": 1},
            respect_prepared_reservations=True,
            prepared_ready_task_attempts=set(),
        )
        is None
    )


def test_batch_claim_ready_allowlist_blocks_unknown_rows_without_retry_budget(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-ready-pressure-unknown-budget-spent",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Reconcile unknown submit")],
    )
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id="controller-1",
            last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
            last_reconciled_status="unknown",
            next_attempt_after=time.time() - 1,
        )
    )

    assert (
        repository.claim_next(
            lease_owner="runner-a",
            policy=LeasePolicy.from_timeout(30),
            max_attempts_by_task={"case-1": 1},
            respect_prepared_reservations=True,
            prepared_ready_task_attempts=set(),
        )
        is None
    )


def test_batch_claim_ready_allowlist_still_holds_local_recovery_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-ready-pressure-local-recovery",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Recover local setup")],
    )
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.WAITING_DEVICE, attempts=1))

    assert (
        repository.claim_next(
            lease_owner="runner-a",
            policy=LeasePolicy.from_timeout(30),
            respect_prepared_reservations=True,
            prepared_ready_task_attempts=set(),
        )
        is None
    )


def test_batch_claim_ignores_prepared_rows_when_policy_disabled(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-disabled-claim",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    assert store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )

    assert repository.claim_next(lease_owner="runner-b", policy=LeasePolicy.from_timeout(30)) is not None


def test_batch_claim_ignores_stale_foreign_prepared_reservation(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-stale-claim",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    assert store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=600,
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE prepared_sandboxes SET updated_at=1 WHERE task_id='case-1'")

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy.from_timeout(30),
        respect_prepared_reservations=True,
        prepared_reservation_stale_after_seconds=30,
    )
    assert claim is not None


def test_prepared_store_expires_stale_unassigned_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-stale-expire",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=600,
    )
    assert record is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE prepared_sandboxes SET updated_at=1 WHERE task_id='case-1'")

    expired = store.expire_stale(stale_after_seconds=30)

    assert [record.task_id for record in expired] == ["case-1"]
    assert store.get(task_id="case-1", attempt=1).status == PreparedSandboxStatus.EXPIRED


def test_prepared_planner_uses_batch_global_buffer(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-global-buffer",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Do two"),
        ],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    assert store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    planner = PreparedSandboxPlanner(
        manifest=manifest,
        store=store,
        policy=PreparedSandboxPolicy(enabled=True, buffer=1),
        max_attempts_by_task={"case-1": 1, "case-2": 1},
    )

    assert store.active_count() == 1
    assert store.active_count(runner_id="runner-b") == 0
    assert planner.candidates(run_concurrency=2, runner_id="runner-b") == []


def test_prepared_planner_reuses_released_attempt_history(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-released-history",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Do two"),
            BatchTaskSpec(task_id="case-3", agent_kind="desktop", query="Do three"),
        ],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assigned = store.assign(task_id="case-1", attempt=1, runner_id="runner-a")
    assert assigned is not None
    released = store.mark_released(assigned)
    assert released is not None

    planner = PreparedSandboxPlanner(
        manifest=manifest,
        store=store,
        policy=PreparedSandboxPolicy(enabled=True, buffer=2),
        max_attempts_by_task={"case-1": 1, "case-2": 1, "case-3": 1},
    )

    candidates = planner.candidates(run_concurrency=2, runner_id="runner-b")

    assert [candidate.task.task_id for candidate in candidates] == ["case-1", "case-2"]
    replacement = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-b",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert replacement is not None
    assert replacement.status == PreparedSandboxStatus.RESERVED
    assert replacement.runner_id == "runner-b"
    assert replacement.sandbox_id == ""
    assert replacement.device_id == ""
    assert replacement.lease_id == ""
    assert replacement.released_at == 0
    assert replacement.error == ""


def test_prepared_planner_allows_next_attempt_after_released_history(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-released-next-attempt",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one")],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assigned = store.assign(task_id="case-1", attempt=1, runner_id="runner-a")
    assert assigned is not None
    assert store.mark_released(assigned) is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE batch_tasks SET status=?, attempts=1 WHERE task_id='case-1'",
            (BatchTaskStatus.FAILED.value,),
        )
    planner = PreparedSandboxPlanner(
        manifest=manifest,
        store=store,
        policy=PreparedSandboxPolicy(enabled=True, buffer=1),
        max_attempts_by_task={"case-1": 2},
    )

    candidates = planner.candidates(run_concurrency=1, runner_id="runner-b")

    assert [(candidate.task.task_id, candidate.attempt) for candidate in candidates] == [("case-1", 2)]
    assert (
        store.reserve(
            task_id="case-1",
            attempt=2,
            runner_id="runner-b",
            depth=PreparedSandboxDepth.FULL_INIT,
            provider="fake",
            template="desktop",
            ttl_seconds=60,
        )
        is not None
    )


def test_foreign_claim_blocking_count_ignores_nonclaimable_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(
        BatchManifest(
            batch_id="prepared-foreign-blockers",
            tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one")],
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    assert store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert store.active_count() == 1
    assert (
        store.foreign_claim_blocking_count(
            runner_id="runner-b",
            max_attempts_by_task={"case-1": 1},
            stale_after_seconds=30,
        )
        == 1
    )

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE batch_tasks SET status=? WHERE task_id='case-1'", (BatchTaskStatus.SUCCEEDED.value,))

    assert store.active_count() == 1
    assert (
        store.foreign_claim_blocking_count(
            runner_id="runner-b",
            max_attempts_by_task={"case-1": 1},
            stale_after_seconds=30,
        )
        == 0
    )


def test_prepared_ttl_does_not_expire_assigned_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(
        BatchManifest(
            batch_id="prepared-ttl",
            tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=0.01,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox",
        device_id="device",
        lease_id="lease",
        prepare_seconds=0.1,
    )
    assert prepared is not None
    assigned = store.assign(task_id="case-1", attempt=1, runner_id="runner-a")
    assert assigned is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE prepared_sandboxes SET expires_at=1 WHERE task_id='case-1'")

    assert store.expire_old() == []
    assert store.get(task_id="case-1", attempt=1).status == PreparedSandboxStatus.ASSIGNED


def test_prepared_failed_attempt_can_be_reused_for_same_prepare_candidate(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-failed-candidate",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Do two"),
        ],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert record is not None
    assert store.mark_failed(record, error="boom") is not None

    assert store.claimable_task_ids(max_attempts_by_task={"case-1": 1, "case-2": 1}, limit=2) == [
        ("case-1", 1),
        ("case-2", 1),
    ]

    replacement = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-b",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert replacement is not None
    assert replacement.status == PreparedSandboxStatus.RESERVED
    assert replacement.runner_id == "runner-b"
    assert replacement.sandbox_id == ""
    assert replacement.error == ""


def test_prepared_expired_attempt_can_be_reused_for_same_prepare_candidate(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-expired-reuse",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert record is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE prepared_sandboxes SET expires_at=1 WHERE task_id='case-1'")
    expired = store.expire_old()

    assert [item.task_id for item in expired] == ["case-1"]
    assert store.claimable_task_ids(max_attempts_by_task={"case-1": 1}, limit=1) == [("case-1", 1)]
    replacement = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-b",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert replacement is not None
    assert replacement.status == PreparedSandboxStatus.RESERVED
    assert replacement.runner_id == "runner-b"


def test_prepared_cleanup_failed_row_blocks_same_attempt_replacement(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-cleanup-block",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert record is not None
    assert store.mark_cleanup_failed(record, error="release failed") is not None

    assert (
        store.reserve(
            task_id="case-1",
            attempt=1,
            runner_id="runner-b",
            depth=PreparedSandboxDepth.FULL_INIT,
            provider="fake",
            template="desktop",
            ttl_seconds=60,
        )
        is None
    )
    assert store.claimable_task_ids(max_attempts_by_task={"case-1": 1}, limit=1) == []


def test_prepared_canceled_attempt_can_be_reused_for_same_prepare_candidate(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-canceled-reuse",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert record is not None
    assert store.mark_canceled(record) is not None

    assert store.claimable_task_ids(max_attempts_by_task={"case-1": 1}, limit=1) == [("case-1", 1)]
    replacement = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-b",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert replacement is not None
    assert replacement.status == PreparedSandboxStatus.RESERVED
    assert replacement.runner_id == "runner-b"


def test_prepared_claimable_task_ids_scan_across_blocked_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-claimable-chunks",
        tasks=[BatchTaskSpec(task_id=f"case-{index}", agent_kind="desktop", query="Do it") for index in range(1, 7)],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    monkeypatch.setattr(sqlite_store_module, "_CLAIMABLE_TASK_SCAN_LIMIT", 2)
    for index in range(1, 4):
        assert (
            store.reserve(
                task_id=f"case-{index}",
                attempt=1,
                runner_id="runner-a",
                depth=PreparedSandboxDepth.FULL_INIT,
                provider="fake",
                template="desktop",
                ttl_seconds=60,
            )
            is not None
        )

    assert store.claimable_task_ids(max_attempts_by_task={f"case-{index}": 1 for index in range(1, 7)}, limit=2) == [
        ("case-4", 1),
        ("case-5", 1),
    ]


def test_prepared_claimable_task_ids_skips_ineligible_rows_across_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-claimable-skips",
        tasks=[BatchTaskSpec(task_id=f"case-{index}", agent_kind="desktop", query="Do it") for index in range(1, 8)],
    )
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    monkeypatch.setattr(sqlite_store_module, "_CLAIMABLE_TASK_SCAN_LIMIT", 2)
    now = time.time()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE batch_tasks SET lease_token='live', lease_expires_at=? WHERE task_id='case-1'",
            (now + 600,),
        )
        connection.execute("UPDATE batch_tasks SET controller_task_id='controller-2' WHERE task_id='case-2'")
        connection.execute("UPDATE batch_tasks SET attempts=1 WHERE task_id='case-3'")
        connection.execute("UPDATE batch_tasks SET cancel_requested=1 WHERE task_id='case-4'")
    assert (
        store.reserve(
            task_id="case-5",
            attempt=1,
            runner_id="runner-a",
            depth=PreparedSandboxDepth.FULL_INIT,
            provider="fake",
            template="desktop",
            ttl_seconds=60,
        )
        is not None
    )

    max_attempts = {f"case-{index}": 1 for index in range(1, 8)}
    assert store.claimable_task_ids(max_attempts_by_task=max_attempts, limit=2) == [
        ("case-6", 1),
        ("case-7", 1),
    ]


def test_prepared_store_creates_claim_order_index_for_existing_batch_table(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(
        BatchManifest(
            batch_id="prepared-claim-index-compat",
            tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
        )
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP INDEX idx_batch_tasks_claim_order")

    SQLitePreparedSandboxStore(db_path)

    with sqlite3.connect(db_path) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_batch_tasks_claim_order'"
            ).fetchone()
            is not None
        )


def test_prepared_claimable_task_ids_uses_order_keyset_index(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-claimable-plan",
        tasks=[BatchTaskSpec(task_id=f"case-{index}", agent_kind="desktop", query="Do it") for index in range(1, 4)],
    )
    repository.initialize_manifest(manifest)
    with sqlite3.connect(db_path) as connection:
        plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT task_id, state_json, status, attempts, lease_token, lease_expires_at,
                   cancel_requested, controller_task_id, last_failure_category,
                   recovery_action, order_index
            FROM batch_tasks INDEXED BY idx_batch_tasks_claim_order
            WHERE cancel_requested=0
              AND order_index > ?
              AND status IN ({placeholders})
              AND controller_task_id=''
              AND (
                lease_token=''
                OR lease_expires_at <= ?
              )
            ORDER BY order_index ASC
            LIMIT ?
            """.format(placeholders=",".join("?" for _ in sqlite_store_module._CLAIMABLE_TASK_STATUSES)),
            (-1, *sorted(sqlite_store_module._CLAIMABLE_TASK_STATUSES), 0, 2048),
        ).fetchall()
    plan_text = "\n".join(str(row) for row in plan)
    assert "idx_batch_tasks_claim_order" in plan_text
    assert "USE TEMP B-TREE" not in plan_text


def test_prepared_reserve_enforces_global_buffer_atomically(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(
        BatchManifest(
            batch_id="prepared-buffer-cas",
            tasks=[
                BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one"),
                BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Do two"),
            ],
        )
    )
    store_a = SQLitePreparedSandboxStore(db_path)
    store_b = SQLitePreparedSandboxStore(db_path)

    assert store_a.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
        buffer_limit=1,
    )
    assert (
        store_b.reserve(
            task_id="case-2",
            attempt=1,
            runner_id="runner-b",
            depth=PreparedSandboxDepth.FULL_INIT,
            provider="fake",
            template="desktop",
            ttl_seconds=60,
            buffer_limit=1,
        )
        is None
    )


def test_prepared_reserve_enforces_live_limit_with_assigned_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(
        BatchManifest(
            batch_id="prepared-live-limit",
            tasks=[
                BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one"),
                BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Do two"),
            ],
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
        live_limit=1,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-1",
        device_id="device-1",
        lease_id="lease-1",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assigned = store.assign(task_id="case-1", attempt=1, runner_id="runner-a")
    assert assigned is not None
    assert store.active_count() == 0
    assert store.live_count() == 1

    assert (
        store.reserve(
            task_id="case-2",
            attempt=1,
            runner_id="runner-b",
            depth=PreparedSandboxDepth.FULL_INIT,
            provider="fake",
            template="desktop",
            ttl_seconds=60,
            live_limit=1,
        )
        is None
    )


def test_prepared_planner_caps_total_live_prepared_capacity(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    manifest = BatchManifest(
        batch_id="prepared-planner-live-cap",
        tasks=[
            BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do one"),
            BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="Do two"),
            BatchTaskSpec(task_id="case-3", agent_kind="desktop", query="Do three"),
            BatchTaskSpec(task_id="case-4", agent_kind="desktop", query="Do four"),
        ],
    )
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(db_path)
    for index in range(1, 4):
        record = store.reserve(
            task_id=f"case-{index}",
            attempt=1,
            runner_id="runner-a",
            depth=PreparedSandboxDepth.FULL_INIT,
            provider="fake",
            template="desktop",
            ttl_seconds=60,
        )
        assert record is not None
        preparing = store.mark_preparing(record)
        assert preparing is not None
        prepared = store.mark_prepared(
            preparing,
            sandbox_id=f"sandbox-{index}",
            device_id=f"device-{index}",
            lease_id=f"lease-{index}",
            prepare_seconds=1.0,
        )
        assert prepared is not None
        assert store.assign(task_id=f"case-{index}", attempt=1, runner_id="runner-a") is not None
    planner = PreparedSandboxPlanner(
        manifest=manifest,
        store=store,
        policy=PreparedSandboxPolicy(enabled=True, buffer=1),
        max_attempts_by_task={task.task_id: 1 for task in manifest.tasks},
    )

    assert planner.candidates(run_concurrency=2, runner_id="runner-a") == []


def test_prepared_claim_blocking_is_attempt_specific(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(
        BatchManifest(
            batch_id="prepared-attempt-specific",
            tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it", max_retries=1)],
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    assert store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=60,
    )
    assert repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.FAILED, attempts=1))

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy.from_timeout(30),
        max_attempts_by_task={"case-1": 2},
        respect_prepared_reservations=True,
    )

    assert claim is not None
    assert claim.state.attempts == 1


def test_prepared_planner_respects_batch_control_pause(tmp_path: Path) -> None:
    db_path = tmp_path / "batch.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    manifest = BatchManifest(
        batch_id="prepared-paused",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    repository.initialize_manifest(manifest)
    repository.set_control_flag("paused", True)
    store = SQLitePreparedSandboxStore(db_path)

    assert store.claimable_task_ids(max_attempts_by_task={"case-1": 1}, limit=1) == []
    assert (
        store.reserve(
            task_id="case-1",
            attempt=1,
            runner_id="runner-a",
            depth=PreparedSandboxDepth.FULL_INIT,
            provider="fake",
            template="desktop",
            ttl_seconds=60,
        )
        is None
    )


@pytest.mark.asyncio
async def test_prepared_queue_runs_full_init_before_controller_submit(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    RecordingSubmitter.provider = provider
    RecordingSubmitter.command_counts_at_submit = []
    RecordingSubmitter.command_labels_at_submit = []
    RecordingSubmitter.submit_times = []
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-demo",
        tasks=[
            BatchTaskSpec(
                task_id="case-1",
                agent_kind="desktop",
                query="Use the open file.",
                startup_commands=[SandboxCommandSpec(label="open-file", command="echo open")],
            ),
            BatchTaskSpec(
                task_id="case-2",
                agent_kind="desktop",
                query="Use the open app.",
                startup_commands=[SandboxCommandSpec(label="open-app", command="echo app")],
            ),
        ],
    )
    manifest.defaults.concurrency = 2
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    manifest.defaults.startup_commands = [SandboxCommandSpec(label="extract-workspace", command="echo setup")]

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: RecordingSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            concurrency=2,
            sandbox_create_concurrency=1,
            prepared_sandbox_policy=PreparedSandboxPolicy(
                enabled=True,
                depth=PreparedSandboxDepth.FULL_INIT,
                buffer=2,
                prepare_concurrency=1,
                claim_wait_seconds=2,
                claim_pending_wait_seconds=2,
            ),
        )
    )

    assert result.succeeded == 2
    assert min(RecordingSubmitter.command_counts_at_submit) >= 3
    assert any("open-file" in labels for labels in RecordingSubmitter.command_labels_at_submit)
    assert any("open-app" in labels for labels in RecordingSubmitter.command_labels_at_submit)
    summary = json.loads((tmp_path / "prepared-demo" / "batch_summary.json").read_text(encoding="utf-8"))
    prepared_hits = summary["metrics"]["prepared_sandbox_hits"]
    prepared_misses = summary["metrics"]["prepared_sandbox_misses"]
    # The warmup gate only waits for the first ready sandbox before starting the
    # claim loop; with prepare_concurrency=1 the second task may legally fall
    # back to normal create if it wins the race against the second prepared row.
    assert prepared_hits >= 1
    assert prepared_hits + prepared_misses == 2
    assert summary["metrics"]["prepared_sandbox_estimated_saved_seconds"] >= 0
    assert summary["metrics"]["prepared_sandbox_avg_estimated_saved_seconds"] >= 0
    assert summary["telemetry"]["summary"]["prepared_sandbox_hits"] == prepared_hits
    assert summary["telemetry"]["summary"]["prepared_sandbox_misses"] == prepared_misses
    assert summary["telemetry"]["summary"]["prepared_sandbox_estimated_saved_seconds"] >= 0
    prepared_events = [
        json.loads(line)
        for line in (tmp_path / "prepared-demo" / "prepared_sandbox_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    prepared_stage_events = [event for event in prepared_events if event["message"] == "prepared_sandbox_stage_timing"]
    assert any(event["details"].get("stage") == "prepared_db_commit" for event in prepared_stage_events)
    ready_events = [event for event in prepared_events if event["message"] == "prepared_sandbox_ready"]
    assert ready_events
    ready_timing = ready_events[0]["details"]["timing"]
    assert ready_timing["candidate_scheduled_to_semaphore_acquired_seconds"] >= 0
    assert ready_timing["coordinator_acquire_seconds"] >= 0
    assert ready_timing["prepared_ready_persist_seconds"] >= 0
    sandbox_events = [
        json.loads(line)
        for line in (tmp_path / "prepared-demo" / "sandbox_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    sandbox_stage_events = [event for event in sandbox_events if event["message"] == "sandbox_stage_timing"]
    sandbox_stages = {event["metadata"].get("stage") for event in sandbox_stage_events}
    assert {"sandbox_create", "startup_commands", "device_worker_start", "device_readiness_wait"} <= sandbox_stages
    prepared_handles = [
        handle for handle in provider.handles.values() if handle.metadata.get("prepared_sandbox") == "true"
    ]
    assert prepared_handles
    assert prepared_handles[0].metadata["northcua"] == "true"
    assert prepared_handles[0].metadata["batch_id"] == "prepared-demo"
    assert prepared_handles[0].metadata["task_id"] in {"case-1", "case-2"}
    assert prepared_handles[0].metadata["attempt"] == "1"
    with sqlite3.connect(tmp_path / "prepared-demo" / "batch_state.sqlite") as connection:
        statuses = [
            row[0] for row in connection.execute("SELECT status FROM prepared_sandboxes ORDER BY task_id").fetchall()
        ]
    assert statuses.count(PreparedSandboxStatus.RELEASED.value) == prepared_hits
    assert set(statuses) <= {
        PreparedSandboxStatus.RELEASED.value,
        PreparedSandboxStatus.CANCELED.value,
    }


@pytest.mark.asyncio
async def test_prepared_fixture_setup_runs_deferred_init_before_submit(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    RecordingSubmitter.provider = provider
    RecordingSubmitter.command_counts_at_submit = []
    RecordingSubmitter.command_labels_at_submit = []
    RecordingSubmitter.submit_times = []
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-fixture-setup",
        tasks=[
            BatchTaskSpec(
                task_id="case-1",
                agent_kind="desktop",
                query="Use the opened file.",
                startup_commands=[SandboxCommandSpec(label="open-file", command="echo open")],
            )
        ],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    manifest.defaults.startup_commands = [SandboxCommandSpec(label="extract-workspace", command="echo setup")]

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: RecordingSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(
                enabled=True,
                depth=PreparedSandboxDepth.FIXTURE_SETUP,
                buffer=1,
                prepare_concurrency=1,
                claim_wait_seconds=2,
                claim_pending_wait_seconds=2,
            ),
        )
    )

    assert result.succeeded == 1
    labels = RecordingSubmitter.command_labels_at_submit[0]
    assert "extract-workspace" in labels
    assert "open-file" in labels


@pytest.mark.asyncio
async def test_prepared_deferred_startup_failure_releases_assigned_sandbox(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    RecordingSubmitter.provider = provider
    RecordingSubmitter.command_counts_at_submit = []
    RecordingSubmitter.command_labels_at_submit = []
    RecordingSubmitter.submit_times = []
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-deferred-failure",
        tasks=[
            BatchTaskSpec(
                task_id="case-1",
                agent_kind="desktop",
                query="Use the opened file.",
                startup_commands=[SandboxCommandSpec(label="open-file", command="fail open")],
            )
        ],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    manifest.defaults.startup_commands = [SandboxCommandSpec(label="extract-workspace", command="echo setup")]

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: RecordingSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(
                enabled=True,
                depth=PreparedSandboxDepth.FIXTURE_SETUP,
                buffer=1,
                prepare_concurrency=1,
                claim_wait_seconds=2,
            ),
        )
    )

    assert result.failed == 1
    assert RecordingSubmitter.command_counts_at_submit == []
    assert provider.terminated == ["fake-sandbox-1"]
    with sqlite3.connect(tmp_path / "prepared-deferred-failure" / "batch_state.sqlite") as connection:
        status = connection.execute("SELECT status FROM prepared_sandboxes WHERE task_id='case-1'").fetchone()[0]
        cleanup_status = connection.execute("SELECT cleanup_status FROM batch_tasks WHERE task_id='case-1'").fetchone()[
            0
        ]
    assert status == PreparedSandboxStatus.RELEASED.value
    assert cleanup_status == "ok"


@pytest.mark.asyncio
async def test_prepared_cleanup_failure_keeps_row_unreleased(tmp_path: Path) -> None:
    provider = TerminateFailsProvider()
    RecordingSubmitter.provider = provider
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-cleanup-failure",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: RecordingSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1, prepare_concurrency=1),
        )
    )

    assert result.succeeded == 1
    with sqlite3.connect(tmp_path / "prepared-cleanup-failure" / "batch_state.sqlite") as connection:
        status = connection.execute("SELECT status FROM prepared_sandboxes WHERE task_id='case-1'").fetchone()[0]
        cleanup_status = connection.execute("SELECT cleanup_status FROM batch_tasks WHERE task_id='case-1'").fetchone()[
            0
        ]
    assert status == PreparedSandboxStatus.CLEANUP_FAILED.value
    assert cleanup_status.startswith("failed:")
    summary = json.loads((tmp_path / "prepared-cleanup-failure" / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["prepared_sandbox_failures"] == 0
    assert summary["metrics"]["prepared_sandbox_cleanup_failures"] == 1


@pytest.mark.asyncio
async def test_prepared_claim_rechecks_device_readiness_before_submit(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    RecordingSubmitter.provider = provider
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-readiness-recheck",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})

    readiness_probe = SequenceReadinessProbe(
        [SandboxReadinessState.READY, SandboxReadinessState.TIMED_OUT, SandboxReadinessState.READY]
    )
    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: RecordingSubmitter(url),
        readiness_probe=readiness_probe,
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            device_readiness_timeout_seconds=123,
            prepared_sandbox_policy=PreparedSandboxPolicy(
                enabled=True,
                buffer=1,
                prepare_concurrency=1,
                claim_wait_seconds=1,
                claim_pending_wait_seconds=1,
            ),
        )
    )

    assert result.succeeded == 1
    assert len(provider.handles) == 2
    assert readiness_probe.timeout_seconds_seen[:3] == [123, 123, 123]
    summary = json.loads((tmp_path / "prepared-readiness-recheck" / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["prepared_sandbox_misses"] == 1


@pytest.mark.asyncio
async def test_prepare_failure_falls_back_without_consuming_attempt_budget(tmp_path: Path) -> None:
    provider = PreparedCreateFailsOnceProvider()
    PreparedCreateFailsOnceProvider.prepared_failures_remaining = 1
    RecordingSubmitter.provider = provider
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-fallback",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: RecordingSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(
                enabled=True,
                buffer=1,
                prepare_concurrency=1,
                claim_wait_seconds=1,
            ),
        )
    )

    assert result.succeeded == 1
    assert result.task_states[0].attempts == 1
    summary = json.loads((tmp_path / "prepared-fallback" / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["prepared_sandbox_failures"] == 1
    prepared_events = [
        json.loads(line)
        for line in (tmp_path / "prepared-fallback" / "prepared_sandbox_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failure_events = [event for event in prepared_events if event["message"] == "prepared_sandbox_failed"]
    assert any(
        event["details"].get("failure_kind") == PreparedSandboxFailureKind.CREATION_FAILED.value
        for event in failure_events
    )
    assert not any(
        event["details"].get("failure_kind") == PreparedSandboxFailureKind.CLAIM_WAIT_TIMEOUT_CANCELLED_FUTURE.value
        for event in failure_events
    )


@pytest.mark.asyncio
async def test_prepared_initial_warmup_returns_after_fast_prepare_failure(tmp_path: Path) -> None:
    class AlwaysFailsPreparedProvider(FakeSandboxProvider):
        async def create(self, spec: SandboxProviderSpec):
            if spec.metadata.get("prepared_sandbox") == "true":
                raise RuntimeError("prepared create failed")
            return await super().create(spec)

    provider = AlwaysFailsPreparedProvider()
    RecordingSubmitter.provider = provider
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-fast-failure-warmup",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})

    started = time.monotonic()
    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: RecordingSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            sandbox_create_concurrency=1,
            prepared_sandbox_policy=PreparedSandboxPolicy(
                enabled=True,
                buffer=1,
                prepare_concurrency=1,
                claim_wait_seconds=30,
            ),
        )
    )

    assert result.succeeded == 1
    assert time.monotonic() - started < 2


@pytest.mark.asyncio
async def test_prepared_initial_warmup_releases_after_first_ready_not_full_buffer(tmp_path: Path) -> None:
    provider = SecondPreparedCreatePausesProvider()
    RecordingSubmitter.provider = provider
    RecordingSubmitter.command_counts_at_submit = []
    RecordingSubmitter.command_labels_at_submit = []
    RecordingSubmitter.submit_times = []
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-first-ready-starts-claim-loop",
        tasks=[BatchTaskSpec(task_id=f"case-{index}", agent_kind="desktop", query="Finish.") for index in range(1, 4)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})

    started = time.monotonic()
    run_task = asyncio.create_task(
        BatchRunner(
            provider_registry=registry,
            submitter_factory=lambda url: RecordingSubmitter(url),
        ).run(
            BatchRunConfig(
                manifest=manifest,
                output_dir=tmp_path,
                concurrency=3,
                sandbox_create_concurrency=1,
                prepared_sandbox_policy=PreparedSandboxPolicy(
                    enabled=True,
                    buffer=3,
                    prepare_concurrency=1,
                    initial_warmup_wait_seconds=5,
                    claim_wait_seconds=0.01,
                ),
            )
        )
    )
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not RecordingSubmitter.submit_times:
            await asyncio.sleep(0.01)
        assert RecordingSubmitter.submit_times
        assert RecordingSubmitter.submit_times[0] - started < 0.5
    finally:
        provider.release_second.set()

    result = await asyncio.wait_for(run_task, timeout=3)

    assert result.succeeded == 3
    output_dir = tmp_path / manifest.batch_id
    warmup_finished = _prepared_events(output_dir, message="prepared_sandbox_initial_warmup_finished")
    assert warmup_finished
    details = warmup_finished[0]["details"]
    assert details["min_ready"] == 1
    assert details["timeout_seconds"] == 5
    assert details["exit_reason"] == "ready"
    assert details["ready_count"] >= 1


@pytest.mark.asyncio
async def test_prepared_claim_timeout_falls_back_without_waiting_for_cleanup(tmp_path: Path) -> None:
    provider = PreparedCreateSlowProvider()
    RecordingSubmitter.provider = provider
    RecordingSubmitter.command_counts_at_submit = []
    RecordingSubmitter.command_labels_at_submit = []
    RecordingSubmitter.submit_times = []
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-timeout-nonblocking",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})

    started = time.monotonic()
    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: RecordingSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            sandbox_create_concurrency=1,
            prepared_sandbox_policy=PreparedSandboxPolicy(
                enabled=True,
                buffer=1,
                prepare_concurrency=1,
                initial_warmup_wait_seconds=0.01,
                claim_wait_seconds=0.01,
                claim_pending_wait_seconds=0.01,
            ),
        )
    )

    assert result.succeeded == 1
    assert "-prep-" not in result.task_states[0].device_id
    assert RecordingSubmitter.submit_times
    assert RecordingSubmitter.submit_times[0] - started < 0.25
    summary = json.loads((tmp_path / "prepared-timeout-nonblocking" / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["prepared_sandbox_misses"] >= 1
    await asyncio.sleep(0.05)
    with sqlite3.connect(tmp_path / "prepared-timeout-nonblocking" / "batch_state.sqlite") as connection:
        row = connection.execute("SELECT status, error FROM prepared_sandboxes WHERE task_id='case-1'").fetchone()
    assert row[0] == PreparedSandboxStatus.FAILED.value
    assert row[1] != PreparedSandboxFailureKind.CLAIM_WAIT_TIMEOUT_CANCELLED_FUTURE.value
    prepared_events = [
        json.loads(line)
        for line in (tmp_path / "prepared-timeout-nonblocking" / "prepared_sandbox_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failure_events = [event for event in prepared_events if event["message"] == "prepared_sandbox_failed"]
    assert not any(
        event["details"].get("failure_kind") == PreparedSandboxFailureKind.CLAIM_WAIT_TIMEOUT_CANCELLED_FUTURE.value
        for event in failure_events
    )
    fallback_events = [event for event in prepared_events if event["message"] == "prepared_sandbox_fallback"]
    assert any(
        event["details"].get("reason") == "timeout" and event["details"].get("prepare_future_continued") is True
        for event in fallback_events
    )


@pytest.mark.asyncio
async def test_prepared_claim_ignores_pre_reserve_future_without_waiting(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-timeout-future-continues",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1, claim_wait_seconds=30),
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=PreparedSandboxPolicy(enabled=True, buffer=1, claim_wait_seconds=0.01),
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )
    future: asyncio.Future[PreparedSandboxAssignment | None] = asyncio.Future()
    key = ("case-1", 1)
    manager._futures[key] = future  # noqa: SLF001 - white-box pre-reserve claim edge.

    started = time.monotonic()
    assignment = await manager.claim(task=manifest.tasks[0], attempt=1)

    assert assignment is None
    assert time.monotonic() - started < 1
    assert not future.cancelled()
    assert not future.done()
    fallback_events = _prepared_fallback_events(tmp_path)
    assert len(fallback_events) == 1
    assert fallback_events[0]["details"]["reason"] == "miss"
    assert fallback_events[0]["details"]["claim_visible_future"] is False
    future.set_result(None)
    manager._prune_completed_futures()  # noqa: SLF001
    assert key not in manager._futures  # noqa: SLF001
    await manager.close()


@pytest.mark.asyncio
async def test_prepared_claim_miss_releases_late_assignment_before_manager_close(tmp_path: Path) -> None:
    provider = PreparedCreateSlowProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-missed-late-assignment-release",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(enabled=True, buffer=1, prepare_concurrency=1, claim_wait_seconds=30)
    manager = PreparedSandboxManager(
        config=BatchRunConfig(manifest=manifest, output_dir=tmp_path, prepared_sandbox_policy=policy),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=policy,
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )

    try:
        await manager.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and ("case-1", 1) not in manager._claim_visible_futures:  # noqa: SLF001
            await asyncio.sleep(0.01)
        assert ("case-1", 1) in manager._claim_visible_futures  # noqa: SLF001

        assignment = await manager.claim(task=manifest.tasks[0], attempt=1)

        assert assignment is None
        fallback_events = _prepared_fallback_events(tmp_path)
        assert fallback_events[-1]["details"]["reason"] == PreparedSandboxFallbackReason.MISS.value
        assert fallback_events[-1]["details"]["claim_visible_future"] is True

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and (
            "fake-sandbox-1" not in provider.terminated or ("case-1", 1) in manager._assignments  # noqa: SLF001
        ):
            await asyncio.sleep(0.01)

        assert "fake-sandbox-1" in provider.terminated
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and ("case-1", 1) in manager._assignments:  # noqa: SLF001
            await asyncio.sleep(0.01)
        assert ("case-1", 1) not in manager._assignments  # noqa: SLF001
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_prepared_abandoned_cleanup_not_double_released_by_close(tmp_path: Path) -> None:
    provider = PreparedSlowTerminateWaitsProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-abandoned-cleanup-close-race",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(enabled=True, buffer=1, prepare_concurrency=1, claim_wait_seconds=30)
    manager = PreparedSandboxManager(
        config=BatchRunConfig(manifest=manifest, output_dir=tmp_path, prepared_sandbox_policy=policy),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=policy,
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )

    await manager.start()
    try:
        key = ("case-1", 1)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and key not in manager._claim_visible_futures:  # noqa: SLF001
            await asyncio.sleep(0.01)
        assert key in manager._claim_visible_futures  # noqa: SLF001

        assignment = await manager.claim(task=manifest.tasks[0], attempt=1)

        assert assignment is None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and key not in manager._assignments:  # noqa: SLF001
            await asyncio.sleep(0.01)
        assert key in manager._assignments  # noqa: SLF001
        assert key in manager._cleanup_inflight_keys  # noqa: SLF001

        close_task = asyncio.create_task(manager.close())
        await asyncio.sleep(0.05)
        provider.release_termination.set()
        await asyncio.wait_for(close_task, timeout=1)

        assert provider.terminated == ["fake-sandbox-1"]
        assert store.get(task_id="case-1", attempt=1).status == PreparedSandboxStatus.CANCELED
    finally:
        provider.release_termination.set()
        if not manager._closed.is_set():  # noqa: SLF001
            await manager.close()


@pytest.mark.asyncio
async def test_prepared_claim_rejects_assignment_when_remaining_ttl_cannot_cover_timeout(tmp_path: Path) -> None:
    provider = NoTimeoutExtensionFakeProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-ttl-guard-rejects",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.timeout_seconds = 1000.0
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(
        enabled=True,
        buffer=1,
        prepare_concurrency=1,
        ttl_seconds=0.5,
        claim_wait_seconds=1,
        assignment_ttl_margin_seconds=0,
    )
    manager = PreparedSandboxManager(
        config=BatchRunConfig(manifest=manifest, output_dir=tmp_path, prepared_sandbox_policy=policy),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=policy,
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )

    try:
        await manager.start()
        await manager.wait_for_initial_ready(min_ready=1, timeout_seconds=1)
        created = next(iter(provider.handles.values()))
        assert created.metadata["prepared_provider_timeout_seconds"] == 1001

        key = ("case-1", 1)
        existing = manager._assignments[key]  # noqa: SLF001 - targeted legacy TTL guard setup.
        manager._assignments[key] = PreparedSandboxAssignment(  # noqa: SLF001
            record=existing.record.model_copy(
                update={
                    "metadata": {
                        **existing.record.metadata,
                        "prepared_provider_timeout_seconds": 0.5,
                    }
                }
            ),
            lease=existing.lease,
            lease_context=existing.lease_context,
        )

        assignment = await manager.claim(task=manifest.tasks[0], attempt=1)

        assert assignment is None
        fallback_events = _prepared_fallback_events(tmp_path)
        fallback = fallback_events[-1]
        assert fallback["details"]["reason"] == PreparedSandboxFallbackReason.INSUFFICIENT_REMAINING_TTL.value
        assert fallback["details"]["result"] == "ttl_guard_rejected"
        assert (
            fallback["details"]["estimated_provider_remaining_ttl_seconds_at_assignment"]
            < fallback["details"]["assignment_ttl_guard_seconds"]
        )
        with sqlite3.connect(tmp_path / "batch_state.sqlite") as connection:
            row = connection.execute("SELECT status FROM prepared_sandboxes WHERE task_id='case-1'").fetchone()
        assert row[0] == PreparedSandboxStatus.CANCELED.value
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_prepared_claim_assignment_emits_ttl_guard_telemetry(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-ttl-guard-hit",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.timeout_seconds = 1.0
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(
        enabled=True,
        buffer=1,
        prepare_concurrency=1,
        ttl_seconds=5,
        claim_wait_seconds=1,
        assignment_ttl_margin_seconds=0,
    )
    manager = PreparedSandboxManager(
        config=BatchRunConfig(manifest=manifest, output_dir=tmp_path, prepared_sandbox_policy=policy),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=policy,
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )

    assignment: PreparedSandboxAssignment | None = None
    try:
        await manager.start()
        await manager.wait_for_initial_ready(min_ready=1, timeout_seconds=1)

        assignment = await manager.claim(task=manifest.tasks[0], attempt=1)

        assert assignment is not None
        prepared_events = [
            json.loads(line)
            for line in (tmp_path / "prepared_sandbox_events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assigned_event = next(event for event in prepared_events if event["message"] == "prepared_sandbox_assigned")
        details = assigned_event["details"]
        assert details["assignment_ttl_guard_passed"] is True
        assert details["case_timeout_seconds"] == 1.0
        assert details["assignment_ttl_guard_seconds"] == 1.0
        assert details["provider_timeout_seconds"] == 300.0
        assert details["prepared_required_provider_timeout_seconds"] == 6.0
        assert provider.extended_timeouts == []
        assert details["prepared_age_seconds_at_assignment"] >= 0.0
        assert not any(event["message"] == "prepared_sandbox_lease_extended" for event in prepared_events)
    finally:
        if assignment is not None:
            await manager._release_helper.try_release_prepared_context(  # noqa: SLF001
                task=manifest.tasks[0],
                attempt=1,
                context=assignment.lease_context,
                close_reason="test cleanup",
            )
            manager.mark_released(assignment)
        await manager.close()


@pytest.mark.asyncio
async def test_attempt_failure_records_prepared_ttl_cliff_for_websocket_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = BatchManifest(
        batch_id="prepared-failure-ttl-telemetry",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    monkeypatch.setattr(attempt_helpers_module.time, "time", lambda: 1300.0)
    lease = SandboxLease(
        lease_id="lease-1",
        handle=SandboxHandle(sandbox_id="sandbox-1", provider="fake", template="desktop"),
        device_id="device-1",
    )
    assignment = PreparedSandboxAssignment(
        record=PreparedSandboxRecord(
            task_id="case-1",
            attempt=1,
            status=PreparedSandboxStatus.ASSIGNED,
            runner_id="runner",
            sandbox_id="sandbox-1",
            device_id="device-1",
            lease_id="lease-1",
            reserved_at=0.0,
            prepared_at=100.0,
            assigned_at=1000.0,
            expires_at=1200.0,
        ),
        lease=lease,
        lease_context=BatchTaskLeaseContext(lease=lease, manager=object()),  # type: ignore[arg-type]
    )
    state = BatchTaskState(
        task_id="case-1",
        attempts=1,
        status=BatchTaskStatus.RUNNING,
        sandbox_id="sandbox-1",
        device_id="device-1",
    )

    await attempt_helpers_module._record_attempt_failure(  # noqa: SLF001
        state=state,
        task=manifest.tasks[0],
        attempt=1,
        status=BatchTaskStatus.FAILED,
        error=RuntimeError("websocket disconnected"),
        events=events,
        prepared_assignment=assignment,
    )

    attempt_events = [
        json.loads(line) for line in (tmp_path / "batch_attempt_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    details = next(event for event in attempt_events if event["message"] == "attempt_failed")["details"]
    assert details["assigned_runtime_seconds_before_disconnect"] == 300.0
    assert details["remaining_ttl_seconds_at_failure"] == -100.0
    assert details["expired_after_assignment"] is True


def test_controller_terminal_device_transport_records_prepared_assigned_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = BatchManifest(
        batch_id="prepared-terminal-device-transport-telemetry",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    monkeypatch.setattr(attempt_submit_module.time, "time", lambda: 1300.0)
    lease = SandboxLease(
        lease_id="lease-1",
        handle=SandboxHandle(sandbox_id="sandbox-1", provider="fake", template="desktop"),
        device_id="device-1",
    )
    assignment = PreparedSandboxAssignment(
        record=PreparedSandboxRecord(
            task_id="case-1",
            attempt=1,
            status=PreparedSandboxStatus.ASSIGNED,
            runner_id="runner",
            sandbox_id="sandbox-1",
            device_id="device-1",
            lease_id="lease-1",
            reserved_at=0.0,
            prepared_at=100.0,
            assigned_at=1000.0,
            expires_at=1200.0,
        ),
        lease=lease,
        lease_context=BatchTaskLeaseContext(lease=lease, manager=object()),  # type: ignore[arg-type]
    )
    state = BatchTaskState(
        task_id="case-1",
        attempts=1,
        status=BatchTaskStatus.RUNNING,
        sandbox_id="sandbox-1",
        device_id="device-1",
    )

    attempt_submit_module._state_from_terminal_controller_result(  # noqa: SLF001
        accepted=TaskAccepted(task_id="controller-task-1", trace_id="trace-1", session_id="session-1"),
        finished=TaskFinished(
            task_id="controller-task-1",
            trace_id="trace-1",
            session_id="session-1",
            state=TaskState.FAILED,
            device_id="device-1",
            error="device closed",
            terminal_decision={"failure_category": FailureCategory.DEVICE_TRANSPORT.value},
        ),
        task=manifest.tasks[0],
        lease=lease,
        state=state,
        metrics=BatchMetrics(),
        events=events,
        task_start=0.0,
        prepared_assignment=assignment,
    )

    attempt_events = [
        json.loads(line) for line in (tmp_path / "batch_attempt_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    details = next(event for event in attempt_events if event["message"] == "controller_submit_finished")["details"]
    assert details["failure_category"] == FailureCategory.DEVICE_TRANSPORT.value
    assert details["assigned_runtime_seconds_before_disconnect"] == 300.0
    assert details["expired_after_assignment"] is True


@pytest.mark.asyncio
async def test_prepared_claim_timeout_keeps_claim_visible_future_alive(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-timeout-future-continues",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(
                enabled=True,
                buffer=1,
                claim_wait_seconds=0.01,
                claim_pending_wait_seconds=0.01,
            ),
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=PreparedSandboxPolicy(enabled=True, buffer=1, claim_wait_seconds=0.01),
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )
    future: asyncio.Future[PreparedSandboxAssignment | None] = asyncio.Future()
    key = ("case-1", 1)
    manager._futures[key] = future  # noqa: SLF001 - white-box claim timeout edge.
    manager._claim_visible_futures[key] = future  # noqa: SLF001 - simulate post-reserve visibility.

    assignment = await manager.claim(task=manifest.tasks[0], attempt=1)

    assert assignment is None
    assert not future.cancelled()
    assert not future.done()
    fallback_events = _prepared_fallback_events(tmp_path)
    assert len(fallback_events) == 1
    assert fallback_events[0]["details"]["reason"] == "timeout"
    future.set_result(None)
    manager._prune_completed_futures()  # noqa: SLF001
    assert key not in manager._futures  # noqa: SLF001
    assert key not in manager._claim_visible_futures  # noqa: SLF001
    await manager.close()


@pytest.mark.asyncio
async def test_prepared_manager_skips_new_work_while_backpressured(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-waits-for-backpressure",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(enabled=True, buffer=1, prepare_concurrency=1)
    gate = BlockingPrepareBackpressureGate()
    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=policy,
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=policy,
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
        backpressure_gate=gate,  # type: ignore[arg-type]
    )

    await manager.start()
    await asyncio.sleep(0.05)
    assert store.active_count() == 0
    assert provider.handles == {}
    assert manager._futures == {}  # noqa: SLF001
    assert manager._claim_visible_futures == {}  # noqa: SLF001

    gate.active = False
    gate.release.set()
    await manager.wait_for_initial_ready(min_ready=1, timeout_seconds=1)

    assert len(provider.handles) == 1
    assert len(manager.ready_unassigned_task_attempts()) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_prepared_scheduler_skips_initial_warmup_wait_when_backpressured(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-initial-pressure-no-warmup-wait",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(enabled=True, buffer=1, prepare_concurrency=1, claim_wait_seconds=30)
    gate = BlockingPrepareBackpressureGate()

    class _ExecutorWithCoordinator:
        lease_coordinator = BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        )

    scheduler = BatchScheduler(provider_registry=registry, executor=_ExecutorWithCoordinator())  # type: ignore[arg-type]
    started = time.monotonic()
    manager = await scheduler._maybe_start_prepared_manager(  # noqa: SLF001
        config=BatchRunConfig(manifest=manifest, output_dir=tmp_path, prepared_sandbox_policy=policy),
        repository=repository,
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        concurrency=1,
        create_concurrency=1,
        backpressure_gate=gate,  # type: ignore[arg-type]
    )

    assert manager is not None
    assert time.monotonic() - started < 1
    assert provider.handles == {}
    assert manager._futures == {}  # noqa: SLF001
    assert manager._claim_visible_futures == {}  # noqa: SLF001
    await manager.close()


@pytest.mark.asyncio
async def test_prepared_scheduler_initialization_and_candidate_scan_use_prepared_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-init-off-loop",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(enabled=True, buffer=1, prepare_concurrency=1)
    gate = BlockingPrepareBackpressureGate()
    observed_threads: dict[str, str] = {}
    store_type = SQLitePreparedSandboxStore
    original_candidates = PreparedSandboxPlanner.candidates

    def store_factory(path: Path) -> SQLitePreparedSandboxStore:
        observed_threads["store_init"] = threading.current_thread().name
        return store_type(path)

    def candidates(
        planner: PreparedSandboxPlanner,
        *,
        run_concurrency: int,
        runner_id: str,
    ) -> list[object]:
        observed_threads["candidate_scan"] = threading.current_thread().name
        return original_candidates(planner, run_concurrency=run_concurrency, runner_id=runner_id)

    monkeypatch.setattr(scheduler_module, "SQLitePreparedSandboxStore", store_factory)
    monkeypatch.setattr(PreparedSandboxPlanner, "candidates", candidates)

    class _ExecutorWithCoordinator:
        lease_coordinator = BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        )

    scheduler = BatchScheduler(provider_registry=registry, executor=_ExecutorWithCoordinator())  # type: ignore[arg-type]
    manager = await scheduler._maybe_start_prepared_manager(  # noqa: SLF001
        config=BatchRunConfig(manifest=manifest, output_dir=tmp_path, prepared_sandbox_policy=policy),
        repository=repository,
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        concurrency=1,
        create_concurrency=1,
        backpressure_gate=gate,  # type: ignore[arg-type]
    )

    assert manager is not None
    assert observed_threads["store_init"].startswith("northcua-prepared-repository")
    assert observed_threads["candidate_scan"].startswith("northcua-prepared-repository")
    await manager.close()


@pytest.mark.asyncio
async def test_prepared_cleanup_timeout_fallback_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = BatchManifest(
        batch_id="prepared-cleanup-timeout-off-loop",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    policy = PreparedSandboxPolicy(enabled=True, buffer=1, prepare_concurrency=1)
    config = BatchRunConfig(manifest=manifest, output_dir=tmp_path, prepared_sandbox_policy=policy)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    coordinator = BatchTaskLeaseCoordinator(
        provider_registry=registry,
        binding_factory=SandboxDeviceBindingFactory(),
    )
    manager = PreparedSandboxManager(
        config=config,
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=policy,
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=coordinator,
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )
    lease = SandboxLease(
        lease_id="lease-1",
        handle=SandboxHandle(sandbox_id="sandbox-1", provider="fake", template="desktop"),
        device_id="device-1",
    )
    assignment = PreparedSandboxAssignment(
        record=PreparedSandboxRecord(
            task_id="case-1",
            attempt=1,
            status=PreparedSandboxStatus.PREPARED,
            runner_id="runner",
            sandbox_id="sandbox-1",
            device_id="device-1",
            lease_id="lease-1",
        ),
        lease=lease,
        lease_context=BatchTaskLeaseContext(lease=lease, manager=object()),  # type: ignore[arg-type]
    )
    manager._assignments[assignment.key] = assignment  # noqa: SLF001
    async_fallback_called = asyncio.Event()

    class _ReleaseHelper:
        async def release_unassigned_assignment(self, **_kwargs: object) -> None:
            await asyncio.sleep(5.0)

        def mark_cleanup_failed(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("synchronous cleanup fallback must not run on the event loop")

        async def mark_cleanup_failed_async(self, *_args: object, **_kwargs: object) -> None:
            async_fallback_called.set()
            await asyncio.sleep(0.05)

    manager._release_helper = _ReleaseHelper()  # type: ignore[assignment]  # noqa: SLF001
    monkeypatch.setattr(preparation_manager_module, "_CLOSE_CLEANUP_WAIT_SECONDS", 0.01)
    ticks = 0
    stop_ticker = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop_ticker.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    ticker_task = asyncio.create_task(ticker())
    await manager._release_unassigned_assignment(  # noqa: SLF001
        task=manifest.tasks[0],
        assignment=assignment,
        final_status=PreparedSandboxStatus.FAILED,
        error="test timeout",
    )
    stop_ticker.set()
    await ticker_task

    assert async_fallback_called.is_set()
    assert ticks >= 5


@pytest.mark.asyncio
async def test_prepared_scheduler_initial_warmup_uses_short_timeout_not_claim_wait(tmp_path: Path) -> None:
    provider = PreparedCreateSlowProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-initial-warmup-short-timeout",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(
        enabled=True,
        buffer=1,
        prepare_concurrency=1,
        initial_warmup_wait_seconds=0.01,
        claim_wait_seconds=30,
    )

    class _ExecutorWithCoordinator:
        lease_coordinator = BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        )

    scheduler = BatchScheduler(provider_registry=registry, executor=_ExecutorWithCoordinator())  # type: ignore[arg-type]
    started = time.monotonic()
    manager = await scheduler._maybe_start_prepared_manager(  # noqa: SLF001
        config=BatchRunConfig(manifest=manifest, output_dir=tmp_path, prepared_sandbox_policy=policy),
        repository=repository,
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        concurrency=1,
        create_concurrency=1,
        backpressure_gate=scheduler_module.ControllerBackpressureGate(controller_db_path=None, enabled=False),
    )

    assert manager is not None
    assert time.monotonic() - started < 0.2
    warmup_finished = _prepared_events(tmp_path, message="prepared_sandbox_initial_warmup_finished")
    assert warmup_finished
    details = warmup_finished[0]["details"]
    assert details["timeout_seconds"] == 0.01
    assert details["exit_reason"] == "timeout"
    await manager.close()


@pytest.mark.asyncio
async def test_prepared_scheduler_p200_buffer100_does_not_wait_for_full_buffer(tmp_path: Path) -> None:
    provider = SecondPreparedCreatePausesProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-p200-buffer100-first-ready",
        tasks=[
            BatchTaskSpec(task_id=f"case-{index}", agent_kind="desktop", query="Finish.") for index in range(1, 201)
        ],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(
        enabled=True,
        buffer=100,
        prepare_concurrency=1,
        initial_warmup_wait_seconds=5,
        claim_wait_seconds=3600,
    )

    class _ExecutorWithCoordinator:
        lease_coordinator = BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        )

    scheduler = BatchScheduler(provider_registry=registry, executor=_ExecutorWithCoordinator())  # type: ignore[arg-type]
    started = time.monotonic()
    manager: PreparedSandboxManager | None = None
    try:
        manager = await scheduler._maybe_start_prepared_manager(  # noqa: SLF001
            config=BatchRunConfig(
                manifest=manifest,
                output_dir=tmp_path,
                concurrency=200,
                prepared_sandbox_policy=policy,
            ),
            repository=repository,
            events=events,
            metrics=BatchMetrics(),
            output_dir=tmp_path,
            concurrency=200,
            create_concurrency=1,
            backpressure_gate=scheduler_module.ControllerBackpressureGate(controller_db_path=None, enabled=False),
        )
        assert manager is not None
        assert time.monotonic() - started < 0.5
        assert provider.prepared_create_calls < 100
        warmup_finished = _prepared_events(tmp_path, message="prepared_sandbox_initial_warmup_finished")
        assert warmup_finished
        details = warmup_finished[0]["details"]
        assert details["min_ready"] == 1
        assert details["prepare_buffer"] == 100
        assert details["run_concurrency"] == 200
        assert details["exit_reason"] == "ready"
    finally:
        provider.release_second.set()
        if manager is not None:
            await manager.close()


@pytest.mark.asyncio
async def test_prepared_manager_logs_backpressure_skip_without_scheduling(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-manager-pool-log",
        tasks=[BatchTaskSpec(task_id=f"case-{index}", agent_kind="desktop", query="Finish.") for index in range(1, 3)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(enabled=True, buffer=2, prepare_concurrency=1)
    gate = BlockingPrepareBackpressureGate()
    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=policy,
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=policy,
            max_attempts_by_task={task.task_id: 1 for task in manifest.tasks},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=2,
        create_concurrency=1,
        backpressure_gate=gate,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(preparation_manager_module, "_POOL_STATUS_LOG_INTERVAL_SECONDS", 0.0)
    caplog.set_level(logging.INFO, logger=preparation_manager_module.__name__)

    try:
        await manager.start()
        deadline = time.monotonic() + 1.0
        pool_records: list[logging.LogRecord] = []
        while time.monotonic() < deadline:
            pool_records = [
                record for record in caplog.records if record.message.startswith("prepared sandbox manager status:")
            ]
            if any(getattr(record, "skipped_backpressure_count", 0) == 2 for record in pool_records):
                break
            await asyncio.sleep(0.02)

        assert store.active_count() == 0
        assert not any(getattr(record, "scheduled_count", None) for record in pool_records)
        skipped_record = next(
            record for record in pool_records if getattr(record, "skipped_backpressure_count", 0) == 2
        )
        assert skipped_record.candidate_count == 2
        assert skipped_record.future_pending_count == 0
        assert skipped_record.claim_visible_future_pending_count == 0
        assert skipped_record.active_unassigned == 0
        assert skipped_record.backpressure_active is True
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_prepared_queued_futures_recheck_backpressure_after_slot_opens(tmp_path: Path) -> None:
    provider = SequencedPreparedCreateProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-queued-futures-recheck-pressure",
        tasks=[BatchTaskSpec(task_id=f"case-{index}", agent_kind="desktop", query="Finish.") for index in range(1, 4)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    policy = PreparedSandboxPolicy(enabled=True, buffer=3, prepare_concurrency=1)
    gate = BlockingPrepareBackpressureGate()
    gate.active = False
    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=policy,
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=policy,
            max_attempts_by_task={task.task_id: 1 for task in manifest.tasks},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=3,
        create_concurrency=1,
        backpressure_gate=gate,  # type: ignore[arg-type]
    )

    await manager.start()
    await asyncio.wait_for(provider.first_started.wait(), timeout=1)
    await asyncio.sleep(0.05)
    assert len(manager._futures) == 3  # noqa: SLF001 - queued future pressure regression.
    assert len(manager._claim_visible_futures) == 1  # noqa: SLF001 - only DB-reserved future is claim-visible.

    gate.active = True
    provider.release_first.set()
    await asyncio.sleep(0.1)

    assert provider.prepared_create_calls == 1
    assert len(manager._claim_visible_futures) == 1  # noqa: SLF001 - waiting-backpressure future stays invisible.

    gate.release.set()
    await manager.wait_for_initial_ready(min_ready=3, timeout_seconds=1)

    assert provider.prepared_create_calls == 3
    assert len(manager.ready_unassigned_task_attempts()) == 3
    await manager.close()


@pytest.mark.asyncio
async def test_prepared_close_bounds_late_cleanup_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preparation_manager_module, "_CLOSE_CLEANUP_WAIT_SECONDS", 0.02)
    provider = PreparedReadyAfterFallbackProvider()
    RecordingSubmitter.provider = provider
    RecordingSubmitter.submit_times = []
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-close-bounded-cleanup",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})

    started = time.monotonic()
    try:
        # An unbounded close never returns because termination is released only
        # afterwards; fail instead of hanging the suite.
        result = await asyncio.wait_for(
            BatchRunner(
                provider_registry=registry,
                submitter_factory=lambda url: RecordingSubmitter(url),
            ).run(
                BatchRunConfig(
                    manifest=manifest,
                    output_dir=tmp_path,
                    sandbox_create_concurrency=1,
                    prepared_sandbox_policy=PreparedSandboxPolicy(
                        enabled=True,
                        buffer=1,
                        prepare_concurrency=1,
                        initial_warmup_wait_seconds=0.01,
                        claim_wait_seconds=0.01,
                        claim_pending_wait_seconds=0.01,
                    ),
                )
            ),
            timeout=30,
        )
        elapsed = time.monotonic() - started
    finally:
        provider.release_termination.set()
    await asyncio.sleep(0.05)

    assert result.succeeded == 1
    assert provider.task_create_started.is_set()
    # Far below the 5 s default close wait: the patched 0.02 s bound applied.
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_prepared_close_bounds_ready_unassigned_cleanup_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preparation_manager_module, "_CLOSE_CLEANUP_WAIT_SECONDS", 0.02)
    provider = PreparedSlowTerminateWaitsProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-ready-close-bounded-cleanup",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1, claim_wait_seconds=30),
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=PreparedSandboxPolicy(enabled=True, buffer=1),
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )
    await manager.start()
    await manager.wait_for_initial_ready(min_ready=1, timeout_seconds=1)

    started = time.monotonic()
    await manager.close()
    elapsed = time.monotonic() - started
    provider.release_termination.set()
    await asyncio.sleep(0.05)

    assert elapsed < 0.25
    assert store.get(task_id="case-1", attempt=1).status == PreparedSandboxStatus.CLEANUP_FAILED


@pytest.mark.asyncio
async def test_prepared_close_retries_broken_pipe_before_cleanup_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preparation_release_module, "_PREPARED_RELEASE_RETRY_DELAYS_SECONDS", (0.01, 0.01))
    provider = PreparedBrokenPipeThenSucceedsProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-close-broken-pipe-retry",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1, claim_wait_seconds=30),
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=PreparedSandboxPolicy(enabled=True, buffer=1),
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )
    await manager.start()
    await manager.wait_for_initial_ready(min_ready=1, timeout_seconds=1)

    await manager.close()

    assert provider.terminate_attempts == {"fake-sandbox-1": 2}
    assert provider.terminated == ["fake-sandbox-1"]
    assert store.get(task_id="case-1", attempt=1).status == PreparedSandboxStatus.CANCELED
    prepared_events = [
        json.loads(line)
        for line in (tmp_path / "prepared_sandbox_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert not any(event["message"] == "prepared_sandbox_cleanup_failed" for event in prepared_events)


@pytest.mark.asyncio
async def test_prepared_close_cleanup_failure_splits_close_reason_and_terminate_error(tmp_path: Path) -> None:
    provider = TerminateFailsProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-close-cleanup-failure-details",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1, claim_wait_seconds=30),
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=PreparedSandboxPolicy(enabled=True, buffer=1),
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )
    await manager.start()
    await manager.wait_for_initial_ready(min_ready=1, timeout_seconds=1)

    await manager.close()

    row = store.get(task_id="case-1", attempt=1)
    assert row.status == PreparedSandboxStatus.CLEANUP_FAILED
    assert row.error == "prepared sandbox manager closed before assignment; terminate_failed: terminate failed"
    prepared_events = [
        json.loads(line)
        for line in (tmp_path / "prepared_sandbox_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cleanup_events = [event for event in prepared_events if event["message"] == "prepared_sandbox_cleanup_failed"]
    assert cleanup_events[-1]["details"]["close_reason"] == "prepared sandbox manager closed before assignment"
    assert cleanup_events[-1]["details"]["terminate_error"] == "terminate failed"
    assert cleanup_events[-1]["details"]["cleanup_result"] == "terminate_failed"


@pytest.mark.asyncio
async def test_prepared_close_does_not_release_claimed_assignment_waiting_on_readiness(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    readiness_probe = BlockSecondReadinessProbe()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-close-during-claim-readiness",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1, claim_wait_seconds=30),
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=PreparedSandboxPolicy(enabled=True, buffer=1),
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
            readiness_probe=readiness_probe,
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )
    await manager.start()
    await manager.wait_for_initial_ready(min_ready=1, timeout_seconds=1)

    claim_task = asyncio.create_task(manager.claim(task=manifest.tasks[0], attempt=1))
    await asyncio.wait_for(readiness_probe.second_started.wait(), timeout=1)
    assert store.get(task_id="case-1", attempt=1).status == PreparedSandboxStatus.ASSIGNED

    await manager.close()
    assert provider.terminated == []

    readiness_probe.release_second.set()
    assignment = await claim_task

    assert assignment is not None
    assert provider.terminated == []
    assert store.get(task_id="case-1", attempt=1).status == PreparedSandboxStatus.ASSIGNED


@pytest.mark.asyncio
async def test_prepared_claim_sees_canceled_prepare_future_as_fallback(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-canceled-future",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    coordinator = BatchTaskLeaseCoordinator(
        provider_registry=registry,
        binding_factory=SandboxDeviceBindingFactory(),
    )

    async def _never_returns() -> PreparedSandboxAssignment | None:
        await asyncio.Event().wait()
        return None

    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, claim_wait_seconds=1),
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=PreparedSandboxPolicy(enabled=True),
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=coordinator,
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )
    future = asyncio.create_task(_never_returns())
    key = ("case-1", 1)
    manager._futures[key] = future  # noqa: SLF001 - white-box cancellation edge.
    manager._claim_visible_futures[key] = future  # noqa: SLF001 - simulate post-reserve visibility.
    future.cancel()

    assignment = await manager.claim(task=manifest.tasks[0], attempt=1)

    assert assignment is None
    fallback_events = _prepared_fallback_events(tmp_path)
    assert len(fallback_events) == 1
    assert fallback_events[0]["details"]["reason"] == "prepare_failed"


@pytest.mark.asyncio
async def test_prepared_claim_prunes_empty_completed_future(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-empty-completed-future",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    coordinator = BatchTaskLeaseCoordinator(
        provider_registry=registry,
        binding_factory=SandboxDeviceBindingFactory(),
    )

    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, claim_wait_seconds=1),
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=PreparedSandboxPolicy(enabled=True),
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=coordinator,
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )
    future: asyncio.Future[PreparedSandboxAssignment | None] = asyncio.Future()
    key = ("case-1", 1)
    future.set_result(None)
    manager._futures[key] = future  # noqa: SLF001 - white-box failed prepare edge.
    manager._claim_visible_futures[key] = future  # noqa: SLF001 - simulate post-reserve visibility.

    assignment = await manager.claim(task=manifest.tasks[0], attempt=1)

    assert assignment is None
    fallback_events = _prepared_fallback_events(tmp_path)
    assert len(fallback_events) == 1
    assert fallback_events[0]["details"]["reason"] == "prepare_failed"
    assert key not in manager._futures  # noqa: SLF001
    assert key not in manager._claim_visible_futures  # noqa: SLF001


@pytest.mark.asyncio
async def test_prepared_manager_prunes_done_empty_futures_before_refill(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-prune-before-refill",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(tmp_path / "batch_state.sqlite")
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id=manifest.batch_id)
    manager = PreparedSandboxManager(
        config=BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1),
        ),
        store=store,
        planner=PreparedSandboxPlanner(
            manifest=manifest,
            store=store,
            policy=PreparedSandboxPolicy(enabled=True, buffer=1),
            max_attempts_by_task={"case-1": 1},
        ),
        coordinator=BatchTaskLeaseCoordinator(
            provider_registry=registry,
            binding_factory=SandboxDeviceBindingFactory(),
        ),
        events=events,
        metrics=BatchMetrics(),
        output_dir=tmp_path,
        run_concurrency=1,
        create_concurrency=1,
    )
    future: asyncio.Future[PreparedSandboxAssignment | None] = asyncio.Future()
    key = ("case-1", 1)
    future.set_result(None)
    manager._futures[key] = future  # noqa: SLF001 - white-box failed prepare edge.
    manager._claim_visible_futures[key] = future  # noqa: SLF001 - simulate post-reserve visibility.

    manager._prune_completed_futures()  # noqa: SLF001

    assert key not in manager._futures  # noqa: SLF001
    assert key not in manager._claim_visible_futures  # noqa: SLF001


@pytest.mark.asyncio
async def test_prepared_late_timeout_cleanup_failure_marks_cleanup_failed(tmp_path: Path) -> None:
    provider = PreparedSlowTerminateFailsProvider()
    RecordingSubmitter.provider = provider
    RecordingSubmitter.submit_times = []
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-timeout-cleanup-failure",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: SlowRecordingSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            sandbox_create_concurrency=1,
            prepared_sandbox_policy=PreparedSandboxPolicy(
                enabled=True,
                buffer=1,
                prepare_concurrency=1,
                initial_warmup_wait_seconds=0.01,
                claim_wait_seconds=0.01,
                claim_pending_wait_seconds=0.01,
            ),
        )
    )

    assert result.succeeded == 1
    with sqlite3.connect(tmp_path / "prepared-timeout-cleanup-failure" / "batch_state.sqlite") as connection:
        row = connection.execute(
            "SELECT status, sandbox_id, error FROM prepared_sandboxes WHERE task_id='case-1'"
        ).fetchone()
    assert row[0] == PreparedSandboxStatus.CLEANUP_FAILED.value
    assert str(row[1]).startswith("fake-sandbox-")
    assert "prepared terminate failed" in row[2]
    summary = json.loads(
        (tmp_path / "prepared-timeout-cleanup-failure" / "batch_summary.json").read_text(encoding="utf-8")
    )
    assert summary["metrics"]["prepared_sandbox_misses"] >= 1
    assert summary["metrics"]["prepared_sandbox_cleanup_failures"] >= 1


@pytest.mark.asyncio
async def test_prepared_startup_cleanup_failure_records_cleanup_failed(tmp_path: Path) -> None:
    provider = TerminateFailsProvider()
    RecordingSubmitter.provider = provider
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-startup-cleanup-failure",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "desktop"})
    manifest.defaults.startup_commands = [SandboxCommandSpec(label="extract-workspace", command="fail setup")]

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: RecordingSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(
                enabled=True,
                buffer=1,
                prepare_concurrency=1,
                claim_wait_seconds=1,
            ),
        )
    )

    assert result.failed == 1
    with sqlite3.connect(tmp_path / "prepared-startup-cleanup-failure" / "batch_state.sqlite") as connection:
        row = connection.execute("SELECT status, sandbox_id FROM prepared_sandboxes WHERE task_id='case-1'").fetchone()
    assert row[0] == PreparedSandboxStatus.CLEANUP_FAILED.value
    assert row[1] == "fake-sandbox-1"


@pytest.mark.asyncio
async def test_prepared_rejects_connect_existing_allocation(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="prepared-connect-existing-conflict",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Finish.")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(
        update={
            "provider": "fake",
            "template": "desktop",
            "allocation_strategy": SandboxAllocationStrategy.CONNECT_EXISTING,
            "sandbox_id": "existing",
        }
    )

    with pytest.raises(ValueError, match="connect_existing"):
        await BatchRunner(
            provider_registry=registry,
            submitter_factory=lambda url: RecordingSubmitter(url),
        ).run(
            BatchRunConfig(
                manifest=manifest,
                output_dir=tmp_path,
                prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1),
            )
        )
    assert not (tmp_path / "prepared-connect-existing-conflict").exists()
