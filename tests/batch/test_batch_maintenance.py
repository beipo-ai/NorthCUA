"""Batch maintenance control-plane tests."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest

from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.controller_db_status import LocalControllerDBStatusReader
from northcua.batch.maintenance.contracts import (
    MaintenanceActionKind,
    MaintenanceApplyResult,
    MaintenancePlan,
    MaintenancePlanItem,
)
from northcua.batch.maintenance.controller_ops import ControllerMaintenanceClient
from northcua.batch.maintenance.executor import MaintenanceExecutor
from northcua.batch.maintenance.operations import BatchMaintenanceOperations
from northcua.batch.maintenance.planner import MaintenancePlanner
from northcua.batch.maintenance.sqlite_ops import SQLiteMaintenanceError, SQLiteMaintenanceStore
from northcua.batch.operations import (
    _controller_residue_cleanup_blocker,
    _ensure_prepared_indexes,
    _pre_resume_blocking_actions,
    _prepared_live_budget_summary,
    _run_pre_resume_cleanup,
)
from northcua.batch.planner import BatchTaskPlanner
from northcua.batch.preparation import PreparedSandboxDepth, PreparedSandboxPolicy, PreparedSandboxStatus
from northcua.batch.preparation.sqlite_store import SQLitePreparedSandboxStore
from northcua.batch.state import LeasePolicy, SQLiteBatchStateRepository
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua.reliability import FailureAction
from northcua.sandboxes.capacity_contracts import SandboxInventoryItem, SandboxInventoryState
from northcua.telemetry.contracts import FailureCategory
from northcua_protocol.payloads import TaskFinished, TaskRequest, TaskStatusSnapshot
from northcua_protocol.state import TaskState

_T = TypeVar("_T")


class FakeControllerMaintenanceClient:
    """Controller maintenance fake for deterministic tests."""

    state_from_snapshot = staticmethod(ControllerMaintenanceClient.state_from_snapshot)

    def __init__(self, snapshots: list[TaskStatusSnapshot] | None = None, *, cancel_error: str = "") -> None:
        self.snapshots = {snapshot.task_id: snapshot for snapshot in snapshots or []}
        self.cancel_error = cancel_error
        self.canceled: list[str] = []
        self.list_limits: list[int] = []
        self.status_task_ids: list[str] = []

    async def statuses(
        self,
        *,
        controller_url: str,
        token: str,
        task_ids: list[str],
        controller_db_path: Path | None = None,
    ) -> list[TaskStatusSnapshot]:
        """Return stored snapshots for requested task ids."""

        _ = controller_db_path
        assert controller_url or controller_db_path is not None
        if controller_url:
            assert token == "tok"
        self.status_task_ids.extend(task_ids)
        return [self.snapshots[task_id] for task_id in task_ids if task_id in self.snapshots]

    async def list_batch(
        self,
        *,
        controller_url: str,
        token: str,
        batch_id: str,
        limit: int = 0,
        controller_db_path: Path | None = None,
    ) -> list[TaskStatusSnapshot]:
        """Return all stored snapshots as a fake batch listing."""

        _ = controller_db_path
        assert controller_url or controller_db_path is not None
        if controller_url:
            assert token == "tok"
        assert batch_id == "batch-1"
        self.list_limits.append(limit)
        return list(self.snapshots.values())

    async def cancel(
        self,
        *,
        controller_url: str,
        token: str,
        state: BatchTaskState,
        reason: str,
    ) -> None:
        """Record best-effort cancellation."""

        assert controller_url
        assert token == "tok"
        assert reason
        if self.cancel_error:
            raise PermissionError(self.cancel_error)
        self.canceled.append(state.controller_task_id)


class _CountingMaintenanceWatcher:
    def __init__(self, snapshot: TaskStatusSnapshot) -> None:
        self.snapshot = snapshot
        self.status_calls = 0
        self.list_calls = 0

    async def statuses(self, **kwargs: object) -> list[TaskStatusSnapshot]:
        self.status_calls += 1
        assert kwargs["task_ids"] == [self.snapshot.task_id]
        return [self.snapshot]

    async def list_batch(self, **kwargs: object) -> list[TaskStatusSnapshot]:
        self.list_calls += 1
        assert kwargs["batch_id"] == "batch-1"
        return [self.snapshot]


class FakeSandboxMaintenanceClient:
    """Sandbox maintenance fake with explicit inventory."""

    def __init__(
        self,
        inventory: list[SandboxInventoryItem],
        *,
        inventory_error: str = "",
        terminate_error: str = "",
    ) -> None:
        self._inventory = inventory
        self.inventory_error = inventory_error
        self.terminate_error = terminate_error
        self.terminated: list[str] = []

    async def inventory(
        self,
        *,
        provider_id: str,
        batch_id: str,
        case_id: str = "",
        include_warm_pool: bool = False,
    ) -> list[SandboxInventoryItem]:
        """Return fake provider inventory rows."""

        assert provider_id == "fake"
        if self.inventory_error:
            raise RuntimeError(self.inventory_error)
        return [
            item
            for item in self._inventory
            if item.sandbox_id not in self.terminated
            and item.metadata.get("batch_id") == batch_id
            and (not case_id or item.metadata.get("task_id") == case_id)
            and (
                include_warm_pool
                or not item.metadata.get("warm_pool")
                and "warm_pool_slot" not in item.metadata
                and not str(item.metadata.get("task_id", "")).startswith("warm-pool-")
            )
        ]

    async def terminate(
        self,
        *,
        item: SandboxInventoryItem,
        batch_id: str,
        include_connect_existing: bool = False,
        include_warm_pool: bool = False,
    ) -> None:
        """Record fake sandbox termination."""

        assert batch_id == "batch-1"
        _ = include_connect_existing
        _ = include_warm_pool
        if self.terminate_error:
            raise RuntimeError(self.terminate_error)
        self.terminated.append(item.sandbox_id)


def _manifest() -> BatchManifest:
    return BatchManifest(
        batch_id="batch-1",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )


def _init_batch(tmp_path: Path) -> SQLiteBatchStateRepository:
    batch_dir = tmp_path / "batch-1"
    batch_dir.mkdir()
    manifest = _manifest()
    (batch_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    return repository


def _set_active_lease(
    repository: SQLiteBatchStateRepository,
    task_id: str,
    *,
    owner: str = "dead-runner",
    expires_in: float = 3600.0,
) -> None:
    with sqlite3.connect(repository.path) as connection:
        connection.execute(
            """
            UPDATE batch_tasks
            SET lease_owner=?, lease_token='lease-token', lease_expires_at=?, generation=generation+1
            WHERE task_id=?
            """,
            (owner, time.time() + expires_in, task_id),
        )


def test_sqlite_maintenance_tolerates_legacy_unknown_recovery_action(tmp_path: Path) -> None:
    """Pre-resume cleanup must keep reading rows with legacy recovery labels."""

    repository = _init_batch(tmp_path)
    legacy_state = BatchTaskState(
        task_id="case-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        retry_budget_remaining=1,
    ).model_dump(mode="json")
    legacy_state["recovery_action"] = "terminal_after_pre_resume_clean_budget_exhausted"
    with sqlite3.connect(repository.path) as connection:
        connection.execute(
            """
            UPDATE batch_tasks
            SET state_json=?, status=?, attempts=?, recovery_action=?,
                retry_budget_remaining=?, updated_at=?
            WHERE task_id=?
            """,
            (
                json.dumps(legacy_state),
                BatchTaskStatus.FAILED.value,
                1,
                "terminal_after_pre_resume_clean_budget_exhausted",
                1,
                time.time(),
                "case-1",
            ),
        )

    store = SQLiteMaintenanceStore(repository.path)
    states = store.pre_resume_states()

    assert len(states) == 1
    assert states[0].recovery_action is None


def test_controller_maintenance_statuses_treats_readable_empty_db_as_authoritative(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.FINISHED,
    )
    watcher = _CountingMaintenanceWatcher(snapshot)
    client = ControllerMaintenanceClient(watcher=watcher)  # type: ignore[arg-type]

    statuses = _run(
        client.statuses(
            controller_url="ws://controller",
            token="tok",
            task_ids=["controller-1"],
            controller_db_path=controller_db,
        )
    )

    assert statuses == []
    assert watcher.status_calls == 0


def test_controller_db_status_lookup_preserves_requested_order(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    for task_id in ("controller-a", "controller-b"):
        controller_store.create_task(
            TaskRequest(
                task_id=task_id,
                trace_id=f"trace-{task_id}",
                agent_kind="desktop",
                request="Do it",
                metadata={"batch_id": "batch-1", "batch_task_id": task_id},
            ),
            session_id=f"session-{task_id}",
            submitter_id="submitter",
        )
    controller_store.close()
    client = ControllerMaintenanceClient()

    statuses = _run(
        client.statuses(
            controller_url="",
            token="",
            task_ids=["controller-b", "missing", "controller-a"],
            controller_db_path=controller_db,
        )
    )

    assert [status.task_id for status in statuses] == ["controller-b", "controller-a"]


def test_controller_db_maintenance_statuses_use_lightweight_projection(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 2},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        "controller-1",
        TaskState.INTERRUPTED,
        result_json=TaskFinished(
            task_id="controller-1",
            trace_id="trace-1",
            session_id="session-result",
            state=TaskState.INTERRUPTED,
            device_id="device-result",
            lease_id="lease-result",
            result={"large_payload": "x" * 20_000},
            error="queued task lost its submitter mapping",
            terminal_decision={
                "action": "fail",
                "reason": "controller submit",
                "metadata": {"failure_category": FailureCategory.CONTROLLER_SUBMIT.value},
            },
        ).model_dump_json(),
        finished=True,
    )
    controller_store.close()

    statuses = LocalControllerDBStatusReader(controller_db).maintenance_statuses(["controller-1"])

    assert statuses is not None
    assert len(statuses) == 1
    snapshot = statuses[0]
    assert snapshot.task_id == "controller-1"
    assert snapshot.trace_id == "trace-1"
    assert snapshot.session_id == "session-result"
    assert snapshot.device_id == "device-result"
    assert snapshot.lease_id == "lease-result"
    assert snapshot.state == TaskState.INTERRUPTED
    assert snapshot.error == "queued task lost its submitter mapping"
    assert snapshot.metadata == {"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 2}
    assert snapshot.terminal_decision["metadata"] == {"failure_category": FailureCategory.CONTROLLER_SUBMIT.value}
    assert snapshot.result is None
    assert len(snapshot.model_dump_json()) < 2_000


def test_controller_db_maintenance_statuses_preserve_finished_envelope(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-row",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        "controller-1",
        TaskState.FINISHED,
        result_json=TaskFinished(
            task_id="controller-1",
            trace_id="trace-1",
            session_id="session-result",
            state=TaskState.FINISHED,
            device_id="device-result",
            lease_id="lease-result",
            result={
                "evaluation_report": {
                    "overall_verdict": "success",
                    "controllability": "uncontrollable",
                    "large_field": "x" * 20_000,
                },
                "large_payload": "y" * 20_000,
            },
            terminal_decision={"action": "finish", "success": True, "metadata": {"note": "kept"}},
        ).model_dump_json(),
        finished=True,
    )
    controller_store.close()

    statuses = LocalControllerDBStatusReader(controller_db).maintenance_statuses(["controller-1"])

    assert statuses is not None
    snapshot = statuses[0]
    assert snapshot.session_id == "session-result"
    assert snapshot.device_id == "device-result"
    assert snapshot.lease_id == "lease-result"
    assert snapshot.terminal_decision == {"action": "finish", "success": True, "metadata": {"note": "kept"}}
    assert snapshot.result == {"evaluation_report": {"overall_verdict": "success", "controllability": "uncontrollable"}}
    assert "large_payload" not in snapshot.model_dump_json()
    assert "large_field" not in snapshot.model_dump_json()


def test_controller_db_maintenance_statuses_match_full_nested_evaluator_projection(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        "controller-1",
        TaskState.FAILED,
        result_json=TaskFinished(
            task_id="controller-1",
            trace_id="trace-1",
            session_id="session-1",
            state=TaskState.FAILED,
            result={
                "result": {
                    "evaluation_report": {
                        "overall_verdict": "success",
                        "outcome_label": "failure",
                        "controllability": "uncontrollable",
                    }
                }
            },
            error="missing physical device",
        ).model_dump_json(),
        finished=True,
    )
    controller_store.close()
    existing = BatchTaskState(task_id="case-1", status=BatchTaskStatus.RUNNING)

    full = LocalControllerDBStatusReader(controller_db).statuses(["controller-1"])
    lightweight = LocalControllerDBStatusReader(controller_db).maintenance_statuses(["controller-1"])

    assert full is not None
    assert lightweight is not None
    full_state = ControllerMaintenanceClient.state_from_snapshot(existing, full[0])
    lightweight_state = ControllerMaintenanceClient.state_from_snapshot(existing, lightweight[0])
    assert lightweight_state.status == BatchTaskStatus.SUCCEEDED
    assert lightweight_state.last_failure_category == FailureCategory.ACCEPTED_INFEASIBLE
    assert lightweight_state.status == full_state.status
    assert lightweight_state.last_failure_category == full_state.last_failure_category
    assert lightweight_state.error == full_state.error


def test_controller_db_maintenance_statuses_match_full_raw_result_projection(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        "controller-1",
        TaskState.FAILED,
        result_json=json.dumps(
            {
                "evaluation_report": {
                    "overall_verdict": "success",
                    "outcome_label": "failure",
                    "controllability": "uncontrollable",
                },
                "error": "missing physical device",
                "large_payload": "x" * 20_000,
            }
        ),
        finished=True,
    )
    controller_store.close()
    existing = BatchTaskState(task_id="case-1", status=BatchTaskStatus.RUNNING)

    full = LocalControllerDBStatusReader(controller_db).statuses(["controller-1"])
    lightweight = LocalControllerDBStatusReader(controller_db).maintenance_statuses(["controller-1"])

    assert full is not None
    assert lightweight is not None
    full_state = ControllerMaintenanceClient.state_from_snapshot(existing, full[0])
    lightweight_state = ControllerMaintenanceClient.state_from_snapshot(existing, lightweight[0])
    assert lightweight_state.status == BatchTaskStatus.SUCCEEDED
    assert lightweight_state.last_failure_category == FailureCategory.ACCEPTED_INFEASIBLE
    assert lightweight_state.status == full_state.status
    assert lightweight_state.last_failure_category == full_state.last_failure_category
    assert lightweight_state.error == full_state.error
    assert "large_payload" not in lightweight[0].model_dump_json()


def test_controller_db_maintenance_statuses_do_not_treat_raw_dict_as_envelope(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-row",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        "controller-1",
        TaskState.FAILED,
        result_json=json.dumps(
            {
                "error": "missing physical device",
                "terminal_decision": {"retryable": False},
                "session_id": "session-raw",
                "device_id": "device-raw",
                "lease_id": "lease-raw",
            }
        ),
        finished=True,
    )
    controller_store.close()
    existing = BatchTaskState(task_id="case-1", status=BatchTaskStatus.RUNNING)

    full = LocalControllerDBStatusReader(controller_db).statuses(["controller-1"])
    lightweight = LocalControllerDBStatusReader(controller_db).maintenance_statuses(["controller-1"])

    assert full is not None
    assert lightweight is not None
    assert full[0].terminal_decision == {}
    assert lightweight[0].terminal_decision == {}
    assert lightweight[0].session_id == full[0].session_id == "session-row"
    assert lightweight[0].device_id == full[0].device_id == ""
    full_state = ControllerMaintenanceClient.state_from_snapshot(existing, full[0])
    lightweight_state = ControllerMaintenanceClient.state_from_snapshot(existing, lightweight[0])
    assert lightweight_state.status == full_state.status
    assert lightweight_state.recovery_action == full_state.recovery_action


def test_controller_db_maintenance_statuses_fail_closed_on_bad_json(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.close()
    with sqlite3.connect(controller_db) as conn:
        conn.execute("UPDATE tasks SET result_json='{bad-json' WHERE task_id='controller-1'")

    statuses = LocalControllerDBStatusReader(controller_db).maintenance_statuses(["controller-1"])

    assert statuses is None


def test_controller_maintenance_statuses_falls_back_when_db_unavailable(tmp_path: Path) -> None:
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.FINISHED,
    )
    watcher = _CountingMaintenanceWatcher(snapshot)
    client = ControllerMaintenanceClient(watcher=watcher)  # type: ignore[arg-type]

    statuses = _run(
        client.statuses(
            controller_url="ws://controller",
            token="tok",
            task_ids=["controller-1"],
            controller_db_path=tmp_path / "missing-controller.sqlite",
        )
    )

    assert statuses == [snapshot]
    assert watcher.status_calls == 1


def test_controller_maintenance_list_batch_treats_readable_empty_db_as_authoritative(tmp_path: Path) -> None:
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        agent_kind="desktop",
        state=TaskState.FINISHED,
    )
    watcher = _CountingMaintenanceWatcher(snapshot)
    client = ControllerMaintenanceClient(watcher=watcher)  # type: ignore[arg-type]

    statuses = _run(
        client.list_batch(
            controller_url="ws://controller",
            token="tok",
            batch_id="batch-1",
            controller_db_path=controller_db,
        )
    )

    assert statuses == []
    assert watcher.list_calls == 0


def test_repair_stale_resets_pre_controller_row(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.STARTING_SANDBOX, attempts=1))
    operations = BatchMaintenanceOperations(
        controller=FakeControllerMaintenanceClient(),
        sandboxes=FakeSandboxMaintenanceClient([]),
    )

    result = _run(operations.repair_stale(batch_id="batch-1", output_dir=tmp_path, older_than_seconds=0, apply=True))

    state = repository.get("case-1")
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.last_failure_category is not None
    assert result.applied_count == 1
    assert (tmp_path / "batch-1" / "batch_maintenance_events.jsonl").exists()


def test_repair_stale_ignores_submitted_rows_with_controller_task_id(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    original = BatchTaskState(
        task_id="case-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
        controller_task_id="controller-1",
    )
    repository.upsert_state(original)
    operations = BatchMaintenanceOperations(
        controller=FakeControllerMaintenanceClient(),
        sandboxes=FakeSandboxMaintenanceClient([]),
    )

    result = _run(operations.repair_stale(batch_id="batch-1", output_dir=tmp_path, older_than_seconds=0, apply=True))

    assert repository.get("case-1") == original
    assert result.applied_count == 0
    assert result.planned_count == 0


def test_repair_stale_ignores_running_rows_without_controller_task_id(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    original = BatchTaskState(task_id="case-1", status=BatchTaskStatus.RUNNING, attempts=1)
    repository.upsert_state(original)
    operations = BatchMaintenanceOperations(
        controller=FakeControllerMaintenanceClient(),
        sandboxes=FakeSandboxMaintenanceClient([]),
    )

    result = _run(operations.repair_stale(batch_id="batch-1", output_dir=tmp_path, older_than_seconds=0, apply=True))

    assert repository.get("case-1") == original
    assert result.planned_count == 0


def test_repair_stale_ignores_active_lease(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )
    assert claim is not None
    original = BatchTaskState(task_id="case-1", status=BatchTaskStatus.STARTING_SANDBOX, attempts=1)
    assert repository.upsert_state(original, lease_token=claim.lease_token)
    operations = BatchMaintenanceOperations(
        controller=FakeControllerMaintenanceClient(),
        sandboxes=FakeSandboxMaintenanceClient([]),
    )

    result = _run(operations.repair_stale(batch_id="batch-1", output_dir=tmp_path, older_than_seconds=0, apply=True))

    assert repository.get("case-1") == original
    assert result.planned_count == 0


def test_repair_stale_rejects_row_changed_after_plan(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.STARTING_SANDBOX, attempts=1))
    store = SQLiteMaintenanceStore(tmp_path / "batch-1" / "batch_state.sqlite")
    planner = MaintenancePlanner()
    plan = planner.repair_stale(
        batch_id="batch-1",
        batch_dir=tmp_path / "batch-1",
        store=store,
        older_than_seconds=0,
    )
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.STARTING_SANDBOX, attempts=2))

    result = _run(
        MaintenanceExecutor(
            store=store,
            controller=FakeControllerMaintenanceClient(),
            sandboxes=FakeSandboxMaintenanceClient([]),
        ).execute(plan=plan, apply=True)
    )

    state = repository.get("case-1")
    assert state is not None
    assert state.status == BatchTaskStatus.STARTING_SANDBOX
    assert state.attempts == 2
    assert result.applied_count == 0
    assert result.errors


def test_pre_resume_stale_repair_reconciles_terminal_submitted_from_controller_db(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
            trace_id="trace-old",
            next_attempt_after=1234.0,
            reconcile_unknown_count=2,
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
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
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.SUCCEEDED
    assert state.controller_task_id == "controller-1"
    assert state.trace_id == "trace-1"
    assert state.next_attempt_after == 0.0
    assert state.reconcile_unknown_count == 0


def test_pre_resume_repair_falls_back_when_lightweight_projection_fails(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.close()
    with sqlite3.connect(controller_db) as conn:
        conn.execute("UPDATE tasks SET result_json='{bad-json' WHERE task_id='controller-1'")
    controller = FakeControllerMaintenanceClient(
        [
            TaskStatusSnapshot(
                task_id="controller-1",
                trace_id="trace-fallback",
                agent_kind="desktop",
                state=TaskState.FINISHED,
                metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
            )
        ]
    )
    operations = BatchMaintenanceOperations(controller=controller, sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert "controller-1" in controller.status_task_ids
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.SUCCEEDED
    assert state.trace_id == "trace-fallback"


def test_pre_resume_stale_repair_does_not_scan_controller_batch_listing(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    controller = FakeControllerMaintenanceClient(
        [
            TaskStatusSnapshot(
                task_id="controller-1",
                trace_id="trace-1",
                agent_kind="desktop",
                state=TaskState.RUNNING,
            )
        ]
    )
    operations = BatchMaintenanceOperations(controller=controller, sandboxes=FakeSandboxMaintenanceClient([]))

    _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=tmp_path / "controller.sqlite",
            older_than_seconds=300,
            apply=False,
        )
    )

    assert controller.list_limits == []


def test_pre_resume_repair_reconciles_recent_terminal_snapshot_without_age_wait(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
            trace_id="trace-old",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        "controller-1",
        TaskState.INTERRUPTED,
        result_json=TaskFinished(
            task_id="controller-1",
            trace_id="trace-1",
            session_id="session-1",
            state=TaskState.INTERRUPTED,
            device_id="device-1",
            error="0 bytes read on a total of 4 expected bytes",
        ).model_dump_json(),
        finished=True,
    )
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=3600,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert state.last_reconciled_status == TaskState.INTERRUPTED.value


def test_pre_resume_repair_reprojects_misclassified_terminal_snapshot(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.CANCELED,
            attempts=1,
            controller_task_id="controller-1",
            trace_id="trace-1",
            terminal_state=TaskState.INTERRUPTED,
            last_reconciled_status=TaskState.INTERRUPTED.value,
            error="0 bytes read on a total of 4 expected bytes",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        "controller-1",
        TaskState.INTERRUPTED,
        result_json=TaskFinished(
            task_id="controller-1",
            trace_id="trace-1",
            session_id="session-1",
            state=TaskState.INTERRUPTED,
            device_id="device-1",
            error="0 bytes read on a total of 4 expected bytes",
        ).model_dump_json(),
        finished=True,
    )
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=3600,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.last_failure_category == FailureCategory.CONTROLLER_SUBMIT


def test_pre_resume_repair_reconciles_next_attempt_deterministic_residue(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            retry_budget_remaining=1,
            last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
        )
    )
    controller_task_id, trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=2,
    )
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id=controller_task_id,
            trace_id=trace_id,
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 2},
        ),
        session_id="session-2",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        controller_task_id,
        TaskState.FINISHED,
        result_json=TaskFinished(
            task_id=controller_task_id,
            trace_id=trace_id,
            session_id="session-2",
            state=TaskState.FINISHED,
            device_id="device-2",
        ).model_dump_json(),
        finished=True,
    )
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=3600,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.SUCCEEDED
    assert state.attempts == 2
    assert state.controller_task_id == controller_task_id
    assert state.trace_id == trace_id


def test_pre_resume_repair_skips_fresh_pending_controller_lookup(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch-1"
    batch_dir.mkdir()
    manifest = BatchManifest(
        batch_id="batch-1",
        tasks=[
            BatchTaskSpec(task_id="fresh-pending", agent_kind="desktop", query="Do it"),
            BatchTaskSpec(task_id="retry-pending", agent_kind="desktop", query="Do it"),
            BatchTaskSpec(task_id="failed-retry", agent_kind="desktop", query="Do it"),
        ],
    )
    (batch_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    repository.upsert_state(BatchTaskState(task_id="fresh-pending", status=BatchTaskStatus.PENDING, attempts=0))
    repository.upsert_state(BatchTaskState(task_id="retry-pending", status=BatchTaskStatus.PENDING, attempts=2))
    repository.upsert_state(
        BatchTaskState(
            task_id="failed-retry",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            retry_budget_remaining=1,
        )
    )
    controller = FakeControllerMaintenanceClient([])
    operations = BatchMaintenanceOperations(controller=controller, sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=tmp_path / "controller.sqlite",
            older_than_seconds=3600,
            apply=False,
        )
    )

    fresh_id, _ = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="fresh-pending",
        attempt=1,
    )
    retry_pending_id, _ = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="retry-pending",
        attempt=2,
    )
    failed_current_id, _ = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="failed-retry",
        attempt=1,
    )
    failed_next_id, _ = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="failed-retry",
        attempt=2,
    )
    assert result.planned_count == 0
    assert fresh_id not in controller.status_task_ids
    assert retry_pending_id in controller.status_task_ids
    assert failed_current_id in controller.status_task_ids
    assert failed_next_id in controller.status_task_ids


def test_pre_resume_repair_does_not_revive_canceled_row_from_next_attempt_residue(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.CANCELED,
            attempts=1,
            retry_budget_remaining=1,
        )
    )
    controller_task_id, trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=2,
    )
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id=controller_task_id,
            trace_id=trace_id,
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 2},
        ),
        session_id="session-2",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        controller_task_id,
        TaskState.FINISHED,
        result_json=TaskFinished(
            task_id=controller_task_id,
            trace_id=trace_id,
            session_id="session-2",
            state=TaskState.FINISHED,
            device_id="device-2",
        ).model_dump_json(),
        finished=True,
    )
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=3600,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 0
    assert state is not None
    assert state.status == BatchTaskStatus.CANCELED
    assert state.attempts == 1
    assert not state.controller_task_id


def test_pre_resume_repair_uses_projection_columns_when_state_json_keeps_old_controller_id(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    controller_id = "controller-stale"
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.CANCELED,
            attempts=1,
            controller_task_id=controller_id,
            trace_id="trace-stale",
            terminal_state=TaskState.INTERRUPTED,
            error="controller residue cleanup interrupted stale task",
            last_reconciled_status=TaskState.INTERRUPTED.value,
            retry_budget_remaining=1,
        )
    )
    with sqlite3.connect(tmp_path / "batch-1" / "batch_state.sqlite") as conn:
        conn.execute("""
            UPDATE batch_tasks
            SET controller_task_id='', trace_id='', session_id=''
            WHERE task_id='case-1'
            """)
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id=controller_id,
            trace_id="trace-stale",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        controller_id,
        TaskState.INTERRUPTED,
        result_json=TaskFinished(
            task_id=controller_id,
            trace_id="trace-stale",
            session_id="session-1",
            state=TaskState.INTERRUPTED,
            error="controller residue cleanup interrupted stale task",
        ).model_dump_json(),
        finished=True,
    )
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=300,
            apply=True,
        )
    )
    second = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=300,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.errors == []
    assert result.applied_count == 1
    assert second.errors == []
    assert second.applied_count == 0
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.controller_task_id == controller_id
    assert state.last_failure_category == FailureCategory.CONTROLLER_SUBMIT


def test_maintenance_store_projects_task_columns_over_stale_state_json(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=2,
            controller_task_id="controller-old",
            trace_id="trace-old",
            terminal_decision={"retryable": True},
            post_run_results=[],
            reconcile_unknown_count=3,
        )
    )
    with sqlite3.connect(tmp_path / "batch-1" / "batch_state.sqlite") as conn:
        conn.execute("""
            UPDATE batch_tasks
            SET status='failed',
                attempts=3,
                controller_task_id='',
                trace_id='',
                session_id='',
                error='projection wins',
                last_failure_category='controller_submit',
                last_reconciled_status='interrupted',
                retry_budget_remaining=1,
                next_attempt_after=123.5
            WHERE task_id='case-1'
            """)
    store = SQLiteMaintenanceStore(tmp_path / "batch-1" / "batch_state.sqlite")

    state = store.get_state("case-1")
    details = store.row_details("case-1")
    all_states = store.all_states()

    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.attempts == 3
    assert state.controller_task_id == ""
    assert state.trace_id == ""
    assert state.error == "projection wins"
    assert state.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert state.last_reconciled_status == TaskState.INTERRUPTED.value
    assert state.retry_budget_remaining == 1
    assert state.next_attempt_after == 123.5
    assert state.terminal_decision == {"retryable": True}
    assert state.reconcile_unknown_count == 3
    assert details["state"] == state.model_dump(mode="json")
    assert all_states == [state]


def test_pre_resume_repair_checks_next_attempt_after_current_snapshot_already_reflected(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    current_id = "controller-current"
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id=current_id,
            terminal_state=TaskState.INTERRUPTED,
            last_reconciled_status=TaskState.INTERRUPTED.value,
            last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
            retry_budget_remaining=1,
        )
    )
    next_id, next_trace = BatchTaskPlanner._deterministic_ids(batch_id="batch-1", task_id="case-1", attempt=2)
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id=current_id,
            trace_id="trace-current",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        current_id,
        TaskState.INTERRUPTED,
        result_json=TaskFinished(
            task_id=current_id,
            trace_id="trace-current",
            session_id="session-1",
            state=TaskState.INTERRUPTED,
            error="0 bytes read on a total of 4 expected bytes",
        ).model_dump_json(),
        finished=True,
    )
    controller_store.create_task(
        TaskRequest(
            task_id=next_id,
            trace_id=next_trace,
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 2},
        ),
        session_id="session-2",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        next_id,
        TaskState.FINISHED,
        result_json=TaskFinished(
            task_id=next_id,
            trace_id=next_trace,
            session_id="session-2",
            state=TaskState.FINISHED,
            device_id="device-2",
        ).model_dump_json(),
        finished=True,
    )
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=3600,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.SUCCEEDED
    assert state.attempts == 2
    assert state.controller_task_id == next_id


def test_pre_resume_stale_repair_resets_submitted_without_controller_task_id(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUBMITTED, attempts=1))
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.controller_task_id == ""
    assert state.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert state.last_reconciled_status == "local_stale_reset"


def test_pre_resume_stale_repair_default_preserves_active_lease(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUBMITTED, attempts=1))
    _set_active_lease(repository, "case-1")
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 0
    assert result.plan.metadata["active_lease_states_planned"] == 0
    assert state is not None
    assert state.status == BatchTaskStatus.SUBMITTED
    with sqlite3.connect(repository.path) as connection:
        row = connection.execute(
            "SELECT lease_owner, lease_token, lease_expires_at FROM batch_tasks WHERE task_id='case-1'"
        ).fetchone()
    assert row[0] == "dead-runner"
    assert row[1] == "lease-token"
    assert float(row[2]) > time.time()


def test_pre_resume_stale_repair_can_release_dead_active_lease_without_controller_id(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUBMITTED, attempts=1))
    _set_active_lease(repository, "case-1")
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
            include_active_leases=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert result.plan.metadata["active_lease_states_planned"] == 1
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert state.last_reconciled_status == "dead_runner_lease_pre_resume"
    with sqlite3.connect(repository.path) as connection:
        row = connection.execute(
            "SELECT lease_owner, lease_token, lease_expires_at FROM batch_tasks WHERE task_id='case-1'"
        ).fetchone()
    assert row == ("", "", 0.0)


def test_pre_resume_stale_repair_can_reconcile_terminal_active_lease(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    _set_active_lease(repository, "case-1")
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
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
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
            include_active_leases=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert result.plan.metadata["active_lease_states_planned"] == 1
    assert state is not None
    assert state.status == BatchTaskStatus.SUCCEEDED
    assert state.controller_task_id == "controller-1"
    with sqlite3.connect(repository.path) as connection:
        row = connection.execute(
            "SELECT lease_owner, lease_token, lease_expires_at FROM batch_tasks WHERE task_id='case-1'"
        ).fetchone()
    assert row == ("", "", 0.0)


def test_pre_resume_stale_repair_releases_active_lease_with_missing_explicit_snapshot(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-missing",
        )
    )
    _set_active_lease(repository, "case-1")
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
            include_active_leases=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert result.plan.items[0].details["allow_active_lease_reset"] is True
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.controller_task_id == "controller-missing"
    assert state.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert state.last_reconciled_status == "dead_runner_lease_pre_resume"
    assert state.next_attempt_after > 0
    with sqlite3.connect(repository.path) as connection:
        row = connection.execute(
            "SELECT lease_owner, lease_token, lease_expires_at FROM batch_tasks WHERE task_id='case-1'"
        ).fetchone()
    assert row == ("", "", 0.0)


def test_pre_resume_stale_repair_releases_active_lease_with_nonterminal_snapshot(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    _set_active_lease(repository, "case-1")
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
            include_active_leases=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.RUNNING
    assert state.controller_task_id == "controller-1"
    assert state.last_reconciled_status
    with sqlite3.connect(repository.path) as connection:
        row = connection.execute(
            "SELECT lease_owner, lease_token, lease_expires_at FROM batch_tasks WHERE task_id='case-1'"
        ).fetchone()
    assert row == ("", "", 0.0)


def test_pre_resume_active_lease_reset_cas_rechecks_lease_identity(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUBMITTED, attempts=1))
    _set_active_lease(repository, "case-1", owner="old-runner")
    store = SQLiteMaintenanceStore(repository.path)
    expected_row = store.row_details("case-1")
    with sqlite3.connect(repository.path) as connection:
        connection.execute(
            """
            UPDATE batch_tasks
            SET lease_owner='still-live-runner', lease_token='new-token', lease_expires_at=?, generation=generation+1
            WHERE task_id='case-1'
            """,
            (time.time() + 3600.0,),
        )

    with pytest.raises(SQLiteMaintenanceError, match="row changed"):
        store.reset_stale_in_flight(
            case_id="case-1",
            reason="release dead active lease",
            allowed_statuses={BatchTaskStatus.SUBMITTED},
            expected_row=expected_row,
            allow_active_lease=True,
            last_reconciled_status="dead_runner_lease_pre_resume",
        )

    state = repository.get("case-1")
    assert state is not None
    assert state.status == BatchTaskStatus.SUBMITTED


def test_pre_resume_active_lease_reset_cas_rechecks_heartbeat_extension(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUBMITTED, attempts=1))
    _set_active_lease(repository, "case-1", owner="old-runner")
    store = SQLiteMaintenanceStore(repository.path)
    expected_row = store.row_details("case-1")
    with sqlite3.connect(repository.path) as connection:
        connection.execute(
            """
            UPDATE batch_tasks
            SET lease_expires_at=?, updated_at=?
            WHERE task_id='case-1'
            """,
            (time.time() + 7200.0, time.time()),
        )

    with pytest.raises(SQLiteMaintenanceError, match="row changed"):
        store.reset_stale_in_flight(
            case_id="case-1",
            reason="release dead active lease",
            allowed_statuses={BatchTaskStatus.SUBMITTED},
            expected_row=expected_row,
            allow_active_lease=True,
            last_reconciled_status="dead_runner_lease_pre_resume",
        )

    state = repository.get("case-1")
    assert state is not None
    assert state.status == BatchTaskStatus.SUBMITTED
    with sqlite3.connect(repository.path) as connection:
        row = connection.execute(
            "SELECT lease_owner, lease_token, lease_expires_at FROM batch_tasks WHERE task_id='case-1'"
        ).fetchone()
    assert row[0] == "old-runner"
    assert row[1] == "lease-token"
    assert float(row[2]) > time.time()


def test_pre_resume_stale_repair_preserves_nonterminal_controller_snapshot(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.WAITING_DEVICE, attempts=1))
    controller_task_id, trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=1,
    )
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id=controller_task_id,
            trace_id=trace_id,
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.RUNNING
    assert state.controller_task_id == controller_task_id
    assert state.trace_id == trace_id
    assert state.last_reconciled_status


def test_pre_resume_stale_repair_missing_explicit_controller_snapshot_preserves_old_id(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-missing",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.controller_task_id == "controller-missing"
    assert state.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert state.last_reconciled_status == "unknown"
    assert state.next_attempt_after > 0


def test_pre_resume_stale_repair_pending_with_old_controller_id_leaves_active_pool(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.PENDING,
            attempts=1,
            controller_task_id="controller-missing",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.controller_task_id == "controller-missing"
    assert state.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert state.last_reconciled_status == "unknown"


def test_pre_resume_stale_repair_pending_attempt_uses_current_deterministic_id(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.PENDING, attempts=2))
    controller_task_id, trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=2,
    )
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id=controller_task_id,
            trace_id=trace_id,
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 2},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
        controller_task_id,
        TaskState.FINISHED,
        result_json=TaskFinished(
            task_id=controller_task_id,
            trace_id=trace_id,
            session_id="session-1",
            state=TaskState.FINISHED,
            device_id="device-1",
        ).model_dump_json(),
        finished=True,
    )
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.SUCCEEDED
    assert state.controller_task_id == controller_task_id
    assert state.trace_id == trace_id


def test_pre_resume_stale_repair_missing_snapshot_preserves_resource_ids(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-missing",
            sandbox_id="sandbox-old",
            device_id="device-old",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.controller_task_id == "controller-missing"
    assert state.sandbox_id == "sandbox-old"
    assert state.device_id == "device-old"


def test_pre_resume_cleanup_builds_active_controller_indexes(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-missing",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            maintenance=BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([])),
        )
    )

    with sqlite3.connect(tmp_path / "batch-1" / "batch_state.sqlite") as connection:
        indexes = {str(row[1]) for row in connection.execute("PRAGMA index_list(batch_tasks)").fetchall()}
    assert "idx_batch_tasks_pre_resume_active_controller_empty_lease" in indexes
    assert "idx_batch_tasks_pre_resume_active_controller_expired_lease" in indexes
    assert "idx_batch_tasks_pre_resume_active_live_lease" in indexes
    assert "idx_batch_tasks_pre_resume_active_controller_live_lease" in indexes
    assert "idx_batch_tasks_pre_resume_pending_history" in indexes
    assert "idx_batch_tasks_pre_resume_pending_history_live_lease" in indexes
    assert "idx_batch_tasks_pre_resume_pending_live_lease" in indexes


def test_pre_resume_cleanup_releases_accident_shaped_dead_active_leases_before_resume(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch-1"
    batch_dir.mkdir()
    task_count = 476
    manifest = BatchManifest(
        batch_id="batch-1",
        tasks=[
            BatchTaskSpec(task_id=f"case-{index:04d}", agent_kind="desktop", query="Do it")
            for index in range(task_count)
        ],
    )
    (batch_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    for index in range(task_count):
        task_id = f"case-{index:04d}"
        repository.upsert_state(BatchTaskState(task_id=task_id, status=BatchTaskStatus.SUBMITTED, attempts=4))
        if index < 240:
            owner = "runner-ac0b843a44ed4cb8aab9acfecfb4b1a0"
        elif index < 391:
            owner = "runner-e25af4ccbce64a5d872140c7e2f4312f"
        else:
            owner = ""
        _set_active_lease(repository, task_id, owner=owner)
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            maintenance=BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([])),
        )
    )

    summary = json.loads((batch_dir / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["pre_resume_repair"]["applied_count"] == task_count
    assert summary["pre_resume_repair"]["metadata"]["include_active_leases"] is True
    assert summary["pre_resume_repair"]["metadata"]["active_lease_states_planned"] == task_count
    assert summary["blocking_actions_count"] == 0
    assert summary["active_leases_after_repair"]["count"] == 0
    with sqlite3.connect(repository.path) as connection:
        assert connection.execute("SELECT count(*) FROM batch_tasks WHERE status='submitted'").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM batch_tasks WHERE lease_token<>''").fetchone()[0] == 0


def test_pre_resume_cleanup_clears_fresh_pending_active_lease_without_failing_case(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    _set_active_lease(repository, "case-1", owner="dead-runner")
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            maintenance=BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([])),
        )
    )

    state = repository.get("case-1")
    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["pre_resume_repair"]["by_kind"] == {"clear_active_lease": 1}
    assert summary["active_leases_after_repair"]["count"] == 0
    assert state is not None
    assert state.status == BatchTaskStatus.PENDING
    with sqlite3.connect(repository.path) as connection:
        row = connection.execute(
            "SELECT lease_owner, lease_token, lease_expires_at FROM batch_tasks WHERE task_id='case-1'"
        ).fetchone()
    assert row == ("", "", 0.0)


def test_pre_resume_cleanup_clears_failed_retryable_active_lease_without_changing_status(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=2,
            controller_task_id="controller-unknown",
            last_reconciled_status="unknown",
            retry_budget_remaining=2,
            next_attempt_after=0.0,
        )
    )
    _set_active_lease(repository, "case-1", owner="dead-runner")
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            maintenance=BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([])),
        )
    )

    state = repository.get("case-1")
    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["pre_resume_repair"]["by_kind"] == {"clear_active_lease": 1}
    assert summary["active_leases_after_repair"]["count"] == 0
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.controller_task_id == "controller-unknown"
    assert state.last_reconciled_status == "unknown"
    assert state.retry_budget_remaining == 2
    with sqlite3.connect(repository.path) as connection:
        row = connection.execute(
            "SELECT lease_owner, lease_token, lease_expires_at FROM batch_tasks WHERE task_id='case-1'"
        ).fetchone()
    assert row == ("", "", 0.0)


def test_pre_resume_cleanup_releases_expired_prepared_resources_before_controller_cleanup(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="dead-runner",
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
        sandbox_id="sandbox-prepared",
        device_id="device-prep",
        lease_id="lease-prep",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE prepared_sandboxes SET expires_at=1 WHERE task_id='case-1'")
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    sandboxes = FakeSandboxMaintenanceClient([])

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            maintenance=BatchMaintenanceOperations(sandboxes=sandboxes),
        )
    )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    resolved = store.get(task_id="case-1", attempt=1)
    assert summary["passed"] is True
    assert summary["prepared_cleanup_needed"] is True
    assert summary["prepared_sandbox_cleanup"]["applied_count"] == 1
    assert summary["prepared_live_budget"]["active_unassigned"] == 0
    assert summary["prepared_live_budget"]["live_total"] == 0
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert sandboxes.terminated == ["sandbox-prepared"]


def test_pre_resume_cleanup_preserves_nonterminal_controller_snapshot_but_clears_dead_lease(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    _set_active_lease(repository, "case-1", owner="dead-runner")
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.close()

    with pytest.raises(RuntimeError, match="controller residue blocked"):
        _run(
            _run_pre_resume_cleanup(
                batch_id="batch-1",
                output_dir=tmp_path,
                controller_db=controller_db,
                provider="fake",
                maintenance=BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([])),
            )
        )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    state = repository.get("case-1")
    assert summary["passed"] is False
    assert summary["pre_resume_repair"]["by_kind"] == {"reconcile_controller_task": 1}
    assert summary["controller_residue_cleanup"]["blocked_by_reason"] == {"current_active_batch_row": 1}
    assert state is not None
    assert state.status == BatchTaskStatus.RUNNING
    assert state.controller_task_id == "controller-1"
    with sqlite3.connect(repository.path) as connection:
        row = connection.execute(
            "SELECT lease_owner, lease_token, lease_expires_at FROM batch_tasks WHERE task_id='case-1'"
        ).fetchone()
    assert row == ("", "", 0.0)


def test_pre_resume_cleanup_applies_post_cleanup_repair_before_verify(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    calls: list[tuple[str, bool | None]] = []

    class FakeMaintenance:
        async def repair_pre_resume_stale(self, *, apply: bool, **kwargs: Any) -> MaintenanceApplyResult:
            calls.append(("repair_pre_resume_stale", apply))
            planned = 1 if apply and len([call for call in calls if call[0] == "repair_pre_resume_stale"]) == 2 else 0
            return MaintenanceApplyResult(
                batch_id="batch-1",
                command="pre-resume-stale-repair",
                dry_run=not apply,
                planned_count=planned,
                applied_count=planned if apply else 0,
                plan=MaintenancePlan(
                    batch_id="batch-1",
                    batch_dir=tmp_path / "batch-1",
                    command="pre-resume-stale-repair",
                    dry_run=not apply,
                    items=[],
                ),
            )

        async def repair_stale(self, *, apply: bool, **kwargs: Any) -> MaintenanceApplyResult:
            calls.append(("repair_stale", apply))
            return MaintenanceApplyResult(
                batch_id="batch-1",
                command="repair-stale",
                dry_run=not apply,
                plan=MaintenancePlan(batch_id="batch-1", batch_dir=tmp_path / "batch-1", command="repair-stale"),
            )

        async def cleanup_controller_residue(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(("cleanup_controller_residue", None))
            return {"planned": {}, "cleaned": {}, "provider_status": {}, "warnings": []}

        async def sweep_provider_orphans(self, *, apply: bool, **kwargs: Any) -> MaintenanceApplyResult:
            calls.append(("sweep_provider_orphans", apply))
            return MaintenanceApplyResult(
                batch_id="batch-1",
                command="sweep-provider-orphans",
                dry_run=not apply,
                plan=MaintenancePlan(
                    batch_id="batch-1",
                    batch_dir=tmp_path / "batch-1",
                    command="sweep-provider-orphans",
                    dry_run=not apply,
                    metadata={
                        "inventory_count": 0,
                        "protected_count": 0,
                        "skipped_count": 0,
                        "candidate_count": 0,
                    },
                ),
            )

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            maintenance=FakeMaintenance(),  # type: ignore[arg-type]
        )
    )

    assert calls == [
        ("repair_pre_resume_stale", True),
        ("repair_stale", True),
        ("cleanup_controller_residue", None),
        ("repair_pre_resume_stale", True),
        ("sweep_provider_orphans", True),
        ("sweep_provider_orphans", False),
        ("repair_pre_resume_stale", False),
    ]
    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["post_cleanup_repair"]["applied_count"] == 1


def test_pre_resume_stale_repair_ignores_fresh_pending_rows(tmp_path: Path) -> None:
    tasks = [BatchTaskSpec(task_id=f"pending-{index}", agent_kind="desktop", query="queued") for index in range(25)]
    tasks.append(BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="stale"))
    repository = SQLiteBatchStateRepository(tmp_path / "batch-1" / "batch_state.sqlite")
    repository.initialize_manifest(BatchManifest(batch_id="batch-1", tasks=tasks))
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-missing",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.repair_pre_resume_stale(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            older_than_seconds=0,
            apply=True,
        )
    )

    assert result.planned_count == 1
    assert result.applied_count == 1
    assert repository.get("case-1").controller_task_id == "controller-missing"
    assert repository.get("pending-0").status == BatchTaskStatus.PENDING
    assert repository.get("pending-0").last_reconciled_status == ""

    with sqlite3.connect(tmp_path / "batch-1" / "batch_state.sqlite") as connection:
        plan = [
            str(row[-1])
            for row in connection.execute("""
                EXPLAIN QUERY PLAN
                SELECT *
                FROM batch_tasks INDEXED BY idx_batch_tasks_pre_resume_pending_history_empty_id
                WHERE status='pending' AND controller_task_id='' AND attempts>0 AND lease_token=''
                ORDER BY order_index ASC
                """).fetchall()
        ]
    assert any("idx_batch_tasks_pre_resume_pending_history_empty_id" in step for step in plan)


def test_pre_resume_cleanup_repairs_local_stale_rows_without_controller_db(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.STARTING_SANDBOX, attempts=1))
    with sqlite3.connect(tmp_path / "batch-1" / "batch_state.sqlite") as connection:
        connection.execute("UPDATE batch_tasks SET updated_at=0 WHERE task_id='case-1'")

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=tmp_path / "missing-controller.sqlite",
            provider="fake",
        )
    )

    state = repository.get("case-1")
    assert state is not None
    assert state.status == BatchTaskStatus.FAILED
    assert state.last_failure_category == FailureCategory.SANDBOX_CREATE
    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["blocking_actions_count"] == 0


def test_pre_resume_cleanup_aborts_when_controller_lookup_attempt_cap_exceeded(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=33,
            retry_budget_remaining=1,
            max_attempts=40,
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    maintenance = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    with pytest.raises(RuntimeError, match="deterministic lookup cap exceeded"):
        _run(
            _run_pre_resume_cleanup(
                batch_id="batch-1",
                output_dir=tmp_path,
                controller_db=controller_db,
                provider="fake",
                maintenance=maintenance,
            )
        )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is False
    assert any("deterministic lookup cap exceeded" in item for item in summary["errors"])


def test_pre_resume_cleanup_aborts_on_current_batch_controller_residue_blocked(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            controller_task_id="controller-current",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    with sqlite3.connect(controller_db) as connection:
        connection.execute(
            "INSERT INTO tasks(task_id, trace_id, agent_kind, submitter_id, request_json, status, result_json, "
            "created_at, finished_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "controller-current",
                "trace-current",
                "desktop",
                "submitter",
                TaskRequest(
                    task_id="controller-current",
                    trace_id="trace-current",
                    agent_kind="desktop",
                    request="Do it",
                    metadata={
                        "batch_id": "batch-1",
                        "batch_task_id": "case-1",
                        "batch_attempt": 1,
                        "sandbox_id": "sandbox-old",
                    },
                ).model_dump_json(),
                "running",
                "{}",
                "2026-06-02T00:00:00Z",
                None,
            ),
        )
    maintenance = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    with pytest.raises(RuntimeError, match="controller residue blocked"):
        _run(
            _run_pre_resume_cleanup(
                batch_id="batch-1",
                output_dir=tmp_path,
                controller_db=controller_db,
                provider="fake",
                maintenance=maintenance,
            )
        )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is False
    assert summary["controller_residue_cleanup"]["blocked_by_reason"] == {"current_active_batch_row": 1}


def test_pre_resume_cleanup_reports_foreign_orphan_device_without_blocking(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    with sqlite3.connect(controller_db) as connection:
        connection.execute(
            "INSERT INTO devices(device_id, status, descriptor_json, last_heartbeat_at, active_sessions, load_json) "
            "VALUES (?,?,?,?,?,?)",
            (
                "batch-old-batch-sc-0001-prep-abcd1234",
                "online",
                json.dumps({"display_name": "old-batch:sc-0001"}),
                "2026-06-02T00:00:00Z",
                0,
                "{}",
            ),
        )
    maintenance = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            maintenance=maintenance,
        )
    )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["controller_residue_cleanup"]["blocked_by_reason"] == {"foreign_batch_online_orphan_device": 1}


def test_pre_resume_cleanup_retries_controller_residue_after_provider_terminate_failure(tmp_path: Path) -> None:
    controller_task_id, _trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=1,
    )
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id=controller_task_id,
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    with sqlite3.connect(controller_db) as connection:
        connection.execute(
            "INSERT INTO tasks(task_id, trace_id, agent_kind, submitter_id, request_json, status, result_json, "
            "created_at, finished_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                controller_task_id,
                "trace-current",
                "desktop",
                "submitter",
                TaskRequest(
                    task_id=controller_task_id,
                    trace_id="trace-current",
                    agent_kind="desktop",
                    request="Do it",
                    metadata={
                        "batch_id": "batch-1",
                        "batch_task_id": "case-1",
                        "batch_attempt": 1,
                        "sandbox_id": "sandbox-old",
                    },
                ).model_dump_json(),
                "failed",
                "{}",
                "2026-06-02T00:00:00Z",
                "2026-06-02T00:00:05Z",
            ),
        )
        connection.execute(
            "INSERT INTO sessions(session_id, task_id, device_id, lease_id, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                "session-old",
                controller_task_id,
                "device-old",
                "lease-old",
                "running",
                "2026-06-02T00:00:00Z",
                "2026-06-02T00:00:00Z",
            ),
        )
        connection.execute(
            "INSERT INTO leases(lease_id, session_id, device_id, status, expires_at, generation) VALUES (?,?,?,?,?,?)",
            ("lease-old", "session-old", "device-old", "leased", "2026-06-03T00:00:00Z", 1),
        )
        connection.execute(
            "INSERT INTO devices(device_id, status, descriptor_json, last_heartbeat_at, active_sessions, load_json) "
            "VALUES (?,?,?,?,?,?)",
            ("device-old", "online", json.dumps({"display_name": "batch-1:case-1"}), "2026-06-02T00:00:00Z", 1, "{}"),
        )
    maintenance = BatchMaintenanceOperations(
        sandboxes=FakeSandboxMaintenanceClient(
            [
                SandboxInventoryItem(
                    sandbox_id="sandbox-old",
                    provider="fake",
                    template="desktop",
                    state=SandboxInventoryState.RUNNING,
                    metadata={"batch_id": "batch-1", "task_id": "case-1", "attempt": 1},
                )
            ],
            terminate_error="sandbox API route returned 404",
        )
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="provider terminate_failed=1"):
            _run(
                _run_pre_resume_cleanup(
                    batch_id="batch-1",
                    output_dir=tmp_path,
                    controller_db=controller_db,
                    provider="fake",
                    maintenance=maintenance,
                )
            )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is False
    assert summary["controller_residue_cleanup"]["provider_status"] == {"terminate_failed": 1}
    with sqlite3.connect(controller_db) as connection:
        assert (
            connection.execute("SELECT status FROM tasks WHERE task_id=?", (controller_task_id,)).fetchone()[0]
            == "failed"
        )
        assert connection.execute("SELECT status FROM devices WHERE device_id='device-old'").fetchone()[0] == "online"


def test_pre_resume_cleanup_resolves_terminal_prepared_orphan_without_provider_row(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assert store.assign(task_id="case-1", attempt=1, runner_id="runner-a") is not None
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.FAILED, attempts=1))
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    sandboxes = FakeSandboxMaintenanceClient([])
    maintenance = BatchMaintenanceOperations(sandboxes=sandboxes)

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            run_concurrency=1,
            maintenance=maintenance,
        )
    )

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert sandboxes.terminated == ["sandbox-prepared"]
    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["prepared_sandbox_cleanup"]["by_kind"] == {"terminate_sandbox": 1}
    assert summary["prepared_live_budget"]["assigned_terminal_with_resource"] == 0


def test_pre_resume_cleanup_aborts_when_direct_prepared_terminate_fails(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assert store.assign(task_id="case-1", attempt=1, runner_id="runner-a") is not None
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.FAILED, attempts=1))
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    maintenance = BatchMaintenanceOperations(
        sandboxes=FakeSandboxMaintenanceClient([], terminate_error="sandbox API route returned 404")
    )

    with pytest.raises(RuntimeError, match="prepared sandbox cleanup failed"):
        _run(
            _run_pre_resume_cleanup(
                batch_id="batch-1",
                output_dir=tmp_path,
                controller_db=controller_db,
                provider="fake",
                run_concurrency=1,
                maintenance=maintenance,
            )
        )

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.ASSIGNED
    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is False
    assert "prepared sandbox cleanup failed" in summary["errors"][0]


def test_pre_resume_cleanup_treats_direct_prepared_provider_missing_as_resolved(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assert store.assign(task_id="case-1", attempt=1, runner_id="runner-a") is not None
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.FAILED, attempts=1))
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    maintenance = BatchMaintenanceOperations(
        sandboxes=FakeSandboxMaintenanceClient([], terminate_error="sandbox sandbox-prepared not found 404")
    )

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            run_concurrency=1,
            maintenance=maintenance,
        )
    )

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["prepared_sandbox_cleanup"]["errors"] == []


def test_pre_resume_cleanup_terminates_and_resolves_prepared_cleanup_debt(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    cleanup_failed = store.mark_cleanup_failed(prepared, error="release failed")
    assert cleanup_failed is not None
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.FAILED, attempts=1))
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    maintenance = BatchMaintenanceOperations(sandboxes=sandboxes)

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            run_concurrency=1,
            maintenance=maintenance,
        )
    )

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert resolved.error == "cleanup resolved by maintenance terminate-sandboxes"
    assert sandboxes.terminated == ["sandbox-prepared"]
    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["prepared_sandbox_cleanup"]["by_kind"] == {"terminate_sandbox": 1}
    assert summary["prepared_live_budget"]["cleanup_required_with_resource"] == 0


def test_pre_resume_cleanup_drains_live_prepared_buffer_before_controller_cleanup(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    store = SQLitePreparedSandboxStore(db_path)
    device_id = "batch-batch-1-sc-case-1-prep-c6ac19d4"
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
    assert (
        store.mark_prepared(
            preparing,
            sandbox_id="sandbox-prepared",
            device_id=device_id,
            lease_id="lease",
            prepare_seconds=1.0,
        )
        is not None
    )
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    with sqlite3.connect(controller_db) as connection:
        connection.execute(
            "INSERT INTO devices(device_id, status, descriptor_json, last_heartbeat_at, active_sessions, load_json) "
            "VALUES (?,?,?,?,?,?)",
            (
                device_id,
                "online",
                json.dumps({"display_name": "batch-1:case-1", "metadata": {"batch_id": "batch-1"}}),
                "2026-06-02T00:00:00Z",
                0,
                "{}",
            ),
        )
    sandboxes = FakeSandboxMaintenanceClient([])
    maintenance = BatchMaintenanceOperations(sandboxes=sandboxes)

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            run_concurrency=1,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1),
            maintenance=maintenance,
        )
    )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert summary["passed"] is True
    assert summary["prepared_unassigned_expired_for_resume"]["count"] == 1
    assert summary["prepared_unassigned_expired_for_resume"]["with_resource"] == 1
    assert summary["prepared_cleanup_needed"] is True
    assert summary["prepared_sandbox_cleanup"]["by_kind"] == {"terminate_sandbox": 1}
    assert summary["controller_residue_cleanup"]["cleaned"]["devices"] == 1
    assert summary["provider_orphan_sweep_verify"]["metadata"]["candidate_count"] == 0
    assert summary["prepared_live_budget"]["active_unassigned"] == 0
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert sandboxes.terminated == ["sandbox-prepared"]
    with sqlite3.connect(controller_db) as connection:
        assert (
            connection.execute("SELECT status FROM devices WHERE device_id=?", (device_id,)).fetchone()[0] == "offline"
        )


def test_pre_resume_cleanup_expires_resource_empty_prepared_placeholder(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="old-runner",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=3600,
    )
    assert record is not None
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    sandboxes = FakeSandboxMaintenanceClient([])

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            run_concurrency=1,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1),
            maintenance=BatchMaintenanceOperations(sandboxes=sandboxes),
        )
    )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert summary["passed"] is True
    assert summary["prepared_unassigned_expired_for_resume"]["count"] == 1
    assert summary["prepared_unassigned_expired_for_resume"]["with_resource"] == 0
    assert summary["prepared_cleanup_needed"] is False
    assert summary["prepared_live_budget"]["active_unassigned"] == 0
    assert resolved.status == PreparedSandboxStatus.EXPIRED
    assert sandboxes.terminated == []


def test_pre_resume_cleanup_terminal_assigned_without_resource_does_not_overrun_budget(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch-1"
    batch_dir.mkdir()
    manifest = BatchManifest(
        batch_id="batch-1",
        tasks=[BatchTaskSpec(task_id=f"case-{index}", agent_kind="desktop", query="Do it") for index in range(3)],
    )
    (batch_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(batch_dir / "batch_state.sqlite")
    for index in range(3):
        task_id = f"case-{index}"
        record = store.reserve(
            task_id=task_id,
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
        assert store.assign(task_id=task_id, attempt=1, runner_id="runner-a") is not None
        repository.upsert_state(BatchTaskState(task_id=task_id, status=BatchTaskStatus.FAILED, attempts=1))
    with sqlite3.connect(batch_dir / "batch_state.sqlite") as connection:
        connection.execute("UPDATE prepared_sandboxes SET sandbox_id='', device_id='', lease_id=''")
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    maintenance = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            run_concurrency=1,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1),
            maintenance=maintenance,
        )
    )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["prepared_cleanup_needed"] is False
    assert summary["provider_orphan_sweep_verify"]["metadata"]["candidate_count"] == 0
    assert summary["prepared_live_budget"]["assigned_total"] == 3
    assert summary["prepared_live_budget"]["assigned_live"] == 0
    assert summary["prepared_live_budget"]["live_e2b_budget_overrun"] == 0


def test_pre_resume_cleanup_treats_evaluation_pending_as_terminal_prepared_assignment(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch-1"
    batch_dir.mkdir()
    manifest = BatchManifest(
        batch_id="batch-1",
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    (batch_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    store = SQLitePreparedSandboxStore(batch_dir / "batch_state.sqlite")
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
        sandbox_id="sandbox-prepared",
        device_id="device-prepared",
        lease_id="lease-prepared",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assert store.assign(task_id="case-1", attempt=1, runner_id="runner-a") is not None
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.EVALUATION_PENDING,
            attempts=1,
            evaluation_status="pending",
        )
    )
    summary = _prepared_live_budget_summary(batch_dir / "batch_state.sqlite", prepared_live_limit=1)

    assert summary["assigned_total"] == 1
    assert summary["assigned_live"] == 0
    assert summary["assigned_terminal_with_resource"] == 1
    assert summary["live_e2b_budget_overrun"] == 0


def test_pre_resume_cleanup_writes_summary_for_unexpected_exception(tmp_path: Path) -> None:
    _init_batch(tmp_path)

    class BoomMaintenance:
        async def repair_stale(self, **kwargs: object) -> object:
            _ = kwargs
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run(
            _run_pre_resume_cleanup(
                batch_id="batch-1",
                output_dir=tmp_path,
                controller_db=tmp_path / "missing-controller.sqlite",
                provider="fake",
                maintenance=BoomMaintenance(),  # type: ignore[arg-type]
            )
        )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is False
    assert "RuntimeError: boom" in summary["errors"]


def test_resume_maintenance_indexes_exist_for_large_table_cleanup(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    SQLiteMaintenanceStore(db_path)
    SQLitePreparedSandboxStore(db_path)
    _ensure_prepared_indexes(db_path)

    with sqlite3.connect(db_path) as connection:
        indexes = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
        }

    assert "idx_batch_tasks_maintenance_status_lease" in indexes
    assert "idx_batch_tasks_controller_task_id" in indexes
    assert "idx_batch_tasks_status_updated" in indexes
    assert "idx_batch_tasks_maintenance_unleased_empty" in indexes
    assert "idx_batch_tasks_maintenance_unleased_expired" in indexes
    assert "idx_prepared_sandboxes_task_status" in indexes
    assert "idx_prepared_sandboxes_sandbox" in indexes
    assert "idx_prepared_sandboxes_resource_debt" in indexes
    assert "idx_batch_tasks_status_task" in indexes


def test_pre_resume_controller_residue_blocker_rejects_incomplete_provider_cleanup() -> None:
    assert (
        _controller_residue_cleanup_blocker(
            {
                "planned": {"sandboxes": 2},
                "cleaned": {"sandboxes": 1},
                "provider_status": {"terminated": 1},
                "warnings": [],
            }
        )
        == "provider cleanup incomplete: planned_sandboxes=2 cleaned_sandboxes=1"
    )


def test_pre_resume_blocking_actions_allows_active_nonterminal_status_drift() -> None:
    active_nonterminal = MaintenancePlanItem(
        action_id="maint-1",
        kind=MaintenanceActionKind.RECONCILE_CONTROLLER_TASK,
        details={
            "controller_snapshot": {"state": "running"},
            "expected_row": {"state": {"status": "running"}},
        },
    )
    terminal = MaintenancePlanItem(
        action_id="maint-2",
        kind=MaintenanceActionKind.RECONCILE_CONTROLLER_TASK,
        details={
            "controller_snapshot": {"state": "finished"},
            "expected_row": {"state": {"status": "running"}},
        },
    )
    next_attempt_nonterminal = MaintenancePlanItem(
        action_id="maint-3",
        kind=MaintenanceActionKind.RECONCILE_CONTROLLER_TASK,
        details={
            "controller_snapshot": {"state": "accepted"},
            "expected_row": {"state": {"status": "failed"}},
        },
    )

    assert _pre_resume_blocking_actions([active_nonterminal]) == []
    assert _pre_resume_blocking_actions([terminal]) == [terminal]
    assert _pre_resume_blocking_actions([next_attempt_nonterminal]) == [next_attempt_nonterminal]


def test_reconcile_terminal_controller_snapshot_updates_sqlite(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    controller = FakeControllerMaintenanceClient(
        [
            TaskStatusSnapshot(
                task_id="controller-1",
                trace_id="trace-1",
                agent_kind="desktop",
                state=TaskState.FINISHED,
            )
        ]
    )
    operations = BatchMaintenanceOperations(controller=controller, sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.reconcile(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="tok",
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert state is not None
    assert state.status == BatchTaskStatus.SUCCEEDED
    assert state.last_reconciled_status == TaskState.FINISHED.value
    assert result.applied_count == 1
    assert controller.list_limits[-1] >= 10_000


def test_reconcile_can_use_local_controller_db_without_websocket_url(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    controller_db = tmp_path / "controller.sqlite"
    controller_store = SQLiteControllerStore(controller_db)
    controller_store.create_task(
        TaskRequest(
            task_id="controller-1",
            trace_id="trace-1",
            agent_kind="desktop",
            request="Do it",
            metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
        ),
        session_id="session-1",
        submitter_id="submitter",
    )
    controller_store.update_task_status(
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
    controller_store.close()
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.reconcile(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert result.applied_count == 1
    assert state is not None
    assert state.status == BatchTaskStatus.SUCCEEDED
    assert state.controller_task_id == "controller-1"
    assert state.device_id == "device-1"


def test_reconcile_db_only_requires_readable_controller_db(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    operations = BatchMaintenanceOperations(sandboxes=FakeSandboxMaintenanceClient([]))

    with pytest.raises(ValueError, match="readable controller DB is required"):
        _run(
            operations.reconcile(
                batch_id="batch-1",
                output_dir=tmp_path,
                controller_db=tmp_path / "missing.sqlite",
            )
        )


def test_reconcile_uses_controller_batch_metadata_when_task_id_is_missing(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.RUNNING, attempts=1))
    controller = FakeControllerMaintenanceClient(
        [
            TaskStatusSnapshot(
                task_id="controller-random",
                trace_id="trace-1",
                agent_kind="desktop",
                state=TaskState.FINISHED,
                metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
            )
        ]
    )
    operations = BatchMaintenanceOperations(controller=controller, sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.reconcile(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="tok",
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert state is not None
    assert state.controller_task_id == "controller-random"
    assert state.status == BatchTaskStatus.SUCCEEDED
    assert result.applied_count == 1
    assert result.items[0].after["controller_task_id"] == "controller-random"


def test_reconcile_metadata_fallback_requires_same_attempt(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.RUNNING, attempts=2))
    controller = FakeControllerMaintenanceClient(
        [
            TaskStatusSnapshot(
                task_id="controller-old-attempt",
                trace_id="trace-1",
                agent_kind="desktop",
                state=TaskState.FINISHED,
                metadata={"batch_id": "batch-1", "batch_task_id": "case-1", "batch_attempt": 1},
            )
        ]
    )
    operations = BatchMaintenanceOperations(controller=controller, sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.reconcile(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="tok",
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert state is not None
    assert state.status == BatchTaskStatus.RUNNING
    assert state.controller_task_id == ""
    assert result.applied_count == 0
    assert result.skipped_count == 1
    assert result.plan.items[0].reason == "controller task was not found"


def test_reconcile_rejects_row_changed_after_plan(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    controller = FakeControllerMaintenanceClient(
        [
            TaskStatusSnapshot(
                task_id="controller-1",
                trace_id="trace-1",
                agent_kind="desktop",
                state=TaskState.FINISHED,
            )
        ]
    )
    store = SQLiteMaintenanceStore(tmp_path / "batch-1" / "batch_state.sqlite")
    plan = _run(
        MaintenancePlanner().reconcile(
            batch_id="batch-1",
            batch_dir=tmp_path / "batch-1",
            store=store,
            manifest=_manifest(),
            controller=controller,
            controller_url="ws://controller",
            token="tok",
        )
    )
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=2,
            controller_task_id="controller-2",
        )
    )

    result = _run(
        MaintenanceExecutor(store=store, controller=controller, sandboxes=FakeSandboxMaintenanceClient([])).execute(
            plan=plan,
            apply=True,
            controller_url="ws://controller",
            token="tok",
        )
    )

    state = repository.get("case-1")
    assert state is not None
    assert state.controller_task_id == "controller-2"
    assert result.applied_count == 0
    assert result.errors


def test_cancel_case_sends_controller_cancel_without_local_terminal_mark(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            controller_task_id="controller-1",
            trace_id="trace-1",
        )
    )
    controller = FakeControllerMaintenanceClient()
    operations = BatchMaintenanceOperations(controller=controller, sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.cancel_case(
            batch_id="batch-1",
            output_dir=tmp_path,
            case_id="case-1",
            controller_url="ws://controller",
            token="tok",
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert state is not None
    assert state.status == BatchTaskStatus.RUNNING
    assert controller.canceled == ["controller-1"]
    assert result.applied_count == 1
    assert result.plan.warnings


def test_cancel_case_rejects_controller_task_changed_after_plan(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    store = SQLiteMaintenanceStore(tmp_path / "batch-1" / "batch_state.sqlite")
    state = store.get_state("case-1")
    assert state is not None
    plan = MaintenancePlanner().cancel_case(
        batch_id="batch-1",
        batch_dir=tmp_path / "batch-1",
        store=store,
        state=state,
    )
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=2,
            controller_task_id="controller-2",
        )
    )
    controller = FakeControllerMaintenanceClient()

    result = _run(
        MaintenanceExecutor(store=store, controller=controller, sandboxes=FakeSandboxMaintenanceClient([])).execute(
            plan=plan,
            apply=True,
            controller_url="ws://controller",
            token="tok",
        )
    )

    assert controller.canceled == []
    state = repository.get("case-1")
    assert state is not None
    assert state.controller_task_id == "controller-2"
    assert result.applied_count == 0
    assert result.errors


def test_cancel_case_marks_local_pre_controller_case(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.STARTING_SANDBOX,
            attempts=1,
        )
    )
    controller = FakeControllerMaintenanceClient()
    operations = BatchMaintenanceOperations(controller=controller, sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.cancel_case(
            batch_id="batch-1",
            output_dir=tmp_path,
            case_id="case-1",
            controller_url="ws://controller",
            token="tok",
            apply=True,
        )
    )

    state = repository.get("case-1")
    assert state is not None
    assert state.status == BatchTaskStatus.CANCELED
    assert controller.canceled == []
    assert result.applied_count == 1


def test_cancel_case_rejects_local_row_gaining_controller_task_after_plan(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.STARTING_SANDBOX,
            attempts=1,
        )
    )
    store = SQLiteMaintenanceStore(tmp_path / "batch-1" / "batch_state.sqlite")
    state = store.get_state("case-1")
    assert state is not None
    plan = MaintenancePlanner().cancel_case(
        batch_id="batch-1",
        batch_dir=tmp_path / "batch-1",
        store=store,
        state=state,
    )
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            controller_task_id="controller-1",
        )
    )

    result = _run(
        MaintenanceExecutor(
            store=store,
            controller=FakeControllerMaintenanceClient(),
            sandboxes=FakeSandboxMaintenanceClient([]),
        ).execute(plan=plan, apply=True)
    )

    state = repository.get("case-1")
    assert state is not None
    assert state.status == BatchTaskStatus.RUNNING
    assert state.controller_task_id == "controller-1"
    assert result.applied_count == 0
    assert result.errors


def test_cancel_case_does_not_mark_sqlite_when_controller_cancel_fails(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    original = BatchTaskState(
        task_id="case-1",
        status=BatchTaskStatus.RUNNING,
        attempts=1,
        controller_task_id="controller-1",
        trace_id="trace-1",
    )
    repository.upsert_state(original)
    controller = FakeControllerMaintenanceClient(cancel_error="not owner")
    operations = BatchMaintenanceOperations(controller=controller, sandboxes=FakeSandboxMaintenanceClient([]))

    result = _run(
        operations.cancel_case(
            batch_id="batch-1",
            output_dir=tmp_path,
            case_id="case-1",
            controller_url="ws://controller",
            token="tok",
            apply=True,
        )
    )

    assert repository.get("case-1") == original
    assert result.applied_count == 0
    assert result.errors


def test_cancel_case_refuses_terminal_failure_reclassification(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    original = BatchTaskState(task_id="case-1", status=BatchTaskStatus.FAILED, attempts=1, error="failed")
    repository.upsert_state(original)
    operations = BatchMaintenanceOperations(
        controller=FakeControllerMaintenanceClient(),
        sandboxes=FakeSandboxMaintenanceClient([]),
    )

    result = _run(
        operations.cancel_case(
            batch_id="batch-1",
            output_dir=tmp_path,
            case_id="case-1",
            controller_url="ws://controller",
            token="tok",
            apply=True,
        )
    )

    assert repository.get("case-1") == original
    assert result.applied_count == 0
    assert result.skipped_count == 1


def test_terminate_sandboxes_filters_by_batch_metadata(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    matching = SandboxInventoryItem(
        sandbox_id="sandbox-1",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={"northcua": True, "batch_id": "batch-1", "task_id": "case-1", "attempt": 1},
    )
    other = SandboxInventoryItem(
        sandbox_id="sandbox-2",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={"northcua": True, "batch_id": "other", "task_id": "case-2", "attempt": 1},
    )
    sandboxes = FakeSandboxMaintenanceClient([matching, other])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(operations.terminate_sandboxes(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True))

    assert sandboxes.terminated == ["sandbox-1"]
    assert result.applied_count == 1


def test_terminate_sandboxes_skips_active_sqlite_lease_without_force(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )
    assert claim is not None
    assert repository.upsert_state(
        BatchTaskState(task_id="case-1", status=BatchTaskStatus.RUNNING, attempts=1),
        lease_token=claim.lease_token,
    )
    matching = SandboxInventoryItem(
        sandbox_id="sandbox-1",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={"northcua": True, "batch_id": "batch-1", "task_id": "case-1", "attempt": 1},
    )
    sandboxes = FakeSandboxMaintenanceClient([matching])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(operations.terminate_sandboxes(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True))

    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.skipped_count == 1


def test_terminate_sandboxes_allows_expired_prepared_reservation_without_force(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
    prepared = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(operations.terminate_sandboxes(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True))

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert resolved.error == "cleanup resolved by maintenance terminate-sandboxes"
    assert sandboxes.terminated == ["sandbox-prepared"]
    assert result.applied_count == 1
    assert result.items[0].after["prepared_cleanup_resolution"]
    assert store.claimable_task_ids(max_attempts_by_task={"case-1": 1}, limit=1) == [("case-1", 1)]


def test_terminate_sandboxes_prepared_only_skips_regular_case_sandboxes(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
    regular = SandboxInventoryItem(
        sandbox_id="sandbox-regular",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={"northcua": True, "batch_id": "batch-1", "task_id": "case-1", "attempt": 1},
    )
    prepared = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([regular, prepared])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.terminate_sandboxes(
            batch_id="batch-1",
            output_dir=tmp_path,
            provider="fake",
            prepared_only=True,
            apply=True,
        )
    )

    assert sandboxes.terminated == ["sandbox-prepared"]
    assert result.applied_count == 1
    assert result.items[0].sandbox_id == "sandbox-prepared"


def test_terminate_sandboxes_prepared_only_marks_expired_prepared_released(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
    prepared_record = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared_record is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE prepared_sandboxes SET expires_at=1 WHERE task_id='case-1' AND attempt=1")
    prepared = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.terminate_sandboxes(
            batch_id="batch-1",
            output_dir=tmp_path,
            provider="fake",
            prepared_only=True,
            apply=True,
        )
    )

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert resolved.error == "cleanup resolved by maintenance terminate-sandboxes"
    assert sandboxes.terminated == ["sandbox-prepared"]
    assert result.applied_count == 1
    assert result.items[0].after["prepared_cleanup_resolution"]


def test_terminate_sandboxes_prepared_only_marks_db_known_prepared_released(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
    prepared_record = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared_record is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE prepared_sandboxes SET expires_at=1 WHERE task_id='case-1' AND attempt=1")
    sandboxes = FakeSandboxMaintenanceClient([])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.terminate_sandboxes(
            batch_id="batch-1",
            output_dir=tmp_path,
            provider="fake",
            prepared_only=True,
            apply=True,
        )
    )

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert resolved.error == "cleanup resolved by maintenance terminate-sandboxes"
    assert sandboxes.terminated == ["sandbox-prepared"]
    assert result.applied_count == 1
    assert result.items[0].after["prepared_cleanup_resolution"]


@pytest.mark.parametrize(
    "prepared_status",
    [
        PreparedSandboxStatus.EXPIRED,
        PreparedSandboxStatus.FAILED,
        PreparedSandboxStatus.CANCELED,
    ],
)
def test_terminate_sandboxes_prepared_only_marks_terminal_prepared_status_released(
    tmp_path: Path, prepared_status: PreparedSandboxStatus
) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=3600,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared_record = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared_record is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE prepared_sandboxes SET status=?, error=? WHERE task_id='case-1' AND attempt=1",
            (prepared_status.value, f"synthetic {prepared_status.value}"),
        )
    sandboxes = FakeSandboxMaintenanceClient([])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.terminate_sandboxes(
            batch_id="batch-1",
            output_dir=tmp_path,
            provider="fake",
            prepared_only=True,
            apply=True,
        )
    )

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert resolved.error == "cleanup resolved by maintenance terminate-sandboxes"
    assert sandboxes.terminated == ["sandbox-prepared"]
    assert result.applied_count == 1
    assert result.items[0].after["prepared_cleanup_resolution"]


def test_terminate_sandboxes_prepared_only_protects_db_known_active_prepared(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=3600,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared_record = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared_record is not None
    sandboxes = FakeSandboxMaintenanceClient([])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.terminate_sandboxes(
            batch_id="batch-1",
            output_dir=tmp_path,
            provider="fake",
            prepared_only=True,
            apply=True,
        )
    )

    protected = store.get(task_id="case-1", attempt=1)
    assert protected is not None
    assert protected.status == PreparedSandboxStatus.PREPARED
    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.skipped_count == 1
    assert result.plan.items[0].kind == MaintenanceActionKind.DIAGNOSE
    assert "prepared sandbox is active" in result.plan.items[0].reason


def test_terminate_sandboxes_prepared_only_protects_retry_backoff_prepared(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=4,
            retry_budget_remaining=2,
            next_attempt_after=time.time() + 300,
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=5,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=3600,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared_record = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared_record is not None
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 5,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.terminate_sandboxes(
            batch_id="batch-1",
            output_dir=tmp_path,
            provider="fake",
            prepared_only=True,
            apply=True,
        )
    )

    protected = store.get(task_id="case-1", attempt=5)
    assert protected is not None
    assert protected.status == PreparedSandboxStatus.PREPARED
    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.skipped_count == 1
    assert result.plan.items[0].kind == MaintenanceActionKind.DIAGNOSE
    assert "retryable terminal case" in result.plan.items[0].reason


def test_terminate_sandboxes_prepared_only_protects_timed_out_retry_even_with_terminal_decision(
    tmp_path: Path,
) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.TIMED_OUT,
            attempts=4,
            retry_budget_remaining=2,
            next_attempt_after=time.time() + 300,
            terminal_decision={"retryable": False},
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=5,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=3600,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared_record = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared_record is not None
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 5,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.terminate_sandboxes(
            batch_id="batch-1",
            output_dir=tmp_path,
            provider="fake",
            prepared_only=True,
            apply=True,
        )
    )

    protected = store.get(task_id="case-1", attempt=5)
    assert protected is not None
    assert protected.status == PreparedSandboxStatus.PREPARED
    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.skipped_count == 1


def test_terminate_sandboxes_prepared_only_cleans_terminal_no_retry_prepared(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=3600,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared_record = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared_record is not None
    assert repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED, attempts=1))
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.terminate_sandboxes(
            batch_id="batch-1",
            output_dir=tmp_path,
            provider="fake",
            prepared_only=True,
            apply=True,
        )
    )

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert sandboxes.terminated == ["sandbox-prepared"]
    assert result.applied_count == 1


def test_terminate_sandboxes_prepared_only_cleans_terminal_assigned_prepared_with_lease(
    tmp_path: Path,
) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=1,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=3600,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared_record = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared_record is not None
    assert store.assign(task_id="case-1", attempt=1, runner_id="runner-a") is not None
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUCCEEDED,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    _set_active_lease(repository, "case-1")
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.terminate_sandboxes(
            batch_id="batch-1",
            output_dir=tmp_path,
            provider="fake",
            prepared_only=True,
            apply=True,
        )
    )

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert sandboxes.terminated == ["sandbox-prepared"]
    assert result.applied_count == 1


@pytest.mark.parametrize(
    ("state", "prepared_attempt"),
    [
        (
            BatchTaskState(
                task_id="case-1",
                status=BatchTaskStatus.CANCELED,
                attempts=4,
                retry_budget_remaining=2,
                next_attempt_after=time.time() + 300,
            ),
            5,
        ),
        (
            BatchTaskState(
                task_id="case-1",
                status=BatchTaskStatus.FAILED,
                attempts=4,
                retry_budget_remaining=2,
                next_attempt_after=time.time() + 300,
            ),
            4,
        ),
        (
            BatchTaskState(
                task_id="case-1",
                status=BatchTaskStatus.FAILED,
                attempts=4,
                retry_budget_remaining=0,
                next_attempt_after=time.time() + 300,
            ),
            5,
        ),
        (
            BatchTaskState(
                task_id="case-1",
                status=BatchTaskStatus.FAILED,
                attempts=4,
                retry_budget_remaining=2,
                next_attempt_after=time.time() + 300,
                recovery_action=FailureAction.FAIL_FAST,
            ),
            5,
        ),
    ],
)
def test_terminate_sandboxes_prepared_only_cleans_non_claimable_terminal_prepared(
    tmp_path: Path, state: BatchTaskState, prepared_attempt: int
) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=4,
            retry_budget_remaining=2,
            next_attempt_after=time.time() + 300,
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=5,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=3600,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared_record = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared_record is not None
    if prepared_attempt != 5:
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                "UPDATE prepared_sandboxes SET attempt=? WHERE task_id='case-1' AND attempt=5",
                (prepared_attempt,),
            )
    assert repository.upsert_state(state)
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": prepared_attempt,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.terminate_sandboxes(
            batch_id="batch-1",
            output_dir=tmp_path,
            provider="fake",
            prepared_only=True,
            apply=True,
        )
    )

    resolved = store.get(task_id="case-1", attempt=prepared_attempt)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert sandboxes.terminated == ["sandbox-prepared"]
    assert result.applied_count == 1


def test_terminate_sandboxes_allows_assigned_prepared_orphan_without_active_lease(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
    assert repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUBMITTED, attempts=1))
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assert store.assign(task_id="case-1", attempt=1, runner_id="runner-a") is not None
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(operations.terminate_sandboxes(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True))

    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert resolved.error == "cleanup resolved by maintenance terminate-sandboxes"
    assert sandboxes.terminated == ["sandbox-prepared"]
    assert result.applied_count == 1
    assert result.items[0].after["prepared_cleanup_resolution"]


def test_terminate_sandboxes_protects_cleanup_deferred_with_controller_owner(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assigned = store.assign(task_id="case-1", attempt=1, runner_id="runner-a")
    assert assigned is not None
    deferred = store.mark_cleanup_deferred(assigned, error="waiting for controller reconcile")
    assert deferred is not None
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(operations.terminate_sandboxes(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True))

    protected = store.get(task_id="case-1", attempt=1)
    assert protected is not None
    assert protected.status == PreparedSandboxStatus.CLEANUP_DEFERRED
    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.skipped_count == 1
    assert result.plan.items[0].kind == MaintenanceActionKind.DIAGNOSE
    assert "controller task id" in result.plan.items[0].reason


def test_terminate_sandboxes_apply_rechecks_assigned_controller_owner(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
    assert repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUBMITTED, attempts=1))
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assigned = store.assign(task_id="case-1", attempt=1, runner_id="runner-a")
    assert assigned is not None
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    plan = MaintenancePlanner().terminate_sandboxes(
        batch_id="batch-1",
        batch_dir=tmp_path / "batch-1",
        store=SQLiteMaintenanceStore(db_path),
        items=[prepared_item],
    )
    assert plan.planned_count == 1
    assert plan.items[0].kind == MaintenanceActionKind.TERMINATE_SANDBOX
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="controller-1",
        )
    )
    executor = MaintenanceExecutor(
        store=SQLiteMaintenanceStore(db_path),
        controller=FakeControllerMaintenanceClient(),
        sandboxes=sandboxes,
    )

    result = _run(executor.execute(plan=plan, apply=True))

    protected = store.get(task_id="case-1", attempt=1)
    assert protected is not None
    assert protected.status == PreparedSandboxStatus.ASSIGNED
    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.errors
    assert "controller-owned assigned prepared sandbox" in result.errors[0]


def test_terminate_sandboxes_apply_rechecks_prepared_row_safety(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
    prepared = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared])
    plan = MaintenancePlanner().terminate_sandboxes(
        batch_id="batch-1",
        batch_dir=tmp_path / "batch-1",
        store=SQLiteMaintenanceStore(db_path),
        items=[prepared],
    )
    assert plan.planned_count == 1
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE prepared_sandboxes
            SET status=?, expires_at=?
            WHERE task_id='case-1' AND attempt=1
            """,
            (PreparedSandboxStatus.PREPARED.value, 9999999999),
        )
    executor = MaintenanceExecutor(
        store=SQLiteMaintenanceStore(db_path),
        controller=FakeControllerMaintenanceClient(),
        sandboxes=sandboxes,
    )

    result = _run(executor.execute(plan=plan, apply=True))

    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.errors


def test_terminate_sandboxes_protects_assigned_prepared_with_active_lease_after_ttl(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    assert store.assign(task_id="case-1", attempt=1, runner_id="runner-a") is not None
    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )
    assert claim is not None
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            sandbox_id="sandbox-prepared",
        ),
        lease_token=claim.lease_token,
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE prepared_sandboxes SET expires_at=1 WHERE task_id='case-1'")
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(operations.terminate_sandboxes(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True))

    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.skipped_count == 1


def test_terminate_sandboxes_inventory_failure_returns_json_result(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    operations = BatchMaintenanceOperations(
        controller=FakeControllerMaintenanceClient(),
        sandboxes=FakeSandboxMaintenanceClient([], inventory_error="inventory unavailable"),
    )

    result = _run(operations.terminate_sandboxes(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True))

    assert result.errors
    assert result.skipped_count == 1
    assert (tmp_path / "batch-1" / "batch_maintenance_events.jsonl").exists()


def test_sweep_provider_orphans_terminates_old_attempt_with_live_current_attempt(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )
    assert claim is not None
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=2,
            sandbox_id="sandbox-current",
        ),
        lease_token=claim.lease_token,
    )
    old_attempt = SandboxInventoryItem(
        sandbox_id="sandbox-old",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={"northcua": True, "batch_id": "batch-1", "task_id": "case-1", "attempt": 1},
    )
    current_attempt = SandboxInventoryItem(
        sandbox_id="sandbox-current",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={"northcua": True, "batch_id": "batch-1", "task_id": "case-1", "attempt": 2},
    )
    sandboxes = FakeSandboxMaintenanceClient([old_attempt, current_attempt])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.sweep_provider_orphans(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True)
    )

    assert sandboxes.terminated == ["sandbox-old"]
    assert result.applied_count == 1
    assert result.plan.metadata["candidate_count"] == 1
    assert result.plan.metadata["candidates_by_reason"] == {"old_attempt": 1}
    assert result.plan.metadata["protected_by_reason"] == {"protected_current_sandbox_id": 1}


def test_sweep_provider_orphans_terminates_terminal_and_missing_rows(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    assert repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED, attempts=1))
    terminal = SandboxInventoryItem(
        sandbox_id="sandbox-terminal",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={"northcua": True, "batch_id": "batch-1", "task_id": "case-1", "attempt": 1},
    )
    missing = SandboxInventoryItem(
        sandbox_id="sandbox-missing",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={"northcua": True, "batch_id": "batch-1", "task_id": "missing-case", "attempt": 1},
    )
    sandboxes = FakeSandboxMaintenanceClient([terminal, missing])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.sweep_provider_orphans(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True)
    )

    assert set(sandboxes.terminated) == {"sandbox-terminal", "sandbox-missing"}
    assert result.applied_count == 2
    assert result.plan.metadata["candidates_by_reason"] == {"terminal_case": 1, "missing_sqlite_row": 1}


def test_sweep_provider_orphans_terminates_terminal_current_sandbox_residue(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUCCEEDED,
            attempts=1,
            sandbox_id="sandbox-current",
        )
    )
    terminal_current = SandboxInventoryItem(
        sandbox_id="sandbox-current",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={"northcua": True, "batch_id": "batch-1", "task_id": "case-1", "attempt": 1},
    )
    sandboxes = FakeSandboxMaintenanceClient([terminal_current])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.sweep_provider_orphans(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True)
    )

    assert sandboxes.terminated == ["sandbox-current"]
    assert result.applied_count == 1
    assert result.plan.metadata["candidates_by_reason"] == {"terminal_case": 1}


def test_sweep_provider_orphans_protects_active_prepared_buffer(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.sweep_provider_orphans(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True)
    )

    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.plan.metadata["protected_by_reason"] == {"protected_prepared": 1}


def test_sweep_provider_orphans_protects_retry_backoff_prepared_buffer(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=4,
            retry_budget_remaining=2,
            next_attempt_after=time.time() + 300,
        )
    )
    store = SQLitePreparedSandboxStore(db_path)
    record = store.reserve(
        task_id="case-1",
        attempt=5,
        runner_id="runner-a",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="desktop",
        ttl_seconds=3600,
    )
    assert record is not None
    preparing = store.mark_preparing(record)
    assert preparing is not None
    prepared = store.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 5,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.sweep_provider_orphans(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True)
    )

    protected = store.get(task_id="case-1", attempt=5)
    assert protected is not None
    assert protected.status == PreparedSandboxStatus.PREPARED
    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.plan.metadata["candidate_count"] == 0
    assert result.plan.metadata["protected_by_reason"] == {"protected_prepared": 1}


def test_sweep_provider_orphans_terminates_duplicate_prepared_sandbox_mismatch(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
        sandbox_id="sandbox-current",
        device_id="device",
        lease_id="lease",
        prepare_seconds=1.0,
    )
    assert prepared is not None
    current = SandboxInventoryItem(
        sandbox_id="sandbox-current",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    duplicate = current.model_copy(update={"sandbox_id": "sandbox-duplicate"})
    sandboxes = FakeSandboxMaintenanceClient([current, duplicate])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.sweep_provider_orphans(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True)
    )

    assert sandboxes.terminated == ["sandbox-duplicate"]
    assert result.applied_count == 1
    assert result.plan.metadata["candidates_by_reason"] == {"prepared_sandbox_mismatch": 1}
    assert result.plan.metadata["protected_by_reason"] == {"protected_prepared": 1}


def test_sweep_provider_orphans_prepared_mismatch_does_not_terminate_current_task_sandbox(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
    assert (
        store.mark_prepared(
            preparing,
            sandbox_id="sandbox-other",
            device_id="device",
            lease_id="lease",
            prepare_seconds=1.0,
        )
        is not None
    )
    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )
    assert claim is not None
    assert repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.RUNNING,
            attempts=1,
            sandbox_id="sandbox-current",
        ),
        lease_token=claim.lease_token,
    )
    provider_current = SandboxInventoryItem(
        sandbox_id="sandbox-current",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([provider_current])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.sweep_provider_orphans(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True)
    )

    assert sandboxes.terminated == []
    assert result.applied_count == 0
    assert result.plan.metadata["protected_by_reason"] == {"protected_current_sandbox_id": 1}


def test_sweep_provider_orphans_resolves_cleanup_failed_prepared(tmp_path: Path) -> None:
    _init_batch(tmp_path)
    db_path = tmp_path / "batch-1" / "batch_state.sqlite"
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
    prepared_item = SandboxInventoryItem(
        sandbox_id="sandbox-prepared",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={
            "northcua": True,
            "batch_id": "batch-1",
            "task_id": "case-1",
            "attempt": 1,
            "prepared_sandbox": "true",
        },
    )
    sandboxes = FakeSandboxMaintenanceClient([prepared_item])
    operations = BatchMaintenanceOperations(controller=FakeControllerMaintenanceClient(), sandboxes=sandboxes)

    result = _run(
        operations.sweep_provider_orphans(batch_id="batch-1", output_dir=tmp_path, provider="fake", apply=True)
    )

    assert sandboxes.terminated == ["sandbox-prepared"]
    assert result.applied_count == 1
    resolved = store.get(task_id="case-1", attempt=1)
    assert resolved is not None
    assert resolved.status == PreparedSandboxStatus.RELEASED
    assert result.plan.metadata["candidates_by_reason"] == {"prepared_orphan": 1}


def test_pre_resume_cleanup_sweeps_provider_orphan_and_verifies_clean(tmp_path: Path) -> None:
    repository = _init_batch(tmp_path)
    assert repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED, attempts=1))
    controller_db = tmp_path / "controller.sqlite"
    SQLiteControllerStore(controller_db).close()
    orphan = SandboxInventoryItem(
        sandbox_id="sandbox-terminal",
        provider="fake",
        template="desktop",
        state=SandboxInventoryState.RUNNING,
        metadata={"northcua": True, "batch_id": "batch-1", "task_id": "case-1", "attempt": 1},
    )
    sandboxes = FakeSandboxMaintenanceClient([orphan])

    _run(
        _run_pre_resume_cleanup(
            batch_id="batch-1",
            output_dir=tmp_path,
            controller_db=controller_db,
            provider="fake",
            maintenance=BatchMaintenanceOperations(sandboxes=sandboxes),
        )
    )

    summary = json.loads((tmp_path / "batch-1" / "pre_resume_cleanup_summary.json").read_text())
    assert summary["passed"] is True
    assert summary["provider_orphan_sweep"]["metadata"]["candidate_count"] == 1
    assert summary["provider_orphan_sweep_verify"]["metadata"]["candidate_count"] == 0
    assert sandboxes.terminated == ["sandbox-terminal"]


def test_interrupted_snapshot_does_not_inherit_stale_transport_category() -> None:
    existing = BatchTaskState(
        task_id="case-1",
        status=BatchTaskStatus.RUNNING,
        last_failure_category="device_transport",
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        agent_kind="desktop",
        state=TaskState.INTERRUPTED,
    )

    reconciled = ControllerMaintenanceClient.state_from_snapshot(existing, snapshot)

    assert reconciled.status == BatchTaskStatus.CANCELED


def test_residue_cleanup_interrupted_snapshot_is_retryable_failure() -> None:
    existing = BatchTaskState(
        task_id="case-1",
        status=BatchTaskStatus.SUBMITTED,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        agent_kind="desktop",
        state=TaskState.INTERRUPTED,
        error="controller residue cleanup interrupted stale task",
    )

    reconciled = ControllerMaintenanceClient.state_from_snapshot(existing, snapshot)

    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.last_failure_category == FailureCategory.CONTROLLER_SUBMIT
    assert reconciled.recovery_action is None


def test_interrupted_timeout_snapshot_maps_to_timed_out() -> None:
    existing = BatchTaskState(
        task_id="case-1",
        status=BatchTaskStatus.RUNNING,
        last_failure_category=FailureCategory.TIMEOUT,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        agent_kind="desktop",
        state=TaskState.INTERRUPTED,
    )

    reconciled = ControllerMaintenanceClient.state_from_snapshot(existing, snapshot)

    assert reconciled.status == BatchTaskStatus.TIMED_OUT
    assert reconciled.last_failure_category == FailureCategory.TIMEOUT


def test_failed_snapshot_with_evaluator_accepted_infeasible_maps_to_succeeded() -> None:
    existing = BatchTaskState(
        task_id="case-1",
        status=BatchTaskStatus.RUNNING,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        agent_kind="desktop",
        state=TaskState.FAILED,
        error="missing physical device",
        result={
            "result": {
                "evaluation_report": {
                    "overall_verdict": "success",
                    "outcome_label": "failure",
                    "controllability": "uncontrollable",
                }
            }
        },
    )

    reconciled = ControllerMaintenanceClient.state_from_snapshot(existing, snapshot)

    assert reconciled.status == BatchTaskStatus.SUCCEEDED
    assert reconciled.terminal_state == TaskState.FAILED
    assert reconciled.error == ""
    assert reconciled.last_failure_category == FailureCategory.ACCEPTED_INFEASIBLE


def test_finished_snapshot_with_evaluator_pending_maps_to_evaluation_pending() -> None:
    existing = BatchTaskState(
        task_id="case-1",
        status=BatchTaskStatus.RUNNING,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        agent_kind="desktop",
        state=TaskState.FINISHED,
        result={"agent_terminal": {"status": "completed"}},
        terminal_decision={
            "action": "finish",
            "success": False,
            "metadata": {
                "evaluation": {
                    "status": "pending",
                    "attempts": 2,
                    "projection_level": "tight",
                    "error_type": "context_budget",
                }
            },
        },
    )

    reconciled = ControllerMaintenanceClient.state_from_snapshot(existing, snapshot)

    assert reconciled.status == BatchTaskStatus.EVALUATION_PENDING
    assert reconciled.terminal_state == TaskState.FINISHED
    assert reconciled.error == ""
    assert reconciled.last_failure_category == FailureCategory.EVALUATOR
    assert reconciled.evaluation_status == "pending"
    assert reconciled.evaluation_attempts == 2
    assert reconciled.evaluation_projection_level == "tight"
    assert reconciled.evaluation_error["error_type"] == "context_budget"
    assert reconciled.agent_terminal["state"] == TaskState.FINISHED.value


def test_finished_snapshot_with_non_success_evaluator_report_maps_to_failed() -> None:
    existing = BatchTaskState(
        task_id="case-1",
        status=BatchTaskStatus.RUNNING,
    )
    snapshot = TaskStatusSnapshot(
        task_id="controller-1",
        trace_id="trace-1",
        agent_kind="desktop",
        state=TaskState.FINISHED,
        result={"evaluation_report": {"overall_verdict": "failure", "outcome_label": "failure"}},
    )

    reconciled = ControllerMaintenanceClient.state_from_snapshot(existing, snapshot)

    assert reconciled.status == BatchTaskStatus.FAILED
    assert reconciled.terminal_state == TaskState.FINISHED


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    import asyncio

    return asyncio.run(coro)
