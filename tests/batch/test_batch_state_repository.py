"""SQLite batch state repository concurrency tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import pytest

from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.preparation.contracts import PreparedSandboxDepth
from northcua.batch.preparation.sqlite_store import SQLitePreparedSandboxStore
from northcua.batch.retry_policy import task_state_blocks_retry
from northcua.batch.selected_retry import SelectedRetryPlan, SelectedRetryTarget
from northcua.batch.state import LeasePolicy, SQLiteBatchStateRepository
from northcua.batch.state import sqlite as sqlite_module
from northcua.reliability import FailureAction
from northcua.telemetry.contracts import FailureCategory


def _manifest() -> BatchManifest:
    return BatchManifest(
        batch_id="local-claims",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="Open page")],
    )


def _manifest_with_task_ids(*task_ids: str) -> BatchManifest:
    return BatchManifest(
        batch_id="local-claims",
        tasks=[BatchTaskSpec(task_id=task_id, agent_kind="web", query=f"Open page {task_id}") for task_id in task_ids],
    )


def test_sqlite_claim_is_exclusive_across_connections(tmp_path: Path) -> None:
    """Only one local runner can own a task before the lease expires."""

    path = tmp_path / "batch_state.sqlite"
    first = SQLiteBatchStateRepository(path)
    second = SQLiteBatchStateRepository(path)
    first.initialize_manifest(_manifest())
    policy = LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20)

    claim = first.claim_next(lease_owner="runner-a", policy=policy)

    assert claim is not None
    assert second.claim_next(lease_owner="runner-b", policy=policy) is None


def test_claim_next_scans_past_unclaimable_rows_across_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunked claim scanning should preserve manifest order across skipped rows."""

    monkeypatch.setattr(sqlite_module, "_CLAIM_SCAN_LIMIT", 2)
    task_ids = tuple(f"task-{index}" for index in range(5))
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest_with_task_ids(*task_ids))
    for task_id in task_ids[:4]:
        assert repository.upsert_state(BatchTaskState(task_id=task_id, status=BatchTaskStatus.FAILED, attempts=1))

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )

    assert claim is not None
    assert claim.task_id == "task-4"


def test_claim_next_prefers_unattempted_pending_before_retry_rows(tmp_path: Path) -> None:
    """Failed early rows must not starve later never-attempted tasks."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest_with_task_ids("task-1", "task-2", "task-3"))
    assert repository.upsert_state(
        BatchTaskState(
            task_id="task-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            retry_budget_remaining=5,
        )
    )
    assert repository.upsert_state(
        BatchTaskState(
            task_id="task-2",
            status=BatchTaskStatus.PENDING,
            attempts=0,
            retry_budget_remaining=5,
        )
    )

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 6, "task-2": 6, "task-3": 6},
    )

    assert claim is not None
    assert claim.task_id == "task-2"


def test_claim_next_prefers_unattempted_pending_before_local_recovery_rows(tmp_path: Path) -> None:
    """Local pre-controller recovery rows must not starve fresh pending work."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest_with_task_ids("task-1", "task-2"))
    assert repository.upsert_state(
        BatchTaskState(
            task_id="task-1",
            status=BatchTaskStatus.WAITING_DEVICE,
            attempts=1,
            retry_budget_remaining=5,
        )
    )
    assert repository.upsert_state(
        BatchTaskState(
            task_id="task-2",
            status=BatchTaskStatus.PENDING,
            attempts=0,
            retry_budget_remaining=5,
        )
    )

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 6, "task-2": 6},
    )

    assert claim is not None
    assert claim.task_id == "task-2"


def test_claim_next_scans_past_blocked_fresh_chunks_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sqlite_module, "_CLAIM_SCAN_LIMIT", 2)
    db_path = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(db_path)
    repository.initialize_manifest(_manifest_with_task_ids("retry", "blocked-1", "blocked-2", "fresh"))
    assert repository.upsert_state(
        BatchTaskState(task_id="retry", status=BatchTaskStatus.FAILED, attempts=1, retry_budget_remaining=5)
    )
    prepared = SQLitePreparedSandboxStore(db_path)
    for task_id in ("blocked-1", "blocked-2"):
        assert (
            prepared.reserve(
                task_id=task_id,
                attempt=1,
                runner_id="other-runner",
                depth=PreparedSandboxDepth.FULL_INIT,
                provider="fake",
                template="desktop",
                ttl_seconds=60,
            )
            is not None
        )

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={task_id: 6 for task_id in ("retry", "blocked-1", "blocked-2", "fresh")},
        respect_prepared_reservations=True,
        prepared_reservation_stale_after_seconds=60,
    )

    assert claim is not None
    assert claim.task_id == "fresh"


def test_evaluation_pending_is_terminal_and_not_claimable(tmp_path: Path) -> None:
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    assert repository.upsert_state(
        BatchTaskState(
            task_id="task-1",
            status=BatchTaskStatus.EVALUATION_PENDING,
            attempts=1,
            retry_budget_remaining=5,
            evaluation_status="pending",
        )
    )

    assert (
        repository.claim_next(
            lease_owner="runner-a",
            policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
            max_attempts_by_task={"task-1": 6},
        )
        is None
    )


def test_evaluation_pending_blocks_whole_case_retry_policy() -> None:
    assert task_state_blocks_retry(
        BatchTaskState(
            task_id="task-1",
            status=BatchTaskStatus.EVALUATION_PENDING,
            attempts=1,
            retry_budget_remaining=5,
            evaluation_status="pending",
        )
    )


def test_selected_retry_reset_is_exact_attempt_preserving_and_idempotent(tmp_path: Path) -> None:
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest_with_task_ids("failed", "canceled", "sandbox", "success"))
    assert repository.upsert_state(
        BatchTaskState(
            task_id="failed",
            status=BatchTaskStatus.FAILED,
            attempts=4,
            retry_budget_remaining=0,
            last_failure_category=FailureCategory.MAIN_AGENT,
            controller_task_id="old-controller",
            terminal_decision={"action": "error", "retryable": False},
            evaluation_status="failed",
        )
    )
    assert repository.upsert_state(
        BatchTaskState(
            task_id="canceled",
            status=BatchTaskStatus.CANCELED,
            attempts=1,
            retry_budget_remaining=3,
            error="operator stop",
        )
    )
    assert repository.upsert_state(
        BatchTaskState(
            task_id="sandbox",
            status=BatchTaskStatus.FAILED,
            attempts=4,
            retry_budget_remaining=0,
            last_failure_category=FailureCategory.SANDBOX_CREATE,
        )
    )
    assert repository.upsert_state(BatchTaskState(task_id="success", status=BatchTaskStatus.SUCCEEDED, attempts=1))
    repository._connection.execute(  # noqa: SLF001 - simulate a prior task-level cancel projection.
        "UPDATE batch_tasks SET cancel_requested=1 WHERE task_id='canceled'"
    )
    repository.set_control_flag("drain_requested", True)
    repository.set_control_flag("cancel_requested", False)
    plan = SelectedRetryPlan(
        path=tmp_path / "plan.json",
        sha256="plan-sha",
        plan_id="selected-v1",
        batch_id="local-claims",
        manifest_sha256="manifest-sha",
        extra_attempts=2,
        targets=(
            SelectedRetryTarget(
                task_id="failed",
                expected_status=BatchTaskStatus.FAILED,
                expected_attempts=4,
                expected_retry_budget_remaining=0,
                expected_failure_category=FailureCategory.MAIN_AGENT.value,
                max_attempts=6,
            ),
            SelectedRetryTarget(
                task_id="canceled",
                expected_status=BatchTaskStatus.CANCELED,
                expected_attempts=1,
                expected_retry_budget_remaining=3,
                expected_failure_category="",
                max_attempts=3,
            ),
        ),
    )

    first = repository.reset_selected_for_retry(plan, actor="test")
    repository.set_control_flag("drain_requested", False)
    second = repository.reset_selected_for_retry(plan, actor="test")
    rows = {
        row["task_id"]: row
        for row in repository._connection.execute(  # noqa: SLF001 - inspect persisted retry limits.
            "SELECT * FROM batch_tasks ORDER BY order_index"
        ).fetchall()
    }
    states = {state.task_id: state for state in repository.all()}

    assert first == {"selected": 2, "source_failed": 1, "source_canceled": 1, "already_applied": False}
    assert second == {"selected": 2, "source_failed": 0, "source_canceled": 0, "already_applied": True}
    assert states["failed"].status == BatchTaskStatus.PENDING
    assert states["failed"].attempts == 4
    assert states["failed"].retry_budget_remaining == 2
    assert states["failed"].terminal_decision == {}
    assert states["failed"].evaluation_status == ""
    assert (rows["failed"]["max_retries"], rows["failed"]["max_attempts"]) == (5, 6)
    assert states["canceled"].status == BatchTaskStatus.PENDING
    assert states["canceled"].attempts == 1
    assert states["canceled"].retry_budget_remaining == 2
    assert rows["canceled"]["cancel_requested"] == 0
    assert (rows["canceled"]["max_retries"], rows["canceled"]["max_attempts"]) == (2, 3)
    assert states["sandbox"].status == BatchTaskStatus.FAILED
    assert states["success"].status == BatchTaskStatus.SUCCEEDED


def test_selected_retry_reset_rolls_back_on_error_hash_drift(tmp_path: Path) -> None:
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest_with_task_ids("first", "drifted"))
    for task_id in ("first", "drifted"):
        assert repository.upsert_state(
            BatchTaskState(
                task_id=task_id,
                status=BatchTaskStatus.FAILED,
                attempts=4,
                retry_budget_remaining=0,
                last_failure_category=FailureCategory.MAIN_AGENT,
                error=f"provider error for {task_id}",
            )
        )
    repository.set_control_flag("drain_requested", True)
    repository.set_control_flag("cancel_requested", False)
    plan = SelectedRetryPlan(
        path=tmp_path / "plan.json",
        sha256="plan-sha",
        plan_id="drift-v1",
        batch_id="local-claims",
        manifest_sha256="manifest-sha",
        extra_attempts=2,
        targets=tuple(
            SelectedRetryTarget(
                task_id=task_id,
                expected_status=BatchTaskStatus.FAILED,
                expected_attempts=4,
                expected_retry_budget_remaining=0,
                expected_failure_category=FailureCategory.MAIN_AGENT.value,
                max_attempts=6,
                expected_error_sha256=(
                    hashlib.sha256(f"provider error for {task_id}".encode()).hexdigest()
                    if task_id == "first"
                    else hashlib.sha256(b"different provider error").hexdigest()
                ),
            )
            for task_id in ("first", "drifted")
        ),
    )

    with pytest.raises(RuntimeError, match="precondition drift"):
        repository.reset_selected_for_retry(plan)

    assert {state.task_id: state.status for state in repository.all()} == {
        "first": BatchTaskStatus.FAILED,
        "drifted": BatchTaskStatus.FAILED,
    }


def test_repository_all_sanitizes_legacy_null_contract_fields(tmp_path: Path) -> None:
    """Old run DBs may contain null in fields that are now typed containers."""

    path = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(path)
    repository.initialize_manifest(_manifest())
    legacy_payload = {
        "task_id": "task-1",
        "status": "failed",
        "attempts": 1,
        "terminal_decision": None,
        "post_run_results": None,
        "reconcile_unknown_count": None,
    }
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE batch_tasks SET state_json=?, status=?, attempts=? WHERE task_id=?",
            (json.dumps(legacy_payload), "failed", 1, "task-1"),
        )

    state = repository.all()[0]

    assert state.terminal_decision == {}
    assert state.post_run_results == []
    assert state.reconcile_unknown_count == 0


def test_claim_next_skips_exhausted_pending_rows_across_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pending rows with no remaining attempt budget must not occupy workers."""

    monkeypatch.setattr(sqlite_module, "_CLAIM_SCAN_LIMIT", 2)
    task_ids = tuple(f"task-{index}" for index in range(5))
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest_with_task_ids(*task_ids))
    for task_id in task_ids[:4]:
        assert repository.upsert_state(
            BatchTaskState(
                task_id=task_id,
                status=BatchTaskStatus.PENDING,
                attempts=3,
                retry_budget_remaining=0,
            )
        )

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )

    assert claim is not None
    assert claim.task_id == "task-4"


def test_legacy_unknown_recovery_action_does_not_block_state_reads_or_claims(tmp_path: Path) -> None:
    """Historical transient recovery labels should not break state decoding."""

    path = tmp_path / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(path)
    repository.initialize_manifest(_manifest())
    legacy_state = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        retry_budget_remaining=1,
    ).model_dump(mode="json")
    legacy_state["recovery_action"] = "terminal_after_pre_resume_clean_budget_exhausted"
    with sqlite3.connect(path) as connection:
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
                "task-1",
            ),
        )

    stored = repository.get("task-1")
    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 3},
    )

    assert stored is not None
    assert stored.recovery_action is None
    assert claim is not None
    assert claim.task_id == "task-1"
    assert claim.state.recovery_action is None


def test_claim_next_ignores_only_exhausted_pending_rows(tmp_path: Path) -> None:
    """A pending projection with no remaining attempts is not runnable work."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    assert repository.upsert_state(
        BatchTaskState(
            task_id="task-1",
            status=BatchTaskStatus.PENDING,
            attempts=3,
            retry_budget_remaining=0,
        )
    )

    assert (
        repository.claim_next(
            lease_owner="runner-a",
            policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        )
        is None
    )


def test_next_claim_after_returns_earliest_delayed_row_across_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunked retry scans should return the global earliest delayed retry."""

    monkeypatch.setattr(sqlite_module, "_CLAIM_SCAN_LIMIT", 2)
    task_ids = tuple(f"task-{index}" for index in range(5))
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest_with_task_ids(*task_ids))
    now = time.time()
    retry_times = {
        "task-0": now + 100,
        "task-1": now + 200,
        "task-2": now + 10,
        "task-3": now + 300,
        "task-4": now + 400,
    }
    for task_id, next_attempt_after in retry_times.items():
        assert repository.upsert_state(
            BatchTaskState(
                task_id=task_id,
                status=BatchTaskStatus.FAILED,
                attempts=0,
                next_attempt_after=next_attempt_after,
            )
        )

    assert (
        repository.next_claim_after(
            lease_owner="runner-a",
            max_attempts_by_task={task_id: 1 for task_id in task_ids},
        )
        == retry_times["task-2"]
    )


def test_prepared_task_lookup_uses_task_id_index(tmp_path: Path) -> None:
    """Prepared-rank lookup should not use the broad claim-order index."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest_with_task_ids("task-1", "task-2"))
    statuses = sorted(sqlite_module._CLAIMABLE_STATUSES)
    plan = repository._connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT task_id
        FROM batch_tasks
        WHERE task_id=?
          AND cancel_requested=0
          AND status IN ({status_placeholders})
          AND (
            lease_token=''
            OR lease_expires_at <= ?
          )
        """.format(status_placeholders=",".join("?" for _ in statuses)),
        ("task-1", *statuses, time.time()),
    ).fetchall()

    plan_text = " | ".join(str(row[3]) for row in plan)
    assert "task_id=?" in plan_text
    assert "idx_batch_tasks_claim_order" not in plan_text


def test_claim_scan_uses_order_keyset_index(tmp_path: Path) -> None:
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest_with_task_ids("task-1", "task-2"))
    statuses = sorted(sqlite_module._CLAIMABLE_STATUSES)
    plan = repository._connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT task_id
        FROM batch_tasks INDEXED BY idx_batch_tasks_claim_order
        WHERE cancel_requested=0
          AND order_index > ?
          AND status IN ({status_placeholders})
          AND (
            lease_token=''
            OR lease_expires_at <= ?
          )
        ORDER BY order_index ASC
        LIMIT ?
        """.format(status_placeholders=",".join("?" for _ in statuses)),
        (-1, *statuses, time.time(), 2048),
    ).fetchall()

    plan_text = " | ".join(str(row[3]) for row in plan)
    assert "idx_batch_tasks_claim_order" in plan_text
    assert "USE TEMP B-TREE" not in plan_text


def test_expired_lease_can_be_reclaimed_and_rejects_stale_owner(tmp_path: Path) -> None:
    """A crashed runner stops owning the task once its lease expires."""

    path = tmp_path / "batch_state.sqlite"
    first = SQLiteBatchStateRepository(path)
    second = SQLiteBatchStateRepository(path)
    first.initialize_manifest(_manifest())
    policy = LeasePolicy(lease_timeout_seconds=0.01, heartbeat_interval_seconds=0.01)
    old_claim = first.claim_next(lease_owner="runner-a", policy=policy)
    assert old_claim is not None
    time.sleep(0.03)

    new_claim = second.claim_next(lease_owner="runner-b", policy=policy)
    assert new_claim is not None
    assert new_claim.generation == old_claim.generation + 1

    stale_state = BatchTaskState(task_id="task-1", status=BatchTaskStatus.SUCCEEDED)
    current_state = BatchTaskState(task_id="task-1", status=BatchTaskStatus.FAILED, error="owned by runner-b")
    assert not first.upsert_state(stale_state, lease_token=old_claim.lease_token)
    assert second.upsert_state(current_state, lease_token=new_claim.lease_token)
    assert second.get("task-1") == current_state


def test_legacy_jsonl_state_initializes_sqlite_repository(tmp_path: Path) -> None:
    """Pre-SQLite batch outputs are migrated when a run is resumed."""

    legacy = tmp_path / "batch_state.jsonl"
    legacy_state = BatchTaskState(task_id="task-1", status=BatchTaskStatus.SUCCEEDED, attempts=1)
    legacy.write_text(legacy_state.model_dump_json() + "\n", encoding="utf-8")
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")

    repository.initialize_manifest(_manifest(), legacy_jsonl_path=str(legacy))

    assert repository.all() == [legacy_state]
    assert (
        repository.claim_next(
            lease_owner="runner-a",
            policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        )
        is None
    )


def test_cancel_marks_unclaimed_tasks_terminal(tmp_path: Path) -> None:
    """Cancel blocks future claims for pending tasks."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())

    repository.mark_cancel_requested()

    state = repository.get("task-1")
    assert state is not None
    assert state.status == BatchTaskStatus.CANCELED
    assert (
        repository.claim_next(
            lease_owner="runner-a",
            policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        )
        is None
    )


def test_cancel_marks_leased_task_terminal_and_stops_heartbeat(tmp_path: Path) -> None:
    """Cancel should interrupt in-flight leases, not just pending rows."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    policy = LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20)
    claim = repository.claim_next(lease_owner="runner-a", policy=policy)
    assert claim is not None

    repository.mark_cancel_requested()

    state = repository.get("task-1")
    assert state is not None
    assert state.status == BatchTaskStatus.CANCELED
    assert not repository.heartbeat(claim, policy=policy)


def test_cancel_preserves_existing_reconcile_metadata(tmp_path: Path) -> None:
    """Cancel should not erase controller ids needed by later audit/reconcile."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    rich_state = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        controller_task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        last_failure_category=FailureCategory.MAIN_AGENT,
        cleanup_status="failed: cleanup timeout",
        error="agent failed",
    )
    assert repository.upsert_state(rich_state)

    repository.mark_cancel_requested()

    canceled = repository.get("task-1")
    assert canceled is not None
    assert canceled.status == BatchTaskStatus.CANCELED
    assert canceled.controller_task_id == "controller-1"
    assert canceled.trace_id == "trace-1"
    assert canceled.session_id == "session-1"
    assert canceled.last_failure_category == FailureCategory.MAIN_AGENT
    assert canceled.cleanup_status == "failed: cleanup timeout"


def test_pause_and_drain_flags_block_new_claims(tmp_path: Path) -> None:
    """Pause/drain are batch-level controls that leave task rows untouched."""

    policy = LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20)
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())

    repository.set_control_flag("paused", True)

    assert repository.control_flags()["paused"] is True
    assert repository.claim_next(lease_owner="runner-a", policy=policy) is None
    state = repository.get("task-1")
    assert state is not None
    assert state.status == BatchTaskStatus.PENDING

    repository.set_control_flag("paused", False)
    repository.set_control_flag("drain_requested", True)

    assert repository.claim_next(lease_owner="runner-a", policy=policy) is None

    repository.set_control_flag("drain_requested", False)

    assert repository.claim_next(lease_owner="runner-a", policy=policy) is not None


def test_control_flags_record_actor_reason_and_events(tmp_path: Path) -> None:
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())

    repository.set_control_flag("paused", True, actor="operator", reason="command backlog investigation")
    repository.set_control_flag("paused", False, actor="batch resume", reason="resume cleared pause")

    current = repository._connection.execute(
        "SELECT value, updated_by, reason FROM batch_control WHERE name='paused'"
    ).fetchone()
    events = repository._connection.execute(
        "SELECT name, value, updated_by, reason FROM batch_control_events ORDER BY event_id"
    ).fetchall()

    assert current is not None
    assert dict(current) == {
        "value": "0",
        "updated_by": "batch resume",
        "reason": "resume cleared pause",
    }
    assert [dict(row) for row in events] == [
        {
            "name": "paused",
            "value": "1",
            "updated_by": "operator",
            "reason": "command backlog investigation",
        },
        {
            "name": "paused",
            "value": "0",
            "updated_by": "batch resume",
            "reason": "resume cleared pause",
        },
    ]


def test_exhausted_failed_task_is_not_claimed_when_budget_is_known(tmp_path: Path) -> None:
    """Retry-exhausted failed rows remain reportable but stop scheduling."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )
    assert claim is not None
    failed = BatchTaskState(task_id="task-1", status=BatchTaskStatus.FAILED, attempts=1, error="failed once")
    assert repository.release_claim(claim, terminal_state=failed)

    assert (
        repository.claim_next(
            lease_owner="runner-b",
            policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
            max_attempts_by_task={"task-1": 1},
        )
        is None
    )


def test_exhausted_in_flight_task_is_claimed_for_reconciliation_when_budget_is_known(tmp_path: Path) -> None:
    """Submitted/running rows need reconciliation even after attempt budget is spent."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    submitted = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
        controller_task_id="controller-1",
    )
    assert repository.upsert_state(submitted)

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 1},
    )
    assert claim is not None
    assert claim.task_id == "task-1"


def test_exhausted_local_setup_task_is_claimed_for_recovery_when_budget_is_known(tmp_path: Path) -> None:
    """Pre-controller rows are recoverable even after their attempt budget is spent."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    starting = BatchTaskState(task_id="task-1", status=BatchTaskStatus.STARTING_SANDBOX, attempts=1)
    assert repository.upsert_state(starting)

    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 1},
    )

    assert claim is not None
    assert claim.task_id == "task-1"


def test_failed_task_can_be_retried_while_budget_remains(tmp_path: Path) -> None:
    """Failed rows stay claimable until the total-attempt budget is spent."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )
    assert claim is not None
    failed = BatchTaskState(task_id="task-1", status=BatchTaskStatus.FAILED, attempts=1, error="failed once")
    assert repository.release_claim(claim, terminal_state=failed)

    retry_claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 2},
    )

    assert retry_claim is not None
    assert retry_claim.task_id == "task-1"


def test_retry_backoff_delays_failed_task_claim(tmp_path: Path) -> None:
    """Retryable failures can be cooled down before the next claim."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    next_attempt_after = time.time() + 60
    failed = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        error="opening handshake timed out",
        last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
        next_attempt_after=next_attempt_after,
    )
    assert repository.upsert_state(failed)

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 2},
    )

    assert claim is None
    assert repository.next_claim_after(lease_owner="runner-b", max_attempts_by_task={"task-1": 2}) == next_attempt_after


def test_retry_backoff_delays_submitted_reconciliation_claim(tmp_path: Path) -> None:
    """Unknown controller-status reconciliation rows can sleep without duplicate submit."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    next_attempt_after = time.time() + 60
    submitted = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
        controller_task_id="controller-1",
        last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
        next_attempt_after=next_attempt_after,
    )
    assert repository.upsert_state(submitted)

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 2},
    )

    assert claim is None
    assert repository.next_claim_after(lease_owner="runner-b", max_attempts_by_task={"task-1": 2}) == next_attempt_after


def test_non_retryable_terminal_decision_blocks_retry_claim(tmp_path: Path) -> None:
    """Terminal decisions with retryable=false stop deterministic reruns."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )
    assert claim is not None
    failed = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        error="round step budget exhausted",
        terminal_decision={
            "action": "fail",
            "reason": "max_steps",
            "success": False,
            "retryable": False,
            "metadata": {"scope": "round", "limit": 2, "current": 2},
        },
    )
    assert repository.release_claim(claim, terminal_state=failed)

    assert (
        repository.claim_next(
            lease_owner="runner-b",
            policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
            max_attempts_by_task={"task-1": 3},
        )
        is None
    )


def test_timed_out_non_retryable_terminal_decision_can_retry_while_budget_remains(tmp_path: Path) -> None:
    """Timeout classification owns retry behavior, not cancel terminal metadata."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    timed_out = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.TIMED_OUT,
        attempts=1,
        error="controller wait timed out",
        terminal_decision={
            "action": "error",
            "reason": "interrupted",
            "success": False,
            "retryable": False,
        },
    )
    assert repository.upsert_state(timed_out)

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 2},
    )

    assert claim is not None
    assert claim.task_id == "task-1"


def test_device_transport_retryable_false_terminal_decision_can_retry(tmp_path: Path) -> None:
    """Device transport failures remain infra-retryable even with stale terminal metadata."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    failed = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        error="websocket disconnected",
        recovery_action="fail_fast",
        last_failure_category=FailureCategory.DEVICE_TRANSPORT,
        terminal_decision={
            "action": "error",
            "reason": "interrupted",
            "success": False,
            "retryable": False,
        },
    )
    assert repository.upsert_state(failed)

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 2},
    )

    assert claim is not None
    assert claim.task_id == "task-1"


def test_controller_submit_retryable_false_terminal_decision_can_retry(tmp_path: Path) -> None:
    """Controller-submit failures stay retryable under the infra retry policy."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    failed = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        error="timed out during opening handshake",
        last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
        terminal_decision={
            "action": "error",
            "reason": "interrupted",
            "success": False,
            "retryable": False,
        },
    )
    assert repository.upsert_state(failed)

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 2},
    )

    assert claim is not None
    assert claim.task_id == "task-1"


def test_unknown_controller_status_without_retry_budget_is_not_reclaimed(tmp_path: Path) -> None:
    """Failed unknown controller rows do not bypass retry budget under pressure."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    failed = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        controller_task_id="controller-1",
        recovery_action=FailureAction.FAIL_FAST,
        last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
        last_reconciled_status="unknown",
        next_attempt_after=time.time() - 1,
    )
    assert repository.upsert_state(failed)

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 1},
    )

    assert claim is None


def test_unknown_controller_status_without_budget_is_not_reclaimed_without_fail_fast(tmp_path: Path) -> None:
    """Budget exhaustion alone blocks failed unknown controller rows."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    failed = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        controller_task_id="controller-1",
        last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
        last_reconciled_status="unknown",
        next_attempt_after=time.time() - 1,
    )
    assert repository.upsert_state(failed)

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 1},
    )

    assert claim is None


def test_active_controller_reconcile_ignores_stale_fail_fast_metadata(tmp_path: Path) -> None:
    """Active controller rows remain claimable for reconcile regardless of retry budget."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    submitted = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
        controller_task_id="controller-1",
        recovery_action=FailureAction.FAIL_FAST,
        last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
        terminal_decision={
            "action": "error",
            "reason": "stale metadata",
            "success": False,
            "retryable": False,
        },
    )
    assert repository.upsert_state(submitted)

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 1},
    )

    assert claim is not None
    assert claim.task_id == "task-1"


def test_delayed_active_controller_reconcile_reports_next_claim_after_with_stale_fail_fast(
    tmp_path: Path,
) -> None:
    """Delayed active controller rows still wake the scheduler after stale fail-fast metadata."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    next_attempt_after = time.time() + 60
    submitted = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.SUBMITTED,
        attempts=1,
        controller_task_id="controller-1",
        recovery_action=FailureAction.FAIL_FAST,
        last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
        next_attempt_after=next_attempt_after,
    )
    assert repository.upsert_state(submitted)

    assert (
        repository.claim_next(
            lease_owner="runner-b",
            policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
            max_attempts_by_task={"task-1": 1},
        )
        is None
    )
    assert repository.next_claim_after(lease_owner="runner-b", max_attempts_by_task={"task-1": 1}) == next_attempt_after


def test_readiness_timeout_retryable_false_terminal_decision_can_retry(tmp_path: Path) -> None:
    """Sandbox readiness failures can recover despite terminal retryable=false metadata."""

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    failed = BatchTaskState(
        task_id="task-1",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        error="sandbox device did not become ready: timed_out",
        last_failure_category=FailureCategory.READINESS_TIMEOUT,
        terminal_decision={
            "action": "error",
            "reason": "runtime_error",
            "success": False,
            "retryable": False,
        },
    )
    assert repository.upsert_state(failed)

    claim = repository.claim_next(
        lease_owner="runner-b",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
        max_attempts_by_task={"task-1": 2},
    )

    assert claim is not None
    assert claim.task_id == "task-1"
