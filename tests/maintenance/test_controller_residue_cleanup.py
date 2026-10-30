from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path

from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.planner import BatchTaskPlanner
from northcua.batch.preparation import PreparedSandboxDepth, PreparedSandboxStatus
from northcua.batch.preparation.sqlite_store import SQLitePreparedSandboxStore
from northcua.batch.state import SQLiteBatchStateRepository
from northcua.maintenance import cleanup_controller_sandboxes
from northcua.maintenance.cleanup_controller_residue import (
    _controller_lookup_task_ids,
    _controller_task_rows,
    cleanup_controller_residue,
)
from northcua.maintenance.cleanup_controller_sandboxes import is_sandbox_missing_error
from northcua.sandboxes.capacity_contracts import SandboxInventoryItem, SandboxInventoryState


class _MissingSandboxClient:
    async def inventory(self, *, provider_id: str, batch_id: str, case_id: str = "", include_warm_pool: bool = False):
        return [
            SandboxInventoryItem(
                sandbox_id="sandbox-old",
                provider="fake",
                template="tpl",
                state=SandboxInventoryState.RUNNING,
                metadata={"northcua": True, "batch_id": batch_id, "task_id": "case-1", "attempt": 1},
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
        raise RuntimeError(f"sandbox {item.sandbox_id} not found 404")


class _Generic404SandboxClient(_MissingSandboxClient):
    async def terminate(
        self,
        *,
        item: SandboxInventoryItem,
        batch_id: str,
        include_connect_existing: bool = False,
        include_warm_pool: bool = False,
    ) -> None:
        raise RuntimeError("sandbox API route returned 404")


class SandboxTemplateNotFoundError(RuntimeError):
    pass


class SandboxNotFoundError(RuntimeError):
    pass


class _RecordingSandboxClient:
    def __init__(self) -> None:
        self.terminated: list[str] = []

    async def inventory(self, *, provider_id: str, batch_id: str, case_id: str = "", include_warm_pool: bool = False):
        return [
            SandboxInventoryItem(
                sandbox_id="sandbox-cleanup-failed",
                provider="fake",
                template="tpl",
                state=SandboxInventoryState.RUNNING,
                metadata={"northcua": True, "batch_id": batch_id, "task_id": "case-1", "attempt": 2},
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
        self.terminated.append(item.sandbox_id)


class _PartialInventorySandboxClient(_RecordingSandboxClient):
    async def inventory(self, *, provider_id: str, batch_id: str, case_id: str = "", include_warm_pool: bool = False):
        return [
            SandboxInventoryItem(
                sandbox_id="sandbox-other",
                provider="fake",
                template="tpl",
                state=SandboxInventoryState.RUNNING,
                metadata={"northcua": True, "batch_id": batch_id, "task_id": "other", "attempt": 1},
            )
        ]


class _InventoryTimeoutSandboxClient(_RecordingSandboxClient):
    async def inventory(self, *, provider_id: str, batch_id: str, case_id: str = "", include_warm_pool: bool = False):
        raise TimeoutError("inventory timed out")


class _HangingTerminateSandboxClient(_RecordingSandboxClient):
    async def terminate(
        self,
        *,
        item: SandboxInventoryItem,
        batch_id: str,
        include_connect_existing: bool = False,
        include_warm_pool: bool = False,
    ) -> None:
        await asyncio.sleep(3600)


def test_cleanup_dry_run_blocks_current_active_controller_task(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.RUNNING, controller_task_id="controller-current")
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id="controller-current",
        case_id="case-1",
        session_id="session-current",
        lease_id="lease-current",
        device_id="device-current",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=False,
            skip_provider=True,
        )
    )

    assert result["planned"]["tasks"] == 0
    assert result["planned"]["blocked"] == 1
    assert "blocked.tsv" in result["outputs"]["blocked_tsv"]


def test_cleanup_can_interrupt_current_active_for_explicit_stop(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.RUNNING, controller_task_id="controller-current")
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id="controller-current",
        case_id="case-1",
        session_id="session-current",
        lease_id="lease-current",
        device_id="device-current",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            include_current_active=True,
            skip_provider=True,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        task_status = connection.execute("SELECT status FROM tasks WHERE task_id='controller-current'").fetchone()[0]
        device_status = connection.execute("SELECT status FROM devices WHERE device_id='device-current'").fetchone()[0]
    assert result["planned"]["tasks"] == 1
    assert result["cleaned"]["tasks"] == 1
    assert task_status == "interrupted"
    assert device_status == "offline"


def test_cleanup_apply_interrupts_old_controller_task_and_writes_audit_outputs(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.RUNNING, controller_task_id="controller-new")
    controller_db = _controller_db(tmp_path)
    old_controller_task_id, _trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=1,
    )
    _insert_controller_task(
        controller_db,
        task_id=old_controller_task_id,
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        sandbox_id="sandbox-old",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            skip_provider=True,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        task_status = connection.execute(
            "SELECT status FROM tasks WHERE task_id=?", (old_controller_task_id,)
        ).fetchone()[0]
        session_status = connection.execute("SELECT status FROM sessions WHERE session_id='session-old'").fetchone()[0]
        lease_status = connection.execute("SELECT status FROM leases WHERE lease_id='lease-old'").fetchone()[0]
        device_status = connection.execute("SELECT status FROM devices WHERE device_id='device-old'").fetchone()[0]
        event_type = connection.execute("SELECT event_type FROM events ORDER BY event_id DESC LIMIT 1").fetchone()[0]
    assert task_status == "interrupted"
    assert session_status == "interrupted"
    assert lease_status == "interrupted"
    assert device_status == "offline"
    assert event_type == "controller_residue_cleanup"
    assert result["cleaned"]["tasks"] == 1
    assert result["cleaned"]["sessions"] == 1
    assert result["cleaned"]["leases"] == 1
    assert result["cleaned"]["devices"] == 1
    assert result["residue_online_after"] == 0
    assert result["backup"]
    assert Path(result["backup"]).exists()
    backup_payload = json.loads(Path(result["backup"]).read_text(encoding="utf-8"))
    assert result["backup_kind"] == "controller_delta_json"
    assert Path(result["backup"]).suffix == ".json"
    assert backup_payload["format"] == "controller_residue_delta_backup_v1"
    assert {row["task_id"] for row in backup_payload["tables"]["tasks"]} == {old_controller_task_id}
    assert {row["session_id"] for row in backup_payload["tables"]["sessions"]} == {"session-old"}
    assert {row["lease_id"] for row in backup_payload["tables"]["leases"]} == {"lease-old"}
    assert {row["device_id"] for row in backup_payload["tables"]["devices"]} == {"device-old"}
    assert Path(result["outputs"]["cleanup_devices_tsv"]).exists()


def test_cleanup_does_not_plan_historical_terminal_rows(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.TIMED_OUT, controller_task_id="controller-old")
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id="controller-old",
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        task_status="interrupted",
        session_status="released",
        lease_status="released",
        device_status="offline",
        command_status="succeeded",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=False,
            skip_provider=True,
        )
    )

    assert result["planned"]["tasks"] == 0
    assert result["planned"]["sessions"] == 0
    assert result["planned"]["leases"] == 0
    assert result["planned"]["devices"] == 0


def test_controller_task_rows_filters_to_batch_and_protected_tasks(tmp_path: Path) -> None:
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id="controller-batch",
        case_id="case-1",
        session_id="session-batch",
        lease_id="lease-batch",
        device_id="device-batch",
        batch_id="batch-1",
    )
    _insert_controller_task(
        controller_db,
        task_id="controller-foreign",
        case_id="case-foreign",
        session_id="session-foreign",
        lease_id="lease-foreign",
        device_id="device-foreign",
        batch_id="batch-2",
    )
    _insert_controller_task(
        controller_db,
        task_id="controller-protected",
        case_id="case-protected",
        session_id="session-protected",
        lease_id="lease-protected",
        device_id="device-protected",
        batch_id="batch-2",
    )

    with sqlite3.connect(controller_db) as connection:
        connection.row_factory = sqlite3.Row
        rows, stats = _controller_task_rows(
            connection,
            protected_task_ids={"controller-protected"},
            lookup_task_ids={"controller-batch"},
        )

    assert {str(row["task_id"]) for row in rows} == {"controller-batch", "controller-protected"}
    assert stats["controller_task_lookup_id_count"] == 2
    assert stats["controller_task_rows_loaded"] == 2
    assert "result_json" not in rows[0].keys()


def test_controller_lookup_skips_historical_success_rows_without_resource_refs(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch-1"
    batch_dir.mkdir(exist_ok=True)
    manifest = BatchManifest(
        batch_id="batch-1",
        tasks=[BatchTaskSpec(task_id=f"case-{index}", agent_kind="desktop", query="do") for index in range(1000)],
    )
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    with sqlite3.connect(batch_dir / "batch_state.sqlite") as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("""
            UPDATE batch_tasks
            SET status='succeeded',
                attempts=1,
                controller_task_id='controller-' || task_id
            """)
        connection.execute("""
            UPDATE batch_tasks
            SET status='failed',
                attempts=2,
                controller_task_id='controller-case-7',
                retry_budget_remaining=1
            WHERE task_id='case-7'
            """)
        rows = connection.execute("SELECT * FROM batch_tasks ORDER BY order_index ASC").fetchall()

    task_ids = _controller_lookup_task_ids(batch_rows=rows, batch_id="batch-1")

    assert "controller-case-0" not in task_ids
    assert "controller-case-7" in task_ids
    assert len(task_ids) == 4


def test_cleanup_keeps_succeeded_resource_rows_in_controller_lookup(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.SUCCEEDED, controller_task_id="controller-old")
    with sqlite3.connect(batch_db) as connection:
        connection.execute("""
            UPDATE batch_tasks
            SET session_id='session-old',
                device_id='device-old',
                sandbox_id='sandbox-old',
                lease_token='lease-token-old'
            WHERE task_id='case-1'
            """)
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id="controller-old",
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        sandbox_id="sandbox-old",
        task_status="finished",
        session_status="released",
        lease_status="released",
        device_status="online",
        command_status="succeeded",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=False,
            skip_provider=True,
        )
    )

    assert result["planned"]["tasks"] == 1
    assert result["planned"]["devices"] == 1
    assert result["protected_counts"]["controller_task_lookup_row_count"] == 1


def test_cleanup_keeps_legacy_succeeded_resource_rows_without_lease_token(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.SUCCEEDED, controller_task_id="controller-old")
    with sqlite3.connect(batch_db) as connection:
        connection.execute("""
            UPDATE batch_tasks
            SET session_id='session-old',
                device_id='device-old',
                sandbox_id='sandbox-old',
                cleanup_status=''
            WHERE task_id='case-1'
            """)
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id="controller-old",
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        sandbox_id="sandbox-old",
        task_status="finished",
        session_status="released",
        lease_status="released",
        device_status="online",
        command_status="succeeded",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=False,
            skip_provider=True,
        )
    )

    assert result["planned"]["tasks"] == 1
    assert result["planned"]["devices"] == 1
    assert result["protected_counts"]["controller_task_lookup_row_count"] == 1


def test_cleanup_skips_succeeded_rows_after_successful_resource_cleanup(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.SUCCEEDED, controller_task_id="controller-old")
    with sqlite3.connect(batch_db) as connection:
        connection.execute("""
            UPDATE batch_tasks
            SET session_id='session-old',
                device_id='device-old',
                sandbox_id='sandbox-old',
                cleanup_status='ok'
            WHERE task_id='case-1'
            """)
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id="controller-old",
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        sandbox_id="sandbox-old",
        task_status="finished",
        session_status="released",
        lease_status="released",
        device_status="offline",
        command_status="succeeded",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=False,
            skip_provider=True,
        )
    )

    assert result["planned"]["tasks"] == 0
    assert result["planned"]["devices"] == 0
    assert result["protected_counts"]["controller_task_lookup_row_count"] == 0
    assert result["protected_counts"]["controller_task_lookup_rows_skipped"] == 1


def test_cleanup_detects_succeeded_controller_only_active_residue(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.SUCCEEDED, controller_task_id="controller-old")
    with sqlite3.connect(batch_db) as connection:
        connection.execute("UPDATE batch_tasks SET cleanup_status='ok' WHERE task_id='case-1'")
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id="controller-old",
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        sandbox_id="sandbox-old",
        task_status="running",
        session_status="running",
        lease_status="leased",
        device_status="online",
        command_status="running",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=False,
            skip_provider=True,
        )
    )

    assert result["planned"]["tasks"] == 1
    assert result["planned"]["devices"] == 1
    assert result["protected_counts"]["controller_task_lookup_row_count"] == 0
    assert result["protected_counts"]["controller_active_residue_task_id_count"] == 1


def test_cleanup_detects_succeeded_controller_only_active_command_residue(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.SUCCEEDED, controller_task_id="controller-old")
    with sqlite3.connect(batch_db) as connection:
        connection.execute("UPDATE batch_tasks SET cleanup_status='ok' WHERE task_id='case-1'")
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id="controller-old",
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        sandbox_id="sandbox-old",
        task_status="finished",
        session_status="released",
        lease_status="released",
        device_status="online",
        command_status="running",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=False,
            skip_provider=True,
        )
    )

    assert result["planned"]["tasks"] == 1
    assert result["planned"]["devices"] == 1
    assert result["protected_counts"]["controller_active_command_task_ids"] == 1
    assert result["protected_counts"]["controller_active_residue_task_id_count"] == 1


def test_cleanup_warns_when_deterministic_lookup_attempt_cap_is_exceeded(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    with sqlite3.connect(batch_db) as connection:
        connection.execute(
            "UPDATE batch_tasks SET attempts=33, retry_budget_remaining=1, max_attempts=40 WHERE task_id='case-1'"
        )
    controller_db = _controller_db(tmp_path)

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=False,
            skip_provider=True,
        )
    )

    assert any("deterministic lookup cap exceeded" in item for item in result["warnings"])


def test_cleanup_plans_terminal_rows_with_live_residue(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.TIMED_OUT, controller_task_id="controller-old")
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id="controller-old",
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        task_status="interrupted",
        session_status="released",
        lease_status="released",
        device_status="online",
        command_status="succeeded",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=False,
            skip_provider=True,
        )
    )

    assert result["planned"]["tasks"] == 1
    assert result["planned"]["devices"] == 1


def test_cleanup_protects_unassigned_prepared_buffer(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    prepared = SQLitePreparedSandboxStore(batch_db)
    reserved = prepared.reserve(
        task_id="case-1",
        attempt=2,
        runner_id="runner",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="tpl",
        ttl_seconds=600,
        buffer_limit=10,
        metadata={"batch_id": "batch-1"},
    )
    assert reserved is not None
    preparing = prepared.mark_preparing(reserved)
    assert preparing is not None
    ready = prepared.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device-prepared",
        lease_id="lease-prepared",
        prepare_seconds=1.0,
    )
    assert ready is not None
    controller_db = _controller_db(tmp_path)
    old_controller_task_id, _trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=1,
    )
    _insert_controller_task(
        controller_db,
        task_id=old_controller_task_id,
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id=ready.device_id,
        sandbox_id=ready.sandbox_id,
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            protect_prepared=True,
            apply=True,
            skip_provider=True,
        )
    )

    assert result["planned"]["tasks"] == 0
    assert result["planned"]["blocked"] == 1


def test_cleanup_can_include_unassigned_prepared_buffer_for_explicit_stop(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    prepared = SQLitePreparedSandboxStore(batch_db)
    reserved = prepared.reserve(
        task_id="case-1",
        attempt=2,
        runner_id="runner",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="tpl",
        ttl_seconds=600,
        buffer_limit=10,
        metadata={"batch_id": "batch-1"},
    )
    assert reserved is not None
    preparing = prepared.mark_preparing(reserved)
    assert preparing is not None
    ready = prepared.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="device-prepared",
        lease_id="lease-prepared",
        prepare_seconds=1.0,
    )
    assert ready is not None
    controller_db = _controller_db(tmp_path)

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            protect_prepared=False,
            include_prepared=True,
            apply=True,
            skip_provider=True,
        )
    )

    assert result["planned"]["tasks"] == 0
    assert result["planned"]["sandboxes"] == 1
    assert result["planned"]["devices"] == 1


def test_cleanup_plans_cleanup_failed_prepared_residue_for_provider_termination(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    prepared = SQLitePreparedSandboxStore(batch_db)
    reserved = prepared.reserve(
        task_id="case-1",
        attempt=2,
        runner_id="runner",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="tpl",
        ttl_seconds=600,
        buffer_limit=10,
        metadata={"batch_id": "batch-1"},
    )
    assert reserved is not None
    failed = prepared.mark_cleanup_failed(
        reserved,
        error="terminate failed",
        sandbox_id="sandbox-cleanup-failed",
        device_id="device-cleanup-failed",
        lease_id="lease-cleanup-failed",
    )
    assert failed is not None
    controller_db = _controller_db(tmp_path)
    _insert_device(controller_db, device_id="device-cleanup-failed", descriptor={"display_name": "batch-1:case-1"})
    sandbox_client = _RecordingSandboxClient()

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            provider="fake",
            sandbox_client=sandbox_client,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        device_status = connection.execute(
            "SELECT status FROM devices WHERE device_id='device-cleanup-failed'"
        ).fetchone()[0]
    assert result["planned"]["sandboxes"] == 1
    assert result["planned"]["devices"] == 1
    assert result["provider_status"]["terminated"] == 1
    assert result["cleaned"]["sandboxes"] == 1
    assert result["cleaned"]["devices"] == 1
    assert sandbox_client.terminated == ["sandbox-cleanup-failed"]
    assert device_status == "offline"


def test_cleanup_directly_terminates_live_prepared_residue_when_inventory_fails(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    prepared = SQLitePreparedSandboxStore(batch_db)
    reserved = prepared.reserve(
        task_id="case-1",
        attempt=2,
        runner_id="runner",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="tpl",
        ttl_seconds=600,
        buffer_limit=10,
        metadata={"batch_id": "batch-1"},
    )
    assert reserved is not None
    failed = prepared.mark_cleanup_failed(
        reserved,
        error="terminate failed",
        sandbox_id="sandbox-cleanup-failed",
        device_id="device-cleanup-failed",
        lease_id="lease-cleanup-failed",
    )
    assert failed is not None
    controller_db = _controller_db(tmp_path)
    _insert_device(controller_db, device_id="device-cleanup-failed", descriptor={"display_name": "batch-1:case-1"})
    sandbox_client = _InventoryTimeoutSandboxClient()

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            provider="fake",
            sandbox_client=sandbox_client,
        )
    )

    assert result["provider_status"]["terminated"] == 1
    assert result["cleaned"]["sandboxes"] == 1
    assert sandbox_client.terminated == ["sandbox-cleanup-failed"]
    assert any("provider inventory failed" in item for item in result["warnings"])


def test_cleanup_directly_terminates_candidate_missing_from_nonempty_inventory(tmp_path: Path) -> None:
    old_controller_task_id, _trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=1,
    )
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED, controller_task_id=old_controller_task_id)
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id=old_controller_task_id,
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        sandbox_id="sandbox-old",
    )
    sandbox_client = _PartialInventorySandboxClient()

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            provider="fake",
            sandbox_client=sandbox_client,
        )
    )

    assert sandbox_client.terminated == ["sandbox-old"]
    assert result["provider_status"] == {"terminated": 1}
    assert result["cleaned"]["sandboxes"] == 1
    assert result["warnings"] == []


def test_cleanup_treats_provider_404_as_cleaned_without_warning_traceback(
    tmp_path: Path,
    caplog,
) -> None:
    old_controller_task_id, _trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=1,
    )
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED, controller_task_id=old_controller_task_id)
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id=old_controller_task_id,
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        sandbox_id="sandbox-old",
    )

    with caplog.at_level(logging.WARNING, logger="northcua.maintenance.cleanup_controller_sandboxes"):
        result = asyncio.run(
            cleanup_controller_residue(
                batch_db=batch_db,
                controller_db=controller_db,
                batch_id="batch-1",
                apply=True,
                provider="fake",
                sandbox_client=_MissingSandboxClient(),
            )
        )

    assert result["provider_status"]["provider_missing"] == 1
    assert result["provider_missing"] == 1
    assert result["cleaned"]["sandboxes"] == 1
    assert not [record for record in caplog.records if "sandbox terminate failed" in record.getMessage()]


def test_cleanup_keeps_generic_provider_404_as_terminate_failed(tmp_path: Path) -> None:
    old_controller_task_id, _trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=1,
    )
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED, controller_task_id=old_controller_task_id)
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id=old_controller_task_id,
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        sandbox_id="sandbox-old",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            provider="fake",
            sandbox_client=_Generic404SandboxClient(),
        )
    )

    assert result["provider_status"]["terminate_failed"] == 1
    assert result["provider_missing"] == 0
    assert result["cleaned"]["sandboxes"] == 0
    assert any("sandbox terminate failed" in item for item in result["warnings"])
    with sqlite3.connect(controller_db) as connection:
        task_status = connection.execute(
            "SELECT status FROM tasks WHERE task_id=?", (old_controller_task_id,)
        ).fetchone()[0]
        session_status = connection.execute("SELECT status FROM sessions WHERE session_id='session-old'").fetchone()[0]
        device_status = connection.execute("SELECT status FROM devices WHERE device_id='device-old'").fetchone()[0]
    assert task_status == "running"
    assert session_status == "running"
    assert device_status == "online"


def test_cleanup_bounds_hanging_provider_terminate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cleanup_controller_sandboxes, "SANDBOX_TERMINATE_TIMEOUT_SECONDS", 0.05)
    old_controller_task_id, _trace_id = BatchTaskPlanner._deterministic_ids(
        batch_id="batch-1",
        task_id="case-1",
        attempt=1,
    )
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED, controller_task_id=old_controller_task_id)
    controller_db = _controller_db(tmp_path)
    _insert_controller_task(
        controller_db,
        task_id=old_controller_task_id,
        case_id="case-1",
        session_id="session-old",
        lease_id="lease-old",
        device_id="device-old",
        sandbox_id="sandbox-old",
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            provider="fake",
            sandbox_client=_HangingTerminateSandboxClient(),
        )
    )

    assert result["provider_status"]["terminate_failed"] == 1
    assert result["cleaned"]["sandboxes"] == 0
    assert any("timed out after 0.1s" in item for item in result["warnings"])
    with sqlite3.connect(controller_db) as connection:
        task_status = connection.execute(
            "SELECT status FROM tasks WHERE task_id=?", (old_controller_task_id,)
        ).fetchone()[0]
        session_status = connection.execute("SELECT status FROM sessions WHERE session_id='session-old'").fetchone()[0]
        device_status = connection.execute("SELECT status FROM devices WHERE device_id='device-old'").fetchone()[0]
    assert task_status == "running"
    assert session_status == "running"
    assert device_status == "online"


def test_provider_missing_match_does_not_treat_template_404_as_sandbox_gone() -> None:
    assert not is_sandbox_missing_error(SandboxNotFoundError("not found"), sandbox_id="sbx-1")
    assert not is_sandbox_missing_error(SandboxTemplateNotFoundError("template not found 404"), sandbox_id="sbx-1")
    assert not is_sandbox_missing_error(RuntimeError("sandbox API route returned 404"), sandbox_id="sbx-1")
    assert is_sandbox_missing_error(RuntimeError("sandbox sbx-1 not found 404"), sandbox_id="sbx-1")


def test_cleanup_blocks_foreign_batch_orphan_device_by_default(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.RUNNING, controller_task_id="controller-current")
    controller_db = _controller_db(tmp_path)
    _insert_device(
        controller_db,
        device_id="batch-old-batch-sc-0001-prep-abcd1234",
        descriptor={"display_name": "old-batch:sc-0001"},
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            skip_provider=True,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        device_status = connection.execute(
            "SELECT status FROM devices WHERE device_id='batch-old-batch-sc-0001-prep-abcd1234'"
        ).fetchone()[0]
    assert result["planned"]["tasks"] == 0
    assert result["planned"]["devices"] == 0
    assert result["planned"]["blocked"] == 1
    assert result["cleaned"]["devices"] == 0
    blocked_tsv = Path(str(result["outputs"]["blocked_tsv"])).read_text(encoding="utf-8")
    assert "foreign_batch_online_orphan_device" in blocked_tsv
    assert "old-batch" in blocked_tsv
    assert device_status == "online"


def test_cleanup_marks_same_batch_orphan_device_offline(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    controller_db = _controller_db(tmp_path)
    _insert_device(
        controller_db,
        device_id="batch-batch-1-sc-0001-prep-abcd1234",
        descriptor={"display_name": "batch-1:sc-0001"},
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            skip_provider=True,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        device_status = connection.execute(
            "SELECT status FROM devices WHERE device_id='batch-batch-1-sc-0001-prep-abcd1234'"
        ).fetchone()[0]
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_northcua_cleanup_%'"
            )
        }
        task_plan = " ".join(
            str(row[3])
            for row in connection.execute("EXPLAIN QUERY PLAN SELECT task_id FROM tasks WHERE status IN ('running')")
        )
        command_plan = " ".join(
            str(row[3])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT session_id FROM commands WHERE status IN ('running')"
            )
        )
    assert result["planned"]["devices"] == 1
    assert result["cleaned"]["devices"] == 1
    assert result["residue_online_before"] == 1
    assert result["residue_online_after"] == 0
    assert device_status == "offline"
    assert "idx_northcua_cleanup_tasks_status_task" in indexes
    assert "idx_northcua_cleanup_devices_status" in indexes
    assert "idx_northcua_cleanup_sessions_device" in indexes
    assert "idx_northcua_cleanup_commands_device_status" in indexes
    assert "idx_northcua_cleanup_commands_session_status" in indexes
    assert "idx_northcua_cleanup_commands_status_session" in indexes
    assert "idx_northcua_cleanup_tasks_status_task" in task_plan
    assert "idx_northcua_cleanup_commands_status_session" in command_plan


def test_cleanup_marks_orphan_device_with_released_session_offline(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    controller_db = _controller_db(tmp_path)
    now = "2026-05-15T00:00:00+00:00"
    with sqlite3.connect(controller_db) as connection:
        connection.execute(
            "INSERT INTO devices(device_id, status, descriptor_json, last_heartbeat_at) VALUES (?, ?, ?, ?)",
            (
                "batch-batch-1-sc-0001-prep-abcd1234",
                "online",
                json.dumps({"display_name": "batch-1:sc-0001"}),
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO sessions(session_id, task_id, device_id, lease_id, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "released-session",
                "finished-task",
                "batch-batch-1-sc-0001-prep-abcd1234",
                "",
                "released",
                now,
                now,
            ),
        )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            skip_provider=True,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        device_status = connection.execute(
            "SELECT status FROM devices WHERE device_id='batch-batch-1-sc-0001-prep-abcd1234'"
        ).fetchone()[0]
    assert result["planned"]["devices"] == 1
    assert result["cleaned"]["devices"] == 1
    assert device_status == "offline"


def test_cleanup_reuses_existing_covering_controller_indexes(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    controller_db = _controller_db(tmp_path)
    with sqlite3.connect(controller_db) as connection:
        connection.execute("CREATE INDEX commands_device_status_idx ON commands(device_id, status)")
        connection.execute("CREATE INDEX commands_session_status_idx ON commands(session_id, status)")
        connection.execute("CREATE INDEX sessions_device_status_idx ON sessions(device_id, status)")

    asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            skip_provider=True,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        command_indexes = {row[1] for row in connection.execute("PRAGMA index_list(commands)")}
        session_indexes = {row[1] for row in connection.execute("PRAGMA index_list(sessions)")}

    assert "commands_device_status_idx" in command_indexes
    assert "commands_session_status_idx" in command_indexes
    assert "sessions_device_status_idx" in session_indexes
    assert "idx_northcua_cleanup_commands_device_status" not in command_indexes
    assert "idx_northcua_cleanup_commands_session_status" not in command_indexes
    assert "idx_northcua_cleanup_sessions_device" not in session_indexes


def test_cleanup_does_not_treat_partial_index_as_covering(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    controller_db = _controller_db(tmp_path)
    with sqlite3.connect(controller_db) as connection:
        connection.execute(
            "CREATE INDEX commands_device_status_running_idx ON commands(device_id, status) WHERE status='running'"
        )

    asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            skip_provider=True,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        command_indexes = {row[1] for row in connection.execute("PRAGMA index_list(commands)")}

    assert "commands_device_status_running_idx" in command_indexes
    assert "idx_northcua_cleanup_commands_device_status" in command_indexes


def test_cleanup_does_not_treat_expression_index_as_covering(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    controller_db = _controller_db(tmp_path)
    with sqlite3.connect(controller_db) as connection:
        connection.execute("CREATE INDEX commands_expr_prefix_idx ON commands(lower(device_id), status)")
        connection.execute("CREATE INDEX commands_expr_middle_idx ON commands(session_id, lower(device_id), status)")

    asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            skip_provider=True,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        command_indexes = {row[1] for row in connection.execute("PRAGMA index_list(commands)")}

    assert "commands_expr_prefix_idx" in command_indexes
    assert "commands_expr_middle_idx" in command_indexes
    assert "idx_northcua_cleanup_commands_device_status" in command_indexes
    assert "idx_northcua_cleanup_commands_session_status" in command_indexes


def test_cleanup_does_not_treat_nocase_index_as_covering(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    controller_db = _controller_db(tmp_path)
    with sqlite3.connect(controller_db) as connection:
        connection.execute(
            "CREATE INDEX commands_device_status_nocase_idx ON commands(device_id COLLATE NOCASE, status)"
        )

    asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            skip_provider=True,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        command_indexes = {row[1] for row in connection.execute("PRAGMA index_list(commands)")}

    assert "commands_device_status_nocase_idx" in command_indexes
    assert "idx_northcua_cleanup_commands_device_status" in command_indexes


def test_cleanup_protects_prepared_orphan_device(tmp_path: Path) -> None:
    batch_db = _batch_db(tmp_path, BatchTaskStatus.FAILED)
    prepared = SQLitePreparedSandboxStore(batch_db)
    reserved = prepared.reserve(
        task_id="case-1",
        attempt=2,
        runner_id="runner",
        depth=PreparedSandboxDepth.FULL_INIT,
        provider="fake",
        template="tpl",
        ttl_seconds=600,
        buffer_limit=10,
        metadata={"batch_id": "batch-1"},
    )
    assert reserved is not None
    preparing = prepared.mark_preparing(reserved)
    assert preparing is not None
    ready = prepared.mark_prepared(
        preparing,
        sandbox_id="sandbox-prepared",
        device_id="batch-old-batch-sc-0001-prep-abcd1234",
        lease_id="lease-prepared",
        prepare_seconds=1.0,
    )
    assert ready is not None
    controller_db = _controller_db(tmp_path)
    _insert_device(
        controller_db,
        device_id=ready.device_id,
        descriptor={"display_name": "old-batch:sc-0001"},
    )

    result = asyncio.run(
        cleanup_controller_residue(
            batch_db=batch_db,
            controller_db=controller_db,
            batch_id="batch-1",
            apply=True,
            skip_provider=True,
        )
    )

    with sqlite3.connect(controller_db) as connection:
        device_status = connection.execute(
            "SELECT status FROM devices WHERE device_id=?", (ready.device_id,)
        ).fetchone()[0]
    assert result["planned"]["devices"] == 0
    assert result["planned"]["blocked"] == 1
    assert device_status == "online"


def _batch_db(tmp_path: Path, status: BatchTaskStatus, *, controller_task_id: str = "") -> Path:
    batch_dir = tmp_path / "batch-1"
    batch_dir.mkdir(exist_ok=True)
    manifest = BatchManifest(
        batch_id="batch-1", tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do")]
    )
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    repository.upsert_state(
        BatchTaskState(
            task_id="case-1",
            status=status,
            attempts=1,
            controller_task_id=controller_task_id,
            session_id="session-current" if controller_task_id == "controller-current" else "",
            device_id="device-current" if controller_task_id == "controller-current" else "",
        )
    )
    return batch_dir / "batch_state.sqlite"


def _controller_db(tmp_path: Path) -> Path:
    path = tmp_path / "controller.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                agent_kind TEXT NOT NULL,
                submitter_id TEXT NOT NULL DEFAULT '',
                request_json TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                finished_at TEXT
            );
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                device_id TEXT,
                lease_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE leases (
                lease_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                status TEXT NOT NULL,
                expires_at TEXT,
                generation INTEGER NOT NULL
            );
            CREATE TABLE devices (
                device_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                descriptor_json TEXT NOT NULL,
                last_heartbeat_at TEXT,
                active_sessions INTEGER NOT NULL DEFAULT 0,
                load_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE commands (
                command_id TEXT PRIMARY KEY,
                protocol_command_id TEXT NOT NULL DEFAULT '',
                idempotency_key TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                lease_id TEXT NOT NULL DEFAULT '',
                device_id TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)
    return path


def _insert_controller_task(
    controller_db: Path,
    *,
    task_id: str,
    case_id: str,
    session_id: str,
    lease_id: str,
    device_id: str,
    batch_id: str = "batch-1",
    sandbox_id: str = "sandbox",
    task_status: str = "running",
    session_status: str = "running",
    lease_status: str = "leased",
    device_status: str = "online",
    command_status: str = "running",
) -> None:
    now = "2026-05-15T00:00:00+00:00"
    request_json = json.dumps(
        {"metadata": {"batch_id": batch_id, "batch_task_id": case_id, "sandbox_id": sandbox_id}},
        ensure_ascii=False,
    )
    with sqlite3.connect(controller_db) as connection:
        connection.execute(
            "INSERT INTO tasks(task_id, trace_id, agent_kind, request_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, "trace", "desktop", request_json, task_status, now),
        )
        connection.execute(
            "INSERT INTO sessions(session_id, task_id, device_id, lease_id, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, task_id, device_id, lease_id, session_status, now, now),
        )
        connection.execute(
            "INSERT INTO leases(lease_id, session_id, device_id, status, generation) VALUES (?, ?, ?, ?, ?)",
            (lease_id, session_id, device_id, lease_status, 1),
        )
        connection.execute(
            "INSERT INTO devices(device_id, status, descriptor_json, last_heartbeat_at) VALUES (?, ?, ?, ?)",
            (device_id, device_status, "{}", now),
        )
        connection.execute(
            "INSERT INTO commands(command_id, idempotency_key, session_id, lease_id, device_id, status, request_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("cmd-" + task_id, "idem-" + task_id, session_id, lease_id, device_id, command_status, "{}", now, now),
        )


def _insert_device(controller_db: Path, *, device_id: str, descriptor: dict[str, object]) -> None:
    now = "2026-05-15T00:00:00+00:00"
    with sqlite3.connect(controller_db) as connection:
        connection.execute(
            "INSERT INTO devices(device_id, status, descriptor_json, last_heartbeat_at) VALUES (?, ?, ?, ?)",
            (device_id, "online", json.dumps(descriptor), now),
        )
