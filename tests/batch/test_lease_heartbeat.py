from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.runner.events import BatchEventWriter
from northcua.batch.runner.lease_heartbeat import start_batch_lease_heartbeat
from northcua.batch.state import LeasePolicy, SQLiteBatchStateRepository
from northcua.telemetry.contracts import FailureCategory


def _manifest() -> BatchManifest:
    return BatchManifest(
        batch_id="lease-heartbeat",
        tasks=[BatchTaskSpec(task_id="task-1", agent_kind="web", query="Open page")],
    )


@pytest.mark.asyncio
async def test_heartbeat_does_not_treat_controller_submit_canceled_as_operator_cancel(tmp_path: Path) -> None:
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    policy = LeasePolicy(lease_timeout_seconds=5.0, heartbeat_interval_seconds=0.05)
    claim = repository.claim_next(lease_owner="runner-a", policy=policy)
    assert claim is not None
    assert repository.upsert_state(
        BatchTaskState(
            task_id=claim.task_id,
            status=BatchTaskStatus.CANCELED,
            attempts=1,
            error="queued task lost its submitter mapping",
            last_failure_category=FailureCategory.CONTROLLER_SUBMIT,
        ),
        lease_token=claim.lease_token,
    )
    assert not repository.cancel_requested(claim.task_id)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id="lease-heartbeat", lease=claim)

    heartbeat = start_batch_lease_heartbeat(events=events, claim=claim, policy=policy)
    try:
        await asyncio.sleep(0.15)
        assert not heartbeat.lost.is_set()
    finally:
        await heartbeat.close()


@pytest.mark.asyncio
async def test_heartbeat_stops_for_explicit_operator_cancel(tmp_path: Path) -> None:
    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(_manifest())
    policy = LeasePolicy(lease_timeout_seconds=5.0, heartbeat_interval_seconds=0.05)
    claim = repository.claim_next(lease_owner="runner-a", policy=policy)
    assert claim is not None
    repository.mark_cancel_requested()
    assert repository.cancel_requested(claim.task_id)
    events = BatchEventWriter(repository=repository, output_dir=tmp_path, batch_id="lease-heartbeat", lease=claim)

    heartbeat = start_batch_lease_heartbeat(events=events, claim=claim, policy=policy)
    try:
        await asyncio.wait_for(heartbeat.lost.wait(), timeout=1.0)
    finally:
        await heartbeat.close()
