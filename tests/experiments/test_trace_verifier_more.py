"""Experiment trace verifier tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
from tests.support.experiment_fixtures import golden_e2b_experiment_spec, run_fixture_experiment
from northcua.experiments.verification import ExperimentTraceVerifier, FakeTraceSeeder
from tests.experiments.trace_verifier_test_utils import (
    move_seeded_artifact_to_controller_blob as _move_seeded_artifact_to_controller_blob,
)
from tests.experiments.trace_verifier_test_utils import seed_controller_artifact_db as _seed_controller_artifact_db


def _batch_result_for_golden(spec) -> BatchRunResult:
    states = [
        BatchTaskState(
            task_id=task.task_id,
            controller_task_id=f"controller-{task.task_id}",
            trace_id=f"trace-{task.task_id}",
            session_id=f"session-{task.task_id}",
            sandbox_id=f"sandbox-{task.task_id}",
            status=BatchTaskStatus.SUCCEEDED,
            attempts=1,
        )
        for task in spec.tasks
    ]
    return BatchRunResult(
        batch_id=spec.experiment_id,
        total=len(states),
        succeeded=len(states),
        failed=0,
        canceled=0,
        output_dir="",
        task_states=states,
    )


def test_trace_verifier_continues_after_corrupt_blob_root(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    task = spec.tasks[0]
    state = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    batch_result = BatchRunResult(
        batch_id=spec.experiment_id,
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[state],
    )
    db_path = tmp_path / "controller.sqlite"
    _seed_controller_artifact_db(db_path, state)
    report_id = f"{state.controller_task_id}-evaluation-desktop_round_completion-round_0001-final_report"
    report_payload = b'{"overall_verdict":"success"}'
    _move_seeded_artifact_to_controller_blob(
        db_path=db_path,
        blob_dir=tmp_path / "controller-blobs",
        artifact_id=report_id,
        payload=report_payload,
        mime_type="application/json",
    )
    corrupt_blob_dir = tmp_path / "stale-controller-blobs"
    corrupt_blob_dir.mkdir()
    (corrupt_blob_dir / hashlib.sha256(report_payload).hexdigest()).write_bytes(b"corrupt")

    rows = ExperimentTraceVerifier().verify(
        output_dir=tmp_path,
        spec=spec,
        batch_result=batch_result,
        extra_roots=[corrupt_blob_dir],
    )
    row = next(item for item in rows if item.task_id == task.task_id)

    assert row.passed
    assert Path(row.evaluator_report_path).read_bytes() == report_payload
