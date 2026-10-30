"""Batch reporting artifact contract tests."""

from __future__ import annotations

import json

from northcua.batch.contracts import BatchRunResult, BatchTaskState
from northcua.batch.reporting import BatchReportWriter


def test_batch_report_writer_creates_empty_prepared_events_contract(tmp_path) -> None:
    """Final reports always expose the prepared-sandbox monitor JSONL path."""

    result = BatchRunResult(
        batch_id="batch-no-prepared",
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        incomplete=0,
        output_dir=str(tmp_path),
        task_states=[BatchTaskState(task_id="case-1")],
    )

    BatchReportWriter(tmp_path).write_summary(result, metrics={})

    prepared_events = tmp_path / "prepared_sandbox_events.jsonl"
    assert prepared_events.exists()
    assert prepared_events.read_text(encoding="utf-8") == ""


def test_batch_report_writer_preserves_existing_prepared_events(tmp_path) -> None:
    """Summary writing must not truncate events already emitted by the queue."""

    prepared_events = tmp_path / "prepared_sandbox_events.jsonl"
    prepared_events.write_text('{"message":"prepared_sandbox_assigned"}\n', encoding="utf-8")
    result = BatchRunResult(
        batch_id="batch-with-prepared",
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        incomplete=0,
        output_dir=str(tmp_path),
        task_states=[BatchTaskState(task_id="case-1")],
    )

    BatchReportWriter(tmp_path).write_summary(result, metrics={})

    assert prepared_events.read_text(encoding="utf-8") == '{"message":"prepared_sandbox_assigned"}\n'
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["prepared_sandbox_hits"] == 1


def test_batch_report_writer_preserves_telemetry_fallback_before_touching_prepared_events(tmp_path) -> None:
    """Legacy telemetry-only prepared rows still reconstruct batch metrics."""

    (tmp_path / "telemetry_events.jsonl").write_text(
        json.dumps(
            {
                "message": "prepared_sandbox_assigned",
                "details": {"prepare_seconds": 12.0, "claim_wait_seconds": 2.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = BatchRunResult(
        batch_id="batch-telemetry-fallback",
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        incomplete=0,
        output_dir=str(tmp_path),
        task_states=[BatchTaskState(task_id="case-1")],
    )

    BatchReportWriter(tmp_path).write_summary(result, metrics={})

    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["prepared_sandbox_hits"] == 1
    assert summary["metrics"]["prepared_sandbox_p50_claim_wait_seconds"] == 2.5
    assert summary["metrics"]["prepared_sandbox_estimated_saved_seconds"] == 9.5
    assert (tmp_path / "prepared_sandbox_events.jsonl").exists()
    assert (tmp_path / "prepared_sandbox_events.jsonl").read_text(encoding="utf-8") == ""
