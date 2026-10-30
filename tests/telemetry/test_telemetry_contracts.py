"""Telemetry contract, sink, and report coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from northcua.telemetry import (
    CaseArtifactTelemetry,
    CasePhase,
    CaseStatus,
    CaseTelemetryRecorder,
    CaseTelemetrySnapshot,
    LocalTelemetrySink,
    PhaseTiming,
    TelemetryEntityKind,
    TelemetryReportWriter,
)


def test_phase_timer_records_successful_timing(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch", experiment_id="experiment")

    with recorder.phase_timer(
        case_id="case-1",
        phase=CasePhase.SANDBOX_CREATE,
        status=CaseStatus.PROVISIONING,
        entity_kind=TelemetryEntityKind.SANDBOX,
    ):
        pass

    timing_rows = _jsonl(tmp_path / "case_timelines.jsonl")
    event_rows = _jsonl(tmp_path / "telemetry_events.jsonl")
    assert timing_rows[0]["case_id"] == "case-1"
    assert timing_rows[0]["phase"] == "sandbox_create"
    assert timing_rows[0]["duration_seconds"] >= 0
    assert [row["event_type"] for row in event_rows] == ["phase_started", "phase_finished"]


def test_phase_timer_records_failure_category(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")

    with pytest.raises(RuntimeError):
        with recorder.phase_timer(
            case_id="case-1",
            phase=CasePhase.MAIN_AGENT,
            status=CaseStatus.RUNNING_AGENT,
        ):
            raise RuntimeError("agent failed")

    timing = _jsonl(tmp_path / "case_timelines.jsonl")[0]
    failure = _jsonl(tmp_path / "telemetry_events.jsonl")[-1]
    assert timing["failure_category"] == "main_agent"
    assert failure["event_type"] == "failure_recorded"
    assert failure["failure_category"] == "main_agent"


def test_phase_timer_classifies_sandbox_readiness_failure(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")

    with pytest.raises(RuntimeError):
        with recorder.phase_timer(
            case_id="case-1",
            phase=CasePhase.SANDBOX_CREATE,
            status=CaseStatus.PROVISIONING,
        ):
            raise RuntimeError("sandbox device did not become ready before timeout")

    timing = _jsonl(tmp_path / "case_timelines.jsonl")[0]
    assert timing["failure_category"] == "readiness_timeout"


def test_phase_timer_classifies_device_command_timeout(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")

    with pytest.raises(RuntimeError):
        with recorder.phase_timer(
            case_id="case-1",
            phase=CasePhase.MAIN_AGENT,
            status=CaseStatus.RUNNING_AGENT,
        ):
            raise RuntimeError("controller command dispatch timed out after 60001 ms")

    timing = _jsonl(tmp_path / "case_timelines.jsonl")[0]
    assert timing["failure_category"] == "device_command_timeout"


def test_phase_timer_redacts_secret_like_error_messages(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")

    with pytest.raises(RuntimeError):
        with recorder.phase_timer(
            case_id="case-1",
            phase=CasePhase.MAIN_AGENT,
            status=CaseStatus.RUNNING_AGENT,
        ):
            raise RuntimeError("provider failed token=super-secret")

    payload = _jsonl(tmp_path / "telemetry_events.jsonl")[-1]
    assert "super-secret" not in payload["message"]
    assert "token=<redacted>" in payload["message"]


def test_telemetry_report_writes_case_metrics_and_summary(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    recorder.record_batch_state(_state("case-1", "succeeded"))
    with recorder.phase_timer(case_id="case-1", phase=CasePhase.MAIN_AGENT, status=CaseStatus.RUNNING_AGENT):
        pass

    summary = TelemetryReportWriter(tmp_path).write()

    assert summary.total_cases == 1
    assert summary.succeeded == 1
    assert (tmp_path / "case_metrics.jsonl").exists()
    assert (tmp_path / "telemetry_summary.json").exists()
    assert (tmp_path / "audit_report.md").read_text(encoding="utf-8").startswith("# Telemetry Audit Report")


def test_telemetry_report_derives_prepared_sandbox_time_saved(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    recorder.record_batch_state(_state("case-1", "succeeded"))
    recorder.record_case_event(
        case_id="case-1",
        entity_kind=TelemetryEntityKind.SANDBOX,
        entity_id="sandbox-1",
        phase=CasePhase.PREPARED_SANDBOX,
        status=CaseStatus.PROVISIONING,
        attempt=1,
        message="prepared_sandbox_ready",
        details={"attempt": 1, "prepare_seconds": 12.0},
    )
    recorder.record_case_event(
        case_id="case-1",
        entity_kind=TelemetryEntityKind.SANDBOX,
        entity_id="sandbox-1",
        phase=CasePhase.PREPARED_SANDBOX,
        status=CaseStatus.PROVISIONING,
        attempt=1,
        message="prepared_sandbox_assigned",
        details={"attempt": 1, "prepare_seconds": 12.0, "claim_wait_seconds": 2.5},
    )

    summary = TelemetryReportWriter(tmp_path).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert metric["prepared_sandbox_hit"] is True
    assert metric["prepared_sandbox_claim_wait_seconds"] == 2.5
    assert metric["prepared_sandbox_estimated_saved_seconds"] == 9.5
    assert summary.prepared_sandbox_hits == 1
    assert summary.prepared_sandbox_estimated_saved_seconds == 9.5
    assert summary.prepared_sandbox_p50_estimated_saved_seconds == 9.5


def test_telemetry_report_counts_prepared_sandbox_diagnostics(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    recorder.record_batch_state(_state("case-1", "failed"))
    recorder.record_case_event(
        case_id="case-1",
        entity_kind=TelemetryEntityKind.SANDBOX,
        entity_id="sandbox-1",
        phase=CasePhase.PREPARED_SANDBOX,
        status=CaseStatus.PROVISIONING,
        attempt=1,
        message="prepared_sandbox_fallback",
        details={
            "attempt": 1,
            "reason": "miss",
            "active_unassigned": 0,
            "prepared_status_counts": {"released": 3, "cleanup_failed": 1},
        },
    )
    recorder.record_case_event(
        case_id="case-1",
        entity_kind=TelemetryEntityKind.SANDBOX,
        entity_id="sandbox-1",
        phase=CasePhase.PREPARED_SANDBOX,
        status=CaseStatus.FAILED,
        attempt=1,
        message="prepared_sandbox_cleanup_failed",
        details={"attempt": 1, "cleanup_result": "failed", "error": "release timed out"},
    )

    summary = TelemetryReportWriter(tmp_path).write()

    assert summary.prepared_sandbox_miss_breakdown == {"miss": 1}
    assert summary.prepared_sandbox_cleanup_failures == 1
    assert summary.prepared_sandbox_status_breakdown == {"cleanup_failed": 1, "released": 3}


def test_telemetry_report_enriches_app_operation_groups_from_manifest(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    recorder.record_batch_state(_state("case-1", "succeeded"))
    recorder.record_batch_state(_state("case-2", "failed"))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "batch_id": "batch",
                "tasks": [
                    {
                        "task_id": "case-1",
                        "metadata": {
                            "app_profile": "libreoffice.calc",
                            "involved_app_profiles": ["libreoffice.calc", "libreoffice.writer"],
                            "involved_app_count": 2,
                            "operation_family": "multi_app_transfer",
                            "expected_feasibility": "feasible",
                        },
                    },
                    {
                        "task_id": "case-2",
                        "metadata": {
                            "app_profile": "desktop.generic",
                            "involved_app_profiles": ["desktop.generic"],
                            "involved_app_count": 1,
                            "operation_family": "infeasible",
                            "expected_feasibility": "infeasible",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = TelemetryReportWriter(tmp_path).write()
    metrics = {row["case_id"]: row for row in _jsonl(tmp_path / "case_metrics.jsonl")}

    assert metrics["case-1"]["operation_family"] == "multi_app_transfer"
    assert metrics["case-1"]["involved_app_count"] == 2
    assert summary.by_involved_app["libreoffice.writer"].total == 1
    assert summary.by_operation_family["multi_app_transfer"].succeeded == 1
    assert summary.by_expected_feasibility["infeasible"].failed == 1


def test_telemetry_summary_counts_expected_infeasible_terminal_as_success(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    from northcua.batch.contracts import BatchTaskState, BatchTaskStatus

    recorder.record_batch_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            terminal_decision={"action": "fail", "reason": "infeasible", "retryable": False},
            attempts=1,
        )
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "batch_id": "batch",
                "tasks": [
                    {
                        "task_id": "case-1",
                        "metadata": {
                            "app_profile": "desktop.generic",
                            "operation_family": "infeasible",
                            "expected_feasibility": "infeasible",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = TelemetryReportWriter(tmp_path).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]
    audit = (tmp_path / "audit_report.md").read_text(encoding="utf-8")

    assert summary.succeeded == 1
    assert summary.failed == 0
    assert metric["status"] == "succeeded"
    assert metric["failure_category"] is None
    assert metric["attempt_failure_category_breakdown"] == {}
    assert "| `case-1` | succeeded |" in audit
    assert summary.by_expected_feasibility["infeasible"].succeeded == 1
    assert summary.by_expected_feasibility["infeasible"].failed == 0


def test_telemetry_report_includes_recovery_and_cleanup_fields(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchTaskState, BatchTaskStatus
    from northcua.reliability.contracts import FailureAction

    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    recorder.record_batch_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=2,
            recovery_action=FailureAction.RETRY_CASE,
            retry_budget_remaining=1,
            cleanup_status="failed: teardown timeout",
            error="failed",
        )
    )

    summary = TelemetryReportWriter(tmp_path).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert summary.retry_summary == {"retry_case": 1}
    assert summary.cleanup_failures == 1
    assert metric["recovery_action"] == "retry_case"
    assert metric["retry_budget_remaining"] == 1


def test_telemetry_report_summarizes_post_run_benchmarks(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchTaskState, BatchTaskStatus
    from northcua.batch.post_run import PostRunHookResult, PostRunHookStatus

    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    recorder.record_batch_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.SUCCEEDED,
            attempts=1,
            post_run_results=[
                PostRunHookResult(
                    hook_id="cuagym_reward",
                    status=PostRunHookStatus.PASSED,
                    passed=True,
                    score=0.8,
                    artifacts={"/tmp/reward.json": str(tmp_path / "reward.json")},
                )
            ],
        )
    )

    summary = TelemetryReportWriter(tmp_path).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]
    audit = (tmp_path / "audit_report.md").read_text(encoding="utf-8")

    assert summary.benchmark_passed == 1
    assert summary.benchmark_failed == 0
    assert summary.benchmark_errors == 0
    assert summary.benchmark_avg_score == 0.8
    assert metric["benchmark_hook_id"] == "cuagym_reward"
    assert metric["benchmark_status"] == "passed"
    assert metric["reward_artifact_path"].endswith("reward.json")
    assert "- benchmark passed: 1" in audit
    assert "passed:0.800:true" in audit


def test_telemetry_report_preserves_transient_retry_failure_categories(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchTaskState, BatchTaskStatus
    from northcua.telemetry.contracts import FailureCategory

    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    recorder.record_batch_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            error="COMMAND_TIMED_OUT",
            last_failure_category=FailureCategory.DEVICE_COMMAND_TIMEOUT,
        )
    )
    recorder.record_batch_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED, attempts=2))

    summary = TelemetryReportWriter(tmp_path).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert metric["status"] == "succeeded"
    assert metric["retry_count"] == 1
    assert metric["failure_category"] is None
    assert metric["attempt_failure_category_breakdown"] == {"device_command_timeout": 1}
    assert summary.failure_breakdown == {}
    assert summary.attempt_failure_breakdown == {"device_command_timeout": 1}


def test_telemetry_seed_preserves_retry_failure_categories_across_recorders(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchTaskState, BatchTaskStatus
    from northcua.telemetry.contracts import FailureCategory

    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    recorder.seed_batch_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            error="COMMAND_TIMED_OUT",
            last_failure_category=FailureCategory.DEVICE_COMMAND_TIMEOUT,
        )
    )
    recorder.record_batch_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED, attempts=2))

    TelemetryReportWriter(tmp_path).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert metric["status"] == "succeeded"
    assert metric["failure_category"] is None
    assert metric["attempt_failure_category_breakdown"] == {"device_command_timeout": 1}


def test_telemetry_report_includes_terminal_retryability(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchTaskState, BatchTaskStatus

    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    recorder.record_batch_state(
        BatchTaskState(
            task_id="case-1",
            status=BatchTaskStatus.FAILED,
            terminal_decision={
                "action": "fail",
                "reason": "max_steps",
                "retryable": False,
                "metadata": {"scope": "round", "limit": 2, "current": 2},
            },
        )
    )

    TelemetryReportWriter(tmp_path).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert metric["terminal_reason"] == "max_steps"
    assert metric["terminal_retryable"] is False
    assert metric["terminal_scope"] == "round"


def test_telemetry_report_includes_evaluator_safety_and_efficiency_metrics(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    sink.upsert_snapshot(
        CaseTelemetrySnapshot(
            batch_id="batch",
            case_id="case-1",
            status=CaseStatus.SUCCEEDED,
            metrics={
                "evaluator_safety_risk_level": "medium",
                "evaluator_safety_finding_count": 2,
                "evaluator_efficiency_rating": 3,
                "evaluator_diagnosis_category_breakdown": {
                    "grounding_localization_error": 1,
                    "environment_or_execution_error": 1,
                    "verification_termination_recovery_error": 2,
                },
                "evaluator_unnecessary_step_count": 4,
                "evaluator_repeated_action_count": 1,
                "evaluator_detour_count": 1,
            },
        )
    )

    summary = TelemetryReportWriter(tmp_path).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert metric["evaluator_safety_risk_level"] == "medium"
    assert metric["evaluator_safety_finding_count"] == 2
    assert metric["evaluator_efficiency_rating"] == 3
    assert metric["evaluator_diagnosis_category_breakdown"] == {
        "grounding_localization_error": 1,
        "environment_or_execution_error": 1,
        "verification_termination_recovery_error": 2,
    }
    assert summary.safety_risk_breakdown == {"medium": 1}
    assert summary.diagnosis_category_breakdown == {
        "grounding_localization_error": 1,
        "environment_or_execution_error": 1,
        "verification_termination_recovery_error": 2,
    }
    assert summary.average_efficiency_rating == 3


def test_report_uses_failed_phase_category_for_final_breakdown(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")
    with pytest.raises(RuntimeError):
        with recorder.phase_timer(case_id="case-1", phase=CasePhase.MAIN_AGENT, status=CaseStatus.RUNNING_AGENT):
            raise RuntimeError("agent failed")
    recorder.record_batch_state(_state("case-1", "failed"))

    summary = TelemetryReportWriter(tmp_path).write()

    assert summary.failure_breakdown == {"main_agent": 1}


def test_telemetry_report_total_seconds_uses_case_wall_clock(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    sink.upsert_snapshot(
        CaseTelemetrySnapshot(
            batch_id="batch",
            case_id="case-1",
            status=CaseStatus.SUCCEEDED,
            started_at="2026-05-05T00:00:00+00:00",
            updated_at="2026-05-05T00:02:00+00:00",
            finished_at="2026-05-05T00:02:00+00:00",
        )
    )
    sink.record_phase(
        PhaseTiming(
            batch_id="batch",
            case_id="case-1",
            phase=CasePhase.MAIN_AGENT,
            status=CaseStatus.RUNNING_AGENT,
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:02:00+00:00",
            duration_seconds=120.0,
        )
    )
    sink.record_phase(
        PhaseTiming(
            batch_id="batch",
            case_id="case-1",
            phase=CasePhase.EVALUATOR,
            status=CaseStatus.EVALUATING,
            started_at="2026-05-05T00:01:00+00:00",
            finished_at="2026-05-05T00:02:00+00:00",
            duration_seconds=60.0,
        )
    )

    summary = TelemetryReportWriter(tmp_path).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert metric["total_seconds"] == 120.0
    assert metric["main_agent_seconds"] == 120.0
    assert metric["evaluator_seconds"] == 60.0
    assert summary.p50_total_seconds == 120.0


def test_telemetry_report_preserves_artifact_projection_when_refreshing_state(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    eval_dir = artifact_dir / "evaluations" / "desktop_round_completion" / "round_0001"
    eval_dir.mkdir(parents=True)
    steps = artifact_dir / "steps.jsonl"
    llm = artifact_dir / "llm_trace.jsonl"
    screenshot = artifact_dir / "screen.png"
    report = eval_dir / "final_report.json"
    evidence = eval_dir / "evidence_manifest.json"
    eval_llm = eval_dir / "llm_trace.jsonl"
    steps.write_text('{"session_step":1}\n', encoding="utf-8")
    llm.write_text(
        '{"assistant_tool_calls":[{"name":"click"}],"usage":{"input_tokens":3,"output_tokens":4}}\n', encoding="utf-8"
    )
    screenshot.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRextra")
    report.write_text(
        '{"overall_verdict":"success","safety":{"risk_level":"low","findings":[]},"efficiency":{"rating":5},"diagnoses":[]}',
        encoding="utf-8",
    )
    evidence.write_text('{"refs":[]}', encoding="utf-8")
    eval_llm.write_text(
        '{"usage":{"input_tokens":5,"output_tokens":6},"tool_checks":[{"name":"client_screenshot"}]}\n',
        encoding="utf-8",
    )
    sink = LocalTelemetrySink(tmp_path)
    sink.upsert_snapshot(
        CaseTelemetrySnapshot(
            batch_id="batch",
            case_id="case-1",
            status=CaseStatus.SUCCEEDED,
            phase=CasePhase.COMPLETED,
            started_at="2026-05-05T00:00:00+00:00",
            updated_at="2026-05-05T00:02:00+00:00",
            finished_at="2026-05-05T00:02:00+00:00",
        )
    )
    TelemetryReportWriter(tmp_path).write(
        case_artifacts=[
            CaseArtifactTelemetry(
                case_id="case-1",
                main_trace_dir=str(artifact_dir),
                main_steps_path=str(steps),
                main_llm_trace_path=str(llm),
                evaluator_trace_dir=str(eval_dir),
                evaluator_report_path=str(report),
                evaluator_evidence_manifest_path=str(evidence),
                evaluator_llm_trace_path=str(eval_llm),
                screenshot_paths=[str(screenshot)],
            )
        ]
    )
    first_metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    TelemetryReportWriter(tmp_path).write()
    refreshed_metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert first_metric["artifact_count"] > 0
    assert refreshed_metric["artifact_count"] == first_metric["artifact_count"]
    assert refreshed_metric["evaluator_report_path"] == str(report)
    assert refreshed_metric["prompt_tokens"] == first_metric["prompt_tokens"]
    assert refreshed_metric["completion_tokens"] == 10


def test_artifact_token_usage_prefers_primary_zero_aliases(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    sink.upsert_snapshot(
        CaseTelemetrySnapshot(
            batch_id="batch",
            case_id="case-1",
            status=CaseStatus.SUCCEEDED,
            phase=CasePhase.COMPLETED,
        )
    )
    artifact_dir = tmp_path / "artifacts"
    eval_dir = artifact_dir / "evaluations" / "desktop_round_completion" / "round_0001"
    eval_dir.mkdir(parents=True)
    llm = artifact_dir / "llm_trace.jsonl"
    eval_llm = eval_dir / "llm_trace.jsonl"
    llm.write_text(
        '{"usage":{"input_tokens":0,"prompt_tokens":99,"completion_tokens":0,"output_tokens":88}}\n',
        encoding="utf-8",
    )
    eval_llm.write_text(
        '{"usage":{"input_tokens":1,"prompt_tokens":99,"completion_tokens":2,"output_tokens":88}}\n',
        encoding="utf-8",
    )

    TelemetryReportWriter(tmp_path).write(
        case_artifacts=[
            CaseArtifactTelemetry(
                case_id="case-1",
                main_llm_trace_path=str(llm),
                evaluator_llm_trace_path=str(eval_llm),
            )
        ]
    )
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert metric["prompt_tokens"] == 1
    assert metric["completion_tokens"] == 2


def test_snapshot_from_batch_state_sets_terminal_total_seconds() -> None:
    from northcua.batch.contracts import BatchTaskState, BatchTaskStatus
    from northcua.telemetry.recorder import snapshot_from_batch_state

    previous = CaseTelemetrySnapshot(
        batch_id="batch",
        case_id="case-1",
        started_at="2026-05-05T00:00:00+00:00",
        updated_at="2026-05-05T00:00:00+00:00",
    )

    snapshot = snapshot_from_batch_state(
        BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED, attempts=1),
        batch_id="batch",
        previous=previous,
        observed_at="2026-05-05T00:02:00+00:00",
    )

    assert snapshot.finished_at == "2026-05-05T00:02:00+00:00"
    assert snapshot.total_seconds == 120.0


def test_sqlite_report_does_not_extend_elapsed_with_unattempted_pending(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState, BatchTaskStatus
    from northcua.batch.state import SQLiteBatchStateRepository

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(
        BatchManifest(
            batch_id="batch",
            tasks=[
                BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="run"),
                BatchTaskSpec(task_id="case-2", agent_kind="desktop", query="pending"),
            ],
        )
    )
    assert repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED, attempts=1))
    (tmp_path / "case_snapshots.jsonl").write_text(
        json.dumps(
            {
                "batch_id": "batch",
                "case_id": "case-1",
                "status": "succeeded",
                "phase": "completed",
                "attempt": 1,
                "started_at": "2026-05-05T00:00:00+00:00",
                "updated_at": "2026-05-05T00:05:00+00:00",
                "finished_at": "2026-05-05T00:05:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = TelemetryReportWriter(tmp_path, state_repository=repository).write()

    assert summary.elapsed_seconds == 300.0
    assert summary.throughput_cases_per_hour == 12.0


def test_telemetry_report_prefers_sqlite_state_over_stale_snapshot(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState, BatchTaskStatus
    from northcua.batch.state import SQLiteBatchStateRepository

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(
        BatchManifest(batch_id="batch", tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="run")])
    )
    assert repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.FAILED, attempts=1))
    (tmp_path / "case_snapshots.jsonl").write_text(
        json.dumps(
            {
                "batch_id": "batch",
                "case_id": "case-1",
                "status": "succeeded",
                "phase": "completed",
                "attempt": 1,
                "started_at": "2026-05-05T00:00:00+00:00",
                "updated_at": "2026-05-05T00:00:01+00:00",
                "finished_at": "2026-05-05T00:00:01+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = TelemetryReportWriter(tmp_path, state_repository=repository).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert metric["status"] == "failed"
    assert summary.failed == 1
    assert summary.succeeded == 0


def test_sqlite_report_preserves_snapshot_derived_metrics(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState, BatchTaskStatus
    from northcua.batch.state import SQLiteBatchStateRepository

    repository = SQLiteBatchStateRepository(tmp_path / "batch_state.sqlite")
    repository.initialize_manifest(
        BatchManifest(batch_id="batch", tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="run")])
    )
    assert repository.upsert_state(BatchTaskState(task_id="case-1", status=BatchTaskStatus.FAILED, attempts=1))
    (tmp_path / "case_snapshots.jsonl").write_text(
        json.dumps(
            {
                "batch_id": "batch",
                "case_id": "case-1",
                "status": "succeeded",
                "phase": "completed",
                "attempt": 1,
                "started_at": "2026-05-05T00:00:00+00:00",
                "updated_at": "2026-05-05T00:00:01+00:00",
                "finished_at": "2026-05-05T00:00:01+00:00",
                "metrics": {
                    "evaluator_safety_risk_level": "medium",
                    "evaluator_efficiency_rating": 4,
                    "evaluator_diagnosis_category_breakdown": {"plan_decomposition_error": 1},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    TelemetryReportWriter(tmp_path, state_repository=repository).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert metric["status"] == "failed"
    assert metric["evaluator_safety_risk_level"] == "medium"
    assert metric["evaluator_efficiency_rating"] == 4
    assert metric["evaluator_diagnosis_category_breakdown"] == {"plan_decomposition_error": 1}


def test_telemetry_report_skips_invalid_snapshot_and_timing_rows(tmp_path: Path) -> None:
    (tmp_path / "case_snapshots.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"batch_id": "batch", "case_id": "bad", "status": "not-a-status", "phase": "completed"}),
                json.dumps(
                    {
                        "batch_id": "batch",
                        "case_id": "case-1",
                        "status": "succeeded",
                        "phase": "completed",
                        "attempt": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "case_timelines.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"batch_id": "batch", "case_id": "case-1", "phase": "not-a-phase"}),
                json.dumps(
                    {
                        "batch_id": "batch",
                        "case_id": "case-1",
                        "phase": "main_agent",
                        "status": "succeeded",
                        "started_at": "2026-05-05T00:00:00+00:00",
                        "finished_at": "2026-05-05T00:00:02+00:00",
                        "duration_seconds": 2.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = TelemetryReportWriter(tmp_path).write()
    metric = _jsonl(tmp_path / "case_metrics.jsonl")[0]

    assert summary.total_cases == 1
    assert metric["case_id"] == "case-1"
    assert metric["main_agent_seconds"] == 2.0


def test_generic_case_event_updates_snapshot(tmp_path: Path) -> None:
    sink = LocalTelemetrySink(tmp_path)
    recorder = CaseTelemetryRecorder(sink=sink, batch_id="batch")

    recorder.record_case_event(
        case_id="case-1",
        entity_kind=TelemetryEntityKind.SANDBOX,
        entity_id="sandbox-1",
        phase=CasePhase.WORKER_READY,
        status=CaseStatus.BOOTSTRAPPING,
        message="device_ready",
        details={"token": "secret"},
        sandbox_id="sandbox-1",
    )

    snapshot = _jsonl(tmp_path / "case_snapshots.jsonl")[-1]
    event = _jsonl(tmp_path / "telemetry_events.jsonl")[-1]
    assert snapshot["phase"] == "worker_ready"
    assert snapshot["sandbox_id"] == "sandbox-1"
    assert event["event_id"] == 1
    assert event["details"]["token"] == "<redacted>"


def _state(task_id: str, status: str):
    from northcua.batch.contracts import BatchTaskState, BatchTaskStatus

    return BatchTaskState(task_id=task_id, status=BatchTaskStatus(status), attempts=1)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
