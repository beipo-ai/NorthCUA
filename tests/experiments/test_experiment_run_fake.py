"""Fake-provider integration coverage for the golden experiment suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from northcua.batch.contracts import (
    BatchManifest,
    BatchRunConfig,
    BatchRunResult,
    BatchTaskSpec,
    BatchTaskState,
    BatchTaskStatus,
)
from northcua.batch.runner import BatchRunner
from northcua.batch.state import SQLiteBatchStateRepository
from northcua.experiments.contracts import (
    ExperimentAppProfile,
    ExperimentControllerSpec,
    ExperimentRunConfig,
    ExperimentSpec,
    ExperimentTaskResult,
    ExperimentTaskSpec,
    SandboxFileInjectionSpec,
    TraceExpectationSpec,
)
from northcua.experiments.materialization import MaterializedExperiment
from northcua.experiments.orchestration import ExperimentOperations
from northcua.experiments.orchestration.operations_runtime import ExperimentOperationsRuntimeMixin
from northcua.experiments.planner import ExperimentBatchPlanner
from tests.support.experiment_fixtures import golden_e2b_experiment_spec, run_fixture_experiment
from northcua.experiments.reporting.telemetry import enrich_telemetry_reports
from northcua.reliability.contracts import FailureAction
from northcua.sandboxes.contracts import SandboxProviderSpec, SandboxResourceSpec
from northcua.sandboxes.provider.fake import FakeSandboxProvider
from northcua.sandboxes.provider.registry import SandboxProviderRegistry
from northcua_protocol.capabilities import Capability, PlatformKind
from northcua_protocol.payloads import TaskAccepted, TaskFinished
from northcua_protocol.state import TaskState


class _GoldenSubmitter:
    """Submitter double that immediately finishes deterministic batch tasks."""

    submitted_batch_task_ids: list[str] = []

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def __aenter__(self) -> "_GoldenSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        metadata = kwargs["metadata"]
        assert isinstance(metadata, dict)
        self.submitted_batch_task_ids.append(str(metadata["batch_task_id"]))
        task_id = str(kwargs["task_id"])
        trace_id = str(kwargs["trace_id"])
        return TaskAccepted(task_id=task_id, trace_id=trace_id, session_id=f"session-{metadata['batch_task_id']}")

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        return TaskFinished(
            task_id=task_id,
            trace_id=f"trace-for-{task_id}",
            session_id="session",
            state=TaskState.FINISHED,
            device_id="device",
            lease_id="lease",
            result={"ok": True},
        )


def test_experiment_runtime_counts_evaluation_pending_as_terminal_failure(tmp_path: Path) -> None:
    states = [
        BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED),
        BatchTaskState(task_id="case-2", status=BatchTaskStatus.EVALUATION_PENDING, evaluation_status="pending"),
        BatchTaskState(task_id="case-3", status=BatchTaskStatus.PENDING),
    ]

    result = ExperimentOperationsRuntimeMixin._batch_result_from_states("batch", tmp_path, states)  # noqa: SLF001

    assert result.succeeded == 1
    assert result.failed == 1
    assert result.incomplete == 1


@pytest.mark.asyncio
async def test_golden_fake_run_creates_three_sandboxes_and_reports(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    _GoldenSubmitter.submitted_batch_task_ids = []

    operations = ExperimentOperations(
        batch_runner_factory=lambda: BatchRunner(
            provider_registry=registry,
            submitter_factory=lambda url: _GoldenSubmitter(url),
        )
    )

    result = await run_fixture_experiment(
        operations,
        public_controller_url="ws://127.0.0.1:8885",
        output_dir=tmp_path,
        provider="fake",
        require_real_e2b=False,
    )

    assert result.batch_result.succeeded == 3
    assert all(row.passed for row in result.task_results)
    assert sorted(_GoldenSubmitter.submitted_batch_task_ids) == [
        "calc-total-csv",
        "chrome-fixture-heading",
        "gimp-open-image",
    ]
    assert len(provider.handles) == 3
    assert len(provider.uploads) == 3
    assert Path(result.dataset_index_path).exists()
    assert Path(result.artifact_index_path).exists()
    assert Path(result.assertions_path).exists()
    assert Path(result.telemetry_summary_path).exists()
    assert Path(result.case_metrics_path).exists()
    assert Path(result.audit_report_path).exists()
    metrics_rows = [
        json.loads(line)
        for line in Path(result.case_metrics_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["evaluator_safety_risk_level"] for row in metrics_rows} == {"none"}
    assert {row["evaluator_efficiency_rating"] for row in metrics_rows} == {5}
    dataset_rows = [
        json.loads(line)
        for line in Path(result.dataset_index_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(dataset_rows) == 3
    assert all(row["query"] for row in dataset_rows)
    assert all(row["main_steps_path"] for row in dataset_rows)
    assert all(row["evaluator_report_path"] for row in dataset_rows)
    assert all(row["evaluator_evidence_manifest_path"] for row in dataset_rows)
    assert all(row["fixture_remote_paths"] for row in dataset_rows)
    assertions = json.loads(Path(result.assertions_path).read_text(encoding="utf-8"))
    assert assertions["passed"] is True
    assert "average efficiency rating: 5.000" in Path(result.audit_report_path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_experiment_run_builds_runtime_bundle_before_planning_and_batch_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    runner = BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: _GoldenSubmitter(url),
    )
    planner = ExperimentBatchPlanner()
    operations = ExperimentOperations(batch_runner_factory=lambda: runner, planner=planner)

    original_runtime_bundle = operations._with_runtime_bundle
    original_plan = planner.plan
    original_batch_run = runner.run

    def _record_runtime_bundle(
        materialized: MaterializedExperiment,
        *,
        dry_run: bool,
    ) -> MaterializedExperiment:
        bundled = original_runtime_bundle(materialized, dry_run=dry_run)
        events.append("runtime_bundle")
        return bundled

    def _record_plan(experiment: MaterializedExperiment, *, controller_url: str) -> BatchManifest:
        events.append("planner")
        return original_plan(experiment, controller_url=controller_url)

    async def _record_batch_run(config: BatchRunConfig) -> BatchRunResult:
        events.append("batch_run")
        return await original_batch_run(config)

    monkeypatch.setattr(operations, "_with_runtime_bundle", _record_runtime_bundle)
    monkeypatch.setattr(planner, "plan", _record_plan)
    monkeypatch.setattr(runner, "run", _record_batch_run)

    result = await run_fixture_experiment(
        operations,
        public_controller_url="ws://127.0.0.1:8885",
        output_dir=tmp_path,
        provider="fake",
        require_real_e2b=False,
    )

    assert result.batch_result.succeeded == 3
    assert events == ["runtime_bundle", "planner", "batch_run"]


@pytest.mark.asyncio
async def test_web_agent_evaluator_three_case_fake_run_produces_web_artifacts_and_qwen_traces(
    tmp_path: Path,
) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    _GoldenSubmitter.submitted_batch_task_ids = []
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for name in ("info.html", "form.html", "tabs.html"):
        (fixtures / name).write_text(f"<!doctype html><title>{name}</title><main>{name}</main>", encoding="utf-8")
    spec = ExperimentSpec(
        experiment_id="web-agent-e2b-three-fake",
        description="Fake-provider Web Agent + evaluator shape test.",
        controller=ExperimentControllerSpec(
            public_url="ws://127.0.0.1:8885",
            app_config_path=Path("config/app.web.e2b.yaml"),
            insecure_dev_no_auth=True,
        ),
        sandbox=SandboxProviderSpec(
            provider="fake",
            template="fixture-web-base",
            desktop=True,
            platform=PlatformKind.LINUX,
            resources=SandboxResourceSpec(timeout_seconds=12000, cpu_count=2, memory_mb=4096),
            metadata={"suite": "web-agent-e2b-three-fake"},
        ),
        concurrency=3,
        trace_expectations=TraceExpectationSpec(evaluator_kind="web_round_evaluator"),
        tasks=[
            _web_fake_task("web-info-lookup", fixtures / "info.html", "http://127.0.0.1:8761/index.html"),
            _web_fake_task("web-form-state", fixtures / "form.html", "http://127.0.0.1:8762/index.html"),
            _web_fake_task("web-tabs-navigation", fixtures / "tabs.html", "http://127.0.0.1:8763/index.html"),
        ],
    )
    operations = ExperimentOperations(
        batch_runner_factory=lambda: BatchRunner(
            provider_registry=registry,
            submitter_factory=lambda url: _GoldenSubmitter(url),
        )
    )

    result = await operations.run(
        ExperimentRunConfig(
            spec=spec,
            output_dir=tmp_path,
            public_controller_url="ws://127.0.0.1:8885",
            require_real_e2b=False,
        )
    )

    assert result.batch_result.succeeded == 3
    assert all(row.passed for row in result.task_results)
    assert {task.agent_kind for task in spec.tasks} == {"web"}
    assert sorted(_GoldenSubmitter.submitted_batch_task_ids) == [
        "web-form-state",
        "web-info-lookup",
        "web-tabs-navigation",
    ]
    for row in result.task_results:
        main_dir = Path(row.main_trace_dir)
        eval_dir = Path(row.evaluator_trace_dir)
        assert (main_dir / "web_agent_trace.jsonl").is_file()
        assert next((main_dir / "dom_snapshots").glob("*.txt")).is_file()
        assert Path(row.main_llm_trace_path).is_file()
        assert Path(row.evaluator_llm_trace_path).is_file()
        assert Path(row.evaluator_report_path).is_file()
        assert eval_dir.name == "round_0001"
    assert Path(result.telemetry_summary_path).exists()
    assert Path(result.summary_path).exists()


def _web_fake_task(task_id: str, source_path: Path, url: str) -> ExperimentTaskSpec:
    return ExperimentTaskSpec(
        task_id=task_id,
        app_profile=ExperimentAppProfile.CHROME,
        agent_kind="web",
        query=f"Complete {task_id}.",
        max_steps=12,
        fixtures=[
            SandboxFileInjectionSpec(
                source_path=source_path,
                remote_path=f"/tmp/northcua-web-e2b-three/{task_id}/index.html",
                mode="0644",
            )
        ],
        required_capabilities={Capability.BROWSER_PAGE},
        metadata={"url": url, "web_case_kind": task_id.replace("web-", "")},
    )


def test_experiment_telemetry_enrichment_aggregates_trace_usage_and_rewrites_audit(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    main_trace = tmp_path / "session" / "llm_trace.jsonl"
    eval_dir = tmp_path / "session" / "evaluations" / "desktop_round_completion" / "round_0001"
    eval_trace = eval_dir / "llm_trace.jsonl"
    final_report = eval_dir / "final_report.json"
    screenshot = tmp_path / "session" / "screenshot.png"
    main_trace.parent.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    screenshot.write_bytes(b"png")
    main_trace.write_text(
        json.dumps(
            {
                "llm_trace": {
                    "usage": {"input_tokens": 10, "completion_tokens": 3},
                    "assistant_tool_calls": [{"name": "click"}],
                    "tool_executions": [{"tool_name": "click"}],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    eval_trace.write_text(
        json.dumps(
            {
                "usage": {"input_tokens": 20, "completion_tokens": 7},
                "tool_checks": [{"name": "final_report", "status": "success"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_dir / "runtime.jsonl").write_text(
        json.dumps({"event": "agentkit_final_report_completed", "duration_ms": 1250}) + "\n",
        encoding="utf-8",
    )
    final_report.write_text(
        json.dumps(
            {
                "safety": {"risk_level": "none", "findings": []},
                "diagnoses": [
                    {
                        "category": "plan_decomposition_error",
                        "code": "detour",
                        "severity": "minor",
                        "reason": "The synthetic report records one plan diagnostic.",
                        "evidence_refs": [],
                    },
                    {
                        "category": "environment_or_execution_error",
                        "code": "slow_widget",
                        "severity": "minor",
                        "reason": "The synthetic report records one environment diagnostic.",
                        "evidence_refs": [],
                    },
                ],
                "efficiency": {
                    "rating": 4,
                    "unnecessary_step_count": 1,
                    "repeated_action_count": 0,
                    "detour_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (batch_dir / "case_metrics.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": "experiment",
                "batch_id": "batch",
                "case_id": "case-1",
                "status": "succeeded",
                "retry_count": 0,
                "total_seconds": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (batch_dir / "case_snapshots.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": "experiment",
                "batch_id": "batch",
                "case_id": "case-1",
                "status": "succeeded",
                "phase": "completed",
                "attempt": 1,
                "started_at": "2026-05-05T00:00:00+00:00",
                "updated_at": "2026-05-05T00:00:02+00:00",
                "finished_at": "2026-05-05T00:00:02+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (batch_dir / "telemetry_summary.json").write_text(
        json.dumps(
            {
                "batch_id": "batch",
                "total_cases": 1,
                "succeeded": 1,
                "failed": 0,
                "timed_out": 0,
                "canceled": 0,
                "success_rate": 1.0,
                "p50_total_seconds": 2.0,
                "p95_total_seconds": 2.0,
                "cleanup_failures": 0,
                "retry_summary": {},
            }
        ),
        encoding="utf-8",
    )
    (batch_dir / "audit_report.md").write_text("- average efficiency rating: 0.000\n", encoding="utf-8")
    (batch_dir / "batch_summary.json").write_text(
        json.dumps(
            {
                "batch_id": "batch",
                "telemetry": {
                    "summary_path": str(batch_dir / "telemetry_summary.json"),
                    "summary": {
                        "artifact_count": 0,
                        "artifact_bytes": 0,
                        "average_efficiency_rating": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    enrich_telemetry_reports(
        batch_output_dir=batch_dir,
        task_results=[
            ExperimentTaskResult(
                task_id="case-1",
                app_profile=ExperimentAppProfile.CHROME,
                status="succeeded",
                main_llm_trace_path=str(main_trace),
                screenshot_paths=[str(screenshot)],
                evaluator_report_path=str(final_report),
                evaluator_llm_trace_path=str(eval_trace),
            )
        ],
    )

    metric = json.loads((batch_dir / "case_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads((batch_dir / "telemetry_summary.json").read_text(encoding="utf-8"))
    batch_summary = json.loads((batch_dir / "batch_summary.json").read_text(encoding="utf-8"))
    audit = (batch_dir / "audit_report.md").read_text(encoding="utf-8")
    assert metric["prompt_tokens"] == 30
    assert metric["completion_tokens"] == 10
    assert metric["tool_call_count"] == 2
    assert metric["command_count"] == 1
    assert metric["evaluator_seconds"] == 1.25
    assert metric["main_trace_dir"] == str(main_trace.parent)
    assert metric["main_llm_trace_path"] == str(main_trace)
    assert metric["evaluator_trace_dir"] == str(eval_dir)
    assert metric["evaluator_report_path"] == str(final_report)
    assert metric["evaluator_llm_trace_path"] == str(eval_trace)
    assert metric["evaluator_diagnosis_category_breakdown"] == {
        "environment_or_execution_error": 1,
        "plan_decomposition_error": 1,
    }
    assert summary["average_efficiency_rating"] == 4
    assert batch_summary["telemetry"]["summary"]["average_efficiency_rating"] == 4
    assert batch_summary["telemetry"]["summary"]["artifact_count"] == summary["artifact_count"]
    assert batch_summary["telemetry"]["summary"]["artifact_bytes"] == summary["artifact_bytes"]
    assert summary["diagnosis_category_breakdown"] == {
        "environment_or_execution_error": 1,
        "plan_decomposition_error": 1,
    }
    assert "average efficiency rating: 4.000" in audit
    assert "environment_or_execution_error" in audit
    assert "plan_decomposition_error:1" in audit
    assert "30/10" in audit


def test_experiment_telemetry_enrichment_preserves_report_when_authoritative_source_missing(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    original_metrics = json.dumps(
        {
            "experiment_id": "experiment",
            "batch_id": "batch",
            "case_id": "case-1",
            "status": "succeeded",
            "retry_count": 0,
            "total_seconds": 2.0,
        }
    )
    (batch_dir / "case_metrics.jsonl").write_text(original_metrics + "\n", encoding="utf-8")
    (batch_dir / "telemetry_summary.json").write_text(json.dumps({"total_cases": 1}), encoding="utf-8")

    enrich_telemetry_reports(
        batch_output_dir=batch_dir,
        task_results=[
            ExperimentTaskResult(
                task_id="case-1",
                app_profile=ExperimentAppProfile.CHROME,
                status="succeeded",
            )
        ],
    )

    assert (batch_dir / "case_metrics.jsonl").read_text(encoding="utf-8") == original_metrics + "\n"
    assert json.loads((batch_dir / "telemetry_summary.json").read_text(encoding="utf-8")) == {"total_cases": 1}


def test_experiment_telemetry_uses_verified_evaluator_metrics_when_report_missing(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (batch_dir / "case_metrics.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": "experiment",
                "batch_id": "batch",
                "case_id": "case-1",
                "status": "succeeded",
                "retry_count": 0,
                "total_seconds": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (batch_dir / "case_snapshots.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": "experiment",
                "batch_id": "batch",
                "case_id": "case-1",
                "status": "succeeded",
                "phase": "completed",
                "attempt": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    enrich_telemetry_reports(
        batch_output_dir=batch_dir,
        task_results=[
            ExperimentTaskResult(
                task_id="case-1",
                app_profile=ExperimentAppProfile.CHROME,
                status="succeeded",
                evaluator_safety_risk_level="low",
                evaluator_efficiency_rating=4,
                evaluator_diagnosis_category_breakdown={"environment_or_execution_error": 1},
            )
        ],
    )

    metric = json.loads((batch_dir / "case_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads((batch_dir / "telemetry_summary.json").read_text(encoding="utf-8"))
    assert metric["evaluator_safety_risk_level"] == "low"
    assert metric["evaluator_efficiency_rating"] == 4
    assert metric["evaluator_diagnosis_category_breakdown"] == {"environment_or_execution_error": 1}
    assert summary["average_efficiency_rating"] == 4


@pytest.mark.asyncio
async def test_experiment_artifacts_do_not_persist_controller_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    secret = "test-secret-controller-token"
    monkeypatch.setenv("NORTHCUA_CONTROLLER_TOKEN", secret)

    operations = ExperimentOperations(
        batch_runner_factory=lambda: BatchRunner(
            provider_registry=registry,
            submitter_factory=lambda url: _GoldenSubmitter(url),
        )
    )

    result = await run_fixture_experiment(
        operations,
        public_controller_url="ws://127.0.0.1:8885",
        output_dir=tmp_path,
        provider="fake",
        require_real_e2b=False,
    )

    experiment_dir = Path(result.output_dir)
    leaked_paths = [
        path
        for path in experiment_dir.rglob("*")
        if path.suffix in {".json", ".jsonl"} and secret in path.read_text(encoding="utf-8")
    ]
    assert leaked_paths == []


def test_experiment_status_reads_live_batch_state_before_summary(tmp_path: Path) -> None:
    experiment_id = "golden-live-status"
    batch_dir = tmp_path / experiment_id / "batch" / experiment_id
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(
        BatchManifest(
            batch_id=experiment_id,
            tasks=[
                BatchTaskSpec(task_id="running-task", agent_kind="desktop", query="run"),
                BatchTaskSpec(task_id="finished-task", agent_kind="desktop", query="done"),
            ],
        )
    )
    assert repository.upsert_state(BatchTaskState(task_id="running-task", status=BatchTaskStatus.RUNNING))
    assert repository.upsert_state(BatchTaskState(task_id="finished-task", status=BatchTaskStatus.SUCCEEDED))

    status = ExperimentOperations().status(experiment_id=experiment_id, output_dir=tmp_path)

    assert status["status"] == "running"
    assert status["total"] == 2
    assert status["running"] == 1
    assert status["succeeded"] == 1
    assert status["batch_state_path"] == str(batch_dir / "batch_state.sqlite")
