"""Fake-provider integration coverage for the golden experiment suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from northcua.batch.contracts import (
    BatchManifest,
    BatchRunConfig,
    BatchRunResult,
    BatchRunResumeConfig,
    BatchTaskSpec,
    BatchTaskState,
    BatchTaskStatus,
)
from northcua.batch.preparation import PreparedSandboxPolicy
from northcua.batch.runner import BatchRunner
from northcua.batch.state import SQLiteBatchStateRepository
from northcua.experiments.contracts import ExperimentAppProfile, ExperimentTaskResult
from northcua.experiments.orchestration import ExperimentOperations
from tests.support.experiment_fixtures import golden_e2b_experiment_spec, run_fixture_experiment
from northcua.experiments.reporting.telemetry import enrich_telemetry_reports
from northcua.reliability.contracts import FailureAction
from northcua.sandboxes.provider.fake import FakeSandboxProvider
from northcua.sandboxes.provider.registry import SandboxProviderRegistry
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


class _PassingVerifier:
    def verify(self, **kwargs: object) -> list[ExperimentTaskResult]:
        return [
            ExperimentTaskResult(
                task_id="case-1",
                app_profile=ExperimentAppProfile.CHROME,
                status=BatchTaskStatus.SUCCEEDED.value,
                passed=True,
            )
        ]


class _CapturingBatchRunner:
    def __init__(self) -> None:
        self.configs: list[BatchRunConfig] = []

    async def run(self, config: BatchRunConfig) -> BatchRunResult:
        self.configs.append(config)
        state = BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED)
        return BatchRunResult(
            batch_id=config.manifest.batch_id,
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            incomplete=0,
            output_dir=str(config.output_dir / config.manifest.batch_id),
            task_states=[state],
        )


def test_experiment_recover_plan_is_read_only_for_materialized_fallback(tmp_path: Path) -> None:
    experiment_id = "read-only-plan"
    experiment_dir = tmp_path / experiment_id
    batch_dir = experiment_dir / "batch" / experiment_id
    spec = golden_e2b_experiment_spec(public_controller_url="ws://127.0.0.1:8885", provider="fake").model_copy(
        update={"experiment_id": experiment_id}
    )
    manifest = BatchManifest(
        batch_id=experiment_id,
        tasks=[
            BatchTaskSpec(task_id=task.task_id, agent_kind=task.agent_kind, query=task.query) for task in spec.tasks
        ],
    )
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "experiment_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (experiment_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)

    ExperimentOperations().recover_plan(experiment_id=experiment_id, output_dir=tmp_path)

    assert not (experiment_dir / "materialized").exists()


@pytest.mark.asyncio
async def test_experiment_recover_plan_identifies_evaluator_only_recovery(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
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
    target = result.task_results[0]
    Path(target.evaluator_llm_trace_path).unlink()

    plan = operations.recover_plan(experiment_id=result.experiment_id, output_dir=tmp_path)

    item = next(item for item in plan.items if item.task_id == target.task_id)
    assert item.action == FailureAction.RETRY_EVALUATOR
    assert item.evaluator_only is True
    assert item.runnable is False


@pytest.mark.asyncio
async def test_experiment_pause_drain_cancel_update_batch_controls(tmp_path: Path) -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
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
    sqlite_path = tmp_path / result.experiment_id / "batch" / result.experiment_id / "batch_state.sqlite"

    assert operations.pause(experiment_id=result.experiment_id, output_dir=tmp_path) == sqlite_path
    assert SQLiteBatchStateRepository(sqlite_path).control_flags()["paused"] is True

    assert operations.drain(experiment_id=result.experiment_id, output_dir=tmp_path) == sqlite_path
    flags = SQLiteBatchStateRepository(sqlite_path).control_flags()
    assert flags["paused"] is True
    assert flags["drain_requested"] is True

    cancel_path = operations.cancel(experiment_id=result.experiment_id, output_dir=tmp_path)
    assert cancel_path.exists()


@pytest.mark.asyncio
async def test_experiment_resume_preserves_persisted_runtime_config_when_cli_omits_overrides(
    tmp_path: Path,
) -> None:
    experiment_id = "resume-keeps-config"
    experiment_dir = tmp_path / experiment_id
    batch_dir = experiment_dir / "batch" / experiment_id
    spec = golden_e2b_experiment_spec(public_controller_url="ws://public-original", provider="fake").model_copy(
        update={
            "experiment_id": experiment_id,
            "concurrency": 3,
            "tasks": [
                golden_e2b_experiment_spec(
                    public_controller_url="ws://public-original",
                    provider="fake",
                )
                .tasks[0]
                .model_copy(update={"task_id": "case-1"}),
            ],
        }
    )
    manifest = BatchManifest(
        batch_id=experiment_id,
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "experiment_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (experiment_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite").initialize_manifest(manifest)
    persisted_policy = PreparedSandboxPolicy(enabled=True, buffer=55, prepare_concurrency=13, ttl_seconds=7200)
    (batch_dir / "run_config.json").write_text(
        BatchRunResumeConfig(
            controller_url="ws://persisted-submit",
            device_controller_url="ws://persisted-device",
            concurrency=77,
            sandbox_create_concurrency=11,
            controller_submit_concurrency=22,
            controller_status_concurrency=7,
            controller_submit_jitter_seconds=3.5,
            device_readiness_timeout_seconds=456,
            infra_retry_backoff_seconds=[12.0, 34.0],
            telemetry_experiment_id=experiment_id,
            telemetry_controller_db_path=experiment_dir / "controller.sqlite",
            prepared_sandbox_policy=persisted_policy,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    runner = _CapturingBatchRunner()
    operations = ExperimentOperations(
        batch_runner_factory=lambda: runner,  # type: ignore[return-value]
        verifier=_PassingVerifier(),  # type: ignore[arg-type]
    )

    result = await operations.resume(experiment_id=experiment_id, output_dir=tmp_path)

    assert result.batch_result.succeeded == 1
    assert len(runner.configs) == 1
    config = runner.configs[0]
    assert config.controller_url == "ws://persisted-submit"
    assert config.device_controller_url == "ws://persisted-device"
    assert config.concurrency == 77
    assert config.sandbox_create_concurrency == 11
    assert config.controller_submit_concurrency == 22
    assert config.controller_status_concurrency == 7
    assert config.controller_submit_jitter_seconds == 3.5
    assert config.device_readiness_timeout_seconds == 456
    assert config.infra_retry_backoff_seconds == [12.0, 34.0]
    assert config.telemetry_experiment_id == experiment_id
    assert config.prepared_sandbox_policy.enabled is True
    assert config.prepared_sandbox_policy.buffer == 55
    assert config.prepared_sandbox_policy.prepare_concurrency == 13


@pytest.mark.asyncio
async def test_experiment_resume_validates_persisted_batch_config_before_clearing_control_flags(
    tmp_path: Path,
) -> None:
    experiment_id = "resume-keeps-control-flags"
    experiment_dir = tmp_path / experiment_id
    batch_dir = experiment_dir / "batch" / experiment_id
    spec = golden_e2b_experiment_spec(public_controller_url="ws://public-original", provider="fake").model_copy(
        update={
            "experiment_id": experiment_id,
            "tasks": [
                golden_e2b_experiment_spec(
                    public_controller_url="ws://public-original",
                    provider="fake",
                )
                .tasks[0]
                .model_copy(update={"task_id": "case-1"}),
            ],
        }
    )
    manifest = BatchManifest(
        batch_id=experiment_id,
        tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Do it")],
    )
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "experiment_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (experiment_dir / "batch_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    repository.set_control_flag("paused", True)
    repository.set_control_flag("drain_requested", True)
    (batch_dir / "run_config.json").write_text(
        BatchRunResumeConfig(
            controller_url="ws://public-original",
            device_controller_url="ws://public-original",
            concurrency=3,
            telemetry_experiment_id=experiment_id,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=55),
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    runner = _CapturingBatchRunner()
    operations = ExperimentOperations(
        batch_runner_factory=lambda: runner,  # type: ignore[return-value]
        verifier=_PassingVerifier(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="prepared_sandbox_policy"):
        await operations.resume(
            experiment_id=experiment_id,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=False),
        )

    flags = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite").control_flags()
    assert flags["paused"] is True
    assert flags["drain_requested"] is True
    assert runner.configs == []
