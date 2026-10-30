"""Batch runner service tests."""

from __future__ import annotations

import asyncio
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
from northcua.batch.operations import BatchOperations
from northcua.batch.preparation import PreparedSandboxDepth, PreparedSandboxPolicy
from northcua.batch.runner import BatchRunner
from northcua.batch.state import SQLiteBatchStateRepository
from northcua.sandboxes.contracts import SandboxAllocationStrategy, SandboxCommandSpec
from northcua.sandboxes.provider.registry import SandboxProviderRegistry
from northcua.telemetry.contracts import FailureCategory
from northcua_protocol.payloads import TaskAccepted, TaskFinished
from northcua_protocol.state import TaskState


class FakeSubmitter:
    """Submitter double that immediately finishes accepted tasks."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self.accepted: TaskAccepted | None = None

    async def __aenter__(self) -> "FakeSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        self.accepted = TaskAccepted(task_id="controller-task", trace_id="trace", session_id="session")
        return self.accepted

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        return TaskFinished(
            task_id=task_id,
            trace_id="trace",
            session_id="session",
            state=TaskState.FINISHED,
            device_id="device",
            lease_id="lease",
            result={"ok": True},
        )


class RecordingRunner:
    """Runner double used by operations tests to inspect reconstructed config."""

    configs: list[BatchRunConfig] = []

    async def run(self, config: BatchRunConfig):
        self.configs.append(config)
        return BatchRunResult(
            batch_id=config.manifest.batch_id,
            total=0,
            succeeded=0,
            failed=0,
            canceled=0,
            output_dir=str(config.output_dir / config.manifest.batch_id),
            task_states=[],
        )


class CountingSubmitter:
    """Submitter double that records deterministic task ids."""

    submitted_task_ids: list[str] = []
    submitted_batch_task_ids: list[str] = []

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def __aenter__(self) -> "CountingSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        task_id = str(kwargs["task_id"])
        trace_id = str(kwargs["trace_id"])
        metadata = kwargs["metadata"]
        assert isinstance(metadata, dict)
        self.submitted_task_ids.append(task_id)
        self.submitted_batch_task_ids.append(str(metadata["batch_task_id"]))
        return TaskAccepted(task_id=task_id, trace_id=trace_id, session_id=f"session-{task_id}")

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        return TaskFinished(
            task_id=task_id,
            trace_id=f"trace-for-{task_id}",
            session_id=f"session-{task_id}",
            state=TaskState.FINISHED,
            device_id="device",
            lease_id="lease",
            result={"ok": True},
        )


class FailingSubmitter:
    """Submitter double that deterministically fails accepted tasks."""

    submitted_batch_task_ids: list[str] = []

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def __aenter__(self) -> "FailingSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        metadata = kwargs["metadata"]
        assert isinstance(metadata, dict)
        self.submitted_batch_task_ids.append(str(metadata["batch_task_id"]))
        task_id = str(kwargs["task_id"])
        return TaskAccepted(task_id=task_id, trace_id=str(kwargs["trace_id"]), session_id=f"session-{task_id}")

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        return TaskFinished(
            task_id=task_id,
            trace_id=f"trace-for-{task_id}",
            session_id=f"session-{task_id}",
            state=TaskState.FAILED,
            device_id="device",
            lease_id="lease",
            error="intentional failure",
        )


class EvaluatorFailingSubmitter(FailingSubmitter):
    """Submitter double that reports an evaluator-scoped terminal failure."""

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        return TaskFinished(
            task_id=task_id,
            trace_id=f"trace-for-{task_id}",
            session_id=f"session-{task_id}",
            state=TaskState.FAILED,
            device_id="device",
            lease_id="lease",
            error="Evaluator model completed without calling final_report.",
            terminal_decision={
                "action": "error",
                "reason": "runtime_error",
                "success": False,
                "retryable": True,
                "final_answer": "Evaluator model completed without calling final_report.",
                "summary": "The desktop evaluator failed before producing required evaluation artifacts.",
                "metadata": {"failure_category": FailureCategory.EVALUATOR.value},
            },
        )


class DeviceTransportFailingSubmitter(FailingSubmitter):
    """Submitter double for a task interrupted by device transport loss."""

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        return TaskFinished(
            task_id=task_id,
            trace_id=f"trace-for-{task_id}",
            session_id=f"session-{task_id}",
            state=TaskState.INTERRUPTED,
            device_id="device",
            lease_id="lease",
            error="websocket disconnected",
            terminal_decision={
                "action": "error",
                "reason": "runtime_error",
                "success": False,
                "retryable": True,
                "final_answer": "Device websocket disconnected while waiting for command result.",
                "summary": "The device websocket disconnected during the main agent run.",
            },
        )


class InterruptedSubmitter(FailingSubmitter):
    """Submitter double that reports a cooperative interrupted task."""

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        return TaskFinished(
            task_id=task_id,
            trace_id=f"trace-for-{task_id}",
            session_id=f"session-{task_id}",
            state=TaskState.INTERRUPTED,
            device_id="device",
            lease_id="lease",
            error="operator interrupted",
            terminal_decision={"action": "error", "reason": "interrupted"},
        )


class DeviceTransportThenSuccessSubmitter(FailingSubmitter):
    """Submitter double that proves device transport failures consume retry budget."""

    wait_calls = 0

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        self.__class__.wait_calls += 1
        if self.__class__.wait_calls == 1:
            return TaskFinished(
                task_id=task_id,
                trace_id=f"trace-for-{task_id}",
                session_id=f"session-{task_id}",
                state=TaskState.INTERRUPTED,
                device_id="device",
                lease_id="lease",
                error="websocket disconnected",
                terminal_decision={
                    "action": "error",
                    "reason": "interrupted",
                    "success": False,
                    "retryable": False,
                    "final_answer": "websocket disconnected",
                    "summary": "The controller interrupted the task.",
                },
            )
        return TaskFinished(
            task_id=task_id,
            trace_id=f"trace-for-{task_id}",
            session_id=f"session-{task_id}",
            state=TaskState.FINISHED,
            device_id="device",
            lease_id="lease",
            result={"ok": True},
        )


class TimeoutThenCancelSubmitter:
    """Submitter double that times out once, then returns interrupted after cancel."""

    cancel_requests: list[object] = []

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self.wait_calls = 0

    async def __aenter__(self) -> "TimeoutThenCancelSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        task_id = str(kwargs["task_id"])
        return TaskAccepted(task_id=task_id, trace_id=str(kwargs["trace_id"]), session_id=f"session-{task_id}")

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise TimeoutError("wait timed out")
        return TaskFinished(
            task_id=task_id,
            trace_id=f"trace-for-{task_id}",
            session_id=f"session-{task_id}",
            state=TaskState.INTERRUPTED,
            device_id="device",
            lease_id="lease",
            error="cancelled after timeout",
        )

    async def cancel_task(self, cancel) -> None:
        self.cancel_requests.append(cancel)


class BlockingCancelSubmitter:
    """Submitter double that stays in-flight until the runner cancels it."""

    accepted_event: asyncio.Event
    cancel_requests: list[object] = []

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def __aenter__(self) -> "BlockingCancelSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        task_id = str(kwargs["task_id"])
        accepted = TaskAccepted(task_id=task_id, trace_id=str(kwargs["trace_id"]), session_id=f"session-{task_id}")
        self.accepted_event.set()
        return accepted

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        _ = task_id, timeout_seconds
        await asyncio.Event().wait()
        raise AssertionError("blocking submitter should be canceled before finishing")

    async def cancel_task(self, cancel) -> None:
        self.cancel_requests.append(cancel)


@pytest.mark.asyncio
async def test_batch_runner_limits_concurrent_sandbox_creates(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    class SlowCreateProvider(FakeSandboxProvider):
        active_creates = 0
        max_active_creates = 0

        async def create(self, spec):
            self.__class__.active_creates += 1
            self.__class__.max_active_creates = max(
                self.__class__.max_active_creates,
                self.__class__.active_creates,
            )
            try:
                await asyncio.sleep(0.02)
                return await super().create(spec)
            finally:
                self.__class__.active_creates -= 1

    SlowCreateProvider.active_creates = 0
    SlowCreateProvider.max_active_creates = 0
    registry = SandboxProviderRegistry()
    registry.register("fake", SlowCreateProvider)
    manifest = BatchManifest(
        batch_id="create-limit",
        tasks=[
            BatchTaskSpec(task_id="t1", agent_kind="web", query="First"),
            BatchTaskSpec(task_id="t2", agent_kind="web", query="Second"),
            BatchTaskSpec(task_id="t3", agent_kind="web", query="Third"),
        ],
    )
    manifest.defaults.concurrency = 3
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: FakeSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            sandbox_create_concurrency=1,
        )
    )

    assert result.succeeded == 3
    assert SlowCreateProvider.max_active_creates == 1


@pytest.mark.asyncio
async def test_batch_cancel_interrupts_in_flight_controller_task(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="operator-cancel",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Keep running", max_retries=0)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    BlockingCancelSubmitter.accepted_event = asyncio.Event()
    BlockingCancelSubmitter.cancel_requests = []

    runner_task = asyncio.create_task(
        BatchRunner(
            provider_registry=registry,
            submitter_factory=lambda url: BlockingCancelSubmitter(url),
        ).run(
            BatchRunConfig(
                manifest=manifest,
                output_dir=tmp_path,
                lease_timeout_seconds=0.03,
            )
        )
    )
    await asyncio.wait_for(BlockingCancelSubmitter.accepted_event.wait(), timeout=1)

    BatchOperations().cancel(batch_id="operator-cancel", output_dir=tmp_path)
    result = await asyncio.wait_for(runner_task, timeout=2)

    assert result.canceled == 1
    assert result.task_states[0].status.value == "canceled"
    assert BlockingCancelSubmitter.cancel_requests


@pytest.mark.asyncio
async def test_batch_pause_keeps_pending_tasks_out_of_failed_count(tmp_path: Path) -> None:
    from northcua.batch.state import SQLiteBatchStateRepository
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="paused-run",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Wait")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    repository = SQLiteBatchStateRepository(tmp_path / "paused-run" / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    repository.set_control_flag("paused", True)
    (tmp_path / "paused-run" / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: FakeSubmitter(url),
    ).run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))

    assert result.succeeded == 0
    assert result.failed == 0
    assert result.canceled == 0
    assert result.incomplete == 1
    assert result.task_states[0].status == BatchTaskStatus.PENDING


@pytest.mark.asyncio
async def test_batch_runner_rejects_manifest_drift_for_existing_sqlite_state(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="manifest-drift",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Original")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: FakeSubmitter(url))
    await runner.run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))
    mutated = manifest.model_copy(update={"tasks": [BatchTaskSpec(task_id="t1", agent_kind="web", query="Mutated")]})
    mutated.defaults.sandbox = manifest.defaults.sandbox

    with pytest.raises(RuntimeError, match="different manifest body"):
        await runner.run(BatchRunConfig(manifest=mutated, output_dir=tmp_path))


def test_batch_report_and_monitor_prefer_sqlite_over_stale_projections(tmp_path: Path) -> None:
    from northcua.batch.state import SQLiteBatchStateRepository

    manifest = BatchManifest(
        batch_id="sqlite-authority",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Done")],
    )
    batch_dir = tmp_path / "sqlite-authority"
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(BatchTaskState(task_id="t1", status=BatchTaskStatus.SUCCEEDED, attempts=1))
    stale_summary = '{"batch_id": "sqlite-authority", "succeeded": 0, "metrics": {"sandbox_create_p50": 12.5}}'
    (batch_dir / "batch_summary.json").write_text(stale_summary, encoding="utf-8")
    (batch_dir / "case_snapshots.jsonl").write_text(
        '{"batch_id":"sqlite-authority","case_id":"t1","status":"failed","phase":"completed"}\n',
        encoding="utf-8",
    )

    report = BatchOperations().report(batch_id="sqlite-authority", output_dir=tmp_path)
    monitor = BatchOperations().monitor(batch_id="sqlite-authority", output_dir=tmp_path)

    assert report["succeeded"] == 1
    assert report["metrics"] == {"sandbox_create_p50": 12.5}
    assert (batch_dir / "batch_summary.json").read_text(encoding="utf-8") == stale_summary
    assert monitor["case_snapshots"][0]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_two_batch_runners_do_not_submit_duplicate_tasks(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    manifest = BatchManifest(
        batch_id="multi-runner",
        tasks=[
            BatchTaskSpec(task_id="t1", agent_kind="web", query="First"),
            BatchTaskSpec(task_id="t2", agent_kind="web", query="Second"),
            BatchTaskSpec(task_id="t3", agent_kind="web", query="Third"),
        ],
    )
    manifest.defaults.concurrency = 2
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    CountingSubmitter.submitted_task_ids = []
    CountingSubmitter.submitted_batch_task_ids = []

    first = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))
    second = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    results = await asyncio.gather(
        first.run(BatchRunConfig(manifest=manifest, output_dir=tmp_path, runner_id="runner-a")),
        second.run(BatchRunConfig(manifest=manifest, output_dir=tmp_path, runner_id="runner-b")),
    )

    assert sum(result.succeeded for result in results) >= 3
    assert sorted(set(CountingSubmitter.submitted_batch_task_ids)) == ["t1", "t2", "t3"]
    assert len(CountingSubmitter.submitted_batch_task_ids) == 3
    assert len(set(CountingSubmitter.submitted_task_ids)) == 3


@pytest.mark.asyncio
async def test_existing_sqlite_batch_rejects_different_controller_runtime(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-config-lock",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller-a",
        )
    )

    with pytest.raises(ValueError, match="different runtime configuration"):
        await runner.run(
            BatchRunConfig(
                manifest=manifest,
                output_dir=tmp_path,
                controller_url="ws://controller-b",
            )
        )


@pytest.mark.asyncio
async def test_existing_sqlite_batch_can_allow_controller_url_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-config-controller-url-drift",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller-a",
        )
    )

    monkeypatch.setenv("NORTHCUA_ALLOW_CONTROLLER_URL_RUNTIME_DRIFT", "1")
    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller-b",
        )
    )


@pytest.mark.asyncio
async def test_existing_sqlite_batch_rejects_prepared_policy_mode_change(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-prepared-lock",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1),
        )
    )

    with pytest.raises(ValueError, match="prepared_sandbox_policy"):
        await runner.run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))


@pytest.mark.asyncio
async def test_existing_sqlite_batch_allows_disabled_prepared_depth_difference(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-prepared-disabled-depth",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(BatchRunConfig(manifest=manifest, output_dir=tmp_path, dry_run=True))
    result = await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            dry_run=True,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=False, depth=PreparedSandboxDepth.FIXTURE_SETUP),
        )
    )

    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_failed_resume_does_not_clear_pause_on_prepared_policy_mismatch(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-prepared-pause-lock",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))
    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            dry_run=True,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1),
        )
    )
    repository = SQLiteBatchStateRepository(tmp_path / "runtime-prepared-pause-lock" / "batch_state.sqlite")
    repository.set_control_flag("paused", True)

    with pytest.raises(ValueError, match="prepared_sandbox_policy"):
        await BatchOperations().resume(
            batch_id="runtime-prepared-pause-lock",
            output_dir=tmp_path,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=False),
        )

    assert SQLiteBatchStateRepository(tmp_path / "runtime-prepared-pause-lock" / "batch_state.sqlite").control_flags()[
        "paused"
    ]


@pytest.mark.asyncio
async def test_existing_sqlite_batch_rejects_different_controller_token_fingerprint(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-token-lock",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="token-a",
            persist_token=False,
        )
    )

    run_config = json.loads((tmp_path / "runtime-token-lock" / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["token"] == ""
    assert run_config["token_fingerprint"]

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="token-a",
            persist_token=False,
        )
    )

    with pytest.raises(ValueError, match="token_fingerprint"):
        await runner.run(
            BatchRunConfig(
                manifest=manifest,
                output_dir=tmp_path,
                controller_url="ws://controller",
                token="token-b",
                persist_token=False,
            )
        )


@pytest.mark.asyncio
async def test_existing_sqlite_batch_preserves_unpersisted_token_fingerprint_on_tokenless_resume(
    tmp_path: Path,
) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-tokenless-resume-lock",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="token-a",
            persist_token=False,
        )
    )

    run_config_path = tmp_path / "runtime-tokenless-resume-lock" / "run_config.json"
    initial = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert initial["token"] == ""
    assert initial["token_fingerprint"]

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="",
            persist_token=False,
        )
    )

    resumed = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert resumed["token"] == ""
    assert resumed["token_fingerprint"] == initial["token_fingerprint"]


@pytest.mark.asyncio
async def test_existing_sqlite_batch_rejects_different_manifest_default_token_on_tokenless_resume(
    tmp_path: Path,
) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-default-token-lock",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    manifest.defaults.token = "token-a"
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="",
            persist_token=False,
        )
    )

    modified = manifest.model_copy(deep=True)
    modified.defaults.token = "token-b"
    with pytest.raises(ValueError, match="token_fingerprint"):
        await runner.run(
            BatchRunConfig(
                manifest=modified,
                output_dir=tmp_path,
                controller_url="ws://controller",
                token="",
                persist_token=False,
            )
        )


@pytest.mark.asyncio
async def test_existing_sqlite_batch_preserves_persisted_token_on_tokenless_shared_runner_resume(
    tmp_path: Path,
) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-persisted-token-resume",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="token-a",
            persist_token=True,
        )
    )

    run_config_path = tmp_path / "runtime-persisted-token-resume" / "run_config.json"
    initial = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert initial["token"] == "token-a"
    assert initial["token_fingerprint"]

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="",
            persist_token=True,
        )
    )

    resumed = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert resumed["token"] == "token-a"
    assert resumed["token_fingerprint"] == initial["token_fingerprint"]


@pytest.mark.asyncio
async def test_existing_sqlite_batch_honors_token_persistence_opt_out_with_supplied_token(
    tmp_path: Path,
) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-persisted-token-opt-out",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="token-a",
            persist_token=True,
        )
    )

    run_config_path = tmp_path / "runtime-persisted-token-opt-out" / "run_config.json"
    initial = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert initial["token"] == "token-a"
    assert initial["token_fingerprint"]

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="token-a",
            persist_token=False,
        )
    )

    resumed = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert resumed["token"] == ""
    assert resumed["token_fingerprint"] == initial["token_fingerprint"]


@pytest.mark.asyncio
async def test_existing_sqlite_batch_rejects_different_token_when_legacy_config_only_persisted_token(
    tmp_path: Path,
) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-legacy-token-only-lock",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="token-a",
            persist_token=True,
        )
    )

    run_config_path = tmp_path / "runtime-legacy-token-only-lock" / "run_config.json"
    legacy = json.loads(run_config_path.read_text(encoding="utf-8"))
    legacy.pop("token_fingerprint", None)
    run_config_path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError, match="token_fingerprint"):
        await runner.run(
            BatchRunConfig(
                manifest=manifest,
                output_dir=tmp_path,
                controller_url="ws://controller",
                token="token-b",
                persist_token=False,
            )
        )


@pytest.mark.asyncio
async def test_existing_sqlite_batch_backfills_legacy_token_fingerprint(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-token-legacy-lock",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="token-a",
            persist_token=False,
        )
    )

    run_config_path = tmp_path / "runtime-token-legacy-lock" / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config.pop("token_fingerprint", None)
    run_config_path.write_text(json.dumps(run_config), encoding="utf-8")

    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://controller",
            token="token-a",
            persist_token=False,
        )
    )

    migrated = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert migrated["token_fingerprint"]

    with pytest.raises(ValueError, match="token_fingerprint"):
        await runner.run(
            BatchRunConfig(
                manifest=manifest,
                output_dir=tmp_path,
                controller_url="ws://controller",
                token="token-b",
                persist_token=False,
            )
        )


@pytest.mark.asyncio
async def test_existing_sqlite_batch_backfills_legacy_controller_db_path(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="runtime-controller-db-legacy-lock",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    await runner.run(BatchRunConfig(manifest=manifest, output_dir=tmp_path, dry_run=True))

    controller_db = tmp_path / "controller.sqlite"
    controller_db.touch()
    await runner.run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            dry_run=True,
            telemetry_controller_db_path=controller_db,
        )
    )

    migrated = json.loads(
        (tmp_path / "runtime-controller-db-legacy-lock" / "run_config.json").read_text(encoding="utf-8")
    )
    assert Path(migrated["telemetry_controller_db_path"]) == controller_db

    with pytest.raises(ValueError, match="telemetry_controller_db_path"):
        await runner.run(
            BatchRunConfig(
                manifest=manifest,
                output_dir=tmp_path,
                dry_run=True,
                telemetry_controller_db_path=tmp_path / "other-controller.sqlite",
            )
        )


@pytest.mark.asyncio
async def test_exhausted_local_setup_recovery_terminalizes_instead_of_spinning(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="local-recovery-terminal",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="First", max_retries=0)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    repository = SQLiteBatchStateRepository(tmp_path / manifest.batch_id / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    repository.upsert_state(
        BatchTaskState(
            task_id="t1",
            status=BatchTaskStatus.STARTING_SANDBOX,
            attempts=1,
        )
    )
    (tmp_path / manifest.batch_id / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: CountingSubmitter(url),
    ).run(BatchRunConfig(manifest=manifest, output_dir=tmp_path, lease_timeout_seconds=0.1))

    state = result.task_states[0]
    assert result.failed == 1
    assert state.status == BatchTaskStatus.FAILED
    assert state.last_failure_category == FailureCategory.SANDBOX_CREATE
    assert "attempt budget exhausted" in state.error


@pytest.mark.asyncio
async def test_concurrent_first_runs_reject_manifest_drift_atomically(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    first_manifest = BatchManifest(
        batch_id="first-run-race",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Original")],
    )
    second_manifest = first_manifest.model_copy(
        update={"tasks": [BatchTaskSpec(task_id="t1", agent_kind="web", query="Mutated")]}
    )
    first_manifest.defaults.sandbox = first_manifest.defaults.sandbox.model_copy(
        update={"provider": "fake", "template": "ubuntu"}
    )
    second_manifest.defaults.sandbox = first_manifest.defaults.sandbox
    first = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))
    second = BatchRunner(provider_registry=registry, submitter_factory=lambda url: CountingSubmitter(url))

    results = await asyncio.gather(
        first.run(BatchRunConfig(manifest=first_manifest, output_dir=tmp_path, runner_id="runner-a")),
        second.run(BatchRunConfig(manifest=second_manifest, output_dir=tmp_path, runner_id="runner-b")),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert any(isinstance(result, RuntimeError) and "different manifest body" in str(result) for result in results)


@pytest.mark.asyncio
async def test_batch_runner_uses_terminal_decision_failure_category(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="evaluator-failed",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Evaluate failure", max_retries=0)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: EvaluatorFailingSubmitter(url),
    ).run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))

    assert result.failed == 1
    assert result.task_states[0].last_failure_category == FailureCategory.EVALUATOR
    assert result.task_states[0].terminal_decision["metadata"]["failure_category"] == FailureCategory.EVALUATOR.value


@pytest.mark.asyncio
async def test_batch_runner_classifies_device_transport_terminal_failure(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="device-transport-failed",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Click save", max_retries=0)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: DeviceTransportFailingSubmitter(url),
    ).run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))

    assert result.failed == 1
    assert result.task_states[0].last_failure_category == FailureCategory.DEVICE_TRANSPORT
    snapshots = [
        json.loads(line)
        for line in (tmp_path / "device-transport-failed" / "case_snapshots.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert snapshots[-1]["failure_category"] == "device_transport"


@pytest.mark.asyncio
async def test_batch_runner_maps_generic_interrupted_terminal_to_canceled(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="generic-interrupted",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Stop cleanly", max_retries=0)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: InterruptedSubmitter(url),
    ).run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))

    assert result.canceled == 1
    assert result.failed == 0
    assert result.task_states[0].status == BatchTaskStatus.CANCELED


@pytest.mark.asyncio
async def test_batch_runner_retries_device_transport_even_when_terminal_marks_non_retryable(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="device-transport-retry",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Click save", max_retries=1)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    DeviceTransportThenSuccessSubmitter.wait_calls = 0

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: DeviceTransportThenSuccessSubmitter(url),
    ).run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))

    assert result.succeeded == 1
    assert result.task_states[0].attempts == 2
    assert result.task_states[0].status == BatchTaskStatus.SUCCEEDED
    metrics = [
        json.loads(line)
        for line in (tmp_path / "device-transport-retry" / "case_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert metrics[0]["attempt_failure_category_breakdown"] == {"device_transport": 1}
