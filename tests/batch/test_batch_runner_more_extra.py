"""Batch runner service tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from northcua.batch.contracts import BatchManifest, BatchRunConfig, BatchRunResult, BatchTaskSpec
from northcua.batch.operations import BatchOperations
from northcua.batch.runner import BatchRunner
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


@pytest.mark.asyncio
async def test_batch_timeout_cancels_controller_task_before_release(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="timeout-cancel",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Slow eval", timeout_seconds=0.01, max_retries=0)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    TimeoutThenCancelSubmitter.cancel_requests = []

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: TimeoutThenCancelSubmitter(url),
    ).run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))

    assert result.failed == 1
    assert result.task_states[0].status.value == "timed_out"
    assert result.task_states[0].terminal_state == TaskState.INTERRUPTED
    assert TimeoutThenCancelSubmitter.cancel_requests


@pytest.mark.asyncio
async def test_batch_resume_reuses_persisted_runtime_overrides(tmp_path: Path) -> None:
    registry = SandboxProviderRegistry()
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="resume-overrides",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Open page")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: FakeSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            controller_url="ws://override-controller",
            token="secret-token",
            concurrency=7,
        )
    )

    RecordingRunner.configs = []
    await BatchOperations(runner_factory=RecordingRunner).resume(batch_id="resume-overrides", output_dir=tmp_path)

    resumed = RecordingRunner.configs[0]
    assert resumed.controller_url == "ws://override-controller"
    assert resumed.token == "secret-token"
    assert resumed.concurrency == 7


@pytest.mark.asyncio
async def test_batch_resume_cli_overrides_persisted_runtime_config(tmp_path: Path) -> None:
    manifest = BatchManifest(
        batch_id="resume-cli-overrides",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Open page")],
    )
    batch_dir = tmp_path / "resume-cli-overrides"
    batch_dir.mkdir()
    (batch_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (batch_dir / "run_config.json").write_text(
        '{"controller_url":"ws://old","token":"old-token","concurrency":2}',
        encoding="utf-8",
    )

    RecordingRunner.configs = []
    await BatchOperations(runner_factory=RecordingRunner).resume(
        batch_id="resume-cli-overrides",
        output_dir=tmp_path,
        controller_url="ws://new",
        token="new-token",
        concurrency=3,
    )

    resumed = RecordingRunner.configs[0]
    assert resumed.controller_url == "ws://new"
    assert resumed.token == "new-token"
    assert resumed.concurrency == 3
