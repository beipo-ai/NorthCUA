"""Batch runner service tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from websockets.exceptions import ConnectionClosedError

from northcua.batch.contracts import (
    BatchManifest,
    BatchRunConfig,
    BatchRunResult,
    BatchTaskSpec,
    BatchTaskState,
    BatchTaskStatus,
)
from northcua.batch.metrics import BatchMetrics
from northcua.batch.operations import BatchOperations, _result_from_states
from northcua.batch.runner import BatchRunner
from northcua.batch.runner import executor as executor_module
from northcua.batch.runner.attempt_submit import _state_from_terminal_controller_result
from northcua.batch.runner.controller import ControllerTaskWatcher
from northcua.batch.runner.executor import BatchAttemptExecutor
from northcua.sandboxes.contracts import SandboxAllocationStrategy, SandboxCommandSpec, SandboxHandle, SandboxLease
from northcua.sandboxes.provider.registry import SandboxProviderRegistry
from northcua.telemetry.contracts import FailureCategory
from northcua_protocol.payloads import TaskAccepted, TaskCancel, TaskFinished, TaskStatusSnapshot
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


class CallbackOrderSubmitter(FakeSubmitter):
    """Submitter double that records connection order around submit callbacks."""

    events: list[str] = []

    @classmethod
    def reset(cls) -> None:
        cls.events = []

    async def __aenter__(self) -> "CallbackOrderSubmitter":
        type(self).events.append("enter")
        return self

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        type(self).events.append("submit")
        return await super().submit_task(**kwargs)


class LegacyStrictSubmitter(FakeSubmitter):
    """Submitter double with the pre-request-sent callback signature."""

    async def submit_task(
        self,
        *,
        agent_kind: str,
        request: str,
        token: str,
        target_device_id: str,
        required_capabilities: object,
        input_artifacts: object,
        metadata: dict[str, object],
        task_id: str = "",
        trace_id: str = "",
    ) -> TaskAccepted:
        self.accepted = TaskAccepted(
            task_id=task_id or "controller-task", trace_id=trace_id or "trace", session_id="session"
        )
        return self.accepted


class LegacyForwardingSubmitter(FakeSubmitter):
    """Submitter wrapper accepting kwargs but forwarding to an older strict implementation."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        type(self).calls += 1
        if "on_request_sent" in kwargs:
            raise TypeError("submit_task() got an unexpected keyword argument 'on_request_sent'")
        return await super().submit_task(**kwargs)


class SlowSubmitter:
    """Submitter double that makes submit and terminal wait overlap measurable."""

    active_submits = 0
    max_active_submits = 0
    active_waits = 0
    max_active_waits = 0

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    @classmethod
    def reset(cls) -> None:
        cls.active_submits = 0
        cls.max_active_submits = 0
        cls.active_waits = 0
        cls.max_active_waits = 0

    async def __aenter__(self) -> "SlowSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        type(self).active_submits += 1
        type(self).max_active_submits = max(type(self).max_active_submits, type(self).active_submits)
        try:
            await asyncio.sleep(0.01)
            task_id = str(kwargs["task_id"])
            return TaskAccepted(task_id=task_id, trace_id=str(kwargs["trace_id"]), session_id=f"session-{task_id}")
        finally:
            type(self).active_submits -= 1

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        type(self).active_waits += 1
        type(self).max_active_waits = max(type(self).max_active_waits, type(self).active_waits)
        try:
            await asyncio.sleep(0.1)
            return TaskFinished(
                task_id=task_id,
                trace_id=f"trace-for-{task_id}",
                session_id=f"session-{task_id}",
                state=TaskState.FINISHED,
                device_id="device",
                lease_id="lease",
                result={"ok": True},
            )
        finally:
            type(self).active_waits -= 1


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


class RaisingSubmitter(FailingSubmitter):
    """Submitter double that raises while waiting for a terminal task."""

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        raise RuntimeError("controller connection closed before terminal result")


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


class _RecordingEvents:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def attempt_event(self, **kwargs: object) -> None:
        self.events.append(kwargs)


@pytest.mark.asyncio
async def test_attempt_executor_returns_submitted_without_starting_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    async def fake_execute_attempt(**kwargs: object) -> BatchTaskState:
        nonlocal calls
        calls += 1
        task = kwargs["task"]
        assert isinstance(task, BatchTaskSpec)
        return BatchTaskState(
            task_id=task.task_id,
            attempts=int(kwargs["attempt"]),
            status=BatchTaskStatus.SUBMITTED,
            controller_task_id="controller-task",
            trace_id="trace",
        )

    monkeypatch.setattr(executor_module, "_execute_attempt", fake_execute_attempt)
    task = BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", max_retries=3)
    manifest = BatchManifest(batch_id="batch", tasks=[task])
    executor = BatchAttemptExecutor(
        provider_registry=SandboxProviderRegistry(),
        planner=object(),  # type: ignore[arg-type]
        watcher=object(),  # type: ignore[arg-type]
    )

    result = await executor.run_task(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        events=object(),  # type: ignore[arg-type]
        metrics=BatchMetrics(),
        output_dir=tmp_path,
    )

    assert calls == 1
    assert result.status == BatchTaskStatus.SUBMITTED
    assert result.controller_task_id == "controller-task"
    assert result.retry_budget_remaining == 3


@pytest.mark.asyncio
async def test_attempt_executor_returns_evaluation_pending_without_whole_case_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    async def fake_execute_attempt(**kwargs: object) -> BatchTaskState:
        nonlocal calls
        calls += 1
        task = kwargs["task"]
        assert isinstance(task, BatchTaskSpec)
        return BatchTaskState(
            task_id=task.task_id,
            attempts=int(kwargs["attempt"]),
            status=BatchTaskStatus.EVALUATION_PENDING,
            controller_task_id="controller-task",
            trace_id="trace",
            evaluation_status="pending",
        )

    monkeypatch.setattr(executor_module, "_execute_attempt", fake_execute_attempt)
    task = BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", max_retries=3)
    manifest = BatchManifest(batch_id="batch", tasks=[task])
    executor = BatchAttemptExecutor(
        provider_registry=SandboxProviderRegistry(),
        planner=object(),  # type: ignore[arg-type]
        watcher=object(),  # type: ignore[arg-type]
    )

    result = await executor.run_task(
        config=BatchRunConfig(manifest=manifest),
        task=task,
        events=object(),  # type: ignore[arg-type]
        metrics=BatchMetrics(),
        output_dir=tmp_path,
    )

    assert calls == 1
    assert result.status == BatchTaskStatus.EVALUATION_PENDING
    assert result.retry_budget_remaining == 3


def test_terminal_controller_result_uses_non_success_evaluator_report_as_failed() -> None:
    task = BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", max_retries=0)
    accepted = TaskAccepted(task_id="controller-1", trace_id="trace-1", session_id="session-1")
    finished = TaskFinished(
        task_id="controller-1",
        trace_id="trace-1",
        session_id="session-1",
        state=TaskState.FINISHED,
        device_id="device",
        lease_id="lease",
        result={"evaluation_report": {"overall_verdict": "failure", "outcome_label": "failure"}},
    )
    lease = SandboxLease(
        lease_id="lease",
        handle=SandboxHandle(sandbox_id="sandbox", provider="fake", template="desktop"),
        device_id="device",
    )
    events = _RecordingEvents()

    result = _state_from_terminal_controller_result(
        accepted=accepted,
        finished=finished,
        task=task,
        lease=lease,
        state=BatchTaskState(task_id=task.task_id, attempts=1, status=BatchTaskStatus.RUNNING),
        metrics=BatchMetrics(),
        events=events,  # type: ignore[arg-type]
        task_start=0.0,
    )

    assert result.status == BatchTaskStatus.FAILED
    assert result.terminal_state == TaskState.FINISHED
    assert result.last_failure_category == FailureCategory.MAIN_AGENT
    assert events.events[0]["details"]["terminal_status"] == BatchTaskStatus.FAILED.value


def test_batch_runner_result_counts_evaluation_pending_as_terminal_failure(tmp_path: Path) -> None:
    states = [
        BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED),
        BatchTaskState(task_id="case-2", status=BatchTaskStatus.EVALUATION_PENDING, evaluation_status="pending"),
        BatchTaskState(task_id="case-3", status=BatchTaskStatus.PENDING),
    ]

    result = _result_from_states(
        batch_id="batch",
        output_dir=tmp_path,
        states=states,
    )

    assert result.succeeded == 1
    assert result.failed == 1
    assert result.incomplete == 1
    assert result.total == 3


class NonCooperativeTimeoutSubmitter:
    """Submitter double whose wait stream stays busy until cancelled."""

    instances: list["NonCooperativeTimeoutSubmitter"] = []

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self.cancelled = False
        self.instances.append(self)

    async def __aenter__(self) -> "NonCooperativeTimeoutSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        task_id = str(kwargs["task_id"])
        return TaskAccepted(task_id=task_id, trace_id=str(kwargs["trace_id"]), session_id=f"session-{task_id}")

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        del timeout_seconds
        if self.cancelled:
            return TaskFinished(
                task_id=task_id,
                trace_id=f"trace-for-{task_id}",
                session_id=f"session-{task_id}",
                state=TaskState.INTERRUPTED,
                device_id="device",
                lease_id="lease",
                error="cancelled",
            )
        await asyncio.sleep(10)
        raise AssertionError("timeout guard should cancel this wait before it completes")

    async def cancel_task(self, cancel: TaskCancel) -> None:
        del cancel
        self.cancelled = True


class DisconnectThenStatusSubmitter:
    """Submitter double that drops during wait but can recover from status."""

    instances: list["DisconnectThenStatusSubmitter"] = []

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self.instances.append(self)

    async def __aenter__(self) -> "DisconnectThenStatusSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        return TaskAccepted(task_id=str(kwargs["task_id"]), trace_id=str(kwargs["trace_id"]), session_id="session")

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        raise ConnectionClosedError(None, None)

    async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
        return [
            TaskStatusSnapshot(
                task_id=task_ids[0],
                trace_id="trace",
                session_id="session",
                agent_kind="desktop",
                state=TaskState.FINISHED,
                device_id="device",
                lease_id="lease",
                result={"ok": True},
                terminal_decision={"action": "finish", "reason": "completed", "success": True},
            )
        ]


class IdleTerminalStatusSubmitter:
    """Submitter double whose wait socket stays idle while status turns terminal."""

    instances: list["IdleTerminalStatusSubmitter"] = []
    status_calls = 0
    wait_cancelled = False

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.status_calls = 0
        cls.wait_cancelled = False

    async def __aenter__(self) -> "IdleTerminalStatusSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        return TaskAccepted(task_id=str(kwargs["task_id"]), trace_id=str(kwargs["trace_id"]), session_id="session")

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        del task_id, timeout_seconds
        try:
            await asyncio.Event().wait()
        finally:
            type(self).wait_cancelled = True
        raise AssertionError("idle wait should be cancelled after status poll recovers terminal state")

    async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
        del token
        type(self).status_calls += 1
        return [
            TaskStatusSnapshot(
                task_id=task_ids[0],
                trace_id="trace",
                session_id="session",
                agent_kind="desktop",
                state=TaskState.INTERRUPTED,
                error="queued task lost its submitter mapping",
            )
        ]


class IdleRunningStatusThenFinishedSubmitter:
    """Submitter double whose status polls stay nonterminal until the wait stream finishes."""

    status_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.status_calls = 0

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def __aenter__(self) -> "IdleRunningStatusThenFinishedSubmitter":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        return TaskAccepted(task_id=str(kwargs["task_id"]), trace_id=str(kwargs["trace_id"]), session_id="session")

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        del timeout_seconds
        await asyncio.sleep(0.03)
        return TaskFinished(
            task_id=task_id,
            trace_id="trace",
            session_id="session",
            state=TaskState.FINISHED,
            device_id="device",
            lease_id="lease",
            result={"ok": True},
        )

    async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
        del token
        type(self).status_calls += 1
        return [
            TaskStatusSnapshot(
                task_id=task_ids[0],
                trace_id="trace",
                session_id="session",
                agent_kind="desktop",
                state=TaskState.RUNNING,
            )
        ]


class IdleStatusErrorThenFinishedSubmitter(IdleRunningStatusThenFinishedSubmitter):
    """Submitter double whose idle status API errors before the wait stream finishes."""

    async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
        del task_ids, token
        type(self).status_calls += 1
        raise RuntimeError("controller returned error: transient status failure")


class TimeoutBeforeIdleStatusSubmitter(NonCooperativeTimeoutSubmitter):
    """Submitter double whose status API must not run after the task deadline expires."""

    status_calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.instances.clear()
        cls.status_calls = 0

    async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
        del task_ids, token
        type(self).status_calls += 1
        await asyncio.sleep(1)
        return []


class DisconnectThenPollingStatusSubmitter(DisconnectThenStatusSubmitter):
    """Submitter double that requires polling because the task is still running."""

    status_calls = 0

    async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
        type(self).status_calls += 1
        state = TaskState.RUNNING if self.status_calls == 1 else TaskState.FINISHED
        return [
            TaskStatusSnapshot(
                task_id=task_ids[0],
                trace_id="trace",
                session_id="session",
                agent_kind="desktop",
                state=state,
                device_id="device",
                lease_id="lease",
                result={"ok": True} if state == TaskState.FINISHED else {},
                terminal_decision=(
                    {"action": "finish", "reason": "completed", "success": True} if state == TaskState.FINISHED else {}
                ),
            )
        ]


class DisconnectThenFlakyPollingStatusSubmitter(DisconnectThenStatusSubmitter):
    """Submitter double that has a transient reconnect failure during recovery."""

    enter_calls = 0
    status_calls = 0

    async def __aenter__(self) -> "DisconnectThenFlakyPollingStatusSubmitter":
        type(self).enter_calls += 1
        if self.enter_calls == 2:
            raise TimeoutError("timed out during opening handshake")
        return self

    async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
        type(self).status_calls += 1
        return [
            TaskStatusSnapshot(
                task_id=task_ids[0],
                trace_id="trace",
                session_id="session",
                agent_kind="desktop",
                state=TaskState.FINISHED,
                device_id="device",
                lease_id="lease",
                result={"ok": True},
                terminal_decision={"action": "finish", "reason": "completed", "success": True},
            )
        ]


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
async def test_batch_runner_executes_with_fake_provider(tmp_path: Path) -> None:
    registry = SandboxProviderRegistry()
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="demo",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Open page")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: FakeSubmitter(url),
    ).run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))

    assert result.succeeded == 1
    assert (tmp_path / "demo" / "batch_summary.json").exists()
    assert (tmp_path / "demo" / "manifest.json").exists()
    assert (tmp_path / "demo" / "telemetry_events.jsonl").exists()
    assert (tmp_path / "demo" / "case_metrics.jsonl").exists()
    assert (tmp_path / "demo" / "audit_report.md").exists()
    assert result.task_states[0].controller_task_id == "controller-task"


@pytest.mark.asyncio
async def test_batch_runner_caps_controller_submit_without_serializing_waits(tmp_path: Path) -> None:
    registry = SandboxProviderRegistry()
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry.register("fake", FakeSandboxProvider)
    tasks = [BatchTaskSpec(task_id=f"t{i}", agent_kind="web", query="Open page") for i in range(3)]
    manifest = BatchManifest(batch_id="submit-cap", tasks=tasks)
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    SlowSubmitter.reset()

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: SlowSubmitter(url),
    ).run(
        BatchRunConfig(
            manifest=manifest,
            output_dir=tmp_path,
            concurrency=3,
            controller_submit_concurrency=1,
        )
    )

    assert result.succeeded == 3
    assert SlowSubmitter.max_active_submits == 1
    assert SlowSubmitter.max_active_waits > 1


@pytest.mark.asyncio
async def test_controller_task_watcher_recovers_finished_status_after_wait_disconnect() -> None:
    from northcua.batch.planner import PlannedTaskSubmission

    DisconnectThenStatusSubmitter.instances.clear()
    watcher = ControllerTaskWatcher(lambda url: DisconnectThenStatusSubmitter(url))
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    accepted, finished = await watcher.submit_and_wait(
        controller_url="ws://controller",
        token="token",
        planned=planned,
        timeout_seconds=10,
    )

    assert accepted.task_id == "controller-task"
    assert finished.state == TaskState.FINISHED
    assert finished.terminal_decision["action"] == "finish"
    assert len(DisconnectThenStatusSubmitter.instances) == 2


@pytest.mark.asyncio
async def test_controller_task_watcher_ignores_accepted_callback_failure() -> None:
    from northcua.batch.planner import PlannedTaskSubmission

    watcher = ControllerTaskWatcher(lambda url: FakeSubmitter(url))
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    accepted, finished = await watcher.submit_and_wait(
        controller_url="ws://controller",
        token="token",
        planned=planned,
        timeout_seconds=10,
        on_accepted=lambda _accepted, _elapsed: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
    )

    assert accepted.task_id == "controller-task"
    assert finished.state == TaskState.FINISHED


@pytest.mark.asyncio
async def test_controller_task_watcher_supports_legacy_submitter_without_request_sent_callback() -> None:
    from northcua.batch.planner import PlannedTaskSubmission

    request_sent_events: list[dict[str, object]] = []
    watcher = ControllerTaskWatcher(lambda url: LegacyStrictSubmitter(url))
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    accepted, finished = await watcher.submit_and_wait(
        controller_url="ws://controller",
        token="token",
        planned=planned,
        timeout_seconds=10,
        on_request_sent=request_sent_events.append,
    )

    assert accepted.task_id == "controller-task"
    assert finished.state == TaskState.FINISHED
    assert len(request_sent_events) == 1
    assert request_sent_events[0]["request_sent_callback_supported"] is False


@pytest.mark.asyncio
async def test_controller_task_watcher_retries_forwarding_submitter_without_request_sent_callback() -> None:
    from northcua.batch.planner import PlannedTaskSubmission

    LegacyForwardingSubmitter.reset()
    request_sent_events: list[dict[str, object]] = []
    watcher = ControllerTaskWatcher(lambda url: LegacyForwardingSubmitter(url))
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    accepted, finished = await watcher.submit_and_wait(
        controller_url="ws://controller",
        token="token",
        planned=planned,
        timeout_seconds=10,
        on_request_sent=request_sent_events.append,
    )

    assert LegacyForwardingSubmitter.calls == 2
    assert accepted.task_id == "controller-task"
    assert finished.state == TaskState.FINISHED
    assert len(request_sent_events) == 1
    assert request_sent_events[0]["request_sent_callback_supported"] is False


@pytest.mark.asyncio
async def test_controller_task_watcher_runs_before_submit_inside_submit_gate() -> None:
    from northcua.batch.planner import PlannedTaskSubmission

    CallbackOrderSubmitter.reset()
    watcher = ControllerTaskWatcher(lambda url: CallbackOrderSubmitter(url))
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )
    submit_gate = asyncio.Semaphore(1)
    await submit_gate.acquire()

    async def before_submit() -> None:
        CallbackOrderSubmitter.events.append("before_submit")

    task = asyncio.create_task(
        watcher.submit_and_wait(
            controller_url="ws://controller",
            token="token",
            planned=planned,
            timeout_seconds=10,
            submit_semaphore=submit_gate,
            before_submit=before_submit,
        )
    )
    await asyncio.sleep(0.01)
    assert CallbackOrderSubmitter.events == []

    submit_gate.release()
    accepted, finished = await task

    assert accepted.task_id == "controller-task"
    assert finished.state == TaskState.FINISHED
    assert CallbackOrderSubmitter.events == ["before_submit", "enter", "submit"]


@pytest.mark.asyncio
async def test_controller_task_watcher_enforces_wall_clock_timeout_for_busy_wait_stream() -> None:
    from northcua.batch.planner import PlannedTaskSubmission
    from northcua.batch.runner.controller import ControllerTaskTimeoutError

    NonCooperativeTimeoutSubmitter.instances.clear()
    watcher = ControllerTaskWatcher(lambda url: NonCooperativeTimeoutSubmitter(url))
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    with pytest.raises(ControllerTaskTimeoutError) as exc_info:
        await watcher.submit_and_wait(
            controller_url="ws://controller",
            token="token",
            planned=planned,
            timeout_seconds=0.02,
        )

    assert NonCooperativeTimeoutSubmitter.instances[0].cancelled is True
    assert exc_info.value.finished is not None
    assert exc_info.value.finished.state == TaskState.INTERRUPTED


@pytest.mark.asyncio
async def test_controller_task_watcher_polls_running_status_after_wait_disconnect() -> None:
    from northcua.batch.planner import PlannedTaskSubmission

    DisconnectThenPollingStatusSubmitter.instances.clear()
    DisconnectThenPollingStatusSubmitter.status_calls = 0
    watcher = ControllerTaskWatcher(
        lambda url: DisconnectThenPollingStatusSubmitter(url),
        disconnect_status_poll_seconds=0,
    )
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    accepted, finished = await watcher.submit_and_wait(
        controller_url="ws://controller",
        token="token",
        planned=planned,
        timeout_seconds=10,
    )

    assert accepted.task_id == "controller-task"
    assert finished.state == TaskState.FINISHED
    assert finished.terminal_decision["action"] == "finish"
    assert DisconnectThenPollingStatusSubmitter.status_calls == 2


@pytest.mark.asyncio
async def test_controller_task_watcher_polls_terminal_status_while_wait_socket_is_idle() -> None:
    from northcua.batch.planner import PlannedTaskSubmission

    IdleTerminalStatusSubmitter.reset()
    watcher = ControllerTaskWatcher(
        lambda url: IdleTerminalStatusSubmitter(url),
        submit_wait_status_poll_seconds=0.01,
        disconnect_status_poll_seconds=0.05,
    )
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    accepted, finished = await asyncio.wait_for(
        watcher.submit_and_wait(
            controller_url="ws://controller",
            token="token",
            planned=planned,
            timeout_seconds=None,
        ),
        timeout=0.3,
    )

    assert accepted.task_id == "controller-task"
    assert finished.state == TaskState.INTERRUPTED
    assert finished.error == "queued task lost its submitter mapping"
    assert IdleTerminalStatusSubmitter.status_calls == 1
    assert IdleTerminalStatusSubmitter.wait_cancelled is True


@pytest.mark.asyncio
async def test_controller_task_watcher_ignores_nonterminal_idle_status_poll() -> None:
    from northcua.batch.planner import PlannedTaskSubmission

    IdleRunningStatusThenFinishedSubmitter.reset()
    watcher = ControllerTaskWatcher(
        lambda url: IdleRunningStatusThenFinishedSubmitter(url),
        submit_wait_status_poll_seconds=0.01,
        disconnect_status_poll_seconds=0.05,
    )
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    _accepted, finished = await asyncio.wait_for(
        watcher.submit_and_wait(
            controller_url="ws://controller",
            token="token",
            planned=planned,
            timeout_seconds=None,
        ),
        timeout=0.3,
    )

    assert finished.state == TaskState.FINISHED
    assert IdleRunningStatusThenFinishedSubmitter.status_calls >= 1


@pytest.mark.asyncio
async def test_controller_task_watcher_ignores_idle_status_poll_error() -> None:
    from northcua.batch.planner import PlannedTaskSubmission

    IdleStatusErrorThenFinishedSubmitter.reset()
    watcher = ControllerTaskWatcher(
        lambda url: IdleStatusErrorThenFinishedSubmitter(url),
        submit_wait_status_poll_seconds=0.01,
        disconnect_status_poll_seconds=0.05,
    )
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    _accepted, finished = await asyncio.wait_for(
        watcher.submit_and_wait(
            controller_url="ws://controller",
            token="token",
            planned=planned,
            timeout_seconds=None,
        ),
        timeout=0.3,
    )

    assert finished.state == TaskState.FINISHED
    assert IdleStatusErrorThenFinishedSubmitter.status_calls >= 1


@pytest.mark.asyncio
async def test_controller_task_watcher_does_not_start_idle_status_poll_after_deadline() -> None:
    from northcua.batch.planner import PlannedTaskSubmission
    from northcua.batch.runner.controller import ControllerTaskTimeoutError

    TimeoutBeforeIdleStatusSubmitter.reset()
    watcher = ControllerTaskWatcher(
        lambda url: TimeoutBeforeIdleStatusSubmitter(url),
        submit_wait_status_poll_seconds=0.02,
        disconnect_status_poll_seconds=1,
    )
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    with pytest.raises(ControllerTaskTimeoutError):
        await asyncio.wait_for(
            watcher.submit_and_wait(
                controller_url="ws://controller",
                token="token",
                planned=planned,
                timeout_seconds=0.02,
            ),
            timeout=0.3,
        )

    assert TimeoutBeforeIdleStatusSubmitter.status_calls == 0


@pytest.mark.asyncio
async def test_controller_task_watcher_retries_status_poll_connect_timeout_after_wait_disconnect() -> None:
    from northcua.batch.planner import PlannedTaskSubmission

    DisconnectThenFlakyPollingStatusSubmitter.instances.clear()
    DisconnectThenFlakyPollingStatusSubmitter.enter_calls = 0
    DisconnectThenFlakyPollingStatusSubmitter.status_calls = 0
    watcher = ControllerTaskWatcher(
        lambda url: DisconnectThenFlakyPollingStatusSubmitter(url),
        disconnect_status_poll_seconds=0,
    )
    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )

    accepted, finished = await watcher.submit_and_wait(
        controller_url="ws://controller",
        token="token",
        planned=planned,
        timeout_seconds=10,
    )

    assert accepted.task_id == "controller-task"
    assert finished.state == TaskState.FINISHED
    assert finished.terminal_decision["action"] == "finish"
    assert DisconnectThenFlakyPollingStatusSubmitter.enter_calls == 3
    assert DisconnectThenFlakyPollingStatusSubmitter.status_calls == 1


def test_batch_monitor_reads_latest_telemetry(tmp_path: Path) -> None:
    operations = BatchOperations()
    batch_dir = tmp_path / "demo"
    batch_dir.mkdir()
    (batch_dir / "case_snapshots.jsonl").write_text(
        '{"case_id": "case-1", "status": "running_agent"}\n{"case_id": "case-1", "status": "succeeded"}\n',
        encoding="utf-8",
    )

    payload = operations.monitor(batch_id="demo", output_dir=tmp_path)

    assert payload["summary"]["total_cases"] == 1
    assert payload["summary"]["succeeded"] == 1
    assert payload["case_snapshots"] == [{"case_id": "case-1", "status": "succeeded"}]


@pytest.mark.asyncio
async def test_batch_runner_dry_run_does_not_require_submitter(tmp_path: Path) -> None:
    registry = SandboxProviderRegistry()
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="dry",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="desktop", query="Do a safe task")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})

    result = await BatchRunner(provider_registry=registry).run(
        BatchRunConfig(manifest=manifest, output_dir=tmp_path, dry_run=True)
    )

    assert result.succeeded == 1
    assert result.task_states[0].device_id.startswith("batch-dry-")


@pytest.mark.asyncio
async def test_batch_runner_skips_previously_succeeded_tasks(tmp_path: Path) -> None:
    registry = SandboxProviderRegistry()
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="resume",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Already done")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    runner = BatchRunner(provider_registry=registry, submitter_factory=lambda url: FakeSubmitter(url))

    first = await runner.run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))
    second = await runner.run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))

    assert first.succeeded == 1
    assert second.succeeded == 1
    assert second.task_states[0].attempts == 1


@pytest.mark.asyncio
async def test_batch_runner_migrates_legacy_jsonl_once_when_sqlite_is_missing(tmp_path: Path) -> None:
    registry = SandboxProviderRegistry()
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="sqlite-only",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Run from SQLite")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    batch_dir = tmp_path / "sqlite-only"
    batch_dir.mkdir()
    batch_dir.joinpath("batch_state.jsonl").write_text(
        BatchTaskState(task_id="t1", status=BatchTaskStatus.SUCCEEDED, attempts=1).model_dump_json() + "\n",
        encoding="utf-8",
    )
    CountingSubmitter.submitted_batch_task_ids = []

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: CountingSubmitter(url),
    ).run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))

    assert result.succeeded == 1
    assert CountingSubmitter.submitted_batch_task_ids == []
    assert result.task_states[0].status == BatchTaskStatus.SUCCEEDED
    assert result.task_states[0].attempts == 1


@pytest.mark.asyncio
async def test_batch_runner_ignores_legacy_jsonl_after_sqlite_exists(tmp_path: Path) -> None:
    registry = SandboxProviderRegistry()
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="sqlite-authoritative",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Run from SQLite")],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    batch_dir = tmp_path / "sqlite-authoritative"
    batch_dir.mkdir()
    batch_dir.joinpath("batch_state.jsonl").write_text(
        BatchTaskState(task_id="t1", status=BatchTaskStatus.SUCCEEDED, attempts=1).model_dump_json() + "\n",
        encoding="utf-8",
    )
    from northcua.batch.state import SQLiteBatchStateRepository

    SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite").initialize_manifest(manifest)
    batch_dir.joinpath("manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    CountingSubmitter.submitted_batch_task_ids = []

    result = await BatchRunner(
        provider_registry=registry,
        submitter_factory=lambda url: CountingSubmitter(url),
    ).run(BatchRunConfig(manifest=manifest, output_dir=tmp_path))

    assert result.succeeded == 1
    assert CountingSubmitter.submitted_batch_task_ids == ["t1"]


@pytest.mark.asyncio
async def test_batch_runner_does_not_reclaim_exhausted_failed_task(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="failed-once",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Always fail", max_retries=0)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})
    FailingSubmitter.submitted_batch_task_ids = []

    result = await asyncio.wait_for(
        BatchRunner(provider_registry=registry, submitter_factory=lambda url: FailingSubmitter(url)).run(
            BatchRunConfig(manifest=manifest, output_dir=tmp_path, concurrency=2)
        ),
        timeout=5,
    )

    assert result.failed == 1
    assert result.task_states[0].attempts == 1
    assert result.task_states[0].error == "intentional failure"
    assert FailingSubmitter.submitted_batch_task_ids == ["t1"]


@pytest.mark.asyncio
async def test_batch_runner_cleanup_preserves_exception_terminal_state(tmp_path: Path) -> None:
    from northcua.sandboxes.provider.fake import FakeSandboxProvider

    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)
    manifest = BatchManifest(
        batch_id="raised-once",
        tasks=[BatchTaskSpec(task_id="t1", agent_kind="web", query="Raise during wait", max_retries=0)],
    )
    manifest.defaults.sandbox = manifest.defaults.sandbox.model_copy(update={"provider": "fake", "template": "ubuntu"})

    result = await asyncio.wait_for(
        BatchRunner(provider_registry=registry, submitter_factory=lambda url: RaisingSubmitter(url)).run(
            BatchRunConfig(manifest=manifest, output_dir=tmp_path, concurrency=1)
        ),
        timeout=5,
    )

    assert result.failed == 1
    assert result.task_states[0].status.value == "failed"
    snapshots = [
        json.loads(line)
        for line in (tmp_path / "raised-once" / "case_snapshots.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert snapshots[-1]["status"] == "failed"
    assert snapshots[-1]["cleanup_status"] == "ok"
    assert snapshots[-1]["failure_category"] == "controller_submit"


def test_submit_jitter_overshoot_triggers_secondary_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    from northcua.batch.planner import PlannedTaskSubmission
    from northcua.batch.runner import controller as controller_module

    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_JITTER_OVERSHOOT_REJITTER_THRESHOLD_SECONDS", "1")
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_JITTER_OVERSHOOT_REJITTER_SECONDS", "10")

    extra_delay, overshoot = controller_module._submit_jitter_overshoot_rejitter_seconds(
        planned,
        requested_jitter_seconds=60,
        planned_delay_seconds=2,
        observed_sleep_seconds=8,
    )

    assert overshoot == 6
    assert extra_delay == controller_module._stable_submit_jitter_seconds(planned, 10)
    assert 0 <= extra_delay <= 10


def test_submit_jitter_overshoot_observes_without_delay_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from northcua.batch.planner import PlannedTaskSubmission
    from northcua.batch.runner import controller as controller_module

    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )
    monkeypatch.delenv("NORTHCUA_BATCH_SUBMIT_JITTER_OVERSHOOT_REJITTER_SECONDS", raising=False)
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_JITTER_OVERSHOOT_REJITTER_THRESHOLD_SECONDS", "1")

    extra_delay, overshoot = controller_module._submit_jitter_overshoot_rejitter_seconds(
        planned,
        requested_jitter_seconds=60,
        planned_delay_seconds=2,
        observed_sleep_seconds=8,
    )

    assert overshoot == 6
    assert extra_delay == 0


def test_submit_jitter_overshoot_ignores_small_scheduler_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    from northcua.batch.planner import PlannedTaskSubmission
    from northcua.batch.runner import controller as controller_module

    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )
    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_JITTER_OVERSHOOT_REJITTER_THRESHOLD_SECONDS", "5")

    extra_delay, overshoot = controller_module._submit_jitter_overshoot_rejitter_seconds(
        planned,
        requested_jitter_seconds=60,
        planned_delay_seconds=2,
        observed_sleep_seconds=4,
    )

    assert overshoot == 2
    assert extra_delay == 0


@pytest.mark.asyncio
async def test_overshot_submit_open_limiter_rechecks_after_its_own_sleep_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from northcua.batch.planner import PlannedTaskSubmission
    from northcua.batch.runner import controller as controller_module

    planned = PlannedTaskSubmission(
        controller_task_id="controller-task",
        trace_id="trace",
        agent_kind="desktop",
        request="finish",
        target_device_id="device",
        required_capabilities=set(),
        input_artifacts=[],
        metadata={},
    )
    watcher = ControllerTaskWatcher(lambda url: FakeSubmitter(url))
    now = 100.0
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return now

    async def fake_sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        # Simulate the limiter's own sleep being delayed by a blocked event loop.
        now += delay + 10.0

    monkeypatch.setenv("NORTHCUA_BATCH_SUBMIT_JITTER_OVERSHOOT_OPEN_INTERVAL_SECONDS", "1")
    monkeypatch.setattr(controller_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(controller_module.asyncio, "sleep", fake_sleep)

    await watcher._rate_limit_overshot_submit_open(planned, overshoot_seconds=20)
    assert now == 100.0
    assert sleeps == []

    await watcher._rate_limit_overshot_submit_open(planned, overshoot_seconds=20)

    assert sleeps == [1.0]
    assert now == 111.0
    assert watcher._next_overshoot_open_at == 112.0
