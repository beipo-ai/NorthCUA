from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from northcua.batch.contracts import BatchManifest, BatchRunConfig, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.post_run import PostRunArtifactSpec, PostRunHookRunner, PostRunHookSpec, PostRunHookStatus
from northcua.batch.runner.attempt_helpers import _run_post_run_hooks
from northcua.batch.runner.events import BatchAttemptEventMessage
from northcua.batch.runner.leases import BatchTaskLeaseContext
from northcua.sandboxes.contracts import SandboxCommandResult, SandboxFileUploadSpec, SandboxHandle, SandboxLease
from northcua_protocol.state import TaskState


class FakeHookManager:
    def __init__(self, result: SandboxCommandResult, *, fail_downloads: bool = False) -> None:
        self.result = result
        self.fail_downloads = fail_downloads
        self.commands: list[str] = []
        self.downloads: list[tuple[str, Path]] = []
        self.staged: list[SandboxFileUploadSpec] = []

    async def run_command(self, lease: SandboxLease, command) -> SandboxCommandResult:
        self.commands.append(command.command)
        return self.result

    async def download_file(self, lease: SandboxLease, *, remote: str, local: Path) -> None:
        self.downloads.append((remote, local))
        if self.fail_downloads:
            raise FileNotFoundError(remote)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text('{"score": 0.25, "passed": false}\n', encoding="utf-8")

    async def stage_evaluator_files(
        self,
        lease: SandboxLease,
        uploads: list[SandboxFileUploadSpec],
    ) -> None:
        del lease
        self.staged.extend(uploads)


class LocalShellHookManager(FakeHookManager):
    async def run_command(self, lease: SandboxLease, command) -> SandboxCommandResult:
        del lease
        self.commands.append(command.command)
        completed = subprocess.run(command.command, shell=True, check=False, capture_output=True, text=True)
        return SandboxCommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class FakeEvents:
    def __init__(self, *, fail_on_message: BatchAttemptEventMessage | None = None) -> None:
        self.events: list[dict[str, object]] = []
        self.fail_on_message = fail_on_message

    def attempt_event(self, **kwargs: object) -> None:
        if kwargs.get("message") == self.fail_on_message:
            raise RuntimeError(f"failed to persist {self.fail_on_message.value}")
        self.events.append(dict(kwargs))

    def phase_timer(self, **kwargs: object):
        return contextlib.nullcontext()


def _lease() -> SandboxLease:
    return SandboxLease(
        lease_id="lease-1",
        handle=SandboxHandle(sandbox_id="sandbox-1", provider="fake", template="template"),
        device_id="device-1",
    )


@pytest.mark.asyncio
async def test_post_run_hook_runner_parses_json_and_collects_artifacts(tmp_path: Path) -> None:
    manager = FakeHookManager(
        SandboxCommandResult(stdout='intermediate log\n{"score": 0.75, "passed": true, "details": {"ok": true}}\n')
    )
    hook = PostRunHookSpec(
        hook_id="reward",
        command="/tmp/reward.py",
        artifacts=[PostRunArtifactSpec(remote_path="/tmp/reward.json", name="reward.json")],
    )

    result = await PostRunHookRunner(output_dir=tmp_path).run_one(
        hook=hook,
        manager=manager,
        lease=_lease(),
        task_id="case-1",
        attempt=2,
    )

    assert result.status == PostRunHookStatus.PASSED
    assert result.score == 0.75
    assert result.passed is True
    assert result.artifacts == {"/tmp/reward.json": str(tmp_path / "post-run/case-1/attempt-2/reward/reward.json")}
    assert manager.commands == ["/tmp/reward.py"]


@pytest.mark.asyncio
async def test_post_run_hook_retries_transport_errors_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FlakyHookManager(FakeHookManager):
        def __init__(self) -> None:
            super().__init__(SandboxCommandResult(stdout='{"score": 1, "passed": true}\n'))
            self.calls = 0

        async def run_command(self, lease: SandboxLease, command) -> SandboxCommandResult:
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("temporary evaluator transport failure")
            return await super().run_command(lease, command)

    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("northcua.batch.post_run.runner.asyncio.sleep", record_sleep)
    manager = FlakyHookManager()
    hook = PostRunHookSpec(
        hook_id="reward",
        command="/tmp/reward.py",
        infra_retries=2,
        infra_retry_backoff_seconds=0.25,
    )

    result = await PostRunHookRunner(output_dir=tmp_path).run_one(
        hook=hook,
        manager=manager,
        lease=_lease(),
        task_id="case-1",
        attempt=1,
    )

    assert result.status is PostRunHookStatus.PASSED
    assert manager.calls == 3
    assert sleeps == [0.25, 0.25]


@pytest.mark.asyncio
async def test_legacy_manager_argv_always_uses_shell_false_trampoline(tmp_path: Path) -> None:
    manager = LocalShellHookManager(SandboxCommandResult())
    marker = tmp_path / "must-not-exist"
    malicious = f"value; touch {marker} $(id)"
    hook = PostRunHookSpec(
        hook_id="reward",
        argv=[
            sys.executable,
            "-c",
            "import json,sys; print(json.dumps({'passed': sys.argv[1].startswith('value;')}))",
            malicious,
        ],
    )

    result = await PostRunHookRunner(output_dir=tmp_path).run_one(
        hook=hook,
        manager=manager,
        lease=_lease(),
        task_id="case-1",
        attempt=1,
    )

    assert result.passed is True
    assert not marker.exists()
    assert malicious not in manager.commands[0]


@pytest.mark.asyncio
async def test_argv_hook_stages_evaluator_files_after_agent(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("print('ok')\n", encoding="utf-8")
    manager = FakeHookManager(SandboxCommandResult(stdout='{"score": 1, "passed": true}\n'))
    upload = SandboxFileUploadSpec(
        local_path=evaluator,
        remote_path="/home/user/.nexverse/evaluator.py",
        visibility="evaluator",
    )
    hook = PostRunHookSpec(
        hook_id="reward",
        argv=["python3", "/home/user/.nexverse/evaluator.py"],
        file_uploads=[upload],
    )

    result = await PostRunHookRunner(output_dir=tmp_path).run_one(
        hook=hook,
        manager=manager,
        lease=_lease(),
        task_id="case-1",
        attempt=1,
    )

    assert result.passed is True
    assert manager.staged == [upload]
    assert len(manager.commands) == 1


@pytest.mark.asyncio
async def test_post_run_hooks_preserve_controller_terminal_status(tmp_path: Path) -> None:
    manager = FakeHookManager(SandboxCommandResult(stdout='{"score": 0.25, "passed": false}\n'))
    lease = _lease()
    events = FakeEvents()
    hook = PostRunHookSpec(hook_id="reward", command="/tmp/reward.py")
    task = BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", post_run_hooks=[hook])
    config = BatchRunConfig(manifest=BatchManifest(batch_id="batch-1", tasks=[task]), output_dir=tmp_path)
    state = BatchTaskState(
        task_id="case-1",
        attempts=1,
        status=BatchTaskStatus.SUCCEEDED,
        terminal_state=TaskState.FINISHED,
        controller_task_id="controller-1",
        trace_id="trace-1",
        sandbox_id="sandbox-1",
        device_id="device-1",
    )

    updated = await _run_post_run_hooks(
        config=config,
        task=task,
        state=state,
        lease=lease,
        lease_context=BatchTaskLeaseContext(lease=lease, manager=manager),
        output_dir=tmp_path,
        attempt=1,
        events=events,
    )

    assert updated.status == BatchTaskStatus.SUCCEEDED
    assert updated.post_run_results[0].status == PostRunHookStatus.FAILED
    assert updated.post_run_results[0].score == 0.25
    assert [event["message"] for event in events.events] == [
        BatchAttemptEventMessage.POST_RUN_HOOK_STARTED,
        BatchAttemptEventMessage.POST_RUN_HOOK_FINISHED,
    ]
    assert events.events[-1]["details"]["failed_count"] == 1


@pytest.mark.asyncio
async def test_post_run_hook_can_receive_the_terminal_decision_after_agent(tmp_path: Path) -> None:
    manager = FakeHookManager(SandboxCommandResult(stdout='{"score": 1, "passed": true}\n'))
    lease = _lease()
    events = FakeEvents()
    hook = PostRunHookSpec(
        hook_id="reward",
        argv=["python3", "/home/user/evaluator.py"],
        terminal_decision_remote_path="/home/user/.nexverse/terminal-decision.json",
    )
    task = BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", post_run_hooks=[hook])
    config = BatchRunConfig(manifest=BatchManifest(batch_id="batch-1", tasks=[task]), output_dir=tmp_path)
    state = BatchTaskState(
        task_id="case-1",
        attempts=1,
        status=BatchTaskStatus.SUCCEEDED,
        terminal_state=TaskState.FINISHED,
        terminal_decision={"action": "fail", "reason": "infeasible"},
    )

    updated = await _run_post_run_hooks(
        config=config,
        task=task,
        state=state,
        lease=lease,
        lease_context=BatchTaskLeaseContext(lease=lease, manager=manager),
        output_dir=tmp_path,
        attempt=1,
        events=events,
    )

    assert updated.post_run_results[0].passed is True
    [upload] = manager.staged
    assert upload.remote_path == "/home/user/.nexverse/terminal-decision.json"
    assert upload.local_path is not None
    payload = json.loads(upload.local_path.read_text(encoding="utf-8"))
    assert payload["terminal_decision"] == {"action": "fail", "reason": "infeasible"}


@pytest.mark.asyncio
async def test_post_run_hooks_skip_pre_terminal_controller_failures(tmp_path: Path) -> None:
    manager = FakeHookManager(SandboxCommandResult(stdout='{"score": 1.0, "passed": true}\n'))
    lease = _lease()
    events = FakeEvents()
    hook = PostRunHookSpec(hook_id="reward", command="/tmp/reward.py")
    task = BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", post_run_hooks=[hook])
    config = BatchRunConfig(manifest=BatchManifest(batch_id="batch-1", tasks=[task]), output_dir=tmp_path)
    state = BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.FAILED, error="submit failed")

    updated = await _run_post_run_hooks(
        config=config,
        task=task,
        state=state,
        lease=lease,
        lease_context=BatchTaskLeaseContext(lease=lease, manager=manager),
        output_dir=tmp_path,
        attempt=1,
        events=events,
    )

    assert updated.post_run_results == []
    assert manager.commands == []
    assert events.events == []


@pytest.mark.asyncio
async def test_post_run_hook_runner_exception_is_persisted_without_failing_controller_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_run_all(*args: object, **kwargs: object):
        _ = args, kwargs
        raise RuntimeError("runner exploded")

    monkeypatch.setattr(PostRunHookRunner, "run_all", fail_run_all)
    manager = FakeHookManager(SandboxCommandResult(stdout='{"score": 1.0, "passed": true}\n'))
    lease = _lease()
    events = FakeEvents()
    hook = PostRunHookSpec(hook_id="reward", command="/tmp/reward.py")
    task = BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", post_run_hooks=[hook])
    config = BatchRunConfig(manifest=BatchManifest(batch_id="batch-1", tasks=[task]), output_dir=tmp_path)
    state = BatchTaskState(
        task_id="case-1",
        attempts=1,
        status=BatchTaskStatus.SUCCEEDED,
        terminal_state=TaskState.FINISHED,
        controller_task_id="controller-1",
    )

    updated = await _run_post_run_hooks(
        config=config,
        task=task,
        state=state,
        lease=lease,
        lease_context=BatchTaskLeaseContext(lease=lease, manager=manager),
        output_dir=tmp_path,
        attempt=1,
        events=events,
    )

    assert updated.status == BatchTaskStatus.SUCCEEDED
    assert updated.post_run_results[0].status == PostRunHookStatus.ERROR
    assert updated.post_run_results[0].error == "runner exploded"
    assert events.events[-1]["message"] == BatchAttemptEventMessage.POST_RUN_HOOK_FINISHED
    assert events.events[-1]["details"]["error_count"] == 1


@pytest.mark.asyncio
async def test_post_run_hook_event_failure_preserves_results_and_status(tmp_path: Path) -> None:
    manager = FakeHookManager(SandboxCommandResult(stdout='{"score": 0.75, "passed": true}\n'))
    lease = _lease()
    events = FakeEvents(fail_on_message=BatchAttemptEventMessage.POST_RUN_HOOK_FINISHED)
    hook = PostRunHookSpec(hook_id="reward", command="/tmp/reward.py")
    task = BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", post_run_hooks=[hook])
    config = BatchRunConfig(manifest=BatchManifest(batch_id="batch-1", tasks=[task]), output_dir=tmp_path)
    state = BatchTaskState(
        task_id="case-1",
        attempts=1,
        status=BatchTaskStatus.SUCCEEDED,
        terminal_state=TaskState.FINISHED,
        controller_task_id="controller-1",
    )

    updated = await _run_post_run_hooks(
        config=config,
        task=task,
        state=state,
        lease=lease,
        lease_context=BatchTaskLeaseContext(lease=lease, manager=manager),
        output_dir=tmp_path,
        attempt=1,
        events=events,
    )

    assert updated.status == BatchTaskStatus.SUCCEEDED
    assert updated.post_run_results[0].status == PostRunHookStatus.PASSED
    assert updated.post_run_results[0].score == 0.75
    assert updated.post_run_results[0].metadata["infra_error"] is True
    assert updated.post_run_results[0].metadata["infra_error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_post_run_hook_runner_reports_missing_json_as_hook_error(tmp_path: Path) -> None:
    manager = FakeHookManager(SandboxCommandResult(stdout="no structured result here\n"))
    hook = PostRunHookSpec(hook_id="reward", command="/tmp/reward.py")

    result = await PostRunHookRunner(output_dir=tmp_path).run_one(
        hook=hook,
        manager=manager,
        lease=_lease(),
        task_id="case-1",
        attempt=1,
    )

    assert result.status == PostRunHookStatus.ERROR
    assert result.error == "post-run hook did not emit a JSON result"


@pytest.mark.asyncio
async def test_post_run_hook_runner_honors_explicit_error_status(tmp_path: Path) -> None:
    manager = FakeHookManager(SandboxCommandResult(stdout='{"status": "error", "error": "reward crashed"}\n'))
    hook = PostRunHookSpec(hook_id="reward", command="/tmp/reward.py")

    result = await PostRunHookRunner(output_dir=tmp_path).run_one(
        hook=hook,
        manager=manager,
        lease=_lease(),
        task_id="case-1",
        attempt=1,
    )

    assert result.status == PostRunHookStatus.ERROR
    assert result.passed is None
    assert result.error == "reward crashed"


@pytest.mark.asyncio
async def test_required_post_run_artifact_failure_marks_hook_error(tmp_path: Path) -> None:
    manager = FakeHookManager(
        SandboxCommandResult(stdout='{"score": 1.0, "passed": true}\n'),
        fail_downloads=True,
    )
    hook = PostRunHookSpec(
        hook_id="reward",
        command="/tmp/reward.py",
        artifacts=[PostRunArtifactSpec(remote_path="/tmp/reward.json", required=True)],
    )

    result = await PostRunHookRunner(output_dir=tmp_path).run_one(
        hook=hook,
        manager=manager,
        lease=_lease(),
        task_id="case-1",
        attempt=1,
    )

    assert result.status == PostRunHookStatus.ERROR
    assert result.passed is None
    assert result.artifact_errors == {"/tmp/reward.json": "/tmp/reward.json"}
    assert result.error == "required post-run artifact download failed: /tmp/reward.json"
