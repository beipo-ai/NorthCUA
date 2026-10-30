from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from northcua.batch.contracts import (
    BatchDefaults,
    BatchManifest,
    BatchRunConfig,
    BatchTaskSpec,
    BatchTaskState,
    BatchTaskStatus,
)
from northcua.batch.runner.attempt_failure import _failure_category_for_attempt
from northcua.batch.runner.attempt_helpers import PreSubmitCommandFailure, _run_pre_submit_commands
from northcua.batch.runner.events import BatchAttemptEventMessage
from northcua.batch.runner.leases import BatchTaskLeaseContext
from northcua.sandboxes.contracts import SandboxCommandResult, SandboxCommandSpec, SandboxHandle, SandboxLease
from northcua.telemetry.contracts import FailureCategory


class FakeCommandManager:
    def __init__(self, result: SandboxCommandResult | None = None) -> None:
        self.result = result or SandboxCommandResult(stdout="ok")
        self.commands: list[SandboxCommandSpec] = []

    async def run_command(self, lease: SandboxLease, command: SandboxCommandSpec) -> SandboxCommandResult:
        _ = lease
        self.commands.append(command)
        return self.result


class FakeEvents:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def attempt_event(self, **kwargs: object) -> None:
        self.events.append(dict(kwargs))

    def phase_timer(self, **kwargs: object):
        _ = kwargs
        return contextlib.nullcontext()


def _lease() -> SandboxLease:
    return SandboxLease(
        lease_id="lease-1",
        handle=SandboxHandle(sandbox_id="sandbox-1", provider="fake", template="template"),
        device_id="device-1",
    )


@pytest.mark.asyncio
async def test_pre_submit_commands_run_after_readiness_before_controller_submit(tmp_path: Path) -> None:
    default_command = SandboxCommandSpec(command="echo default", label="default")
    task_command = SandboxCommandSpec(command="echo task", label="task", allow_failure=True)
    task = BatchTaskSpec(
        task_id="case-1",
        agent_kind="desktop",
        query="do it",
        pre_submit_commands=[task_command],
    )
    config = BatchRunConfig(
        manifest=BatchManifest(
            batch_id="batch-1",
            defaults=BatchDefaults(pre_submit_commands=[default_command]),
            tasks=[task],
        ),
        output_dir=tmp_path,
    )
    lease = _lease()
    manager = FakeCommandManager()
    events = FakeEvents()

    await _run_pre_submit_commands(
        config=config,
        task=task,
        lease=lease,
        lease_context=BatchTaskLeaseContext(lease=lease, manager=manager),
        attempt=1,
        events=events,
    )

    assert manager.commands == [default_command, task_command]
    assert [event["message"] for event in events.events] == [
        BatchAttemptEventMessage.PRE_SUBMIT_COMMANDS_STARTED,
        BatchAttemptEventMessage.PRE_SUBMIT_COMMANDS_FINISHED,
    ]
    assert events.events[-1]["details"]["completed_count"] == 2


@pytest.mark.asyncio
async def test_pre_submit_commands_raise_on_required_failure(tmp_path: Path) -> None:
    command = SandboxCommandSpec(command="false", label="required")
    task = BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", pre_submit_commands=[command])
    config = BatchRunConfig(manifest=BatchManifest(batch_id="batch-1", tasks=[task]), output_dir=tmp_path)
    lease = _lease()
    manager = FakeCommandManager(SandboxCommandResult(exit_code=1, stderr="boom"))
    events = FakeEvents()

    with pytest.raises(PreSubmitCommandFailure, match="pre-submit command failed") as exc_info:
        await _run_pre_submit_commands(
            config=config,
            task=task,
            lease=lease,
            lease_context=BatchTaskLeaseContext(lease=lease, manager=manager),
            attempt=1,
            events=events,
        )

    assert events.events[-1]["message"] == BatchAttemptEventMessage.PRE_SUBMIT_COMMANDS_FINISHED
    assert events.events[-1]["details"]["results"][0]["ok"] is False
    assert (
        _failure_category_for_attempt(
            state=BatchTaskState(task_id="case-1", attempts=1, status=BatchTaskStatus.WAITING_DEVICE),
            status=BatchTaskStatus.FAILED,
            error=exc_info.value,
        )
        == FailureCategory.STARTUP_COMMAND
    )


@pytest.mark.asyncio
async def test_pre_submit_commands_skip_without_live_lease(tmp_path: Path) -> None:
    command = SandboxCommandSpec(command="echo nope", label="skip")
    task = BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="do it", pre_submit_commands=[command])
    config = BatchRunConfig(manifest=BatchManifest(batch_id="batch-1", tasks=[task]), output_dir=tmp_path)
    events = FakeEvents()

    await _run_pre_submit_commands(
        config=config,
        task=task,
        lease=None,
        lease_context=None,
        attempt=1,
        events=events,
    )

    assert events.events == []
