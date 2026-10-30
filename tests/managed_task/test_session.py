from __future__ import annotations

import json
from pathlib import Path

import pytest

from northcua.evaluation.script import ScriptEvaluatorSpec, ScriptEvaluatorStatus
from northcua.managed_task import ManagedRuntimeTrustBinding, ManagedTaskFactory, ManagedTaskSpec
from northcua.sandboxes.contracts import (
    SandboxCommandResult,
    SandboxCommandSpec,
    SandboxDeviceBinding,
    SandboxProviderSpec,
)
from northcua.sandboxes.provider.fake import FakeSandboxProvider
from northcua.sandboxes.provider.registry import SandboxProviderRegistry
from northcua.submitter.specs import DesktopSubmitSpec, WebSubmitSpec
from northcua_protocol.capabilities import Capability
from northcua_protocol.payloads import TaskAccepted, TaskFinished, TaskStatusSnapshot
from northcua_protocol.state import TaskState


class _TrackingProvider(FakeSandboxProvider):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    async def run_command(self, handle, command: SandboxCommandSpec) -> SandboxCommandResult:  # type: ignore[no-untyped-def]
        if command.command == "evaluate-task":
            self.events.append("script-evaluator")
            return SandboxCommandResult(stdout='{"passed":true,"score":1.0,"status":"passed"}')
        return await super().run_command(handle, command)

    async def terminate(self, handle) -> None:  # type: ignore[no-untyped-def]
        self.events.append("sandbox-terminate")
        await super().terminate(handle)


class _ControllerBackend:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []
        self.cancellations: list[str] = []
        self.fail_first_wait = False
        self.wait_failed = False
        self.duplicate_first_submit = False
        self.fail_before_accept = False
        self.status_missing = False

    def finished(self, *, task_id: str, trace_id: str, session_id: str) -> TaskFinished:
        return TaskFinished(
            task_id=task_id,
            trace_id=trace_id,
            session_id=session_id,
            state=TaskState.FINISHED,
            device_id="managed-device-1",
            result={"answer": "done"},
        )


class _ControllerClient:
    def __init__(self, backend: _ControllerBackend) -> None:
        self.backend = backend
        self.connected = False
        self.accepted: TaskAccepted | None = None

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        self.backend.submissions.append(dict(kwargs))
        if self.backend.fail_before_accept:
            raise ConnectionError("synthetic failure before Controller acceptance")
        self.accepted = TaskAccepted(
            task_id=str(kwargs["task_id"]),
            trace_id=str(kwargs["trace_id"] or "controller-trace-1"),
            session_id="controller-session-1",
            device_id=str(kwargs["target_device_id"]),
        )
        if self.backend.duplicate_first_submit:
            self.backend.duplicate_first_submit = False
            raise RuntimeError("controller rejected task: task_id already exists")
        return self.accepted

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        del timeout_seconds
        if self.backend.fail_first_wait and not self.backend.wait_failed:
            self.backend.wait_failed = True
            raise ConnectionError("synthetic submitter connection loss")
        assert self.accepted is not None
        return self.backend.finished(
            task_id=task_id,
            trace_id=self.accepted.trace_id,
            session_id=self.accepted.session_id,
        )

    async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
        del token
        if self.backend.status_missing:
            return []
        task_id = task_ids[0]
        submission = self.backend.submissions[-1] if self.backend.submissions else {}
        return [
            TaskStatusSnapshot(
                task_id=task_id,
                trace_id="controller-trace-1",
                session_id="controller-session-1",
                agent_kind=str(submission.get("agent_kind", "desktop")),
                state=TaskState.FINISHED,
                device_id=str(submission.get("target_device_id", "managed-device-1")),
                result={"answer": "recovered"},
                request=str(submission.get("request", "")),
                metadata=submission.get("metadata", {}),
            )
        ]

    async def cancel_task(self, cancel: object) -> None:
        self.backend.cancellations.append(str(getattr(cancel, "task_id")))


def _factory(provider: _TrackingProvider, backend: _ControllerBackend) -> ManagedTaskFactory:
    providers = SandboxProviderRegistry()
    providers.register("fake", lambda: provider)
    return ManagedTaskFactory(
        provider_registry=providers,
        controller_client_factory=lambda _url: _ControllerClient(backend),
    )


def _spec(tmp_path: Path, *, web: bool = False) -> ManagedTaskSpec:
    common = {
        "controller_url": "ws://controller.example.invalid/ws",
        "controller_token": "controller-secret",
        "trace_id": "controller-trace-1",
        "output_dir": tmp_path,
        "sandbox": SandboxProviderSpec(provider="fake", template="e2b-nex/test", desktop=not web),
        "device": SandboxDeviceBinding(
            device_id="managed-device-1",
            display_name="managed-device-1",
            server_url="wss://controller-public.example.invalid/ws",
            capabilities={Capability.BROWSER_PAGE} if web else {Capability.DESKTOP_SCREENSHOT},
            agentkit_agent_kinds={"web" if web else "desktop"},
        ),
        "evaluator": ScriptEvaluatorSpec(hook_id="score", command="evaluate-task"),
        "policy_route_ref": "northcua-policy-route://session/member-1",
        "runtime_trust": ManagedRuntimeTrustBinding(
            provider_ref="northcua.sandbox.fake",
            provider_kind="fake",
            provider_version="1.0.0",
            provider_build_digest="1" * 64,
            descriptor_digest="2" * 64,
            conformance_profile_digest="3" * 64,
            conformance_report_digest="4" * 64,
            resolved_runtime_ref="e2b-nex/test",
            resolved_runtime_digest="5" * 64,
            runtime_spec_digest="6" * 64,
            sdk_name="northcua-fake",
            sdk_version="1.0.0",
        ),
        "recovery_poll_seconds": 0.001,
    }
    if web:
        common["web"] = WebSubmitSpec(query="read the page", url="https://example.invalid")
    else:
        common["desktop"] = DesktopSubmitSpec(query="click the button", max_steps=2)
    return ManagedTaskSpec.model_validate(common)


@pytest.mark.asyncio
@pytest.mark.parametrize("web", [False, True])
async def test_one_task_evaluates_before_cleanup_and_projects_safe_route(tmp_path: Path, web: bool) -> None:
    provider = _TrackingProvider()
    backend = _ControllerBackend()
    session = await _factory(provider, backend).open(_spec(tmp_path, web=web))

    assert not provider.handles
    accepted = await session.submit(task_id="task-1")
    assert accepted.task_id == "task-1"
    assert (await session.submit(task_id="task-1")) == accepted
    terminal = await session.wait_terminal(timeout_seconds=5)
    await session.close()

    submission = backend.submissions[0]
    metadata = submission["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["evaluation_mode"] == "external_script"
    assert metadata["policy_route_ref"] == "northcua-policy-route://session/member-1"
    assert "controller-secret" not in repr(metadata)
    assert submission["target_device_id"] == "managed-device-1"
    assert submission["agent_kind"] == ("web" if web else "desktop")
    assert provider.events == ["script-evaluator", "sandbox-terminate"]
    assert terminal.script_evaluation.status is ScriptEvaluatorStatus.PASSED
    assert terminal.cleanup_receipt.sha256
    assert terminal.runtime_custody_fact.sha256
    custody_path = tmp_path / terminal.runtime_custody_fact.ref.removeprefix("agent-artifacts://path/")
    custody = json.loads(custody_path.read_text(encoding="utf-8"))
    assert custody["sandbox_id"]
    assert custody["lease_id"]
    assert custody["release_state"] == "released"
    assert (
        custody["lease_acquired_at"]
        <= custody["controller_accepted_at"]
        <= custody["release_started_at"]
        <= custody["release_finished_at"]
    )
    assert custody["cleanup_receipt_ref"] == terminal.cleanup_receipt.ref
    assert custody["cleanup_receipt_sha256"] == terminal.cleanup_receipt.sha256
    runtime_path = tmp_path / custody["runtime_session_observation_ref"].removeprefix("agent-artifacts://path/")
    release_path = tmp_path / custody["provider_release_observation_ref"].removeprefix("agent-artifacts://path/")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    release = json.loads(release_path.read_text(encoding="utf-8"))
    assert runtime["sandbox_id"] == custody["sandbox_id"]
    assert runtime["provider_build_digest"] == "1" * 64
    assert runtime["observed_at"] == custody["lease_acquired_at"]
    assert release["sandbox_id"] == custody["sandbox_id"]
    assert release["release_state"] == "released"
    assert release["observed_at"] == custody["release_finished_at"]
    assert terminal.execution_fact.ref.startswith("agent-artifacts://path/")
    assert len(provider.terminated) == 1


@pytest.mark.asyncio
async def test_fresh_connection_status_recovers_terminal_then_cleans_up(tmp_path: Path) -> None:
    provider = _TrackingProvider()
    backend = _ControllerBackend()
    backend.fail_first_wait = True
    session = await _factory(provider, backend).open(_spec(tmp_path))

    await session.submit(task_id="task-recover")
    terminal = await session.wait_terminal(timeout_seconds=5)
    await session.close()

    assert backend.wait_failed
    assert terminal.state == "finished"
    assert terminal.execution_fact.sha256
    assert provider.events == ["script-evaluator", "sandbox-terminate"]


@pytest.mark.asyncio
async def test_duplicate_submit_recovers_only_identical_durable_request(tmp_path: Path) -> None:
    provider = _TrackingProvider()
    backend = _ControllerBackend()
    backend.duplicate_first_submit = True
    session = await _factory(provider, backend).open(_spec(tmp_path))

    accepted = await session.submit(task_id="task-duplicate")
    terminal = await session.wait_terminal(timeout_seconds=5)
    await session.close()

    assert accepted.session_id == "controller-session-1"
    assert terminal.state == "finished"
    assert len(backend.submissions) == 1
    assert provider.events == ["script-evaluator", "sandbox-terminate"]


@pytest.mark.asyncio
async def test_process_restart_reconnects_durable_sandbox_and_controller_task(tmp_path: Path) -> None:
    provider = _TrackingProvider()
    backend = _ControllerBackend()
    factory = _factory(provider, backend)
    first = await factory.open(_spec(tmp_path))

    accepted = await first.submit(task_id="task-process-restart")
    custody = tmp_path / "managed-tasks" / "task-process-restart" / "custody.json"
    assert custody.is_file()
    assert accepted.session_id == "controller-session-1"

    recovered = await factory.recover(_spec(tmp_path), task_id="task-process-restart")
    terminal = await recovered.wait_terminal(timeout_seconds=5)
    await recovered.close()

    assert terminal.task_id == "task-process-restart"
    assert len(backend.submissions) == 1
    assert provider.events == ["script-evaluator", "sandbox-terminate"]
    assert len(provider.terminated) == 1
    assert '"released":true' in custody.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_process_restart_releases_and_quarantines_pre_accept_sandbox(tmp_path: Path) -> None:
    provider = _TrackingProvider()
    backend = _ControllerBackend()
    backend.fail_before_accept = True
    backend.status_missing = True
    factory = _factory(provider, backend)
    first = await factory.open(_spec(tmp_path))

    with pytest.raises(ConnectionError, match="before Controller acceptance"):
        await first.submit(task_id="task-pre-accept-crash")
    custody = tmp_path / "managed-tasks" / "task-pre-accept-crash" / "custody.json"
    assert custody.is_file()

    recovered = await factory.recover(_spec(tmp_path), task_id="task-pre-accept-crash")
    assert not recovered.has_controller_acceptance
    terminal = await recovered.abandon_before_accept(reason="old session recovery")
    await recovered.close()

    assert terminal.state == "failed"
    assert terminal.session_id == "pre-accept"
    assert terminal.script_evaluation.status is ScriptEvaluatorStatus.ERROR
    assert terminal.script_evaluation.metadata["training_disposition"] == "quarantine"
    assert len(provider.terminated) == 1
    assert '"released":true' in custody.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_close_cancels_and_releases_submitted_nonterminal_task(tmp_path: Path) -> None:
    provider = _TrackingProvider()
    backend = _ControllerBackend()
    session = await _factory(provider, backend).open(_spec(tmp_path))

    await session.submit(task_id="task-cancel")
    await session.close()

    assert backend.cancellations == ["task-cancel"]
    assert provider.events == ["sandbox-terminate"]
    assert len(provider.terminated) == 1


def test_spec_rejects_ambiguous_agent_and_allows_controller_public_ingress(tmp_path: Path) -> None:
    payload = _spec(tmp_path).model_dump()
    payload["web"] = WebSubmitSpec(query="also web")
    with pytest.raises(ValueError, match="exactly one"):
        ManagedTaskSpec.model_validate(payload)

    spec = _spec(tmp_path)
    assert spec.controller_url == "ws://controller.example.invalid/ws"
    assert spec.device.server_url == "wss://controller-public.example.invalid/ws"
