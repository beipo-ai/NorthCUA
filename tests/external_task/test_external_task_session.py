from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr

from northcua.external_task import (
    ExternalTaskDeviceBinding,
    ExternalTaskFactory,
    ExternalTaskObservationState,
    ExternalTaskSpec,
)
from northcua.submitter.specs import DesktopSubmitSpec, WebSubmitSpec
from northcua_protocol.capabilities import Capability
from northcua_protocol.payloads import (
    TaskAccepted,
    TaskCancel,
    TaskFinished,
    TaskRequest,
    TaskStatusSnapshot,
    task_request_fingerprint_sha256,
)
from northcua_protocol.state import TaskState

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


class _ControllerFake:
    def __init__(
        self,
        *,
        lose_acceptance: bool = False,
        foreign_snapshot: bool = False,
        intent_path: Path | None = None,
        status_state: TaskState | None = TaskState.ACCEPTED,
        status_error: bool = False,
    ) -> None:
        self.lose_acceptance = lose_acceptance
        self.foreign_snapshot = foreign_snapshot
        self.connected = 0
        self.closed = 0
        self.cancelled: list[TaskCancel] = []
        self.request: TaskRequest | None = None
        self.accepted: TaskAccepted | None = None
        self.intent_path = intent_path
        self.status_state = status_state
        self.status_error = status_error
        self.submit_calls = 0

    async def connect(self) -> None:
        self.connected += 1

    async def close(self) -> None:
        self.closed += 1

    async def submit_task(self, **kwargs: object) -> TaskAccepted:
        self.submit_calls += 1
        if self.intent_path is not None:
            assert self.intent_path.is_file()
        self.request = TaskRequest.model_validate(kwargs)
        self.accepted = TaskAccepted(
            task_id=self.request.task_id,
            trace_id=self.request.trace_id,
            session_id="session-1",
        )
        if self.lose_acceptance:
            raise ConnectionError("acceptance response lost")
        return self.accepted

    async def wait_for_finished(self, task_id: str, *, timeout_seconds: float | None = None) -> TaskFinished:
        _ = timeout_seconds
        assert self.request is not None
        return TaskFinished(
            task_id=task_id,
            trace_id=self.request.trace_id,
            session_id="session-1",
            state=TaskState.FINISHED,
            device_id=self.request.target_device_id,
            result={"steps": 1},
        )

    async def task_statuses(self, task_ids: list[str], *, token: str = "") -> list[TaskStatusSnapshot]:
        _ = token
        if self.status_error:
            raise ConnectionError("private controller transport detail")
        assert self.request is not None
        assert task_ids == [self.request.task_id]
        if self.status_state is None:
            return []
        fingerprint = task_request_fingerprint_sha256(self.request)
        if self.foreign_snapshot:
            fingerprint = SHA_C
        return [
            TaskStatusSnapshot(
                task_id=self.request.task_id,
                trace_id=self.request.trace_id,
                session_id="session-1",
                agent_kind=self.request.agent_kind,
                state=self.status_state,
                device_id=self.request.target_device_id,
                request=self.request.request,
                metadata=dict(self.request.metadata),
                request_fingerprint_sha256=fingerprint,
            )
        ]

    async def cancel_task(self, cancel: TaskCancel) -> None:
        self.cancelled.append(cancel)


def _spec(tmp_path: Path, *, web: bool = False) -> ExternalTaskSpec:
    binding = ExternalTaskDeviceBinding(
        runtime_session_ref="runtime://session-1",
        device_id="device-1",
        bootstrap_operation_id="bootstrap-1",
        bootstrap_plan_sha256=SHA_A,
        registration_receipt_ref="artifact://registration",
        registration_receipt_sha256=SHA_B,
    )
    return ExternalTaskSpec(
        controller_url="ws://controller.test",
        trace_id="trace-1",
        output_dir=tmp_path,
        device=binding,
        desktop=None if web else DesktopSubmitSpec(query="do desktop task", session_id="agent-session", max_steps=3),
        web=WebSubmitSpec(query="do web task", url="https://example.test", max_steps=4) if web else None,
        policy_route_ref="policy-route://member-1",
        source_projection_sha256=SHA_C,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("web", [False, True])
async def test_external_task_desktop_and_web_submit_collect_without_runtime_ownership(
    tmp_path: Path, web: bool
) -> None:
    client = _ControllerFake()
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(_spec(tmp_path, web=web))

    accepted = await session.submit(task_id="task-1")
    terminal = await session.wait_terminal(timeout_seconds=1)
    await session.close_transport()

    assert accepted.session_id == "session-1"
    assert terminal.state == TaskState.FINISHED.value
    assert terminal.controller_terminal_fact.sha256
    assert terminal.execution_fact.sha256
    assert client.request is not None
    assert client.request.agent_kind == ("web" if web else "desktop")
    assert client.request.target_device_id == "device-1"
    assert client.request.metadata["evaluation_mode"] == "external_script"
    assert client.request.metadata["policy_route_ref"] == "policy-route://member-1"
    assert client.cancelled == []
    assert not hasattr(session, "release")


@pytest.mark.asyncio
async def test_acceptance_response_loss_recovers_only_exact_request(tmp_path: Path) -> None:
    client = _ControllerFake(lose_acceptance=True)
    spec = _spec(tmp_path)
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(spec)

    accepted = await session.submit(task_id="task-1")
    restarted = await ExternalTaskFactory(controller_client_factory=lambda _url: client).recover(spec, task_id="task-1")

    assert accepted.session_id == "session-1"
    assert (tmp_path / "external-tasks" / "task-1" / "intent.json").is_file()
    await restarted.cancel(reason="test cancellation")
    assert client.cancelled[0].task_id == "task-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("controller_state", "expected"),
    [
        (None, ExternalTaskObservationState.ABSENT),
        (TaskState.ACCEPTED, ExternalTaskObservationState.ACTIVE),
        (TaskState.FINISHED, ExternalTaskObservationState.TERMINAL),
    ],
)
async def test_recovered_session_observes_without_resubmitting(
    tmp_path: Path,
    controller_state: TaskState | None,
    expected: ExternalTaskObservationState,
) -> None:
    client = _ControllerFake(status_state=controller_state)
    spec = _spec(tmp_path)
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(spec)
    await session.submit(task_id="task-1")
    restarted = await ExternalTaskFactory(controller_client_factory=lambda _url: client).recover(spec, task_id="task-1")

    observed = await restarted.observe()

    assert observed.state is expected
    assert (observed.terminal is not None) == (expected is ExternalTaskObservationState.TERMINAL)
    assert client.submit_calls == 1


@pytest.mark.asyncio
async def test_recovery_repairs_acceptance_commit_window_and_waits_terminal(tmp_path: Path) -> None:
    client = _ControllerFake(status_state=TaskState.ACCEPTED)
    spec = _spec(tmp_path)
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(spec)
    await session.submit(task_id="task-1")
    intent_path = tmp_path / "external-tasks" / "task-1" / "intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["accepted"] = None
    intent_path.write_text(json.dumps(intent), encoding="utf-8")

    restarted = await ExternalTaskFactory(controller_client_factory=lambda _url: client).recover(spec, task_id="task-1")
    observed = await restarted.observe()

    assert observed.state is ExternalTaskObservationState.ACTIVE
    repaired = json.loads(intent_path.read_text(encoding="utf-8"))
    assert repaired["accepted"]["session_id"] == "session-1"
    client.status_state = TaskState.FINISHED
    terminal = await restarted.wait_terminal(timeout_seconds=1)
    assert terminal.state == TaskState.FINISHED.value
    assert client.submit_calls == 1


@pytest.mark.asyncio
async def test_recovered_session_observation_transport_failure_is_unknown(tmp_path: Path) -> None:
    client = _ControllerFake()
    spec = _spec(tmp_path)
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(spec)
    await session.submit(task_id="task-1")
    client.status_error = True
    restarted = await ExternalTaskFactory(controller_client_factory=lambda _url: client).recover(spec, task_id="task-1")

    observed = await restarted.observe()

    assert observed.state is ExternalTaskObservationState.UNKNOWN
    assert client.submit_calls == 1


@pytest.mark.asyncio
async def test_factory_recovers_from_secret_free_intent(tmp_path: Path) -> None:
    client = _ControllerFake()
    spec = _spec(tmp_path).model_copy(update={"controller_token": SecretStr("first-private-token")})
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(spec)
    await session.submit(task_id="task-1")
    intent_bytes = (tmp_path / "external-tasks" / "task-1" / "intent.json").read_bytes()

    restarted = await ExternalTaskFactory(controller_client_factory=lambda _url: client).recover_from_intent(
        output_dir=tmp_path,
        task_id="task-1",
        controller_token=SecretStr("fresh-private-token"),
    )

    assert (await restarted.observe()).state is ExternalTaskObservationState.ACTIVE
    assert b"first-private-token" not in intent_bytes
    assert b"fresh-private-token" not in intent_bytes
    assert client.submit_calls == 1


@pytest.mark.asyncio
async def test_submit_intent_is_durable_before_controller_network_side_effect(tmp_path: Path) -> None:
    intent_path = tmp_path / "external-tasks" / "task-1" / "intent.json"
    client = _ControllerFake(intent_path=intent_path)
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(_spec(tmp_path))

    await session.submit(task_id="task-1")

    assert intent_path.is_file()


@pytest.mark.asyncio
async def test_recovery_validates_terminal_identity_and_fact_bytes(tmp_path: Path) -> None:
    client = _ControllerFake()
    spec = _spec(tmp_path)
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(spec)
    await session.submit(task_id="task-1")
    original = await session.wait_terminal(timeout_seconds=1)

    recovered = await ExternalTaskFactory(controller_client_factory=lambda _url: client).recover(spec, task_id="task-1")
    assert await recovered.collect() == original

    terminal_path = tmp_path / "external-tasks" / "task-1" / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["task_id"] = "foreign-task"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from durable intent"):
        await ExternalTaskFactory(controller_client_factory=lambda _url: client).recover(spec, task_id="task-1")


@pytest.mark.asyncio
async def test_recovery_rejects_tampered_or_missing_terminal_fact(tmp_path: Path) -> None:
    client = _ControllerFake()
    spec = _spec(tmp_path)
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(spec)
    await session.submit(task_id="task-1")
    await session.wait_terminal(timeout_seconds=1)
    fact = tmp_path / "external-tasks" / "task-1" / "facts" / "execution-fact.json"
    fact.write_text('{"task_id":"foreign"}', encoding="utf-8")

    with pytest.raises(ValueError, match="fact integrity mismatch"):
        await ExternalTaskFactory(controller_client_factory=lambda _url: client).recover(spec, task_id="task-1")

    fact.unlink()
    with pytest.raises(FileNotFoundError):
        await ExternalTaskFactory(controller_client_factory=lambda _url: client).recover(spec, task_id="task-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["artifact_refs", "trace_refs", "public_error", "terminal_decision"])
async def test_recovery_rejects_tampered_terminal_projected_fields(tmp_path: Path, field: str) -> None:
    client = _ControllerFake()
    spec = _spec(tmp_path)
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(spec)
    await session.submit(task_id="task-1")
    await session.wait_terminal(timeout_seconds=1)
    terminal_path = tmp_path / "external-tasks" / "task-1" / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal[field] = (
        "forged-public-error"
        if field == "public_error"
        else {"decision": "forged"}
        if field == "terminal_decision"
        else ["artifact://foreign"]
    )
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicts|unbound"):
        await ExternalTaskFactory(controller_client_factory=lambda _url: client).recover(spec, task_id="task-1")


@pytest.mark.asyncio
async def test_acceptance_response_loss_rejects_foreign_duplicate(tmp_path: Path) -> None:
    client = _ControllerFake(lose_acceptance=True, foreign_snapshot=True)
    session = await ExternalTaskFactory(controller_client_factory=lambda _url: client).open(_spec(tmp_path))

    with pytest.raises(RuntimeError, match="could not be reconciled"):
        await session.submit(task_id="task-1")


def test_external_contract_rejects_agent_or_device_ambiguity(tmp_path: Path) -> None:
    base = _spec(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        ExternalTaskSpec.model_validate({**base.model_dump(), "web": WebSubmitSpec(query="also web")})
    with pytest.raises(ValueError, match="target device"):
        ExternalTaskSpec.model_validate(
            {
                **base.model_dump(),
                "desktop": DesktopSubmitSpec(query="task", target_device_id="some-other-device"),
            }
        )


def test_request_fingerprint_is_secret_free_and_capability_order_independent() -> None:
    common = {
        "task_id": "task-1",
        "trace_id": "trace-1",
        "agent_kind": "desktop",
        "request": "task",
        "target_device_id": "device-1",
    }
    first = TaskRequest(
        **common,
        token="first-secret",
        required_capabilities={Capability.DESKTOP_INPUT, Capability.DESKTOP_SCREENSHOT},
    )
    second = TaskRequest(
        **common,
        token="different-secret",
        required_capabilities={Capability.DESKTOP_SCREENSHOT, Capability.DESKTOP_INPUT},
    )

    assert task_request_fingerprint_sha256(first) == task_request_fingerprint_sha256(second)


def test_external_import_does_not_load_provider_or_batch_scheduler() -> None:
    script = (
        "import sys; import northcua.external_task; "
        "assert 'northcua.sandboxes.provider.e2b' not in sys.modules; "
        "assert 'northcua.batch.runner.service' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
