from __future__ import annotations

import asyncio
import json
import logging
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from northcua.cli.controller_command_runner import _process_runner_app_config_path, _validate_process_runner_sources
from northcua.controller.execution.executor_protocol import ControllerExecutionContext
from northcua.controller.execution.processes import worker_client
from northcua.controller.execution.processes.contracts import (
    WorkerCommandDispatchErrorCode,
    WorkerCommandDispatchResponse,
    WorkerEnvelope,
    WorkerHeartbeat,
    WorkerJobFailureKind,
    WorkerJobStatus,
    WorkerMessageType,
    WorkerRunRequest,
)
from northcua.controller.execution.processes.executor import ProcessTaskExecutorService
from northcua.controller.execution.processes.ipc import WorkerIPCError, read_envelope
from northcua.controller.execution.processes.runners import ProcessWorkerRunnerRegistry
from northcua.controller.execution.processes.supervisor import ProcessWorkerSupervisor
from northcua.controller.execution.processes.worker_client import WorkerCommandTransport
from northcua.controller.execution.processes.worker_main import (
    _execute_request,
    _execute_while_control_plane_alive,
    _heartbeat_loop,
    _progress_is_worker_heartbeat_only,
    _WorkerStartupPhaseReporter,
)
from northcua.controller.execution.runner_registry import RunnerRegistry, TaskExecutorService
from northcua.controller.storage.identifiers import command_storage_key
from northcua.controller.storage.persistence import SQLiteControllerStore
from northcua_protocol.enums import MessageType
from northcua_protocol.payloads import (
    CommandRequest,
    CommandResult,
    SessionLeaseGranted,
    TaskFinished,
    TaskRequest,
    TelemetryUpdate,
)
from northcua_protocol.state import CommandState, TaskState
from northcua_runtime_shared.validation import ToolParameterValidationError


def test_worker_heartbeat_reports_current_and_peak_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_client, "_current_rss_bytes", lambda: 123456)

    heartbeat = worker_client.worker_heartbeat(
        phase="running",
        artifact_root="/tmp/artifacts",
        active_command_id="cmd-1",
    )

    assert heartbeat.phase == "running"
    assert heartbeat.step_index is None
    assert "step_index" not in heartbeat.model_dump_json(exclude_none=True)
    assert heartbeat.artifact_root == "/tmp/artifacts"
    assert heartbeat.active_command_id == "cmd-1"
    assert heartbeat.current_rss_bytes == 123456
    assert heartbeat.peak_rss_bytes > 0
    assert heartbeat.rss_bytes == heartbeat.peak_rss_bytes


def test_worker_heartbeat_treats_legacy_zero_step_as_unknown() -> None:
    assert WorkerHeartbeat(step_index=0).step_index is None
    assert WorkerHeartbeat(step_index="0").step_index is None
    assert WorkerHeartbeat(step_index=-1).step_index is None
    assert WorkerHeartbeat(step_index=1).step_index == 1


class _DisconnectingIPC:
    def __init__(self) -> None:
        self.disconnected = asyncio.Event()

    async def wait_disconnected(self) -> None:
        await self.disconnected.wait()
        raise RuntimeError("synthetic controller disconnect")


class _FakeStreamWriter:
    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


@pytest.mark.asyncio
async def test_worker_ipc_exposes_unexpected_eof_as_disconnect() -> None:
    reader = asyncio.StreamReader()
    ipc = worker_client.WorkerIPCClient(
        reader=reader,
        writer=_FakeStreamWriter(),  # type: ignore[arg-type]
        trace_id="trace-eof",
    )
    await ipc.start()
    reader.feed_eof()

    with pytest.raises(RuntimeError, match="disconnected from controller") as exc_info:
        await ipc.wait_disconnected()
    assert isinstance(exc_info.value.__cause__, asyncio.IncompleteReadError)
    await ipc.close()


@pytest.mark.asyncio
async def test_worker_cancels_execution_when_controller_disconnects() -> None:
    ipc = _DisconnectingIPC()
    execution_started = asyncio.Event()
    execution_cancelled = asyncio.Event()

    async def execution() -> TaskFinished:
        execution_started.set()
        try:
            await asyncio.sleep(60)
        finally:
            execution_cancelled.set()
        raise AssertionError("unreachable")

    heartbeat_task = asyncio.create_task(asyncio.sleep(60))
    guarded = asyncio.create_task(
        _execute_while_control_plane_alive(
            ipc,  # type: ignore[arg-type]
            heartbeat_task,
            execution(),
        )
    )
    await execution_started.wait()
    ipc.disconnected.set()

    with pytest.raises(RuntimeError, match="synthetic controller disconnect"):
        await guarded
    assert execution_cancelled.is_set()
    assert heartbeat_task.cancelled() is False
    heartbeat_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await heartbeat_task


@pytest.mark.asyncio
async def test_worker_cancels_execution_when_heartbeat_fails() -> None:
    ipc = _DisconnectingIPC()
    execution_started = asyncio.Event()
    execution_cancelled = asyncio.Event()

    async def execution() -> TaskFinished:
        execution_started.set()
        try:
            await asyncio.sleep(60)
        finally:
            execution_cancelled.set()
        raise AssertionError("unreachable")

    async def heartbeat() -> None:
        await execution_started.wait()
        raise ConnectionError("synthetic heartbeat failure")

    with pytest.raises(ConnectionError, match="synthetic heartbeat failure"):
        await _execute_while_control_plane_alive(
            ipc,  # type: ignore[arg-type]
            asyncio.create_task(heartbeat()),
            execution(),
        )
    assert execution_cancelled.is_set()


class _FakeSDK:
    def __init__(self) -> None:
        self.commands: list[CommandRequest] = []

    async def send_command(self, command: CommandRequest, *, trace_id: str) -> CommandResult:
        del trace_id
        self.commands.append(command)
        return CommandResult(
            command_id=command.command_id,
            session_id=command.session_id,
            lease_id=command.lease_id,
            device_id=command.device_id,
            lease_generation=command.lease_generation,
            status=CommandState.SUCCEEDED,
            result={"ok": True, "tool_id": command.tool_id},
        )


class _FakeTaskService:
    def __init__(self) -> None:
        self.payloads: list[TaskFinished] = []

    async def complete_task(self, *, payload: TaskFinished, correlation_id: str | None = None) -> None:
        del correlation_id
        self.payloads.append(payload)


class _FailingTaskService:
    async def complete_task(self, *, payload: TaskFinished, correlation_id: str | None = None) -> None:
        del payload, correlation_id
        raise RuntimeError("synthetic complete_task failure")


async def _failing_progress_sink(_message_type, _payload) -> None:
    raise RuntimeError("synthetic progress sink failure")


class _CapturingIPC:
    def __init__(self) -> None:
        self.requests = []

    async def send_command(self, request):
        self.requests.append(request)
        return WorkerCommandDispatchResponse(
            request_id=request.request_id,
            result=CommandResult(
                command_id=request.command_id,
                session_id="session-transport",
                lease_id="lease-transport",
                device_id="device-transport",
                status=CommandState.SUCCEEDED,
            ),
        )


class _FailingCommandIPC:
    def __init__(self) -> None:
        self.requests = []

    async def send_command(self, request):
        self.requests.append(request)
        raise TimeoutError("synthetic command dispatch timeout")


class _ParameterValidationCommandIPC:
    async def send_command(self, request):
        return WorkerCommandDispatchResponse(
            request_id=request.request_id,
            error="command parameter 'parameters' must match exactly one schema",
            error_code=WorkerCommandDispatchErrorCode.TOOL_PARAMETER_VALIDATION,
        )


class _CapturingHeartbeatIPC:
    def __init__(self) -> None:
        self.heartbeats = []
        self.progress = []

    async def send_heartbeat(self, heartbeat):
        self.heartbeats.append((time.monotonic(), heartbeat))

    async def send_progress(self, event):
        self.progress.append(event)


class _CapturingCommandStateReporter:
    def __init__(self) -> None:
        self.events = []

    async def set_active_command(self, command_id: str) -> None:
        self.events.append(("set", command_id))

    async def clear_active_command(self, command_id: str) -> None:
        self.events.append(("clear", command_id))


class _SlowCommandStateReporter:
    async def set_active_command(self, command_id: str) -> None:
        del command_id
        await asyncio.sleep(1.0)

    async def clear_active_command(self, command_id: str) -> None:
        del command_id
        await asyncio.sleep(1.0)


def test_worker_command_transport_exposes_protocol_properties() -> None:
    lease = SessionLeaseGranted(
        lease_id="lease-transport",
        session_id="session-transport",
        task_id="task-transport",
        device_id="device-transport",
        trace_id="trace-transport",
    )
    transport = WorkerCommandTransport(
        ipc=object(),  # type: ignore[arg-type]
        lease=lease,
        metadata_overrides={"northcua.desktop.coordinate_space": "screen"},
    )

    assert transport.lease is lease
    assert transport.metadata_overrides == {"northcua.desktop.coordinate_space": "screen"}


@pytest.mark.asyncio
async def test_worker_command_transport_uses_controller_response_timeout_budget() -> None:
    lease = SessionLeaseGranted(
        lease_id="lease-transport",
        session_id="session-transport",
        task_id="task-transport",
        device_id="device-transport",
        trace_id="trace-transport",
    )
    ipc = _CapturingIPC()
    transport = WorkerCommandTransport(
        ipc=ipc,  # type: ignore[arg-type]
        lease=lease,
        command_response_timeout_seconds=330,
    )
    command = CommandRequest(
        command_id="command-short",
        idempotency_key="idem-command-short",
        session_id=lease.session_id,
        lease_id=lease.lease_id,
        device_id=lease.device_id,
        tool_id="fake.tool",
        timeout_seconds=10,
    )

    await transport.send_command(command, trace_id=lease.trace_id)

    assert ipc.requests[0].timeout_seconds == 10
    assert ipc.requests[0].response_timeout_seconds == 330


def test_process_worker_response_budget_uses_configured_timeout_floor(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    supervisor = ProcessWorkerSupervisor(
        context=_context("task-floor", fake_worker_seconds=0.0),
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        job_id="job-floor",
        attempt_id="task-floor:1",
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        start_timeout_seconds=10.0,
        heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
        command_timeout_floor_seconds=180.0,
    )

    assert supervisor._worker_command_response_timeout_seconds() == 210.0


@pytest.mark.asyncio
async def test_worker_command_transport_reports_active_command() -> None:
    lease = SessionLeaseGranted(
        lease_id="lease-transport",
        session_id="session-transport",
        task_id="task-transport",
        device_id="device-transport",
        trace_id="trace-transport",
    )
    ipc = _CapturingIPC()
    reporter = _CapturingCommandStateReporter()
    transport = WorkerCommandTransport(
        ipc=ipc,  # type: ignore[arg-type]
        lease=lease,
        heartbeat_state_reporter=reporter,
    )
    command = CommandRequest(
        command_id="command-active",
        idempotency_key="idem-command-active",
        session_id=lease.session_id,
        lease_id=lease.lease_id,
        device_id=lease.device_id,
        tool_id="fake.tool",
        timeout_seconds=10,
    )

    await transport.send_command(command, trace_id=lease.trace_id)

    assert reporter.events == [("set", "command-active"), ("clear", "command-active")]


@pytest.mark.asyncio
async def test_worker_command_transport_clears_active_command_on_dispatch_failure() -> None:
    lease = SessionLeaseGranted(
        lease_id="lease-transport",
        session_id="session-transport",
        task_id="task-transport",
        device_id="device-transport",
        trace_id="trace-transport",
    )
    ipc = _FailingCommandIPC()
    reporter = _CapturingCommandStateReporter()
    transport = WorkerCommandTransport(
        ipc=ipc,  # type: ignore[arg-type]
        lease=lease,
        heartbeat_state_reporter=reporter,
    )
    command = CommandRequest(
        command_id="command-active",
        idempotency_key="idem-command-active",
        session_id=lease.session_id,
        lease_id=lease.lease_id,
        device_id=lease.device_id,
        tool_id="fake.tool",
        timeout_seconds=10,
    )

    with pytest.raises(TimeoutError, match="synthetic command dispatch timeout"):
        await transport.send_command(command, trace_id=lease.trace_id)

    assert reporter.events == [("set", "command-active"), ("clear", "command-active")]


@pytest.mark.asyncio
async def test_worker_command_transport_does_not_wait_on_slow_heartbeat_reporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_client, "_HEARTBEAT_STATE_REPORT_TIMEOUT_SECONDS", 0.01)
    lease = SessionLeaseGranted(
        lease_id="lease-transport",
        session_id="session-transport",
        task_id="task-transport",
        device_id="device-transport",
        trace_id="trace-transport",
    )
    ipc = _CapturingIPC()
    transport = WorkerCommandTransport(
        ipc=ipc,  # type: ignore[arg-type]
        lease=lease,
        heartbeat_state_reporter=_SlowCommandStateReporter(),
    )
    command = CommandRequest(
        command_id="command-active",
        idempotency_key="idem-command-active",
        session_id=lease.session_id,
        lease_id=lease.lease_id,
        device_id=lease.device_id,
        tool_id="fake.tool",
        timeout_seconds=10,
    )

    started_at = time.monotonic()
    await transport.send_command(command, trace_id=lease.trace_id)

    assert time.monotonic() - started_at < 0.2
    assert len(ipc.requests) == 1


@pytest.mark.asyncio
async def test_worker_command_transport_preserves_parameter_validation_error_type() -> None:
    lease = SessionLeaseGranted(
        lease_id="lease-transport",
        session_id="session-transport",
        task_id="task-transport",
        device_id="device-transport",
        trace_id="trace-transport",
    )
    transport = WorkerCommandTransport(
        ipc=_ParameterValidationCommandIPC(),  # type: ignore[arg-type]
        lease=lease,
    )
    command = CommandRequest(
        command_id="command-invalid-params",
        idempotency_key="idem-command-invalid-params",
        session_id=lease.session_id,
        lease_id=lease.lease_id,
        device_id=lease.device_id,
        tool_id="web.web_wait",
        parameters={"duration_ms": 1000, "load_state": None},
        timeout_seconds=10,
    )

    with pytest.raises(ToolParameterValidationError, match="must match exactly one schema"):
        await transport.send_command(command, trace_id=lease.trace_id)


@pytest.mark.asyncio
async def test_worker_progress_updates_known_step_and_phase() -> None:
    ipc = _CapturingHeartbeatIPC()
    request = WorkerRunRequest(
        job_id="job-progress",
        attempt_id="task-progress:1",
        app_config_path="config/app.yaml",
        artifact_root_dir_override="/tmp/artifacts",
        command_timeout_seconds=1.0,
        request=TaskRequest(
            task_id="task-progress",
            trace_id="trace-progress",
            agent_kind="desktop",
            request="fake",
        ),
        session_id="session-progress",
        lease=SessionLeaseGranted(
            lease_id="lease-progress",
            session_id="session-progress",
            task_id="task-progress",
            device_id="device-progress",
            trace_id="trace-progress",
        ),
    )
    reporter = _WorkerStartupPhaseReporter(ipc=ipc, request=request)

    await reporter.observe_progress(
        MessageType.TELEMETRY_UPDATE,
        TelemetryUpdate(
            events=[
                {
                    "event_type": "desktop_worker_runtime_phase",
                    "phase": "llm_started",
                    "step_index": 7,
                }
            ]
        ),
    )

    assert reporter.phase == "llm_started"
    assert reporter.step_index == 7
    assert ipc.heartbeats[-1][1].phase == "llm_started"
    assert ipc.heartbeats[-1][1].step_index == 7

    await reporter.observe_progress(
        MessageType.TELEMETRY_UPDATE,
        TelemetryUpdate(
            events=[
                {
                    "event_type": "desktop_worker_runtime_phase",
                    "phase": "llm_failed",
                    "step_index": 0,
                }
            ]
        ),
    )

    assert reporter.phase == "llm_failed"
    assert reporter.step_index == 7
    assert ipc.heartbeats[-1][1].phase == "llm_failed"
    assert ipc.heartbeats[-1][1].step_index == 7

    await reporter.observe_progress(
        MessageType.TELEMETRY_UPDATE,
        TelemetryUpdate(events=[{"event_type": "desktop_step", "step": {"step_index": 8}}]),
    )

    assert reporter.phase == "step_finished"
    assert reporter.step_index == 8
    assert ipc.heartbeats[-1][1].phase == "step_finished"
    assert ipc.heartbeats[-1][1].step_index == 8

    await reporter.set_runtime_state(phase="llm_started", step_index=9)
    await reporter.observe_progress(
        MessageType.TELEMETRY_UPDATE,
        TelemetryUpdate(events=[{"event_type": "desktop_step", "step": {"step_index": 8}}]),
    )

    assert reporter.phase == "llm_started"
    assert reporter.step_index == 9
    assert ipc.heartbeats[-1][1].phase == "llm_started"
    assert ipc.heartbeats[-1][1].step_index == 9


def test_worker_runtime_progress_can_be_heartbeat_only() -> None:
    payload = TelemetryUpdate(
        events=[
            {
                "event_type": "desktop_worker_runtime_phase",
                "phase": "llm_started",
                "step_index": 3,
                "worker_heartbeat_only": True,
            }
        ]
    )

    assert _progress_is_worker_heartbeat_only(MessageType.TELEMETRY_UPDATE, payload)

    external_payload = TelemetryUpdate(
        events=[
            {
                "event_type": "desktop_worker_runtime_phase",
                "phase": "llm_started",
                "step_index": 3,
            }
        ]
    )

    assert not _progress_is_worker_heartbeat_only(MessageType.TELEMETRY_UPDATE, external_payload)


@pytest.mark.asyncio
async def test_supervisor_persists_heartbeat_state_changes_inside_sample_window(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    store.create_worker_job(
        job_id="job-heartbeat-state",
        task_id="task-heartbeat-state",
        attempt_id="task-heartbeat-state:1",
        session_id="session-heartbeat-state",
        lease_id="lease-heartbeat-state",
        device_id="device-heartbeat-state",
        agent_kind="desktop",
    )
    supervisor = ProcessWorkerSupervisor.__new__(ProcessWorkerSupervisor)
    supervisor.store = store
    supervisor.job_id = "job-heartbeat-state"
    supervisor._last_heartbeat_at = time.monotonic()
    supervisor._last_heartbeat_persisted_at = 0.0
    supervisor._last_heartbeat_state_key = None
    supervisor._heartbeat_batcher = None

    async def _send(heartbeat: WorkerHeartbeat) -> dict:
        await supervisor._handle_heartbeat(
            WorkerEnvelope(
                message_type=WorkerMessageType.HEARTBEAT,
                payload_json=heartbeat.model_dump_json(exclude_none=True),
            )
        )
        return json.loads(store.worker_job_row("job-heartbeat-state")["heartbeat_payload_json"])

    first = await _send(WorkerHeartbeat(phase="llm_started", step_index=2, worker_pid=111))
    assert first["phase"] == "llm_started"
    assert first["step_index"] == 2
    assert first["worker_pid"] == 111

    unchanged = await _send(WorkerHeartbeat(phase="llm_started", step_index=2, worker_pid=222))
    assert unchanged["worker_pid"] == 111

    changed = await _send(
        WorkerHeartbeat(phase="command_dispatch", step_index=2, active_command_id="cmd-1", worker_pid=333)
    )
    assert changed["phase"] == "command_dispatch"
    assert changed["active_command_id"] == "cmd-1"
    assert changed["worker_pid"] == 333


class _FakeDesktopRemoteRunnerConfig:
    def __init__(
        self,
        *,
        app_config_path: str,
        artifact_root_dir_override: str | None = None,
        command_timeout_seconds: float = 0.0,
        max_steps_per_task: int = 100,
    ) -> None:
        self.app_config_path = app_config_path
        self.artifact_root_dir_override = artifact_root_dir_override
        self.command_timeout_seconds = command_timeout_seconds
        self.max_steps_per_task = max_steps_per_task


class _FakeDesktopRemoteTaskRunner:
    def __init__(
        self,
        *,
        config,
        command_transport_factory,
        startup_phase_sink=None,
        runtime_progress_sink_enabled=False,
    ) -> None:
        self.config = config
        self.command_transport_factory = command_transport_factory
        self.startup_phase_sink = startup_phase_sink
        self.runtime_progress_sink_enabled = runtime_progress_sink_enabled

    async def run(self, context: ControllerExecutionContext) -> TaskFinished:
        assert self.runtime_progress_sink_enabled
        if self.startup_phase_sink is not None:
            self.startup_phase_sink("loading_app_config")
        return TaskFinished(
            task_id=context.request.task_id,
            trace_id=context.request.trace_id,
            session_id=context.session_id,
            device_id=context.lease.device_id,
            lease_id=context.lease.lease_id,
            state=TaskState.FINISHED,
            result={"ok": True},
        )


@pytest.mark.asyncio
async def test_worker_execute_request_heartbeats_during_threaded_runner_import(monkeypatch) -> None:
    ipc = _CapturingHeartbeatIPC()
    lease = SessionLeaseGranted(
        lease_id="lease-startup",
        session_id="session-startup",
        task_id="task-startup",
        device_id="device-startup",
        trace_id="trace-startup",
    )
    request = WorkerRunRequest(
        job_id="job-startup",
        attempt_id="task-startup:1",
        app_config_path="config/app.yaml",
        artifact_root_dir_override="/tmp/artifacts",
        command_timeout_seconds=1.0,
        heartbeat_interval_seconds=0.01,
        request=TaskRequest(
            task_id="task-startup",
            trace_id="trace-startup",
            agent_kind="desktop",
            request="fake",
            target_device_id="device-startup",
        ),
        session_id="session-startup",
        lease=lease,
    )
    import_started = threading.Event()
    import_started_at = 0.0
    import_finished_at = 0.0

    def _blocking_import():
        nonlocal import_started_at, import_finished_at
        import_started_at = time.monotonic()
        import_started.set()
        time.sleep(0.08)
        import_finished_at = time.monotonic()
        return _FakeDesktopRemoteRunnerConfig, _FakeDesktopRemoteTaskRunner

    monkeypatch.setattr(
        "northcua.controller.execution.processes.runners.desktop._import_desktop_runner",
        _blocking_import,
    )

    reporter = _WorkerStartupPhaseReporter(ipc=ipc, request=request)
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(ipc, request, reporter),
    )
    try:
        finished = await _execute_request(ipc, request, phase_reporter=reporter)
        assert finished.state == TaskState.FINISHED
        assert import_started.is_set()
        await asyncio.sleep(0.03)
    finally:
        heartbeat_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_task
        await reporter.close()

    import_window_heartbeats = [
        heartbeat for timestamp, heartbeat in ipc.heartbeats if import_started_at <= timestamp <= import_finished_at
    ]
    assert len(import_window_heartbeats) >= 3
    phases = [heartbeat.phase for _timestamp, heartbeat in ipc.heartbeats]
    assert "importing_runner" in [heartbeat.phase for heartbeat in import_window_heartbeats]
    assert "loading_app_config" in phases
    assert any("worker_startup_phase" in event.payload_json for event in ipc.progress)


def test_process_worker_registry_exposes_desktop_and_web() -> None:
    registry = ProcessWorkerRunnerRegistry.default()

    assert registry.supported_agent_kinds() == ("desktop", "web")
    assert registry.resolve("desktop") is not None
    assert registry.resolve("web") is not None
    assert registry.resolve("unknown") is None


def test_process_mode_rejects_inline_runner_override_that_worker_cannot_load() -> None:
    async def plugin_runner(context: ControllerExecutionContext) -> TaskFinished:
        raise AssertionError(context)

    registry = RunnerRegistry()
    with registry.registration_scope("third_party.plugin:register"):
        registry.register("web", plugin_runner)

    with pytest.raises(RuntimeError, match="third_party.plugin:register"):
        _validate_process_runner_sources(registry, {"web"})


def test_process_mode_accepts_exact_builtin_runner_plugin_identity() -> None:
    async def builtin_runner(context: ControllerExecutionContext) -> TaskFinished:
        raise AssertionError(context)

    registry = RunnerRegistry()
    with registry.registration_scope("northcua_controller_plugins.desktop:register_controller_runner"):
        registry.register("desktop", builtin_runner)
        registry.register("web", builtin_runner)

    _validate_process_runner_sources(registry, {"desktop", "web"})


def test_process_mode_reuses_registered_immutable_app_config(tmp_path) -> None:
    source = tmp_path / "app.yaml"
    source.write_text("context_management:\n  mode: enforce\n", encoding="utf-8")
    frozen = tmp_path / "snapshot" / "app.yaml"
    frozen.parent.mkdir()
    frozen.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    async def builtin_runner(context: ControllerExecutionContext) -> TaskFinished:
        raise AssertionError(context)

    registry = RunnerRegistry()
    with registry.registration_scope("northcua_controller_plugins.desktop:register_controller_runner"):
        registry.register("desktop", builtin_runner, app_config_path=str(frozen))
        registry.register("web", builtin_runner, app_config_path=str(frozen))
    source.write_text("context_management:\n  mode: off\n", encoding="utf-8")

    process_path = _process_runner_app_config_path(
        registry,
        {"desktop", "web"},
        fallback=str(source),
    )

    assert process_path == str(frozen)
    assert Path(process_path).read_text(encoding="utf-8") == "context_management:\n  mode: enforce\n"


@pytest.mark.asyncio
async def test_worker_execute_request_dispatches_web_runner_strategy() -> None:
    ipc = _CapturingHeartbeatIPC()
    lease = SessionLeaseGranted(
        lease_id="lease-web-process",
        session_id="session-web-process",
        task_id="task-web-process",
        device_id="device-web-process",
        trace_id="trace-web-process",
    )
    request = WorkerRunRequest(
        job_id="job-web-process",
        attempt_id="task-web-process:1",
        app_config_path="config/app.yaml",
        command_timeout_seconds=1.0,
        request=TaskRequest(
            task_id="task-web-process",
            trace_id="trace-web-process",
            agent_kind="web",
            request="fake",
            target_device_id="device-web-process",
        ),
        session_id="session-web-process",
        lease=lease,
    )
    built: list[str] = []

    class _FakeWebRunner:
        async def run(self, context: ControllerExecutionContext) -> TaskFinished:
            assert context.request.agent_kind == "web"
            return TaskFinished(
                task_id=context.request.task_id,
                trace_id=context.request.trace_id,
                session_id=context.session_id,
                device_id=context.lease.device_id,
                lease_id=context.lease.lease_id,
                state=TaskState.FINISHED,
            )

    class _FakeWebFactory:
        agent_kind = "web"

        async def build(self, context):
            built.append(context.request.request.agent_kind)
            transport = context.command_transport_factory()(  # prove shared IPC transport wiring
                ControllerExecutionContext(
                    request=context.request.request,
                    session_id=context.request.session_id,
                    lease=context.request.lease,
                ),
                {"safe": "value"},
            )
            assert isinstance(transport, WorkerCommandTransport)
            assert transport.lease == lease
            assert transport.metadata_overrides == {"safe": "value"}
            return _FakeWebRunner()

    registry = ProcessWorkerRunnerRegistry((_FakeWebFactory(),))
    reporter = _WorkerStartupPhaseReporter(ipc=ipc, request=request)
    try:
        finished = await _execute_request(
            ipc,
            request,
            phase_reporter=reporter,
            runner_registry=registry,
        )
    finally:
        await reporter.close()

    assert finished.state == TaskState.FINISHED
    assert built == ["web"]


@pytest.mark.asyncio
async def test_process_worker_supervisor_runs_fake_worker(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    request = TaskRequest(
        task_id="task-1",
        trace_id="trace-1",
        agent_kind="desktop",
        request="fake",
        target_device_id="device-1",
        metadata={"batch_attempt": 2, "fake_worker_seconds": 0.01, "fake_command_count": 2},
    )
    lease = SessionLeaseGranted(
        lease_id="lease-1",
        session_id="session-1",
        task_id="task-1",
        device_id="device-1",
        trace_id="trace-1",
        metadata={"batch_attempt": 2},
    )
    context = ControllerExecutionContext(
        request=request,
        session_id="session-1",
        lease=lease,
        blob_store_root=tmp_path / "blobs",
    )
    store.create_worker_job(
        job_id="job-1",
        task_id="task-1",
        attempt_id="task-1:2",
        session_id="session-1",
        lease_id="lease-1",
        device_id="device-1",
        agent_kind="desktop",
    )

    sdk = _FakeSDK()
    supervisor = ProcessWorkerSupervisor(
        context=context,
        sdk=sdk,  # type: ignore[arg-type]
        store=store,
        job_id="job-1",
        attempt_id="task-1:2",
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        start_timeout_seconds=10.0,
        heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
    )

    finished = await supervisor.run()

    assert finished.state == TaskState.FINISHED
    assert finished.result == {
        "status": "ok",
        "worker_mode": "fake",
        "command_results": [{"ok": True, "tool_id": "fake.tool"}, {"ok": True, "tool_id": "fake.tool"}],
    }
    assert [command.tool_id for command in sdk.commands] == ["fake.tool", "fake.tool"]
    assert all("worker_job_id" not in command.metadata for command in sdk.commands)
    assert all(command.session_id == "session-1" and command.lease_id == "lease-1" for command in sdk.commands)
    metric_names = [
        json.loads(row["payload_json"])["metric_name"]
        for row in store.event_rows()
        if row["event_type"] == "process_executor_metric"
    ]
    assert metric_names == ["worker_connect", "worker_ready", "worker_command_roundtrip"]
    row = store.worker_job_row("job-1")
    assert row is not None
    assert row["status"] == WorkerJobStatus.FINISHING.value
    assert row["worker_pid"] > 0


@pytest.mark.asyncio
async def test_process_worker_rejects_spoofed_progress_messages(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    context = _context("task-spoof-progress", fake_worker_seconds=0.01)
    context = ControllerExecutionContext(
        request=context.request,
        session_id=context.session_id,
        lease=context.lease,
        progress_sink=lambda _message_type, _payload: None,
    )
    supervisor = _supervisor(
        tmp_path=tmp_path,
        store=store,
        context=context,
        job_id="job-spoof-progress",
        heartbeat_timeout_seconds=10.0,
    )

    with pytest.raises(WorkerIPCError, match="worker progress cannot forward controller message type"):
        await supervisor._handle_progress(  # noqa: SLF001 - lock the private IPC guard directly.
            WorkerEnvelope(
                message_type=WorkerMessageType.PROGRESS,
                payload_json=json.dumps({"message_type": "task_finished", "payload_json": "{}"}),
            )
        )


@pytest.mark.asyncio
async def test_process_worker_progress_sink_failure_is_best_effort(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    context = _context("task-progress-sink", fake_worker_seconds=0.01, metadata={"fake_command_count": 1})
    context = ControllerExecutionContext(
        request=context.request,
        session_id=context.session_id,
        lease=context.lease,
        progress_sink=_failing_progress_sink,
    )
    store.create_worker_job(
        job_id="job-progress-sink",
        task_id=context.request.task_id,
        attempt_id="task-progress-sink:1",
        session_id=context.session_id,
        lease_id=context.lease.lease_id,
        device_id=context.lease.device_id,
        agent_kind="desktop",
    )

    finished = await _supervisor(
        tmp_path=tmp_path,
        store=store,
        context=context,
        job_id="job-progress-sink",
        heartbeat_timeout_seconds=10.0,
    ).run()

    assert finished.state == TaskState.FINISHED


@pytest.mark.asyncio
async def test_process_worker_rejects_terminal_payload_mismatch(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    context = _context("task-bad-terminal", fake_worker_seconds=0.01, metadata={"fake_bad_terminal": True})
    store.create_worker_job(
        job_id="job-bad-terminal",
        task_id=context.request.task_id,
        attempt_id="task-bad-terminal:1",
        session_id=context.session_id,
        lease_id=context.lease.lease_id,
        device_id=context.lease.device_id,
        agent_kind="desktop",
    )

    finished = await _supervisor(
        tmp_path=tmp_path,
        store=store,
        context=context,
        job_id="job-bad-terminal",
        heartbeat_timeout_seconds=10.0,
    ).run()

    assert finished.state == TaskState.FAILED
    assert "worker terminal payload mismatch" in (finished.error or "")
    row = store.worker_job_row("job-bad-terminal")
    assert row is not None
    assert row["failure_kind"] == "ipc_protocol_error"


@pytest.mark.asyncio
async def test_process_worker_heartbeat_timeout_is_classified(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    context = _context(
        "task-no-heartbeat",
        fake_worker_seconds=3.0,
        metadata={"fake_disable_heartbeat": True},
    )
    store.create_worker_job(
        job_id="job-no-heartbeat",
        task_id=context.request.task_id,
        attempt_id="task-no-heartbeat:1",
        session_id=context.session_id,
        lease_id=context.lease.lease_id,
        device_id=context.lease.device_id,
        agent_kind="desktop",
    )

    finished = await _supervisor(
        tmp_path=tmp_path,
        store=store,
        context=context,
        job_id="job-no-heartbeat",
        heartbeat_timeout_seconds=1.0,
    ).run()

    assert finished.state == TaskState.FAILED
    assert "heartbeat stale" in (finished.error or "")
    row = store.worker_job_row("job-no-heartbeat")
    assert row is not None
    assert row["failure_kind"] == "heartbeat_timeout"


@pytest.mark.asyncio
async def test_process_worker_preserves_structured_worker_error(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    context = _context("task-worker-error", fake_worker_seconds=0.01, metadata={"fake_worker_error": True})
    store.create_worker_job(
        job_id="job-worker-error",
        task_id=context.request.task_id,
        attempt_id="task-worker-error:1",
        session_id=context.session_id,
        lease_id=context.lease.lease_id,
        device_id=context.lease.device_id,
        agent_kind="desktop",
    )

    finished = await _supervisor(
        tmp_path=tmp_path,
        store=store,
        context=context,
        job_id="job-worker-error",
        heartbeat_timeout_seconds=10.0,
    ).run()

    assert finished.state == TaskState.FAILED
    assert finished.error == "fake worker reported error"
    assert finished.terminal_decision["retryable"] is False
    row = store.worker_job_row("job-worker-error")
    assert row is not None
    assert row["status"] == WorkerJobStatus.FINISHING.value
    assert row["failure_kind"] == "task_failed"


@pytest.mark.asyncio
async def test_process_executor_rejects_when_worker_queue_full(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    task_service = _FakeTaskService()
    executor = ProcessTaskExecutorService(
        task_service=task_service,  # type: ignore[arg-type]
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        runner_workers=1,
        worker_queue_max=1,
        worker_start_timeout_seconds=10.0,
        worker_heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
    )

    await executor.schedule(_context("task-1", fake_worker_seconds=0.2))
    await executor.schedule(_context("task-2", fake_worker_seconds=0.01))
    await executor.drain()

    assert [payload.task_id for payload in task_service.payloads] == ["task-2", "task-1"]
    rejected, completed = task_service.payloads
    assert rejected.state == TaskState.FAILED
    assert rejected.error == "process executor worker queue is full"
    assert completed.state == TaskState.FINISHED
    assert len(store.non_terminal_worker_job_rows()) == 0


@pytest.mark.asyncio
async def test_process_executor_duplicate_attempt_is_db_claimed(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    task_service = _FakeTaskService()
    executor = ProcessTaskExecutorService(
        task_service=task_service,  # type: ignore[arg-type]
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        runner_workers=1,
        worker_queue_max=10,
        worker_start_timeout_seconds=10.0,
        worker_heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
    )

    context = _context("task-duplicate", fake_worker_seconds=0.01)
    await executor.schedule(context)
    await executor.schedule(context)
    await executor.drain()

    assert [payload.task_id for payload in task_service.payloads] == ["task-duplicate"]
    rows = store.connection.execute("SELECT * FROM worker_jobs WHERE task_id='task-duplicate'").fetchall()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_process_executor_db_claim_does_not_block_event_loop(tmp_path, monkeypatch) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    executor = ProcessTaskExecutorService(
        task_service=_FakeTaskService(),  # type: ignore[arg-type]
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        runner_workers=1,
        worker_queue_max=10,
        worker_start_timeout_seconds=10.0,
        worker_heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
    )
    original_claim = SQLiteControllerStore.try_create_worker_job
    loop_thread_id = threading.get_ident()
    claim_thread_ids: list[int] = []

    def slow_claim(store_instance, *args, **kwargs):
        claim_thread_ids.append(threading.get_ident())
        time.sleep(0.1)
        return original_claim(store_instance, *args, **kwargs)

    monkeypatch.setattr(SQLiteControllerStore, "try_create_worker_job", slow_claim)
    loop_ticks = 0

    async def tick_while_claim_waits() -> None:
        nonlocal loop_ticks
        deadline = asyncio.get_running_loop().time() + 0.08
        while asyncio.get_running_loop().time() < deadline:
            loop_ticks += 1
            await asyncio.sleep(0.005)

    ticker = asyncio.create_task(tick_while_claim_waits())
    await executor.schedule(_context("task-slow-db-claim", fake_worker_seconds=0.01))
    await ticker
    await executor.drain()

    assert loop_ticks >= 5
    assert claim_thread_ids
    assert all(thread_id != loop_thread_id for thread_id in claim_thread_ids)


@pytest.mark.asyncio
async def test_process_executor_keeps_worker_job_finishing_until_task_completion_succeeds(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    executor = ProcessTaskExecutorService(
        task_service=_FailingTaskService(),  # type: ignore[arg-type]
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        runner_workers=1,
        worker_queue_max=10,
        worker_start_timeout_seconds=10.0,
        worker_heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
    )

    await executor.schedule(_context("task-complete-fails", fake_worker_seconds=0.01))
    with pytest.raises(RuntimeError, match="synthetic complete_task failure"):
        await executor.drain()

    row = store.connection.execute("SELECT * FROM worker_jobs WHERE task_id='task-complete-fails'").fetchone()
    assert row is not None
    assert row["status"] == WorkerJobStatus.FINISHING.value


@pytest.mark.asyncio
async def test_process_executor_cancel_queued_task_before_worker_starts(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    task_service = _FakeTaskService()
    executor = ProcessTaskExecutorService(
        task_service=task_service,  # type: ignore[arg-type]
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        runner_workers=1,
        worker_queue_max=10,
        worker_start_timeout_seconds=10.0,
        worker_heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
    )

    await executor.schedule(_context("task-active", fake_worker_seconds=0.2))
    await executor.schedule(_context("task-queued", fake_worker_seconds=0.01))
    await executor.cancel_task("task-queued", "operator canceled queued task")
    await executor.drain()

    assert [payload.task_id for payload in task_service.payloads] == ["task-active"]
    row = store.connection.execute("SELECT * FROM worker_jobs WHERE task_id='task-queued'").fetchone()
    assert row is not None
    assert row["status"] == WorkerJobStatus.CANCELED.value
    assert row["failure_kind"] == "canceled"
    assert row["worker_pid"] == 0


@pytest.mark.asyncio
async def test_process_executor_cancel_active_task_keeps_worker_job_canceled(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    task_service = _FakeTaskService()
    executor = ProcessTaskExecutorService(
        task_service=task_service,  # type: ignore[arg-type]
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        runner_workers=1,
        worker_queue_max=10,
        worker_start_timeout_seconds=10.0,
        worker_heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
    )

    await executor.schedule(_context("task-active-cancel", fake_worker_seconds=5.0))
    for _ in range(50):
        row = store.connection.execute(
            "SELECT worker_pid FROM worker_jobs WHERE task_id='task-active-cancel'"
        ).fetchone()
        if row is not None and row["worker_pid"]:
            break
        await asyncio.sleep(0.02)
    await executor.cancel_task("task-active-cancel", "operator canceled active task")
    await executor.drain()

    assert [payload.task_id for payload in task_service.payloads] == ["task-active-cancel"]
    assert task_service.payloads[0].state == TaskState.INTERRUPTED
    row = store.connection.execute("SELECT * FROM worker_jobs WHERE task_id='task-active-cancel'").fetchone()
    assert row is not None
    assert row["status"] == WorkerJobStatus.CANCELED.value
    assert row["failure_kind"] == "canceled"


@pytest.mark.asyncio
async def test_process_executor_no_llm_simulation_records_metrics(tmp_path, caplog) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    task_service = _FakeTaskService()
    caplog.set_level(logging.INFO, logger="northcua.controller.execution.processes")
    executor = ProcessTaskExecutorService(
        task_service=task_service,  # type: ignore[arg-type]
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        runner_workers=4,
        worker_queue_max=16,
        worker_start_timeout_seconds=10.0,
        worker_heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
    )

    for index in range(8):
        await executor.schedule(
            _context(
                f"task-sim-{index}",
                fake_worker_seconds=0.01,
                metadata={"fake_command_count": 2},
            )
        )
    await executor.drain()

    assert len(task_service.payloads) == 8
    assert {payload.state for payload in task_service.payloads} == {TaskState.FINISHED}
    rows = store.connection.execute("SELECT status FROM worker_jobs").fetchall()
    assert [row["status"] for row in rows] == [WorkerJobStatus.FINISHED.value] * 8
    metrics = [
        record.process_executor_metric for record in caplog.records if hasattr(record, "process_executor_metric")
    ]
    assert sum(metric["metric_name"] == "worker_connect" for metric in metrics) == 8
    assert sum(metric["metric_name"] == "worker_ready" for metric in metrics) == 8
    assert sum(metric["metric_name"] == "worker_command_roundtrip" for metric in metrics) == 24
    assert all(metric["value_ms"] >= 0 for metric in metrics)
    persisted_metrics = [
        row
        for row in store.event_rows()
        if row["event_type"] == "process_executor_metric" and row["entity_type"] == "worker_job"
    ]
    assert len(persisted_metrics) == 24


def test_worker_jobs_schema_and_non_terminal_rows(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    store.create_worker_job(
        job_id="job-queued",
        task_id="task-queued",
        attempt_id="task-queued:1",
        session_id="session-queued",
        lease_id="lease-queued",
        device_id="device-queued",
        agent_kind="desktop",
    )
    store.create_worker_job(
        job_id="job-finished",
        task_id="task-finished",
        attempt_id="task-finished:1",
        session_id="session-finished",
        lease_id="lease-finished",
        device_id="device-finished",
        agent_kind="desktop",
    )
    store.mark_worker_job_status(job_id="job-finished", status=WorkerJobStatus.FINISHED)

    assert [row["job_id"] for row in store.non_terminal_worker_job_rows()] == ["job-queued"]


@pytest.mark.asyncio
async def test_in_process_executor_cancel_does_not_raise_on_drain() -> None:
    async def never_finishes(_context: ControllerExecutionContext) -> TaskFinished:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    executor = TaskExecutorService(
        task_service=_FakeTaskService(),  # type: ignore[arg-type]
        runner_registry=RunnerRegistry({"desktop": never_finishes}),
    )
    await executor.schedule(_context("task-in-process-cancel", fake_worker_seconds=0.01))
    await executor.cancel_task("task-in-process-cancel", "operator canceled")
    await executor.drain()


@pytest.mark.asyncio
async def test_process_executor_startup_marks_stale_worker_jobs_retryable(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    stale_request = TaskRequest(
        task_id="task-stale",
        trace_id="trace-stale",
        agent_kind="desktop",
        request="fake",
        target_device_id="device-stale",
    )
    store.create_task(stale_request, session_id="session-stale", submitter_id="submitter-1")
    store.create_lease(
        lease_id="lease-stale",
        session_id="session-stale",
        task_id="task-stale",
        device_id="device-stale",
    )
    store.create_worker_job(
        job_id="job-stale",
        task_id="task-stale",
        attempt_id="task-stale:1",
        session_id="session-stale",
        lease_id="lease-stale",
        device_id="device-stale",
        agent_kind="desktop",
    )

    executor = ProcessTaskExecutorService(
        task_service=_FakeTaskService(),  # type: ignore[arg-type]
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        runner_workers=1,
        worker_queue_max=1,
        worker_start_timeout_seconds=10.0,
        worker_heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
    )
    await executor._ensure_recovered()  # noqa: SLF001 - focused startup recovery contract

    row = store.worker_job_row("job-stale")
    assert row is not None
    assert row["status"] == WorkerJobStatus.FAILED_RETRYABLE.value
    assert row["failure_kind"] == "worker_crash"
    assert store.task_row("task-stale")["status"] == TaskState.FAILED.value
    assert store.session_row("session-stale")["status"] == "failed"
    assert store.lease_row("lease-stale")["status"] == "released"


@pytest.mark.asyncio
async def test_process_executor_startup_recovery_does_not_block_event_loop(tmp_path, monkeypatch) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_store_call(_path, operation, **_kwargs):
        started.set()
        await release.wait()
        return operation(store)

    monkeypatch.setattr(
        "northcua.controller.execution.processes.executor.controller_store_call",
        _slow_store_call,
    )
    executor = ProcessTaskExecutorService(
        task_service=_FakeTaskService(),  # type: ignore[arg-type]
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        runner_workers=1,
        worker_queue_max=1,
        worker_start_timeout_seconds=10.0,
        worker_heartbeat_timeout_seconds=10.0,
        worker_module="tests.support.fake_worker_main",
    )

    recovery = asyncio.create_task(executor._ensure_recovered())  # noqa: SLF001
    await started.wait()
    loop_progressed = False

    async def _tick() -> None:
        nonlocal loop_progressed
        await asyncio.sleep(0)
        loop_progressed = True

    await _tick()
    assert loop_progressed
    release.set()
    await recovery
    await executor.drain()


def test_process_executor_startup_preserves_already_terminal_task_rows(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    terminal_request = TaskRequest(
        task_id="task-terminal",
        trace_id="trace-terminal",
        agent_kind="desktop",
        request="fake",
        target_device_id="device-terminal",
    )
    store.create_task(terminal_request, session_id="session-terminal", submitter_id="submitter-1")
    store.create_lease(
        lease_id="lease-terminal",
        session_id="session-terminal",
        task_id="task-terminal",
        device_id="device-terminal",
    )
    store.connection.execute(
        "UPDATE tasks SET status=?, result_json=?, finished_at=? WHERE task_id=?",
        (TaskState.INTERRUPTED.value, '{"preserve": true}', "already-terminal", "task-terminal"),
    )
    store.connection.commit()
    command = CommandRequest(
        command_id="command-terminal-active",
        idempotency_key="idem-command-terminal-active",
        session_id="session-terminal",
        lease_id="lease-terminal",
        device_id="device-terminal",
        tool_id="fake.tool",
        lease_generation=3,
    )
    assert store.claim_command(command)[0] == "new"
    store.create_worker_job(
        job_id="job-terminal",
        task_id="task-terminal",
        attempt_id="task-terminal:1",
        session_id="session-terminal",
        lease_id="lease-terminal",
        device_id="device-terminal",
        agent_kind="desktop",
    )

    recovered = store.fail_non_terminal_worker_jobs_and_tasks(
        failure_kind=WorkerJobFailureKind.WORKER_CRASH,
        error="controller restarted",
    )

    assert recovered == 1
    assert store.task_row("task-terminal")["status"] == TaskState.INTERRUPTED.value
    assert store.task_row("task-terminal")["result_json"] == '{"preserve": true}'
    assert store.session_row("session-terminal")["status"] == "interrupted"
    assert store.lease_row("lease-terminal")["status"] == "released"
    command_row = store.command_row(
        command_storage_key(
            session_id="session-terminal",
            lease_id="lease-terminal",
            command_id="command-terminal-active",
        )
    )
    assert command_row is not None
    assert command_row["status"] == CommandState.CANCELLED.value
    assert command_row["result_json"]
    state, cached = store.claim_command(command)
    assert state == "cached"
    assert cached is not None
    assert cached.status == CommandState.CANCELLED
    assert cached.lease_generation == 3
    row = store.worker_job_row("job-terminal")
    assert row is not None
    assert row["status"] == WorkerJobStatus.CANCELED.value
    assert row["failure_kind"] == "canceled"


def test_claim_command_returns_terminal_synthetic_result_when_result_missing(tmp_path) -> None:
    store = SQLiteControllerStore(tmp_path / "controller.sqlite")
    command = CommandRequest(
        command_id="command-cancelled-no-result",
        idempotency_key="idem-command-cancelled-no-result",
        session_id="session-command",
        lease_id="lease-command",
        device_id="device-command",
        tool_id="fake.tool",
    )
    assert store.claim_command(command)[0] == "new"
    store.update_command_status_for_request(command, CommandState.CANCELLED)

    state, cached = store.claim_command(command)

    assert state == "cached"
    assert cached is not None
    assert cached.status == CommandState.CANCELLED
    assert "without cached result" in (cached.error or "")


@pytest.mark.asyncio
async def test_worker_ipc_rejects_oversized_and_invalid_frames() -> None:
    envelope = WorkerEnvelope(message_type=WorkerMessageType.READY, trace_id="trace", payload_json="{}")
    reader = asyncio.StreamReader()
    _feed_frame(reader, envelope.model_dump_json().encode("utf-8"))
    assert await read_envelope(reader) == envelope

    oversized = asyncio.StreamReader()
    _feed_frame(oversized, b"{}")
    with pytest.raises(WorkerIPCError, match="invalid worker IPC frame size"):
        await read_envelope(oversized, max_frame_bytes=1)

    invalid = asyncio.StreamReader()
    _feed_frame(invalid, b"{}")
    with pytest.raises(WorkerIPCError, match="invalid worker IPC envelope"):
        await read_envelope(invalid)


def _context(
    task_id: str,
    *,
    fake_worker_seconds: float,
    metadata: dict[str, object] | None = None,
) -> ControllerExecutionContext:
    merged_metadata: dict[str, object] = {"batch_attempt": 1, "fake_worker_seconds": fake_worker_seconds}
    merged_metadata.update(metadata or {})
    request = TaskRequest(
        task_id=task_id,
        trace_id=f"trace-{task_id}",
        agent_kind="desktop",
        request="fake",
        target_device_id="device-1",
        metadata=merged_metadata,
    )
    lease = SessionLeaseGranted(
        lease_id=f"lease-{task_id}",
        session_id=f"session-{task_id}",
        task_id=task_id,
        device_id="device-1",
        trace_id=f"trace-{task_id}",
        metadata={"batch_attempt": 1},
    )
    return ControllerExecutionContext(
        request=request,
        session_id=f"session-{task_id}",
        lease=lease,
    )


def _supervisor(
    *,
    tmp_path,
    store: SQLiteControllerStore,
    context: ControllerExecutionContext,
    job_id: str,
    heartbeat_timeout_seconds: float,
) -> ProcessWorkerSupervisor:
    return ProcessWorkerSupervisor(
        context=context,
        sdk=_FakeSDK(),  # type: ignore[arg-type]
        store=store,
        job_id=job_id,
        attempt_id=f"{context.request.task_id}:1",
        app_config_path="config/app.local.yaml",
        artifact_root_dir_override=str(tmp_path / "artifacts"),
        command_timeout_seconds=1.0,
        start_timeout_seconds=10.0,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        worker_module="tests.support.fake_worker_main",
    )


def _feed_frame(reader: asyncio.StreamReader, payload: bytes) -> None:
    reader.feed_data(struct.pack("!I", len(payload)))
    reader.feed_data(payload)
