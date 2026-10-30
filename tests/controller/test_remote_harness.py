"""Focused tests for the controller-owned desktop remote harness runner."""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from types import SimpleNamespace

from northcua.controller.execution.executor_protocol import ControllerExecutionContext
from northcua.core.budget import AgentStepBudget
from northcua.core.enums import RuntimeMode, TaskStatus
from northcua.core.models import StepLogRecord
from northcua.telemetry.progress import RuntimeProgressEventType
from northcua.telemetry.contracts import FailureCategory
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult
from northcua_controller_plugins.desktop_evaluation_progress import drain_evaluators_with_progress
from northcua_controller_plugins.desktop_runner import DesktopRemoteRunnerConfig, DesktopRemoteTaskRunner
from northcua_controller_plugins.desktop_runner_lifecycle import stop_app_best_effort
from northcua_protocol.capabilities import Capability
from northcua_protocol.payloads import CommandResult, SessionLeaseGranted, TaskRequest
from northcua_protocol.state import CommandState, TaskState


class _FakeWebSocketConfig:
    """Minimal websocket config supporting the model_copy API used by the runner."""

    def __init__(self, runtime_mode: RuntimeMode = RuntimeMode.LOCAL) -> None:
        self.runtime_mode = runtime_mode

    def model_copy(self, update: dict[str, object]) -> "_FakeWebSocketConfig":
        return _FakeWebSocketConfig(runtime_mode=update["runtime_mode"])  # type: ignore[arg-type]


class _FakeFacade:
    """Facade stub that only satisfies the runner's high-level control flow."""

    def __init__(self) -> None:
        self.enqueued_queries: list[str] = []
        self.enqueued_metadata: list[dict[str, object] | None] = []
        self.drained = False

    async def enqueue_initial_request(self, query: str, metadata: dict[str, object] | None = None) -> None:
        self.enqueued_queries.append(query)
        self.enqueued_metadata.append(metadata)

    async def next_round(self) -> SimpleNamespace:
        return SimpleNamespace(
            query=self.enqueued_queries[-1] if self.enqueued_queries else "",
            metadata=self.enqueued_metadata[-1] if self.enqueued_metadata else None,
        )

    async def drain_background_tasks(self) -> None:
        self.drained = True


class _FakeArtifactStore:
    """Small artifact-store stand-in for remote runner tests."""

    def __init__(self, artifact_root: Path) -> None:
        self.root = artifact_root
        self.step_log_path = artifact_root / "steps.jsonl"
        self.llm_trace_log_path = artifact_root / "llm_trace.jsonl"
        self.runtime_log_path = artifact_root / "runtime.jsonl"
        self.evaluation_log_path = artifact_root / "evaluation.jsonl"
        self.round_evaluation_log_path = artifact_root / "round_evaluations.jsonl"
        self.eval_runtime_log_path = artifact_root / "eval_runtime.jsonl"
        self.session_evaluation_rollup_path = artifact_root / "session_evaluation.json"

    def eval_round_result_path(self, round_num: int) -> Path:
        """Return the legacy evaluator result path for one round."""

        return self.root / f"eval_round_{round_num:04d}_result.json"


class _FakeApp:
    """Application stub returned by DesktopApplication.from_app_config in tests."""

    def __init__(self, artifact_root: Path) -> None:
        self.app_config = SimpleNamespace(websocket=_FakeWebSocketConfig())
        self.artifact_store = _FakeArtifactStore(artifact_root)
        self.facade = _FakeFacade()
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


def _context() -> ControllerExecutionContext:
    return ControllerExecutionContext(
        request=TaskRequest(
            task_id="task-remote",
            trace_id="trace-remote",
            agent_kind="desktop",
            request="Open Calculator",
            required_capabilities={Capability.DESKTOP_INPUT},
        ),
        session_id="sess-remote",
        lease=SessionLeaseGranted(
            lease_id="lease-remote",
            session_id="sess-remote",
            task_id="task-remote",
            device_id="dev-remote",
            trace_id="trace-remote",
            lease_generation=2,
        ),
    )


def test_desktop_remote_task_runner_forces_remote_mode_and_attaches_trace_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    async def _run() -> None:
        artifact_root = tmp_path / "desktop-remote"
        artifact_root.mkdir(parents=True)
        (artifact_root / "steps.jsonl").write_text('{"step":1}\n', encoding="utf-8")
        (artifact_root / "llm_trace.jsonl").write_text('{"provider":"test"}\n', encoding="utf-8")
        (artifact_root / "runtime.jsonl").write_text('{"event":"task_running"}\n', encoding="utf-8")

        fake_app = _FakeApp(artifact_root)
        startup_phases: list[str] = []
        runner = DesktopRemoteTaskRunner(
            config=DesktopRemoteRunnerConfig(app_config_path="config/app.yaml"),
            sdk=SimpleNamespace(),
            startup_phase_sink=startup_phases.append,
        )
        assert runner.config.max_steps_per_task == 100

        async def _fake_run_desktop_task_runtime(*, facade, round_input, dispatcher, budget, context):
            del facade, round_input, dispatcher, context
            assert isinstance(budget, AgentStepBudget)
            assert budget.round_limit(runner.config.max_steps_per_task) == runner.config.max_steps_per_task
            return StepLogRecord(
                session_id="sess-remote",
                session_step=1,
                round_step=1,
                round_num=1,
                status=TaskStatus.FINISH,
                request="Open Calculator",
                result="done",
            )

        captured_bootstrap: dict[str, object] = {}

        def _fake_from_app_config(*args, **kwargs):
            del args
            captured_bootstrap.update(kwargs)
            phase_callback = kwargs["bootstrap_phase_callback"]
            phase_callback("loading_app_config")
            time.sleep(0.05)
            phase_callback("building_artifact_store")
            fake_app.app_config.websocket = fake_app.app_config.websocket.model_copy(
                update={"runtime_mode": kwargs["runtime_mode_override"]}
            )
            return fake_app

        monkeypatch.setattr(
            "northcua_controller_plugins.desktop_runner.DesktopApplication.from_app_config",
            _fake_from_app_config,
        )
        monkeypatch.setattr(runner, "_run_desktop_task_runtime", _fake_run_desktop_task_runtime)

        tick_count = 0

        async def _ticker() -> None:
            nonlocal tick_count
            while True:
                await asyncio.sleep(0.005)
                tick_count += 1

        ticker_task = asyncio.create_task(_ticker())
        try:
            payload = await runner.run(_context())
        finally:
            ticker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker_task

        assert fake_app.app_config.websocket.runtime_mode == RuntimeMode.REMOTE
        assert captured_bootstrap["runtime_mode_override"] == RuntimeMode.REMOTE
        assert captured_bootstrap["bootstrap_phase_callback"] is not None
        assert tick_count >= 3
        assert startup_phases.index("loading_app_config") < startup_phases.index("building_artifact_store")
        assert startup_phases.index("building_artifact_store") < startup_phases.index("runner_started")
        assert fake_app.stopped is True
        assert fake_app.facade.drained is True
        assert payload.state == TaskState.FINISHED
        assert payload.result == "done"
        assert {artifact.artifact_id for artifact in payload.artifacts} == {
            "task-remote-steps",
            "task-remote-llm-trace",
            "task-remote-runtime",
        }
        assert all(artifact.storage_kind.value == "device_local" for artifact in payload.artifacts)
        assert all(artifact.inline_data is None for artifact in payload.artifacts)

    asyncio.run(_run())


def test_desktop_remote_task_runner_attaches_latest_evaluation_payload(tmp_path: Path, monkeypatch) -> None:
    async def _run() -> None:
        artifact_root = tmp_path / "desktop-remote-eval"
        artifact_root.mkdir(parents=True)
        fake_app = _FakeApp(artifact_root)
        fake_app.artifact_store.round_evaluation_log_path.write_text(
            '{"session_id":"sess-remote","round_num":1,"overall_verdict":"partial","reason":"Initial row."}\n',
            encoding="utf-8",
        )
        fake_app.artifact_store.eval_round_result_path(1).write_text(
            (
                '{"record":{"session_id":"sess-remote","round_num":1,'
                '"overall_verdict":"success","reason":"Verified from final screenshot."}}'
            ),
            encoding="utf-8",
        )
        runner = DesktopRemoteTaskRunner(
            config=DesktopRemoteRunnerConfig(app_config_path="config/app.yaml"),
            sdk=SimpleNamespace(),
        )

        async def _fake_run_desktop_task_runtime(*, facade, round_input, dispatcher, budget, context):
            del facade, round_input, dispatcher, budget, context
            return StepLogRecord(
                session_id="sess-remote",
                session_step=1,
                round_step=1,
                round_num=1,
                status=TaskStatus.FINISH,
                request="Open Calculator",
                result="done",
            )

        monkeypatch.setattr(
            "northcua_controller_plugins.desktop_runner.DesktopApplication.from_app_config",
            lambda *args, **kwargs: fake_app,
        )
        monkeypatch.setattr(runner, "_run_desktop_task_runtime", _fake_run_desktop_task_runtime)

        payload = await runner.run(_context())

        assert isinstance(payload.result, dict)
        assert payload.result["answer"] == "done"
        assert payload.result["evaluation_report"]["overall_verdict"] == "success"
        assert payload.result["evaluation_report"]["reason"] == "Verified from final screenshot."
        assert str(payload.result["evaluation_report_path"]).endswith("eval_round_0001_result.json")
        assert "task-remote-round-evaluations" in {artifact.artifact_id for artifact in payload.artifacts}

    asyncio.run(_run())


def test_desktop_remote_evaluator_progress_emits_report_instead_of_ready_status(tmp_path: Path) -> None:
    async def _run() -> None:
        artifact_root = tmp_path / "desktop-remote-eval-progress"
        artifact_root.mkdir(parents=True)
        store = _FakeArtifactStore(artifact_root)
        store.round_evaluation_log_path.write_text(
            '{"session_id":"sess-remote","round_num":1,"overall_verdict":"success","reason":"Report row."}\n',
            encoding="utf-8",
        )
        store.eval_round_result_path(1).write_text(
            (
                '{"record":{"session_id":"sess-remote","round_num":1,'
                '"overall_verdict":"success","reason":"Verified from final screenshot."}}'
            ),
            encoding="utf-8",
        )
        emitted: list[object] = []

        async def _progress_sink(message_type, payload) -> None:
            del message_type
            emitted.append(payload)

        async def _drain(_facade) -> None:
            return None

        base = _context()
        context = ControllerExecutionContext(
            request=base.request,
            session_id=base.session_id,
            lease=base.lease,
            progress_sink=_progress_sink,
        )
        await drain_evaluators_with_progress(
            facade=SimpleNamespace(artifact_store=store),
            context=context,
            drain=_drain,
        )

        event_types = [payload.events[0]["event_type"] for payload in emitted if hasattr(payload, "events")]
        assert event_types == [
            RuntimeProgressEventType.EVALUATION_STATUS.value,
            RuntimeProgressEventType.EVALUATION_REPORT.value,
        ]
        report_payload = emitted[-1]
        assert report_payload.events[0]["report"]["overall_verdict"] == "success"
        assert report_payload.events[0]["report"]["reason"] == "Verified from final screenshot."

    asyncio.run(_run())


def test_desktop_remote_app_stop_is_bounded() -> None:
    async def _run() -> None:
        stop_started = asyncio.Event()

        class _HungApp:
            async def stop(self) -> None:
                stop_started.set()
                await asyncio.Event().wait()

        await asyncio.wait_for(
            stop_app_best_effort(app=_HungApp(), context=_context(), timeout_seconds=0.01),
            timeout=1.0,
        )
        assert stop_started.is_set()

    asyncio.run(_run())


def test_desktop_remote_app_stop_returns_when_cleanup_suppresses_cancellation() -> None:
    async def _run() -> None:
        stop_started = asyncio.Event()

        class _StubbornApp:
            async def stop(self) -> None:
                stop_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await asyncio.Event().wait()

        await asyncio.wait_for(
            stop_app_best_effort(app=_StubbornApp(), context=_context(), timeout_seconds=0.01),
            timeout=1.0,
        )
        assert stop_started.is_set()

    asyncio.run(_run())


def test_desktop_remote_task_runner_uses_request_specific_max_steps(tmp_path: Path, monkeypatch) -> None:
    async def _run() -> None:
        artifact_root = tmp_path / "desktop-remote-budget"
        artifact_root.mkdir(parents=True)
        fake_app = _FakeApp(artifact_root)
        runner = DesktopRemoteTaskRunner(
            config=DesktopRemoteRunnerConfig(app_config_path="config/app.yaml", max_steps_per_task=32),
            sdk=SimpleNamespace(),
        )

        async def _fake_run_desktop_task_runtime(*, facade, round_input, dispatcher, budget, context):
            del facade, round_input, dispatcher, context
            assert isinstance(budget, AgentStepBudget)
            assert budget.max_round_steps == 7
            assert budget.max_session_steps == 21
            return StepLogRecord(
                session_id="sess-remote",
                session_step=1,
                round_step=1,
                round_num=1,
                status=TaskStatus.FINISH,
                request="Open Calculator",
                result="done",
            )

        monkeypatch.setattr(
            "northcua_controller_plugins.desktop_runner.DesktopApplication.from_app_config",
            lambda *args, **kwargs: fake_app,
        )
        monkeypatch.setattr(runner, "_run_desktop_task_runtime", _fake_run_desktop_task_runtime)

        context = ControllerExecutionContext(
            request=_context().request.model_copy(update={"metadata": {"max_steps": 7, "max_session_steps": 21}}),
            session_id="sess-remote",
            lease=_context().lease,
        )
        payload = await runner.run(context)

        assert payload.state == TaskState.FINISHED

    asyncio.run(_run())


def test_desktop_remote_task_runner_marks_evaluator_drain_failure(tmp_path: Path, monkeypatch) -> None:
    async def _run() -> None:
        artifact_root = tmp_path / "desktop-remote-evaluator-failure"
        artifact_root.mkdir(parents=True)
        fake_app = _FakeApp(artifact_root)
        runner = DesktopRemoteTaskRunner(
            config=DesktopRemoteRunnerConfig(app_config_path="config/app.yaml"),
            sdk=SimpleNamespace(),
        )

        async def _fake_run_desktop_task_runtime(*, facade, round_input, dispatcher, budget, context):
            del facade, round_input, dispatcher, budget, context
            return StepLogRecord(
                session_id="sess-remote",
                session_step=1,
                round_step=1,
                round_num=1,
                status=TaskStatus.FINISH,
                request="Open Calculator",
                result="main agent done",
            )

        async def _failing_drain() -> None:
            raise RuntimeError("evaluator did not submit final_report")

        monkeypatch.setattr(
            "northcua_controller_plugins.desktop_runner.DesktopApplication.from_app_config",
            lambda *args, **kwargs: fake_app,
        )
        monkeypatch.setattr(runner, "_run_desktop_task_runtime", _fake_run_desktop_task_runtime)
        fake_app.facade.drain_background_tasks = _failing_drain  # type: ignore[method-assign]

        payload = await runner.run(_context())

        assert payload.state == TaskState.FAILED
        assert payload.error == "evaluator did not submit final_report"
        assert payload.terminal_decision["summary"] == (
            "The desktop evaluator failed before producing required evaluation artifacts."
        )
        assert payload.terminal_decision["metadata"]["failure_category"] == FailureCategory.EVALUATOR.value

    asyncio.run(_run())
