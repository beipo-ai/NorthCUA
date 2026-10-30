"""Focused tests for the controller-owned desktop remote harness runner."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("nexau")

from northcua.agentkit.runtime import AgentRuntimeCursor, AgentRuntimeTerminalContext, AgentRuntimeTerminalPoint
from northcua.agents.desktop.runtime import DesktopTerminalProjectionPort
from northcua.controller.execution.executor_protocol import ControllerExecutionContext
from northcua.core.budget import AgentBudgetPolicy, AgentBudgetScope, AgentStepBudget
from northcua.core.enums import RuntimeMode, TaskStatus
from northcua.core.models import StepLogRecord
from northcua.telemetry.contracts import FailureCategory
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult
from northcua_controller_plugins.desktop_runner import DesktopRemoteRunnerConfig, DesktopRemoteTaskRunner
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


def test_desktop_remote_task_runner_clamps_request_max_steps_to_controller_cap(tmp_path: Path, monkeypatch) -> None:
    async def _run() -> None:
        artifact_root = tmp_path / "desktop-remote-budget-cap"
        artifact_root.mkdir(parents=True)
        fake_app = _FakeApp(artifact_root)
        runner = DesktopRemoteTaskRunner(
            config=DesktopRemoteRunnerConfig(app_config_path="config/app.yaml", max_steps_per_task=4),
            sdk=SimpleNamespace(),
        )

        async def _fake_run_desktop_task_runtime(*, facade, round_input, dispatcher, budget, context):
            del facade, round_input, dispatcher, context
            assert isinstance(budget, AgentStepBudget)
            assert budget.max_round_steps == 4
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
            request=_context().request.model_copy(update={"metadata": {"max_steps": 7}}),
            session_id="sess-remote",
            lease=_context().lease,
        )
        payload = await runner.run(context)

        assert payload.state == TaskState.FINISHED

    asyncio.run(_run())


def test_desktop_remote_tool_dispatch_validation_error_becomes_tool_result() -> None:
    """Invalid model tool arguments should not crash the controller runner."""

    async def _run() -> None:
        runner = DesktopRemoteTaskRunner(
            config=DesktopRemoteRunnerConfig(app_config_path="config/app.yaml"),
            sdk=SimpleNamespace(),
        )

        class _RejectingDispatcher:
            async def dispatch(self, **kwargs):
                del kwargs
                raise ValueError("command parameter 'parameters' must match exactly one schema")

        dispatch_result = await runner._dispatch_external_tool_calls(
            dispatcher=_RejectingDispatcher(),
            pending_tool_calls=[
                {
                    "id": "call-keypress",
                    "name": "keypress",
                    "input": {"key": "enter", "keys": ["ctrl+l"]},
                }
            ],
        )

        assert isinstance(dispatch_result, RemoteToolDispatchResult)
        assert len(dispatch_result.command_results) == 1
        command_result = dispatch_result.command_results[0]
        assert command_result.call_id == "call-keypress"
        assert command_result.status.value == "failure"
        assert "exactly one schema" in (command_result.error or "")
        assert dispatch_result.tool_blocks[0].is_error is True

    asyncio.run(_run())


def test_desktop_remote_terminal_projection_marks_max_step_decision() -> None:
    class _BudgetFacade:
        def __init__(self) -> None:
            self.persisted: list[StepLogRecord] = []
            self.advanced: list[bool] = []
            self.dispatchers: list[object] = []

        async def persist_step_limit_terminal(
            self,
            *,
            request: str,
            step_log: StepLogRecord,
            decision,
            advance_counters: bool = True,
            remote_tool_dispatcher=None,
        ) -> None:
            del request
            step_log.status = TaskStatus.FAIL
            step_log.result = decision.final_answer
            step_log.terminal_decision = decision.to_json()
            self.advanced.append(advance_counters)
            self.dispatchers.append(remote_tool_dispatcher)
            self.persisted.append(step_log)

    async def _run() -> None:
        facade = _BudgetFacade()
        dispatcher = object()
        decision = AgentBudgetPolicy.max_step_decision(
            scope=AgentBudgetScope.ROUND,
            limit=1,
            current=1,
            final_answer="Task exceeded the controller remote step budget before reaching a terminal state.",
        )
        projection = await DesktopTerminalProjectionPort(facade, remote_tool_dispatcher=dispatcher).project_terminal(
            AgentRuntimeTerminalContext(
                query="Open Calculator",
                decision=decision,
                point=AgentRuntimeTerminalPoint.PRE_STEP_LIMIT,
                cursor=AgentRuntimeCursor(session_id="sess-remote", round_num=1, round_step=1, session_step=1),
            )
        )

        assert projection is not None
        step_log = projection.result
        assert projection.should_advance is True
        assert step_log.status == TaskStatus.FAIL
        assert step_log.terminal_decision["reason"] == "max_steps"
        assert facade.persisted == [step_log]
        assert facade.dispatchers == [dispatcher]
        assert facade.advanced == [False]

    asyncio.run(_run())
