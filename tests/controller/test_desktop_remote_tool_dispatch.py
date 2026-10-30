"""Remote desktop tool-dispatch conversion tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("nexau")

from northcua.transport.ws.result_blocks import RemoteToolDispatchResult
from northcua_controller_plugins.desktop_runner import DesktopRemoteRunnerConfig, DesktopRemoteTaskRunner
from northcua_protocol.payloads import CommandResult
from northcua_protocol.state import CommandState
from northcua_runtime_shared.execution import RuntimeExecutionError


def test_desktop_remote_tool_dispatcher_attaches_raw_command_results() -> None:
    async def _run() -> None:
        class _FakeDispatcher:
            async def dispatch(self, **kwargs):
                return CommandResult(
                    command_id=str(kwargs["command_id"]),
                    session_id="sess-remote",
                    lease_id="lease-remote",
                    device_id="dev-remote",
                    lease_generation=2,
                    status=CommandState.SUCCEEDED,
                    result={"text": "clicked", "screenshot_path": "/tmp/click.png"},
                )

        runner = DesktopRemoteTaskRunner(
            config=DesktopRemoteRunnerConfig(app_config_path="config/app.yaml"),
            sdk=SimpleNamespace(),
        )

        batch = await runner._dispatch_external_tool_calls(
            dispatcher=_FakeDispatcher(),  # type: ignore[arg-type]
            pending_tool_calls=[{"id": "call-1", "name": "click", "input": {"x": 1, "y": 2}}],
        )

        assert isinstance(batch, RemoteToolDispatchResult)
        assert len(batch.command_results) == 1
        assert batch.command_results[0].call_id == "call-1"
        assert batch.command_results[0].result == {"text": "clicked", "screenshot_path": "/tmp/click.png"}

    asyncio.run(_run())


@pytest.mark.parametrize(
    "status,error",
    [
        (CommandState.TIMED_OUT, "COMMAND_TIMED_OUT"),
        (CommandState.CANCELLED, "COMMAND_DISPATCH_CANCELLED"),
        (CommandState.FAILED, "COMMAND_DELIVERY_UNKNOWN"),
        (CommandState.FAILED, "COMMAND_TRANSPORT_FAILED"),
        (CommandState.FAILED, "COMMAND_EXECUTION_STILL_RUNNING"),
    ],
)
def test_infrastructure_failure_reaches_task_boundary_without_next_action(status, error):
    async def run():
        calls = []

        class Dispatcher:
            async def dispatch(self, **kwargs):
                calls.append(kwargs["command_id"])
                return CommandResult(
                    command_id=kwargs["command_id"],
                    session_id="session",
                    lease_id="lease",
                    device_id="device",
                    status=status,
                    error=error,
                )

        runner = DesktopRemoteTaskRunner(
            config=DesktopRemoteRunnerConfig(app_config_path="unused"), sdk=SimpleNamespace()
        )
        with pytest.raises(RuntimeExecutionError, match=error):
            await runner._dispatch_external_tool_calls(
                dispatcher=Dispatcher(),
                pending_tool_calls=[
                    {"id": "one", "name": "click", "input": {}},
                    {"id": "two", "name": "click", "input": {}},
                ],
            )
        assert calls == ["one"]

    asyncio.run(run())


def test_ordinary_bad_parameters_remain_agent_visible_and_allow_next_tool():
    async def run():
        class Dispatcher:
            async def dispatch(self, **kwargs):
                return CommandResult(
                    command_id=kwargs["command_id"],
                    session_id="session",
                    lease_id="lease",
                    device_id="device",
                    status=CommandState.FAILED,
                    error="invalid key",
                )

        runner = DesktopRemoteTaskRunner(
            config=DesktopRemoteRunnerConfig(app_config_path="unused"), sdk=SimpleNamespace()
        )
        result = await runner._dispatch_external_tool_calls(
            dispatcher=Dispatcher(),
            pending_tool_calls=[
                {"id": "one", "name": "keypress", "input": {}},
                {"id": "two", "name": "keypress", "input": {}},
            ],
        )
        assert [r.error for r in result.command_results] == ["invalid key", "invalid key"]

    asyncio.run(run())


def test_device_failure_does_not_trigger_llm_fallback(monkeypatch):
    from northcua.agents.desktop.runtime.model import DesktopModelStepRunner
    from northcua.core.enums import RuntimeMode

    async def run():
        calls = []

        async def candidate(**kwargs):
            calls.append(kwargs["candidate"])
            raise RuntimeExecutionError("COMMAND_DELIVERY_UNKNOWN")

        fake = SimpleNamespace(
            app_config=SimpleNamespace(websocket=SimpleNamespace(runtime_mode=RuntimeMode.REMOTE)),
            agent_preparer=SimpleNamespace(llm_fallback_candidates=lambda: ["primary", "fallback"]),
            _run_candidate=candidate,
        )
        with pytest.raises(RuntimeExecutionError, match="COMMAND_DELIVERY_UNKNOWN"):
            await DesktopModelStepRunner.run(
                fake,
                task="task",
                session_id="session",
                round_num=1,
                round_step=1,
                session_step=1,
                step_budget=None,
                observation=None,
                history_turns=[],
                prompt_projector=None,
                trace_context_metadata={},
                remote_tool_dispatcher=None,
            )
        assert calls == ["primary"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "command_status,expected_state,retryable",
    [
        (CommandState.CANCELLED, "interrupted", False),
        (CommandState.TIMED_OUT, "failed", True),
        (CommandState.FAILED, "failed", True),
    ],
)
def test_runner_preserves_cancel_and_infra_terminal_semantics(
    monkeypatch, tmp_path, command_status, expected_state, retryable
):
    from unittest.mock import AsyncMock

    from northcua_controller_plugins import desktop_runner as module

    async def run():
        app = SimpleNamespace(artifact_store=SimpleNamespace(runtime_log_path=tmp_path / "runtime.jsonl"))
        monkeypatch.setattr(module.DesktopApplication, "from_app_config", lambda *args, **kwargs: app)
        monkeypatch.setattr(module, "disable_agentic_evaluator_for_external_script", lambda *args: None)
        monkeypatch.setattr(module, "apply_task_policy_route", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "stop_app_best_effort", AsyncMock())
        monkeypatch.setattr(module, "log_lifecycle", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "finalize_payload_with_artifacts", lambda **kwargs: kwargs["payload"])
        runner = DesktopRemoteTaskRunner(
            config=DesktopRemoteRunnerConfig(app_config_path="unused"), sdk=SimpleNamespace()
        )
        monkeypatch.setattr(runner, "_startup_phase_callback", lambda context: lambda phase: None)
        monkeypatch.setattr(
            runner,
            "_execute_with_app",
            AsyncMock(side_effect=RuntimeExecutionError("COMMAND_TIMED_OUT", status=command_status)),
        )
        context = SimpleNamespace(
            request=SimpleNamespace(task_id="task", trace_id="trace", metadata={}),
            session_id="session",
            lease=SimpleNamespace(device_id="device", lease_id="lease"),
        )
        result = await runner.run(context)
        assert result.state.value == expected_state
        assert result.terminal_decision["retryable"] is retryable
        assert result.terminal_decision["reason"] == ("interrupted" if not retryable else "runtime_error")

    asyncio.run(run())
