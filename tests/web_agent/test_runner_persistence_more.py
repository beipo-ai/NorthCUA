from __future__ import annotations

import asyncio
import base64

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.model_step import AgentKitModelStepExecutor
from northcua.agents.web.backend import WebActionCommandResult
from northcua.agents.web.domain.observations import (
    WebObservationContext,
    WebScreenshotObservation,
    WebSnapshotObservation,
)
from northcua.agents.web.domain.results import WebRunStatus, WebTerminalStatus
from northcua.agents.web.runner import (
    WebAgentModelStepResult,
    WebAgentRunner,
    WebAgentRunnerComponents,
    WebAgentRunnerRunSpec,
)
from northcua.core.config import ArtifactConfig, NorthCUAAppConfig
from northcua.core.models import AssistantToolCallRecord, ToolExecutionRecord
from tests.web_agent.test_runner_persistence import _build_runner


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def open(self, url: str) -> WebActionCommandResult:
        return WebActionCommandResult(success=True, data={"url": url})

    async def wait(self, params, *, timeout_seconds: float | None = None) -> WebActionCommandResult:
        del timeout_seconds
        return WebActionCommandResult(success=True, data={"waited": True})

    async def snapshot(self, **kwargs) -> WebSnapshotObservation:
        return WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            snapshot="button @e1 Finish",
            raw={"refs": {"e1": {"role": "button"}}},
        )

    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        data = base64.b64encode(b"fake-png").decode("ascii")
        return WebScreenshotObservation(data_url=f"data:image/png;base64,{data}", raw={"source": "fake"})

    async def close(self) -> WebActionCommandResult:
        self.closed = True
        return WebActionCommandResult(success=True, data={})


class _DoneRunner(WebAgentRunner):
    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        state = TraceCaptureState(assistant_text="finished")
        return WebAgentModelStepResult(trace_state=state, response_text="finished", meta={}, envelope=None)


class _ToolErrorRunner(_DoneRunner):
    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        state = TraceCaptureState(assistant_text="clicked")
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id="call-1",
                tool_name="web_click_ref",
                tool_input={"ref": "@stale"},
                tool_output={"status": "error", "error": "stale ref"},
                is_error=True,
            )
        )
        return WebAgentModelStepResult(trace_state=state, response_text="clicked", meta={}, envelope=None)


class _NoToolRunner(_DoneRunner):
    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        return WebAgentModelStepResult(
            trace_state=TraceCaptureState(assistant_text=""), response_text="", meta={}, envelope=None
        )


class _MultiToolRunner(_DoneRunner):
    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        state = TraceCaptureState(
            assistant_text="inspect tabs and URL", reasoning="Need URL and tabs before final answer."
        )
        state.assistant_tool_calls.extend(
            [
                AssistantToolCallRecord(id="call-1", name="web_get_url", arguments="{}"),
                AssistantToolCallRecord(id="call-2", name="web_list_tabs", arguments="{}"),
            ]
        )
        state.tool_executions.extend(
            [
                ToolExecutionRecord(
                    tool_call_id="call-1",
                    tool_name="web_get_url",
                    tool_output={"status": "ok", "text": "Read URL.", "result": {"url": "https://example.com"}},
                ),
                ToolExecutionRecord(
                    tool_call_id="call-2",
                    tool_name="web_list_tabs",
                    tool_output={"status": "ok", "text": "Listed tabs.", "result": {"tabs": []}},
                ),
            ]
        )
        return WebAgentModelStepResult(trace_state=state, response_text="inspect tabs and URL", meta={}, envelope=None)


def test_runner_applies_session_step_budget_to_existing_trace(tmp_path) -> None:
    class _RecordingRunner(_DoneRunner):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.seen_session_steps: list[int] = []
            self.seen_step_budgets: list[dict[str, object] | None] = []

        async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
            self.seen_session_steps.append(kwargs["session_step"])
            self.seen_step_budgets.append(kwargs["step_budget"])
            state = TraceCaptureState(assistant_text="scroll")
            state.tool_executions.append(
                ToolExecutionRecord(
                    tool_call_id="call-1",
                    tool_name="web_scroll",
                    tool_input={"direction": "down", "amount": 600},
                    tool_output={"status": "ok", "text": "Scrolled."},
                )
            )
            return WebAgentModelStepResult(trace_state=state, response_text="scroll", meta={}, envelope=None)

    runner, store, _client = _build_runner(tmp_path, _RecordingRunner)
    store.trace_path.write_text("{}\n{}\n", encoding="utf-8")

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="scroll", max_steps=5, max_session_steps=3)))

    assert runner.seen_session_steps == [3]
    assert runner.seen_step_budgets == [
        {
            "phase": "main",
            "current_round_step": 1,
            "round_limit": 5,
            "remaining_round_steps": 1,
            "current_session_step": 3,
            "session_limit": 3,
        }
    ]
    assert result.status == WebRunStatus.MAX_STEPS
    assert result.terminal_metadata == {"scope": "session", "limit": 3, "current": 3}


def test_runner_stops_before_model_when_session_budget_is_exhausted(tmp_path) -> None:
    class _RecordingRunner(_DoneRunner):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.model_calls = 0

        async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
            self.model_calls += 1
            return await super()._run_model_step(**kwargs)

    runner, store, client = _build_runner(tmp_path, _RecordingRunner)
    store.trace_path.write_text("{}\n{}\n", encoding="utf-8")

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="finish", max_steps=5, max_session_steps=2)))

    assert runner.model_calls == 0
    assert result.status == WebRunStatus.MAX_STEPS
    assert result.steps == []
    assert result.terminal_metadata == {"scope": "session", "limit": 2, "current": 2}
    assert client.closed is False


def test_runner_preserves_result_when_browser_cleanup_fails(tmp_path) -> None:
    class _CloseFailClient(_FakeClient):
        async def close(self) -> WebActionCommandResult:
            raise RuntimeError("close failed")

    app_config = NorthCUAAppConfig(
        agent_config_path="agent.example.yaml",
        artifacts=ArtifactConfig(root_dir=str(tmp_path)),
    )
    fake_client = _CloseFailClient()
    from northcua.agents.web.artifacts import WebArtifactStore

    store = WebArtifactStore.for_session(tmp_path, "web-runner-test", app_config.artifacts)
    runner = _DoneRunner(
        WebAgentRunnerComponents(
            app_config=app_config,
            agent_config=object(),
            artifact_store=store,
            execution_transport=object(),
            session_manager=object(),
            user_id="user",
            transport_capabilities=object(),
            client_factory=lambda _session_id: fake_client,
        )
    )

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="finish")))

    assert result.status == WebRunStatus.ERROR
    assert result.terminal_status == WebTerminalStatus.RUNTIME_ERROR
    assert result.final_answer == "finished\n\nCleanup failed: close failed"
    assert result.cleanup_errors == ["cleanup_failed: close failed"]
    assert result.tool_errors == ["cleanup_failed: close failed"]
