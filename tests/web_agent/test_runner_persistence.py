from __future__ import annotations

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.model_step import AgentKitModelStepExecutor
from northcua.agents.web.backend import WebActionCommandResult
from northcua.agents.web.domain.observations import (
    WebElementRef,
    WebObservationContext,
    WebScreenshotObservation,
    WebSnapshotObservation,
)
from northcua.agents.web.domain.results import WebRunStatus, WebStepRecord, WebStepStatus, WebTerminalStatus
from northcua.agents.web.runner import (
    WebAgentModelStepResult,
    WebAgentRunner,
    WebAgentRunnerComponents,
    WebAgentRunnerRunSpec,
)
from northcua.agents.web.runner.model_step import WebAgentModelStepComponents, WebAgentModelStepExecutor
from northcua.core.config import ArtifactConfig, NorthCUAAppConfig
from northcua.core.models import AssistantToolCallRecord, ToolExecutionRecord
from northcua.core.terminal import AgentTerminalReason


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.snapshot_calls = 0

    async def open(self, url: str) -> WebActionCommandResult:
        return WebActionCommandResult(success=True, data={"url": url})

    async def wait(self, params, *, timeout_seconds: float | None = None) -> WebActionCommandResult:
        del timeout_seconds
        return WebActionCommandResult(success=True, data={"waited": True})

    async def snapshot(self, **kwargs) -> WebSnapshotObservation:
        self.snapshot_calls += 1
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


def test_web_step_record_rejects_empty_session_identity() -> None:
    with pytest.raises(ValidationError, match="session_id"):
        WebStepRecord(step_index=1, session_id="", status=WebStepStatus.OK)


class _SomFakeClient(_FakeClient):
    async def snapshot(self, **kwargs) -> WebSnapshotObservation:
        del kwargs
        from northcua.agents.web.domain.observations import WebElementRef

        return WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            snapshot='button "Finish" [ref=@e1]',
            elements=[WebElementRef(ref_id="e1", role="button", name="Finish")],
            raw={"refs": {"e1": {"role": "button"}}},
        )

    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        del kwargs
        buffer = io.BytesIO()
        Image.new("RGB", (640, 360), "white").save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return WebScreenshotObservation(data_url=f"data:image/png;base64,{data}", raw={"source": "fake"})

    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        return WebActionCommandResult(
            success=True,
            data={"boxes": {ref: {"x": 48, "y": 40, "width": 120, "height": 34} for ref in refs}},
        )

    async def get_ref_states(self, refs: list[str]) -> WebActionCommandResult:
        return WebActionCommandResult(
            success=True,
            data={"states": {ref: {"visible": True, "enabled": True} for ref in refs}},
        )

    async def get_viewport_size(self) -> WebActionCommandResult:
        return WebActionCommandResult(
            success=True,
            data={"result": {"width": 640, "height": 360, "devicePixelRatio": 1}},
        )


class _DoneRunner(WebAgentRunner):
    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        state = TraceCaptureState(assistant_text="finished")
        return WebAgentModelStepResult(trace_state=state, response_text="finished", meta={}, envelope=None)


class _CaptureAllowedRunner(_DoneRunner):
    captured_allowed_tool_names: tuple[str, ...] = ()

    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        self.captured_allowed_tool_names = tuple(kwargs["allowed_tool_names"])
        return await super()._run_model_step(**kwargs)


class _ToolErrorRunner(_DoneRunner):
    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        state = TraceCaptureState(assistant_text="clicked")
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id="call-1",
                tool_name="web_click_ref",
                tool_input={"ref": "@stale"},
                tool_output={"status": "error", "error": "browser backend crashed"},
                is_error=True,
            )
        )
        return WebAgentModelStepResult(trace_state=state, response_text="clicked", meta={}, envelope=None)


class _RetryableToolErrorThenDoneRunner(_DoneRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_steps = 0
        self.recent_action_results: list[dict[str, object] | None] = []
        self.observed_snapshots: list[str] = []
        self.observation_ids: list[str] = []

    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        self.model_steps += 1
        observation = kwargs["observation"]
        self.recent_action_results.append(observation.recent_action_result)
        self.observed_snapshots.append(observation.snapshot.snapshot)
        self.observation_ids.append(observation.observation_id)
        state = TraceCaptureState(assistant_text="recovering" if self.model_steps == 1 else "recovered")
        if self.model_steps == 1:
            state.tool_executions.append(
                ToolExecutionRecord(
                    tool_call_id="call-1",
                    tool_name="web_click_ref",
                    tool_input={"ref": "@stale"},
                    tool_output={
                        "status": "error",
                        "code": "stale_ref",
                        "error": "stale ref",
                        "retryable": True,
                        "hint": "Capture a fresh snapshot.",
                    },
                    is_error=True,
                )
            )
            return WebAgentModelStepResult(trace_state=state, response_text="recovering", meta={}, envelope=None)
        return WebAgentModelStepResult(trace_state=state, response_text="recovered", meta={}, envelope=None)


class _BareValidationToolErrorThenDoneRunner(_RetryableToolErrorThenDoneRunner):
    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        self.model_steps += 1
        observation = kwargs["observation"]
        self.recent_action_results.append(observation.recent_action_result)
        self.observed_snapshots.append(observation.snapshot.snapshot)
        self.observation_ids.append(observation.observation_id)
        state = TraceCaptureState(assistant_text="recovering" if self.model_steps == 1 else "recovered")
        if self.model_steps == 1:
            state.tool_executions.append(
                ToolExecutionRecord(
                    tool_call_id="call-1",
                    tool_name="web_wait",
                    tool_input={"duration_ms": 1000, "load_state": "load", "text": "Ready"},
                    tool_output={
                        "status": "error",
                        "error": "1 validation error for WebWaitParams\nValue error, web_wait expects exactly one wait target.",
                        "error_type": "ToolExecutionError",
                    },
                    is_error=True,
                )
            )
            return WebAgentModelStepResult(trace_state=state, response_text="recovering", meta={}, envelope=None)
        return WebAgentModelStepResult(trace_state=state, response_text="recovered", meta={}, envelope=None)


class _ClickWithoutObservationThenDoneRunner(_DoneRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_steps = 0
        self.observed_snapshots: list[str] = []

    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        self.model_steps += 1
        observation = kwargs["observation"]
        self.observed_snapshots.append(observation.snapshot.snapshot)
        state = TraceCaptureState(assistant_text="clicked" if self.model_steps == 1 else "done")
        if self.model_steps == 1:
            state.tool_executions.append(
                ToolExecutionRecord(
                    tool_call_id="call-click",
                    tool_name="web_click_ref",
                    tool_input={"ref": "@e1"},
                    tool_output={"status": "ok", "action": "click_ref", "text": "Clicked.", "result": {}},
                )
            )
            return WebAgentModelStepResult(trace_state=state, response_text="clicked", meta={}, envelope=None)
        return WebAgentModelStepResult(trace_state=state, response_text="done", meta={}, envelope=None)


class _EarlierObservationThenMutationRunner(_DoneRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_steps = 0
        self.observed_snapshots: list[str] = []
        self.observed_ref_counts: list[int] = []
        self.observed_screenshots: list[WebScreenshotObservation | None] = []

    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        self.model_steps += 1
        observation = kwargs["observation"]
        self.observed_snapshots.append(observation.snapshot.snapshot)
        self.observed_ref_counts.append(len(observation.snapshot.elements))
        self.observed_screenshots.append(observation.screenshot)
        state = TraceCaptureState(assistant_text="acting" if self.model_steps == 1 else "done")
        if self.model_steps == 1:
            state.tool_executions.extend(
                [
                    ToolExecutionRecord(
                        tool_call_id="call-snapshot",
                        tool_name="web_snapshot",
                        tool_output={
                            "status": "ok",
                            "action": "snapshot",
                            "observation": {
                                "observation_id": "earlier-valid-observation",
                                "settled": True,
                                "settle_reason": "stable",
                                "page": {"url": "https://example.com", "title": "Example"},
                                "snapshot": {
                                    "text": '7 button "Stale target"',
                                    "refs": [{"ref": "7", "role": "button", "label": "Stale target"}],
                                    "ref_count": 1,
                                },
                                "screenshot": {"kind": "none", "image_attached": False},
                            },
                        },
                    ),
                    ToolExecutionRecord(
                        tool_call_id="call-click",
                        tool_name="web_click_ref",
                        tool_output={"status": "ok", "action": "click_ref", "text": "Clicked.", "result": {}},
                    ),
                ]
            )
            return WebAgentModelStepResult(trace_state=state, response_text="acting", meta={}, envelope=None)
        return WebAgentModelStepResult(trace_state=state, response_text="done", meta={}, envelope=None)


class _RecoveryFailClient(_FakeClient):
    async def snapshot(self, **kwargs) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        if self.snapshot_calls > 1:
            raise RuntimeError("recovery snapshot unavailable")
        return WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            snapshot='button "Initial target" [ref=@e1]',
            elements=[WebElementRef(ref_id="e1", role="button", name="Initial target")],
            raw={"refs": {"e1": {"role": "button"}}},
        )


class _NoteThenDoneRunner(_DoneRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_steps = 0
        self.recent_action_results: list[dict[str, object] | None] = []

    async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
        self.model_steps += 1
        observation = kwargs["observation"]
        self.recent_action_results.append(observation.recent_action_result)
        state = TraceCaptureState(assistant_text="noting" if self.model_steps == 1 else "noted")
        if self.model_steps == 1:
            state.tool_executions.append(
                ToolExecutionRecord(
                    tool_call_id="call-note",
                    tool_name="web_note",
                    tool_input={"text": "Remember Ada.", "reason": "Use later.", "kind": "fact"},
                    tool_output={
                        "status": "recorded",
                        "action": "note",
                        "text": "[fact] Remember Ada.\nReason: Use later.",
                        "note_text": "Remember Ada.",
                        "reason": "Use later.",
                        "kind": "fact",
                    },
                )
            )
            return WebAgentModelStepResult(trace_state=state, response_text="noting", meta={}, envelope=None)
        return WebAgentModelStepResult(trace_state=state, response_text="noted", meta={}, envelope=None)


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


class _CapturePromptBuilder:
    def __init__(self) -> None:
        self.allow_current_observation_image_values: list[bool] = []
        self.step_budget_values: list[dict[str, object] | None] = []

    def build_system_prompt_context(self, *, step_budget: dict[str, object] | None = None) -> dict[str, object]:
        self.step_budget_values.append(step_budget)
        return {}

    def build(self, **kwargs):
        self.allow_current_observation_image_values.append(bool(kwargs["allow_current_observation_image"]))
        from northcua.agents.web.prompt import WebAgentPromptEnvelope

        return WebAgentPromptEnvelope(messages=[], tools=kwargs["tools"], tool_choice=kwargs["tool_choice"])


class _FakeSharedStepExecutor:
    def __init__(self) -> None:
        self.context_values: list[dict[str, object]] = []

    async def run_step(self, **kwargs):
        self.context_values.append(dict(kwargs["context"]))
        envelope = kwargs["build_envelope"]([], None)
        return SimpleNamespace(
            trace_state=TraceCaptureState(assistant_text=""),
            response_text="",
            meta={},
            envelope=envelope,
        )


def _build_runner(tmp_path, runner_cls):
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.example.yaml",
        artifacts=ArtifactConfig(root_dir=str(tmp_path)),
    )
    app_config.web.allow_private_network_urls = True
    fake_client = _FakeClient()
    from northcua.agents.web.artifacts import WebArtifactStore

    store = WebArtifactStore.for_session(tmp_path, "web-runner-test", app_config.artifacts)
    runner = runner_cls(
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
    return runner, store, fake_client


def test_web_model_step_allows_current_observation_image_only_for_first_session_step(tmp_path) -> None:
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.example.yaml",
        artifacts=ArtifactConfig(root_dir=str(tmp_path)),
    )
    prompt_builder = _CapturePromptBuilder()
    executor = WebAgentModelStepExecutor(
        WebAgentModelStepComponents(
            app_config=app_config,
            agent_config=object(),
            execution_transport=object(),
            user_id="user",
            transport_capabilities=object(),
            prompt_builder=prompt_builder,
            tool_surface_service=object(),
        )
    )
    fake_step_executor = _FakeSharedStepExecutor()
    executor.step_executor = fake_step_executor
    observation = WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body"))
    step_budget = {
        "current_round_step": 1,
        "round_limit": 5,
        "remaining_round_steps": 5,
        "current_session_step": 1,
        "session_limit": 10,
    }

    asyncio.run(
        executor.run_step(
            user_query="first",
            session_id="session",
            step_index=1,
            session_step=1,
            step_budget=step_budget,
            allowed_tool_names=("web_snapshot", "web_click_ref"),
            observation=observation,
            prior_messages=[],
        )
    )
    asyncio.run(
        executor.run_step(
            user_query="next round",
            session_id="session",
            step_index=1,
            session_step=2,
            observation=observation,
            prior_messages=[],
        )
    )

    assert prompt_builder.allow_current_observation_image_values == [True, False]
    assert prompt_builder.step_budget_values == [step_budget, None]
    assert fake_step_executor.context_values[0]["step_budget"] == step_budget
    assert fake_step_executor.context_values[0]["allowed_tool_names"] == ["web_snapshot", "web_click_ref"]
    assert fake_step_executor.context_values[1]["allowed_tool_names"] == []


def test_web_runner_forwards_case_tool_allowlist_to_each_model_step(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _CaptureAllowedRunner)

    result = asyncio.run(
        runner.run(
            WebAgentRunnerRunSpec(
                user_query="finish",
                allowed_tool_names=("web_snapshot", "web_click_ref"),
            )
        )
    )

    assert result.status == WebRunStatus.SUCCESS
    assert runner.captured_allowed_tool_names == ("web_snapshot", "web_click_ref")


def test_runner_persists_dom_and_screenshot_each_step(tmp_path) -> None:
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.example.yaml",
        artifacts=ArtifactConfig(root_dir=str(tmp_path)),
    )
    fake_client = _FakeClient()
    artifact_store = app_config.artifacts
    from northcua.agents.web.artifacts import WebArtifactStore

    store = WebArtifactStore.for_session(tmp_path, "web-runner-test", artifact_store)
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

    assert result.status == WebRunStatus.SUCCESS
    assert result.final_answer == "finished"
    assert result.steps[0].session_id == "web-runner-test"
    assert store.dom_snapshot_path(1).read_text(encoding="utf-8") == "button @e1 Finish"
    assert store.dom_snapshot_json_path(1).exists()
    assert store.screenshot_path(1).read_bytes() == b"fake-png"
    assert store.result_path.exists()
    assert store.trace_path.exists()
    assert store.llm_trace_log_path.exists()
    assert fake_client.closed is True


def test_web_dom_artifact_projector_adds_visible_static_text() -> None:
    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.observation.artifact_dom import WebDomArtifactProjector

    snapshot = WebSnapshotObservation(
        url="https://example.test/search",
        title="Search",
        snapshot='link "climate-viz/climate-change-dashboard" [ref=@e30]',
        raw={
            "visible_text_frames": [
                {
                    "frame_index": 0,
                    "text": "climate-viz/climate-change-dashboard Interactive dashboard 12.8k stars",
                    "chars": 68,
                    "truncated": False,
                }
            ]
        },
    )

    artifact_text = WebDomArtifactProjector().text_for_snapshot(snapshot)

    assert "Page:" in artifact_text
    assert "Interactive refs:" in artifact_text
    assert 'link "climate-viz/climate-change-dashboard" [ref=@e30]' in artifact_text
    assert "Visible page text:" in artifact_text
    assert "12.8k stars" in artifact_text


def test_runner_persists_web_llm_trace_with_desktop_style_wrapper(tmp_path) -> None:
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.example.yaml",
        artifacts=ArtifactConfig(root_dir=str(tmp_path)),
    )
    fake_client = _FakeClient()
    from northcua.agents.web.artifacts import WebArtifactStore

    store = WebArtifactStore.for_session(tmp_path, "web-runner-trace-test", app_config.artifacts)
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

    assert result.status == WebRunStatus.SUCCESS
    trace_lines = store.llm_trace_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(trace_lines) == 1
    payload = json.loads(trace_lines[0])
    assert payload["source"] == "web"
    assert payload["model_request_stage"] == "logical_pre_provider_projection"
    assert payload["session_id"] == "web-runner-trace-test"
    assert payload["session_step"] == 1
    assert payload["round_num"] == 1
    assert payload["round_step"] == 1
    assert payload["request"] == "finish"
    assert payload["runtime_cursor"]["round_step"] == 1
    assert payload["llm_trace"]["assistant_text"] == "finished"
    assert payload["llm_trace"]["tool_executions"] == []


def test_runner_resumes_round_number_from_existing_web_trace(tmp_path) -> None:
    runner, store, _client = _build_runner(tmp_path, _DoneRunner)

    first = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="first")))
    second = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="second")))

    assert first.status == WebRunStatus.SUCCESS
    assert second.status == WebRunStatus.SUCCESS
    rows = [json.loads(line) for line in store.trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [(row["round_num"], row["session_step"]) for row in rows] == [(1, 1), (2, 2)]
    llm_rows = [
        json.loads(line) for line in store.llm_trace_log_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert [(row["round_num"], row["session_step"]) for row in llm_rows] == [(1, 1), (2, 2)]


def test_web_llm_trace_wrapper_preserves_llm_images_and_redacts_runtime_cursor(tmp_path) -> None:
    from northcua.agents.web.artifacts import WebArtifactStore
    from northcua.agents.web.runner.artifact_persistence import WebAgentArtifactPersistence

    app_config = NorthCUAAppConfig(
        agent_config_path="agent.example.yaml",
        artifacts=ArtifactConfig(root_dir=str(tmp_path)),
    )
    store = WebArtifactStore.for_session(tmp_path, "web-trace-redaction-test", app_config.artifacts)
    persistence = WebAgentArtifactPersistence(artifact_store=store, config=app_config.web)
    state = TraceCaptureState(assistant_text="look")
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-shot",
            tool_name="web_screenshot",
            tool_output={
                "status": "ok",
                "screenshot_url": "data:IMAGE/PNG;base64,QUJD",
                "message": "embedded screenshot=data:Image/Png;base64,QUJD after text",
                "result": {"data_url": "data:image/png;base64,QUJD"},
            },
        )
    )

    persistence.persist_llm_trace(
        state,
        session_id="web-trace-redaction-test",
        session_step=3,
        round_num=2,
        round_step=1,
        request="inspect",
        runtime_cursor={"round_step": 1, "image": "data:IMAGE/PNG;base64,QUJD"},
    )

    raw_line = store.llm_trace_log_path.read_text(encoding="utf-8").strip()
    payload = json.loads(raw_line)
    assert payload["source"] == "web"
    assert payload["runtime_cursor"]["image"] == "[redacted data URL; see screenshot artifact path]"
    assert payload["llm_trace"]["tool_executions"][0]["tool_output"]["screenshot_url"] == ("data:IMAGE/PNG;base64,QUJD")
    assert payload["llm_trace"]["tool_executions"][0]["tool_output"]["message"] == (
        "embedded screenshot=data:Image/Png;base64,QUJD after text"
    )
    assert "data:image" in raw_line.lower()
    assert "runtime_cursor" in payload


def test_runner_persists_initial_som_raw_and_annotated_artifacts(tmp_path) -> None:
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.example.yaml",
        artifacts=ArtifactConfig(root_dir=str(tmp_path)),
    )
    fake_client = _SomFakeClient()
    from northcua.agents.web.artifacts import WebArtifactStore

    store = WebArtifactStore.for_session(tmp_path, "web-runner-som-test", app_config.artifacts)
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

    assert result.status == WebRunStatus.SUCCESS
    assert store.screenshot_path(1).exists()
    assert store.screenshot_path(1, annotated=True).exists()
    assert store.screenshot_path(1).read_bytes() != store.screenshot_path(1, annotated=True).read_bytes()
    assert any(path.name.endswith(".annotation.json") for path in store.screenshots_dir.iterdir())


def test_runner_links_post_action_tool_som_artifact_without_exposing_path_in_tool_output(tmp_path) -> None:
    class _ToolSomRunner(_DoneRunner):
        async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
            tool_som_path = self.artifact_store.screenshots_dir / "tool_001_som.png"
            tool_som_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (80, 40), "white").save(tool_som_path, format="PNG")
            tool_som_data_url = "data:image/png;base64," + base64.b64encode(tool_som_path.read_bytes()).decode("ascii")
            state = TraceCaptureState(assistant_text="click")
            state.tool_executions.append(
                ToolExecutionRecord(
                    tool_call_id="call-click",
                    tool_name="web_click_ref",
                    tool_input={"ref": "@e1"},
                    tool_output={
                        "status": "ok",
                        "action": "click_ref",
                        "text": "Clicked.",
                        "screenshot_url": tool_som_data_url,
                        "observation": {
                            "observation_id": "post-click",
                            "settled": True,
                            "settle_reason": "stable",
                            "page": {"url": "https://example.com/after", "title": "After"},
                            "snapshot": {"text": "button after [ref=@e2]", "refs": [], "ref_count": 0},
                            "screenshot": {"kind": "set_of_mark", "image_attached": True, "annotation": {}},
                        },
                    },
                )
            )
            return WebAgentModelStepResult(trace_state=state, response_text="click", meta={}, envelope=None)

    runner, _store, _client = _build_runner(tmp_path, _ToolSomRunner)

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="click", max_steps=1)))

    assert result.steps[0].screenshot_path.endswith("step_001_terminal_raw.png")
    assert result.steps[0].terminal_verification is not None
    assert result.steps[0].terminal_verification.previous_screenshot_path.endswith("step_001_raw.png")
    assert result.steps[0].tool_artifacts[0].selected_image_path.endswith("tool_001_som.png")
    assert any(path.endswith("tool_001_som.png") for path in result.screenshots)
    assert "tool_001_som.png" not in json.dumps(result.steps[0].tools[0].tool_output)


def test_web_model_step_hides_iteration_limit_text_when_tool_succeeded() -> None:
    state = TraceCaptureState(assistant_text="")
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-1",
            tool_name="web_snapshot",
            tool_output={"status": "ok"},
        )
    )

    response = AgentKitModelStepExecutor.normalize_response_text(
        response_text="\n\n[Error: Maximum iteration limit reached.]",
        trace_state=state,
    )

    assert response == ""


def test_web_step_record_uses_captured_assistant_text_reasoning_and_tool_calls(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _DoneRunner)
    state = TraceCaptureState(assistant_text="I will inspect the page.", reasoning="Need current DOM first.")
    state.assistant_tool_calls.append(AssistantToolCallRecord(id="call-1", name="web_snapshot", arguments="{}"))
    step_result = WebAgentModelStepResult(
        trace_state=state,
        response_text="[Error: Maximum iteration limit reached.]",
        meta={},
        envelope=None,
    )

    step_record = runner._build_step_record(
        step_index=1,
        observation=WebObservationContext(snapshot=WebSnapshotObservation(snapshot="button @e1")),
        step_result=step_result,
        resolved_actions=[],
        artifact_paths={},
    )

    assert step_record.assistant_text == "I will inspect the page."
    assert step_record.reasoning == "Need current DOM first."
    assert step_record.assistant_tool_calls[0].name == "web_snapshot"


def test_web_step_record_hides_captured_iteration_limit_error_text(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _DoneRunner)
    state = TraceCaptureState(assistant_text="\n\n[Error: Maximum iteration limit reached.]")
    step_result = WebAgentModelStepResult(
        trace_state=state,
        response_text="\n\n[Error: Maximum iteration limit reached.]",
        meta={},
        envelope=None,
    )

    step_record = runner._build_step_record(
        step_index=1,
        observation=WebObservationContext(snapshot=WebSnapshotObservation(snapshot="button @e1")),
        step_result=step_result,
        resolved_actions=runner.action_resolver.resolve_all(state.tool_executions),
        artifact_paths={},
    )

    assert step_record.assistant_text == ""


def test_runner_preserves_multiple_web_tools_from_one_model_step(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _MultiToolRunner)

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="inspect", max_steps=1)))

    assert result.status == WebRunStatus.MAX_STEPS
    assert result.steps[0].tool_summary == "web_get_url, web_list_tabs"
    assert [tool.tool_name for tool in result.steps[0].tools] == ["web_get_url", "web_list_tabs"]
    assert result.steps[0].reasoning == "Need URL and tabs before final answer."
    assert result.steps[0].assistant_text == "inspect tabs and URL"


def test_runner_finishes_plain_final_text_without_web_tool_call(tmp_path) -> None:
    class _FinalTextRunner(_DoneRunner):
        async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
            state = TraceCaptureState(assistant_text="finish now")
            return WebAgentModelStepResult(
                trace_state=state,
                response_text="finish now",
                meta={},
                envelope=None,
            )

    runner, _store, _client = _build_runner(tmp_path, _FinalTextRunner)

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="finish")))

    assert result.status == WebRunStatus.SUCCESS
    assert result.terminal_status == WebTerminalStatus.COMPLETED
    assert result.final_answer == "finish now"
    assert result.steps[0].status == WebStepStatus.OK
    assert result.steps[0].error_text == ""
    assert result.steps[0].tool_summary == ""
    assert result.steps[0].tools == []


def test_runner_maps_infeasible_final_text_without_web_done(tmp_path) -> None:
    class _InfeasibleFinalTextRunner(_DoneRunner):
        async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
            text = "The requested file does not exist in this environment."
            state = TraceCaptureState(assistant_text=text)
            return WebAgentModelStepResult(
                trace_state=state,
                response_text=text,
                meta={},
                envelope=None,
            )

    runner, _store, _client = _build_runner(tmp_path, _InfeasibleFinalTextRunner)

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="finish")))

    assert result.status == WebRunStatus.FAILED
    assert result.success is False
    assert result.terminal_status == WebTerminalStatus.COMPLETED
    assert result.steps[0].status == WebStepStatus.OK
    assert result.steps[0].error_text == ""
    assert result.to_terminal_decision().reason == AgentTerminalReason.INFEASIBLE


def test_runner_records_tool_error_terminal_status(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _ToolErrorRunner)

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="click")))

    assert result.status == WebRunStatus.ERROR
    assert result.terminal_status == WebTerminalStatus.TOOL_ERROR
    assert result.tool_errors == ["browser backend crashed"]


def test_runner_reprompts_after_retryable_web_tool_error(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _RetryableToolErrorThenDoneRunner)

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="click then recover", max_steps=2)))

    assert result.status == WebRunStatus.SUCCESS
    assert result.final_answer == "recovered"
    assert len(result.steps) == 2
    assert result.steps[0].status == WebStepStatus.ERROR
    assert result.steps[0].tools[0].tool_output["retryable"] is True
    assert runner.recent_action_results[0] is None
    assert runner.recent_action_results[1]["retryable"] is True
    assert _client.snapshot_calls >= 2
    assert len(runner.observed_snapshots) == 2
    assert runner.observation_ids[0].startswith("initial-")
    assert runner.observation_ids[1].startswith("recovery-")
    recovery = result.steps[0].tools[0].tool_output["observation_recovery"]
    assert recovery["status"] == "recovered"
    assert recovery["reason"] == "stale_or_missing_ref"
    assert result.steps[0].tools[0].tool_output["observation"]["observation_id"].startswith("recovery-")


def test_runner_reprompts_after_framework_validation_tool_error(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _BareValidationToolErrorThenDoneRunner)

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="wait then recover", max_steps=2)))

    assert result.status == WebRunStatus.SUCCESS
    assert result.final_answer == "recovered"
    assert len(result.steps) == 2
    first_output = result.steps[0].tools[0].tool_output
    assert first_output["code"] == "invalid_params"
    assert first_output["retryable"] is True
    assert runner.recent_action_results[1]["retryable"] is True
    assert "observation_recovery" not in first_output
    assert runner.observation_ids[0] == runner.observation_ids[1]


def test_runner_refreshes_after_mutating_action_without_post_observation_bundle(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _ClickWithoutObservationThenDoneRunner)
    runner.app_config.web.post_action_observation_mode = "none"

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="click then finish", max_steps=2)))

    assert result.status == WebRunStatus.SUCCESS
    assert result.final_answer == "done"
    assert len(result.steps) == 2
    assert _client.snapshot_calls >= 2
    assert len(runner.observed_snapshots) == 2
    first_output = result.steps[0].tools[0].tool_output
    assert first_output["observation_recovery"]["reason"] == "mutation_without_observation"
    assert first_output["observation"]["observation_id"].startswith("recovery-")


def test_failed_recovery_does_not_restore_earlier_batch_observation(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _EarlierObservationThenMutationRunner)
    failing_client = _RecoveryFailClient()
    runner.client_factory = lambda _session_id: failing_client
    runner.app_config.web.initial_observation_mode = "snapshot"
    runner.app_config.web.post_action_observation_mode = "snapshot"

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="inspect then click", max_steps=2)))

    assert result.status == WebRunStatus.SUCCESS
    assert result.final_answer == "done"
    assert runner.observed_ref_counts == [1, 0]
    assert runner.observed_snapshots[1] == ""
    assert runner.observed_screenshots[1] is not None
    assert runner.observed_screenshots[1].annotation is None
    click_output = result.steps[0].tools[1].tool_output
    assert click_output["observation_recovery"]["status"] == "failed"
    assert click_output["observation_recovery"]["refs_invalidated"] is True
    assert "observation" not in click_output


def test_runner_treats_web_note_as_nonterminal_prompt_visible_action(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _NoteThenDoneRunner)

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="remember then finish", max_steps=2)))

    assert result.status == WebRunStatus.SUCCESS
    assert result.final_answer == "noted"
    assert result.steps[0].status == WebStepStatus.OK
    assert result.steps[0].tool_summary == "web_note"
    assert result.steps[0].tools[0].tool_output["text"] == "[fact] Remember Ada.\nReason: Use later."
    assert runner.recent_action_results[0] is None
    assert runner.recent_action_results[1]["text"] == "[fact] Remember Ada.\nReason: Use later."
    assert "note_text" not in runner.recent_action_results[1]
    assert "reason" not in runner.recent_action_results[1]


def test_runner_records_no_tool_terminal_status(tmp_path) -> None:
    runner, _store, _client = _build_runner(tmp_path, _NoToolRunner)

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="think")))

    assert result.status == WebRunStatus.ERROR
    assert result.terminal_status == WebTerminalStatus.NO_TOOL_CALL
    assert result.steps[0].status == WebStepStatus.ERROR
    assert result.steps[0].error_text == "no web-agent tool call"


def test_runner_records_max_steps_terminal_status(tmp_path) -> None:
    class _NonTerminalRunner(_DoneRunner):
        async def _run_model_step(self, **kwargs) -> WebAgentModelStepResult:
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

    runner, _store, _client = _build_runner(tmp_path, _NonTerminalRunner)

    result = asyncio.run(runner.run(WebAgentRunnerRunSpec(user_query="scroll", max_steps=1)))

    assert result.status == WebRunStatus.MAX_STEPS
    assert result.terminal_status == WebTerminalStatus.MAX_STEPS
    assert result.final_answer == "Task reached the round step limit before completion."
