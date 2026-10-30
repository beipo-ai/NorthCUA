"""Regression coverage for desktop observation sources, adapters, and executors."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from northcua.agents.desktop.observation import (
    LocalInternalProbeExecutor,
    ObservationCoordinator,
    ProbeExecutionRecord,
    RemoteInternalProbeExecutor,
    ToolResultStateSource,
)
from northcua.core.config import (
    AccessibilityObservationConfig,
    ArtifactConfig,
    NorthCUAAppConfig,
    ObservationConfig,
    ProbeToolCallConfig,
)
from northcua.core.enums import ResultStatus, ScreenshotSelectionKind, WebSocketMessageType
from northcua.core.errors import ConfigValidationError, ObservationProbeExecutionError
from northcua.core.models import CommandExecutionResultRecord, StepLogRecord, ToolResultArtifact
from northcua.runtime.artifacts.store import NorthCUAArtifactStore


def _png_data_url() -> str:
    image = Image.new("RGB", (2, 3), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


@pytest.mark.parametrize("backed_by_file", [True, False])
def test_tool_result_state_source_prefers_raw_over_selected(backed_by_file) -> None:
    source = ToolResultStateSource()
    step = StepLogRecord(
        session_id="sess-1",
        session_step=1,
        round_step=1,
        round_num=1,
        status="continue",
        request="Open Calculator",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-1",
                name="click",
                content="Clicked",
                status="SUCCESS",
                raw_image_path="/tmp/raw.png" if backed_by_file else "",
                raw_image_url="data:image/png;base64,BBBB",
                image_path="/tmp/selected.png",
                image_url="data:image/png;base64,AAAA",
                selected_image_kind=ScreenshotSelectionKind.OVERLAY,
                metadata={"cua_target": {"pid": 123, "window_id": 456}},
            )
        ],
    )

    result = asyncio.run(source.collect(session_step=2, previous_steps=[step]))

    assert result.current_clean_screenshot_path == ("/tmp/raw.png" if backed_by_file else "")
    assert result.current_clean_screenshot_url == "data:image/png;base64,BBBB"
    assert result.artifacts[0].url == ("" if backed_by_file else "data:image/png;base64,BBBB")
    assert result.metadata["current_state_source"] == "latest_tool_result:raw"
    assert result.metadata["selected_image_kind"] == "raw"
    assert result.metadata["cua_target"] == {"pid": 123, "window_id": 456}
    assert result.artifacts[0].metadata["cua_target"] == {"pid": 123, "window_id": 456}


def test_tool_result_state_source_never_uses_overlay_as_clean_current_state() -> None:
    source = ToolResultStateSource()
    step = StepLogRecord(
        session_id="sess-1",
        session_step=1,
        round_step=1,
        round_num=1,
        status="continue",
        request="Open Calculator",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-1",
                name="click",
                content="Clicked",
                status="SUCCESS",
                image_path="/tmp/overlay.png",
                image_url="data:image/png;base64,AAAA",
                selected_image_kind=ScreenshotSelectionKind.OVERLAY,
                source_image_path="/tmp/probe.png",
            )
        ],
    )

    result = asyncio.run(source.collect(session_step=2, previous_steps=[step]))

    assert result.current_clean_screenshot_path == ""
    assert result.current_clean_screenshot_url == ""
    assert result.selected_history_image_path == "/tmp/overlay.png"
    assert result.selected_history_image_url == "data:image/png;base64,AAAA"
    assert result.artifacts[0].role == "history"
    assert result.metadata["history_image_source"] == "latest_tool_result:overlay"


def test_local_internal_probe_executor_uses_registry_build_once(monkeypatch) -> None:
    build_calls: list[str] = []
    tool_calls: list[tuple[str, dict[str, object]]] = []

    class _DummyServer:
        async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> dict[str, str]:
            tool_calls.append((tool_name, dict(arguments)))
            return {"text": "ok"}

    def _build(name: str, context: object) -> _DummyServer:
        del context
        build_calls.append(name)
        return _DummyServer()

    monkeypatch.setattr("northcua.agents.desktop.observation.executors.NorthCUAMCPRegistry.build", _build)

    executor = LocalInternalProbeExecutor()
    steps = [
        ProbeToolCallConfig(
            server="DesktopNativeDataExecutor",
            tool="capture_desktop_screenshot",
            arguments={"all_screens": False},
            adapter="desktop_screenshot",
        ),
        ProbeToolCallConfig(
            server="DesktopNativeDataExecutor",
            tool="capture_desktop_screenshot",
            arguments={"all_screens": True},
            adapter="desktop_screenshot",
        ),
    ]

    records = asyncio.run(executor.execute_chain("command_probe", session_step=1, steps=steps))

    assert build_calls == ["DesktopNativeDataExecutor"]
    assert [record.tool_name for record in records] == ["capture_desktop_screenshot", "capture_desktop_screenshot"]
    assert tool_calls == [
        ("capture_desktop_screenshot", {"all_screens": False}),
        ("capture_desktop_screenshot", {"all_screens": True}),
    ]


def test_remote_internal_probe_executor_relays_results_through_websocket() -> None:
    sent_envelopes = []

    async def _send_payload(envelope):
        sent_envelopes.append(envelope)

    async def _await_results(call_ids):
        return [
            CommandExecutionResultRecord(
                call_id=call_ids[0],
                status=ResultStatus.SUCCESS,
                result={"text": "focused"},
            )
        ]

    executor = RemoteInternalProbeExecutor(
        session_id="sess-1",
        round_num=1,
        round_step=1,
        send_payload=_send_payload,
        await_results=_await_results,
    )

    records = asyncio.run(
        executor.execute_chain(
            "remote_probe",
            session_step=1,
            steps=[
                ProbeToolCallConfig(
                    server="DesktopNativeDataExecutor",
                    tool="capture_desktop_screenshot",
                    arguments={"all_screens": False},
                    adapter="desktop_screenshot",
                )
            ],
        )
    )

    assert len(sent_envelopes) == 1
    assert sent_envelopes[0].type == WebSocketMessageType.COMMAND_REQUEST
    assert sent_envelopes[0].payload.commands[0].tool_name == "capture_desktop_screenshot"
    assert sent_envelopes[0].payload.commands[0].server_name == "DesktopNativeDataExecutor"
    assert records[0].output == {"text": "focused"}


def test_remote_internal_probe_executor_fails_fast_on_probe_error() -> None:
    async def _send_payload(_envelope):
        return None

    async def _await_results(call_ids):
        return [
            CommandExecutionResultRecord(
                call_id=call_ids[0],
                status=ResultStatus.FAILURE,
                error="remote screenshot failed",
                result=None,
            )
        ]

    executor = RemoteInternalProbeExecutor(
        session_id="sess-1",
        round_num=1,
        round_step=1,
        send_payload=_send_payload,
        await_results=_await_results,
    )

    with pytest.raises(ObservationProbeExecutionError, match="remote screenshot failed"):
        asyncio.run(
            executor.execute_chain(
                "remote_probe",
                session_step=1,
                steps=[
                    ProbeToolCallConfig(
                        server="DesktopNativeDataExecutor",
                        tool="capture_desktop_screenshot",
                        arguments={"all_screens": False},
                        adapter="desktop_screenshot",
                    )
                ],
            )
        )


def test_coordinator_captures_initial_state_then_reuses_tool_image(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "direct", ArtifactConfig())
    app = NorthCUAAppConfig(
        agent_config_path="agent.yaml",
        observation=ObservationConfig(a11y=AccessibilityObservationConfig(enabled=False)),
    )
    app.agent_mcp_servers = [{"name": "DesktopNativeDataExecutor", "type": "stdio", "model_visible": False}]
    calls = []

    class Executor:
        async def execute_chain(self, chain_name, *, session_step, steps):
            calls.append((chain_name, session_step))
            assert [step.tool for step in steps] == ["capture_desktop_screenshot"]
            return [
                ProbeExecutionRecord(
                    chain_name=chain_name,
                    step_index=0,
                    server_name=steps[0].server,
                    tool_name=steps[0].tool,
                    arguments=steps[0].arguments,
                    output=_png_data_url(),
                    source="initial_screenshot",
                )
            ]

    coordinator = ObservationCoordinator(
        app_config=app,
        agent_config=SimpleNamespace(mcp_servers=[]),
        artifact_store=store,
        probe_executor=Executor(),
    )
    initial = asyncio.run(coordinator.collect(1, []))
    assert initial.current_clean_screenshot_url == _png_data_url()
    path = tmp_path / "returned.png"
    Image.new("RGB", (2, 3), "blue").save(path)
    step = StepLogRecord(
        session_id="direct",
        session_step=1,
        round_step=1,
        round_num=1,
        status="continue",
        request="click",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call",
                name="click",
                content="clicked",
                status="SUCCESS",
                raw_image_path=str(path),
            )
        ],
    )
    state = asyncio.run(coordinator.collect(2, [step]))
    assert state.current_clean_screenshot_path == str(path)
    assert state.current_clean_screenshot_url != initial.current_clean_screenshot_url
    assert calls == [("desktop_snapshot_default", 1)]
    assert state.observation_metadata["current_state_source"] == "latest_tool_result:raw"


def test_coordinator_rejects_model_visible_internal_capture(tmp_path) -> None:
    app = NorthCUAAppConfig(
        agent_config_path="agent.yaml",
        observation=ObservationConfig(a11y=AccessibilityObservationConfig(enabled=True, server_name="CustomA11y")),
    )
    app.agent_mcp_servers = [
        {"name": "DesktopNativeDataExecutor", "type": "stdio", "model_visible": False},
        {"name": "CustomA11y", "type": "stdio", "model_visible": True},
    ]
    with pytest.raises(ConfigValidationError, match="internal-only"):
        ObservationCoordinator(
            app_config=app,
            agent_config=SimpleNamespace(mcp_servers=[]),
            artifact_store=NorthCUAArtifactStore.for_session(tmp_path, "direct", ArtifactConfig()),
            probe_executor=SimpleNamespace(),
        )


@pytest.mark.parametrize(
    "section, values",
    [
        ("observation", {"policy": {"type": "pre_round_probe"}}),
        ("observation", {"policy": {"type": "tool_result_state", "projection": "user_observation"}}),
        ("observation", {"probe_chains": {}}),
        ("desktop", {"use_post_action_image_as_current": False}),
        ("desktop", {"action_post_wait_seconds": 1.0}),
        ("desktop", {"clipboard_first_typing": False}),
        ("prompt", {"image_window_policy": "generic_budget"}),
    ],
)
def test_removed_observation_switches_fail_instead_of_being_silently_ignored(section, values) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NorthCUAAppConfig.model_validate({"agent_config_path": "agent.yaml", section: values})
