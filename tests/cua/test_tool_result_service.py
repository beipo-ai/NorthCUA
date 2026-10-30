"""Regression coverage for desktop tool result normalization and artifact selection."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

from northcua.agentkit.controller_dispatch import command_result_record
from northcua.agentkit.prompt_history import tool_result_history_messages
from northcua.agents.desktop.prompt import HistoryTurn
from northcua.agents.desktop.services.tool_results import ToolResultService
from northcua.core.config import ArtifactConfig, PromptPolicyConfig
from northcua.core.models import ToolExecutionRecord
from northcua.runtime.artifacts.store import NorthCUAArtifactStore
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef
from northcua_protocol.payloads import CommandResult
from northcua_protocol.state import CommandState


def test_tool_result_service_unwraps_nested_result_json_and_persists_raw_image(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"png-bytes")

    store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-1",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    service = ToolResultService(
        PromptPolicyConfig(history_action_trace_enabled=False),
        store,
    )

    execution = ToolExecutionRecord(
        tool_call_id="call-1",
        tool_name="keypress",
        tool_input={"key": "enter"},
        tool_output={
            "result": json.dumps(
                {
                    "text": "Pressed enter.",
                    "screenshot_path": str(source),
                    "screenshot_format": "local_path",
                }
            )
        },
        llm_tool_output=None,
    )

    artifacts = service.build(
        executions=[execution],
        session_step=0,
        pre_action_screenshot_path="",
        is_initial_turn=False,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.content == "Pressed enter."
    assert artifact.raw_image_path.endswith("action_step0_tool_1.png")
    assert Path(artifact.raw_image_path).exists()
    assert artifact.selected_image_kind.value == "raw"


def test_tool_result_service_persists_bare_data_url_without_text_bloat(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(buffer, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-data-url",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    service = ToolResultService(
        PromptPolicyConfig(history_action_trace_enabled=False),
        store,
    )

    artifacts = service.build(
        executions=[
            ToolExecutionRecord(
                tool_call_id="call-screenshot",
                tool_name="capture_desktop_screenshot",
                tool_input={},
                tool_output=data_url,
                llm_tool_output=data_url,
            )
        ],
        session_step=1,
        pre_action_screenshot_path="",
        is_initial_turn=False,
    )

    artifact = artifacts[0]
    assert artifact.content == "Screenshot captured."
    assert "data:image" not in artifact.content
    assert Path(artifact.raw_image_path).exists()
    assert artifact.raw_image_url.startswith("data:image/png;base64,")


def test_tool_result_service_persists_remote_inline_artifact_on_controller(tmp_path: Path) -> None:
    """Remote image artifacts are stored locally before evaluator evidence reads them."""

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    artifact_ref = ArtifactRef.from_inline_bytes(
        artifact_id="cmd-click-screenshot_url",
        kind=ArtifactKind.SCREENSHOT,
        data=png_bytes,
        mime_type="image/png",
    )
    remote_record = command_result_record(
        "call-click",
        CommandResult(
            command_id="cmd-click",
            session_id="sess-remote",
            lease_id="lease",
            device_id="sandbox-device",
            status=CommandState.SUCCEEDED,
            result={
                "text": "Clicked",
                "screenshot_path": "/tmp/sandbox-local-action.png",
                "screenshot_url": "",
                "screenshot_url_artifact_id": artifact_ref.artifact_id,
            },
            artifacts=[artifact_ref],
        ),
    )

    store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-remote",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    service = ToolResultService(PromptPolicyConfig(history_action_trace_enabled=False), store)

    artifacts = service.build(
        executions=[
            ToolExecutionRecord(
                tool_call_id=remote_record.call_id,
                tool_name="click",
                tool_input={"x": 10, "y": 20},
                tool_output=remote_record.result,
                llm_tool_output=remote_record.result,
            )
        ],
        session_step=2,
        pre_action_screenshot_path="",
        is_initial_turn=False,
    )

    assert len(artifacts) == 1
    tool_artifact = artifacts[0]
    assert tool_artifact.raw_image_path.endswith("action_step2_tool_1.png")
    assert Path(tool_artifact.raw_image_path).exists()
    with Image.open(tool_artifact.raw_image_path) as persisted:
        assert persisted.size == (1, 1)
    assert tool_artifact.source_image_path == ""
    assert tool_artifact.selected_image_kind.value == "raw"


def test_tool_result_service_keeps_error_results_text_only_even_with_screenshot(tmp_path: Path) -> None:
    source = tmp_path / "failed-action.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(source)
    pre_action = tmp_path / "pre-action.png"
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(pre_action)
    store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-tool-error",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    service = ToolResultService(PromptPolicyConfig(history_action_trace_enabled=True), store)

    artifacts = service.build(
        executions=[
            ToolExecutionRecord(
                tool_call_id="call-failed-click",
                tool_name="click",
                tool_input={"x": 10, "y": 20},
                tool_input_pixels={"x": 10, "y": 20},
                tool_output={
                    "status": "error",
                    "error": "Click target was outside the visible screenshot.",
                    "screenshot_path": str(source),
                },
                llm_tool_output={
                    "status": "error",
                    "error": "Click target was outside the visible screenshot.",
                    "screenshot_path": str(source),
                },
                is_error=True,
            )
        ],
        session_step=4,
        pre_action_screenshot_path=str(pre_action),
        is_initial_turn=False,
    )

    artifact = artifacts[0]
    assert artifact.is_error is True
    assert artifact.content == "click failed: Click target was outside the visible screenshot."
    assert artifact.raw_image_path == ""
    assert artifact.raw_image_url == ""
    assert artifact.overlay_image_path == ""
    assert artifact.overlay_image_url == ""
    assert artifact.image_path == ""
    assert artifact.image_url == ""
    assert artifact.source_image_path == ""
    assert artifact.overlay_spec is None
    assert artifact.selected_image_kind.value == "none"
    assert not list((tmp_path / "logs").glob("**/action_step4_tool_1.png"))
    assert not list((tmp_path / "logs").glob("**/action_step4_history_overlay_1.png"))


def test_tool_result_service_uses_shared_overlay_service_for_action_traces(tmp_path: Path) -> None:
    pre_action = tmp_path / "pre-action.png"
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(pre_action)

    store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-2",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    service = ToolResultService(
        PromptPolicyConfig(history_action_trace_enabled=True),
        store,
    )

    execution = ToolExecutionRecord(
        tool_call_id="call-2",
        tool_name="click",
        tool_input={"x": 500, "y": 500},
        tool_input_model={"x": 500, "y": 500},
        tool_input_pixels={"x": 40, "y": 30},
        tool_output={"text": "Clicked primary CTA."},
        llm_tool_output={"text": "Clicked primary CTA."},
    )

    artifacts = service.build(
        executions=[execution],
        session_step=1,
        pre_action_screenshot_path=str(pre_action),
        is_initial_turn=False,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.content == 'click succeeded with model-space input {"x":500,"y":500}.'
    assert artifact.overlay_image_path.endswith("action_step1_history_overlay_1.png")
    assert Path(artifact.overlay_image_path).exists()
    assert artifact.overlay_spec["action_function"] == "click"
    assert artifact.overlay_spec["primary_point"] == {"x": 40, "y": 30}
    assert artifact.selected_image_kind.value == "overlay"
    assert execution.tool_output == {"text": "Clicked primary CTA."}


def test_tool_result_service_skips_spatial_overlay_without_pixel_arguments(tmp_path: Path) -> None:
    pre_action = tmp_path / "pre-action.png"
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(pre_action)

    store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-no-pixels",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    service = ToolResultService(
        PromptPolicyConfig(history_action_trace_enabled=True),
        store,
    )

    execution = ToolExecutionRecord(
        tool_call_id="call-legacy-click",
        tool_name="click",
        tool_input={"x": 500, "y": 500},
        tool_output={"text": "Clicked without converted pixel metadata."},
    )

    artifacts = service.build(
        executions=[execution],
        session_step=2,
        pre_action_screenshot_path=str(pre_action),
        is_initial_turn=False,
    )

    artifact = artifacts[0]
    assert artifact.overlay_image_path == ""
    assert artifact.overlay_spec is None
    assert artifact.selected_image_kind.value == "none"


def test_note_tool_result_stays_text_only_without_overlay_or_image_history(tmp_path: Path) -> None:
    pre_action = tmp_path / "pre-action.png"
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(pre_action)

    store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-note",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    service = ToolResultService(
        PromptPolicyConfig(history_action_trace_enabled=True),
        store,
    )
    payload = {
        "text": "[fact] Ada email is ada.lovelace@northcua.test.",
        "status": "recorded",
        "kind": "fact",
        "note_text": "Ada email is ada.lovelace@northcua.test.",
        "reason": "Use the exact value when filling the contact sheet.",
        "screenshot_policy": "none",
    }
    execution = ToolExecutionRecord(
        tool_call_id="call-note",
        tool_name="note",
        tool_input={"text": payload["note_text"], "reason": payload["reason"], "kind": "fact"},
        tool_output=payload,
        llm_tool_output=payload,
    )

    artifacts = service.build(
        executions=[execution],
        session_step=3,
        pre_action_screenshot_path=str(pre_action),
        is_initial_turn=False,
    )
    artifact = artifacts[0]
    desktop_history = HistoryTurn(
        tool_results=[
            {
                "tool_call_id": artifact.tool_call_id,
                "content": artifact.content,
                "image_url": artifact.image_url,
                "is_error": artifact.is_error,
            }
        ]
    ).to_messages()
    agentkit_history = tool_result_history_messages([execution])

    assert artifact.content == "[fact] Ada email is ada.lovelace@northcua.test."
    assert artifact.raw_image_path == ""
    assert artifact.raw_image_url == ""
    assert artifact.overlay_image_path == ""
    assert artifact.overlay_image_url == ""
    assert artifact.image_path == ""
    assert artifact.image_url == ""
    assert artifact.source_image_path == ""
    assert artifact.overlay_spec is None
    assert artifact.selected_image_kind.value == "none"
    desktop_tool_result = desktop_history[0].content[0]
    assert desktop_tool_result.content == "[fact] Ada email is ada.lovelace@northcua.test."
    assert getattr(desktop_tool_result.content, "type", None) != "image"
    agentkit_tool_result = agentkit_history[0].content[0]
    assert isinstance(agentkit_tool_result.content, str)
    assert "ada.lovelace@northcua.test" in agentkit_tool_result.content
    assert "data:image" not in agentkit_tool_result.content
