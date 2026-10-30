from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nexau.core.messages import ImageBlock, TextBlock, ToolResultBlock
from PIL import Image
from pydantic import ValidationError

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.evaluator.artifacts import EvaluatorArtifactLayout, EvaluatorArtifactWriter
from northcua.agentkit.evaluator.evidence.selection import EvidenceSelector
from northcua.agentkit.evaluator.protocol import (
    EvaluationEvidenceManifest,
    EvaluationEvidenceRef,
    RoundEvaluationBundle,
    RoundEvaluationStepRef,
)
from northcua.agentkit.evaluator.routing import RemoteEvaluatorClientToolDispatcher
from northcua.agentkit.evaluator.tools import ArtifactInspectionToolFactory
from northcua.agents.web.artifacts import WebArtifactStore
from northcua.agents.web.backend import WebActionCommandResult
from northcua.agents.web.domain.actions import WebAgentActionType
from northcua.agents.web.domain.evaluation_artifacts import (
    WebObservationArtifactRole,
    WebTerminalVerificationStatus,
    WebToolArtifactAttributionSource,
    WebToolArtifactImageKind,
    WebToolImageArtifactEvent,
    WebToolInvocationArtifactEvent,
)
from northcua.agents.web.domain.observations import (
    WebBoundingBox,
    WebElementRef,
    WebObservationContext,
    WebScreenshotAnnotation,
    WebScreenshotMark,
    WebScreenshotObservation,
    WebSnapshotObservation,
)
from northcua.agents.web.domain.results import WebAgentResult, WebRunStatus, WebTerminalStatus
from northcua.agents.web.evaluation.adapters import WebRoundEvaluationBundleAdapter
from northcua.agents.web.evaluation.domain import WebRoundEvaluationReport
from northcua.agents.web.evaluation.offline import load_web_offline_evaluation_bundle
from northcua.agents.web.evaluation.prompts import WebEvaluatorPromptBuilder
from northcua.agents.web.evaluation.service import build_web_round_evaluator_spec
from northcua.agents.web.evaluation.service.report_normalizer import normalize_web_final_report_page_state
from northcua.agents.web.evaluation.service.round_evaluator import WebRoundEvaluatorService
from northcua.agents.web.evaluation.tools import WebBrowserLiveProbeDispatcher
from northcua.agents.web.mcp.action_service import WebAgentMCPActionService
from northcua.agents.web.runner.action_resolution import ResolvedWebAgentAction, WebAgentActionResolver
from northcua.agents.web.runner.artifact_persistence import WebAgentArtifactPersistence
from northcua.agents.web.runner.model_step import WebAgentModelStepResult
from northcua.agents.web.runner.step_records import WebAgentStepRecordProjector
from northcua.agents.web.session import RegisteredWebAgentSession, WebAgentSessionRegistry
from northcua.core.config import ArtifactConfig, WebAgentConfig
from northcua.core.enums import EvaluationEvidenceKind, EvaluationSchedule, EvaluationToolPlacement, ResultStatus
from northcua.core.models import CommandExecutionResultRecord, ToolExecutionRecord
from northcua.runtime.mcp.invocation_context import MCPInvocationContext, use_mcp_invocation_context
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult
from tests.support.web_agent_trace_qa import run_qa


class _TerminalClient:
    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        del kwargs
        return WebScreenshotObservation(data_url=_png_data_url("blue"))


class _NoScreenshotClient:
    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        del kwargs
        return WebScreenshotObservation()


class _ProbeClient(_TerminalClient):
    def __init__(self) -> None:
        self.screenshot_kwargs: list[dict[str, object]] = []
        self.opened_urls: list[str] = []
        self.clicked_refs: list[str] = []

    async def snapshot(self, **kwargs) -> WebSnapshotObservation:
        return WebSnapshotObservation(
            url="http://127.0.0.1/probe",
            title="Probe",
            snapshot=f"snapshot depth={kwargs.get('depth')}",
            elements=[WebElementRef(ref_id="e1", role="button", name="Probe")],
            raw={"debug_path": "/tmp/private-dom.json", "image": "data:IMAGE/PNG;base64,AAAA"},
        )

    async def list_tabs(self) -> WebActionCommandResult:
        return WebActionCommandResult(success=True, data={"tabs": [{"title": "Probe", "url": "http://127.0.0.1"}]})

    async def get_url(self) -> WebActionCommandResult:
        return WebActionCommandResult(success=True, data={"url": "http://127.0.0.1/probe"})

    async def open(self, url: str) -> WebActionCommandResult:
        self.opened_urls.append(url)
        return WebActionCommandResult(success=True, data={"url": url, "opened": True})

    async def click_ref(self, ref: str) -> WebActionCommandResult:
        self.clicked_refs.append(ref)
        return WebActionCommandResult(success=True, data={"ref": ref, "clicked": True})

    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        return WebActionCommandResult(
            success=True,
            data={"boxes": {ref: {"x": 10.0, "y": 20.0, "width": 120.0, "height": 30.0} for ref in refs}},
        )

    async def get_ref_states(self, refs: list[str]) -> WebActionCommandResult:
        return WebActionCommandResult(
            success=True,
            data={"states": {ref: {"visible": True, "enabled": True} for ref in refs}},
        )

    async def get_viewport_size(self) -> WebActionCommandResult:
        return WebActionCommandResult(success=True, data={"width": 640, "height": 480})

    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        self.screenshot_kwargs.append(dict(kwargs))
        return WebScreenshotObservation(
            data_url=_png_data_url("blue"),
            width=640,
            height=480,
            annotation=WebScreenshotAnnotation(
                source_screenshot_path="/tmp/source.png",
                overlay_screenshot_path="/tmp/overlay.png",
            ),
        )


class _CaptureErrorProbeClient(_ProbeClient):
    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        self.screenshot_kwargs.append(dict(kwargs))
        return WebScreenshotObservation(
            width=320,
            height=240,
            raw={"capture_error": "failed opening /tmp/private-shot.png data:IMAGE/PNG;base64,AAAA"},
        )


def test_web_step_keeps_current_screenshot_separate_from_tool_result_image(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    observation = _observation("Before", "https://example.test/before", "button @e1 Before", "white")
    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=1, observation=observation)
    )
    observation_artifact = persistence.build_observation_artifact(observation=observation, paths=paths)

    tool_som_path = persistence.artifact_store.screenshots_dir / "tool_0001_som.png"
    _write_png(tool_som_path, "green")
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True},
        tool_call_id="call-shot",
        tool_output={
            "status": "ok",
            "text": "Captured.",
            "action": "screenshot",
            "screenshot_url": _png_data_url("green"),
            "observation": {
                "page": {"url": "https://example.test/after", "title": "After"},
                "screenshot": {"annotation": {"kind": "set_of_mark", "marks": [{"ref": "@e1", "label": "1"}]}},
            },
        },
    )
    tool_artifacts = persistence.build_tool_result_artifacts(actions=[action], action_screenshot_paths=[tool_som_path])
    step_record = WebAgentStepRecordProjector(action_resolver=WebAgentActionResolver()).build_step_record(
        session_id="test-session",
        step_index=1,
        round_num=1,
        session_step=1,
        runtime_cursor={},
        observation=observation,
        step_result=_step_result(action),
        resolved_actions=[action],
        artifact_paths=paths,
        observation_artifact=observation_artifact,
        tool_artifacts=tool_artifacts,
    )

    assert step_record.screenshot_path == str(paths["screenshot_path"])
    assert step_record.observation_artifact is not None
    assert step_record.observation_artifact.role == WebObservationArtifactRole.CURRENT
    assert step_record.tool_artifacts[0].selected_image_kind == WebToolArtifactImageKind.SOM
    assert step_record.tool_artifacts[0].selected_image_path == str(tool_som_path)
    assert step_record.tool_artifacts[0].tool_call_id == "call-shot"


def test_current_annotated_screenshot_persists_som_sidecar(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    source = tmp_path / "source.png"
    overlay = tmp_path / "source_som.png"
    _write_png(source, "white")
    _write_png(overlay, "green")
    overlay.with_suffix(".annotation.json").write_text(
        json.dumps({"kind": "set_of_mark", "marks": [{"ref": "@e1", "label": "1"}]}),
        encoding="utf-8",
    )
    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(
            url="https://example.test",
            title="Annotated",
            snapshot='button "Annotated" [ref=@e1]',
            elements=[WebElementRef(ref_id="e1", role="button", name="Annotated")],
        ),
        screenshot=WebScreenshotObservation(
            annotation=WebScreenshotAnnotation(
                marks=[
                    WebScreenshotMark(
                        ref="@e1",
                        label="1",
                        role="button",
                        name="Annotated",
                        bbox_css=WebBoundingBox(x=1, y=2, width=3, height=4),
                        bbox_image=WebBoundingBox(x=1, y=2, width=3, height=4),
                    )
                ],
                source_screenshot_path=str(source),
                overlay_screenshot_path=str(overlay),
            )
        ),
    )

    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=1, observation=observation)
    )

    annotated_path = paths["annotated_screenshot_path"]
    sidecar = annotated_path.with_suffix(".annotation.json")
    assert annotated_path.name == "step_001_annotated.png"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["marks"][0]["ref"] == "@e1"


def test_tool_result_images_attach_to_last_image_producing_action_not_note(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    tool_som_path = persistence.artifact_store.screenshots_dir / "tool_0001_som.png"
    _write_png(tool_som_path, "green")
    note_action = ResolvedWebAgentAction(
        action=WebAgentActionType.NOTE,
        tool_name="web_note",
        tool_input={"text": "remember"},
        tool_output={"status": "ok", "text": "noted"},
        tool_call_id="call-note",
    )
    screenshot_action = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True},
        tool_output={
            "status": "ok",
            "text": "Captured.",
            "screenshot_url": _png_data_url("green"),
            "result": {"kind": "set_of_mark", "image_attached": True},
        },
        tool_call_id="call-shot",
    )
    trailing_note_action = ResolvedWebAgentAction(
        action=WebAgentActionType.NOTE,
        tool_name="web_note",
        tool_input={"text": "done"},
        tool_output={"status": "ok", "text": "done"},
        tool_call_id="call-note-2",
    )

    artifacts = persistence.build_tool_result_artifacts(
        actions=[note_action, screenshot_action, trailing_note_action],
        action_screenshot_paths=[tool_som_path],
    )

    by_call_id = {artifact.tool_call_id: artifact for artifact in artifacts}
    assert by_call_id["call-note"].selected_image_kind == WebToolArtifactImageKind.NONE
    assert by_call_id["call-shot"].selected_image_path == str(tool_som_path)
    assert by_call_id["call-note-2"].selected_image_kind == WebToolArtifactImageKind.NONE


def test_tool_result_images_attribute_by_data_url_hash_not_path_order(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    raw_one = persistence.artifact_store.screenshots_dir / "tool_0001.png"
    som_one = persistence.artifact_store.screenshots_dir / "tool_0001_som.png"
    raw_two = persistence.artifact_store.screenshots_dir / "tool_0002.png"
    som_two = persistence.artifact_store.screenshots_dir / "tool_0002_som.png"
    for path, color in ((raw_one, "red"), (som_one, "green"), (raw_two, "blue"), (som_two, "yellow")):
        _write_png(path, color)
    _write_som_sidecar(som_one, source=raw_one)
    _write_som_sidecar(som_two, source=raw_two)
    first = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True},
        tool_call_id="call-shot-1",
        tool_output={"status": "ok", "screenshot_url": _png_data_url("green"), "result": {"image_attached": True}},
    )
    second = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True},
        tool_call_id="call-shot-2",
        tool_output={"status": "ok", "screenshot_url": _png_data_url("yellow"), "result": {"image_attached": True}},
    )

    artifacts = persistence.build_tool_result_artifacts(
        actions=[first, second],
        action_screenshot_paths=[som_two, raw_two, raw_one, som_one],
    )

    assert artifacts[0].raw_image_path == str(raw_one)
    assert artifacts[0].som_image_path == str(som_one)
    assert artifacts[0].metadata["attribution_source"] == WebToolArtifactAttributionSource.IMAGE_HASH.value
    assert artifacts[1].raw_image_path == str(raw_two)
    assert artifacts[1].som_image_path == str(som_two)
    assert artifacts[1].metadata["attribution_source"] == WebToolArtifactAttributionSource.IMAGE_HASH.value


def test_tool_result_images_do_not_fallback_to_order_without_attribution(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    raw_path = persistence.artifact_store.screenshots_dir / "tool_0001.png"
    som_path = persistence.artifact_store.screenshots_dir / "tool_0001_som.png"
    _write_png(raw_path, "red")
    _write_png(som_path, "green")
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True},
        tool_call_id="call-shot",
        tool_output={"status": "ok", "result": {"image_attached": True}},
    )

    artifacts = persistence.build_tool_result_artifacts(
        actions=[action],
        action_screenshot_paths=[raw_path, som_path],
    )

    assert artifacts[0].selected_image_kind == WebToolArtifactImageKind.NONE
    assert artifacts[0].raw_image_path == ""
    assert artifacts[0].som_image_path == ""
    assert artifacts[0].metadata["attribution_source"] == WebToolArtifactAttributionSource.NONE.value


def test_tool_result_images_do_not_hash_match_ambiguous_identical_images(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    first_path = persistence.artifact_store.screenshots_dir / "tool_0001.png"
    second_path = persistence.artifact_store.screenshots_dir / "tool_0002.png"
    _write_png(first_path, "green")
    _write_png(second_path, "green")
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": False},
        tool_call_id="call-shot",
        tool_output={"status": "ok", "screenshot_url": _png_data_url("green")},
    )

    artifacts = persistence.build_tool_result_artifacts(
        actions=[action],
        action_screenshot_paths=[first_path, second_path],
    )

    assert artifacts[0].selected_image_kind == WebToolArtifactImageKind.NONE
    assert artifacts[0].metadata["attribution_source"] == WebToolArtifactAttributionSource.NONE.value


def test_tool_result_images_use_tool_artifact_events_for_large_pathless_outputs(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    raw_path = persistence.artifact_store.screenshots_dir / "tool_0001.png"
    som_path = persistence.artifact_store.screenshots_dir / "tool_0001_som.png"
    _write_png(raw_path, "green")
    _write_png(som_path, "green")
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.CLICK_REF,
        tool_name="web_click_ref",
        tool_input={"ref": "@e1"},
        tool_call_id="call-click",
        tool_output={
            "status": "ok",
            "result": {"clicked": True},
            "observation": {
                "page": {"url": "https://example.test/after", "title": "After"},
                "screenshot": {"kind": "set_of_mark", "image_attached": False, "annotation": {}},
            },
        },
    )
    artifact_events = [
        WebToolImageArtifactEvent(
            sequence=1,
            invocation_id=1,
            action=WebAgentActionType.CLICK_REF.value,
            raw_image_path=str(raw_path),
            selected_image_path=str(raw_path),
            selected_image_kind=WebToolArtifactImageKind.RAW,
        ),
        WebToolImageArtifactEvent(
            sequence=2,
            invocation_id=1,
            action=WebAgentActionType.CLICK_REF.value,
            raw_image_path=str(raw_path),
            som_image_path=str(som_path),
            selected_image_path=str(som_path),
            selected_image_kind=WebToolArtifactImageKind.SOM,
        ),
    ]

    artifacts = persistence.build_tool_result_artifacts(
        actions=[action],
        action_screenshot_paths=[],
        artifact_invocations=[
            WebToolInvocationArtifactEvent(
                sequence=1,
                invocation_id=1,
                action=WebAgentActionType.CLICK_REF.value,
            )
        ],
        artifact_events=artifact_events,
    )

    assert artifacts[0].selected_image_kind == WebToolArtifactImageKind.RAW
    assert artifacts[0].raw_image_path == str(raw_path)
    assert artifacts[0].som_image_path == str(som_path)
    assert artifacts[0].metadata["attribution_source"] == WebToolArtifactAttributionSource.EXPLICIT_PATH.value


def test_tool_result_images_bind_events_by_exact_invocation_not_same_action_order(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    raw_path = persistence.artifact_store.screenshots_dir / "tool_0002.png"
    som_path = persistence.artifact_store.screenshots_dir / "tool_0002_som.png"
    _write_png(raw_path, "blue")
    _write_png(som_path, "yellow")
    first_no_image = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True},
        tool_call_id="call-shot-1",
        tool_output={"status": "ok", "result": {"image_attached": False}},
    )
    second_with_image = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True},
        tool_call_id="call-shot-2",
        tool_output={"status": "ok", "result": {"image_attached": False}},
    )
    invocation_events = [
        WebToolInvocationArtifactEvent(
            sequence=1,
            invocation_id=1,
            action=WebAgentActionType.SCREENSHOT.value,
        ),
        WebToolInvocationArtifactEvent(
            sequence=2,
            invocation_id=2,
            action=WebAgentActionType.SCREENSHOT.value,
        ),
    ]
    artifact_events = [
        WebToolImageArtifactEvent(
            sequence=1,
            invocation_id=2,
            action=WebAgentActionType.SCREENSHOT.value,
            raw_image_path=str(raw_path),
            som_image_path=str(som_path),
            selected_image_path=str(som_path),
            selected_image_kind=WebToolArtifactImageKind.SOM,
        )
    ]

    artifacts = persistence.build_tool_result_artifacts(
        actions=[first_no_image, second_with_image],
        action_screenshot_paths=[],
        artifact_invocations=invocation_events,
        artifact_events=artifact_events,
    )

    assert artifacts[0].selected_image_kind == WebToolArtifactImageKind.NONE
    assert artifacts[0].selected_image_path == ""
    assert artifacts[1].selected_image_kind == WebToolArtifactImageKind.RAW
    assert artifacts[1].raw_image_path == str(raw_path)
    assert artifacts[1].som_image_path == str(som_path)
    assert artifacts[1].metadata["invocation_id"] == 2


def test_preflight_tool_error_placeholder_prevents_same_action_image_shift(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    registered = _registered_session(tmp_path, client=_ProbeClient(), session_id="preflight-placeholder")
    registry = WebAgentSessionRegistry()
    registry.register(registered)
    service = WebAgentMCPActionService(registry=registry)
    raw_path = persistence.artifact_store.screenshots_dir / "tool_0002.png"
    som_path = persistence.artifact_store.screenshots_dir / "tool_0002_som.png"
    _write_png(raw_path, "blue")
    _write_png(som_path, "yellow")
    with use_mcp_invocation_context(MCPInvocationContext(session_id="preflight-placeholder")):
        service.record_preflight_tool_invocation(WebAgentActionType.SCREENSHOT)
    valid_context = registered.begin_tool_artifact_invocation(WebAgentActionType.SCREENSHOT.value)
    registered.record_tool_image_artifact(
        raw_image_path=str(raw_path),
        som_image_path=str(som_path),
        selected_image_path=str(som_path),
        selected_image_kind=WebToolArtifactImageKind.SOM,
    )
    registered.restore_tool_artifact_invocation(valid_context)
    failed_preflight = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True, "full_page": True},
        tool_call_id="call-shot-invalid",
        tool_output={"status": "error", "action": "screenshot", "code": "validation_error"},
        is_error=True,
    )
    valid_screenshot = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True},
        tool_call_id="call-shot-valid",
        tool_output={"status": "ok", "result": {"image_attached": False}},
    )

    artifacts = persistence.build_tool_result_artifacts(
        actions=[failed_preflight, valid_screenshot],
        action_screenshot_paths=[],
        artifact_invocations=list(registered.tool_invocation_artifact_events),
        artifact_events=list(registered.tool_image_artifact_events),
    )

    assert artifacts[0].selected_image_kind == WebToolArtifactImageKind.NONE
    assert artifacts[0].selected_image_path == ""
    assert artifacts[1].selected_image_kind == WebToolArtifactImageKind.RAW
    assert artifacts[1].selected_image_path == str(raw_path)
    assert artifacts[1].metadata["invocation_id"] == 2


def test_recovery_images_merge_with_existing_pre_action_overlay(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    source_path = persistence.artifact_store.screenshots_dir / "tool_source.png"
    overlay_path = persistence.artifact_store.screenshots_dir / "tool_action_overlay.png"
    _write_png(source_path, "white")
    _write_png(overlay_path, "red")
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.CLICK_REF,
        tool_name="web_click_ref",
        tool_input={"ref": "999"},
        tool_call_id="call-recovery",
        tool_output={
            "status": "error",
            "code": "stale_ref",
            "retryable": True,
            "raw_image_url": _png_data_url("blue"),
            "som_image_url": _png_data_url("yellow"),
            "observation": {
                "observation_id": "recovery-observation",
                "page": {"url": "https://example.test", "title": "Recovered"},
                "snapshot": {"refs": [{"ref": "1", "role": "button", "label": "Continue"}]},
                "screenshot": {"kind": "set_of_mark", "image_attached": True},
            },
            "observation_recovery": {"status": "recovered"},
        },
        is_error=True,
    )

    artifacts = persistence.build_tool_result_artifacts(
        actions=[action],
        action_screenshot_paths=[],
        artifact_invocations=[
            WebToolInvocationArtifactEvent(
                sequence=1,
                invocation_id=1,
                action=WebAgentActionType.CLICK_REF.value,
            )
        ],
        artifact_events=[
            WebToolImageArtifactEvent(
                sequence=1,
                invocation_id=1,
                action=WebAgentActionType.CLICK_REF.value,
                raw_image_path=str(source_path),
                action_overlay_image_path=str(overlay_path),
                selected_image_path=str(overlay_path),
                selected_image_kind=WebToolArtifactImageKind.ACTION_OVERLAY,
            )
        ],
        session_step=1,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.action_overlay_image_path == str(overlay_path)
    assert artifact.raw_image_path
    assert artifact.raw_image_path != str(source_path)
    assert artifact.som_image_path
    assert Path(artifact.raw_image_path).is_file()
    assert Path(artifact.som_image_path).is_file()
    assert artifact.metadata["observation_recovered"] is True
    assert artifact.metadata["observation_recovery"] == {"status": "recovered"}


def test_tool_result_events_do_not_shift_when_invocation_ledger_is_shorter(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    raw_path = persistence.artifact_store.screenshots_dir / "tool_0001.png"
    som_path = persistence.artifact_store.screenshots_dir / "tool_0001_som.png"
    _write_png(raw_path, "blue")
    _write_png(som_path, "yellow")
    failed_external_validation = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True, "full_page": True},
        tool_call_id="call-shot-invalid",
        tool_output={"status": "error", "action": "screenshot", "code": "validation_error"},
        is_error=True,
    )
    valid_screenshot = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True},
        tool_call_id="call-shot-valid",
        tool_output={"status": "ok", "result": {"image_attached": False}},
    )
    invocation_events = [
        WebToolInvocationArtifactEvent(
            sequence=1,
            invocation_id=1,
            action=WebAgentActionType.SCREENSHOT.value,
        )
    ]
    artifact_events = [
        WebToolImageArtifactEvent(
            sequence=1,
            invocation_id=1,
            action=WebAgentActionType.SCREENSHOT.value,
            raw_image_path=str(raw_path),
            som_image_path=str(som_path),
            selected_image_path=str(som_path),
            selected_image_kind=WebToolArtifactImageKind.SOM,
        )
    ]

    artifacts = persistence.build_tool_result_artifacts(
        actions=[failed_external_validation, valid_screenshot],
        action_screenshot_paths=[],
        artifact_invocations=invocation_events,
        artifact_events=artifact_events,
    )

    assert artifacts[0].selected_image_kind == WebToolArtifactImageKind.NONE
    assert artifacts[1].selected_image_kind == WebToolArtifactImageKind.NONE


def test_tool_image_artifact_context_restores_nested_invocations(tmp_path: Path) -> None:
    registered = _registered_session(tmp_path, client=_ProbeClient(), session_id="artifact-context")
    first_context = registered.begin_tool_artifact_invocation(WebAgentActionType.CLICK_REF.value)
    second_context = registered.begin_tool_artifact_invocation(WebAgentActionType.SCREENSHOT.value)
    registered.record_tool_image_artifact(
        raw_image_path="/tmp/second.png",
        selected_image_kind=WebToolArtifactImageKind.RAW,
    )
    registered.restore_tool_artifact_invocation(second_context)
    registered.record_tool_image_artifact(
        raw_image_path="/tmp/first.png",
        selected_image_kind=WebToolArtifactImageKind.RAW,
    )
    registered.restore_tool_artifact_invocation(first_context)

    assert [event.invocation_id for event in registered.tool_invocation_artifact_events] == [1, 2]
    assert [
        (event.invocation_id, event.action, event.raw_image_path) for event in registered.tool_image_artifact_events
    ] == [
        (2, WebAgentActionType.SCREENSHOT.value, "/tmp/second.png"),
        (1, WebAgentActionType.CLICK_REF.value, "/tmp/first.png"),
    ]


def test_tool_result_artifacts_accept_remote_materialized_screenshot_paths(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    materialized_path = persistence.artifact_store.screenshots_dir / "remote_materialized.png"
    _write_png(materialized_path, "green")
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": False},
        tool_call_id="call-remote-shot",
        tool_output={"status": "ok", "screenshot_path": str(materialized_path), "text": "Captured."},
    )

    artifacts = persistence.build_tool_result_artifacts(actions=[action], action_screenshot_paths=[])

    assert artifacts[0].selected_image_kind == WebToolArtifactImageKind.RAW
    assert artifacts[0].raw_image_path == str(materialized_path)
    assert artifacts[0].selected_image_path == str(materialized_path)
    assert artifacts[0].metadata["attribution_source"] == WebToolArtifactAttributionSource.EXPLICIT_PATH.value


def test_tool_result_artifacts_materialize_remote_inline_som_without_action_path(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    annotation = {
        "kind": "set_of_mark",
        "marks": [{"ref": "19", "label": "19", "bbox": {"x": 8, "y": 10, "width": 40, "height": 20}}],
        "marked_count": 1,
        "visible_interactive_count": 1,
        "omitted_count": 0,
    }
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.CLICK_REF,
        tool_name="web_click_ref",
        tool_input={"ref": "19"},
        tool_call_id="call-remote-click",
        tool_output={
            "status": "ok",
            "text": "Clicked 19.",
            "screenshot_url": _png_data_url("green"),
            "observation": {
                "page": {"url": "https://example.test/after", "title": "After"},
                "screenshot": {"kind": "set_of_mark", "annotation": annotation},
            },
        },
    )

    artifacts = persistence.build_tool_result_artifacts(
        actions=[action],
        action_screenshot_paths=[],
        session_step=4,
    )

    selected_path = Path(artifacts[0].selected_image_path)
    assert artifacts[0].selected_image_kind == WebToolArtifactImageKind.SOM
    assert artifacts[0].som_image_path == str(selected_path)
    assert artifacts[0].raw_image_path == ""
    assert artifacts[0].metadata["attribution_source"] == WebToolArtifactAttributionSource.IMAGE_HASH.value
    assert selected_path.is_file()
    assert selected_path.name.startswith("remote_tool_004_001_web_click_ref_screenshot_url_som")
    assert selected_path.with_suffix(".annotation.json").is_file()
    assert "screenshot_path" not in action.tool_output


def test_tool_result_artifacts_materialize_remote_inline_clean_and_som_roles(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.CLICK_REF,
        tool_name="web_click_ref",
        tool_input={"ref": "19"},
        tool_call_id="call-remote-click-roles",
        tool_output={
            "status": "ok",
            "text": "Clicked 19.",
            "raw_image_url": _png_data_url("blue"),
            "som_image_url": _png_data_url("green"),
        },
    )

    artifacts = persistence.build_tool_result_artifacts(
        actions=[action],
        action_screenshot_paths=[],
        session_step=4,
    )

    artifact = artifacts[0]
    assert artifact.selected_image_kind == WebToolArtifactImageKind.RAW
    assert artifact.raw_image_path
    assert artifact.som_image_path
    assert artifact.selected_image_path == artifact.raw_image_path
    assert Path(artifact.raw_image_path).name.startswith("remote_tool_004_001_web_click_ref_raw_image_url")
    assert Path(artifact.som_image_path).name.startswith("remote_tool_004_002_web_click_ref_som_image_url_som")
    assert artifact.metadata["attribution_source"] == WebToolArtifactAttributionSource.IMAGE_HASH.value


def test_remote_inline_clean_with_som_annotation_is_not_classified_as_som(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    annotation = {
        "kind": "set_of_mark",
        "marked_count": 1,
        "marks": [{"ref": "1", "label": "1", "role": "button", "name": "Submit"}],
    }
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.CLICK_REF,
        tool_name="web_click_ref",
        tool_input={"ref": "1"},
        tool_call_id="call-remote-click-annotated-roles",
        tool_output={
            "status": "ok",
            "text": "Clicked 1.",
            "raw_image_url": _png_data_url("blue"),
            "som_image_url": _png_data_url("green"),
            "observation": {"screenshot": {"annotation": annotation}},
        },
    )

    artifacts = persistence.build_tool_result_artifacts(
        actions=[action],
        action_screenshot_paths=[],
        session_step=4,
    )

    artifact = artifacts[0]
    raw_path = Path(artifact.raw_image_path)
    som_path = Path(artifact.som_image_path)
    assert artifact.selected_image_kind == WebToolArtifactImageKind.RAW
    assert artifact.selected_image_path == artifact.raw_image_path
    assert raw_path.name.startswith("remote_tool_004_001_web_click_ref_raw_image_url")
    assert som_path.name.startswith("remote_tool_004_002_web_click_ref_som_image_url_som")
    assert not raw_path.with_suffix(".annotation.json").exists()
    assert som_path.with_suffix(".annotation.json").is_file()


def test_tool_result_artifacts_materialize_remote_inline_action_overlay(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.CLICK_REF,
        tool_name="web_click_ref",
        tool_input={"ref": "19"},
        tool_call_id="call-remote-click-overlay",
        tool_output={
            "status": "ok",
            "text": "Clicked 19.",
            "action_overlay_image_url": _png_data_url("red"),
        },
    )

    artifacts = persistence.build_tool_result_artifacts(
        actions=[action],
        action_screenshot_paths=[],
        session_step=4,
    )

    selected_path = Path(artifacts[0].selected_image_path)
    assert artifacts[0].selected_image_kind == WebToolArtifactImageKind.ACTION_OVERLAY
    assert artifacts[0].action_overlay_image_path == str(selected_path)
    assert artifacts[0].raw_image_path == ""
    assert artifacts[0].som_image_path == ""
    assert artifacts[0].metadata["attribution_source"] == WebToolArtifactAttributionSource.IMAGE_HASH.value
    assert selected_path.is_file()
    assert selected_path.name.startswith("remote_tool_004_001_web_click_ref_action_overlay_image_url_action_overlay")


def test_terminal_verification_replaces_final_current_and_preserves_previous_paths(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    observation = _observation("Before", "https://example.test/before", "button @e1 Before", "white")
    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=1, observation=observation)
    )
    observation_artifact = persistence.build_observation_artifact(observation=observation, paths=paths)
    step_record = _terminal_step_record(
        persistence,
        observation=observation,
        paths=paths,
    )
    previous_screenshot = step_record.screenshot_path
    previous_dom = step_record.dom_snapshot_path
    terminal_observation = _observation("After", "https://example.test/after", "main @e2 After", "yellow")

    record = _run(
        persistence.attach_terminal_verification(
            client=_TerminalClient(),
            step_record=step_record,
            observation=terminal_observation,
            reason="done",
        )
    )

    assert record.status == WebTerminalVerificationStatus.CAPTURED
    assert record.previous_screenshot_path == previous_screenshot
    assert record.previous_dom_snapshot_path == previous_dom
    assert step_record.screenshot_path.endswith("step_001_terminal_raw.png")
    assert step_record.dom_snapshot_path.endswith("step_001_terminal.txt")
    assert step_record.observation_artifact is not None
    assert step_record.observation_artifact.role == WebObservationArtifactRole.TERMINAL_VERIFICATION
    assert step_record.observation_metadata["web_terminal_previous_screenshot_path"] == previous_screenshot
    assert step_record.observation_metadata["web_current_observation_role"] == "round_end_verify"
    assert step_record.observation_metadata["web_current_screenshot_path"] == step_record.screenshot_path
    assert step_record.observation_metadata["web_current_dom_snapshot_path"] == step_record.dom_snapshot_path
    assert step_record.observation_metadata["web_current_dom_snapshot_json_path"] == step_record.dom_snapshot_json_path
    assert step_record.observation_metadata["web_current_page_title"] == "After"


def test_terminal_verification_failure_is_persisted_without_extra_artifacts(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    observation = _observation("Before", "https://example.test/before", "button @e1 Before", "white")
    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=1, observation=observation)
    )
    step_record = _terminal_step_record(
        persistence,
        observation=observation,
        paths=paths,
    )

    record = persistence.mark_terminal_verification_failed(
        step_record=step_record,
        reason="runtime_error",
        error=RuntimeError("snapshot failed"),
    )
    persistence.replace_step_trace([step_record])

    assert record.status == WebTerminalVerificationStatus.FAILED
    assert step_record.screenshot_path.endswith("step_001_raw.png")
    trace = json.loads(persistence.artifact_store.trace_path.read_text(encoding="utf-8").strip())
    assert trace["terminal_verification"]["status"] == "failed"


def test_terminal_verification_missing_screenshot_is_failed_not_captured(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    observation = _observation("Before", "https://example.test/before", "button @e1 Before", "white")
    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=1, observation=observation)
    )
    step_record = _terminal_step_record(
        persistence,
        observation=observation,
        paths=paths,
    )
    previous_screenshot = step_record.screenshot_path

    record = _run(
        persistence.attach_terminal_verification(
            client=_NoScreenshotClient(),
            step_record=step_record,
            observation=WebObservationContext(
                snapshot=_observation("After", "https://example.test/after", "main @e2 After", "yellow").snapshot
            ),
            reason="done",
        )
    )

    assert record.status == WebTerminalVerificationStatus.FAILED
    assert step_record.screenshot_path == previous_screenshot
    assert not persistence.artifact_store.terminal_dom_snapshot_path(1).exists()


def test_web_bundle_adapter_emits_dom_tool_and_terminal_evidence(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    observation = _observation("Before", "https://example.test/before", "button @e1 Before", "white")
    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=1, observation=observation)
    )
    step_record = _terminal_step_record(
        persistence,
        observation=observation,
        paths=paths,
    )
    _run(
        persistence.attach_terminal_verification(
            client=_TerminalClient(),
            step_record=step_record,
            observation=_observation("After", "https://example.test/after", "main @e2 After", "yellow"),
            reason="done",
        )
    )
    result = WebAgentResult(
        status=WebRunStatus.SUCCESS,
        terminal_status=WebTerminalStatus.COMPLETED,
        session_id="session",
        user_query="finish the page",
        success=True,
        steps=[step_record],
    )
    bundle = WebRoundEvaluationBundleAdapter(result=result).build_bundle()

    assert any(ref.kind == EvaluationEvidenceKind.DOM for ref in bundle.dom_refs)
    assert any(ref.metadata.get("variant") == "terminal_verification" for ref in bundle.dom_refs)
    assert any(ref.metadata.get("variant") == "initial_state" for ref in bundle.dom_refs)
    assert not any(ref.metadata.get("variant") == "pre_terminal_current" for ref in bundle.dom_refs)
    terminal_ref = next(ref for ref in bundle.screenshot_refs if ref.metadata.get("variant") == "terminal_verification")
    assert terminal_ref.metadata["pinned"] is True
    assert terminal_ref.metadata["pinned_reason"] == "web_terminal_verification"
    assert terminal_ref.metadata["initial_state"] is False


def test_bundle_adapter_emits_raw_and_som_tool_image_refs(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    observation = _observation("Before", "https://example.test/before", "button @e1 Before", "white")
    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=1, observation=observation)
    )
    raw_path = persistence.artifact_store.screenshots_dir / "tool_0001.png"
    som_path = persistence.artifact_store.screenshots_dir / "tool_0001_som.png"
    _write_png(raw_path, "blue")
    _write_png(som_path, "green")
    _write_som_sidecar(som_path, source=raw_path)
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": True},
        tool_output={
            "status": "ok",
            "text": "Captured.",
            "screenshot_url": _png_data_url("green"),
            "result": {"kind": "set_of_mark", "image_attached": True},
        },
        tool_call_id="call-shot",
    )
    tool_artifacts = persistence.build_tool_result_artifacts(
        actions=[action],
        action_screenshot_paths=[raw_path, som_path],
    )
    step_record = WebAgentStepRecordProjector(action_resolver=WebAgentActionResolver()).build_step_record(
        session_id="test-session",
        step_index=1,
        round_num=1,
        session_step=1,
        runtime_cursor={},
        observation=observation,
        step_result=_step_result(action),
        resolved_actions=[action],
        artifact_paths=paths,
        observation_artifact=persistence.build_observation_artifact(observation=observation, paths=paths),
        tool_artifacts=tool_artifacts,
    )
    result = WebAgentResult(
        status=WebRunStatus.SUCCESS,
        terminal_status=WebTerminalStatus.COMPLETED,
        session_id="session",
        user_query="capture annotated screenshot",
        success=True,
        steps=[step_record],
    )

    bundle = WebRoundEvaluationBundleAdapter(result=result).build_bundle()

    tool_result = bundle.tool_results[0]
    assert tool_result["raw_image_ref"]
    assert tool_result["som_image_ref"]
    assert tool_result["image_ref"] == tool_result["raw_image_ref"]
    assert tool_result["default_image_ref_policy"] == "clean_then_action_overlay_then_som"
    ref_paths = {ref.id: ref.path for ref in bundle.screenshot_refs}
    assert ref_paths[tool_result["raw_image_ref"]] == str(raw_path)
    assert ref_paths[tool_result["som_image_ref"]] == str(som_path)


def test_bundle_adapter_pins_error_tool_image_refs_for_final_report_validation(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    observation = _observation("Before", "https://example.test/before", "button @e1 Before", "white")
    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=1, observation=observation)
    )
    raw_path = persistence.artifact_store.screenshots_dir / "tool_error.png"
    som_path = persistence.artifact_store.screenshots_dir / "tool_error_som.png"
    _write_png(raw_path, "red")
    _write_png(som_path, "orange")
    _write_som_sidecar(som_path, source=raw_path)
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.CLICK_REF,
        tool_name="web_click_ref",
        tool_input={"ref": 1},
        tool_output={"status": "error", "text": "stale ref", "error": "stale ref"},
        tool_call_id="call-click",
        is_error=True,
    )
    tool_artifacts = persistence.build_tool_result_artifacts(
        actions=[action],
        action_screenshot_paths=[],
        artifact_invocations=[
            WebToolInvocationArtifactEvent(
                sequence=1,
                invocation_id=1,
                action=WebAgentActionType.CLICK_REF.value,
            )
        ],
        artifact_events=[
            WebToolImageArtifactEvent(
                sequence=1,
                invocation_id=1,
                action=WebAgentActionType.CLICK_REF.value,
                raw_image_path=str(raw_path),
                selected_image_path=str(raw_path),
                selected_image_kind=WebToolArtifactImageKind.RAW,
            ),
            WebToolImageArtifactEvent(
                sequence=2,
                invocation_id=1,
                action=WebAgentActionType.CLICK_REF.value,
                raw_image_path=str(raw_path),
                som_image_path=str(som_path),
                selected_image_path=str(som_path),
                selected_image_kind=WebToolArtifactImageKind.SOM,
            ),
        ],
    )
    step_record = WebAgentStepRecordProjector(action_resolver=WebAgentActionResolver()).build_step_record(
        session_id="test-session",
        step_index=1,
        round_num=1,
        session_step=1,
        runtime_cursor={},
        observation=observation,
        step_result=_step_result(action),
        resolved_actions=[action],
        artifact_paths=paths,
        observation_artifact=persistence.build_observation_artifact(observation=observation, paths=paths),
        tool_artifacts=tool_artifacts,
    )
    result = WebAgentResult(
        status=WebRunStatus.FAILED,
        terminal_status=WebTerminalStatus.TOOL_ERROR,
        session_id="session",
        user_query="click the target",
        success=False,
        steps=[step_record],
    )

    bundle = WebRoundEvaluationBundleAdapter(result=result).build_bundle()
    tool_result = bundle.tool_results[0]
    manifest = EvidenceSelector(build_web_round_evaluator_spec().evidence_policy).select(bundle=bundle)

    assert tool_result["som_image_ref"]
    error_image_ref = next(ref for ref in bundle.screenshot_refs if ref.id == tool_result["som_image_ref"])
    assert error_image_ref.metadata["pinned"] is True
    assert error_image_ref.metadata["pinned_reason"] == "web_tool_error_evidence"
    assert error_image_ref.metadata["tool_is_error"] is True
    assert tool_result["som_image_ref"] in manifest.ref_ids


def test_bundle_adapter_exposes_final_answer_as_trace_evidence(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    step_record = _step_record(
        persistence,
        round_num=1,
        round_step=1,
        session_step=1,
        title="Done",
    )
    final_answer = "| Item | Value |\n|---|---|\n| Macroblade 110 3WD | 110 mm, 3 wheels |"
    result = WebAgentResult(
        status=WebRunStatus.SUCCESS,
        terminal_status=WebTerminalStatus.COMPLETED,
        session_id="session",
        user_query="compare wheel sizes",
        success=True,
        final_answer=final_answer,
        steps=[step_record],
    )

    bundle = WebRoundEvaluationBundleAdapter(result=result).build_bundle()
    final_refs = [
        ref
        for ref in bundle.runtime_event_refs
        if ref.kind == EvaluationEvidenceKind.TRACE and ref.metadata.get("event") == "final_answer"
    ]

    assert len(final_refs) == 1
    assert final_refs[0].label == "Final answer"
    assert final_refs[0].summary == final_answer
    assert final_refs[0].selection_reason == "web_final_answer"
    assert final_refs[0].pinned is True

    spec = build_web_round_evaluator_spec()
    evidence_manifest = EvidenceSelector(spec.evidence_policy).select(
        bundle=bundle,
    )
    prompt_messages = WebEvaluatorPromptBuilder().build_messages(
        bundle=bundle,
        evidence_manifest=evidence_manifest,
        artifact_root=persistence.artifact_store.root,
    )
    prompt_text = "\n".join(block.text for block in prompt_messages[1].content if isinstance(block, TextBlock))
    assert "Final answer" in prompt_text
    assert "Macroblade 110 3WD | 110 mm, 3 wheels" in prompt_text


def test_web_evaluator_prompt_uses_compact_path_free_evidence_index(tmp_path: Path) -> None:
    screenshot_path = tmp_path / "step.png"
    action_overlay_path = tmp_path / "step_action_overlay.png"
    _write_png(screenshot_path, "blue")
    _write_png(action_overlay_path, "red")
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="inspect page",
        session_request="inspect page",
    )
    manifest = EvaluationEvidenceManifest(
        refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Step 1 SoM image",
                summary="A useful SoM screenshot",
                path=str(screenshot_path),
                session_step=1,
                round_step=1,
                metadata={
                    "variant": "som",
                    "tool_name": "web_click_ref",
                    "selected_image_kind": "som",
                    "som_metadata": {
                        "marks": [
                            {
                                "ref": "1",
                                "label": "1",
                                "bbox_css": {"x": 1, "y": 2, "width": 3, "height": 4},
                            }
                        ],
                        "omitted_count": 7,
                        "truncated": False,
                    },
                    "source_screenshot_path": "/tmp/private/source.png",
                },
            ),
            EvaluationEvidenceRef(
                id="E002",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Step 1 action overlay image",
                summary="Where the click was attempted",
                path=str(action_overlay_path),
                session_step=1,
                round_step=1,
                metadata={"variant": "action_overlay", "tool_name": "web_click_ref"},
            ),
        ]
    )

    messages = WebEvaluatorPromptBuilder().build_messages(
        bundle=bundle,
        evidence_manifest=manifest,
        artifact_root=tmp_path,
    )

    system_text = "\n".join(block.text for block in messages[0].content if isinstance(block, TextBlock))
    prompt_text = "\n".join(block.text for block in messages[1].content if isinstance(block, TextBlock))
    assert "Selected evidence index" in prompt_text
    assert "E001" in prompt_text
    assert "som_mark_count=1" in prompt_text
    assert "som_omitted_count=7" in prompt_text
    assert "variant=action_overlay" in prompt_text
    assert "action-overlay history screenshot" in prompt_text
    assert "not current page state" in prompt_text
    assert "copy the verified terminal/page-state values exactly" in prompt_text
    assert "query parameter" in prompt_text
    assert "copy verified terminal/page-state strings exactly" in system_text
    assert str(screenshot_path) not in prompt_text
    assert "/tmp/private/source.png" not in prompt_text
    assert "som_metadata" not in prompt_text
    assert "bbox_css" not in prompt_text
    assert '"marks"' not in prompt_text


def test_query_dom_snapshot_prefers_terminal_dom_for_session_step(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    observation = _observation("Before", "https://example.test/before", "button @e1 Before", "white")
    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=1, observation=observation)
    )
    step_record = _terminal_step_record(
        persistence,
        observation=observation,
        paths=paths,
    )
    _run(
        persistence.attach_terminal_verification(
            client=_TerminalClient(),
            step_record=step_record,
            observation=_observation("After", "https://example.test/after", "main @e2 After", "yellow"),
            reason="done",
        )
    )
    result = WebAgentResult(
        status=WebRunStatus.SUCCESS,
        terminal_status=WebTerminalStatus.COMPLETED,
        session_id="session",
        user_query="finish the page",
        success=True,
        steps=[step_record],
    )
    bundle = WebRoundEvaluationBundleAdapter(result=result).build_bundle()
    factory = ArtifactInspectionToolFactory(
        artifact_root=persistence.artifact_store.root,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=[*bundle.screenshot_refs, *bundle.dom_refs]),
        step_trace_by_session_step={1: step_record.model_dump(mode="json")},
    )

    by_session_step = factory._query_dom_snapshot(session_step=1)
    previous_ref = next(ref for ref in bundle.dom_refs if ref.metadata.get("variant") == "initial_state")
    by_previous_ref = factory._query_dom_snapshot(evidence_ref=previous_ref.id)

    assert by_session_step["status"] == "success"
    assert by_session_step["evidence_ref"] != previous_ref.id
    assert "After" in by_session_step["text_preview"]
    assert any(candidate["text"] == "After" for candidate in by_session_step["visible_text_candidates"])
    assert by_previous_ref["status"] == "success"
    assert "Before" in by_previous_ref["text_preview"]


def test_bundle_adapter_does_not_emit_terminal_previous_when_initial_current_exists(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    first = _step_record(persistence, round_num=1, round_step=1, session_step=1, title="Initial")
    observation = _observation("Before Done", "https://example.test/before-done", "button @e1 Before Done", "white")
    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=2, observation=observation)
    )
    second = _terminal_step_record(
        persistence,
        observation=observation,
        paths=paths,
        step_index=2,
        session_step=2,
    )
    _run(
        persistence.attach_terminal_verification(
            client=_TerminalClient(),
            step_record=second,
            observation=_observation("After Done", "https://example.test/after-done", "main @e2 After Done", "yellow"),
            reason="done",
        )
    )
    result = WebAgentResult(
        status=WebRunStatus.SUCCESS,
        terminal_status=WebTerminalStatus.COMPLETED,
        session_id="session",
        user_query="finish the page",
        success=True,
        steps=[first, second],
    )

    bundle = WebRoundEvaluationBundleAdapter(result=result).build_bundle()

    assert [ref.metadata.get("variant") for ref in bundle.dom_refs] == ["initial_state", "terminal_verification"]
    assert [ref.metadata.get("variant") for ref in bundle.screenshot_refs] == [
        "initial_state",
        "terminal_verification",
    ]
    assert not any("pre-verification" in ref.label for ref in [*bundle.dom_refs, *bundle.screenshot_refs])


def test_query_web_terminal_verification_defaults_to_current_round(tmp_path: Path) -> None:
    first_trace = {
        "session_step": 1,
        "round_num": 1,
        "terminal_verification": {"status": "captured", "reason": "round-one"},
    }
    second_trace = {
        "session_step": 2,
        "round_num": 2,
        "terminal_verification": {"status": "captured", "reason": "round-two"},
    }
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="first",
        session_request="first",
        step_refs=[
            # Evaluate round 1 while a later round exists in the session trace.
            RoundEvaluationStepRef(
                session_step=1,
                round_step=1,
                status="ok",
                request="first",
                result="done",
            )
        ],
    )
    factory = ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=[]),
        step_trace_by_session_step={1: first_trace, 2: second_trace},
    )

    result = factory._query_web_terminal_verification()

    assert result["status"] == "success"
    assert result["terminal_verification"]["reason"] == "round-one"


def test_replace_step_trace_preserves_prior_round_rows_and_session_step_artifacts(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    first = _step_record(persistence, round_num=1, round_step=1, session_step=1, title="First")
    second = _step_record(persistence, round_num=2, round_step=1, session_step=2, title="Second")
    persistence.persist_step(first)
    persistence.persist_step(second)

    assert first.screenshot_path.endswith("step_001_raw.png")
    assert second.screenshot_path.endswith("step_002_raw.png")
    second.assistant_text = "updated"
    persistence.replace_step_trace([second])

    rows = [
        json.loads(line)
        for line in persistence.artifact_store.trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["session_step"] for row in rows] == [1, 2]
    assert rows[0]["assistant_text"] != "updated"
    assert rows[1]["assistant_text"] == "updated"


def test_trace_qa_accepts_resumed_positive_session_steps(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    root.mkdir()
    (root / "result.json").write_text(json.dumps({"status": "success", "screenshots": []}), encoding="utf-8")
    rows = [
        _llm_trace_wrapper_row(session_step=3, round_num=2, round_step=1, request="resume one"),
        _llm_trace_wrapper_row(session_step=4, round_num=2, round_step=2, request="resume two"),
    ]
    (root / "llm_trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    (root / "web_agent_trace.jsonl").write_text("", encoding="utf-8")
    screenshots = root / "screenshots"
    screenshots.mkdir()
    (screenshots / "tool_som.annotation.json").write_text(
        json.dumps(
            {
                "marks": [
                    {
                        "ref": "@e1",
                        "bbox_css": {"x": 1, "y": 1, "width": 2, "height": 2},
                        "bbox_image": {"x": 1, "y": 1, "width": 2, "height": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_qa(root)

    assert report["status"] == "pass"
    assert not [
        finding for finding in report["findings"] if finding["title"] == "llm_trace wrapper fields are inconsistent"
    ]


def test_trace_qa_flags_model_visible_som_refs_mark_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    screenshots = root / "screenshots"
    screenshots.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"status": "success", "screenshots": []}), encoding="utf-8")
    (root / "llm_trace.jsonl").write_text("", encoding="utf-8")
    (screenshots / "tool_som.annotation.json").write_text(
        json.dumps(
            {
                "marks": [
                    {
                        "ref": "@e1",
                        "label": "@e1",
                        "bbox_css": {"x": 1, "y": 1, "width": 4, "height": 4},
                        "bbox_image": {"x": 1, "y": 1, "width": 4, "height": 4},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    trace_row = {
        "tools": [
            {
                "tool_name": "web_click_ref",
                "tool_output": {
                    "status": "success",
                    "observation": {
                        "page": {"url": "https://example.test"},
                        "snapshot": {
                            "refs": [
                                {"ref": "@e1", "label": "First"},
                                {"ref": "@e2", "label": "Second"},
                            ],
                        },
                        "screenshot": {
                            "image_attached": True,
                            "annotation": {
                                "kind": "set_of_mark",
                                "marks": [{"ref": "@e1", "label": "@e1"}],
                            },
                        },
                    },
                },
            }
        ]
    }
    (root / "web_agent_trace.jsonl").write_text(json.dumps(trace_row) + "\n", encoding="utf-8")

    report = run_qa(root)

    assert report["status"] == "fail"
    assert report["web_trace"]["som_ref_alignment_issues"]
    assert any(
        finding["title"] == "model-visible SoM refs do not match annotation marks" for finding in report["findings"]
    )


def test_trace_qa_rejects_undeclared_prompt_history_function_call(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    screenshots = root / "screenshots"
    screenshots.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"status": "success", "screenshots": []}), encoding="utf-8")
    request = {
        "input": [
            {
                "type": "function_call",
                "call_id": "call-observation",
                "name": "web_current_observation",
                "arguments": "{}",
            }
        ],
        "tools": [{"type": "function", "name": "web_snapshot"}],
    }
    (root / "llm_trace.jsonl").write_text(
        json.dumps(_llm_trace_wrapper_row(session_step=1, request="inspect", llm_request=request)),
        encoding="utf-8",
    )
    (root / "web_agent_trace.jsonl").write_text("", encoding="utf-8")
    _write_png(screenshots / "tool_som.png", "green")
    (screenshots / "tool_som.annotation.json").write_text(
        json.dumps(
            {
                "marks": [
                    {
                        "ref": "@e1",
                        "bbox_css": {"x": 1, "y": 1, "width": 2, "height": 2},
                        "bbox_image": {"x": 1, "y": 1, "width": 2, "height": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_qa(root)

    assert report["status"] == "fail"
    assert report["llm_trace"]["undeclared_tool_calls"] == ["row=1:input:web_current_observation"]
    assert any(
        finding["title"] == "prompt history contains undeclared function calls" for finding in report["findings"]
    )


def test_trace_qa_accepts_declared_prompt_history_function_call(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    screenshots = root / "screenshots"
    screenshots.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"status": "success", "screenshots": []}), encoding="utf-8")
    request = {
        "input": [
            {
                "type": "function_call",
                "call_id": "call-snapshot",
                "name": "web_snapshot",
                "arguments": "{}",
            }
        ],
        "tools": [{"type": "function", "name": "web_snapshot"}],
    }
    (root / "llm_trace.jsonl").write_text(
        json.dumps(_llm_trace_wrapper_row(session_step=1, request="inspect", llm_request=request)),
        encoding="utf-8",
    )
    (root / "web_agent_trace.jsonl").write_text("", encoding="utf-8")
    _write_png(screenshots / "tool_som.png", "green")
    (screenshots / "tool_som.annotation.json").write_text(
        json.dumps(
            {
                "marks": [
                    {
                        "ref": "@e1",
                        "bbox_css": {"x": 1, "y": 1, "width": 2, "height": 2},
                        "bbox_image": {"x": 1, "y": 1, "width": 2, "height": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_qa(root)

    assert report["status"] == "pass"
    assert report["llm_trace"]["undeclared_tool_calls"] == []


def test_trace_qa_rejects_chat_style_undeclared_function_call(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    screenshots = root / "screenshots"
    screenshots.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"status": "success", "screenshots": []}), encoding="utf-8")
    request = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-observation",
                        "type": "function",
                        "function": {"name": "web_current_observation", "arguments": "{}"},
                    }
                ],
            }
        ],
        "tools": [{"type": "function", "function": {"name": "web_snapshot", "parameters": {}}}],
    }
    (root / "llm_trace.jsonl").write_text(
        json.dumps(_llm_trace_wrapper_row(session_step=1, request="inspect", llm_request=request)),
        encoding="utf-8",
    )
    (root / "web_agent_trace.jsonl").write_text("", encoding="utf-8")
    _write_png(screenshots / "tool_som.png", "green")
    (screenshots / "tool_som.annotation.json").write_text(
        json.dumps(
            {
                "marks": [
                    {
                        "ref": "@e1",
                        "bbox_css": {"x": 1, "y": 1, "width": 2, "height": 2},
                        "bbox_image": {"x": 1, "y": 1, "width": 2, "height": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_qa(root)

    assert report["status"] == "fail"
    assert report["llm_trace"]["undeclared_tool_calls"] == ["row=1:messages:web_current_observation"]


def test_trace_qa_flags_som_boxes_outside_screenshot_bounds(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    screenshots = root / "screenshots"
    screenshots.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"status": "success", "screenshots": []}), encoding="utf-8")
    (root / "llm_trace.jsonl").write_text("", encoding="utf-8")
    (root / "web_agent_trace.jsonl").write_text("", encoding="utf-8")
    _write_png(screenshots / "tool_som.png", "green")
    (screenshots / "tool_som.annotation.json").write_text(
        json.dumps(
            {
                "marks": [
                    {
                        "ref": "@e1",
                        "bbox_css": {"x": 1, "y": 1, "width": 2, "height": 2},
                        "bbox_image": {"x": 6, "y": 6, "width": 4, "height": 4},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_qa(root)

    assert report["status"] == "fail"
    assert any(finding["title"] == "SoM annotation contains invalid mark geometry" for finding in report["findings"])


def test_trace_qa_detects_mixed_case_runtime_data_urls(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    screenshots = root / "screenshots"
    screenshots.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"status": "success", "screenshots": []}), encoding="utf-8")
    (root / "llm_trace.jsonl").write_text(
        json.dumps(
            _llm_trace_wrapper_row(
                session_step=1,
                request="inspect",
                llm_request={"image_url": "data:IMAGE/PNG;base64,QUJD", "tools": []},
            )
        ),
        encoding="utf-8",
    )
    (root / "web_agent_trace.jsonl").write_text(
        json.dumps(
            {
                "tool_summary": "web_screenshot",
                "tools": [
                    {
                        "tool_name": "web_screenshot",
                        "tool_output": {"text": "prefix data:Image/Png;base64,QUJD suffix"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_png(screenshots / "tool_som.png", "green")
    (screenshots / "tool_som.annotation.json").write_text(
        json.dumps(
            {
                "marks": [
                    {
                        "ref": "@e1",
                        "bbox_css": {"x": 1, "y": 1, "width": 2, "height": 2},
                        "bbox_image": {"x": 1, "y": 1, "width": 2, "height": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_qa(root)

    assert report["status"] == "fail"
    titles = {finding["title"] for finding in report["findings"]}
    assert "llm_trace persisted raw data URLs" not in titles
    assert "model request contains local artifact path" not in titles
    assert "model-visible tool output contains path or raw data URL" in titles


def test_trace_qa_rejects_incomplete_llm_wrapper_and_memory_images(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    screenshots = root / "screenshots"
    screenshots.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"status": "success", "screenshots": []}), encoding="utf-8")
    (root / "llm_trace.jsonl").write_text(
        json.dumps(
            {
                "source": "web",
                "session_step": 1,
                "llm_trace": {
                    "request": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "input_image", "image_url": "memory://initial"}],
                            }
                        ],
                        "tools": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "web_agent_trace.jsonl").write_text("", encoding="utf-8")
    _write_png(screenshots / "tool_som.png", "green")
    (screenshots / "tool_som.annotation.json").write_text(
        json.dumps(
            {
                "marks": [
                    {
                        "ref": "@e1",
                        "bbox_css": {"x": 1, "y": 1, "width": 2, "height": 2},
                        "bbox_image": {"x": 1, "y": 1, "width": 2, "height": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_qa(root)

    assert report["status"] == "fail"
    titles = {finding["title"] for finding in report["findings"]}
    assert "llm_trace wrapper fields are inconsistent" in titles
    assert "llm_trace image blocks are not raw data URLs" in titles


def test_trace_qa_rejects_incomplete_tool_artifact_attribution(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    screenshots = root / "screenshots"
    screenshots.mkdir(parents=True)
    tool_som = screenshots / "tool_001_som.png"
    _write_png(tool_som, "green")
    (screenshots / "tool_001_som.annotation.json").write_text(
        json.dumps(
            {
                "marks": [
                    {
                        "ref": "@e1",
                        "bbox_css": {"x": 1, "y": 1, "width": 2, "height": 2},
                        "bbox_image": {"x": 1, "y": 1, "width": 2, "height": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "result.json").write_text(
        json.dumps({"status": "success", "screenshots": [str(tool_som)]}),
        encoding="utf-8",
    )
    (root / "llm_trace.jsonl").write_text(
        json.dumps(
            _llm_trace_wrapper_row(
                session_step=1,
                request="click",
                llm_request={"messages": [], "tools": [{"name": "web_click_ref"}]},
            )
        ),
        encoding="utf-8",
    )
    (root / "web_agent_trace.jsonl").write_text(
        json.dumps(
            {
                "session_step": 1,
                "step_index": 1,
                "status": "success",
                "screenshot_path": str(screenshots / "step_001_raw.png"),
                "tools": [
                    {
                        "tool_name": "web_click_ref",
                        "tool_output": {
                            "status": "ok",
                            "screenshot_url": "memory://tool-result",
                            "observation": {"page": {"url": "https://example.test", "title": "Done"}},
                        },
                    }
                ],
                "tool_artifacts": [{"selected_image_path": str(tool_som)}],
            }
        ),
        encoding="utf-8",
    )

    report = run_qa(root)

    assert report["status"] == "fail"
    assert any(
        finding["title"] == "post-action tool artifact attribution is incomplete" for finding in report["findings"]
    )


def test_trace_qa_rejects_removed_top_level_web_trace_fields(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    root.mkdir()
    (root / "result.json").write_text(json.dumps({"status": "success", "screenshots": []}), encoding="utf-8")
    (root / "llm_trace.jsonl").write_text("", encoding="utf-8")
    (root / "web_agent_trace.jsonl").write_text(
        json.dumps(
            {
                "session_step": 1,
                "action": "web_click_ref",
                "tool_input": {"ref": "@e1"},
                "tool_output": {"status": "ok"},
                "tools": [],
            }
        ),
        encoding="utf-8",
    )

    report = run_qa(root)

    assert report["status"] == "fail"
    assert report["web_trace"]["removed_top_level_fields"] == ["row 1: action, tool_input, tool_output"]
    assert any(finding["title"] == "removed top-level web trace fields present" for finding in report["findings"])


def test_web_evaluator_contracts_use_web_report_and_wrapped_browser_tools() -> None:
    report = WebRoundEvaluationReport.model_validate(
        {
            "overall_verdict": "success",
            "process_rating": 5,
            "outcome_label": "success",
            "completion_rating": 5,
            "confidence_rating": 4,
            "controllability": "controllable",
            "reason": "Terminal DOM and screenshot prove completion.",
            "criteria": [],
            "diagnoses": [],
            "side_effects": [],
            "safety": {"risk_level": "none", "findings": []},
            "efficiency": {"rating": 5, "reason": "Direct path."},
            "final_url": "https://example.test/done",
            "final_title": "Done",
            "page_state_verified": True,
            "dom_state_verified": True,
            "visual_state_verified": True,
            "critical_web_evidence_refs": ["E001"],
        }
    )
    assert WebRoundEvaluationReport.model_validate_json(report.model_dump_json()).final_url.endswith("/done")

    artifact_only = build_web_round_evaluator_spec()
    assert artifact_only.max_steps == 12
    assert artifact_only.force_finalizer_after_steps == 8
    assert not any(tool.name.startswith("browser_") for tool in artifact_only.tools)

    live = build_web_round_evaluator_spec(enable_live_probe=True)
    names = {tool.name for tool in live.tools}
    assert "client_browser_snapshot" in names
    assert "client_browser_tabs" in names
    assert "client_browser_get_url" in names
    assert "client_browser_screenshot" in names
    assert "client_browser_click_ref" not in names
    assert "client_browser_open" not in names
    assert "browser_snapshot" not in names
    assert "web_screenshot" not in names
    assert all(tool.placement != EvaluationToolPlacement.CLIENT_MUTATING for tool in live.tools)
    for wrapper_name, dispatch_name in {
        "client_browser_snapshot": "web.web_snapshot",
        "client_browser_tabs": "web.web_list_tabs",
        "client_browser_get_url": "web.web_get_url",
        "client_browser_screenshot": "web.web_screenshot",
    }.items():
        tool = {tool.name: tool for tool in live.tools}[wrapper_name]
        assert tool.server_name == "WebAgentActionExecutor"
        assert tool.dispatch_name == dispatch_name

    mutating = build_web_round_evaluator_spec(enable_live_probe=True, allow_client_mutating_tools=True)
    mutating_by_name = {tool.name: tool for tool in mutating.tools}
    assert mutating_by_name["client_browser_snapshot"].placement == EvaluationToolPlacement.CLIENT_PROBE
    assert mutating_by_name["client_browser_open"].placement == EvaluationToolPlacement.CLIENT_MUTATING
    assert mutating_by_name["client_browser_click_ref"].placement == EvaluationToolPlacement.CLIENT_MUTATING
    assert mutating_by_name["client_browser_wait"].placement == EvaluationToolPlacement.CLIENT_MUTATING
    assert "client_browser_note" not in mutating_by_name
    assert "web_open" not in mutating_by_name
    assert mutating_by_name["client_browser_open"].dispatch_name == "web.web_open"
    assert mutating_by_name["client_browser_click_ref"].dispatch_name == "web.web_click_ref"
    assert mutating_by_name["client_browser_wait"].dispatch_name == "web.web_wait"
    assert EvaluationToolPlacement.CLIENT_MUTATING in mutating.placement_policy.allowed_placements
    assert mutating.effective_schedule() == EvaluationSchedule.AFTER_ROUND_SYNC


def test_web_final_report_copies_verified_terminal_page_state(tmp_path: Path) -> None:
    writer = EvaluatorArtifactWriter(
        EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="web_round_evaluator")
    )
    stale_report = _web_report_payload()
    stale_report["final_url"] = (
        "http://127.0.0.1:40001/search?q=Xbox+Wireless+Controller+green&sort=&min_rating=4"
        "&min_price=&max_price=&brand=&color=&size=&features="
    )
    stale_report["final_title"] = "Amazon Search"
    writer.write_final_report(round_num=1, payload=stale_report)

    normalized = normalize_web_final_report_page_state(
        writer=writer,
        round_num=1,
        step_trace_by_session_step={
            4: {
                "terminal_verification": {
                    "status": "captured",
                    "page_url": (
                        "http://127.0.0.1:40001/search?q=Xbox+Wireless+Controller+green&sort=&min_rating=4"
                        "&min_price=&max_price=&brand=&color=&size=&feature="
                    ),
                    "page_title": "Xbox Wireless Controller green - Amazon.com Search",
                }
            }
        },
    )

    report_path = writer.layout.round_paths(1).final_report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime_events = writer.layout.round_paths(1).runtime_log_path.read_text(encoding="utf-8")
    assert normalized is True
    assert report["final_url"].endswith("&feature=")
    assert "features=" not in report["final_url"]
    assert report["final_title"] == "Xbox Wireless Controller green - Amazon.com Search"
    assert "web_final_report_page_state_normalized" in runtime_events


def test_web_final_report_page_state_normalizer_stays_in_current_round(tmp_path: Path) -> None:
    writer = EvaluatorArtifactWriter(
        EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="web_round_evaluator")
    )
    stale_report = _web_report_payload()
    stale_report["final_url"] = "http://127.0.0.1:40001/round-one-stale"
    stale_report["final_title"] = "Round one stale"
    writer.write_final_report(round_num=1, payload=stale_report)

    normalized = normalize_web_final_report_page_state(
        writer=writer,
        round_num=1,
        step_trace_by_session_step={
            1: {
                "round_num": 1,
                "terminal_verification": {
                    "status": "captured",
                    "page_url": "http://127.0.0.1:40001/round-one",
                    "page_title": "Round one",
                },
            },
            2: {
                "round_num": 2,
                "terminal_verification": {
                    "status": "captured",
                    "page_url": "http://127.0.0.1:40001/round-two",
                    "page_title": "Round two",
                },
            },
        },
    )

    report = json.loads(writer.layout.round_paths(1).final_report_path.read_text(encoding="utf-8"))
    assert normalized is True
    assert report["final_url"] == "http://127.0.0.1:40001/round-one"
    assert report["final_title"] == "Round one"


def test_web_evaluator_prompt_attaches_image_blocks_without_manifest_paths(tmp_path: Path) -> None:
    screenshot_path = tmp_path / "screenshots" / "step_001_raw.png"
    _write_png(screenshot_path, "white")
    raw_manifest_path = str(Path("logs") / "old-artifact-run" / tmp_path.name / "screenshots" / "step_001_raw.png")
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="inspect page",
        session_request="inspect page",
    )
    manifest = EvaluationEvidenceManifest(
        refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Step 1 initial screenshot",
                path=raw_manifest_path,
                session_step=1,
                round_step=1,
                metadata={"variant": "initial_state", "source": "web"},
            )
        ]
    )

    messages = WebEvaluatorPromptBuilder().build_messages(
        bundle=bundle,
        evidence_manifest=manifest,
        artifact_root=tmp_path,
    )

    user_blocks = messages[1].content
    prompt_text = "\n".join(block.text for block in user_blocks if isinstance(block, TextBlock))
    image_blocks = [block for block in user_blocks if isinstance(block, ImageBlock)]
    assert image_blocks
    assert image_blocks[0].url.startswith("data:image/jpeg;base64,")
    assert str(screenshot_path) not in prompt_text
    assert raw_manifest_path not in prompt_text
    assert "Step 1 initial screenshot" in prompt_text


def test_web_evaluator_service_trace_preserves_prompt_images_and_redacts_safety_fields(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    step_record = _step_record(persistence, round_num=1, round_step=1, session_step=1, title="TraceImage")
    persistence.persist_step(step_record)
    persistence.persist_result(
        WebAgentResult(
            status=WebRunStatus.SUCCESS,
            terminal_status=WebTerminalStatus.COMPLETED,
            session_id="session",
            user_query="judge trace image",
            success=True,
            steps=[step_record],
        )
    )

    class _TraceImageExecutor:
        async def run_step(self, **kwargs: object) -> SimpleNamespace:
            envelope = kwargs["build_envelope"]([], None)
            messages = envelope.messages
            image_blocks = [block for message in messages for block in message.content if isinstance(block, ImageBlock)]
            assert image_blocks
            return SimpleNamespace(
                response_text="calling final_report",
                meta={
                    "pending_tool_calls": [{"id": "call-final", "name": "final_report", "input": _web_report_payload()}]
                },
                trace_state=TraceCaptureState(
                    request_payload={
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": image_blocks[0].url,
                                        "safety_identifier": "raw-user-safety-id",
                                    }
                                ],
                            }
                        ]
                    },
                    assistant_text="calling final_report",
                ),
            )

    _run(
        WebRoundEvaluatorService(model_step_executor=_TraceImageExecutor()).evaluate_artifact_root(
            persistence.artifact_store.root
        )
    )

    trace_path = (
        persistence.artifact_store.root / "evaluations" / "web_round_evaluator" / "round_0001" / "llm_trace.jsonl"
    )
    trace_text = trace_path.read_text(encoding="utf-8")
    row = json.loads(trace_text.splitlines()[0])
    assert "data:image/jpeg;base64," in trace_text
    assert "raw-user-safety-id" not in trace_text
    assert row["source"] == "evaluator"
    assert row["llm_trace"]["request"]["messages"][0]["content"][0]["safety_identifier"] == (
        "[redacted sensitive value]"
    )
    runtime_path = trace_path.with_name("runtime.jsonl")
    runtime_events = [
        json.loads(line).get("event") for line in runtime_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert "bundle_built" in runtime_events
    assert "evidence_selected" in runtime_events
    assert "agentkit_final_report_completed" in runtime_events


def test_web_live_probe_dispatchers_bind_local_and_remote_browser_tools(tmp_path: Path) -> None:
    live = build_web_round_evaluator_spec(enable_live_probe=True)
    tools = {tool.name: tool for tool in live.tools}
    mutating_live = build_web_round_evaluator_spec(enable_live_probe=True, allow_client_mutating_tools=True)
    mutating_tools = {tool.name: tool for tool in mutating_live.tools}
    probe_client = _ProbeClient()
    local_dispatcher = WebBrowserLiveProbeDispatcher(
        registered_session=_registered_session(tmp_path, client=probe_client)
    )

    snapshot = _run(local_dispatcher.dispatch(tool=tools["client_browser_snapshot"], parameters={}, remote=False))
    tabs = _run(local_dispatcher.dispatch(tool=tools["client_browser_tabs"], parameters={}, remote=False))
    url = _run(local_dispatcher.dispatch(tool=tools["client_browser_get_url"], parameters={}, remote=False))
    screenshot = _run(
        local_dispatcher.dispatch(
            tool=tools["client_browser_screenshot"],
            parameters={"annotate": False},
            remote=False,
        )
    )

    assert snapshot["status"] == "success"
    assert snapshot["result"]["result"]["title"] == "Probe"
    assert "raw" not in snapshot["result"]["result"]
    assert "/tmp/private-dom.json" not in json.dumps(snapshot, ensure_ascii=False)
    assert "data:IMAGE/PNG" not in json.dumps(snapshot, ensure_ascii=False)
    assert tabs["result"]["result"]["tabs"][0]["title"] == "Probe"
    assert url["result"]["result"]["url"] == "http://127.0.0.1/probe"
    assert screenshot["result"]["result"]["width"] == 640
    assert screenshot["result"]["screenshot_url"].startswith("[image data omitted; chars=")
    assert "image_omitted" not in screenshot["result"]
    assert "data_url" not in screenshot["result"]["result"]
    assert "source_screenshot_path" not in screenshot["result"]["result"]["annotation"]
    assert "overlay_screenshot_path" not in screenshot["result"]["result"]["annotation"]
    assert probe_client.screenshot_kwargs == [{"annotate": False}]

    annotated_screenshot = _run(
        local_dispatcher.dispatch(
            tool=tools["client_browser_screenshot"],
            parameters={"annotate": True},
            remote=False,
        )
    )
    annotation = annotated_screenshot["result"]["result"]["annotation"]
    assert annotation["kind"] == "set_of_mark"
    assert annotation["marks"][0]["ref"] == "1"
    assert annotation["marks"][0]["bbox_css"] == {"x": 10.0, "y": 20.0, "width": 120.0, "height": 30.0}
    assert "source_screenshot_path" not in annotation
    assert "overlay_screenshot_path" not in annotation
    assert probe_client.screenshot_kwargs[-1] == {"annotate": False}

    before_invalid_calls = len(probe_client.screenshot_kwargs)
    invalid_screenshot = _run(
        local_dispatcher.dispatch(
            tool=tools["client_browser_screenshot"],
            parameters={"annotate": True, "full_page": True},
            remote=False,
        )
    )
    assert invalid_screenshot["status"] == "failure"
    assert invalid_screenshot["result"]["status"] == "error"
    assert invalid_screenshot["result"]["action"] == "screenshot"
    assert "Extra inputs are not permitted" in invalid_screenshot["error"]
    assert len(probe_client.screenshot_kwargs) == before_invalid_calls

    invalid_raw_input = _run(
        local_dispatcher.dispatch(
            tool=tools["client_browser_screenshot"],
            parameters={"annotate": "data:IMAGE/PNG;base64,AAAA /tmp/private.png"},
            remote=False,
        )
    )
    invalid_raw_json = json.dumps(invalid_raw_input, ensure_ascii=False)
    assert invalid_raw_input["status"] == "failure"
    assert "data:IMAGE/PNG" not in invalid_raw_json
    assert "/tmp/private.png" not in invalid_raw_json

    error_screenshot = _run(
        WebBrowserLiveProbeDispatcher(
            registered_session=_registered_session(
                tmp_path, client=_CaptureErrorProbeClient(), session_id="capture-error"
            )
        ).dispatch(
            tool=tools["client_browser_screenshot"],
            parameters={},
            remote=False,
        )
    )
    error_json = json.dumps(error_screenshot, ensure_ascii=False)
    assert error_screenshot["result"]["result"]["capture_error"] == "[redacted capture error]"
    assert error_screenshot["result"]["result"]["capture_error_present"] is True
    assert error_screenshot["result"]["result"]["image_omitted_reason"] == "[redacted image_omitted_reason]"
    assert error_screenshot["result"]["result"]["image_omitted_reason_present"] is True
    assert "/tmp/private-shot.png" not in error_json
    assert "data:IMAGE/PNG" not in error_json

    opened = _run(
        local_dispatcher.dispatch(
            tool=mutating_tools["client_browser_open"],
            parameters={"url": "http://127.0.0.1/mutating"},
            remote=False,
        )
    )
    clicked = _run(
        local_dispatcher.dispatch(
            tool=mutating_tools["client_browser_click_ref"],
            parameters={"ref": "1"},
            remote=False,
        )
    )
    assert opened["status"] == "success"
    assert clicked["status"] == "success"
    assert probe_client.opened_urls == ["http://127.0.0.1/mutating"]
    assert probe_client.clicked_refs == ["@e1"]

    dispatched: list[dict[str, object]] = []

    async def _dispatch(pending):
        dispatched.extend(pending)
        call_id = str(pending[0]["id"])
        name = str(pending[0]["name"])
        result = {"status": "ok", "result": {"snapshot": "remote snapshot", "title": "Remote"}}
        if name == "web.web_list_tabs":
            result = {"status": "ok", "result": {"tabs": [{"title": "Remote"}]}}
        if name == "web.web_get_url":
            result = {"status": "ok", "result": {"url": "https://remote.test/probe"}}
        elif name == "web.web_screenshot":
            result = {
                "status": "ok",
                "result": {
                    "width": 800,
                    "height": 600,
                    "screenshot_url": _png_data_url("red").replace("image", "IMAGE"),
                    "capture_error": "remote capture failed at /tmp/remote.png",
                    "image_omitted_reason": "remote capture failed at /tmp/remote.png",
                },
                "screenshot_url": _png_data_url("red").replace("image", "IMAGE"),
            }
        return RemoteToolDispatchResult(
            tool_blocks=[ToolResultBlock(tool_use_id=call_id, content=name, is_error=False)],
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result=result,
                )
            ],
        )

    remote_snapshot = _run(
        RemoteEvaluatorClientToolDispatcher(_dispatch).dispatch(
            tool=tools["client_browser_snapshot"],
            parameters={},
            remote=True,
        )
    )
    remote_tabs = _run(
        RemoteEvaluatorClientToolDispatcher(_dispatch).dispatch(
            tool=tools["client_browser_tabs"],
            parameters={},
            remote=True,
        )
    )

    assert dispatched[0]["name"] == "web.web_snapshot"
    assert dispatched[0]["server_name"] == "WebAgentActionExecutor"
    assert dispatched[0]["input"] == {}
    assert remote_snapshot["result"]["result"]["title"] == "Remote"
    assert dispatched[1]["name"] == "web.web_list_tabs"
    assert dispatched[1]["server_name"] == "WebAgentActionExecutor"
    assert dispatched[1]["input"] == {}
    assert remote_tabs["result"]["result"]["tabs"][0]["title"] == "Remote"

    remote_url = _run(
        RemoteEvaluatorClientToolDispatcher(_dispatch).dispatch(
            tool=tools["client_browser_get_url"],
            parameters={},
            remote=True,
        )
    )
    remote_screenshot = _run(
        RemoteEvaluatorClientToolDispatcher(_dispatch).dispatch(
            tool=tools["client_browser_screenshot"],
            parameters={"annotate": False},
            remote=True,
        )
    )

    assert dispatched[2]["name"] == "web.web_get_url"
    assert dispatched[2]["server_name"] == "WebAgentActionExecutor"
    assert dispatched[2]["input"] == {}
    assert remote_url["result"]["result"]["url"] == "https://remote.test/probe"
    assert dispatched[3]["name"] == "web.web_screenshot"
    assert dispatched[3]["server_name"] == "WebAgentActionExecutor"
    assert dispatched[3]["input"] == {"annotate": False}
    assert remote_screenshot["result"]["result"]["screenshot_url"].startswith("[image data omitted; chars=")
    assert remote_screenshot["result"]["result"]["capture_error"] == "[redacted capture_error]"
    assert remote_screenshot["result"]["result"]["capture_error_present"] is True
    assert remote_screenshot["result"]["result"]["image_omitted_reason"] == "[redacted image_omitted_reason]"
    assert remote_screenshot["result"]["result"]["image_omitted_reason_present"] is True
    assert remote_screenshot["result"]["screenshot_url"].startswith("[image data omitted; chars=")
    assert "/tmp/remote.png" not in json.dumps(remote_screenshot, ensure_ascii=False)

    remote_click = _run(
        RemoteEvaluatorClientToolDispatcher(_dispatch).dispatch(
            tool=mutating_tools["client_browser_click_ref"],
            parameters={"ref": "1"},
            remote=True,
        )
    )

    assert dispatched[4]["name"] == "web.web_click_ref"
    assert dispatched[4]["server_name"] == "WebAgentActionExecutor"
    assert dispatched[4]["input"] == {"ref": "1"}
    assert remote_click["status"] == "success"


def test_web_live_probe_requires_explicit_dispatcher_for_service(tmp_path: Path) -> None:
    service = WebRoundEvaluatorService(model_step_executor=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="requires a bound"):
        _run(service.evaluate_artifact_root(tmp_path, enable_live_probe=True))


def test_step_trace_rewrite_redacts_data_urls(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    observation = _observation("Before", "https://example.test/before", "button @e1 Before", "white")
    paths = _run(
        persistence.persist_observation_artifacts(client=_TerminalClient(), step_index=1, observation=observation)
    )
    action = ResolvedWebAgentAction(
        action=WebAgentActionType.SCREENSHOT,
        tool_name="web_screenshot",
        tool_input={"annotate": False},
        tool_output={"status": "ok", "text": "Captured.", "screenshot_url": _png_data_url("red")},
    )
    step_record = WebAgentStepRecordProjector(action_resolver=WebAgentActionResolver()).build_step_record(
        session_id="test-session",
        step_index=1,
        round_num=1,
        session_step=1,
        runtime_cursor={},
        observation=observation,
        step_result=_step_result(action),
        resolved_actions=[action],
        artifact_paths=paths,
        observation_artifact=persistence.build_observation_artifact(observation=observation, paths=paths),
        tool_artifacts=[],
    )
    persistence.replace_step_trace([step_record])

    trace_text = persistence.artifact_store.trace_path.read_text(encoding="utf-8")
    assert "data:image/png;base64" not in trace_text
    assert "redacted data URL" in trace_text


def test_offline_replay_loads_realistic_web_artifact_root(tmp_path: Path) -> None:
    persistence = _persistence(tmp_path)
    step_record = _step_record(persistence, round_num=1, round_step=1, session_step=1, title="Replay")
    persistence.persist_step(step_record)
    result = WebAgentResult(
        status=WebRunStatus.SUCCESS,
        terminal_status=WebTerminalStatus.COMPLETED,
        session_id="session",
        user_query="replay",
        success=True,
        steps=[step_record],
    )
    persistence.persist_result(result)

    offline = load_web_offline_evaluation_bundle(persistence.artifact_store.root)

    assert offline.result.session_id == "session"
    assert offline.bundle.dom_refs
    assert offline.step_trace_by_session_step[1]["session_step"] == 1


def test_offline_replay_rejects_removed_step_mirror_fields(tmp_path: Path) -> None:
    root = tmp_path / "legacy-result"
    root.mkdir()
    (root / "result.json").write_text(
        json.dumps(
            {
                "status": "success",
                "terminal_status": "completed",
                "session_id": "legacy",
                "user_query": "legacy replay",
                "success": True,
                "steps": [
                    {
                        "step_index": 1,
                        "status": "ok",
                        "action": "web_click_ref",
                        "tool_input": {"ref": "@e1"},
                        "tool_output": {"status": "ok"},
                        "tools": [],
                    }
                ],
                "screenshots": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "web_agent_trace.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_web_offline_evaluation_bundle(root)


def test_query_som_annotation_resolves_tool_image_ref_not_current_ref(tmp_path: Path) -> None:
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="check som",
        session_request="check som",
        screenshot_refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.IMAGE,
                label="current",
                path=str(tmp_path / "current.png"),
                session_step=1,
                round_step=1,
            ),
            EvaluationEvidenceRef(
                id="E002",
                kind=EvaluationEvidenceKind.IMAGE,
                label="tool som",
                path=str(tmp_path / "tool_som.png"),
                session_step=1,
                round_step=1,
                metadata={"som_metadata": {"kind": "set_of_mark", "marks": [{"ref": "@e1", "label": "1"}]}},
            ),
        ],
    )
    factory = ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=list(bundle.screenshot_refs)),
        step_trace_by_session_step={},
    )

    result = factory._query_som_annotation(session_step=1)

    assert result["status"] == "success"
    assert result["evidence_ref"] == "E002"
    assert result["marks"] == [{"ref": "@e1", "label": "1"}]


def test_query_som_annotation_falls_back_from_clean_ref_to_step_sidecar(tmp_path: Path) -> None:
    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    clean_path = screenshots / "step_001_raw.png"
    annotated_path = screenshots / "step_001_annotated.png"
    _write_png(clean_path, "white")
    _write_png(annotated_path, "green")
    annotated_path.with_suffix(".annotation.json").write_text(
        json.dumps(
            {
                "kind": "set_of_mark",
                "truncated": True,
                "omitted_count": 7,
                "max_marks": 60,
                "marks": [{"ref": "@e1", "label": "1"}],
            }
        ),
        encoding="utf-8",
    )
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="check som",
        session_request="check som",
        screenshot_refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.IMAGE,
                label="current",
                path=str(clean_path),
                session_step=1,
                round_step=1,
            )
        ],
    )
    factory = ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=list(bundle.screenshot_refs)),
        step_trace_by_session_step={1: {"observation_artifact": {"annotated_screenshot_path": str(annotated_path)}}},
    )

    result = factory._query_som_annotation(evidence_ref="E001")

    assert result["status"] == "success"
    assert result["session_step"] == 1
    assert result["truncated"] is True
    assert result["omitted_count"] == 7
    assert result["marks"] == [{"ref": "@e1", "label": "1"}]


def test_web_evaluator_tools_accept_assistant_final_text(tmp_path: Path) -> None:
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="answer title",
        session_request="answer title",
        assistant_texts=["I will inspect the page first.", "Top title"],
        step_refs=[
            RoundEvaluationStepRef(
                session_step=1,
                round_step=1,
                status="ok",
                request="answer title",
                result="Top title",
                assistant_text="Top title",
            )
        ],
    )
    factory = ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=[]),
        step_trace_by_session_step={1: {"session_step": 1}},
    )

    assert factory._read_round_bundle(section="trace")["status"] == "success"
    trajectory_section = factory._read_round_bundle(section="trajectory")
    assert trajectory_section["status"] == "success"
    assert trajectory_section["section"] == "trace"
    debug_trajectory_section = factory._read_round_bundle(section="trajectory", detail_level="debug")
    assert debug_trajectory_section["status"] == "success"
    assert debug_trajectory_section["section"] == "trace"
    final_section = factory._read_round_bundle(section="final")
    assert final_section["status"] == "success"
    assert final_section["value"]["final_output_candidates"] == [
        {"source": "assistant_final_text", "index": 2, "text": "Top title"}
    ]
    page_state = factory._query_web_page_state(session_step=1)
    assert page_state["final_output_candidates"] == [
        {"source": "assistant_final_text", "index": 2, "text": "Top title"}
    ]
    assert page_state["final_output_verification"]["status"] == "missing_dom"
    assert factory._query_tool_results(tool_name="done", session_step=1)["matches"] == []
    assert factory._query_tool_results(tool_name="final", session_step=1)["matches"] == []


def test_web_page_state_verifies_final_output_against_terminal_dom(tmp_path: Path) -> None:
    dom_path = tmp_path / "step_001.json"
    text_path = tmp_path / "step_001.txt"
    dom_path.write_text(
        json.dumps(
            {
                "url": "https://example.test",
                "title": "Example",
                "snapshot": 'link "Top title" [ref=@e1]',
                "elements": [{"ref_id": "e1", "role": "link", "name": "Top title", "text": "Top title"}],
            }
        ),
        encoding="utf-8",
    )
    text_path.write_text('link "Top title" [ref=@e1]', encoding="utf-8")
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="answer title",
        session_request="answer title",
        assistant_texts=["Top title"],
        dom_refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.DOM,
                label="terminal DOM",
                path=str(dom_path),
                session_step=1,
                round_step=1,
                metadata={"text_path": str(text_path), "variant": "terminal_verification"},
            )
        ],
    )
    factory = ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=list(bundle.dom_refs)),
        step_trace_by_session_step={1: {"session_step": 1}},
    )

    page_state = factory._query_web_page_state(session_step=1)

    assert page_state["final_output_verification"]["status"] == "matched"
    assert page_state["final_output_verification"]["matches"][0]["match_type"] == "exact"


def test_web_page_state_marks_final_output_substring_as_partial(tmp_path: Path) -> None:
    dom_path = tmp_path / "step_001.json"
    text_path = tmp_path / "step_001.txt"
    dom_path.write_text(
        json.dumps(
            {
                "url": "https://example.test",
                "title": "Example",
                "snapshot": 'main "Top title - extra context" [ref=@e1]',
                "elements": [
                    {
                        "ref_id": "e1",
                        "role": "main",
                        "name": "Top title - extra context",
                        "text": "Top title - extra context",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    text_path.write_text('main "Top title - extra context" [ref=@e1]', encoding="utf-8")
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="answer title",
        session_request="answer title",
        assistant_texts=["Top title"],
        dom_refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.DOM,
                label="terminal DOM",
                path=str(dom_path),
                session_step=1,
                round_step=1,
                metadata={"text_path": str(text_path), "variant": "terminal_verification"},
            )
        ],
    )
    factory = ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=list(bundle.dom_refs)),
        step_trace_by_session_step={1: {"session_step": 1}},
    )

    page_state = factory._query_web_page_state(session_step=1)

    assert page_state["final_output_verification"]["status"] == "partial_match"
    assert page_state["final_output_verification"]["matches"][0]["match_type"] == "substring"


def _persistence(tmp_path: Path) -> WebAgentArtifactPersistence:
    store = WebArtifactStore.for_session(tmp_path, "session", ArtifactConfig())
    assert isinstance(store, WebArtifactStore)
    return WebAgentArtifactPersistence(artifact_store=store, config=WebAgentConfig())


def _step_record(
    persistence: WebAgentArtifactPersistence,
    *,
    round_num: int,
    round_step: int,
    session_step: int,
    title: str,
):
    observation = _observation(title, f"https://example.test/{title.lower()}", f"button @e1 {title}", "white")
    paths = _run(
        persistence.persist_observation_artifacts(
            client=_TerminalClient(),
            step_index=session_step,
            observation=observation,
        )
    )
    return _terminal_step_record(
        persistence,
        observation=observation,
        paths=paths,
        step_index=round_step,
        round_num=round_num,
        session_step=session_step,
    )


def _terminal_step_record(
    persistence: WebAgentArtifactPersistence,
    *,
    observation: WebObservationContext,
    paths: dict[str, Path],
    step_index: int = 1,
    round_num: int = 1,
    session_step: int = 1,
    final_text: str = "done",
):
    return WebAgentStepRecordProjector(action_resolver=WebAgentActionResolver()).build_step_record(
        session_id="test-session",
        step_index=step_index,
        round_num=round_num,
        session_step=session_step,
        runtime_cursor={},
        observation=observation,
        step_result=_final_step_result(final_text),
        resolved_actions=[],
        artifact_paths=paths,
        observation_artifact=persistence.build_observation_artifact(observation=observation, paths=paths),
        tool_artifacts=[],
    )


def _step_record_from_action(
    persistence: WebAgentArtifactPersistence,
    *,
    observation: WebObservationContext,
    paths: dict[str, Path],
    action: ResolvedWebAgentAction,
):
    return WebAgentStepRecordProjector(action_resolver=WebAgentActionResolver()).build_step_record(
        session_id="test-session",
        step_index=1,
        round_num=1,
        session_step=1,
        runtime_cursor={},
        observation=observation,
        step_result=_step_result(action),
        resolved_actions=[action],
        artifact_paths=paths,
        observation_artifact=persistence.build_observation_artifact(observation=observation, paths=paths),
        tool_artifacts=[],
    )


def _observation(title: str, url: str, snapshot: str, color: str) -> WebObservationContext:
    return WebObservationContext(
        snapshot=WebSnapshotObservation(
            url=url,
            title=title,
            snapshot=snapshot,
            elements=[WebElementRef(ref_id="e1", role="button", name=title)],
            raw={"viewport": {"width": 640, "height": 360}},
        ),
        screenshot=WebScreenshotObservation(data_url=_png_data_url(color)),
        observation_id=f"obs-{title.lower()}",
    )


def _step_result(action: ResolvedWebAgentAction) -> WebAgentModelStepResult:
    state = TraceCaptureState(assistant_text="step")
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id=action.tool_call_id or "call-1",
            tool_name=action.tool_name,
            tool_input=action.tool_input,
            tool_output=action.tool_output,
            is_error=action.is_error,
        )
    )
    return WebAgentModelStepResult(trace_state=state, response_text="step", meta={}, envelope=None)


def _final_step_result(text: str = "done") -> WebAgentModelStepResult:
    state = TraceCaptureState(assistant_text=text)
    return WebAgentModelStepResult(trace_state=state, response_text=text, meta={}, envelope=None)


def _web_report_payload() -> dict[str, object]:
    return {
        "overall_verdict": "success",
        "process_rating": 5,
        "outcome_label": "success",
        "completion_rating": 5,
        "confidence_rating": 4,
        "controllability": "controllable",
        "reason": "Terminal DOM and screenshot prove completion.",
        "criteria": [],
        "diagnoses": [],
        "side_effects": [],
        "safety": {"risk_level": "none", "findings": []},
        "efficiency": {"rating": 5, "reason": "Direct path."},
        "final_url": "https://example.test/done",
        "final_title": "Done",
        "page_state_verified": True,
        "dom_state_verified": True,
        "visual_state_verified": True,
        "critical_web_evidence_refs": ["E001"],
    }


def _llm_trace_wrapper_row(
    *,
    session_step: int,
    request: str,
    round_num: int = 1,
    round_step: int = 1,
    llm_request: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "source": "web",
        "session_id": "web-trace-test",
        "session_step": session_step,
        "round_num": round_num,
        "round_step": round_step,
        "request": request,
        "runtime_cursor": {"round_step": round_step},
        "llm_trace": {"request": llm_request or {"messages": [], "tools": []}},
    }


def _png_data_url(color: str) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_png(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path, format="PNG")


def _write_som_sidecar(path: Path, *, source: Path) -> None:
    path.with_suffix(".annotation.json").write_text(
        json.dumps(
            {
                "kind": "set_of_mark",
                "source_screenshot_path": str(source),
                "overlay_screenshot_path": str(path),
            }
        ),
        encoding="utf-8",
    )


def _registered_session(
    tmp_path: Path,
    *,
    client,
    session_id: str = "probe-session",
) -> RegisteredWebAgentSession:
    config = WebAgentConfig()
    return RegisteredWebAgentSession(
        session_id=session_id,
        client=client,
        artifact_store=WebArtifactStore.for_session(tmp_path, session_id, ArtifactConfig(root_dir=str(tmp_path))),
        settings=config,
    )


def _run(coro):
    import asyncio

    return asyncio.run(coro)
