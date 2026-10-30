"""Desktop-style action overlay adapter tests for Web Agent."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from northcua.agents.web.artifacts import WebArtifactStore
from northcua.agents.web.domain.actions import WebAgentActionType
from northcua.agents.web.domain.evaluation_artifacts import WebToolArtifactImageKind
from northcua.agents.web.domain.observations import (
    WebBoundingBox,
    WebObservationContext,
    WebScreenshotAnnotation,
    WebScreenshotMark,
    WebScreenshotObservation,
    WebSnapshotObservation,
)
from northcua.agents.web.mcp import action_service as action_service_module
from northcua.agents.web.mcp.action_service import WebAgentMCPActionService
from northcua.agents.web.overlay import WebActionOverlayAdapter
from northcua.agents.web.session.models import RegisteredWebAgentSession
from northcua.core.config import ArtifactConfig, WebAgentConfig


def test_click_ref_maps_numeric_som_mark_to_shared_click_command() -> None:
    observation = _observation_with_marks(
        WebScreenshotMark(
            ref="19",
            label="19",
            source_ref="@e39",
            bbox_css=WebBoundingBox(x=10, y=20, width=30, height=20),
            bbox_image=WebBoundingBox(x=20, y=40, width=60, height=40),
        )
    )

    plan = WebActionOverlayAdapter().build_plan(
        action=WebAgentActionType.CLICK_REF,
        params={"ref": 19},
        observation=observation,
    )

    assert plan is not None
    assert plan.source_image_path == "/tmp/pre_raw.png"
    assert plan.command.function == "click"
    assert plan.command.arguments == {"x": 50, "y": 60, "ref": "19"}


def test_hover_ref_maps_to_shared_move_command_and_source_ref_lookup() -> None:
    observation = _observation_with_marks(
        WebScreenshotMark(
            ref="4",
            label="4",
            source_ref="@e12",
            bbox_css=WebBoundingBox(x=10, y=20, width=30, height=20),
            bbox_image=WebBoundingBox(x=20, y=40, width=60, height=40),
        )
    )

    plan = WebActionOverlayAdapter().build_plan(
        action=WebAgentActionType.HOVER_REF,
        params={"ref": "@e12"},
        observation=observation,
    )

    assert plan is not None
    assert plan.command.function == "move"
    assert plan.command.arguments == {"x": 50, "y": 60, "ref": "@e12"}


def test_ref_overlay_prefers_resolved_source_ref_over_model_alias() -> None:
    observation = _observation_with_marks(
        WebScreenshotMark(
            ref="4",
            label="4",
            source_ref="@e39",
            bbox_css=WebBoundingBox(x=10, y=20, width=30, height=20),
            bbox_image=WebBoundingBox(x=20, y=40, width=60, height=40),
        )
    )

    plan = WebActionOverlayAdapter().build_plan(
        action=WebAgentActionType.CLICK_REF,
        params={"ref": "7", "source_ref": "@e39"},
        observation=observation,
    )

    assert plan is not None
    assert plan.command.function == "click"
    assert plan.command.arguments == {"x": 50, "y": 60, "ref": "7"}


def test_fill_ref_overlay_uses_type_badge_instead_of_click_marker() -> None:
    observation = _observation_with_marks(
        WebScreenshotMark(
            ref="4",
            label="4",
            source_ref="@e39",
            bbox_css=WebBoundingBox(x=10, y=20, width=30, height=20),
            bbox_image=WebBoundingBox(x=20, y=40, width=60, height=40),
        )
    )

    plan = WebActionOverlayAdapter().build_plan(
        action=WebAgentActionType.FILL_REF,
        params={"ref": "4", "text": "Xbox Wireless Controller"},
        observation=observation,
    )

    assert plan is not None
    assert plan.command.function == "type"
    assert plan.command.arguments == {"text": "Xbox Wireless Controller"}


def test_drag_ref_maps_source_and_target_mark_centers() -> None:
    observation = _observation_with_marks(
        WebScreenshotMark(
            ref="1",
            label="1",
            source_ref="@e1",
            bbox_css=WebBoundingBox(x=5, y=5, width=10, height=10),
            bbox_image=WebBoundingBox(x=10, y=10, width=20, height=20),
        ),
        WebScreenshotMark(
            ref="2",
            label="2",
            source_ref="@e2",
            bbox_css=WebBoundingBox(x=50, y=60, width=10, height=20),
            bbox_image=WebBoundingBox(x=100, y=120, width=20, height=40),
        ),
    )

    plan = WebActionOverlayAdapter().build_plan(
        action=WebAgentActionType.DRAG_REF,
        params={"from_ref": "1", "to_ref": "2"},
        observation=observation,
    )

    assert plan is not None
    assert plan.command.function == "drag"
    assert plan.command.arguments["path"] == [{"x": 20, "y": 20}, {"x": 110, "y": 140}]
    assert plan.command.arguments["from_ref"] == "1"
    assert plan.command.arguments["to_ref"] == "2"


def test_point_actions_scale_css_coordinates_to_screenshot_pixels() -> None:
    observation = _observation_with_marks(
        WebScreenshotMark(
            ref="1",
            label="1",
            source_ref="@e1",
            bbox_css=WebBoundingBox(x=0, y=0, width=40, height=20),
            bbox_image=WebBoundingBox(x=0, y=0, width=80, height=40),
        )
    )

    plan = WebActionOverlayAdapter().build_plan(
        action=WebAgentActionType.DOUBLE_CLICK_POINT,
        params={"x": 12.5, "y": 20},
        observation=observation,
    )

    assert plan is not None
    assert plan.command.function == "double_click"
    assert plan.command.arguments == {"x": 25, "y": 40}


def test_scroll_maps_to_shared_scroll_command() -> None:
    observation = _observation_with_marks()

    plan = WebActionOverlayAdapter().build_plan(
        action=WebAgentActionType.SCROLL,
        params={"direction": "down", "amount": 900},
        observation=observation,
    )

    assert plan is not None
    assert plan.command.function == "scroll"
    assert plan.command.arguments == {"delta_x": 0, "delta_y": 900}


def test_navigation_actions_do_not_emit_spatial_overlay() -> None:
    observation = _observation_with_marks()

    plan = WebActionOverlayAdapter().build_plan(
        action=WebAgentActionType.OPEN,
        params={"url": "https://example.test"},
        observation=observation,
    )

    assert plan is None


def test_action_service_renders_pre_action_overlay_artifact(tmp_path: Path) -> None:
    raw_path = tmp_path / "pre_raw.png"
    som_path = tmp_path / "pre_som.png"
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(raw_path)
    Image.new("RGB", (200, 100), color=(245, 245, 245)).save(som_path)
    observation = _observation_with_marks(
        WebScreenshotMark(
            ref="7",
            label="7",
            source_ref="@e7",
            bbox_css=WebBoundingBox(x=10, y=20, width=30, height=20),
            bbox_image=WebBoundingBox(x=20, y=40, width=60, height=40),
        ),
        source_screenshot_path=str(raw_path),
        overlay_screenshot_path=str(som_path),
    )
    store = WebArtifactStore.for_session(tmp_path / "artifacts", "sess-overlay", ArtifactConfig())
    registered = RegisteredWebAgentSession(
        session_id="sess-overlay",
        client=object(),  # type: ignore[arg-type]
        artifact_store=store,
        settings=WebAgentConfig(),
    )
    previous = registered.begin_tool_artifact_invocation(WebAgentActionType.CLICK_REF.value)
    try:
        output_path = WebAgentMCPActionService().render_pre_action_overlay(
            registered=registered,
            action=WebAgentActionType.CLICK_REF,
            params={"ref": "7"},
            observation=observation,
        )
    finally:
        registered.restore_tool_artifact_invocation(previous)

    assert output_path.endswith("_action_overlay.png")
    assert Path(output_path).is_file()
    assert Path(output_path).with_suffix(".overlay.json").is_file()
    assert len(registered.tool_image_artifact_events) == 1
    event = registered.tool_image_artifact_events[0]
    assert event.raw_image_path == str(raw_path)
    assert event.action_overlay_image_path == output_path
    assert event.selected_image_kind == WebToolArtifactImageKind.ACTION_OVERLAY


def test_action_overlay_runtime_log_is_path_free(tmp_path: Path, monkeypatch) -> None:
    raw_path = tmp_path / "pre_raw.png"
    som_path = tmp_path / "pre_som.png"
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(raw_path)
    Image.new("RGB", (200, 100), color=(245, 245, 245)).save(som_path)
    observation = _observation_with_marks(
        WebScreenshotMark(
            ref="7",
            label="7",
            source_ref="@e7",
            bbox_css=WebBoundingBox(x=10, y=20, width=30, height=20),
            bbox_image=WebBoundingBox(x=20, y=40, width=60, height=40),
        ),
        source_screenshot_path=str(raw_path),
        overlay_screenshot_path=str(som_path),
    )
    store = WebArtifactStore.for_session(tmp_path / "artifacts", "sess-overlay-log", ArtifactConfig())
    registered = RegisteredWebAgentSession(
        session_id="sess-overlay-log",
        client=object(),  # type: ignore[arg-type]
        artifact_store=store,
        settings=WebAgentConfig(),
    )
    runtime_events: list[dict[str, object]] = []

    def _capture_log(_logger, event_name: str, **fields: object) -> None:
        runtime_events.append({"event": event_name, **fields})

    monkeypatch.setattr(action_service_module, "log_runtime_event", _capture_log)
    previous = registered.begin_tool_artifact_invocation(WebAgentActionType.CLICK_REF.value)
    try:
        WebAgentMCPActionService().render_pre_action_overlay(
            registered=registered,
            action=WebAgentActionType.CLICK_REF,
            params={"ref": "7"},
            observation=observation,
        )
    finally:
        registered.restore_tool_artifact_invocation(previous)

    rendered = [event for event in runtime_events if event["event"] == "web_agent_action_overlay_rendered"]
    assert rendered
    rendered_text = str(rendered[-1])
    assert str(raw_path) not in rendered_text
    assert "_action_overlay.png" not in rendered_text
    assert "data:image" not in rendered_text.lower()


def _observation_with_marks(
    *marks: WebScreenshotMark,
    source_screenshot_path: str = "/tmp/pre_raw.png",
    overlay_screenshot_path: str = "/tmp/pre_som.png",
) -> WebObservationContext:
    return WebObservationContext(
        snapshot=WebSnapshotObservation(url="https://example.test", title="Example"),
        screenshot=WebScreenshotObservation(
            path=overlay_screenshot_path,
            width=200,
            height=100,
            annotation=WebScreenshotAnnotation(
                marks=list(marks),
                source_screenshot_path=source_screenshot_path,
                overlay_screenshot_path=overlay_screenshot_path,
            ),
        ),
        observation_id="pre-click-ref-1",
    )
