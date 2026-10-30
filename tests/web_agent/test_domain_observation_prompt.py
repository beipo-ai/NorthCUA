from __future__ import annotations

import base64
import json
from itertools import combinations
from pathlib import Path

import pytest
from pydantic import ValidationError

from northcua.agentkernel.tools.surface import ToolSurfaceService
from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agents.web.backend import WebActionBackendTimeoutError
from northcua.agents.web.domain import (
    WebAgentActionType,
    WebClickPointParams,
    WebCloseTabParams,
    WebDragPointParams,
    WebDragRefParams,
    WebFocusTabParams,
    WebMouseButton,
    WebNewTabParams,
    WebNoteParams,
    WebRefParams,
    WebRefsParams,
    WebScreenshotParams,
    WebSelectOptionRefParams,
    WebWaitParams,
    action_from_tool_name,
)
from northcua.agents.web.domain.observations import (
    WebBoundingBox,
    WebObservationContext,
    WebScreenshotAnnotation,
    WebScreenshotMark,
    WebScreenshotObservation,
)
from northcua.agents.web.domain.results import WebStepRecord, WebStepStatus
from northcua.agents.web.mcp.descriptors import (
    web_agent_action_contracts,
    web_agent_backend_contracts,
    web_agent_probe_contracts,
)
from northcua.agents.web.mcp.error_classification import classify_web_tool_error
from northcua.agents.web.observation import WebSnapshotNormalizer
from northcua.agents.web.observation.projector import WebObservationProjector
from northcua.agents.web.prompt import WebTemplateRenderer
from northcua.core.config import PromptPolicyConfig, WebAgentConfig
from northcua.core.models import AssistantToolCallRecord, ToolExecutionRecord
from northcua_runtime_shared.tool_contracts import ToolProfile, ToolSurfaceResolver


def _top_level_image_urls(messages: list[object]) -> list[str]:
    return [
        str(getattr(block, "url", ""))
        for message in messages
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "image"
    ]


def _tool_result_image_urls(messages: list[object]) -> list[str]:
    return [
        str(getattr(content_block, "url", ""))
        for message in messages
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "tool_result" and isinstance(getattr(block, "content", None), list)
        for content_block in block.content
        if getattr(content_block, "type", None) == "image"
    ]


def _all_prompt_image_urls(messages: list[object]) -> list[str]:
    return _top_level_image_urls(messages) + _tool_result_image_urls(messages)


def _leading_user_text(messages: list[object]) -> str:
    for message in messages:
        role = getattr(message, "role", None)
        if role != "user" and getattr(role, "value", role) != "user":
            continue
        return "\n".join(
            str(getattr(block, "text", "")) for block in getattr(message, "content", []) if block.type == "text"
        )
    return ""


def _rendered_tool_text(messages: list[object]) -> str:
    parts: list[str] = []
    for message in messages:
        for block in getattr(message, "content", []):
            if getattr(block, "type", None) != "tool_result":
                continue
            content = getattr(block, "content", "")
            if isinstance(content, str):
                parts.append(content)
                continue
            if isinstance(content, list):
                parts.extend(
                    str(getattr(content_block, "text", ""))
                    for content_block in content
                    if getattr(content_block, "type", None) == "text"
                )
    return "\n".join(parts)


def _write_tiny_png(path: Path) -> Path:
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/Ob1G3wAAAABJRU5ErkJggg=="
        )
    )
    return path


def _write_distinct_tiny_png(path: Path, marker: str) -> Path:
    _write_tiny_png(path)
    path.write_bytes(path.read_bytes() + marker.encode("utf-8"))
    return path


def _data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def test_action_tool_mapping_and_params_are_typed() -> None:
    assert action_from_tool_name("web_done") is None
    assert action_from_tool_name("web_note") == WebAgentActionType.NOTE
    assert action_from_tool_name("web_hover_ref") == WebAgentActionType.HOVER_REF
    assert action_from_tool_name("web_click_point") == WebAgentActionType.CLICK_POINT
    assert action_from_tool_name("web_list_tabs") == WebAgentActionType.LIST_TABS
    assert WebNoteParams(text=" Remember Ada ", reason=" later ", kind="fact").model_dump(mode="json") == {
        "key": "",
        "text": "Remember Ada",
        "reason": "later",
        "kind": "fact",
    }
    assert WebWaitParams(duration_ms=100).duration_ms == 100
    assert WebWaitParams(text="x").text == "x"
    assert WebWaitParams(selector="placeholder").selector == "placeholder"
    with pytest.raises(ValidationError, match="url_pattern"):
        WebWaitParams.model_validate(
            {
                "url_pattern": "*",
            }
        )


def test_web_wait_params_accept_exactly_one_active_target_and_nullable_placeholders() -> None:
    valid_payloads = (
        {"duration_ms": 1000},
        {"load_state": "domcontentloaded"},
        {"text": "Ready"},
        {"url_pattern": "**/complete"},
        {"selector": "#result"},
    )

    for payload in valid_payloads:
        assert WebWaitParams.model_validate(payload).model_dump(mode="json", exclude_none=True) == payload

    with_null_placeholders = WebWaitParams.model_validate(
        {
            "duration_ms": 1000,
            "load_state": None,
            "text": None,
            "url_pattern": None,
            "selector": None,
        }
    )
    assert with_null_placeholders.model_dump(mode="json", exclude_none=True) == {"duration_ms": 1000}

    with_blank_placeholders = WebWaitParams.model_validate(
        {"duration_ms": 1000, "text": " ", "url_pattern": "", "selector": "\t"}
    )
    assert with_blank_placeholders.model_dump(mode="json", exclude_none=True) == {"duration_ms": 1000}


def test_web_wait_params_reject_every_pair_of_active_targets_without_priority() -> None:
    target_values: dict[str, object] = {
        "duration_ms": 1000,
        "load_state": "load",
        "text": "Ready",
        "url_pattern": "**/complete",
        "selector": "#result",
    }

    for first, second in combinations(target_values, 2):
        with pytest.raises(ValidationError) as error:
            WebWaitParams.model_validate({first: target_values[first], second: target_values[second]})
        issue = error.value.errors(include_url=False)[0]
        assert issue["type"] == "web_wait_target_combination"
        assert issue["ctx"]["received_targets"] == (first, second)
        assert f"received: {first}, {second}" in issue["msg"]


def test_web_wait_params_do_not_silently_remove_legacy_placeholder_words() -> None:
    for placeholder in ("x", "placeholder", "unused", "none", "null", "n/a", "na"):
        assert WebWaitParams(text=placeholder).text == placeholder

    with pytest.raises(ValidationError) as error:
        WebWaitParams.model_validate(
            {
                "duration_ms": 1000,
                "text": "placeholder",
                "selector": "x",
            }
        )
    issue = error.value.errors(include_url=False)[0]
    assert issue["type"] == "web_wait_target_combination"
    assert issue["ctx"]["received_targets"] == ("duration_ms", "text", "selector")


def test_web_wait_params_reject_zero_targets_and_unknown_fields() -> None:
    with pytest.raises(ValidationError) as error:
        WebWaitParams.model_validate({})
    issue = error.value.errors(include_url=False)[0]
    assert issue["type"] == "web_wait_target_combination"
    assert issue["ctx"]["received_targets"] == ()
    assert "received: <none>" in issue["msg"]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WebWaitParams.model_validate({"duration": 1000})


def test_web_post_action_observation_config_defaults_and_validation() -> None:
    config = WebAgentConfig()

    assert config.initial_observation_mode == "som"
    assert config.post_action_observation_mode == "som"
    assert config.post_action_observation_timeout_seconds == 15.0
    assert config.screenshot_timeout_seconds == 4.0
    assert config.readonly_post_observation is False
    assert config.post_action_settle.enabled is True
    assert config.post_action_settle.profile == "balanced"
    assert config.post_action_settle.debounce_ms == 150
    assert config.post_action_settle.max_wait_ms == 3000
    assert config.post_action_settle.snapshot_stable_samples == 2
    assert config.post_action_settle.bbox_stable_samples == 2
    assert config.post_action_settle.load_wait_timeout_ms == 0
    assert PromptPolicyConfig().ref_snapshot_window_observations == 1

    with pytest.raises(ValidationError):
        WebAgentConfig(post_action_observation_mode="raw")

    with pytest.raises(ValidationError):
        WebAgentConfig(post_action_observation_timeout_seconds=0)

    with pytest.raises(ValidationError, match="include_screenshot_by_default"):
        WebAgentConfig(include_screenshot_by_default=True)


def test_web_step_record_rejects_removed_primary_tool_mirrors() -> None:
    payload = {
        "step_index": 1,
        "status": WebStepStatus.OK.value,
        "action": "web_click_ref",
        "tool_input": {"ref": "@e1"},
        "tool_output": {"status": "ok"},
    }

    with pytest.raises(ValidationError, match="Extra inputs"):
        WebStepRecord.model_validate(payload)


def test_web_timeout_errors_are_retryable_for_agent_recovery() -> None:
    classification = classify_web_tool_error(WebActionBackendTimeoutError("playwright command timed out"))

    assert classification.code == "browser_timeout"
    assert classification.retryable is True
    assert "timed out" in classification.hint

    generic_timeout = classify_web_tool_error(RuntimeError("playwright command timed out after 30s"))
    assert generic_timeout.code == "browser_timeout"
    assert generic_timeout.retryable is True

    generic_validation = classify_web_tool_error(RuntimeError("1 validation error for WebWaitParams"))
    assert generic_validation.code == "invalid_params"
    assert generic_validation.retryable is True

    stale_alias = classify_web_tool_error(
        ValueError("Unknown or stale web ref '999'; use a numeric ref from the latest observation.")
    )
    assert stale_alias.code == "stale_ref"
    assert stale_alias.retryable is True


def test_web_wait_target_validation_error_reports_received_targets() -> None:
    with pytest.raises(ValidationError) as error:
        WebWaitParams.model_validate({"duration_ms": 1000, "load_state": "networkidle"})

    classification = classify_web_tool_error(error.value)

    assert classification.code == "invalid_wait_target_combination"
    assert classification.reason_code == "invalid_wait_target_combination"
    assert classification.retryable is True
    assert classification.received_targets == ("duration_ms", "load_state")
    assert "duration_ms, load_state" in classification.hint
    assert classification.feedback_source == "contract_validator"


def test_web_actionability_timeout_errors_get_specific_feedback() -> None:
    disabled = classify_web_tool_error(
        RuntimeError(
            "Locator.click: Timeout 30000ms exceeded.\n"
            "Call log:\n"
            '  - waiting for locator("button").first\n'
            "  - element is not enabled\n"
            "  - retrying click action"
        )
    )
    assert disabled.code == "element_disabled"
    assert disabled.reason_code == "element_disabled"
    assert disabled.retryable is True
    assert "disabled" in disabled.hint
    assert "not enabled" in disabled.concise_log_summary

    intercepted = classify_web_tool_error(
        RuntimeError(
            "Locator.hover: Timeout 30000ms exceeded.\n"
            "Call log:\n"
            "  - element is visible and stable\n"
            '  - <div class="min-h-screen">…</div> intercepts pointer events'
        )
    )
    assert intercepted.code == "element_not_receiving_events"
    assert intercepted.retryable is True
    assert "covered" in intercepted.hint or "overlay" in intercepted.hint
    assert "intercepts pointer events" in intercepted.concise_log_summary


def test_web_tool_result_replay_includes_actionability_feedback() -> None:
    from northcua.agents.web.prompt.history import WebHistoryToolResultProjector

    rendered = WebHistoryToolResultProjector().render_web_tool_output(
        {
            "status": "error",
            "action": "click_ref",
            "code": "element_disabled",
            "reason_code": "element_disabled",
            "concise_log_summary": "element is not enabled",
            "hint": "The target is disabled. Do not retry the same ref.",
            "retryable": True,
        }
    )

    assert "Tool feedback:" in rendered
    assert "- code: element_disabled" in rendered
    assert "- summary: element is not enabled" in rendered
    assert "Do not retry" in rendered


def test_web_wrong_element_action_errors_are_retryable_for_agent_recovery() -> None:
    classification = classify_web_tool_error(
        RuntimeError('Locator.check: Error: Not a checkbox or radio button - locator resolved to <input type="number">')
    )

    assert classification.code == "wrong_action_for_element"
    assert classification.retryable is True
    assert "fill_ref" in classification.hint


def test_web_disallowed_url_scheme_errors_are_retryable_for_agent_recovery() -> None:
    classification = classify_web_tool_error(RuntimeError("URL scheme is not allowed: view-source"))

    assert classification.code == "invalid_url_scheme"
    assert classification.retryable is True
    assert "http" in classification.hint
    assert "view-source" in classification.hint


def test_web_wrong_select_action_errors_are_retryable_for_agent_recovery() -> None:
    classification = classify_web_tool_error(
        RuntimeError("Locator.select_option: Error: Element is not a <select> element")
    )

    assert classification.code == "wrong_action_for_element"
    assert classification.retryable is True
    assert "matches the element role" in classification.hint


def test_web_wrong_fill_action_errors_are_retryable_for_agent_recovery() -> None:
    classification = classify_web_tool_error(
        RuntimeError("Locator.fill: Error: Element is not an <input>, <textarea> or [contenteditable] element")
    )

    assert classification.code == "wrong_action_for_element"
    assert classification.retryable is True
    assert "matches the element role" in classification.hint


def test_web_action_params_canonicalize_refs_and_validate_points_tabs() -> None:
    assert WebRefParams(ref=" e3 ").ref == "@e3"
    assert WebRefParams(ref=" 3 ").ref == "3"
    assert WebRefParams(ref=3).ref == "3"
    assert WebRefParams(ref=" @3 ").ref == "@3"
    assert WebSelectOptionRefParams(ref="@country", values=[" US "]).model_dump(mode="json") == {
        "ref": "@country",
        "values": [" US "],
    }
    assert WebDragRefParams(from_ref=3, to_ref="@dst").model_dump(mode="json") == {
        "from_ref": "3",
        "to_ref": "@dst",
    }
    assert WebClickPointParams(x=10, y=20, button=WebMouseButton.RIGHT).button == WebMouseButton.RIGHT
    assert WebDragPointParams(from_x=0, from_y=1.5, to_x=20, to_y=30).to_y == 30
    assert WebNewTabParams(label=" docs ").label == "docs"
    assert WebCloseTabParams(tab_ref=" t2 ").tab_ref == "t2"
    assert WebFocusTabParams(tab_ref="docs").tab_ref == "docs"
    assert WebRefsParams(refs=[" e3 ", "@4", 3, "@3", "e3"]).refs == ["@e3", "@4", "3", "@3"]

    with pytest.raises(ValidationError):
        WebRefParams(ref="@")
    with pytest.raises(ValidationError):
        WebRefsParams(refs=[])
    with pytest.raises(ValidationError):
        WebNoteParams(text="   ")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WebScreenshotParams.model_validate({"annotate": True, "full_page": False})
    with pytest.raises(ValidationError):
        WebSelectOptionRefParams(ref="country", values=["US", " "])
    with pytest.raises(ValidationError):
        WebClickPointParams(x=-1, y=0)
    with pytest.raises(ValidationError):
        WebFocusTabParams(tab_ref="bad tab")


def test_snapshot_normalizer_supports_ref_dicts() -> None:
    snapshot = WebSnapshotNormalizer().normalize(
        {
            "url": "https://example.com",
            "title": "Example",
            "snapshot": 'button "Submit" [ref=e1]',
            "refs": {"e1": {"role": "button", "name": "Submit"}},
        }
    )

    assert snapshot.url == "https://example.com"
    assert snapshot.snapshot == 'button "Submit" [ref=@e1]'
    assert snapshot.raw["snapshot"] == 'button "Submit" [ref=@e1]'
    assert snapshot.elements[0].model_ref == "@e1"
    assert snapshot.elements[0].role == "button"


def test_snapshot_normalizer_uses_origin_as_url_fallback() -> None:
    snapshot = WebSnapshotNormalizer().normalize(
        {
            "origin": "http://127.0.0.1:8123/index.html",
            "snapshot": "button @e1 Save",
            "refs": {"e1": {"role": "button", "name": "Save"}},
        }
    )
    observation = WebObservationContext(snapshot=snapshot)

    assert snapshot.url == "http://127.0.0.1:8123/index.html"
    assert observation.page_url == "http://127.0.0.1:8123/index.html"
    assert "127.0.0.1:8123" in snapshot.summary


def test_system_prompt_template_does_not_embed_tool_list() -> None:
    template_path = (
        Path(__file__).resolve().parents[2]
        / "northcua"
        / "agents"
        / "web"
        / "prompt"
        / "templates"
        / "web_agent_system.j2"
    )

    rendered = WebTemplateRenderer(template_path).render()

    assert "web_open" not in rendered
    assert "Do not return tool calls alone" in rendered
    assert "short non-empty user-facing assistant text" in rendered
    assert "finish with plain text only and no tool calls" in rendered
    assert "Default to one tool call per response" in rendered
    assert "read-only inspection" in rendered
    assert "This task has a finite step budget of" not in rendered
    assert "Task completion contract:" in rendered
    assert "answer_task" in rendered
    assert "state_task" in rendered
    assert "artifact_task" in rendered
    assert "Reasoning protocol for every step:" in rendered
    assert "Step budget is finite" in rendered
    assert "reserve enough remaining steps for final verification" in rendered
    assert "After any mutating browser action" in rendered
    assert "fresh post-observation" in rendered
    assert "does not expose local/device screenshot paths" in rendered
    assert "Step 1 has one leading user message containing the task plus the initial observation" in rendered
    assert "newest page state comes from the most recent retained tool result" in rendered
    assert "Explicit `web_snapshot` and annotated `web_screenshot` results return the newest usable refs" in rendered
    assert "Historical tool-result images may remain in the image window" in rendered
    assert "actionable refs are exposed only for the latest retained observation" in rendered
    assert "clean page screenshot: page-state pixels" in rendered
    assert "action-overlay screenshot: visual history showing where a previous action was attempted" in rendered
    assert "Set-of-Mark screenshot: the newest ref-grounding image" in rendered
    assert "snapshot refs were omitted" in rendered
    assert "settled=false" in rendered
    assert "Use the returned refs/SoM directly" in rendered
    assert "Older retained refs/images are historical reference" in rendered
    assert "Do not batch any mutating browser action" in rendered
    assert "Use viewport point actions only" in rendered
    assert "when a reliable ref exists" in rendered
    assert "no reliable ref for the target" in rendered
    assert "Use the note tool sparingly as explicit durable working memory" in rendered
    assert "Do not record routine reasoning or every next action" in rendered
    assert "compact numeric labels" in rendered
    assert "annotated screenshot" in rendered
    assert "direct numeric Set-of-Mark ref labels" in rendered
    assert "separate label map" in rendered
    assert "Do not invent element refs" in rendered
    assert "bbox metadata in the viewport CSS coordinate frame" in rendered
    assert "viewport top-left" in rendered
    assert "Screenshot image pixel dimensions may be reported in metadata" in rendered
    assert "Every screenshot captures only the current viewport" in rendered
    assert "Full-page capture is unavailable" in rendered
    assert "viewport by viewport with `web_scroll`" in rendered
    assert "retryable tool error" in rendered
    assert "For wait actions, the target parameters are mutually exclusive" in rendered
    assert "Specify exactly one wait target" in rendered
    assert "do not send null placeholders" in rendered


def test_system_prompt_template_includes_concrete_step_budget_when_available() -> None:
    template_path = (
        Path(__file__).resolve().parents[2]
        / "northcua"
        / "agents"
        / "web"
        / "prompt"
        / "templates"
        / "web_agent_system.j2"
    )

    rendered = WebTemplateRenderer(template_path).render(
        step_budget={
            "available": True,
            "round_limit": 20,
        }
    )

    assert "This task has a finite step budget of 20 model turns." in rendered
    assert "Step budget is finite. Use it deliberately" in rendered
    assert "remaining budget is low" in rendered


def test_web_tool_schemas_project_into_prompt_envelope_with_agent_friendly_descriptions() -> None:
    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder

    resolved = ToolSurfaceResolver(
        (*web_agent_action_contracts(), *web_agent_backend_contracts(), *web_agent_probe_contracts())
    ).resolve(ToolProfile(name="agent.web"))
    action_contracts = web_agent_action_contracts()
    assert {contract.timeout_seconds for contract in action_contracts} == {60.0}
    assert {contract.timeout_seconds for contract in web_agent_probe_contracts()} == {60.0}
    assert {contract.timeout_seconds for contract in web_agent_backend_contracts()} == {60.0}
    provider_surface = ToolSurfaceService().build_from_resolved_surface(resolved, api_type="openai_responses")
    tools_by_name = {str(tool["name"]): tool for tool in provider_surface.tools}

    assert provider_surface.tool_choice == "auto"
    assert len(provider_surface.tools) == 24
    assert "web_done" not in tools_by_name
    assert "web_agent_get_box" not in tools_by_name
    assert "web_agent_get_boxes" not in tools_by_name
    assert "web_agent_get_ref_states" not in tools_by_name
    assert "web_agent_viewport" not in tools_by_name
    assert not any(name.startswith("web_agent_backend_") for name in tools_by_name)
    assert {
        "web_note",
        "web_click_point",
        "web_drag_point",
        "web_select_option_ref",
        "web_screenshot",
        "web_focus_tab",
    }.issubset(tools_by_name)

    note = tools_by_name["web_note"]
    assert "explicit durable working memory" in str(note["description"])
    assert "Do not record routine reasoning or every next action" in str(note["description"])
    note_params = note["parameters"]
    assert "Durable, exact information to preserve" in note_params["properties"]["text"]["description"]
    assert "Use sparingly" in note_params["properties"]["text"]["description"]
    assert note_params["properties"]["kind"]["enum"] == [
        "fact",
        "mistake",
        "precaution",
        "plan",
        "constraint",
        "decision",
        "other",
    ]

    click_point = tools_by_name["web_click_point"]
    assert "viewport CSS coordinates" in str(click_point["description"])
    assert "no reliable element ref exists" in str(click_point["description"])
    assert "use the ref click action instead" in str(click_point["description"])
    assert "fresh post-action observation" in str(click_point["description"])
    assert "clean page image" in str(click_point["description"])
    assert "action-overlay history image" in str(click_point["description"])
    click_params = click_point["parameters"]
    assert "viewport CSS x coordinate" in click_params["properties"]["x"]["description"]
    assert "no reliable numeric ref" in click_params["properties"]["x"]["description"]
    assert "left is the normal primary click" in click_params["properties"]["button"]["description"]
    assert click_params["properties"]["button"]["enum"] == ["left", "right", "middle"]

    assert "Use this instead of viewport coordinates" in str(tools_by_name["web_click_ref"]["description"])
    assert "Use this for hover menus" in str(tools_by_name["web_hover_ref"]["description"])
    assert "instead of click/type sequences" in str(tools_by_name["web_fill_ref"]["description"])
    assert "instead of clicking option coordinates" in str(tools_by_name["web_select_option_ref"]["description"])
    assert "without toggling it off accidentally" in str(tools_by_name["web_check_ref"]["description"])
    assert "Use this before coordinate drag" in str(tools_by_name["web_drag_ref"]["description"])
    assert "Use this only when drag_ref is unavailable" in str(tools_by_name["web_drag_point"]["description"])
    drag_point_params = tools_by_name["web_drag_point"]["parameters"]
    for property_name in ("from_x", "from_y", "to_x", "to_y"):
        description = drag_point_params["properties"][property_name]["description"]
        assert "either endpoint lacks a reliable numeric ref" in description
        assert "bbox center" in description

    select_params = tools_by_name["web_select_option_ref"]["parameters"]
    assert "Numeric element ref" in select_params["properties"]["ref"]["description"]
    assert "for example 3" in select_params["properties"]["ref"]["description"]
    assert "fall back to viewport point actions only" in select_params["properties"]["ref"]["description"]
    assert "HTML option value strings" in select_params["properties"]["values"]["description"]
    assert (
        "Prefer this ref action over clicking dropdown coordinates"
        in select_params["properties"]["values"]["description"]
    )

    wait_tool = tools_by_name["web_wait"]
    assert "mutually exclusive target" in str(wait_tool["description"])
    assert "Choose exactly one of" in str(wait_tool["description"])
    assert "Do not combine wait fields" in str(wait_tool["description"])
    assert "do not provide null or invented placeholder values" in str(wait_tool["description"])
    wait_params = wait_tool["parameters"]
    assert wait_tool["strict"] is False
    assert len(wait_params["oneOf"]) == 5
    assert [branch["required"] for branch in wait_params["oneOf"]] == [
        ["duration_ms"],
        ["load_state"],
        ["text"],
        ["url_pattern"],
        ["selector"],
    ]
    assert all(len(branch["properties"]) == 1 for branch in wait_params["oneOf"])
    assert "Mutually exclusive web_wait arguments" in str(wait_params.get("description", ""))
    assert wait_params["additionalProperties"] is False
    for property_name in ("duration_ms", "load_state", "text", "url_pattern", "selector"):
        assert "Mutually exclusive" in wait_params["properties"][property_name]["description"]

    screenshot_params = tools_by_name["web_screenshot"]["parameters"]
    assert "direct numeric ref labels" in str(tools_by_name["web_screenshot"]["description"])
    assert "annotate=false returns a clean screenshot" in str(tools_by_name["web_screenshot"]["description"])
    assert "annotate=true returns a current-viewport Set-of-Mark screenshot" in str(
        tools_by_name["web_screenshot"]["description"]
    )
    assert "Set-of-Mark" in screenshot_params["properties"]["annotate"]["description"]
    assert "When false, capture a clean screenshot" in screenshot_params["properties"]["annotate"]["description"]
    assert "does not expose screenshot paths" in str(tools_by_name["web_screenshot"]["description"])
    assert "visible numeric label directly" in screenshot_params["properties"]["annotate"]["description"]
    assert set(screenshot_params["properties"]) == {"annotate"}
    assert screenshot_params.get("additionalProperties") is False
    assert "current-viewport visual inspection screenshot" in str(tools_by_name["web_screenshot"]["description"])
    assert "viewport by viewport with web_scroll" in str(tools_by_name["web_screenshot"]["description"])
    assert "direct numeric ref labels" in screenshot_params["description"]
    assert "element refs" not in screenshot_params["description"]
    provider_surface_text = json.dumps(provider_surface.tools, sort_keys=True)
    assert "Playwright" not in provider_surface_text
    assert "substrate" not in provider_surface_text
    assert "screenshot_url" not in provider_surface_text
    assert "data_url" not in provider_surface_text
    focus_params = tools_by_name["web_focus_tab"]["parameters"]
    assert "Stable tab id or label" in focus_params["properties"]["tab_ref"]["description"]

    envelope = WebAgentPromptBuilder().build(
        user_query="Use the page",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=[],
        tools=provider_surface.tools,
        tool_choice=provider_surface.tool_choice,
    )

    assert envelope.tools == provider_surface.tools
    assert envelope.tool_choice == "auto"
    assert any(tool["name"] == "web_click_point" for tool in envelope.tools)


def test_observation_prompt_omits_screenshot_when_not_in_prompt() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder

    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            snapshot="button @e1 Submit",
            elements=[WebElementRef(ref_id="e1", role="button", name="Submit")],
        )
    )

    envelope = WebAgentPromptBuilder(history_window_rounds=1).build(
        user_query="Click submit",
        observation=observation,
        prior_messages=[],
        tools=[],
        tool_choice=None,
    )

    rendered_text = "\n".join(
        block.text for message in envelope.messages for block in message.content if block.type == "text"
    )
    assert '- 1 button "Submit"' in rendered_text
    assert "Screenshot artifact" not in rendered_text

    envelope_with_history = WebAgentPromptBuilder(history_window_rounds=1).build(
        user_query="Never click save",
        observation=observation,
        prior_messages=[Message(role=Role.USER, content=[TextBlock(text="Tool result from web_get_url:\nok")])],
        tools=[],
        tool_choice=None,
    )
    rendered_history_text = "\n".join(
        block.text for message in envelope_with_history.messages for block in message.content if block.type == "text"
    )
    assert "Task:\nNever click save" in rendered_history_text


def test_observation_prompt_uses_numeric_ref_som_labels() -> None:
    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder

    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            snapshot='link "Pricing" [ref=@e194]\nlink "Docs" [ref=@e195]',
            elements=[
                WebElementRef(ref_id="e194", role="link", name="Pricing"),
                WebElementRef(ref_id="e195", role="link", name="Docs"),
            ],
        ),
        screenshot=WebScreenshotObservation(
            data_url="data:image/png;base64,QUJD",
            annotation=WebScreenshotAnnotation(
                marks=[
                    WebScreenshotMark(
                        ref="1",
                        label="1",
                        source_ref="@e194",
                        role="link",
                        name="Pricing",
                        bbox_css=WebBoundingBox(x=12, y=20, width=90, height=24),
                        bbox_image=WebBoundingBox(x=12, y=20, width=90, height=24),
                    ),
                    WebScreenshotMark(
                        ref="2",
                        label="2",
                        source_ref="@e195",
                        role="link",
                        name="Docs",
                        bbox_css=WebBoundingBox(x=112, y=20, width=70, height=24),
                        bbox_image=WebBoundingBox(x=112, y=20, width=70, height=24),
                    ),
                ],
                omitted_count=3,
            ),
        ),
    )

    envelope = WebAgentPromptBuilder(screenshot_window_images=1).build(
        user_query="Click pricing",
        observation=observation,
        prior_messages=[],
        tools=[],
        tool_choice=None,
    )
    rendered_text = "\n".join(
        block.text for message in envelope.messages for block in message.content if block.type == "text"
    )

    assert "SoM marks are numeric refs" in rendered_text
    assert "Actionable refs (2):" in rendered_text
    assert "Snapshot:" not in rendered_text
    assert "Set-of-Mark refs" not in rendered_text
    assert '- 1 link "Pricing"' in rendered_text
    assert '- 2 link "Docs"' in rendered_text
    assert 'Pricing"\n- 2 link "Docs"' in rendered_text
    assert "link- 2" not in rendered_text
    assert "@e194" not in rendered_text
    assert "@e195" not in rendered_text
    assert "omitted candidates: 3" in rendered_text
    assert any(
        getattr(block, "type", None) == "image" and getattr(block, "url", "") == "data:image/png;base64,QUJD"
        for message in envelope.messages
        for block in message.content
    )


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
@pytest.mark.parametrize("append_history", [False, True])
def test_initial_observation_attaches_clean_and_som_images(tmp_path: Path, budget: int, append_history: bool) -> None:
    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder
    from northcua.core.image_payloads import image_data_url_from_path

    clean_path = tmp_path / "initial_clean.png"
    som_path = tmp_path / "initial_som.png"
    clean_path.write_bytes(b"clean")
    som_path.write_bytes(b"som")
    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            elements=[WebElementRef(ref_id="e1", role="button", name="Buy")],
        ),
        screenshot=WebScreenshotObservation(
            path=str(som_path),
            data_url=image_data_url_from_path(str(som_path)),
            annotation=WebScreenshotAnnotation(
                marks=[
                    WebScreenshotMark(
                        ref="1",
                        label="1",
                        source_ref="@e1",
                        role="button",
                        name="Buy",
                        bbox_css=WebBoundingBox(x=1, y=2, width=30, height=12),
                        bbox_image=WebBoundingBox(x=1, y=2, width=30, height=12),
                    )
                ],
                source_screenshot_path=str(clean_path),
                overlay_screenshot_path=str(som_path),
            ),
        ),
    )

    from nexau.core.messages import Message, Role, ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock

    history = (
        [
            Message(role=Role.USER, content=[TextBlock(text="Task: test")]),
            Message(role=Role.ASSISTANT, content=[ToolUseBlock(id="prior", name="web_screenshot", input={})]),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_use_id="prior", content=[ImageBlock(url="data:image/png;base64,b2xk")])],
            ),
        ]
        if append_history
        else []
    )
    envelope = WebAgentPromptBuilder(
        history_strategy="append" if append_history else "desktop",
        screenshot_window_images=budget,
        ref_snapshot_window_observations=1,
    ).build(
        user_query="Buy the item",
        observation=observation,
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )

    image_urls = _top_level_image_urls(envelope.messages)
    assert image_urls == [image_data_url_from_path(str(clean_path)), image_data_url_from_path(str(som_path))][:budget]
    images = [block for message in envelope.messages for block in message.content if isinstance(block, ImageBlock)]
    images += [
        item
        for message in envelope.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and isinstance(block.content, list)
        for item in block.content
        if isinstance(item, ImageBlock)
    ]
    assert len(images) <= budget
    leading_text = "\n".join(
        block.text for message in envelope.messages for block in message.content if isinstance(block, TextBlock)
    )
    assert ("Current clean page screenshot." in leading_text) == (budget > 0)
    assert ("Current Set-of-Mark image for actionable refs in this observation." in leading_text) == (budget >= 2)


def test_initial_observation_does_not_attach_som_when_refs_disabled(tmp_path: Path) -> None:
    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder
    from northcua.core.image_payloads import image_data_url_from_path

    clean_path = tmp_path / "initial_clean.png"
    som_path = tmp_path / "initial_som.png"
    clean_path.write_bytes(b"clean")
    som_path.write_bytes(b"som")
    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            elements=[WebElementRef(ref_id="e1", role="button", name="Buy")],
        ),
        screenshot=WebScreenshotObservation(
            path=str(som_path),
            data_url=image_data_url_from_path(str(som_path)),
            annotation=WebScreenshotAnnotation(
                marks=[
                    WebScreenshotMark(
                        ref="1",
                        label="1",
                        source_ref="@e1",
                        role="button",
                        name="Buy",
                        bbox_css=WebBoundingBox(x=1, y=2, width=30, height=12),
                        bbox_image=WebBoundingBox(x=1, y=2, width=30, height=12),
                    )
                ],
                source_screenshot_path=str(clean_path),
                overlay_screenshot_path=str(som_path),
            ),
        ),
    )

    envelope = WebAgentPromptBuilder(screenshot_window_images=1, ref_snapshot_window_observations=0).build(
        user_query="Buy the item",
        observation=observation,
        prior_messages=[],
        tools=[],
        tool_choice=None,
    )

    assert _top_level_image_urls(envelope.messages) == [image_data_url_from_path(str(clean_path))]
    assert "Current Set-of-Mark image for actionable refs in this observation." not in _leading_user_text(
        envelope.messages
    )


def test_observation_projector_filters_snapshot_text_by_exact_ref_token() -> None:
    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation

    snapshot = WebSnapshotObservation(
        url="https://example.com",
        title="Example",
        snapshot="\n".join(
            [
                'link "Wrong larger ref" [ref=@e10]',
                'button "Target mentions [ref=@e2]" [ref=@e1]',
            ]
        ),
        elements=[
            WebElementRef(ref_id="e1", role="button", name="Target"),
            WebElementRef(ref_id="e2", role="button", name="Unmarked"),
            WebElementRef(ref_id="e10", role="link", name="Wrong larger ref"),
        ],
    )
    annotation = WebScreenshotAnnotation(
        marks=[
            WebScreenshotMark(
                ref="1",
                label="1",
                source_ref="@e1",
                role="button",
                name="Target",
                bbox_css=WebBoundingBox(x=1, y=2, width=30, height=12),
                bbox_image=WebBoundingBox(x=1, y=2, width=30, height=12),
            )
        ]
    )

    text = WebObservationProjector().snapshot_text(snapshot, annotation=annotation)

    assert '1 button "Target"' in text
    assert "@e1" not in text
    assert "@e2" not in text
    assert "@e10" not in text
    assert "Wrong larger ref" not in text


def test_observation_projector_keeps_numeric_model_visible_som_labels() -> None:
    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation

    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            snapshot='button "Target" [ref=@e1]',
            elements=[WebElementRef(ref_id="e1", role="button", name="Target")],
        ),
        screenshot=WebScreenshotObservation(
            data_url="data:image/png;base64,QUJD",
            annotation=WebScreenshotAnnotation(
                marks=[
                    WebScreenshotMark(
                        ref="1",
                        label="1",
                        source_ref="@e1",
                        role="button",
                        name="Target",
                        bbox_css=WebBoundingBox(x=1, y=2, width=30, height=12),
                        bbox_image=WebBoundingBox(x=1, y=2, width=30, height=12),
                    )
                ],
            ),
        ),
    )

    payload = WebObservationProjector().bundle_dict(observation)

    refs = payload["snapshot"]["refs"]
    marks = payload["screenshot"]["annotation"]["marks"]
    assert refs == [{"ref": "1", "role": "button", "label": "Target"}]
    assert marks == [
        {
            "ref": "1",
            "label": "1",
            "role": "button",
            "name": "Target",
            "bbox_css": {"x": 1.0, "y": 2.0, "width": 30.0, "height": 12.0},
            "bbox_image": {"x": 1.0, "y": 2.0, "width": 30.0, "height": 12.0},
            "visible": True,
            "enabled": True,
        }
    ]


def test_ref_alias_service_maps_numeric_model_refs_to_source_refs() -> None:
    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation
    from northcua.agents.web.observation.ref_alias import WebRefAliasService

    snapshot = WebSnapshotObservation(
        elements=[
            WebElementRef(ref_id="e39", role="link", name="Shop"),
            WebElementRef(ref_id="e194", role="button", name="Pricing"),
        ]
    )
    aliases = WebRefAliasService().aliases_for_observation(snapshot, observation_id="obs-1")

    assert aliases.observation_id == "obs-1"
    assert aliases.alias_to_source == {"1": "@e39", "2": "@e194"}
    assert aliases.source_to_alias == {"@e39": "1", "@e194": "2"}


def test_unannotated_snapshot_projection_filters_definitely_non_actionable_refs() -> None:
    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation
    from northcua.agents.web.observation.projector import WebObservationProjector
    from northcua.agents.web.observation.ref_alias import WebRefAliasService

    snapshot = WebSnapshotObservation(
        elements=[
            WebElementRef(ref_id="e1", role="button", name="Submit"),
            WebElementRef(ref_id="e2", role="button", name="Hidden", attributes={"aria-hidden": "true"}),
            WebElementRef(ref_id="e3", role="button", name="Disabled", attributes={"disabled": ""}),
            WebElementRef(ref_id="e4", role="link", name="Docs"),
            WebElementRef(ref_id="e5", role="button", name="Runtime disabled", attributes={"enabled": False}),
        ]
    )

    aliases = WebRefAliasService().aliases_for_observation(snapshot)
    refs = WebObservationProjector().snapshot_refs(snapshot)
    text = WebObservationProjector().snapshot_text(snapshot)

    assert aliases.alias_to_source == {"1": "@e1", "2": "@e4"}
    assert [ref["ref"] for ref in refs] == ["1", "2"]
    assert [ref["label"] for ref in refs] == ["Submit", "Docs"]
    assert '1 button "Submit"' in text
    assert '2 link "Docs"' in text
    assert "Hidden" not in text
    assert "Disabled" not in text


def test_unannotated_snapshot_projection_does_not_fallback_to_raw_filtered_refs() -> None:
    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation
    from northcua.agents.web.observation.projector import WebObservationProjector

    snapshot = WebSnapshotObservation(
        snapshot='button "Disabled" [ref=@e7]',
        elements=[
            WebElementRef(ref_id="e7", role="button", name="Disabled", attributes={"disabled": ""}),
        ],
    )

    projector = WebObservationProjector()
    assert projector.snapshot_refs(snapshot) == []
    assert projector.snapshot_text(snapshot) == ""


def test_snapshot_refs_include_form_state_when_refs_table_is_rendered() -> None:
    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation
    from northcua.agents.web.observation.projector import WebObservationProjector

    snapshot = WebSnapshotObservation(
        elements=[
            WebElementRef(ref_id="e1", role="textbox", name="Full name", attributes={"value": "Ada Lovelace"}),
            WebElementRef(ref_id="e2", role="checkbox", name="Terms", attributes={"checked": True}),
        ],
    )

    refs = WebObservationProjector().snapshot_refs(snapshot)
    assert refs == [
        {"ref": "1", "role": "textbox", "label": "Full name", "state": 'value="Ada Lovelace"'},
        {"ref": "2", "role": "checkbox", "label": "Terms", "state": "checked=true"},
    ]


def test_empty_som_annotation_does_not_fallback_to_unseen_aliases() -> None:
    from northcua.agents.web.domain.observations import (
        WebElementRef,
        WebScreenshotAnnotation,
        WebSnapshotObservation,
    )
    from northcua.agents.web.observation.ref_alias import WebRefAliasService

    snapshot = WebSnapshotObservation(
        elements=[
            WebElementRef(ref_id="e1", role="button", name="Hidden Alias"),
        ]
    )

    aliases = WebRefAliasService().aliases_for_observation(snapshot, annotation=WebScreenshotAnnotation())

    assert aliases.aliases == []


def test_context_from_payload_preserves_ref_state_for_history_projection() -> None:
    from northcua.agents.web.observation.projector import WebObservationProjector

    projector = WebObservationProjector()
    context = projector.context_from_payload(
        {
            "observation_id": "obs-form",
            "page": {"url": "https://example.com/form", "title": "Form"},
            "snapshot": {
                "text": "",
                "ref_count": 2,
                "refs": [
                    {"ref": "1", "role": "textbox", "label": "Full name", "state": 'value="Ada Lovelace"'},
                    {"ref": "2", "role": "checkbox", "label": "Terms", "state": "checked=true"},
                ],
            },
            "screenshot": {"kind": "none", "image_attached": False},
        }
    )

    assert context is not None
    assert projector.snapshot_refs(context.snapshot) == [
        {"ref": "1", "role": "textbox", "label": "Full name", "state": 'value="Ada Lovelace"'},
        {"ref": "2", "role": "checkbox", "label": "Terms", "state": "checked=true"},
    ]


def test_context_from_payload_preserves_ref_state_for_annotated_history_projection() -> None:
    from northcua.agents.web.observation.projector import WebObservationProjector

    projector = WebObservationProjector()
    box = {"x": 1, "y": 2, "width": 30, "height": 12}
    context = projector.context_from_payload(
        {
            "observation_id": "obs-form-som",
            "page": {"url": "https://example.com/form", "title": "Form"},
            "snapshot": {
                "text": "",
                "ref_count": 1,
                "refs": [
                    {"ref": "1", "role": "textbox", "label": "Full name", "state": 'value="Ada Lovelace"'},
                ],
            },
            "screenshot": {
                "kind": "set_of_mark",
                "image_attached": True,
                "annotation": {
                    "kind": "set_of_mark",
                    "marks": [
                        {
                            "ref": "1",
                            "label": "1",
                            "role": "textbox",
                            "name": "Full name",
                            "bbox_css": box,
                            "bbox_image": box,
                            "visible": True,
                            "enabled": True,
                        }
                    ],
                    "marked_count": 1,
                    "visible_interactive_count": 1,
                    "max_marks": 150,
                    "max_box_lookup": 512,
                },
            },
        },
        screenshot_url="data:image/png;base64,abc",
    )

    assert context is not None
    assert context.screenshot is not None
    assert context.screenshot.annotation is not None
    assert projector.snapshot_refs(context.snapshot, annotation=context.screenshot.annotation) == [
        {"ref": "1", "role": "textbox", "label": "Full name", "state": 'value="Ada Lovelace"'},
    ]
    assert projector.snapshot_text(context.snapshot, annotation=context.screenshot.annotation) == (
        '1 textbox "Full name" value="Ada Lovelace"'
    )


def test_web_prompt_desktop_strategy_uses_single_leading_user_then_history() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(history_window_rounds=2)
    state = TraceCaptureState()
    state.assistant_tool_calls.append(AssistantToolCallRecord(id="call-1", name="web_get_url", arguments="{}"))
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-1",
            tool_name="web_get_url",
            tool_output={"status": "ok", "result": {"url": "https://example.com"}},
            llm_tool_output={"status": "ok"},
        )
    )
    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=[Message(role=Role.USER, content=[TextBlock(text="old obs")])]),
        prior_messages=[],
        trace_state=state,
    )
    envelope = builder.build(
        user_query="Inspect only",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )

    assert envelope.messages[0].role == Role.USER
    assert "Task:\nInspect only" in envelope.messages[0].content[0].text
    assert envelope.messages[1].role == Role.ASSISTANT
    assert envelope.messages[1].content[0].type == "tool_use"
    assert envelope.messages[2].role == Role.TOOL
    assert envelope.messages[2].content[0].type == "tool_result"
    assert envelope.messages[-1].role == Role.TOOL
    assert [message.role for message in envelope.messages].count(Role.USER) == 1


def test_web_prompt_append_strategy_keeps_prior_messages_before_current_observation() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder

    builder = WebAgentPromptBuilder(history_window_rounds=2, history_strategy="append")
    prior = [Message(role=Role.USER, content=[TextBlock(text="previous observation")])]
    envelope = builder.build(
        user_query="Inspect only",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=prior,
        tools=[],
        tool_choice=None,
    )

    assert envelope.messages[0].content[0].text == "previous observation"
    assert "Latest page observation:" in envelope.messages[-1].content[0].text


def test_web_prompt_append_strategy_strips_initial_top_level_image_from_history() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder

    builder = WebAgentPromptBuilder(history_strategy="append", screenshot_window_images=2)
    initial_envelope = builder.build(
        user_query="Inspect",
        observation=WebObservationContext(
            snapshot=WebSnapshotObservation(url="https://example.com", snapshot="button @e1"),
            screenshot=WebScreenshotObservation(data_url="data:image/png;base64,SU5JVElBTA=="),
            observation_id="initial",
        ),
        prior_messages=[],
        tools=[],
        tool_choice=None,
    )
    state = TraceCaptureState()
    state.assistant_tool_calls.append(AssistantToolCallRecord(id="call-url", name="web_get_url", arguments="{}"))
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-url",
            tool_name="web_get_url",
            tool_output={"status": "ok", "text": "Read URL.", "result": {"url": "https://example.com"}},
        )
    )
    history = builder.append_step_history(
        envelope=initial_envelope,
        prior_messages=[Message(role=Role.USER, content=[TextBlock(text="unused")])],
        trace_state=state,
    )

    step_two_envelope = builder.build(
        user_query="Inspect",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
        allow_current_observation_image=False,
    )
    image_blocks = [
        block
        for message in step_two_envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "image"
    ]
    rendered_text = "\n".join(
        block.text for message in step_two_envelope.messages for block in message.content if block.type == "text"
    )

    assert image_blocks == []
    assert "top-level observation image omitted from this replay turn" in rendered_text
    assert "subsequent visual state is carried by tool results" in rendered_text


def test_web_prompt_does_not_duplicate_recent_action_result_in_user_observation() -> None:
    from northcua.agents.web.domain.observations import WebObservationContext, WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder

    builder = WebAgentPromptBuilder()
    envelope = builder.build(
        user_query="Inspect only",
        observation=WebObservationContext(
            snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body"),
            recent_action_result={"status": "ok", "action": "snapshot"},
        ),
        prior_messages=[],
        tools=[],
        tool_choice=None,
    )

    observation_text = _leading_user_text(envelope.messages)
    assert "Initial page observation:" in observation_text
    assert "Recent action result:" not in observation_text
    assert '"status": "ok"' not in observation_text


def test_note_tool_result_replays_as_compact_text_without_full_payload_json() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(history_window_rounds=2)
    state = TraceCaptureState()
    state.assistant_tool_calls.append(
        AssistantToolCallRecord(
            id="call-note",
            name="web_note",
            arguments='{"text": "Remember Ada.", "reason": "Use later.", "kind": "fact"}',
        )
    )
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-note",
            tool_name="web_note",
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

    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=[Message(role=Role.USER, content=[TextBlock(text="old obs")])]),
        prior_messages=[],
        trace_state=state,
    )

    tool_message = next(message for message in history if message.role == Role.TOOL)
    tool_result = tool_message.content[0]
    assert tool_result.type == "tool_result"
    assert tool_result.content == "[fact] Remember Ada.\nReason: Use later."
    assert "note_text" not in tool_result.content
    assert "Use later" in tool_result.content


def test_screenshot_tool_result_replays_as_image_block_without_base64_text() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    data_url = "data:image/png;base64,QUJDREVGRw=="
    builder = WebAgentPromptBuilder(history_window_rounds=2)
    state = TraceCaptureState()
    state.assistant_tool_calls.append(
        AssistantToolCallRecord(id="call-shot", name="web_screenshot", arguments='{"annotate": true}')
    )
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-shot",
            tool_name="web_screenshot",
            tool_output={
                "status": "ok",
                "action": "screenshot",
                "text": "Captured screenshot.",
                "screenshot_url": data_url,
                "result": {
                    "data_url": data_url,
                    "annotation": {
                        "kind": "set_of_mark",
                        "marks": [
                            {
                                "label": "1",
                                "ref": "@e194",
                                "role": "link",
                                "name": "Pricing",
                                "bbox_css": {"x": 12, "y": 20, "width": 90, "height": 24},
                                "bbox_image": {"x": 12, "y": 20, "width": 90, "height": 24},
                            }
                        ],
                    },
                },
            },
            llm_tool_output={
                "status": "ok",
                "action": "screenshot",
                "text": "Captured screenshot.",
                "screenshot_url": data_url,
                "result": {
                    "data_url": data_url,
                    "annotation": {
                        "kind": "set_of_mark",
                        "marks": [
                            {
                                "label": "1",
                                "ref": "@e194",
                                "role": "link",
                                "name": "Pricing",
                                "bbox_css": {"x": 12, "y": 20, "width": 90, "height": 24},
                                "bbox_image": {"x": 12, "y": 20, "width": 90, "height": 24},
                            }
                        ],
                    },
                },
            },
        )
    )

    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=[Message(role=Role.USER, content=[TextBlock(text="old obs")])]),
        prior_messages=[],
        trace_state=state,
    )

    tool_message = next(message for message in history if message.role == Role.TOOL)
    tool_result = tool_message.content[0]
    assert tool_result.type == "tool_result"
    assert isinstance(tool_result.content, list)
    assert any(
        getattr(block, "type", None) == "image" and getattr(block, "url", "") == data_url
        for block in tool_result.content
    )
    text = "\n".join(
        getattr(block, "text", "") for block in tool_result.content if getattr(block, "type", None) == "text"
    )
    assert "Captured screenshot." in text
    assert '"ref"' not in text
    assert "annotation" not in text
    assert "bbox_css" not in text
    assert "QUJDREVGRw" not in text
    assert "base64" not in text


def test_annotated_web_screenshot_replays_standard_observation_refs() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    data_url = "data:image/png;base64,QUJDREVGRw=="
    observation = {
        "observation_id": "screenshot-tool",
        "settled": True,
        "settle_reason": "stable",
        "page": {"url": "https://example.com/pricing", "title": "Pricing"},
        "snapshot": {
            "text": '7 link "Pricing"',
            "refs": [{"ref": "7", "role": "link", "label": "Pricing"}],
            "ref_count": 1,
        },
        "screenshot": {
            "kind": "set_of_mark",
            "image_attached": True,
            "annotation": {
                "kind": "set_of_mark",
                "marks": [
                    {
                        "label": "7",
                        "ref": "7",
                        "role": "link",
                        "name": "Pricing",
                        "bbox_css": {"x": 12, "y": 20, "width": 90, "height": 24},
                        "bbox_image": {"x": 12, "y": 20, "width": 90, "height": 24},
                    }
                ],
                "marked_count": 1,
                "visible_interactive_count": 1,
                "omitted_count": 0,
            },
        },
    }
    output = {
        "status": "ok",
        "action": "screenshot",
        "text": "Captured annotated viewport screenshot.",
        "screenshot_url": data_url,
        "result": {
            "kind": "set_of_mark",
            "image_attached": True,
            "annotation": observation["screenshot"]["annotation"],
        },
        "observation": observation,
    }
    builder = WebAgentPromptBuilder(
        history_window_rounds=2,
        screenshot_window_images=1,
        ref_snapshot_window_observations=1,
    )
    state = TraceCaptureState()
    state.assistant_tool_calls.append(
        AssistantToolCallRecord(id="call-annotated-shot", name="web_screenshot", arguments='{"annotate": true}')
    )
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-annotated-shot",
            tool_name="web_screenshot",
            tool_output=output,
            llm_tool_output=output,
        )
    )
    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=[Message(role=Role.USER, content=[TextBlock(text="old obs")])]),
        prior_messages=[],
        trace_state=state,
    )

    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )
    rendered_tool_text = _rendered_tool_text(envelope.messages)

    assert _tool_result_image_urls(envelope.messages) == [data_url]
    assert "Captured annotated viewport screenshot." in rendered_tool_text
    assert "Post-action page observation:" in rendered_tool_text
    assert "Actionable refs (1):" in rendered_tool_text
    assert '7 link "Pricing"' in rendered_tool_text
    assert "bbox_css" not in rendered_tool_text
    assert '"annotation"' not in rendered_tool_text


def test_web_screenshot_tool_result_uses_same_image_window_budget() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(history_window_rounds=6, screenshot_window_images=1)
    history: list[Message] = []

    def append_result(
        *,
        call_id: str,
        tool_name: str,
        arguments: str,
        output: dict[str, object],
    ) -> None:
        nonlocal history
        state = TraceCaptureState()
        state.assistant_tool_calls.append(AssistantToolCallRecord(id=call_id, name=tool_name, arguments=arguments))
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id=call_id,
                tool_name=tool_name,
                tool_output=output,
                llm_tool_output=output,
            )
        )
        history = builder.append_step_history(
            envelope=WebAgentPromptEnvelope(messages=history),
            prior_messages=history,
            trace_state=state,
        )

    post_action_url = "data:image/png;base64,cG9zdC1hY3Rpb24="
    explicit_screenshot_url = "data:image/png;base64,d2ViLXNjcmVlbnNob3Q="
    append_result(
        call_id="call-click",
        tool_name="web_click_ref",
        arguments='{"ref":"@e1"}',
        output={
            "status": "ok",
            "action": "click_ref",
            "text": "Clicked.",
            "screenshot_url": post_action_url,
            "observation": {
                "observation_id": "post-click",
                "settled": True,
                "settle_reason": "stable",
                "page": {"url": "https://example.com", "title": "Example"},
                "snapshot": {"text": "button @e1", "refs": [], "ref_count": 0},
                "screenshot": {"kind": "set_of_mark", "image_attached": True, "annotation": {}},
            },
        },
    )
    append_result(
        call_id="call-screenshot",
        tool_name="web_screenshot",
        arguments='{"annotate":true}',
        output={
            "status": "ok",
            "action": "screenshot",
            "text": "Captured annotated viewport screenshot.",
            "screenshot_url": explicit_screenshot_url,
            "result": {
                "annotation": {
                    "kind": "set_of_mark",
                    "marks": [{"label": "1", "ref": "@e1", "bbox_css": {"x": 1, "y": 2, "width": 3, "height": 4}}],
                }
            },
        },
    )

    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
        allow_current_observation_image=False,
    )
    retained_images = [
        getattr(content_block, "url", "")
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "tool_result" and isinstance(block.content, list)
        for content_block in block.content
        if getattr(content_block, "type", None) == "image"
    ]
    rendered_tool_text = "\n".join(
        str(content_block.text)
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "tool_result" and isinstance(block.content, list)
        for content_block in block.content
        if getattr(content_block, "type", None) == "text"
    )

    assert retained_images == [explicit_screenshot_url]
    assert post_action_url not in retained_images
    assert "Captured annotated viewport screenshot." in rendered_tool_text
    assert '"ref"' not in rendered_tool_text
    assert '"annotation"' not in rendered_tool_text
    assert "bbox_css" not in rendered_tool_text
    assert "image omitted from this replay turn" in rendered_tool_text


def test_web_prompt_five_round_mixed_tool_matrix_uses_desktop_like_image_and_ref_windows() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(
        history_window_rounds=20,
        screenshot_window_images=3,
        ref_snapshot_window_observations=3,
    )
    history: list[Message] = []
    task = "exercise window matrix"
    initial_image = "data:image/png;base64,aW5pdGlhbA=="

    initial_envelope = builder.build(
        user_query=task,
        observation=WebObservationContext(
            snapshot=WebSnapshotObservation(url="https://example.com", snapshot="initial body"),
            screenshot=WebScreenshotObservation(data_url=initial_image),
            observation_id="matrix-initial",
        ),
        prior_messages=[],
        tools=[],
        tool_choice=None,
        allow_current_observation_image=True,
    )
    assert _top_level_image_urls(initial_envelope.messages) == [initial_image]
    assert _tool_result_image_urls(initial_envelope.messages) == []
    history = initial_envelope.messages

    image_urls_by_step = {
        1: "data:image/png;base64,bWF0cml4LWltYWdlLTE=",
        3: "data:image/png;base64,bWF0cml4LWltYWdlLTM=",
        4: "data:image/png;base64,bWF0cml4LWltYWdlLTQ=",
        5: "data:image/png;base64,bWF0cml4LWltYWdlLTU=",
    }

    def observation_payload(index: int) -> dict[str, object]:
        return {
            "observation_id": f"matrix-post-{index}",
            "settled": True,
            "settle_reason": "stable",
            "page": {"url": f"https://example.com/step-{index}", "title": f"Step {index}"},
            "snapshot": {
                "text": f'{index} button "Matrix {index}"',
                "refs": [{"ref": str(index), "role": "button", "label": f"Matrix {index}"}],
                "ref_count": 1,
            },
            "screenshot": {
                "kind": "set_of_mark",
                "image_attached": index in image_urls_by_step,
                "annotation": {"marks": [{"label": str(index), "ref": str(index)}]},
            },
        }

    step_executions: list[list[tuple[str, str, dict[str, object]]]] = [
        [
            (
                "web_click_ref",
                '{"ref":"@e1"}',
                {
                    "status": "ok",
                    "action": "click_ref",
                    "text": "Matrix click 1.",
                    "screenshot_url": image_urls_by_step[1],
                    "observation": observation_payload(1),
                },
            )
        ],
        [
            (
                "web_get_url",
                "{}",
                {
                    "status": "ok",
                    "action": "get_url",
                    "text": "Matrix read-only URL with no image or refs.",
                    "result": {"url": "https://example.com/after-1"},
                },
            ),
            (
                "web_wait",
                '{"duration_ms":1}',
                {
                    "status": "ok",
                    "action": "wait",
                    "text": "Matrix wait 2 without image.",
                    "observation": observation_payload(2),
                },
            ),
        ],
        [
            (
                "web_screenshot",
                '{"annotate":true}',
                {
                    "status": "ok",
                    "action": "screenshot",
                    "text": "Matrix explicit screenshot 3.",
                    "screenshot_url": image_urls_by_step[3],
                    "result": {
                        "annotation": {
                            "kind": "set_of_mark",
                            "marks": [
                                {
                                    "label": "3",
                                    "ref": "3",
                                    "bbox_css": {"x": 3, "y": 4, "width": 30, "height": 12},
                                }
                            ],
                        }
                    },
                },
            )
        ],
        [
            (
                "web_scroll",
                '{"direction":"down","amount":600}',
                {
                    "status": "ok",
                    "action": "scroll",
                    "text": "Matrix scroll 4.",
                    "screenshot_url": image_urls_by_step[4],
                    "observation": observation_payload(4),
                },
            )
        ],
        [
            (
                "web_click_point",
                '{"x":10,"y":20}',
                {
                    "status": "ok",
                    "action": "click_point",
                    "text": "Matrix point click 5.",
                    "screenshot_url": image_urls_by_step[5],
                    "observation": observation_payload(5),
                },
            )
        ],
    ]

    seen_image_urls: list[str] = []
    seen_snapshot_indices: list[int] = []
    for step_index, executions in enumerate(step_executions, start=1):
        state = TraceCaptureState()
        for execution_index, (tool_name, arguments, output) in enumerate(executions, start=1):
            call_id = f"call-{step_index}-{execution_index}"
            state.assistant_tool_calls.append(AssistantToolCallRecord(id=call_id, name=tool_name, arguments=arguments))
            state.tool_executions.append(
                ToolExecutionRecord(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    tool_output=output,
                    llm_tool_output=output,
                )
            )
            image_url = output.get("screenshot_url")
            if isinstance(image_url, str):
                seen_image_urls.append(image_url)
            if isinstance(output.get("observation"), dict):
                seen_snapshot_indices.append(step_index)

        history = builder.append_step_history(
            envelope=WebAgentPromptEnvelope(messages=history),
            prior_messages=history,
            trace_state=state,
        )
        current_image = f"data:image/png;base64,Y3VycmVudC1hZnRlci0{step_index}"
        envelope = builder.build(
            user_query=task,
            observation=WebObservationContext(
                snapshot=WebSnapshotObservation(
                    url=f"https://example.com/current-{step_index}",
                    snapshot=f"current snapshot after {step_index}",
                ),
                screenshot=WebScreenshotObservation(data_url=current_image),
                observation_id=f"matrix-current-{step_index}",
            ),
            prior_messages=history,
            tools=[],
            tool_choice=None,
            allow_current_observation_image=False,
        )
        rendered_tool_text = _rendered_tool_text(envelope.messages)
        expected_images = seen_image_urls[-3:]
        expected_snapshot_indices = seen_snapshot_indices[-1:]
        omitted_snapshot_indices = seen_snapshot_indices[:-1]

        assert _all_prompt_image_urls(envelope.messages) == ([initial_image, *seen_image_urls])[-3:]
        assert _tool_result_image_urls(envelope.messages) == [
            image for image in expected_images if image != initial_image
        ]
        assert current_image not in _all_prompt_image_urls(envelope.messages)
        if step_index >= 2:
            assert "Matrix read-only URL with no image or refs." in rendered_tool_text

        for image_url in expected_images:
            assert image_url in _tool_result_image_urls(envelope.messages)
        for image_url in seen_image_urls[:-3]:
            assert image_url not in _tool_result_image_urls(envelope.messages)
        if len(seen_image_urls) > 3:
            assert "image omitted from this replay turn" in rendered_tool_text

            for index in expected_snapshot_indices:
                assert f'{index} button "Matrix {index}"' in rendered_tool_text
            for index in omitted_snapshot_indices:
                assert f'{index} button "Matrix {index}"' not in rendered_tool_text
        if omitted_snapshot_indices:
            assert "historical observation only: its refs are expired and invalid" in rendered_tool_text


def test_web_prompt_desktop_initial_image_and_refs_share_history_window_budget() -> None:
    from northcua.agents.web.domain.observations import WebElementRef, WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(
        history_window_rounds=20,
        screenshot_window_images=3,
        ref_snapshot_window_observations=3,
    )
    task = "exercise desktop window sequence"
    initial_image = "data:image/png;base64,aW5pdGlhbC13aW5kb3c="
    initial_id = "initial"

    def history_text(messages: list[object]) -> str:
        return (
            "\n".join(
                str(getattr(block, "text", ""))
                for message in messages[1:]
                for block in getattr(message, "content", [])
                if getattr(block, "type", None) == "text"
            )
            + "\n"
            + _rendered_tool_text(messages[1:])
        )

    initial_envelope = builder.build(
        user_query=task,
        observation=WebObservationContext(
            snapshot=WebSnapshotObservation(
                url="https://example.com",
                title="Initial",
                snapshot=f"full snapshot {initial_id}",
                elements=[WebElementRef(ref_id="e0", role="button", name="Initial")],
            ),
            screenshot=WebScreenshotObservation(data_url=initial_image),
            observation_id="window-initial",
        ),
        prior_messages=[],
        tools=[],
        tool_choice=None,
        allow_current_observation_image=True,
    )

    history = []
    prompt_image_counts = [len(_all_prompt_image_urls(initial_envelope.messages))]
    retained_ref_counts = [int('1 button "Initial"' in _leading_user_text(initial_envelope.messages))]
    assert _all_prompt_image_urls(initial_envelope.messages) == [initial_image]
    assert '1 button "Initial"' in _leading_user_text(initial_envelope.messages)

    post_image_urls = [
        "data:image/png;base64,cG9zdC13aW5kb3ctMQ==",
        "data:image/png;base64,cG9zdC13aW5kb3ctMg==",
        "data:image/png;base64,cG9zdC13aW5kb3ctMw==",
        "data:image/png;base64,cG9zdC13aW5kb3ctNA==",
        "data:image/png;base64,cG9zdC13aW5kb3ctNQ==",
    ]

    def post_observation_payload(index: int) -> dict[str, object]:
        return {
            "observation_id": f"window-post-{index}",
            "settled": True,
            "settle_reason": "stable",
            "page": {"url": f"https://example.com/post-{index}", "title": f"Post {index}"},
            "snapshot": {
                "text": f'{index} button "Post {index}"',
                "refs": [{"ref": str(index), "role": "button", "label": f"Post {index}"}],
                "ref_count": 1,
            },
            "screenshot": {
                "kind": "set_of_mark",
                "image_attached": True,
                "annotation": {"marks": [{"label": str(index), "ref": str(index)}]},
            },
        }

    for step_index, post_image in enumerate(post_image_urls, start=1):
        state = TraceCaptureState()
        call_id = f"call-window-{step_index}"
        state.assistant_tool_calls.append(
            AssistantToolCallRecord(id=call_id, name="web_click_ref", arguments=f'{{"ref":"{step_index}"}}')
        )
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id=call_id,
                tool_name="web_click_ref",
                tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked window {step_index}.",
                    "screenshot_url": post_image,
                    "observation": post_observation_payload(step_index),
                },
                llm_tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked window {step_index}.",
                    "screenshot_url": post_image,
                    "observation": post_observation_payload(step_index),
                },
            )
        )
        history = builder.append_step_history(
            envelope=initial_envelope if step_index == 1 else WebAgentPromptEnvelope(messages=history),
            prior_messages=history,
            trace_state=state,
        )
        current_image = f"data:image/png;base64,Y3VycmVudC13aW5kb3ct{step_index}"
        envelope = builder.build(
            user_query=task,
            observation=WebObservationContext(
                snapshot=WebSnapshotObservation(
                    url=f"https://example.com/current-{step_index}",
                    snapshot=f"current snapshot after post-{step_index}",
                ),
                screenshot=WebScreenshotObservation(data_url=current_image),
                observation_id=f"window-current-{step_index}",
            ),
            prior_messages=history,
            tools=[],
            tool_choice=None,
            allow_current_observation_image=False,
        )

        expected_images = ([initial_image, *post_image_urls[:step_index]])[-3:]
        all_images = _all_prompt_image_urls(envelope.messages)
        prompt_image_counts.append(len(all_images))
        assert all_images == expected_images
        assert current_image not in all_images

        all_snapshot_texts = [
            '1 button "Initial"',
            *[f'{index} button "Post {index}"' for index in range(1, step_index + 1)],
        ]
        rendered_history_text = f"{_leading_user_text(envelope.messages)}\n{history_text(envelope.messages)}"
        expected_retained_refs = all_snapshot_texts[-1:]
        omitted_refs = all_snapshot_texts[:-1]
        for snapshot_text in expected_retained_refs:
            assert snapshot_text in rendered_history_text
        for snapshot_text in omitted_refs:
            assert snapshot_text not in rendered_history_text
        retained_ref_counts.append(len(expected_retained_refs))
        if omitted_refs:
            assert "historical observation only: its refs are expired and invalid" in rendered_history_text
        if '1 button "Initial"' in omitted_refs:
            assert "initial page observation omitted from this replay turn" in rendered_history_text
            assert "observation_id: window-initial" not in rendered_history_text
            assert "- title: Initial" not in rendered_history_text

    assert prompt_image_counts == [1, 2, 3, 3, 3, 3]
    assert retained_ref_counts == [1, 1, 1, 1, 1, 1]


def test_web_prompt_image_window_omits_subsequent_current_image_and_keeps_tool_result_image() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    old_url = "data:image/png;base64,T0xE"
    current_url = "data:image/png;base64,Q1VSUkVOVA=="
    builder = WebAgentPromptBuilder(history_window_rounds=4, screenshot_window_images=1)
    state = TraceCaptureState()
    state.assistant_tool_calls.append(
        AssistantToolCallRecord(id="call-shot", name="web_click_ref", arguments='{"ref":"@e1"}')
    )
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-shot",
            tool_name="web_click_ref",
            tool_output={
                "status": "ok",
                "action": "click_ref",
                "text": "Clicked.",
                "screenshot_url": old_url,
                "observation": {
                    "observation_id": "old",
                    "settled": True,
                    "settle_reason": "stable",
                    "page": {"url": "https://example.com", "title": "Example"},
                    "snapshot": {"text": "button @e1", "refs": [], "ref_count": 0},
                    "screenshot": {"kind": "set_of_mark", "image_attached": True, "annotation": {}},
                },
            },
            llm_tool_output={
                "status": "ok",
                "action": "click_ref",
                "text": "Clicked.",
                "screenshot_url": old_url,
            },
        )
    )
    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=[Message(role=Role.USER, content=[TextBlock(text="old obs")])]),
        prior_messages=[],
        trace_state=state,
    )
    observation = WebObservationContext(
        snapshot=WebSnapshotObservation(url="https://example.com", snapshot="button @e1"),
        screenshot=WebScreenshotObservation(data_url=current_url),
        observation_id="current",
    )

    envelope = builder.build(
        user_query="continue",
        observation=observation,
        prior_messages=history,
        tools=[],
        tool_choice=None,
        allow_current_observation_image=False,
    )

    current_observation_image_urls = [
        getattr(block, "url", "")
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "image"
    ]
    retained_tool_result_images = [
        getattr(content_block, "url", "")
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "tool_result" and isinstance(block.content, list)
        for content_block in block.content
        if getattr(content_block, "type", None) == "image"
    ]
    tool_text = "\n".join(
        str(content_block.text)
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "tool_result" and isinstance(block.content, list)
        for content_block in block.content
        if getattr(content_block, "type", None) == "text"
    )
    observation_text = _leading_user_text(envelope.messages)

    assert current_observation_image_urls == []
    assert retained_tool_result_images == [old_url]
    assert observation_text == "Task:\ncontinue"
    assert "image omitted from this replay turn" not in tool_text
    assert "Q1VSUkVOVA" not in tool_text


def test_web_prompt_image_window_keeps_newest_history_images_beyond_desktop_turn_default() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(history_window_rounds=20, screenshot_window_images=2)
    history: list[Message] = [Message(role=Role.USER, content=[TextBlock(text="old obs")])]
    image_urls = [
        f"data:image/png;base64,{base64.b64encode(f'history-{index}'.encode()).decode('ascii')}" for index in range(10)
    ]
    for index, image_url in enumerate(image_urls):
        state = TraceCaptureState()
        state.assistant_tool_calls.append(
            AssistantToolCallRecord(id=f"call-image-{index}", name="web_click_ref", arguments='{"ref":"@e1"}')
        )
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id=f"call-image-{index}",
                tool_name="web_click_ref",
                tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked {index}.",
                    "screenshot_url": image_url,
                },
                llm_tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked {index}.",
                    "screenshot_url": image_url,
                },
            )
        )
        history = builder.append_step_history(
            envelope=WebAgentPromptEnvelope(messages=history),
            prior_messages=history,
            trace_state=state,
        )

    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )
    retained_images = [
        getattr(content_block, "url", "")
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "tool_result" and isinstance(block.content, list)
        for content_block in block.content
        if getattr(content_block, "type", None) == "image"
    ]
    rendered_tool_text_parts: list[str] = []
    for message in envelope.messages:
        for block in message.content:
            if getattr(block, "type", None) != "tool_result":
                continue
            if isinstance(block.content, str):
                rendered_tool_text_parts.append(block.content)
                continue
            if isinstance(block.content, list):
                rendered_tool_text_parts.extend(
                    str(content_block.text)
                    for content_block in block.content
                    if getattr(content_block, "type", None) == "text"
                )
    rendered_tool_text = "\n".join(rendered_tool_text_parts)

    assert retained_images == image_urls[-2:]
    assert "image omitted from this replay turn" in rendered_tool_text


def test_web_prompt_step_three_keeps_two_tool_result_images_without_duplicate_current_observation_image() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(history_window_rounds=8, screenshot_window_images=2)
    history: list[Message] = [Message(role=Role.USER, content=[TextBlock(text="initial observation")])]
    post_action_urls = [
        "data:image/png;base64,cG9zdC0x",
        "data:image/png;base64,cG9zdC0y",
    ]
    for index, image_url in enumerate(post_action_urls, start=1):
        observation_payload = {
            "observation_id": f"post-{index}",
            "settled": True,
            "settle_reason": "stable",
            "page": {"url": "https://example.com", "title": "Example"},
            "snapshot": {"text": f"button @e{index}", "refs": [], "ref_count": 0},
            "screenshot": {
                "kind": "set_of_mark",
                "image_attached": True,
                "annotation": {
                    "kind": "set_of_mark",
                    "marked_count": 1,
                    "visible_interactive_count": 1,
                    "omitted_count": 0,
                    "marks": [
                        {
                            "ref": "19",
                            "label": "19",
                            "bbox_css": {"x": 1, "y": 2, "width": 30, "height": 12},
                        }
                    ],
                },
            },
        }
        state = TraceCaptureState()
        state.assistant_tool_calls.append(
            AssistantToolCallRecord(id=f"call-{index}", name="web_click_ref", arguments='{"ref":"@e1"}')
        )
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id=f"call-{index}",
                tool_name="web_click_ref",
                tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked {index}.",
                    "screenshot_url": image_url,
                    "observation": observation_payload,
                },
                llm_tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked {index}.",
                    "screenshot_url": image_url,
                    "observation": observation_payload,
                },
            )
        )
        history = builder.append_step_history(
            envelope=WebAgentPromptEnvelope(messages=history),
            prior_messages=history,
            trace_state=state,
        )

    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(
            snapshot=WebSnapshotObservation(url="https://example.com", snapshot="button @e2"),
            screenshot=WebScreenshotObservation(data_url=post_action_urls[-1]),
            observation_id="post-2",
        ),
        prior_messages=history,
        tools=[],
        tool_choice=None,
        allow_current_observation_image=False,
    )

    top_level_images = [
        getattr(block, "url", "")
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "image"
    ]
    tool_result_images = [
        getattr(content_block, "url", "")
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "tool_result" and isinstance(block.content, list)
        for content_block in block.content
        if getattr(content_block, "type", None) == "image"
    ]
    observation_text = _leading_user_text(envelope.messages)

    assert top_level_images == []
    assert tool_result_images == post_action_urls
    assert tool_result_images.count(post_action_urls[-1]) == 1
    assert observation_text == "Task:\ncontinue"
    assert "post-2" in _rendered_tool_text(envelope.messages)


def test_web_prompt_image_and_ref_windows_skip_non_observation_results_without_misalignment() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(
        history_window_rounds=8,
        screenshot_window_images=1,
        ref_snapshot_window_observations=1,
    )
    history: list[Message] = [Message(role=Role.USER, content=[TextBlock(text="initial observation")])]

    def observation_payload(index: int) -> dict[str, object]:
        return {
            "observation_id": f"post-{index}",
            "settled": True,
            "settle_reason": "stable",
            "page": {"url": "https://example.com", "title": "Example"},
            "snapshot": {
                "text": f"full snapshot {index}",
                "refs": [{"ref": str(index), "role": "button", "label": f"Button {index}"}],
                "ref_count": 1,
            },
            "screenshot": {
                "kind": "set_of_mark",
                "image_attached": True,
                "annotation": {
                    "kind": "set_of_mark",
                    "marked_count": 1,
                    "visible_interactive_count": 1,
                    "omitted_count": 0,
                    "marks": [
                        {
                            "ref": "19",
                            "label": "19",
                            "bbox_css": {"x": 1, "y": 2, "width": 30, "height": 12},
                        }
                    ],
                },
            },
        }

    def append_history(
        *,
        call_id: str,
        tool_name: str,
        arguments: str,
        output: dict[str, object],
    ) -> None:
        nonlocal history
        state = TraceCaptureState()
        state.assistant_tool_calls.append(AssistantToolCallRecord(id=call_id, name=tool_name, arguments=arguments))
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id=call_id,
                tool_name=tool_name,
                tool_output=output,
                llm_tool_output=output,
            )
        )
        history = builder.append_step_history(
            envelope=WebAgentPromptEnvelope(messages=history),
            prior_messages=history,
            trace_state=state,
        )

    image_a = "data:image/png;base64,aW1hZ2UtYQ=="
    image_b = "data:image/png;base64,aW1hZ2UtYg=="
    append_history(
        call_id="call-a",
        tool_name="web_click_ref",
        arguments='{"ref":"@e1"}',
        output={
            "status": "ok",
            "action": "click_ref",
            "text": "Clicked A.",
            "screenshot_url": image_a,
            "observation": observation_payload(1),
        },
    )
    append_history(
        call_id="call-note",
        tool_name="web_note",
        arguments='{"text":"middle note","kind":"fact"}',
        output={"status": "recorded", "action": "note", "text": "[fact] middle note without refs"},
    )
    append_history(
        call_id="call-b",
        tool_name="web_click_ref",
        arguments='{"ref":"@e2"}',
        output={
            "status": "ok",
            "action": "click_ref",
            "text": "Clicked B.",
            "screenshot_url": image_b,
            "observation": observation_payload(2),
        },
    )

    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
        allow_current_observation_image=False,
    )
    retained_images = [
        getattr(content_block, "url", "")
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "tool_result" and isinstance(block.content, list)
        for content_block in block.content
        if getattr(content_block, "type", None) == "image"
    ]
    rendered_tool_text_parts: list[str] = []
    for message in envelope.messages:
        for block in message.content:
            if getattr(block, "type", None) != "tool_result":
                continue
            if isinstance(block.content, str):
                rendered_tool_text_parts.append(block.content)
                continue
            if isinstance(block.content, list):
                rendered_tool_text_parts.extend(
                    str(content_block.text)
                    for content_block in block.content
                    if getattr(content_block, "type", None) == "text"
                )
    rendered_tool_text = "\n".join(rendered_tool_text_parts)

    assert retained_images == [image_b]
    assert image_a not in retained_images
    assert "image omitted from this replay turn" in rendered_tool_text
    assert '2 button "Button 2"' in rendered_tool_text
    assert '1 button "Button 1"' not in rendered_tool_text
    assert "full snapshot 2" not in rendered_tool_text
    assert "full snapshot 1" not in rendered_tool_text
    assert "historical observation only: its refs are expired and invalid" in rendered_tool_text
    assert '"observation"' not in rendered_tool_text
    assert "bbox_css" not in rendered_tool_text
    assert "[fact] middle note without refs" in rendered_tool_text


def test_web_prompt_history_omits_stale_ref_args_without_placeholder_fields() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(history_window_rounds=4)
    state = TraceCaptureState()
    state.assistant_tool_calls.append(
        AssistantToolCallRecord(id="call-stale-ref", name="web_click_ref", arguments='{"ref":"31"}')
    )
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-stale-ref",
            tool_name="web_click_ref",
            tool_output={
                "status": "ok",
                "action": "click_ref",
                "text": "Clicked 31.",
                "observation": {
                    "observation_id": "post-stale-ref",
                    "settled": True,
                    "settle_reason": "stable",
                    "page": {"url": "https://example.com", "title": "Example"},
                    "snapshot": {"text": "old page", "refs": [{"ref": "31", "role": "button"}], "ref_count": 1},
                    "screenshot": {"kind": "set_of_mark", "image_attached": False, "annotation": {}},
                },
            },
            llm_tool_output={
                "status": "ok",
                "action": "click_ref",
                "text": "Clicked 31.",
                "observation": {
                    "observation_id": "post-stale-ref",
                    "settled": True,
                    "settle_reason": "stable",
                    "page": {"url": "https://example.com", "title": "Example"},
                    "snapshot": {"text": "old page", "refs": [{"ref": "31", "role": "button"}], "ref_count": 1},
                    "screenshot": {"kind": "set_of_mark", "image_attached": False, "annotation": {}},
                },
            },
        )
    )
    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=[Message(role=Role.USER, content=[TextBlock(text="old obs")])]),
        prior_messages=[],
        trace_state=state,
    )
    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )

    tool_use_inputs = [
        getattr(block, "input", {})
        for message in envelope.messages
        for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "tool_use"
    ]
    rendered_tool_text = _rendered_tool_text(envelope.messages)

    assert tool_use_inputs == [{"ref": "31"}]
    assert "historical ref omitted" not in json.dumps(tool_use_inputs)
    assert "Post-action page observation:" in rendered_tool_text
    assert "31 button" in rendered_tool_text
    assert '"ref": "31"' not in rendered_tool_text
    assert "old page" not in rendered_tool_text


def test_web_prompt_history_replays_observation_like_single_prompt_surface() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    image_url = "data:image/png;base64,aW1hZ2U="
    builder = WebAgentPromptBuilder(history_window_rounds=4)
    state = TraceCaptureState()
    state.assistant_tool_calls.append(
        AssistantToolCallRecord(id="call-click", name="web_click_ref", arguments='{"ref":"1"}')
    )
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-click",
            tool_name="web_click_ref",
            tool_output={
                "status": "ok",
                "action": "click_ref",
                "text": "Clicked 1.",
                "screenshot_url": image_url,
                "observation": {
                    "observation_id": "post-click",
                    "settled": True,
                    "settle_reason": "stable_2_samples",
                    "page": {"url": "https://example.com", "title": "Example"},
                    "snapshot": {
                        "text": '1 button "Target"',
                        "refs": [
                            {
                                "ref": "1",
                                "role": "button",
                                "label": "Target",
                                "bbox": "1,2,30,12",
                                "bbox_css": {"x": 1, "y": 2, "width": 30, "height": 12},
                            }
                        ],
                        "ref_count": 1,
                    },
                    "screenshot": {
                        "kind": "set_of_mark",
                        "image_attached": True,
                        "annotation": {
                            "marks": [
                                {
                                    "ref": "1",
                                    "label": "1",
                                    "name": "Target",
                                    "bbox_css": {"x": 1, "y": 2, "width": 30, "height": 12},
                                }
                            ],
                            "omitted_count": 7,
                            "truncated": False,
                        },
                    },
                },
            },
            llm_tool_output={
                "status": "ok",
                "action": "click_ref",
                "text": "Clicked 1.",
                "screenshot_url": image_url,
                "observation": {
                    "observation_id": "post-click",
                    "settled": True,
                    "settle_reason": "stable_2_samples",
                    "page": {"url": "https://example.com", "title": "Example"},
                    "snapshot": {
                        "text": '1 button "Target"',
                        "refs": [
                            {
                                "ref": "1",
                                "role": "button",
                                "label": "Target",
                                "bbox": "1,2,30,12",
                                "bbox_css": {"x": 1, "y": 2, "width": 30, "height": 12},
                            }
                        ],
                        "ref_count": 1,
                    },
                    "screenshot": {
                        "kind": "set_of_mark",
                        "image_attached": True,
                        "annotation": {
                            "marks": [
                                {
                                    "ref": "1",
                                    "label": "1",
                                    "name": "Target",
                                    "bbox_css": {"x": 1, "y": 2, "width": 30, "height": 12},
                                }
                            ],
                            "omitted_count": 7,
                            "truncated": False,
                        },
                    },
                },
            },
        )
    )
    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=[Message(role=Role.USER, content=[TextBlock(text="old obs")])]),
        prior_messages=[],
        trace_state=state,
    )
    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )
    rendered_tool_text = _rendered_tool_text(envelope.messages)
    retained_images = [
        getattr(content_block, "url", "")
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "tool_result" and isinstance(block.content, list)
        for content_block in block.content
        if getattr(content_block, "type", None) == "image"
    ]

    assert retained_images == [image_url]
    assert "Clicked 1." in rendered_tool_text
    assert "Post-action page observation:" in rendered_tool_text
    assert "- title: Example" in rendered_tool_text
    assert "Actionable refs (1):" in rendered_tool_text
    assert '- 1 button "Target"' in rendered_tool_text
    assert "- omitted candidates: 7" in rendered_tool_text
    assert '"observation"' not in rendered_tool_text
    assert "bbox_css" not in rendered_tool_text
    assert '"x": 1' not in rendered_tool_text


def test_web_prompt_history_replays_explicit_snapshot_as_compact_observation() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(history_window_rounds=4)
    observation_payload = {
        "observation_id": "snapshot-tool",
        "settled": True,
        "settle_reason": "stable",
        "page": {"url": "https://example.com", "title": "Example"},
        "snapshot": {
            "text": "\n".join(f'{index} link "Item {index}"' for index in range(1, 80)),
            "refs": [{"ref": str(index), "role": "link", "label": f"Item {index}"} for index in range(1, 80)],
            "ref_count": 79,
        },
        "screenshot": {"kind": "none", "image_attached": False, "annotation": {}},
    }
    state = TraceCaptureState()
    state.assistant_tool_calls.append(AssistantToolCallRecord(id="call-snapshot", name="web_snapshot", arguments="{}"))
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-snapshot",
            tool_name="web_snapshot",
            tool_output={
                "status": "ok",
                "action": "snapshot",
                "text": "Captured current snapshot.",
                "result": {
                    "url": "https://example.com",
                    "title": "Example",
                    "snapshot": observation_payload["snapshot"]["text"],
                    "refs": observation_payload["snapshot"]["refs"],
                    "ref_count": 79,
                    "observation": observation_payload,
                },
            },
            llm_tool_output={
                "status": "ok",
                "action": "snapshot",
                "text": "Captured current snapshot.",
                "result": {
                    "url": "https://example.com",
                    "title": "Example",
                    "snapshot": observation_payload["snapshot"]["text"],
                    "refs": observation_payload["snapshot"]["refs"],
                    "ref_count": 79,
                    "observation": observation_payload,
                },
            },
        )
    )
    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=[Message(role=Role.USER, content=[TextBlock(text="old obs")])]),
        prior_messages=[],
        trace_state=state,
    )
    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )
    rendered_tool_text = _rendered_tool_text(envelope.messages)

    assert "Captured current snapshot." in rendered_tool_text
    assert "Post-action page observation:" in rendered_tool_text
    assert "Observation refs (79):" in rendered_tool_text
    assert '1 link "Item 1"' in rendered_tool_text
    assert '"snapshot":' not in rendered_tool_text
    assert '"observation":' not in rendered_tool_text
    assert '"refs":' not in rendered_tool_text


def test_web_prompt_history_rewrites_ref_tool_text_without_observation() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(history_window_rounds=4)
    state = TraceCaptureState()
    state.assistant_tool_calls.append(
        AssistantToolCallRecord(id="call-get-text", name="web_get_text", arguments='{"ref":"31"}')
    )
    state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-get-text",
            tool_name="web_get_text",
            tool_output={
                "status": "ok",
                "action": "get_text",
                "text": "Read text from 31.",
                "params": {"ref": "31"},
                "result": {"text": "Hello"},
            },
            llm_tool_output={
                "status": "ok",
                "action": "get_text",
                "text": "Read text from 31.",
                "params": {"ref": "31"},
                "result": {"text": "Hello"},
            },
        )
    )
    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=[Message(role=Role.USER, content=[TextBlock(text="old obs")])]),
        prior_messages=[],
        trace_state=state,
    )
    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )

    rendered_tool_text = _rendered_tool_text(envelope.messages)

    assert "Read text from 31" in rendered_tool_text
    assert "historical refs omitted" not in rendered_tool_text
    assert "Hello" in rendered_tool_text


def test_web_prompt_ref_snapshot_window_omits_older_history_refs_with_notice() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(
        history_window_rounds=4,
        screenshot_window_images=0,
        ref_snapshot_window_observations=1,
    )
    history: list[Message] = [Message(role=Role.USER, content=[TextBlock(text="old obs")])]
    for index in range(2):
        state = TraceCaptureState()
        state.assistant_tool_calls.append(
            AssistantToolCallRecord(id=f"call-{index}", name="web_click_ref", arguments='{"ref":"@e1"}')
        )
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id=f"call-{index}",
                tool_name="web_click_ref",
                tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked {index}.",
                    "screenshot_url": "data:image/png;base64,QUJD",
                    "observation": {
                        "observation_id": f"post-{index}",
                        "settled": True,
                        "settle_reason": "stable",
                        "page": {"url": "https://example.com", "title": "Example"},
                        "snapshot": {
                            "text": f"full snapshot {index}",
                            "refs": [{"ref": str(index), "role": "button", "label": f"Button {index}"}],
                            "ref_count": 1,
                        },
                        "screenshot": {
                            "kind": "set_of_mark",
                            "image_attached": True,
                            "annotation": {
                                "kind": "set_of_mark",
                                "marked_count": 1,
                                "visible_interactive_count": 1,
                                "omitted_count": 0,
                                "marks": [
                                    {
                                        "ref": str(index),
                                        "label": str(index),
                                        "bbox_css": {"x": 1, "y": 2, "width": 3, "height": 4},
                                    }
                                ],
                            },
                        },
                    },
                },
                llm_tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked {index}.",
                    "screenshot_url": "data:image/png;base64,QUJD",
                    "observation": {
                        "observation_id": f"post-{index}",
                        "settled": True,
                        "settle_reason": "stable",
                        "page": {"url": "https://example.com", "title": "Example"},
                        "snapshot": {
                            "text": f"full snapshot {index}",
                            "refs": [{"ref": str(index), "role": "button", "label": f"Button {index}"}],
                            "ref_count": 1,
                        },
                        "screenshot": {"kind": "set_of_mark", "image_attached": True, "annotation": {}},
                    },
                },
            )
        )
        history = builder.append_step_history(
            envelope=WebAgentPromptEnvelope(messages=history),
            prior_messages=history,
            trace_state=state,
        )

    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )
    rendered = "\n".join(
        str(block.content)
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "tool_result"
    )

    assert '1 button "Button 1"' in rendered
    assert '0 button "Button 0"' not in rendered
    assert "full snapshot 1" not in rendered
    assert "full snapshot 0" not in rendered
    assert "historical observation only: its refs are expired and invalid" in rendered
    assert '"observation"' not in rendered


def test_web_prompt_only_latest_retained_observation_exposes_actionable_numeric_refs() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(
        history_window_rounds=8,
        screenshot_window_images=3,
        ref_snapshot_window_observations=3,
    )
    history: list[Message] = [Message(role=Role.USER, content=[TextBlock(text="old obs")])]
    labels = ["Old color field", "Historical green link", "Current color textbox"]
    roles = ["textbox", "link", "textbox"]
    for index, label in enumerate(labels):
        observation_payload = {
            "observation_id": f"post-ref-19-{index}",
            "settled": True,
            "settle_reason": "stable",
            "page": {"url": "https://example.com", "title": "Example"},
            "snapshot": {
                "text": f'19 {roles[index]} "{label}"',
                "refs": [{"ref": "19", "role": roles[index], "label": label}],
                "ref_count": 1,
            },
            "screenshot": {
                "kind": "set_of_mark",
                "image_attached": True,
                "annotation": {
                    "kind": "set_of_mark",
                    "marked_count": 1,
                    "visible_interactive_count": 1,
                    "omitted_count": 0,
                    "marks": [
                        {
                            "ref": "19",
                            "label": "19",
                            "bbox_css": {"x": 1, "y": 2, "width": 30, "height": 12},
                        }
                    ],
                },
            },
        }
        state = TraceCaptureState()
        state.assistant_tool_calls.append(
            AssistantToolCallRecord(id=f"call-ref-19-{index}", name="web_click_ref", arguments='{"ref":"19"}')
        )
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id=f"call-ref-19-{index}",
                tool_name="web_click_ref",
                tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked {index}.",
                    "screenshot_url": f"data:image/png;base64,QUJD{index}",
                    "observation": observation_payload,
                },
                llm_tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked {index}.",
                    "screenshot_url": f"data:image/png;base64,QUJD{index}",
                    "observation": observation_payload,
                },
            )
        )
        history = builder.append_step_history(
            envelope=WebAgentPromptEnvelope(messages=history),
            prior_messages=history,
            trace_state=state,
        )

    envelope = builder.build(
        user_query="set color to green",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )
    rendered_tool_text = _rendered_tool_text(envelope.messages)

    assert '19 textbox "Current color textbox"' in rendered_tool_text
    assert "Old color field" not in rendered_tool_text
    assert "Historical green link" not in rendered_tool_text
    assert rendered_tool_text.count("Actionable refs (1):") == 1
    assert rendered_tool_text.count("SoM marks are numeric refs") == 1
    assert "Historical observation visual only" in rendered_tool_text
    assert "Historical SoM image only" not in rendered_tool_text
    assert "historical observation only: its refs are expired and invalid" in rendered_tool_text


def test_web_visual_history_path_keeps_only_latest_refs_actionable(tmp_path: Path) -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.evaluation_artifacts import WebToolResultArtifact
    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope
    from northcua.agents.web.prompt.visual_window import WebPromptVisualHistoryFactory

    builder = WebAgentPromptBuilder(
        history_window_rounds=8,
        screenshot_window_images=2,
        ref_snapshot_window_observations=1,
    )
    history: list[Message] = [Message(role=Role.USER, content=[TextBlock(text="Task:\nset color")])]
    visual_turns = []
    labels = ["Historical green link", "Current color textbox"]
    roles = ["link", "textbox"]
    factory = WebPromptVisualHistoryFactory()
    for index, label in enumerate(labels):
        observation_payload = {
            "observation_id": f"visual-post-{index}",
            "settled": True,
            "settle_reason": "stable",
            "page": {"url": "https://example.com", "title": "Example"},
            "snapshot": {
                "text": f'19 {roles[index]} "{label}"',
                "refs": [{"ref": "19", "role": roles[index], "label": label}],
                "ref_count": 1,
            },
            "screenshot": {
                "kind": "set_of_mark",
                "image_attached": True,
                "annotation": {
                    "kind": "set_of_mark",
                    "marked_count": 1,
                    "visible_interactive_count": 1,
                    "omitted_count": 0,
                    "marks": [
                        {
                            "ref": "19",
                            "label": "19",
                            "bbox_css": {"x": 1, "y": 2, "width": 30, "height": 12},
                        }
                    ],
                },
            },
        }
        state = TraceCaptureState()
        state.assistant_tool_calls.append(
            AssistantToolCallRecord(id=f"call-visual-ref-{index}", name="web_click_ref", arguments='{"ref":"19"}')
        )
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id=f"call-visual-ref-{index}",
                tool_name="web_click_ref",
                tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked {index}.",
                    "screenshot_url": "data:image/png;base64,QUJD",
                    "observation": observation_payload,
                },
                llm_tool_output={
                    "status": "ok",
                    "action": "click_ref",
                    "text": f"Clicked {index}.",
                    "screenshot_url": "data:image/png;base64,QUJD",
                    "observation": observation_payload,
                },
            )
        )
        history = builder.append_step_history(
            envelope=WebAgentPromptEnvelope(messages=history),
            prior_messages=history,
            trace_state=state,
        )
        visual_turn = factory.build_turn(
            session_step=index + 1,
            round_num=1,
            tool_artifacts=[
                WebToolResultArtifact(
                    tool_name="web_click_ref",
                    tool_call_id=f"call-visual-ref-{index}",
                    action="click_ref",
                    status="ok",
                    text=f"Clicked {index}.",
                    raw_image_path=str(_write_tiny_png(tmp_path / f"raw-{index}.png")),
                    action_overlay_image_path=str(_write_tiny_png(tmp_path / f"overlay-{index}.png")),
                    som_image_path=str(_write_tiny_png(tmp_path / f"som-{index}.png")),
                )
            ],
        )
        assert visual_turn is not None
        visual_turns.append(visual_turn)

    envelope = builder.build(
        user_query="set color",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
        visual_history_turns=visual_turns,
    )
    rendered_tool_text = _rendered_tool_text(envelope.messages)

    assert '19 textbox "Current color textbox"' in rendered_tool_text
    assert "Historical green link" not in rendered_tool_text
    assert rendered_tool_text.count("Actionable refs (1):") == 1
    assert rendered_tool_text.count("SoM marks are numeric refs") == 1
    assert "historical observation only: its refs are expired and invalid" in rendered_tool_text
    assert "Historical observation visual only" in rendered_tool_text
    assert "Historical SoM image only" not in rendered_tool_text


def test_web_visual_history_does_not_pin_stale_som_after_newer_snapshot_refs(tmp_path: Path) -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.evaluation_artifacts import WebToolResultArtifact
    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope
    from northcua.agents.web.prompt.visual_window import WebPromptVisualHistoryFactory

    builder = WebAgentPromptBuilder(
        history_window_rounds=8,
        screenshot_window_images=2,
        ref_snapshot_window_observations=1,
    )
    history: list[Message] = [Message(role=Role.USER, content=[TextBlock(text="Task:\ninspect")])]
    click_observation = {
        "observation_id": "post-click-with-som",
        "settled": True,
        "settle_reason": "stable",
        "page": {"url": "https://example.com", "title": "Example"},
        "snapshot": {
            "text": '7 link "Old result"',
            "refs": [{"ref": "7", "role": "link", "label": "Old result"}],
            "ref_count": 1,
        },
        "screenshot": {
            "kind": "set_of_mark",
            "image_attached": True,
            "annotation": {
                "kind": "set_of_mark",
                "marked_count": 1,
                "visible_interactive_count": 1,
                "omitted_count": 0,
                "marks": [{"ref": "7", "label": "7", "bbox_css": {"x": 1, "y": 2, "width": 30, "height": 12}}],
            },
        },
    }
    snapshot_observation = {
        "observation_id": "newer-snapshot-refs",
        "settled": True,
        "settle_reason": "snapshot",
        "page": {"url": "https://example.com", "title": "Example"},
        "snapshot": {
            "text": '8 button "New current result"',
            "refs": [{"ref": "8", "role": "button", "label": "New current result"}],
            "ref_count": 1,
        },
    }
    click_state = TraceCaptureState()
    click_state.assistant_tool_calls.append(
        AssistantToolCallRecord(id="call-click-with-som", name="web_click_ref", arguments='{"ref":"7"}')
    )
    click_state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-click-with-som",
            tool_name="web_click_ref",
            tool_output={
                "status": "ok",
                "action": "click_ref",
                "text": "Clicked 7.",
                "screenshot_url": "data:image/png;base64,QUJD",
                "observation": click_observation,
            },
            llm_tool_output={
                "status": "ok",
                "action": "click_ref",
                "text": "Clicked 7.",
                "screenshot_url": "data:image/png;base64,QUJD",
                "observation": click_observation,
            },
        )
    )
    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=history),
        prior_messages=history,
        trace_state=click_state,
    )
    visual_turn = WebPromptVisualHistoryFactory().build_turn(
        session_step=1,
        round_num=1,
        tool_artifacts=[
            WebToolResultArtifact(
                tool_name="web_click_ref",
                tool_call_id="call-click-with-som",
                action="click_ref",
                status="ok",
                text="Clicked 7.",
                raw_image_path=str(_write_tiny_png(tmp_path / "raw.png")),
                action_overlay_image_path=str(_write_tiny_png(tmp_path / "overlay.png")),
                som_image_path=str(_write_tiny_png(tmp_path / "som.png")),
            )
        ],
    )
    assert visual_turn is not None
    snapshot_state = TraceCaptureState()
    snapshot_state.assistant_tool_calls.append(
        AssistantToolCallRecord(id="call-newer-snapshot", name="web_snapshot", arguments="{}")
    )
    snapshot_state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-newer-snapshot",
            tool_name="web_snapshot",
            tool_output={
                "status": "ok",
                "action": "snapshot",
                "text": "Captured current snapshot.",
                "observation": snapshot_observation,
            },
            llm_tool_output={
                "status": "ok",
                "action": "snapshot",
                "text": "Captured current snapshot.",
                "observation": snapshot_observation,
            },
        )
    )
    history = builder.append_step_history(
        envelope=WebAgentPromptEnvelope(messages=history),
        prior_messages=history,
        trace_state=snapshot_state,
    )
    budgeted_history = builder.ref_snapshot_window_budgeter.apply(builder._trim_history(history))
    assert builder.ref_snapshot_window_budgeter.latest_retained_tool_result_ids(budgeted_history) == {
        "call-newer-snapshot"
    }

    envelope = builder.build(
        user_query="inspect",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
        visual_history_turns=[visual_turn],
    )
    rendered_tool_text = _rendered_tool_text(envelope.messages)

    assert '8 button "New current result"' in rendered_tool_text
    assert '7 link "Old result"' not in rendered_tool_text
    assert "Latest Set-of-Mark image for actionable refs" not in rendered_tool_text
    assert "Historical observation visual only" in rendered_tool_text
    assert "Historical SoM image only" not in rendered_tool_text


def test_web_ref_snapshot_window_compacts_raw_json_annotation_marks() -> None:
    from nexau.core.messages import Message, Role, ToolResultBlock

    from northcua.agents.web.prompt.ref_window import WebRefSnapshotWindowBudgeter

    payload = {
        "status": "ok",
        "action": "snapshot",
        "observation": {
            "observation_id": "raw-json-old-observation",
            "snapshot": {
                "text": '19 textbox "Old color field"',
                "refs": [{"ref": "19", "role": "textbox", "label": "Old color field"}],
                "ref_count": 1,
            },
            "screenshot": {
                "kind": "set_of_mark",
                "image_attached": True,
                "annotation": {
                    "kind": "set_of_mark",
                    "marked_count": 1,
                    "visible_interactive_count": 1,
                    "omitted_count": 0,
                    "marks": [
                        {
                            "ref": "19",
                            "label": "19",
                            "bbox_css": {"x": 1, "y": 2, "width": 30, "height": 12},
                        }
                    ],
                },
            },
        },
    }
    message = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="call-snapshot", content=json.dumps(payload), is_error=False)],
    )

    compacted = WebRefSnapshotWindowBudgeter(max_history_snapshots=0).apply([message])
    block = compacted[0].content[0]
    assert isinstance(block, ToolResultBlock)
    compacted_payload = json.loads(str(block.content))

    observation = compacted_payload["observation"]
    assert observation["snapshot"]["refs"] == []
    assert observation["snapshot"]["text"] == "[omitted by web ref-snapshot window]"
    assert observation["screenshot"]["annotation"]["marks"] == []
    assert "bbox_css" not in json.dumps(compacted_payload)
    assert compacted_payload["snapshot_omitted_notice"].startswith("[historical observation only")


def test_web_ref_snapshot_window_compacts_annotation_marks_without_snapshot_refs() -> None:
    from nexau.core.messages import Message, Role, ToolResultBlock

    from northcua.agents.web.prompt.ref_window import WebRefSnapshotWindowBudgeter

    payload = {
        "status": "ok",
        "action": "click_ref",
        "observation": {
            "observation_id": "annotation-only-old-observation",
            "snapshot": {"text": "", "refs": [], "ref_count": 0},
            "screenshot": {
                "kind": "set_of_mark",
                "image_attached": True,
                "annotation": {
                    "kind": "set_of_mark",
                    "marked_count": 1,
                    "visible_interactive_count": 1,
                    "omitted_count": 0,
                    "marks": [
                        {
                            "ref": "19",
                            "label": "19",
                            "bbox_css": {"x": 8, "y": 12, "width": 44, "height": 16},
                        }
                    ],
                },
            },
        },
    }
    message = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="call-click", content=json.dumps(payload), is_error=False)],
    )

    compacted = WebRefSnapshotWindowBudgeter(max_history_snapshots=0).apply([message])
    block = compacted[0].content[0]
    assert isinstance(block, ToolResultBlock)
    compacted_payload = json.loads(str(block.content))

    observation = compacted_payload["observation"]
    assert observation["snapshot"]["refs"] == []
    assert observation["screenshot"]["annotation"]["marks"] == []
    assert "bbox_css" not in json.dumps(compacted_payload)
    assert compacted_payload["snapshot_omitted_notice"].startswith("[historical observation only")


def test_web_ref_snapshot_window_budgets_all_text_blocks_in_tool_result() -> None:
    from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock

    from northcua.agents.web.prompt.ref_window import WebRefSnapshotWindowBudgeter

    def payload(ref: str, label: str) -> str:
        return json.dumps(
            {
                "status": "ok",
                "action": "snapshot",
                "observation": {
                    "observation_id": f"obs-{ref}",
                    "snapshot": {
                        "text": f'{ref} button "{label}"',
                        "refs": [{"ref": ref, "role": "button", "label": label}],
                        "ref_count": 1,
                    },
                    "screenshot": {
                        "kind": "set_of_mark",
                        "image_attached": True,
                        "annotation": {
                            "kind": "set_of_mark",
                            "marked_count": 1,
                            "visible_interactive_count": 1,
                            "omitted_count": 0,
                            "marks": [{"ref": ref, "label": ref}],
                        },
                    },
                },
            }
        )

    message = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call-multi-text",
                content=[
                    TextBlock(text=payload("19", "old result")),
                    TextBlock(text="non-observation note"),
                    TextBlock(text=payload("20", "latest result")),
                ],
                is_error=False,
            )
        ],
    )

    compacted = WebRefSnapshotWindowBudgeter(max_history_snapshots=1).apply([message])
    block = compacted[0].content[0]
    assert isinstance(block, ToolResultBlock)
    assert isinstance(block.content, list)

    old_payload = json.loads(block.content[0].text)
    latest_payload = json.loads(block.content[2].text)

    assert old_payload["observation"]["snapshot"]["refs"] == []
    assert old_payload["observation"]["screenshot"]["annotation"]["marks"] == []
    assert old_payload["snapshot_omitted_notice"].startswith("[historical observation only")
    assert latest_payload["observation"]["snapshot"]["refs"] == [
        {"ref": "20", "role": "button", "label": "latest result"}
    ]
    assert latest_payload["observation"]["screenshot"]["annotation"]["marks"] == [{"ref": "20", "label": "20"}]


def test_web_prompt_ref_snapshot_window_handles_observation_without_image_side_channel() -> None:
    from nexau.core.messages import Message, Role, TextBlock

    from northcua.agents.web.domain.observations import WebSnapshotObservation
    from northcua.agents.web.prompt import WebAgentPromptBuilder, WebAgentPromptEnvelope

    builder = WebAgentPromptBuilder(
        history_window_rounds=4,
        screenshot_window_images=0,
        ref_snapshot_window_observations=1,
    )
    history: list[Message] = [Message(role=Role.USER, content=[TextBlock(text="old obs")])]
    for index in range(2):
        observation_payload = {
            "observation_id": f"snapshot-only-{index}",
            "settled": True,
            "settle_reason": "stable",
            "page": {"url": "https://example.com", "title": "Example"},
            "snapshot": {
                "text": f"snapshot-only full refs {index}",
                "refs": [{"ref": str(index), "role": "button"}],
                "ref_count": 1,
            },
            "screenshot": {"kind": "none", "image_attached": False, "annotation": {}},
        }
        state = TraceCaptureState()
        state.assistant_tool_calls.append(
            AssistantToolCallRecord(id=f"call-snapshot-only-{index}", name="web_wait", arguments='{"duration_ms":1}')
        )
        state.tool_executions.append(
            ToolExecutionRecord(
                tool_call_id=f"call-snapshot-only-{index}",
                tool_name="web_wait",
                tool_output={
                    "status": "ok",
                    "action": "wait",
                    "text": f"Waited {index}.",
                    "observation": observation_payload,
                },
                llm_tool_output={
                    "status": "ok",
                    "action": "wait",
                    "text": f"Waited {index}.",
                    "observation": observation_payload,
                },
            )
        )
        history = builder.append_step_history(
            envelope=WebAgentPromptEnvelope(messages=history),
            prior_messages=history,
            trace_state=state,
        )

    envelope = builder.build(
        user_query="continue",
        observation=WebObservationContext(snapshot=WebSnapshotObservation(url="https://example.com", snapshot="body")),
        prior_messages=history,
        tools=[],
        tool_choice=None,
    )
    rendered = "\n".join(
        str(block.content)
        for message in envelope.messages
        for block in message.content
        if getattr(block, "type", None) == "tool_result"
    )

    assert "1 button" in rendered
    assert "0 button" not in rendered
    assert "snapshot-only full refs 1" not in rendered
    assert "snapshot-only full refs 0" not in rendered
    assert "historical observation only: its refs are expired and invalid" in rendered
    assert '"observation"' not in rendered


def test_web_observation_projector_rehydrates_post_action_bundle_without_paths() -> None:
    payload = {
        "observation_id": "post-click-1",
        "settled": False,
        "settle_reason": "timeout_best_effort",
        "page": {"url": "https://example.com", "title": "Example"},
        "snapshot": {
            "text": "button @e1 Submit",
            "refs": [{"ref": "@e1", "role": "button", "name": "Submit", "text": "", "tag": "button"}],
            "ref_count": 1,
        },
        "screenshot": {
            "kind": "set_of_mark",
            "width": 640,
            "height": 360,
            "image_attached": True,
            "annotation": {
                "kind": "set_of_mark",
                "coordinate_frame": "viewport_css",
                "source_screenshot_path": "/device/raw.png",
                "overlay_screenshot_path": "/device/som.png",
                "marks": [],
            },
        },
    }

    context = WebObservationProjector().context_from_payload(
        payload,
        screenshot_url="data:image/png;base64,QUJD",
    )
    assert context is not None
    assert context.observation_id == "post-click-1"
    assert context.settled is False
    assert context.settle_reason == "timeout_best_effort"
    assert context.snapshot.elements[0].model_ref == "@e1"
    assert context.screenshot is not None
    assert context.screenshot.data_url == "data:image/png;base64,QUJD"
    assert context.screenshot.annotation is not None
    assert context.screenshot.annotation.source_screenshot_path == ""
    assert context.screenshot.annotation.overlay_screenshot_path == ""
