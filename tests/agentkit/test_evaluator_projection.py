"""Regression tests for low-noise evaluator artifact projection."""

from __future__ import annotations

import json

from northcua.agentkit.evaluator.artifacts import ArtifactPathResolver
from northcua.agentkit.evaluator.evidence import EvidenceSelector
from northcua.agentkit.evaluator.projection import (
    EvaluatorDetailLevel,
    EvaluatorEvidenceProjector,
    EvaluatorEvidenceView,
)
from northcua.agentkit.evaluator.protocol import (
    EvaluationEvidenceManifest,
    EvaluationEvidenceRef,
    RoundEvaluationBundle,
    RoundEvaluationStepRef,
)
from northcua.agentkit.evaluator.runtime.tool_calls import EvaluatorToolCallExecutor
from northcua.agentkit.evaluator.spec import EvidencePolicy
from northcua.agentkit.evaluator.tools import ArtifactInspectionToolFactory
from northcua.agents.desktop.evaluation.prompting import DesktopEvaluatorPromptBuilder
from northcua.core.config import EvaluationPromptBudgetConfig
from northcua.core.enums import EvaluationEvidenceKind


def _projected_factory(tmp_path):
    som_metadata = {
        "kind": "set_of_mark",
        "marks": [
            {
                "ref": "1",
                "label": "1",
                "role": "button",
                "name": "Buy",
                "bbox_css": {"x": 1, "y": 2, "width": 3, "height": 4},
            }
        ],
        "omitted_count": 2,
        "truncated": False,
        "max_marks": 150,
    }
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="Click Buy",
        session_request="Click Buy",
        tool_results=[
            {
                "name": "web_click_ref",
                "action": "click_ref",
                "status": "ok",
                "text": "Clicked 1.",
                "session_step": 1,
                "params": {"ref": "1"},
                "selected_image_path": "/tmp/private/tool_som.png",
                "result": {"ref": "1", "bbox_css": {"x": 1, "y": 2, "width": 3, "height": 4}},
                "observation": {"page": {"url": "https://example.test", "title": "Example"}, "settled": True},
                "observation_recovery": {
                    "status": "recovered",
                    "reason": "stale_or_missing_ref",
                    "refs_invalidated": False,
                },
                "som_image_ref": "E001",
            }
        ],
        screenshot_refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Step 1 SoM",
                summary="clicked buy",
                path="/tmp/private/tool_som.png",
                session_step=1,
                round_step=1,
                metadata={"variant": "som", "som_metadata": som_metadata},
            )
        ],
    )
    return ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=list(bundle.screenshot_refs)),
        step_trace_by_session_step={},
    )


def test_projector_models_roundtrip() -> None:
    view = EvaluatorEvidenceView(
        session_id="session",
        round_num=1,
        round_request="task",
        session_request="task",
        projection={"detail_level": EvaluatorDetailLevel.STANDARD.value},
    )

    assert EvaluatorEvidenceView.model_validate(view.model_dump(mode="json")).session_id == "session"


def test_read_round_bundle_default_is_path_and_bbox_free(tmp_path) -> None:
    factory = _projected_factory(tmp_path)

    projected = factory._read_round_bundle()
    raw = factory._read_round_bundle(detail_level="debug")

    projected_text = json.dumps(projected, ensure_ascii=False)
    raw_text = json.dumps(raw, ensure_ascii=False)
    assert "/tmp/private" not in projected_text
    assert "bbox_css" not in projected_text
    assert "som_metadata" not in projected_text
    assert "marked_count" in projected_text
    assert "/tmp/private" in raw_text
    assert "bbox_css" in raw_text


def test_query_tool_results_compacts_paths_by_default(tmp_path) -> None:
    factory = _projected_factory(tmp_path)

    projected = factory._query_tool_results(tool_name="web_click_ref")
    raw = factory._query_tool_results(tool_name="web_click_ref", detail_level="debug")

    projected_text = json.dumps(projected, ensure_ascii=False)
    raw_text = json.dumps(raw, ensure_ascii=False)
    assert projected["matches"][0]["name"] == "web_click_ref"
    assert projected["matches"][0]["evidence_refs"]["som_image_ref"] == "E001"
    assert projected["matches"][0]["recovery"] == {
        "status": "recovered",
        "reason": "stale_or_missing_ref",
        "refs_invalidated": False,
    }
    assert "/tmp/private" not in projected_text
    assert "bbox_css" not in projected_text
    assert "/tmp/private" in raw_text
    assert "bbox_css" in raw_text


def test_query_som_annotation_geometry_requires_explicit_ref_filter(tmp_path) -> None:
    factory = _projected_factory(tmp_path)

    projected = factory._query_som_annotation(session_step=1)
    with_geometry = factory._query_som_annotation(
        session_step=1,
        include_geometry=True,
        ref_filter=["1"],
    )

    assert projected["marked_count"] == 1
    assert projected["marks"][0]["ref"] == "1"
    assert "bbox_css" not in projected["marks"][0]
    assert with_geometry["geometry_policy"] == "selected_refs"
    assert with_geometry["marks"][0]["bbox_css"]["x"] == 1


def test_query_a11y_snapshot_default_strips_bounds(tmp_path) -> None:
    a11y_path = tmp_path / "a11y.json"
    a11y_path.write_text(
        json.dumps(
            {
                "tree": {
                    "role": "button",
                    "name": "Submit",
                    "bounds": {"x": 1, "y": 2, "width": 3, "height": 4},
                    "children": [],
                }
            }
        ),
        encoding="utf-8",
    )
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="Submit",
        session_request="Submit",
        a11y_refs=[
            EvaluationEvidenceRef(
                id="A001",
                kind=EvaluationEvidenceKind.A11Y,
                label="Step 1 a11y",
                path=str(a11y_path),
                session_step=1,
            )
        ],
    )
    factory = ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=list(bundle.a11y_refs)),
        step_trace_by_session_step={},
    )

    projected = factory._query_a11y_snapshot(evidence_ref="A001")
    raw = factory._query_a11y_snapshot(evidence_ref="A001", detail_level="debug")

    projected_text = json.dumps(projected, ensure_ascii=False)
    raw_text = json.dumps(raw, ensure_ascii=False)
    assert "Submit" in projected_text
    assert "bounds" not in projected_text
    assert "bounds" in raw_text


def test_projector_redacts_mixed_case_data_urls(tmp_path) -> None:
    factory = _projected_factory(tmp_path)
    projector = EvaluatorEvidenceProjector(bundle=factory.bundle, evidence_manifest=factory.evidence_manifest)

    payload = projector.model_facing_object(
        {
            "screenshot_url": "data:IMAGE/PNG;base64,abc",
            "nested": {"data_url": "data:image/png;base64,abc"},
            "page": {"url": "https://example.test/path"},
        }
    )

    text = json.dumps(payload, ensure_ascii=False)
    assert "base64" not in text
    assert "screenshot_url" not in text
    assert "data_url" not in text
    assert payload["page"]["url"] == "https://example.test/path"


def test_projector_omits_replayed_image_url_placeholders(tmp_path) -> None:
    factory = _projected_factory(tmp_path)
    projector = EvaluatorEvidenceProjector(bundle=factory.bundle, evidence_manifest=factory.evidence_manifest)

    payload = projector.model_facing_object(
        {
            "trace": {
                "tool_results": [
                    {
                        "overlay_image_url": "[redacted image data url; chars=100]",
                        "image_url": "data:image/png;base64,abc",
                        "page_url": "https://example.test/ok",
                    }
                ]
            }
        }
    )

    text = json.dumps(payload, ensure_ascii=False)
    assert "redacted image data url" not in text
    assert "data:image" not in text
    assert "overlay_image_url" not in text
    assert "image_url" not in text
    assert "page_url" in text


def test_evaluator_tool_execution_separates_raw_and_model_outputs() -> None:
    raw_output = {
        "status": "success",
        "trace": {
            "tool_results": [
                {
                    "overlay_image_url": "[redacted image data url; chars=100]",
                    "image_url": "data:image/png;base64,abc",
                    "page_url": "https://example.test/ok",
                }
            ]
        },
    }

    record = EvaluatorToolCallExecutor._record(
        call_id="call_1",
        tool_name="read_step",
        tool_input={},
        output=raw_output,
        is_error=False,
    )

    raw_text = json.dumps(record.tool_output, ensure_ascii=False)
    model_text = json.dumps(record.llm_tool_output, ensure_ascii=False)
    assert "redacted image data url" in raw_text
    assert "data:image" in raw_text
    assert "redacted image data url" not in model_text
    assert "data:image" not in model_text
    assert "overlay_image_url" not in model_text
    assert "image_url" not in model_text
    assert "page_url" in model_text


def test_debug_tool_payload_is_not_replaced_by_budget_preview(tmp_path) -> None:
    factory = _projected_factory(tmp_path)
    large_payload = {"status": "success", "items": [{"index": index} for index in range(80)]}
    bounded = factory._bounded_implementation(lambda detail_level="standard": large_payload)

    standard = bounded(detail_level="standard")
    debug = bounded(detail_level="debug")

    assert len(standard["items"]) < 80
    assert standard["items"][-1]["truncated"] is True
    assert len(debug["items"]) == 80


def test_unselected_sibling_image_ref_can_still_resolve(tmp_path) -> None:
    som_path = tmp_path / "tool_som.png"
    raw_path = tmp_path / "tool_raw.png"
    som_path.write_bytes(b"not-a-real-image")
    raw_path.write_bytes(b"not-a-real-image")
    selected_ref = EvaluationEvidenceRef(
        id="E001",
        kind=EvaluationEvidenceKind.IMAGE,
        label="Step 1 SoM",
        path=str(som_path),
        session_step=1,
        metadata={"variant": "som"},
    )
    raw_ref = EvaluationEvidenceRef(
        id="E002",
        kind=EvaluationEvidenceKind.IMAGE,
        label="Step 1 raw",
        path=str(raw_path),
        session_step=1,
        metadata={"variant": "raw"},
    )
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="Click",
        session_request="Click",
        screenshot_refs=[selected_ref, raw_ref],
        tool_results=[
            {
                "name": "web_click_ref",
                "status": "ok",
                "raw_image_ref": "E002",
                "som_image_ref": "E001",
            }
        ],
    )
    factory = ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=[selected_ref]),
        step_trace_by_session_step={},
    )

    assert factory._image_path_for_ref("E002") == raw_path
    projected = factory._query_tool_results(tool_name="web_click_ref")
    assert projected["matches"][0]["evidence_refs"]["raw_image_ref"] == "E002"


def test_artifact_path_resolver_rebases_copied_artifact_suffixes(tmp_path) -> None:
    screenshot = tmp_path / "screenshots" / "step_001.png"
    screenshot.parent.mkdir()
    screenshot.write_bytes(b"png")
    stale_absolute = "/old/worktree/session-artifacts/screenshots/step_001.png"

    resolver = ArtifactPathResolver(tmp_path)

    assert resolver.existing_file(stale_absolute) == screenshot.resolve()


def test_artifact_path_resolver_rebases_top_level_copied_artifact_files(tmp_path) -> None:
    screenshot = tmp_path / "action_step1_probe_1.png"
    screenshot.write_bytes(b"png")
    stale_absolute = "/old/worktree/session-artifacts/action_step1_probe_1.png"

    resolver = ArtifactPathResolver(tmp_path)

    assert resolver.existing_file(stale_absolute) == screenshot.resolve()


def test_artifact_path_resolver_rebases_desktop_query_artifact_suffixes(tmp_path) -> None:
    screenshot = tmp_path / "query_0001" / "action_step1_probe_1.png"
    screenshot.parent.mkdir()
    screenshot.write_bytes(b"png")
    stale_absolute = "/old/worktree/session-artifacts/query_0001/action_step1_probe_1.png"

    resolver = ArtifactPathResolver(tmp_path)

    assert resolver.existing_file(stale_absolute) == screenshot.resolve()


def test_artifact_path_resolver_rejects_non_artifact_absolute_paths(tmp_path) -> None:
    secret = tmp_path.parent / "outside-secret.txt"
    secret.write_text("secret", encoding="utf-8")

    resolver = ArtifactPathResolver(tmp_path)

    assert resolver.existing_file(str(secret)) is None


def test_artifact_path_resolver_does_not_rebase_unrecognized_basename_collisions(tmp_path) -> None:
    in_root = tmp_path / "outside.png"
    in_root.write_bytes(b"png")
    stale_absolute = "/old/worktree/outside.png"

    resolver = ArtifactPathResolver(tmp_path)

    assert resolver.existing_file(stale_absolute) is None


def test_projector_compact_bundle_is_path_free(tmp_path) -> None:
    factory = _projected_factory(tmp_path)
    projector = EvaluatorEvidenceProjector(bundle=factory.bundle, evidence_manifest=factory.evidence_manifest)

    payload = projector.compact_bundle()

    assert payload["projection"]["path_free"] is True
    assert payload["evidence_counts"]["screenshots"] == 1
    assert "/tmp/private" not in json.dumps(payload, ensure_ascii=False)


def test_evidence_selector_prefers_clean_over_action_overlay_and_som_within_budget() -> None:
    refs = [
        EvaluationEvidenceRef(
            id="initial",
            kind=EvaluationEvidenceKind.IMAGE,
            label="initial",
            session_step=1,
            metadata={"variant": "initial_state"},
        ),
        EvaluationEvidenceRef(
            id="current2",
            kind=EvaluationEvidenceKind.IMAGE,
            label="current step 2",
            session_step=2,
            metadata={"variant": "current"},
        ),
        EvaluationEvidenceRef(
            id="action-overlay2",
            kind=EvaluationEvidenceKind.IMAGE,
            label="tool action overlay step 2",
            session_step=2,
            metadata={"variant": "action_overlay", "tool_name": "web_click_ref"},
        ),
        EvaluationEvidenceRef(
            id="som2",
            kind=EvaluationEvidenceKind.IMAGE,
            label="tool SoM step 2",
            session_step=2,
            metadata={"variant": "som", "tool_name": "web_click_ref"},
        ),
        EvaluationEvidenceRef(
            id="terminal",
            kind=EvaluationEvidenceKind.IMAGE,
            label="terminal",
            session_step=3,
            metadata={"variant": "terminal_verification"},
        ),
    ]
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="task",
        session_request="task",
        screenshot_refs=refs,
    )

    manifest = EvidenceSelector(
        EvidencePolicy(
            screenshot_max=3,
            include_initial_screenshot=True,
            include_last_screenshot=True,
            include_pinned_evidence=True,
        )
    ).select(bundle=bundle)

    assert [ref.id for ref in manifest.refs] == ["initial", "terminal", "current2"]


def test_desktop_prompt_uses_projected_payload(tmp_path) -> None:
    factory = _projected_factory(tmp_path)

    messages = DesktopEvaluatorPromptBuilder(artifact_root=tmp_path).build_messages(
        bundle=factory.bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=[]),
        criteria=[],
    )

    text = "\n".join(block.text for message in messages for block in message.content if getattr(block, "text", None))
    assert "/tmp/private" not in text
    assert "bbox_css" not in text
    assert "path_free" in text


def test_desktop_prompt_tightens_long_trace_and_omits_duplicate_text(tmp_path) -> None:
    long_request = "repeat the long task request " * 200
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request=long_request,
        session_request=long_request,
        step_refs=[
            RoundEvaluationStepRef(
                session_step=index + 1,
                round_step=index + 1,
                status="continue",
                request=long_request,
                result=f"intermediate result {index} " * 20,
                assistant_text=f"assistant text {index} " * 20,
            )
            for index in range(400)
        ],
        step_summaries=[f"summary {index} " * 20 for index in range(400)],
        assistant_texts=["duplicated assistant text " * 200],
        tool_calls=[{"name": "tool", "arguments": {"request": long_request}} for _ in range(120)],
        tool_results=[{"name": "tool", "result": "result " * 100} for _ in range(120)],
    )
    builder = DesktopEvaluatorPromptBuilder(
        artifact_root=tmp_path,
        prompt_budget=EvaluationPromptBudgetConfig(
            target_tokens=1_024,
            hard_tokens=75_000,
            final_full_steps=20,
        ),
    )

    result = builder.build_prompt(
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(),
        criteria=[],
    )

    text = "\n".join(
        block.text for message in result.messages for block in message.content if getattr(block, "text", None)
    )
    assert result.projection_level == "tight"
    assert result.estimated_tokens < result.hard_tokens
    assert '"assistant_texts"' not in text
    assert "assistant_summary" in text
    payload = EvaluatorEvidenceProjector(
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(),
    ).compact_bundle(projection_level="tight")
    assert all("request" not in step for step in payload["step_refs"])


def test_artifact_tool_does_not_expose_assistant_texts_section(tmp_path) -> None:
    bundle = RoundEvaluationBundle(
        session_id="session",
        round_num=1,
        round_request="task",
        session_request="task",
        step_refs=[
            RoundEvaluationStepRef(
                session_step=1,
                round_step=1,
                status="finish",
                request="task",
                result="done",
                assistant_text="hidden full assistant transcript",
            )
        ],
        assistant_texts=["hidden full assistant transcript"],
    )
    factory = ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(),
        step_trace_by_session_step={},
    )

    all_payload = factory._read_round_bundle(section="all")
    messages_payload = factory._read_round_bundle(section="messages")
    explicit_payload = factory._read_round_bundle(section="assistant_texts")

    assert "assistant_texts" not in json.dumps(all_payload, ensure_ascii=False)
    assert messages_payload["section"] == "trace"
    assert "assistant_texts" not in json.dumps(messages_payload, ensure_ascii=False)
    assert explicit_payload["status"] == "failure"
