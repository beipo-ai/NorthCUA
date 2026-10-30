"""Regression coverage for desktop evaluator bundles and evidence selection."""

from __future__ import annotations

import json
from pathlib import Path

from northcua.agentkit.evaluator import EvidencePolicy
from northcua.agents.desktop.evaluation.bundle import BundleBuilder
from northcua.agents.desktop.evaluation.protocol import (
    EvaluationCriterion,
    EvaluationEvidenceManifest,
    EvaluationEvidenceRef,
    EvaluationJudgeMetadata,
    RoundEvaluationRecord,
)
from northcua.agents.desktop.evaluation.selector import EvidenceSelector
from northcua.core.accessibility import AccessibilityDiffSummary, AccessibilityStepRecord
from northcua.core.config import ArtifactConfig, NorthCUAAppConfig
from northcua.core.enums import (
    AccessibilityCaptureStatus,
    AccessibilityDiffMode,
    AccessibilityScope,
    AccessibilitySourceKind,
    EvaluationControllability,
    EvaluationEvidenceKind,
    EvaluationOutcome,
    EvaluationVerdict,
    ObservationArtifactKind,
    TaskStatus,
)
from northcua.core.models import ObservationArtifact
from northcua.runtime.artifacts.store import NorthCUAArtifactStore
from tests.cua.evaluation_helpers import image, step


def test_bundle_builder_groups_round_steps_and_prior_summaries(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-eval", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    builder = BundleBuilder(artifact_store=store, config=app_config.evaluation.bundle)

    store.append_jsonl(
        store.round_evaluation_log_path,
        RoundEvaluationRecord(
            session_id="sess-eval",
            round_num=1,
            round_request="Open Calculator",
            overall_verdict=EvaluationVerdict.SUCCESS,
            process_rating=5,
            outcome_label=EvaluationOutcome.SUCCESS,
            completion_rating=5,
            confidence_rating=5,
            controllability=EvaluationControllability.CONTROLLABLE,
            reason="Completed.",
            evidence_manifest=EvaluationEvidenceManifest(),
            judge_metadata=EvaluationJudgeMetadata(),
        ).model_dump(mode="json"),
    )

    current = image(store.initial_screenshot_path(2), (255, 0, 0))
    tool = image(store.tool_screenshot_path(2, 0), (0, 255, 0))
    a11y_path = str(store.a11y_snapshot_path(2))
    store.write_text(Path(a11y_path), json.dumps({"status": "success"}))
    store.append_jsonl(
        store.runtime_log_path,
        {
            "message": "round_stop_decision_made",
            "extra": {"session_step": 2, "round_step": 1, "event": "round_stop_decision_made"},
        },
    )

    step_logs = [
        step(
            session_id="sess-eval",
            session_step=1,
            round_num=1,
            round_step=1,
            request="Open Calculator",
            result="done",
            status=TaskStatus.FINISH,
            clean_screenshot_path=current,
            tool_image_path=tool,
            a11y_path=a11y_path,
        ),
        step(
            session_id="sess-eval",
            session_step=2,
            round_num=2,
            round_step=1,
            request="Compute 12 + 34",
            result="continue",
            status=TaskStatus.CONTINUE,
            clean_screenshot_path=current,
            tool_image_path=tool,
            a11y_path=a11y_path,
        ),
    ]
    step_logs[1].tool_results[0].image_url = "data:image/png;base64,abc"
    step_logs[1].tool_results[0].overlay_image_url = "data:image/png;base64,def"
    overlay = image(store.tool_screenshot_path(2, 1), (0, 0, 255))
    step_logs[1].tool_results[0].overlay_image_path = overlay

    bundle = builder.build(round_num=2, round_request="Compute 12 + 34", step_logs=step_logs)

    assert bundle.round_num == 2
    assert bundle.session_request == "Open Calculator"
    assert len(bundle.step_refs) == 1
    assert len(bundle.screenshot_refs) >= 2
    assert len(bundle.a11y_refs) == 1
    assert len(bundle.runtime_event_refs) == 1
    assert bundle.prior_round_summaries[0].round_num == 1
    assert any(ref.path == overlay and ref.metadata.get("variant") == "overlay" for ref in bundle.screenshot_refs)
    assert bundle.tool_results[0]["image_url"] == "[image data omitted; chars=25]"
    assert bundle.tool_results[0]["overlay_image_url"] == "[image data omitted; chars=25]"


def test_bundle_builder_marks_first_clean_screenshot_as_initial_state(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-eval-initial", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    builder = BundleBuilder(artifact_store=store, config=app_config.evaluation.bundle)
    first_path = image(store.initial_screenshot_path(1), (80, 120, 160))
    second_path = image(store.initial_screenshot_path(2), (20, 40, 60))
    step_logs = [
        step(
            session_id="sess-eval-initial",
            session_step=1,
            round_num=1,
            round_step=1,
            request="Read Calculator",
            result="continue",
            status=TaskStatus.CONTINUE,
            clean_screenshot_path=first_path,
            tool_image_path="",
            a11y_path="",
        ),
        step(
            session_id="sess-eval-initial",
            session_step=2,
            round_num=1,
            round_step=2,
            request="Read Calculator",
            result="done",
            status=TaskStatus.FINISH,
            clean_screenshot_path=second_path,
            tool_image_path="",
            a11y_path="",
        ),
    ]

    bundle = builder.build(round_num=1, round_request="Read Calculator", step_logs=step_logs)

    first_ref = next(ref for ref in bundle.screenshot_refs if ref.path == first_path)
    second_ref = next(ref for ref in bundle.screenshot_refs if ref.path == second_path)
    assert first_ref.metadata["variant"] == "initial_state"
    assert first_ref.metadata["initial_state"] is True
    assert "initial screenshot" in first_ref.label
    assert "initial_state" not in second_ref.metadata


def test_bundle_builder_filters_prompt_runtime_events(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-eval", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    builder = BundleBuilder(artifact_store=store, config=app_config.evaluation.bundle)
    for event_name in (
        "round_state_loaded",
        "round_state_hydrated",
        "round_started",
        "round_history_replay_built",
        "llm_fallback_attempt_started",
        "round_tool_surface_ready",
        "round_prompt_envelope_built",
        "llm_fallback_attempt_selected",
        "round_trace_record_built",
        "round_external_tool_pause_received",
        "round_remote_external_tools_completed",
        "round_state_saved",
        "round_evaluation_skipped",
        "round_resume_results_replayed",
        "round_step_persisted",
    ):
        store.append_jsonl(
            store.runtime_log_path,
            {"message": event_name, "extra": {"session_step": 1, "round_step": 1, "event": event_name}},
        )
    for event_name in (
        "round_snapshot_ready",
        "round_stop_decision_made",
        "tool_result_artifact_built",
        "round_end_verification_capture_completed",
        "round_terminal_status_reached",
    ):
        store.append_jsonl(
            store.runtime_log_path,
            {"message": event_name, "extra": {"session_step": 1, "round_step": 1, "event": event_name}},
        )

    bundle = builder.build(
        round_num=1,
        round_request="Edit image",
        step_logs=[
            step(
                session_id="sess-eval",
                session_step=1,
                round_num=1,
                round_step=1,
                request="Edit image",
                result="done",
                status=TaskStatus.FINISH,
                clean_screenshot_path=image(store.initial_screenshot_path(1), (255, 0, 0)),
                tool_image_path=image(store.tool_screenshot_path(1, 0), (0, 255, 0)),
                a11y_path=str(store.a11y_snapshot_path(1)),
            )
        ],
    )

    assert [ref.metadata["event"] for ref in bundle.runtime_event_refs] == [
        "round_snapshot_ready",
        "round_stop_decision_made",
        "tool_result_artifact_built",
        "round_end_verification_capture_completed",
        "round_terminal_status_reached",
    ]


def test_bundle_builder_aggregates_repeated_unavailable_a11y_refs(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-eval", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    builder = BundleBuilder(artifact_store=store, config=app_config.evaluation.bundle)
    records = []
    for session_step in range(1, 4):
        base = step(
            session_id="sess-eval",
            session_step=session_step,
            round_num=1,
            round_step=session_step,
            request="Edit image",
            result="continue",
            status=TaskStatus.CONTINUE,
            clean_screenshot_path=image(store.initial_screenshot_path(session_step), (255, 0, 0)),
            tool_image_path=image(store.tool_screenshot_path(session_step, 0), (0, 255, 0)),
            a11y_path="",
        )
        records.append(
            base.model_copy(
                update={
                    "accessibility": AccessibilityStepRecord(
                        status=AccessibilityCaptureStatus.UNSUPPORTED,
                        source=AccessibilitySourceKind.ATSPI,
                        scope=AccessibilityScope.DESKTOP,
                        error="AT-SPI backend unavailable",
                    )
                }
            )
        )

    bundle = builder.build(round_num=1, round_request="Edit image", step_logs=records)

    assert len(bundle.a11y_refs) == 1
    ref = bundle.a11y_refs[0]
    assert ref.label == "Accessibility unsupported summary"
    assert ref.path == ""
    assert ref.metadata["aggregated"] is True
    assert ref.metadata["affected_steps"] == [1, 2, 3]
    assert ref.metadata["original_ref_count"] == 3
    assert "3 steps" in ref.summary


def test_bundle_builder_keeps_successful_a11y_refs_per_step(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-eval", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    builder = BundleBuilder(artifact_store=store, config=app_config.evaluation.bundle)
    records = [
        step(
            session_id="sess-eval",
            session_step=session_step,
            round_num=1,
            round_step=session_step,
            request="Edit image",
            result="continue",
            status=TaskStatus.CONTINUE,
            clean_screenshot_path=image(store.initial_screenshot_path(session_step), (255, 0, 0)),
            tool_image_path=image(store.tool_screenshot_path(session_step, 0), (0, 255, 0)),
            a11y_path=str(store.a11y_snapshot_path(session_step)),
        )
        for session_step in range(1, 3)
    ]

    bundle = builder.build(round_num=1, round_request="Edit image", step_logs=records)

    assert len(bundle.a11y_refs) == 2
    assert all(ref.metadata.get("aggregated") is None for ref in bundle.a11y_refs)
    assert [ref.session_step for ref in bundle.a11y_refs] == [1, 2]


def test_bundle_builder_includes_accessibility_diff_artifact_refs(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-eval", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    builder = BundleBuilder(artifact_store=store, config=app_config.evaluation.bundle)
    diff_path = store.a11y_diff_path(1)
    store.write_text(diff_path, json.dumps({"added_nodes": 2, "focus_changed": True}))
    base = step(
        session_id="sess-eval",
        session_step=1,
        round_num=1,
        round_step=1,
        request="Edit image",
        result="continue",
        status=TaskStatus.CONTINUE,
        clean_screenshot_path=image(store.initial_screenshot_path(1), (255, 0, 0)),
        tool_image_path=image(store.tool_screenshot_path(1, 0), (0, 255, 0)),
        a11y_path=str(store.a11y_snapshot_path(1)),
    )
    record = base.model_copy(
        update={
            "accessibility": base.accessibility.model_copy(
                update={
                    "diff_mode": AccessibilityDiffMode.BASIC,
                    "diff": AccessibilityDiffSummary(added_nodes=2, removed_nodes=1, focus_changed=True),
                }
            ),
            "observation_artifacts": [
                ObservationArtifact(
                    kind=ObservationArtifactKind.ACCESSIBILITY_DIFF,
                    role="accessibility_diff",
                    source="accessibility_diff_builder",
                    path=str(diff_path),
                    metadata={"added_nodes": 2, "focus_changed": True},
                )
            ],
        }
    )

    bundle = builder.build(round_num=1, round_request="Edit image", step_logs=[record])
    diff_refs = [ref for ref in bundle.a11y_refs if ref.metadata.get("variant") == "accessibility_diff"]

    assert len(diff_refs) == 1
    assert diff_refs[0].path == str(diff_path)
    assert "changed 3 nodes" in diff_refs[0].summary
    assert diff_refs[0].metadata["focus_changed"] is True


def test_evidence_selector_limits_images_but_keeps_non_images() -> None:
    selector = EvidenceSelector(
        NorthCUAAppConfig(agent_config_path="agent.yaml").evaluation.selector,
        policy=EvidencePolicy(screenshot_max=2),
    )
    criteria = [
        EvaluationCriterion(id="criterion_1", name="Click Button", description="Use the right button", weight=1.0)
    ]
    bundle = type(
        "_Bundle",
        (),
        {
            "screenshot_refs": [
                EvaluationEvidenceRef(
                    id=f"E{i:03d}",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label=f"button state {i}",
                    summary="button screenshot",
                    path=f"/tmp/{i}.png",
                    session_step=i,
                    round_step=i,
                )
                for i in range(1, 12)
            ],
            "a11y_refs": [
                EvaluationEvidenceRef(
                    id="EA11Y",
                    kind=EvaluationEvidenceKind.A11Y,
                    label="Focused accessibility snapshot",
                    summary="focus on button",
                    path="/tmp/a11y.json",
                    session_step=1,
                    round_step=1,
                )
            ],
            "runtime_event_refs": [],
        },
    )()

    manifest = selector.select(bundle=bundle, criteria=criteria)

    image_refs = [ref for ref in manifest.refs if ref.kind == EvaluationEvidenceKind.IMAGE]
    assert len(image_refs) == 2
    assert any(ref.kind == EvaluationEvidenceKind.A11Y for ref in manifest.refs)


def test_evidence_selector_caps_non_image_refs() -> None:
    config = NorthCUAAppConfig(agent_config_path="agent.yaml").evaluation.selector.model_copy(
        update={"max_non_image_refs": 4}
    )
    selector = EvidenceSelector(config, policy=EvidencePolicy(screenshot_max=1))
    bundle = type(
        "_Bundle",
        (),
        {
            "screenshot_refs": [
                EvaluationEvidenceRef(
                    id="EIMG",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label="terminal image",
                    path="/tmp/final.png",
                    session_step=8,
                    round_step=8,
                    metadata={"variant": "terminal_verification", "pinned": True},
                )
            ],
            "a11y_refs": [
                EvaluationEvidenceRef(
                    id=f"EA{i}",
                    kind=EvaluationEvidenceKind.A11Y,
                    label=f"a11y {i}",
                    session_step=i,
                    round_step=i,
                )
                for i in range(1, 9)
            ],
            "runtime_event_refs": [],
        },
    )()

    manifest = selector.select(bundle=bundle, criteria=[])

    assert [ref.id for ref in manifest.refs if ref.kind != EvaluationEvidenceKind.IMAGE] == [
        "EA1",
        "EA2",
        "EA7",
        "EA8",
    ]
    assert any(ref.id == "EIMG" for ref in manifest.refs)
