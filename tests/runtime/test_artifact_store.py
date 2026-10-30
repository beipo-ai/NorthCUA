from __future__ import annotations

from northcua.core.config import ArtifactConfig
from northcua.core.enums import AccessibilityRawFormat
from northcua.runtime.artifacts import NorthCUAArtifactStore


def test_artifact_store_paths_and_jsonl(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-1", ArtifactConfig())

    assert store.step_log_path.name == "steps.jsonl"
    assert store.llm_trace_log_path.name == "llm_trace.jsonl"
    assert store.runtime_log_path.name == "runtime.jsonl"
    assert store.round_evaluation_log_path.name == "round_evaluations.jsonl"
    assert store.session_evaluation_rollup_path.name == "session_evaluation.json"
    assert store.eval_runtime_log_path.name == "eval_runtime.jsonl"
    assert store.initial_screenshot_path(2).name == "action_step2.png"
    assert store.initial_screenshot_path(2).parent.name == "query_0001"
    assert store.overlay_path(3, 1).name == "action_step3_history_overlay_1.png"
    assert store.probe_screenshot_path(4, 2).name == "action_step4_probe_3.png"
    assert store.verification_screenshot_path(4).name == "action_step4_verify.png"
    assert store.a11y_snapshot_path(5).name == "a11y_step5.json"
    assert store.a11y_raw_path(5, AccessibilityRawFormat.JSON).name == "a11y_step5.raw.json"
    assert store.a11y_diff_path(5).name == "a11y_step5_diff.json"
    assert store.eval_round_bundle_path(2).name == "eval_round_0002_bundle.json"
    assert store.eval_round_rubric_path(2).name == "eval_round_0002_rubric.json"
    assert store.eval_round_selection_path(2).name == "eval_round_0002_selection.json"
    assert store.eval_round_result_path(2).name == "eval_round_0002_result.json"

    store.append_jsonl(store.step_log_path, {"ok": True})
    assert store.step_log_path.read_text(encoding="utf-8").strip() == '{"ok": true}'

    store.rewrite_jsonl(store.step_log_path, [{"ok": False}, {"nested": {"value": "中文"}}])
    assert store.step_log_path.read_text(encoding="utf-8").splitlines() == [
        '{"ok": false}',
        '{"nested": {"value": "中文"}}',
    ]


def test_artifact_store_query_scoped_step_artifacts(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-1", ArtifactConfig())

    store.set_query_index(3)

    assert store.query_dir().name == "query_0003"
    assert store.query_dir(4).name == "query_0004"
    assert store.initial_screenshot_path(1).parent.name == "query_0003"
    assert store.desktop_screenshot_path(1).parent.name == "query_0003"
    assert store.tool_screenshot_path(1, 0).parent.name == "query_0003"
    assert store.a11y_raw_path(1, AccessibilityRawFormat.XML).parent.name == "query_0003"

    store.set_query_index(0)
    assert store.query_dir().name == "query_0001"
