"""Desktop remote artifact collection tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from northcua_controller_plugins.desktop_artifacts import DesktopRemoteArtifactCollector


def test_desktop_remote_collector_includes_structured_evaluator_trace(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    round_root = session_root / "evaluations" / "desktop_round_completion" / "round_0001"
    round_root.mkdir(parents=True)
    (round_root / "final_report.json").write_text(json.dumps({"overall_verdict": "success"}), encoding="utf-8")
    (round_root / "evidence_manifest.json").write_text(json.dumps({"refs": []}), encoding="utf-8")
    (round_root / "llm_trace.jsonl").write_text(json.dumps({"phase": "judge"}) + "\n", encoding="utf-8")

    artifacts = DesktopRemoteArtifactCollector().collect(
        _context(task_id="task-1"),
        SimpleNamespace(root=session_root),
    )
    artifact_ids = {artifact.artifact_id for artifact in artifacts}
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}

    assert "task-1-evaluation-desktop_round_completion-round_0001-final_report" in artifact_ids
    assert "task-1-evaluation-desktop_round_completion-round_0001-evidence_manifest" in artifact_ids
    assert "task-1-evaluation-desktop_round_completion-round_0001-llm_trace" in artifact_ids
    trace = artifacts_by_id["task-1-evaluation-desktop_round_completion-round_0001-llm_trace"]
    assert trace.storage_kind.value == "device_local"
    assert trace.inline_data is None
    assert trace.uri.endswith("evaluations/desktop_round_completion/round_0001/llm_trace.jsonl")


def test_desktop_remote_collector_does_not_inline_session_images(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    image = session_root / "query_0001" / "action_step1_history_overlay_1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    artifacts = DesktopRemoteArtifactCollector().collect(
        _context(task_id="task-1"),
        SimpleNamespace(root=session_root),
    )

    assert all(artifact.kind.value != "screenshot" for artifact in artifacts)


def test_desktop_remote_collector_includes_latest_round_verify_screenshot(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    older = session_root / "query_0001" / "action_step1_verify.png"
    latest = session_root / "query_0002" / "action_step3_verify.png"
    older.parent.mkdir(parents=True)
    latest.parent.mkdir(parents=True)
    older.write_bytes(b"\x89PNG\r\n\x1a\nolder")
    latest.write_bytes(b"\x89PNG\r\n\x1a\nlatest")

    artifacts = DesktopRemoteArtifactCollector().collect(
        _context(task_id="task-1"),
        SimpleNamespace(root=session_root),
    )
    screenshots = [artifact for artifact in artifacts if artifact.kind.value == "screenshot"]

    assert len(screenshots) == 1
    assert screenshots[0].artifact_id == "task-1-round-verify-screenshot"
    assert screenshots[0].storage_kind.value == "device_local"
    assert screenshots[0].uri.endswith("query_0002/action_step3_verify.png")
    assert screenshots[0].inline_data is None


def test_desktop_remote_collector_does_not_duplicate_verify_screenshot_to_controller_blob(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    blob_root = tmp_path / "controller-blobs"
    latest = session_root / "query_0002" / "action_step3_verify.png"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"\x89PNG\r\n\x1a\nlatest")

    artifacts = DesktopRemoteArtifactCollector().collect(
        _context(task_id="task-1", blob_store_root=blob_root),
        SimpleNamespace(root=session_root),
    )
    screenshots = [artifact for artifact in artifacts if artifact.kind.value == "screenshot"]

    assert len(screenshots) == 1
    assert screenshots[0].artifact_id == "task-1-round-verify-screenshot"
    assert screenshots[0].storage_kind.value == "device_local"
    assert not blob_root.exists()
    assert screenshots[0].inline_data is None


def _context(*, task_id: str, blob_store_root: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(request=SimpleNamespace(task_id=task_id), blob_store_root=blob_store_root)
