"""Experiment trace verifier tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
from northcua.experiments.contracts import TraceExpectationSpec
from tests.support.experiment_fixtures import golden_e2b_experiment_spec, run_fixture_experiment
from northcua.experiments.verification import ExperimentTraceVerifier, FakeTraceSeeder
from tests.experiments.trace_verifier_test_utils import (
    move_seeded_artifact_to_controller_blob as _move_seeded_artifact_to_controller_blob,
)
from tests.experiments.trace_verifier_test_utils import seed_agent_artifact_session as _seed_agent_artifact_session
from tests.experiments.trace_verifier_test_utils import seed_controller_artifact_db as _seed_controller_artifact_db


def _batch_result_for_golden(spec) -> BatchRunResult:
    states = [
        BatchTaskState(
            task_id=task.task_id,
            controller_task_id=f"controller-{task.task_id}",
            trace_id=f"trace-{task.task_id}",
            session_id=f"session-{task.task_id}",
            sandbox_id=f"sandbox-{task.task_id}",
            status=BatchTaskStatus.SUCCEEDED,
            attempts=1,
        )
        for task in spec.tasks
    ]
    return BatchRunResult(
        batch_id=spec.experiment_id,
        total=len(states),
        succeeded=len(states),
        failed=0,
        canceled=0,
        output_dir="",
        task_states=states,
    )


def test_trace_verifier_passes_seeded_fake_artifacts(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    batch_result = _batch_result_for_golden(spec)
    FakeTraceSeeder().seed(output_dir=tmp_path, spec=spec, batch_result=batch_result)

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)

    assert len(rows) == 3
    assert all(row.passed for row in rows)
    assert all(Path(row.evaluator_report_path).exists() for row in rows)


def test_trace_verifier_reports_missing_evaluator_trace(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    batch_result = _batch_result_for_golden(spec)
    FakeTraceSeeder().seed(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    missing = next(tmp_path.rglob(f"evaluations/{spec.trace_expectations.evaluator_kind}/round_0001/llm_trace.jsonl"))
    missing.unlink()

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)

    assert any("evaluator llm_trace.jsonl missing" in error for row in rows for error in row.errors)


def test_trace_verifier_rejects_corrupt_image_payloads(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    batch_result = _batch_result_for_golden(spec)
    FakeTraceSeeder().seed(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    first = batch_result.task_states[0]
    screenshot = next((tmp_path / "sessions" / first.session_id).glob("*.png"))
    screenshot.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRnot-a-real-image")

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    row = next(item for item in rows if item.task_id == first.task_id)

    assert not row.passed
    assert "main-agent screenshot missing" in row.errors


def test_trace_verifier_rejects_evaluator_images_as_main_screenshots(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    batch_result = _batch_result_for_golden(spec)
    FakeTraceSeeder().seed(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    first = batch_result.task_states[0]
    session_root = tmp_path / "sessions" / first.session_id
    for screenshot in session_root.glob("*.png"):
        screenshot.unlink()
    evaluator_screenshot = (
        session_root / "evaluations" / spec.trace_expectations.evaluator_kind / "round_0001" / "evidence.png"
    )
    evaluator_screenshot.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
        b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    row = next(item for item in rows if item.task_id == first.task_id)

    assert not row.passed
    assert "main-agent screenshot missing" in row.errors


def test_trace_verifier_rejects_decompression_bomb_images(tmp_path: Path) -> None:
    path = tmp_path / "bomb.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        + (100000).to_bytes(4, "big")
        + (100000).to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )

    assert not ExperimentTraceVerifier._is_valid_image(path)


def test_trace_verifier_rejects_pillow_syntax_error_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    def _raise_syntax_error(_path: Path) -> object:
        raise SyntaxError("broken PNG file")

    monkeypatch.setattr(Image, "open", _raise_syntax_error)

    assert not ExperimentTraceVerifier._is_valid_image(path)


def test_trace_verifier_does_not_borrow_other_task_artifacts(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    batch_result = _batch_result_for_golden(spec)
    FakeTraceSeeder().seed(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    missing_state = batch_result.task_states[0]
    missing_root = tmp_path / "sessions" / missing_state.session_id
    for path in sorted(missing_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    missing_row = next(row for row in rows if row.task_id == missing_state.task_id)

    assert not missing_row.passed
    assert "main-agent steps.jsonl missing or empty" in missing_row.errors
    assert not missing_row.main_steps_path


def test_trace_verifier_requires_success_evaluator_verdict(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    batch_result = _batch_result_for_golden(spec)
    FakeTraceSeeder().seed(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    report = next(tmp_path.rglob(f"evaluations/{spec.trace_expectations.evaluator_kind}/round_0001/final_report.json"))
    report.write_text('{"overall_verdict":"failure"}', encoding="utf-8")

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)

    assert any("overall_verdict is not success" in error for row in rows for error in row.errors)


def test_trace_verifier_materializes_controller_inline_artifacts(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    task = spec.tasks[0]
    state = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    batch_result = BatchRunResult(
        batch_id=spec.experiment_id,
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[state],
    )
    _seed_controller_artifact_db(tmp_path / "controller.sqlite", state)

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    row = next(item for item in rows if item.task_id == task.task_id)

    assert row.passed
    assert Path(row.main_steps_path).is_file()
    assert Path(row.main_llm_trace_path).is_file()
    assert Path(row.evaluator_report_path).is_file()
    assert Path(row.evaluator_evidence_manifest_path).is_file()
    assert Path(row.evaluator_llm_trace_path).is_file()
    assert row.screenshot_paths and all(Path(path).is_file() for path in row.screenshot_paths)
    assert len(row.screenshot_paths) == len(set(row.screenshot_paths))
    assert any(
        path.endswith("logs/session-1/query_0001/action_step1_history_overlay_1.png") for path in row.screenshot_paths
    )


def test_trace_verifier_prefers_agent_artifacts_without_duplicate_materialization(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    task = spec.tasks[0]
    state = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    batch_result = BatchRunResult(
        batch_id=spec.experiment_id,
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[state],
    )
    _seed_controller_artifact_db(tmp_path / "controller.sqlite", state)
    session_root = _seed_agent_artifact_session(tmp_path, spec, state)

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    row = next(item for item in rows if item.task_id == task.task_id)

    assert row.passed
    assert Path(row.main_steps_path).resolve() == (session_root / "steps.jsonl").resolve()
    assert (
        Path(row.evaluator_report_path).resolve()
        == (
            session_root / "evaluations" / spec.trace_expectations.evaluator_kind / "round_0001" / "final_report.json"
        ).resolve()
    )
    assert row.screenshot_paths
    assert all("agent-artifacts" in path for path in row.screenshot_paths)
    assert not (tmp_path / "materialized").exists()


def test_trace_verifier_accepts_web_agent_trace_as_main_steps(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake").model_copy(
        update={
            "trace_expectations": TraceExpectationSpec(evaluator_kind="web_round_evaluator"),
            "tasks": [
                task.model_copy(update={"agent_kind": "web"})
                for task in golden_e2b_experiment_spec(provider="fake").tasks
            ],
        }
    )
    task = spec.tasks[0]
    state = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    batch_result = BatchRunResult(
        batch_id=spec.experiment_id,
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[state],
    )
    session_root = _seed_agent_artifact_session(tmp_path, spec, state)
    (session_root / "steps.jsonl").unlink()
    (session_root / "web_agent_trace.jsonl").write_text('{"step":1,"action":"web_open"}\n', encoding="utf-8")

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    row = next(item for item in rows if item.task_id == task.task_id)

    assert row.passed
    assert Path(row.main_steps_path).resolve() == (session_root / "web_agent_trace.jsonl").resolve()


def test_fake_trace_seeder_uses_agent_kind_for_web_trace_sources(tmp_path: Path) -> None:
    base = golden_e2b_experiment_spec(provider="fake")
    spec = base.model_copy(
        update={
            "trace_expectations": TraceExpectationSpec(evaluator_kind="web_round_evaluator"),
            "tasks": [
                base.tasks[0].model_copy(update={"agent_kind": "web", "task_id": "browser-task-without-web-prefix"})
            ],
        }
    )
    batch_result = _batch_result_for_golden(spec)

    FakeTraceSeeder().seed(output_dir=tmp_path, spec=spec, batch_result=batch_result)

    session_root = tmp_path / "sessions" / batch_result.task_states[0].session_id
    main_trace = json.loads((session_root / "llm_trace.jsonl").read_text(encoding="utf-8").splitlines()[0])
    evaluator_trace = json.loads(
        (session_root / "evaluations" / "web_round_evaluator" / "round_0001" / "llm_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert main_trace["source"] == "web"
    assert evaluator_trace["source"] == "web_evaluator"


def test_trace_verifier_materializes_controller_web_trace_artifact(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake").model_copy(
        update={
            "trace_expectations": TraceExpectationSpec(evaluator_kind="web_round_evaluator"),
            "tasks": [
                task.model_copy(update={"agent_kind": "web"})
                for task in golden_e2b_experiment_spec(provider="fake").tasks
            ],
        }
    )
    task = spec.tasks[0]
    state = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    batch_result = BatchRunResult(
        batch_id=spec.experiment_id,
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[state],
    )
    db_path = tmp_path / "controller.sqlite"
    _seed_controller_artifact_db(db_path, state)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE artifacts SET original_artifact_id=? WHERE original_artifact_id=?",
            (f"{state.controller_task_id}-web-trace", f"{state.controller_task_id}-steps"),
        )
        connection.execute(
            "UPDATE artifacts SET original_artifact_id=replace(original_artifact_id, ?, ?)",
            ("desktop_round_completion", "web_round_evaluator"),
        )
        connection.commit()

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    row = next(item for item in rows if item.task_id == task.task_id)

    assert row.passed
    assert row.main_steps_path.endswith("logs/session-1/web_agent_trace.jsonl")
    assert Path(row.main_steps_path).is_file()


def test_trace_verifier_does_not_materialize_generic_trace_label_as_web_trace(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake").model_copy(
        update={
            "trace_expectations": TraceExpectationSpec(evaluator_kind="web_round_evaluator"),
            "tasks": [
                task.model_copy(update={"agent_kind": "web"})
                for task in golden_e2b_experiment_spec(provider="fake").tasks
            ],
        }
    )
    task = spec.tasks[0]
    state = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    batch_result = BatchRunResult(
        batch_id=spec.experiment_id,
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[state],
    )
    db_path = tmp_path / "controller.sqlite"
    _seed_controller_artifact_db(db_path, state)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE artifacts SET original_artifact_id=? WHERE original_artifact_id=?",
            (f"{state.controller_task_id}-trace", f"{state.controller_task_id}-steps"),
        )
        connection.execute(
            "UPDATE artifacts SET original_artifact_id=replace(original_artifact_id, ?, ?)",
            ("desktop_round_completion", "web_round_evaluator"),
        )
        connection.commit()

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    row = next(item for item in rows if item.task_id == task.task_id)

    assert not row.passed
    assert any("web_agent_trace.jsonl missing or empty" in error for error in row.errors)


def test_trace_verifier_does_not_accept_web_trace_for_desktop_agent_kind(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake").model_copy(
        update={"trace_expectations": TraceExpectationSpec(evaluator_kind="web_round_evaluator")}
    )
    task = spec.tasks[0]
    state = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    batch_result = BatchRunResult(
        batch_id=spec.experiment_id,
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[state],
    )
    session_root = _seed_agent_artifact_session(tmp_path, spec, state)
    (session_root / "steps.jsonl").unlink()
    (session_root / "web_agent_trace.jsonl").write_text('{"step":1,"action":"web_open"}\n', encoding="utf-8")

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    row = next(item for item in rows if item.task_id == task.task_id)

    assert not row.passed
    assert "main-agent steps.jsonl missing or empty" in row.errors


def test_trace_verifier_prefers_custom_artifact_root_without_duplicate_materialization(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    task = spec.tasks[0]
    state = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    batch_result = BatchRunResult(
        batch_id=spec.experiment_id,
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[state],
    )
    custom_root = tmp_path / "custom-agent-artifacts"
    session_root = _seed_agent_artifact_session(custom_root.parent, spec, state, root_name=custom_root.name)
    _seed_controller_artifact_db(tmp_path / "controller.sqlite", state)

    rows = ExperimentTraceVerifier().verify(
        output_dir=tmp_path,
        spec=spec,
        batch_result=batch_result,
        extra_roots=[custom_root],
    )
    row = next(item for item in rows if item.task_id == task.task_id)

    assert row.passed
    assert Path(row.main_steps_path).resolve() == (session_root / "steps.jsonl").resolve()
    assert not (tmp_path / "materialized").exists()


def test_trace_verifier_materializes_controller_blob_artifacts(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    task = spec.tasks[0]
    state = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    batch_result = BatchRunResult(
        batch_id=spec.experiment_id,
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[state],
    )
    db_path = tmp_path / "controller.sqlite"
    _seed_controller_artifact_db(db_path, state)
    report_id = f"{state.controller_task_id}-evaluation-desktop_round_completion-round_0001-final_report"
    report_payload = b'{"overall_verdict":"success"}'
    _move_seeded_artifact_to_controller_blob(
        db_path=db_path,
        blob_dir=tmp_path / "controller-blobs",
        artifact_id=report_id,
        payload=report_payload,
        mime_type="application/json",
    )

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    row = next(item for item in rows if item.task_id == task.task_id)

    assert row.passed
    assert Path(row.evaluator_report_path).read_bytes() == report_payload


def test_trace_verifier_uses_custom_controller_db_path(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    task = spec.tasks[0]
    state = BatchTaskState(
        task_id=task.task_id,
        controller_task_id="controller-task-1",
        trace_id="trace-1",
        session_id="session-1",
        sandbox_id="sandbox-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    batch_result = BatchRunResult(
        batch_id=spec.experiment_id,
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[state],
    )
    custom_db = tmp_path / "custom-controller" / "controller.sqlite"
    custom_db.parent.mkdir(parents=True)
    _seed_controller_artifact_db(custom_db, state)
    spec = spec.model_copy(update={"controller": spec.controller.model_copy(update={"db_path": custom_db})})

    rows = ExperimentTraceVerifier().verify(output_dir=tmp_path, spec=spec, batch_result=batch_result)
    row = next(item for item in rows if item.task_id == task.task_id)

    assert row.passed
    assert Path(row.main_steps_path).is_file()
