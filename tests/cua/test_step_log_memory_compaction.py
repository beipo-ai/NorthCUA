from __future__ import annotations

import json

import pytest

from northcua.agents.desktop.services.memory_compaction import (
    compact_replay_payloads_enabled,
    retained_step_payload_window,
)
from northcua.agents.desktop.services.step_log_rewrite import rewrite_step_logs_preserving_existing
from northcua.core.enums import ObservationArtifactKind, TaskStatus
from northcua.core.models import ObservationArtifact, StepLogRecord, ToolResultArtifact


def test_step_log_rewrite_preserves_existing_disk_row_for_compacted_step(tmp_path) -> None:
    data_url = "data:image/png;base64,FULLROW"
    old_step = StepLogRecord(
        session_id="sess",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Click",
        llm_reasoning="old reasoning",
        llm_thinking="old thinking",
        llm_reasoning_encrypted_content="old encrypted",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-old",
                name="screenshot",
                content="Screenshot captured.",
                status="success",
                raw_image_url=data_url,
            )
        ],
    )
    step_log_path = tmp_path / "steps.jsonl"
    step_log_path.write_text(json.dumps(old_step.model_dump(mode="json")) + "\n", encoding="utf-8")

    old_step.release_retained_payloads(drop_unbacked_inline=True, keep_replay_payloads=False)
    recent_step = StepLogRecord(
        session_id="sess",
        session_step=2,
        round_step=2,
        round_num=1,
        status=TaskStatus.FINISH,
        request="Done",
        result="done",
    )

    rewrite_step_logs_preserving_existing([old_step, recent_step], step_log_path)
    payloads = [json.loads(line) for line in step_log_path.read_text(encoding="utf-8").splitlines()]

    assert payloads[0]["llm_reasoning"] == "old reasoning"
    assert payloads[0]["llm_thinking"] == "old thinking"
    assert payloads[0]["llm_reasoning_encrypted_content"] == "old encrypted"
    assert payloads[0]["tool_results"][0]["raw_image_url"] == data_url
    assert payloads[1]["session_step"] == 2
    assert payloads[1]["result"] == "done"


def test_step_log_rewrite_preserves_disk_row_for_released_path_backed_images(tmp_path) -> None:
    data_url = "data:image/png;base64,PATHBACKED"
    old_step = StepLogRecord(
        session_id="sess",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Click",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-old",
                name="screenshot",
                content="Screenshot captured.",
                status="success",
                raw_image_path="/tmp/raw.png",
                raw_image_url=data_url,
            )
        ],
    )
    step_log_path = tmp_path / "steps.jsonl"
    step_log_path.write_text(json.dumps(old_step.model_dump(mode="json")) + "\n", encoding="utf-8")
    old_step.release_retained_payloads()
    old_step.clean_screenshot_path = "/tmp/verify.png"
    old_step.observation_metadata["round_end_verification_status"] = "success"

    rewrite_step_logs_preserving_existing([old_step], step_log_path)
    payload = json.loads(step_log_path.read_text(encoding="utf-8"))

    assert old_step.tool_results[0].raw_image_url == ""
    assert old_step.clean_screenshot_path == "/tmp/verify.png"
    assert old_step.retained_payloads_compacted is False
    assert payload["clean_screenshot_path"] == "/tmp/verify.png"
    assert payload["observation_metadata"]["round_end_verification_status"] == "success"
    assert payload["tool_results"][0]["raw_image_path"] == "/tmp/raw.png"
    assert payload["tool_results"][0]["raw_image_url"] == data_url


def test_step_log_rewrite_does_not_preserve_disk_row_for_terminal_override(tmp_path) -> None:
    data_url = "data:image/png;base64,TERMINAL"
    original = StepLogRecord(
        session_id="sess",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Click",
        result="",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-old",
                name="screenshot",
                content="Screenshot captured.",
                status="success",
                raw_image_path="/tmp/raw.png",
                raw_image_url=data_url,
            )
        ],
    )
    override = StepLogRecord(
        session_id="sess",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.FINISH,
        request="Click",
        result="done",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-old",
                name="screenshot",
                content="Screenshot captured.",
                status="success",
                raw_image_path="/tmp/raw.png",
            )
        ],
    )
    step_log_path = tmp_path / "steps.jsonl"
    step_log_path.write_text(json.dumps(original.model_dump(mode="json")) + "\n", encoding="utf-8")

    rewrite_step_logs_preserving_existing([override], step_log_path)
    payload = json.loads(step_log_path.read_text(encoding="utf-8"))

    assert payload["status"] == "finish"
    assert payload["result"] == "done"
    assert payload["tool_results"][0]["raw_image_url"] == data_url


def test_step_log_rewrite_merges_tool_result_images_by_call_id(tmp_path) -> None:
    first_url = "data:image/png;base64,FIRST"
    second_url = "data:image/png;base64,SECOND"
    original = StepLogRecord(
        session_id="sess",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Click",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-first",
                name="screenshot",
                content="first",
                status="success",
                raw_image_path="/tmp/first.png",
                raw_image_url=first_url,
            ),
            ToolResultArtifact(
                tool_call_id="call-second",
                name="screenshot",
                content="second",
                status="success",
                raw_image_path="/tmp/second.png",
                raw_image_url=second_url,
            ),
        ],
    )
    current = original.model_copy(deep=True)
    current.tool_results = [current.tool_results[1], current.tool_results[0]]
    for result in current.tool_results:
        result.raw_image_url = ""
    step_log_path = tmp_path / "steps.jsonl"
    step_log_path.write_text(json.dumps(original.model_dump(mode="json")) + "\n", encoding="utf-8")

    rewrite_step_logs_preserving_existing([current], step_log_path)
    payload = json.loads(step_log_path.read_text(encoding="utf-8"))

    assert payload["tool_results"][0]["tool_call_id"] == "call-second"
    assert payload["tool_results"][0]["raw_image_url"] == second_url
    assert payload["tool_results"][1]["tool_call_id"] == "call-first"
    assert payload["tool_results"][1]["raw_image_url"] == first_url


def test_step_log_rewrite_does_not_positionally_merge_new_tool_call_id(tmp_path) -> None:
    old_url = "data:image/png;base64,OLD"
    original = StepLogRecord(
        session_id="sess",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Click",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-old",
                name="screenshot",
                content="old",
                status="success",
                raw_image_path="/tmp/old.png",
                raw_image_url=old_url,
            )
        ],
    )
    current = StepLogRecord(
        session_id="sess",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Click",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-new",
                name="screenshot",
                content="new",
                status="success",
                raw_image_path="/tmp/new.png",
            )
        ],
    )
    step_log_path = tmp_path / "steps.jsonl"
    step_log_path.write_text(json.dumps(original.model_dump(mode="json")) + "\n", encoding="utf-8")

    rewrite_step_logs_preserving_existing([current], step_log_path)
    payload = json.loads(step_log_path.read_text(encoding="utf-8"))

    assert payload["tool_results"][0]["tool_call_id"] == "call-new"
    assert payload["tool_results"][0]["raw_image_url"] == ""


def test_step_log_rewrite_does_not_positionally_merge_new_observation_artifact(tmp_path) -> None:
    old_url = "data:image/png;base64,OLDARTIFACT"
    original = StepLogRecord(
        session_id="sess",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Click",
        observation_artifacts=[
            ObservationArtifact(
                kind=ObservationArtifactKind.IMAGE,
                role="raw",
                source="screen",
                path="/tmp/old.png",
                url=old_url,
            )
        ],
    )
    current = StepLogRecord(
        session_id="sess",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Click",
        observation_artifacts=[
            ObservationArtifact(
                kind=ObservationArtifactKind.IMAGE,
                role="raw",
                source="screen",
                path="/tmp/new.png",
            )
        ],
    )
    step_log_path = tmp_path / "steps.jsonl"
    step_log_path.write_text(json.dumps(original.model_dump(mode="json")) + "\n", encoding="utf-8")

    rewrite_step_logs_preserving_existing([current], step_log_path)
    payload = json.loads(step_log_path.read_text(encoding="utf-8"))

    assert payload["observation_artifacts"][0]["path"] == "/tmp/new.png"
    assert payload["observation_artifacts"][0]["url"] == ""


def test_worker_step_payload_window_is_independent_from_prompt_history(monkeypatch) -> None:
    monkeypatch.delenv("NORTHCUA_WORKER_RETAIN_RECENT_STEP_PAYLOADS", raising=False)
    assert retained_step_payload_window() == 16

    monkeypatch.setenv("NORTHCUA_WORKER_RETAIN_RECENT_STEP_PAYLOADS", "0")
    assert retained_step_payload_window() is None

    monkeypatch.setenv("NORTHCUA_WORKER_RETAIN_RECENT_STEP_PAYLOADS", "1")
    assert retained_step_payload_window() == 2

    monkeypatch.setenv("NORTHCUA_WORKER_RETAIN_RECENT_STEP_PAYLOADS", "bad")
    assert retained_step_payload_window() == 16


def test_worker_replay_payload_compaction_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("NORTHCUA_WORKER_COMPACT_REPLAY_PAYLOADS", raising=False)
    assert compact_replay_payloads_enabled() is False

    monkeypatch.setenv("NORTHCUA_WORKER_COMPACT_REPLAY_PAYLOADS", "1")
    assert compact_replay_payloads_enabled() is True


def _simple_record() -> StepLogRecord:
    return StepLogRecord(
        session_id="s",
        session_step=0,
        request="r",
        status=TaskStatus.CONTINUE,
        round_step=0,
        round_num=0,
    )


def test_step_log_rewrite_succeeds_when_existing_file_present(tmp_path) -> None:
    """The rewrite must release its own read handle before replacing (Windows)."""

    step_log_path = tmp_path / "steps.jsonl"
    step_log_path.write_text(json.dumps({"session_id": "s", "session_step": 0, "request": "r"}) + "\n")

    rewrite_step_logs_preserving_existing([_simple_record()], step_log_path)

    assert json.loads(step_log_path.read_text().splitlines()[0])["session_id"] == "s"
    assert not (tmp_path / "steps.jsonl.rewrite.tmp").exists()


def test_step_log_rewrite_retries_transient_permission_error(tmp_path, monkeypatch) -> None:
    from northcua.agents.desktop.services import step_log_rewrite as module

    real_replace = module.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.os, "replace", flaky_replace)
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    step_log_path = tmp_path / "steps.jsonl"

    rewrite_step_logs_preserving_existing([_simple_record()], step_log_path)

    assert calls["n"] == 3
    assert step_log_path.exists()


def test_step_log_rewrite_raises_when_destination_stays_locked(tmp_path, monkeypatch) -> None:
    from northcua.agents.desktop.services import step_log_rewrite as module

    def locked_replace(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.os, "replace", locked_replace)
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    step_log_path = tmp_path / "steps.jsonl"

    with pytest.raises(PermissionError):
        rewrite_step_logs_preserving_existing([_simple_record()], step_log_path)

    assert not (tmp_path / "steps.jsonl.rewrite.tmp").exists()


def test_step_log_rewrite_does_not_retry_on_posix(tmp_path, monkeypatch) -> None:
    from northcua.agents.desktop.services import step_log_rewrite as module

    calls = {"n": 0}

    def locked_replace(src, dst):
        calls["n"] += 1
        raise PermissionError(13, "denied")

    monkeypatch.setattr(module.os, "name", "posix")
    monkeypatch.setattr(module.os, "replace", locked_replace)

    with pytest.raises(PermissionError):
        rewrite_step_logs_preserving_existing([_simple_record()], tmp_path / "steps.jsonl")

    assert calls["n"] == 1
