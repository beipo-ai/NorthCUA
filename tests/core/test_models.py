from __future__ import annotations

import json

from northcua.core.enums import TaskStatus
from northcua.core.models import LLMTraceRecord, RoundState, StepLogRecord, ToolExecutionRecord, ToolResultArtifact


def test_tool_result_artifact_text_alias_matches_content() -> None:
    artifact = ToolResultArtifact(
        tool_call_id="call-1",
        name="click",
        content="Clicked Calculator icon.",
        status="ok",
    )

    assert artifact.text == "Clicked Calculator icon."


def test_tool_execution_record_backfills_coordinate_input_views_from_tool_output() -> None:
    record = ToolExecutionRecord(
        tool_call_id="call-click",
        tool_name="click",
        tool_input={"x": 500, "y": 500},
        tool_output={
            "tool_input_model": {"x": 500, "y": 500, "button": "left"},
            "tool_input_pixels": {"x": 40, "y": 30, "button": "left"},
            "tool_input_runtime_pixels": {"x": 140, "y": 230, "button": "left"},
        },
    )

    assert record.tool_input_model == {"x": 500, "y": 500, "button": "left"}
    assert record.tool_input_pixels == {"x": 40, "y": 30, "button": "left"}
    assert record.tool_input_runtime_pixels == {"x": 140, "y": 230, "button": "left"}


def test_tool_execution_record_does_not_treat_raw_model_input_as_pixels() -> None:
    record = ToolExecutionRecord(
        tool_call_id="call-legacy",
        tool_name="click",
        tool_input={"x": 500, "y": 500},
        tool_output={"text": "Clicked."},
    )

    assert record.tool_input_model == {"x": 500, "y": 500}
    assert record.tool_input_pixels == {}


def test_tool_execution_record_parses_string_tool_output_once(monkeypatch) -> None:
    calls = 0
    real_loads = json.loads

    def _counting_loads(value: str) -> object:
        nonlocal calls
        calls += 1
        return real_loads(value)

    monkeypatch.setattr("northcua.core.models.json.loads", _counting_loads)

    record = ToolExecutionRecord(
        tool_call_id="call-click",
        tool_name="click",
        tool_input={"x": 500, "y": 500},
        tool_output=json.dumps(
            {
                "tool_input_model": {"x": 500, "y": 500},
                "tool_input_pixels": {"x": 40, "y": 30},
                "tool_input_runtime_pixels": {"x": 140, "y": 230},
            }
        ),
    )

    assert calls == 1
    assert record.tool_input_model == {"x": 500, "y": 500}
    assert record.tool_input_pixels == {"x": 40, "y": 30}
    assert record.tool_input_runtime_pixels == {"x": 140, "y": 230}


def test_step_log_release_retained_payloads_keeps_paths_and_drops_inline_images() -> None:
    step = StepLogRecord(
        session_id="sess",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Click",
        llm_trace=LLMTraceRecord(
            request={"messages": [{"content": "data:image/png;base64,AAAA"}]},
            response={"output": [{"content": "large"}]},
            tool_executions=[
                ToolExecutionRecord(
                    tool_call_id="call-1",
                    tool_name="click",
                    tool_output={"screenshot_path": "/tmp/raw.png", "screenshot_url": "data:image/png;base64,BBBB"},
                )
            ],
        ),
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-1",
                name="click",
                content="Clicked",
                status="success",
                raw_image_path="/tmp/raw.png",
                raw_image_url="data:image/png;base64,BBBB",
                overlay_image_path="/tmp/overlay.png",
                overlay_image_url="data:image/png;base64,CCCC",
                image_path="/tmp/overlay.png",
                image_url="data:image/png;base64,CCCC",
            )
        ],
    )

    step.release_retained_payloads()

    assert step.llm_trace.request == {}
    assert step.llm_trace.response == {}
    assert step.llm_trace.tool_executions == []
    result = step.tool_results[0]
    assert result.raw_image_path == "/tmp/raw.png"
    assert result.overlay_image_path == "/tmp/overlay.png"
    assert result.raw_image_url == ""
    assert result.overlay_image_url == ""
    assert result.image_url == ""


def test_release_image_payloads_preserves_url_only_command_results() -> None:
    data_url = "data:image/png;base64,URLONLY"
    record = ToolExecutionRecord(
        tool_call_id="call-url-only",
        tool_name="screenshot",
        tool_output={"screenshot_url": data_url, "text": "visible"},
        llm_tool_output=data_url,
    )

    record.release_image_payloads()

    assert record.tool_output == {"screenshot_url": data_url, "text": "visible"}
    assert record.llm_tool_output == data_url


def test_release_image_payloads_drops_path_backed_nested_command_images() -> None:
    record = ToolExecutionRecord(
        tool_call_id="call-path-backed",
        tool_name="screenshot",
        tool_output={"screenshot_path": "/tmp/raw.png", "screenshot_url": "data:image/png;base64,AAAA"},
        llm_tool_output={"image_path": "/tmp/selected.png", "image_url": "data:image/png;base64,BBBB"},
    )

    record.release_image_payloads()

    assert record.tool_output == {"screenshot_path": "/tmp/raw.png", "screenshot_url": ""}
    assert record.llm_tool_output == {"image_path": "/tmp/selected.png", "image_url": ""}


def test_round_state_release_retained_payloads_preserves_replay_payloads_by_default() -> None:
    data_url = "data:image/png;base64,URLONLY"
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
    state = RoundState(session_id="sess", step_logs=[old_step])

    state.release_retained_payloads(retain_recent_steps=0)

    assert old_step.llm_reasoning == "old reasoning"
    assert old_step.llm_thinking == "old thinking"
    assert old_step.llm_reasoning_encrypted_content == "old encrypted"
    assert old_step.tool_results[0].raw_image_url == data_url
    assert old_step.retained_payloads_compacted is False


def test_round_state_release_retained_payloads_compacts_steps_outside_history_window_when_enabled() -> None:
    data_url = "data:image/png;base64,URLONLY"
    path_backed_url = "data:image/png;base64,PATHBACKED"
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
                image_path="/tmp/selected.png",
                image_url=path_backed_url,
            )
        ],
        llm_trace=LLMTraceRecord(
            request={"input": [data_url]},
            response={"output": [{"type": "message"}]},
            tool_executions=[
                ToolExecutionRecord(
                    tool_call_id="call-old",
                    tool_name="screenshot",
                    tool_output={"data_url": data_url},
                    llm_tool_output=data_url,
                )
            ],
        ),
    )
    recent_step = StepLogRecord(
        session_id="sess",
        session_step=2,
        round_step=2,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Click again",
        llm_reasoning="recent reasoning",
        llm_thinking="recent thinking",
        llm_reasoning_encrypted_content="recent encrypted",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-recent",
                name="screenshot",
                content="Screenshot captured.",
                status="success",
                raw_image_url=data_url,
            )
        ],
    )
    state = RoundState(session_id="sess", step_logs=[old_step, recent_step])

    state.release_retained_payloads(retain_recent_steps=1, compact_replay_payloads=True)

    assert old_step.llm_reasoning == ""
    assert old_step.llm_thinking == ""
    assert old_step.llm_reasoning_encrypted_content == ""
    assert old_step.llm_trace.request == {}
    assert old_step.llm_trace.response == {}
    assert old_step.llm_trace.tool_executions == []
    assert old_step.tool_results[0].raw_image_url == ""
    assert old_step.tool_results[0].image_url == ""
    assert old_step.tool_results[0].image_path == "/tmp/selected.png"
    assert old_step.retained_payloads_compacted is True
    assert old_step.result == ""
    assert old_step.status == TaskStatus.CONTINUE

    assert recent_step.llm_reasoning == "recent reasoning"
    assert recent_step.llm_thinking == "recent thinking"
    assert recent_step.llm_reasoning_encrypted_content == "recent encrypted"
    assert recent_step.tool_results[0].raw_image_url == data_url
    assert recent_step.retained_payloads_compacted is False
