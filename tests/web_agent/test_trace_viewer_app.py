from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None,
    reason="web trace viewer helpers require the optional streamlit viewer dependency",
)

if importlib.util.find_spec("streamlit") is not None:
    from northcua.agents.web.trace_viewer.app import (
        discover_trace_sources,
        display_messages,
        evaluator_final_report_summary,
        is_user_observation_message,
        output_image_ref_items,
        output_image_refs,
        parsed_function_call_output_payload,
        request_message_label,
        text_from_content,
        trace_noise_metrics,
    )
    from northcua.agents.web.trace_viewer.domain import TraceKind


def test_text_from_content_reads_responses_function_call_output_blocks() -> None:
    message = {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {
                "type": "input_text",
                "text": '{"status":"ok","text":"Clicked @e1.","observation":{"snapshot":{"refs":[{"ref":"@e1"}]},"screenshot":{"annotation":{"marks":[{"ref":"@e1","label":"@e1"}]}}}}',
            },
            {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
        ],
    }

    texts = text_from_content(message)
    parsed = parsed_function_call_output_payload(message)

    assert any("Clicked @e1." in text for text in texts)
    assert parsed is not None
    assert parsed["text"] == "Clicked @e1."
    assert parsed["observation"]["snapshot"]["refs"][0]["ref"] == "@e1"
    assert output_image_refs(message) == ["data:image/png;base64,QUJD"]


def test_request_message_labels_observations_without_current_history_split() -> None:
    current = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Task:\nDo it\n\nCurrent page:\n- title: A"}],
    }
    history = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Task:\nDo it\n\nCurrent page:\n- title: Before"}],
    }
    output = {"type": "function_call_output", "call_id": "call_1", "output": []}

    assert is_user_observation_message(current)
    assert request_message_label(current, 1) == "user observation"
    assert request_message_label(history, 2) == "user observation"
    assert request_message_label(output, 3) == "history function_call_output · call_1"


def test_display_messages_preserves_tool_result_images_from_function_call_output() -> None:
    request = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "input_text", "text": '{"status":"ok","text":"Clicked @e1."}'},
                    {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
                ],
            }
        ]
    }

    messages = display_messages(request, {"output": []})

    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "input_text", "text": '{"status":"ok","text":"Clicked @e1."}'},
                {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
            ],
        }
    ]
    assert (
        parsed_function_call_output_payload({"type": "function_call_output", "output": messages[0]["content"]})["text"]
        == "Clicked @e1."
    )
    assert output_image_refs(messages[0]["content"]) == ["data:image/png;base64,QUJD"]


def test_output_image_refs_collects_local_web_artifact_paths() -> None:
    payload = {
        "status": "ok",
        "screenshot_path": "/tmp/web/step_001_raw.png",
        "result": {
            "raw_image_path": "/tmp/web/tool_001.png",
            "som_image_path": "/tmp/web/tool_001_som.png",
            "url": "https://example.com/not-an-image",
        },
    }

    assert output_image_refs(payload) == [
        "/tmp/web/step_001_raw.png",
        "/tmp/web/tool_001.png",
        "/tmp/web/tool_001_som.png",
    ]


def test_output_image_ref_items_label_web_visual_roles() -> None:
    payload = {
        "raw_image_path": "/tmp/web/tool_001.png",
        "action_overlay_image_path": "/tmp/web/tool_001_action_overlay.png",
        "som_image_path": "/tmp/web/tool_001_som.png",
    }

    items = output_image_ref_items(payload)

    assert [(item.raw, item.role) for item in items] == [
        ("/tmp/web/tool_001.png", "clean"),
        ("/tmp/web/tool_001_action_overlay.png", "action_overlay"),
        ("/tmp/web/tool_001_som.png", "som"),
    ]


def test_output_image_ref_items_label_remote_image_url_roles() -> None:
    payload = {
        "raw_image_url": "data:image/png;base64,Q0xFQU4=",
        "action_overlay_image_url": "data:image/png;base64,QUNUSU9O",
        "overlay_image_url": "data:image/png;base64,U09N",
    }

    items = output_image_ref_items(payload)

    assert {(item.raw, item.role) for item in items} == {
        ("data:image/png;base64,QUNUSU9O", "action_overlay"),
        ("data:image/png;base64,Q0xFQU4=", "clean"),
        ("data:image/png;base64,U09N", "som"),
    }


def test_evaluator_final_report_summary_exposes_key_report_fields() -> None:
    summary = evaluator_final_report_summary(
        {
            "overall_verdict": "fail",
            "outcome_label": "incomplete",
            "completion_rating": 2,
            "confidence_rating": 4,
            "needs_more_evidence": False,
            "web_blocker": "none",
            "final_url": "https://example.test/final",
            "final_title": "Final",
            "first_divergence_step": 3,
            "first_divergence_reason": "Clicked wrong product.",
            "criteria": [{"criterion_id": "task_completion", "score": 2}],
        }
    )

    assert summary == {
        "overall_verdict": "fail",
        "outcome_label": "incomplete",
        "completion_rating": 2,
        "confidence_rating": 4,
        "needs_more_evidence": False,
        "web_blocker": "none",
        "final_url": "https://example.test/final",
        "final_title": "Final",
        "first_divergence_step": 3,
        "first_divergence_reason": "Clicked wrong product.",
    }


def test_trace_noise_metrics_separates_model_visible_from_raw_runtime_noise() -> None:
    row = {
        "llm_trace": {
            "request": {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Judge compact evidence."}],
                    }
                ]
            },
            "response": {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "query_tool_results",
                        "arguments": "{}",
                    }
                ]
            },
        },
        "tool_executions": [
            {
                "tool_call_id": "call_1",
                "llm_tool_output": {"status": "success", "matches": [{"text": "compact"}]},
                "tool_output": {
                    "status": "success",
                    "matches": [
                        {
                            "bbox_css": {"x": 1, "y": 2, "width": 3, "height": 4},
                            "screenshot_path": "/tmp/raw.png",
                        }
                    ],
                },
            }
        ],
    }

    metrics = trace_noise_metrics(row)

    assert metrics["bbox_key_count"] == 0
    assert metrics["local_path_count"] == 0
    assert metrics["raw_bbox_key_count"] > 0
    assert metrics["raw_local_path_count"] > 0


def test_discover_trace_sources_keeps_context_summary_separate(tmp_path) -> None:
    (tmp_path / "llm_trace.jsonl").write_text('{"main":true}\n', encoding="utf-8")
    (tmp_path / "context_summary_llm_trace.jsonl").write_text('{"status":"succeeded"}\n', encoding="utf-8")

    sources = discover_trace_sources(tmp_path, [])

    assert [source.kind for source in sources] == [TraceKind.MAIN, TraceKind.CONTEXT_SUMMARY]
