"""Regression coverage for shared AgentKit prompt history replay."""

from __future__ import annotations

from nexau.archs.llm.llm_config import LLMConfig

from northcua.agentinfra.providers.llm_fallback import runtime_llm_config
from northcua.agentinfra.providers.provider_payloads import build_provider_request_payload
from northcua.agentinfra.providers.reasoning_replay import (
    ReasoningReplayProjector,
    reasoning_replay_policy_from_config,
)
from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.evaluator.runtime.prompt_history import EvaluatorToolResultContentProjector
from northcua.agentkit.prompt_history import PromptHistoryReplayer
from northcua.agents.desktop.prompt.envelope import HistoryTurn
from northcua.core.models import AssistantToolCallRecord, ToolExecutionRecord


def test_prompt_history_replays_reasoning_before_assistant_tool_call_for_responses() -> None:
    trace_state = TraceCaptureState(
        reasoning="The evaluator should inspect the final artifact before reporting.",
        reasoning_encrypted_content="enc-evaluator-reasoning",
        assistant_text="I will inspect the artifact.",
        assistant_tool_calls=[AssistantToolCallRecord(id="call_1", name="read_trace", arguments='{"session_step": 3}')],
    )

    messages = PromptHistoryReplayer().append_tool_step_history(
        messages=[],
        trace_state=trace_state,
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id="call_1",
                tool_name="read_trace",
                tool_input={"session_step": 3},
                tool_output={"text": "Step 3 edited the file."},
                llm_tool_output={"text": "Step 3 edited the file."},
            )
        ],
    )
    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="gpt-5.5",
            base_url="https://api.openai.com/v1",
            api_key="key",
            api_type="openai_responses",
            max_tokens=128,
        ),
        messages=messages,
    )

    input_items = prepared.request_payload["input"]

    assert input_items[0] == {
        "type": "reasoning",
        "summary": [
            {
                "type": "summary_text",
                "text": "The evaluator should inspect the final artifact before reporting.",
            }
        ],
        "encrypted_content": "enc-evaluator-reasoning",
    }
    assert input_items[1]["type"] == "message"
    assert input_items[1]["role"] == "assistant"
    assert input_items[2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "read_trace",
        "arguments": '{"session_step": 3}',
    }
    assert input_items[3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "Step 3 edited the file.",
    }


def test_runtime_chat_candidate_replays_trace_reasoning_content() -> None:
    messages = HistoryTurn(
        reasoning="Inspect the current screenshot before clicking.",
        assistant_tool_calls=[
            {
                "id": "call_1",
                "type": "function_call",
                "function": {"name": "click", "arguments": '{"x": 500}'},
            }
        ],
        tool_results=[{"tool_call_id": "call_1", "name": "click", "content": {"ok": True}}],
    ).to_messages()
    routed_config = LLMConfig(
        model="qwen",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="openai_chat_completion",
        max_tokens=128,
        reasoning_replay_policy="chat_reasoning_content",
    )
    messages = ReasoningReplayProjector().project_messages(
        messages,
        api_type=routed_config.api_type,
        policy=reasoning_replay_policy_from_config(routed_config),
    )
    llm_config = runtime_llm_config(routed_config)

    prepared = build_provider_request_payload(llm_config=llm_config, messages=messages)

    assistant = prepared.request_payload["messages"][0]
    assert assistant["reasoning_content"] == "Inspect the current screenshot before clicking."
    assert assistant["tool_calls"][0]["function"]["name"] == "click"
    assert "reasoning_replay_policy" not in prepared.request_payload


def test_prompt_history_replays_structured_dict_without_summary_as_json() -> None:
    trace_state = TraceCaptureState(
        assistant_tool_calls=[AssistantToolCallRecord(id="call_1", name="get_url", arguments="{}")],
    )

    messages = PromptHistoryReplayer().append_tool_step_history(
        messages=[],
        trace_state=trace_state,
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id="call_1",
                tool_name="get_url",
                tool_output={"url": "https://example.com"},
                llm_tool_output={"url": "https://example.com"},
            )
        ],
    )
    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="gpt-5.5",
            base_url="https://api.openai.com/v1",
            api_key="key",
            api_type="openai_responses",
            max_tokens=128,
        ),
        messages=messages,
    )

    assert prepared.request_payload["input"][1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"url": "https://example.com"}',
    }


def test_prompt_history_keeps_structured_result_when_summary_text_is_present() -> None:
    trace_state = TraceCaptureState(
        assistant_tool_calls=[AssistantToolCallRecord(id="call_1", name="web_list_tabs", arguments="{}")],
    )

    messages = PromptHistoryReplayer().append_tool_step_history(
        messages=[],
        trace_state=trace_state,
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id="call_1",
                tool_name="web_list_tabs",
                tool_output={
                    "status": "ok",
                    "action": "list_tabs",
                    "text": "Listed browser tabs.",
                    "result": {"tabs": [{"id": "t1", "label": "docs", "url": "https://example.com/docs"}]},
                },
                llm_tool_output={
                    "status": "ok",
                    "action": "list_tabs",
                    "text": "Listed browser tabs.",
                    "result": {"tabs": [{"id": "t1", "label": "docs", "url": "https://example.com/docs"}]},
                },
            )
        ],
    )
    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="gpt-5.5",
            base_url="https://api.openai.com/v1",
            api_key="key",
            api_type="openai_responses",
            max_tokens=128,
        ),
        messages=messages,
    )

    assert prepared.request_payload["input"][1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": (
            'Listed browser tabs.\nresult: {"tabs": [{"id": "t1", "label": "docs", "url": "https://example.com/docs"}]}'
        ),
    }


def test_evaluator_prompt_history_preserves_evidence_tool_json() -> None:
    trace_state = TraceCaptureState(
        assistant_tool_calls=[AssistantToolCallRecord(id="call_1", name="read_trace", arguments='{"session_step": 3}')],
    )

    messages = PromptHistoryReplayer(
        tool_result_projector=EvaluatorToolResultContentProjector()
    ).append_tool_step_history(
        messages=[],
        trace_state=trace_state,
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id="call_1",
                tool_name="read_trace",
                tool_input={"session_step": 3},
                tool_output={
                    "status": "success",
                    "trace": {
                        "assistant_text": "opened product page",
                        "page_url": "https://example.com/product",
                    },
                },
                llm_tool_output={
                    "status": "success",
                    "trace": {
                        "assistant_text": "opened product page",
                        "page_url": "https://example.com/product",
                    },
                },
            )
        ],
    )
    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="gpt-5.5",
            base_url="https://api.openai.com/v1",
            api_key="key",
            api_type="openai_responses",
            max_tokens=128,
        ),
        messages=messages,
    )

    output = prepared.request_payload["input"][1]["output"]
    assert "Tool result [success]" not in output
    assert '"trace"' in output
    assert "opened product page" in output


def test_evaluator_prompt_history_preserves_desktop_artifact_tool_json() -> None:
    trace_state = TraceCaptureState(
        assistant_tool_calls=[
            AssistantToolCallRecord(id="call_matches", name="query_tool_results", arguments='{"tool_name": "click"}'),
            AssistantToolCallRecord(id="call_snapshot", name="query_a11y_snapshot", arguments='{"session_step": 4}'),
            AssistantToolCallRecord(id="call_value", name="read_round_bundle", arguments='{"section": "final"}'),
            AssistantToolCallRecord(id="call_truncated", name="read_trace", arguments='{"session_step": 9}'),
        ],
    )
    payloads = {
        "call_matches": {
            "status": "success",
            "matches": [{"name": "click", "session_step": 4, "summary": "Clicked Save"}],
        },
        "call_snapshot": {
            "status": "success",
            "snapshot": {"role": "window", "tree_preview": {"role": "button", "name": "Save"}},
        },
        "call_value": {
            "status": "success",
            "section": "final",
            "value": {"final_output_candidates": [{"text": "Saved"}]},
        },
        "call_truncated": {
            "status": "success",
            "truncated": True,
            "summary": "Evaluator artifact tool output exceeded the prompt budget and was truncated.",
            "preview": '{"trace": {"assistant_text": "large but useful preview"',
            "original_chars": 99999,
        },
    }

    messages = PromptHistoryReplayer(
        tool_result_projector=EvaluatorToolResultContentProjector()
    ).append_tool_step_history(
        messages=[],
        trace_state=trace_state,
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id=call_id,
                tool_name=trace_state.assistant_tool_calls[index].name,
                tool_input={},
                tool_output=payload,
                llm_tool_output=payload,
            )
            for index, (call_id, payload) in enumerate(payloads.items())
        ],
    )
    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="gpt-5.5",
            base_url="https://api.openai.com/v1",
            api_key="key",
            api_type="openai_responses",
            max_tokens=128,
        ),
        messages=messages,
    )

    outputs = [
        item["output"] for item in prepared.request_payload["input"] if item.get("type") == "function_call_output"
    ]
    assert len(outputs) == 4
    assert all("Tool result [success]" not in output for output in outputs)
    joined = "\n".join(outputs)
    assert '"matches"' in joined
    assert '"snapshot"' in joined
    assert '"value"' in joined
    assert '"truncated": true' in joined
    assert "large but useful preview" in joined
