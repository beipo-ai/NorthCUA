"""Regression coverage for desktop LLM trace capture payloads."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
from nexau.archs.tracer.composite import CompositeTracer
from nexau.archs.tracer.core import SpanType
from nexau.core.messages import ImageBlock, Message, ReasoningBlock, Role, TextBlock, ToolResultBlock, ToolUseBlock
from PIL import Image

from northcua.agentkernel.tracing.capture import TraceCaptureService, TraceCaptureState


class _DummyLLMConfig:
    api_type = "openai_responses"

    def to_openai_params(self) -> dict[str, object]:
        return {
            "model": "gpt-5.4",
            "stream": False,
            "store": False,
            "max_tokens": 256,
        }


class _DummyMessage:
    def __init__(self, role: str, content: str) -> None:
        self._payload = {"role": role, "content": content}

    def _as_legacy_openai_chat_dict(self) -> dict[str, str]:
        return self._payload


def _png_b64() -> str:
    image = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _chat_trace_service() -> TraceCaptureService:
    return TraceCaptureService(
        LLMConfig(
            model="gpt-5.5",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_chat_completion",
            tool_image_policy="embed_in_tool_message",
            supports_images=True,
        )
    )


def test_trace_capture_builds_raw_request_and_response() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    state = TraceCaptureState()

    before_hook, after_hook, after_tool_hook = service.build_hooks(state)
    before_hook[0](
        SimpleNamespace(
            messages=[_DummyMessage("system", "System"), _DummyMessage("user", "Task")],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    tool_call = SimpleNamespace(
        call_id="call-1",
        name="click",
        raw_arguments='{"x": 10, "y": 20}',
        arguments={"x": 10, "y": 20},
        call_type="function",
    )
    after_hook[0](
        SimpleNamespace(
            model_response=SimpleNamespace(
                content="I'll click the launcher.",
                raw_message={
                    "output": [
                        {
                            "type": "reasoning",
                            "encrypted_content": "enc-123",
                        }
                    ],
                    "stop_reason": "tool_calls",
                },
                reasoning_content="Reasoning summary",
                tool_calls=[tool_call],
            ),
            original_response="",
        )
    )
    after_tool_hook[0](
        SimpleNamespace(
            tool_call_id="call-1",
            tool_name="click",
            tool_input={"x": 10, "y": 20},
            tool_output={"text": "Clicked launcher.", "status": "success"},
            llm_tool_output={"text": "Clicked launcher."},
        )
    )

    trace = service.build_trace_record(state)

    assert trace.request["model"] == "gpt-5.4"
    assert trace.request["max_output_tokens"] == 256
    assert trace.response["stop_reason"] == "tool_calls"
    assert trace.usage == {}
    assert state.assistant_tool_calls[0].arguments == '{"x": 10, "y": 20}'
    assert trace.assistant_tool_calls[0].name == "click"
    assert trace.tool_executions[0].tool_name == "click"
    assert trace.reasoning == "Reasoning summary"
    assert trace.reasoning_encrypted_content == "enc-123"
    assert trace.assistant_text == "I'll click the launcher."
    assert state.tool_executions[0].tool_name == "click"


def test_trace_capture_extracts_assistant_text_from_openai_responses_output() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    state = TraceCaptureState()
    _, after_hook, _ = service.build_hooks(state)

    after_hook[0](
        SimpleNamespace(
            model_response=SimpleNamespace(
                content="",
                raw_message={
                    "output": [
                        {"type": "reasoning", "summary": []},
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "The figure has been mirrored horizontally."}],
                        },
                    ]
                },
                reasoning_content="",
                tool_calls=[],
            ),
            original_response="",
        )
    )

    trace = service.build_trace_record(state)

    assert state.assistant_text == "The figure has been mirrored horizontally."
    assert trace.assistant_text == "The figure has been mirrored horizontally."


def test_trace_capture_extracts_assistant_text_from_chat_and_anthropic_payloads() -> None:
    chat_service = TraceCaptureService(_DummyLLMConfig())
    chat_state = TraceCaptureState()
    _, chat_after_hook, _ = chat_service.build_hooks(chat_state)
    chat_after_hook[0](
        SimpleNamespace(
            model_response=SimpleNamespace(
                content="",
                raw_message={"choices": [{"message": {"content": "Chat completion text."}}]},
                reasoning_content="",
                tool_calls=[],
            ),
            original_response="",
        )
    )

    anthropic_service = TraceCaptureService(_DummyLLMConfig())
    anthropic_state = TraceCaptureState()
    _, anthropic_after_hook, _ = anthropic_service.build_hooks(anthropic_state)
    anthropic_after_hook[0](
        SimpleNamespace(
            model_response=SimpleNamespace(
                content="",
                raw_message={"content": [{"type": "text", "text": "Anthropic message text."}]},
                reasoning_content="",
                tool_calls=[],
            ),
            original_response="",
        )
    )

    assert chat_service.build_trace_record(chat_state).assistant_text == "Chat completion text."
    assert anthropic_service.build_trace_record(anthropic_state).assistant_text == "Anthropic message text."


def test_trace_capture_persists_openai_chat_reasoning_content() -> None:
    service = TraceCaptureService(
        LLMConfig(
            model="gpt-5.5",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_chat_completion",
            reasoning_effort="xhigh",
        )
    )
    state = TraceCaptureState()
    _, after_hook, _ = service.build_hooks(state)

    after_hook[0](
        SimpleNamespace(
            model_response=SimpleNamespace(
                content="Done.",
                raw_message={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Done.",
                                "reasoning_content": "The screenshot already shows the requested state.",
                            }
                        }
                    ]
                },
                reasoning_content="The screenshot already shows the requested state.",
                tool_calls=[],
            ),
            original_response="",
        )
    )

    trace = service.build_trace_record(state)

    assert trace.assistant_text == "Done."
    assert trace.reasoning == "The screenshot already shows the requested state."


def test_trace_capture_prefers_tracer_llm_request_payload_when_available() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    state = TraceCaptureState()
    tracer = InMemoryTracer()

    before_hook, _, _ = service.build_hooks(state)
    before_hook[0](
        SimpleNamespace(
            messages=[_DummyMessage("user", "Task")],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    baseline = service.snapshot_tracer(tracer)
    span = tracer.start_span(
        "OpenAI Responses", SpanType.LLM, inputs={"model": "real-model", "tool_choice": {"type": "auto"}}
    )
    tracer.end_span(span, outputs={"result": {"stop_reason": "tool_calls"}})

    service.enrich_from_tracer(state, tracer, baseline)
    trace = service.build_trace_record(state)

    assert trace.request["model"] == "real-model"
    assert trace.request["tool_choice"] == {"type": "auto"}


def test_trace_capture_preserves_private_neutral_tools_when_public_span_is_flattened() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    state = TraceCaptureState(
        provider_tools=[
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ]
    )
    tracer = InMemoryTracer()

    before_hook, _, _ = service.build_hooks(state)
    before_hook[0](
        SimpleNamespace(
            messages=[_DummyMessage("user", "Task")],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )
    baseline = service.snapshot_tracer(tracer)
    span = tracer.start_span(
        "OpenAI responses.create (async)",
        SpanType.LLM,
        inputs={
            "model": "real-model",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ],
        },
    )
    tracer.end_span(span, outputs={"result": {"stop_reason": "completed"}})

    service.enrich_from_tracer(state, tracer, baseline)

    assert state.request_payload["model"] == "real-model"
    assert state.request_payload["tools"] == [
        {
            "name": "lookup",
            "description": "Look up a value",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            "kind": "tool",
        }
    ]


def test_trace_capture_does_not_reinject_hook_tools_when_final_span_omits_them() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    state = TraceCaptureState(
        provider_tools=[
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {"type": "object"},
            }
        ]
    )
    tracer = InMemoryTracer()
    before_hook, _, _ = service.build_hooks(state)
    before_hook[0](
        SimpleNamespace(messages=[_DummyMessage("user", "Task")], agent_state=SimpleNamespace(_tool_registry=None))
    )
    baseline = service.snapshot_tracer(tracer)
    span = tracer.start_span("OpenAI responses.create", SpanType.LLM, inputs={"model": "no-tools"})
    tracer.end_span(span, outputs={"result": {"stop_reason": "completed"}})

    service.enrich_from_tracer(state, tracer, baseline)

    assert "tools" not in state.request_payload


def test_trace_capture_keeps_embedded_tool_images_when_tracer_reinjects_user_images() -> None:
    service = TraceCaptureService(
        LLMConfig(
            model="gpt-5.5",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_chat_completion",
            tool_image_policy="embed_in_tool_message",
            supports_images=True,
        )
    )
    state = TraceCaptureState()
    tracer = InMemoryTracer()

    before_hook, _, _ = service.build_hooks(state)
    before_hook[0](
        SimpleNamespace(
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content=[ToolUseBlock(id="call-1", name="screenshot", input={}, raw_input="{}")],
                ),
                Message(
                    role=Role.TOOL,
                    content=[
                        ToolResultBlock(
                            tool_use_id="call-1",
                            content=[
                                TextBlock(text="Captured."),
                                ImageBlock(url="data:image/png;base64,AAAA"),
                            ],
                        )
                    ],
                ),
            ],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    baseline = service.snapshot_tracer(tracer)
    span = tracer.start_span(
        "openai.chat.completions.create",
        SpanType.LLM,
        inputs={
            "model": "gpt-5.5",
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function"}]},
                {"role": "tool", "tool_call_id": "call-1", "content": "Captured.<image>"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Images returned by tool call call-1:"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                },
            ],
        },
    )
    tracer.end_span(span, outputs={"result": {"stop_reason": "tool_calls"}})

    service.enrich_from_tracer(state, tracer, baseline)
    trace = service.build_trace_record(state)

    messages = trace.request["messages"]
    assert [message["role"] for message in messages] == ["assistant", "tool"]
    assert messages[1]["content"][0] == {"type": "text", "text": "Captured."}
    assert messages[1]["content"][1]["type"] == "image_url"
    assert trace.stop_reason == "tool_calls"


def test_trace_capture_prefers_provider_specific_final_llm_span() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    state = TraceCaptureState()
    tracer = InMemoryTracer()

    baseline = service.snapshot_tracer(tracer)
    generic = tracer.start_span("generic llm", SpanType.LLM, inputs={"model": "generic"})
    tracer.end_span(generic, outputs={"result": {"stop_reason": "tool_calls"}})
    provider = tracer.start_span(
        "OpenAI responses.create (async)",
        SpanType.LLM,
        inputs={"model": "provider-final", "parallel_tool_calls": True},
    )
    tracer.end_span(provider, outputs={"result": {"stop_reason": "completed"}})

    service.enrich_from_tracer(state, tracer, baseline)
    trace = service.build_trace_record(state)

    assert trace.request["model"] == "provider-final"
    assert trace.stop_reason == "completed"


def test_trace_capture_clears_new_in_memory_tracer_spans() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    state = TraceCaptureState()
    tracer = InMemoryTracer()
    baseline_root = tracer.start_span("preexisting diagnostic span", SpanType.LLM, inputs={"keep": True})
    baseline = service.snapshot_tracer(tracer)

    provider = tracer.start_span(
        "OpenAI Responses",
        SpanType.LLM,
        inputs={"model": "provider-final", "large_image": "data:image/png;base64,AAAA"},
        parent_span=baseline_root,
    )
    tracer.end_span(provider, outputs={"result": {"stop_reason": "completed"}})
    transient_root = tracer.start_span("non-provider helper", SpanType.LLM, inputs={"helper": True})
    tracer.end_span(transient_root, outputs={"result": {"ok": True}})

    captured_span_ids = service.new_tracer_span_ids(tracer, baseline)
    service.enrich_from_tracer_span_ids(state, tracer, captured_span_ids)
    trace = service.build_trace_record(state)
    cleared = service.clear_tracer_spans(tracer, captured_span_ids)

    assert trace.request["model"] == "provider-final"
    assert cleared == 2
    assert set(tracer.spans) == {baseline_root.id}
    assert tracer.root_spans == [baseline_root.id]
    assert provider.id not in tracer.children.get(baseline_root.id, [])
    assert transient_root.id not in tracer.children
    assert service.clear_new_tracer_spans(tracer, baseline) == 0
    assert service.clear_new_tracer_spans(None, baseline) == 0
    assert tracer.dump_traces()[0]["id"] == baseline_root.id


def test_trace_capture_clears_only_explicit_span_ids() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    tracer = InMemoryTracer()

    baseline = service.snapshot_tracer(tracer)
    owned_span = tracer.start_span("OpenAI Responses", SpanType.LLM, inputs={"model": "owned"})
    tracer.end_span(owned_span, outputs={"result": {"stop_reason": "completed"}})
    captured_span_ids = service.new_tracer_span_ids(tracer, baseline)
    overlapping_span = tracer.start_span("OpenAI Responses", SpanType.LLM, inputs={"model": "overlap"})
    tracer.end_span(overlapping_span, outputs={"result": {"stop_reason": "completed"}})

    assert service.clear_tracer_spans(tracer, captured_span_ids) == 1
    assert owned_span.id not in tracer.spans
    assert overlapping_span.id in tracer.spans
    assert tracer.root_spans == [overlapping_span.id]


def test_trace_capture_clear_tracer_spans_removes_descendants() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    tracer = InMemoryTracer()
    parent = tracer.start_span("parent", SpanType.LLM, inputs={"model": "parent"})
    child = tracer.start_span("child", SpanType.LLM, inputs={"model": "child"}, parent_span=parent)

    assert service.clear_tracer_spans(tracer, {parent.id}) == 2
    assert parent.id not in tracer.spans
    assert child.id not in tracer.spans
    assert tracer.root_spans == []
    assert tracer.children == {}
    assert tracer.dump_traces() == []


def test_trace_capture_best_effort_clear_does_not_raise(monkeypatch) -> None:
    service = TraceCaptureService(_DummyLLMConfig())

    def _raise(*_args, **_kwargs):
        raise RuntimeError("diagnostic cleanup failed")

    monkeypatch.setattr(service._tracer_enricher, "clear_tracer_spans", _raise)

    assert service.clear_tracer_spans_best_effort(InMemoryTracer(), {"span-id"}) == 0


def test_trace_capture_best_effort_new_span_ids_does_not_raise(monkeypatch) -> None:
    service = TraceCaptureService(_DummyLLMConfig())

    def _raise(*_args, **_kwargs):
        raise RuntimeError("diagnostic snapshot failed")

    monkeypatch.setattr(service._tracer_enricher, "new_span_ids", _raise)

    assert service.new_tracer_span_ids_best_effort(InMemoryTracer(), set()) == set()


def test_trace_capture_clears_nested_composite_in_memory_tracers() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    first = InMemoryTracer()
    second = InMemoryTracer()
    tracer = CompositeTracer([CompositeTracer([first]), second])

    baseline = service.snapshot_tracer(tracer)
    span = tracer.start_span("OpenAI Responses", SpanType.LLM, inputs={"model": "composite"})
    tracer.end_span(span, outputs={"result": {"stop_reason": "completed"}})
    captured_span_ids = service.new_tracer_span_ids(tracer, baseline)

    assert len(captured_span_ids) == 2
    assert service.clear_tracer_spans(tracer, captured_span_ids) == 2
    assert first.spans == {}
    assert second.spans == {}
    assert first.root_spans == []
    assert second.root_spans == []


def test_trace_capture_preserves_tool_result_images_for_openai_responses() -> None:
    service = TraceCaptureService(_DummyLLMConfig())
    state = TraceCaptureState()

    before_hook, _, _ = service.build_hooks(state)
    before_hook[0](
        SimpleNamespace(
            messages=[
                Message(role=Role.USER, content=[TextBlock(text="Task")]),
                Message(
                    role=Role.TOOL,
                    content=[
                        ToolResultBlock(
                            tool_use_id="call-1",
                            content=[
                                TextBlock(text="Clicked."),
                                ImageBlock(url="data:image/png;base64,AAAA"),
                            ],
                        )
                    ],
                ),
            ],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    items = state.request_payload["input"]
    tool_output = next(item for item in items if item.get("type") == "function_call_output")

    assert isinstance(tool_output["output"], list)
    assert tool_output["output"][0]["type"] == "input_text"
    assert tool_output["output"][1]["type"] == "input_image"


def test_trace_capture_chat_tool_images_and_reasoning_replay_can_be_released() -> None:
    service = TraceCaptureService(
        LLMConfig(
            model="gpt-5.5",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_chat_completion",
            reasoning_replay_policy="chat_reasoning_content",
            tool_image_policy="embed_in_tool_message",
            supports_images=True,
        )
    )
    state = TraceCaptureState()

    before_hook, after_hook, _ = service.build_hooks(state)
    hook_result = before_hook[0](
        SimpleNamespace(
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content=[TextBlock(text="Prior step.")],
                    metadata={
                        "reasoning": [
                            {
                                "type": "reasoning",
                                "summary": [{"type": "summary_text", "text": "Kept the target visible."}],
                            }
                        ]
                    },
                ),
                Message(
                    role=Role.ASSISTANT,
                    content=[ToolUseBlock(id="call-1", name="screenshot", input={}, raw_input="{}")],
                ),
                Message(
                    role=Role.TOOL,
                    content=[
                        ToolResultBlock(
                            tool_use_id="call-1",
                            content=[
                                TextBlock(text="Captured."),
                                ImageBlock(url="data:image/png;base64,AAAA"),
                            ],
                        )
                    ],
                ),
            ],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )
    after_hook[0](
        SimpleNamespace(
            model_response=SimpleNamespace(
                content="Done.",
                raw_message={"choices": [{"message": {"reasoning_content": "New reasoning."}}]},
                reasoning_content="New reasoning.",
                tool_calls=[],
            ),
            original_response="",
        )
    )

    messages = state.request_payload["messages"]
    trace = service.build_trace_record(state)

    assert hook_result.messages is not None
    assert messages[0]["reasoning_content"] == "Kept the target visible."
    assert [message["role"] for message in messages] == ["assistant", "assistant", "tool"]
    assert messages[2]["content"][0] == {"type": "text", "text": "Captured."}
    assert messages[2]["content"][1]["type"] == "image_url"
    assert trace.request["messages"][0]["reasoning_content"] == "Kept the target visible."
    assert trace.reasoning == "New reasoning."
    state.release_provider_payloads()
    trace.release_provider_payloads()
    assert state.request_payload == {}
    assert state.response_payload == {}
    assert trace.request == {}
    assert trace.response == {}


def test_trace_capture_keeps_step_observation_image_after_note_iteration() -> None:
    service = _chat_trace_service()
    state = TraceCaptureState()
    user_image = "data:image/png;base64,USERPROMPT"

    before_hook, _, _ = service.build_hooks(state)
    before_hook[0](
        SimpleNamespace(
            messages=[
                Message(
                    role=Role.USER,
                    content=[TextBlock(text="Task"), ImageBlock(url=user_image)],
                )
            ],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )
    before_hook[0](
        SimpleNamespace(
            messages=[
                Message(
                    role=Role.USER,
                    content=[TextBlock(text="Task"), ImageBlock(url=user_image)],
                ),
                Message(
                    role=Role.ASSISTANT,
                    content=[
                        ToolUseBlock(
                            id="call-note",
                            name="note",
                            input={"text": "Remember this value."},
                            raw_input='{"text": "Remember this value."}',
                        )
                    ],
                ),
                Message(
                    role=Role.TOOL,
                    content=[
                        ToolResultBlock(
                            tool_use_id="call-note",
                            content="[fact] Remember this value.",
                        )
                    ],
                ),
            ],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    trace = service.build_trace_record(state)
    messages = trace.request["messages"]

    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[0]["content"][1]["image_url"]["url"] == user_image
    assert messages[2]["content"] == "[fact] Remember this value."


def test_trace_capture_keeps_step_observation_image_after_screenshot_iteration() -> None:
    service = _chat_trace_service()
    state = TraceCaptureState()
    user_image = "data:image/png;base64,USERPROMPT"
    screenshot_image = "data:image/png;base64,SCREENSHOT"

    before_hook, _, _ = service.build_hooks(state)
    before_hook[0](
        SimpleNamespace(
            messages=[
                Message(
                    role=Role.USER,
                    content=[TextBlock(text="Task"), ImageBlock(url=user_image)],
                )
            ],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )
    before_hook[0](
        SimpleNamespace(
            messages=[
                Message(
                    role=Role.USER,
                    content=[TextBlock(text="Task"), ImageBlock(url=user_image)],
                ),
                Message(
                    role=Role.ASSISTANT,
                    content=[ToolUseBlock(id="call-shot", name="screenshot", input={}, raw_input="{}")],
                ),
                Message(
                    role=Role.TOOL,
                    content=[
                        ToolResultBlock(
                            tool_use_id="call-shot",
                            content=[
                                TextBlock(text="Captured."),
                                ImageBlock(url=screenshot_image),
                            ],
                        )
                    ],
                ),
            ],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    trace = service.build_trace_record(state)
    messages = trace.request["messages"]

    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[0]["content"][1]["image_url"]["url"] == user_image
    assert messages[2]["content"][0] == {"type": "text", "text": "Captured."}
    assert messages[2]["content"][1]["image_url"]["url"] == screenshot_image


def test_trace_capture_keeps_step_observation_image_after_note_followup_iteration() -> None:
    service = _chat_trace_service()
    state = TraceCaptureState()
    user_image = "data:image/png;base64,USERPROMPT"

    before_hook, _, _ = service.build_hooks(state)
    base_user = Message(role=Role.USER, content=[TextBlock(text="Task"), ImageBlock(url=user_image)])
    note_call = Message(
        role=Role.ASSISTANT,
        content=[
            ToolUseBlock(
                id="call-note",
                name="note",
                input={"text": "Remember this value."},
                raw_input='{"text": "Remember this value."}',
            )
        ],
    )
    note_result = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="call-note", content="[fact] Remember this value.")],
    )
    click_call = Message(
        role=Role.ASSISTANT,
        content=[ToolUseBlock(id="call-click", name="click", input={"x": 10, "y": 20}, raw_input='{"x":10,"y":20}')],
    )
    click_result = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="call-click", content="Clicked.")],
    )

    before_hook[0](SimpleNamespace(messages=[base_user], agent_state=SimpleNamespace(_tool_registry=None)))
    before_hook[0](
        SimpleNamespace(messages=[base_user, note_call, note_result], agent_state=SimpleNamespace(_tool_registry=None))
    )
    before_hook[0](
        SimpleNamespace(
            messages=[base_user, note_call, note_result, click_call, click_result],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    trace = service.build_trace_record(state)
    messages = trace.request["messages"]

    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant", "tool"]
    assert messages[0]["content"][1]["image_url"]["url"] == user_image
    assert messages[2]["content"] == "[fact] Remember this value."
    assert messages[4]["content"] == "Clicked."


def test_trace_capture_keeps_observation_and_screenshot_images_after_followup_iteration() -> None:
    service = _chat_trace_service()
    state = TraceCaptureState()
    user_image = "data:image/png;base64,USERPROMPT"
    screenshot_image = "data:image/png;base64,SCREENSHOT"

    before_hook, _, _ = service.build_hooks(state)
    base_user = Message(role=Role.USER, content=[TextBlock(text="Task"), ImageBlock(url=user_image)])
    screenshot_call = Message(
        role=Role.ASSISTANT,
        content=[ToolUseBlock(id="call-shot", name="screenshot", input={}, raw_input="{}")],
    )
    screenshot_result = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call-shot",
                content=[TextBlock(text="Captured."), ImageBlock(url=screenshot_image)],
            )
        ],
    )
    note_call = Message(
        role=Role.ASSISTANT,
        content=[
            ToolUseBlock(
                id="call-note",
                name="note",
                input={"text": "The settings page is visible."},
                raw_input='{"text": "The settings page is visible."}',
            )
        ],
    )
    note_result = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="call-note", content="[fact] The settings page is visible.")],
    )

    before_hook[0](SimpleNamespace(messages=[base_user], agent_state=SimpleNamespace(_tool_registry=None)))
    before_hook[0](
        SimpleNamespace(
            messages=[base_user, screenshot_call, screenshot_result],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )
    before_hook[0](
        SimpleNamespace(
            messages=[base_user, screenshot_call, screenshot_result, note_call, note_result],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    trace = service.build_trace_record(state)
    messages = trace.request["messages"]

    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "assistant", "tool"]
    assert messages[0]["content"][1]["image_url"]["url"] == user_image
    assert messages[2]["content"][0] == {"type": "text", "text": "Captured."}
    assert messages[2]["content"][1]["image_url"]["url"] == screenshot_image
    assert messages[4]["content"] == "[fact] The settings page is visible."


def test_trace_capture_rewrites_anthropic_runtime_messages_to_match_trace_payload() -> None:
    service = TraceCaptureService(
        LLMConfig(
            model="claude-opus-4-6",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="anthropic_chat_completion",
        )
    )
    state = TraceCaptureState()
    png_b64 = _png_b64()

    before_hook, _, _ = service.build_hooks(state)
    hook_result = before_hook[0](
        SimpleNamespace(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        TextBlock(text="Task"),
                        ImageBlock(url=f"data:image/png;base64,{png_b64}"),
                    ],
                )
            ],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    traced_image = state.request_payload["messages"][0]["content"][1]
    assert traced_image["source"]["type"] == "base64"
    assert traced_image["source"]["media_type"] == "image/jpeg"
    assert traced_image["source"]["data"] != png_b64
    assert hook_result.messages is not None
    runtime_image = hook_result.messages[0].content[1]
    assert isinstance(runtime_image, ImageBlock)
    assert runtime_image.url is None
    assert runtime_image.mime_type == "image/jpeg"
    assert runtime_image.base64 == traced_image["source"]["data"]


def test_trace_capture_provider_payload_failure_falls_back_without_blocking() -> None:
    service = TraceCaptureService(
        LLMConfig(
            model="gpt-5.4",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_responses",
            provider_profile="no-responses-gateway",
            supports_responses_api=False,
        )
    )
    state = TraceCaptureState()

    before_hook, _, _ = service.build_hooks(state)
    before_hook[0](
        SimpleNamespace(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Task")])],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    assert state.trace_error
    assert state.request_payload["model"] == "gpt-5.4"
    assert "input" in state.request_payload
    assert "provider_profile" not in state.request_payload


def test_trace_capture_fallback_matches_private_empty_reasoning_wire() -> None:
    service = TraceCaptureService(
        LLMConfig(
            model="gpt-5.4",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_responses",
            provider_profile="no-responses-gateway",
            supports_responses_api=False,
        )
    )
    state = TraceCaptureState()
    before_hook, _, _ = service.build_hooks(state)

    before_hook[0](
        SimpleNamespace(
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content=[ReasoningBlock(text=""), TextBlock(text="answer")],
                )
            ],
            agent_state=SimpleNamespace(_tool_registry=None),
        )
    )

    assert state.trace_error
    assert state.request_payload["input"] == [
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer"}],
            "type": "message",
        }
    ]
