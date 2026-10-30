from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import anthropic
import httpx
import pytest
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.hooks import Middleware, MiddlewareManager
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.core.messages import Message, ReasoningBlock, Role, TextBlock, ToolResultBlock, ToolUseBlock

from northcua.agentinfra.nexau_compat.adapters.openai_responses import wrap_openai_responses_client
from northcua.agentinfra.nexau_compat.adapters.providers import (
    LegacyLLMCallRequest,
    NexAUProviderAdapter,
    NexAUProviderCapabilities,
    NexAUProviderCapabilityError,
    NexAUProviderCompatibilityPolicy,
    UnsignedReasoningReplayPolicy,
)
from northcua.agentkernel.tools.surface import install_active_provider_tool_payload
from northcua.policy_invocation import POLICY_STEP_HEADER, use_policy_step_ref


def _llm_config(api_type: str = "anthropic_chat_completion") -> LLMConfig:
    return LLMConfig(
        model="fixture-model",
        base_url="https://example.invalid",
        api_key="fixture-key",
        api_type=api_type,
    )


def test_anthropic_unsigned_reasoning_matches_private_text_replay() -> None:
    messages = [
        Message(
            role=Role.ASSISTANT,
            content=[ReasoningBlock(text="internal"), TextBlock(text="answer")],
        )
    ]

    payload = NexAUProviderAdapter().serialize_anthropic(messages)

    assert payload.system == []
    assert payload.messages == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "internal"},
                {"type": "text", "text": "answer"},
            ],
        }
    ]
    assert isinstance(messages[0].content[0], ReasoningBlock)


def test_anthropic_drop_policy_does_not_mutate_source_message() -> None:
    messages = [Message(role=Role.ASSISTANT, content=[ReasoningBlock(text="internal"), TextBlock(text="answer")])]
    adapter = NexAUProviderAdapter(
        NexAUProviderCompatibilityPolicy(unsigned_reasoning=UnsignedReasoningReplayPolicy.DROP)
    )

    payload = adapter.serialize_anthropic(messages)

    assert payload.messages == [{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}]
    assert isinstance(messages[0].content[0], ReasoningBlock)


def test_provider_thinking_requires_an_explicit_supported_flag() -> None:
    capabilities = NexAUProviderCapabilities.detect()
    policy = NexAUProviderCompatibilityPolicy(
        unsigned_reasoning=UnsignedReasoningReplayPolicy.PROVIDER_THINKING,
        public_allow_unsigned_thinking=True,
    )

    if capabilities.anthropic_unsigned_thinking_flag:
        adapter = NexAUProviderAdapter(policy, capabilities)
        assert adapter.policy.public_allow_unsigned_thinking is True
    else:
        with pytest.raises(NexAUProviderCapabilityError):
            NexAUProviderAdapter(policy, capabilities)


def test_openai_responses_drops_empty_reasoning_reconstruction() -> None:
    messages = [Message(role=Role.ASSISTANT, content=[ReasoningBlock(text=""), TextBlock(text="answer")])]

    adapter = NexAUProviderAdapter()
    payload = adapter.prepare_openai_responses_input(
        adapter.serialize_openai_chat(messages, tool_image_policy="embed_in_tool_message")
    )

    assert payload.instructions is None
    assert all(item.get("type") != "reasoning" for item in payload.items)
    assert payload.items == [
        {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}], "type": "message"}
    ]


def test_openai_responses_preserves_encrypted_empty_reasoning_with_private_summary_shape() -> None:
    messages = [
        Message(
            role=Role.ASSISTANT,
            content=[ReasoningBlock(text="", redacted_data="encrypted"), TextBlock(text="answer")],
        )
    ]

    adapter = NexAUProviderAdapter()
    payload = adapter.prepare_openai_responses_input(
        adapter.serialize_openai_chat(messages, tool_image_policy="embed_in_tool_message")
    )

    assert payload.items[0] == {"type": "reasoning", "encrypted_content": "encrypted", "summary": []}


class _ResponsesCapture:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            id="response-1",
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="answer", annotations=[])],
                )
            ],
            output_text="answer",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            status="completed",
            error=None,
            incomplete_details=None,
            model="fixture-model",
        )

    def stream(self, **kwargs: object) -> SimpleNamespace:
        return self.create(**kwargs)


class _ResponsesClient:
    def __init__(self) -> None:
        self.responses = _ResponsesCapture()


class _AsyncResponsesCapture(_ResponsesCapture):
    async def create(self, **kwargs: object) -> SimpleNamespace:
        return super().create(**kwargs)


class _AsyncResponsesClient:
    def __init__(self) -> None:
        self.responses = _AsyncResponsesCapture()


def test_responses_stream_boundary_normalizes_empty_reasoning() -> None:
    client = _ResponsesClient()
    wrapped = wrap_openai_responses_client(client)
    assert wrapped is not None
    stream = getattr(getattr(wrapped, "responses"), "stream")
    cast(Callable[..., object], stream)(
        input=[
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "content": []},
        ]
    )

    assert client.responses.requests == [{"input": [{"type": "message", "role": "assistant", "content": []}]}]


def test_real_llm_caller_send_boundary_drops_public_empty_reasoning_item() -> None:
    client = _ResponsesClient()
    adapter = NexAUProviderAdapter()
    caller = adapter.create_llm_caller(
        openai_client=client,
        llm_config=_llm_config("openai_responses"),
        retry_attempts=1,
    )

    adapter.call_llm(
        caller,
        LegacyLLMCallRequest(
            messages=[Message(role=Role.ASSISTANT, content=[ReasoningBlock(text=""), TextBlock(text="answer")])],
            max_tokens=32,
        ),
    )

    assert len(client.responses.requests) == 1
    assert all(item.get("type") != "reasoning" for item in client.responses.requests[0]["input"])  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_real_async_llm_caller_send_boundary_drops_public_empty_reasoning_item() -> None:
    client = _AsyncResponsesClient()
    adapter = NexAUProviderAdapter()
    caller = adapter.create_llm_caller(
        openai_client=object(),
        async_openai_client=client,
        llm_config=_llm_config("openai_responses"),
        retry_attempts=1,
    )

    await adapter.call_llm_async(
        caller,
        LegacyLLMCallRequest(
            messages=[Message(role=Role.ASSISTANT, content=[ReasoningBlock(text=""), TextBlock(text="answer")])],
            max_tokens=32,
        ),
    )

    assert len(client.responses.requests) == 1
    assert all(item.get("type") != "reasoning" for item in client.responses.requests[0]["input"])  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_second_async_responses_client_is_wrapped_after_agent_reinitialization() -> None:
    initial = _AsyncResponsesClient()
    replacement = _AsyncResponsesClient()
    adapter = NexAUProviderAdapter()
    caller = adapter.create_llm_caller(
        openai_client=object(),
        async_openai_client=initial,
        llm_config=_llm_config("openai_responses"),
        retry_attempts=1,
    )
    request = LegacyLLMCallRequest(
        messages=[Message(role=Role.ASSISTANT, content=[ReasoningBlock(text=""), TextBlock(text="answer")])],
        max_tokens=32,
    )

    await adapter.call_llm_async(caller, request)
    caller.async_openai_client = replacement
    await adapter.call_llm_async(caller, request)

    assert len(initial.responses.requests) == 1
    assert len(replacement.responses.requests) == 1
    assert all(
        item.get("type") != "reasoning"
        for item in replacement.responses.requests[0]["input"]  # type: ignore[union-attr]
    )


def test_request_override_client_is_normalized_at_final_responses_send_boundary() -> None:
    default_client = _ResponsesClient()
    override_client = _ResponsesClient()
    adapter = NexAUProviderAdapter()
    caller = adapter.create_llm_caller(
        openai_client=default_client,
        llm_config=_llm_config("openai_responses"),
        retry_attempts=1,
    )

    adapter.call_llm(
        caller,
        LegacyLLMCallRequest(
            messages=[Message(role=Role.ASSISTANT, content=[ReasoningBlock(text=""), TextBlock(text="answer")])],
            max_tokens=32,
            openai_client=override_client,
        ),
    )

    assert default_client.responses.requests == []
    assert len(override_client.responses.requests) == 1
    assert all(
        item.get("type") != "reasoning"
        for item in override_client.responses.requests[0]["input"]  # type: ignore[union-attr]
    )


def test_four_provider_serializers_preserve_tool_call_identity() -> None:
    messages = [
        Message(role=Role.SYSTEM, content=[TextBlock(text="system")]),
        Message(
            role=Role.ASSISTANT,
            content=[ToolUseBlock(id="call-7", name="lookup", input={"q": "x"})],
        ),
        Message(
            role=Role.TOOL,
            content=[ToolResultBlock(tool_use_id="call-7", content="result")],
            metadata={"tool_name": "lookup"},
        ),
    ]
    adapter = NexAUProviderAdapter()

    openai_chat = adapter.serialize_openai_chat(messages)
    openai_responses = adapter.prepare_openai_responses_input(
        adapter.serialize_openai_chat(messages, tool_image_policy="embed_in_tool_message")
    )
    anthropic = adapter.serialize_anthropic(messages)
    from nexau.core.serializers.gemini_messages import serialize_ump_to_gemini_messages_payload

    gemini_contents, _gemini_system = serialize_ump_to_gemini_messages_payload(messages)

    assert openai_chat[1]["tool_calls"][0]["id"] == "call-7"  # type: ignore[index]
    assert any(item.get("call_id") == "call-7" for item in openai_responses.items)
    assert anthropic.messages[0]["content"][0]["id"] == "call-7"  # type: ignore[index]
    assert gemini_contents[-1] == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "lookup", "response": {"result": "result"}}}],
    }


def test_prepare_llm_config_preserves_type_and_sets_public_flags_explicitly() -> None:
    original = _llm_config()
    adapter = NexAUProviderAdapter()

    prepared = adapter.prepare_llm_config(original)

    assert type(prepared) is LLMConfig
    assert prepared is not original
    if adapter.capabilities.llm_config_tool_streaming:
        assert prepared.tool_streaming is True
        assert prepared.allow_unsigned_thinking is False


def test_create_llm_caller_returns_installed_nexau_type() -> None:
    adapter = NexAUProviderAdapter()

    caller = adapter.create_llm_caller(openai_client=object(), llm_config=_llm_config())

    assert type(caller) is LLMCaller
    assert type(caller.llm_config) is LLMConfig
    if adapter.capabilities.llm_config_tool_streaming:
        assert caller.llm_config.tool_streaming is True
        assert caller.llm_config.allow_unsigned_thinking is False


class _CaptureCaller:
    def __init__(self) -> None:
        self.llm_config = _llm_config()
        self.messages: list[Message] = []
        self.kwargs: dict[str, object] = {}

    def call_llm(self, messages: list[Message], **kwargs: object) -> None:
        self.messages = messages
        self.kwargs = kwargs

    async def call_llm_async(self, messages: list[Message], **kwargs: object) -> None:
        self.messages = messages
        self.kwargs = kwargs


def test_llm_call_wrapper_uses_private_surface_and_projects_messages() -> None:
    caller = _CaptureCaller()
    source = Message(role=Role.ASSISTANT, content=[ReasoningBlock(text="internal")])

    result = NexAUProviderAdapter().call_llm(
        cast(LLMCaller, caller),
        LegacyLLMCallRequest(messages=[source], max_tokens=64),
    )

    assert result is None
    assert isinstance(caller.messages[0].content[0], TextBlock)
    assert caller.messages[0].content[0].text == "internal"
    assert "framework_context" not in caller.kwargs
    assert caller.kwargs["max_tokens"] == 64


@pytest.mark.asyncio
async def test_async_llm_call_wrapper_matches_sync_projection() -> None:
    caller = _CaptureCaller()
    source = Message(role=Role.ASSISTANT, content=[ReasoningBlock(text="internal")])

    result = await NexAUProviderAdapter().call_llm_async(
        cast(LLMCaller, caller),
        LegacyLLMCallRequest(messages=[source], max_tokens=32),
    )

    assert result is None
    assert isinstance(caller.messages[0].content[0], TextBlock)
    assert "framework_context" not in caller.kwargs
    assert caller.kwargs["max_tokens"] == 32


class _RetryBoundaryCaller:
    def __init__(self, api_type: str = "anthropic_chat_completion") -> None:
        self.llm_config = _llm_config(api_type)
        self.sync_params: object | None = None
        self.async_params: object | None = None
        self.async_extra: tuple[tuple[object, ...], dict[str, object]] | None = None
        self.response: ModelResponse | None = None

    def _call_with_retry(self, params: object) -> ModelResponse | None:
        self.sync_params = params
        return self.response

    async def _call_with_retry_async(self, params: object, *args: object, **kwargs: object) -> ModelResponse | None:
        self.async_params = params
        self.async_extra = (args, kwargs)
        return self.response


def _retry_params(*messages: Message, api_type: str = "anthropic_chat_completion") -> SimpleNamespace:
    return SimpleNamespace(messages=list(messages), llm_config=_llm_config(api_type))


def test_installed_retry_boundary_projects_complete_agent_calls_without_mutating_history() -> None:
    caller = _RetryBoundaryCaller()
    source = Message(role=Role.ASSISTANT, content=[ReasoningBlock(text="internal"), TextBlock(text="answer")])
    params = _retry_params(source)
    adapter = NexAUProviderAdapter()

    adapter.install_on_caller(cast(LLMCaller, caller))
    caller._call_with_retry(params)

    assert caller.sync_params is not params
    projected = cast(SimpleNamespace, caller.sync_params)
    assert isinstance(projected.messages[0].content[0], TextBlock)
    assert projected.messages[0].content[0].text == "internal"
    assert isinstance(source.content[0], ReasoningBlock)


@pytest.mark.asyncio
async def test_installed_retry_boundary_projects_async_fallback_config_and_is_idempotent() -> None:
    caller = _RetryBoundaryCaller(api_type="openai_responses")
    source = Message(role=Role.ASSISTANT, content=[ReasoningBlock(text="internal")])
    params = _retry_params(source, api_type="anthropic_chat_completion")
    adapter = NexAUProviderAdapter()

    adapter.install_on_caller(cast(LLMCaller, caller))
    adapter.install_on_caller(cast(LLMCaller, caller))
    sentinel = object()
    await caller._call_with_retry_async(params, sentinel, marker="kept")

    projected = cast(SimpleNamespace, caller.async_params)
    assert isinstance(projected.messages[0].content[0], TextBlock)
    assert isinstance(source.content[0], ReasoningBlock)
    assert caller.async_extra == ((sentinel,), {"marker": "kept"})


def test_installed_retry_boundary_leaves_other_provider_params_by_identity() -> None:
    caller = _RetryBoundaryCaller(api_type="openai_responses")
    params = _retry_params(Message.user("hello"), api_type="openai_responses")

    NexAUProviderAdapter().install_on_caller(cast(LLMCaller, caller))
    caller._call_with_retry(params)

    assert caller.sync_params is params


def test_openai_chat_retry_boundary_adds_task_local_step_header() -> None:
    caller = _RetryBoundaryCaller(api_type="openai_chat_completion")
    params = _retry_params(Message.user("hello"), api_type="openai_chat_completion")
    params.api_params = {"temperature": 0.2}
    NexAUProviderAdapter().install_on_caller(cast(LLMCaller, caller))

    with use_policy_step_ref("controller-session:desktop:1:2"):
        caller._call_with_retry(params)

    projected = cast(SimpleNamespace, caller.sync_params)
    assert projected.api_params["extra_headers"] == {POLICY_STEP_HEADER: "controller-session:desktop:1:2"}
    assert projected.api_params["temperature"] == 0.2
    assert params.api_params == {"temperature": 0.2}


def test_openai_retry_boundary_replaces_reexpanded_tools_with_active_task_surface() -> None:
    caller = _RetryBoundaryCaller(api_type="openai_chat_completion")
    click = {
        "type": "function",
        "function": {
            "name": "click",
            "description": "click",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    screenshot = {
        "type": "function",
        "function": {
            "name": "screenshot",
            "description": "screenshot",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    params = _retry_params(Message.user("click"), api_type="openai_chat_completion")
    params.api_params = {"tools": [click, screenshot], "tool_choice": "required"}
    install_active_provider_tool_payload(caller, [click])
    NexAUProviderAdapter().install_on_caller(cast(LLMCaller, caller))

    caller._call_with_retry(params)

    projected = cast(SimpleNamespace, caller.sync_params)
    assert [tool["function"]["name"] for tool in projected.api_params["tools"]] == ["click"]
    assert [tool["function"]["name"] for tool in params.api_params["tools"]] == ["click", "screenshot"]


def test_anthropic_retry_boundary_replaces_reexpanded_tools_with_active_task_surface() -> None:
    caller = _RetryBoundaryCaller(api_type="anthropic_chat_completion")
    click = {
        "name": "click",
        "description": "click",
        "input_schema": {"type": "object", "properties": {}},
    }
    screenshot = {
        "name": "screenshot",
        "description": "screenshot",
        "input_schema": {"type": "object", "properties": {}},
    }
    params = _retry_params(Message.user("click"), api_type="anthropic_chat_completion")
    params.api_params = {"tools": [click, screenshot], "tool_choice": {"type": "any"}}
    install_active_provider_tool_payload(caller, [click])
    NexAUProviderAdapter().install_on_caller(cast(LLMCaller, caller))

    caller._call_with_retry(params)

    projected = cast(SimpleNamespace, caller.sync_params)
    assert [tool["name"] for tool in projected.api_params["tools"]] == ["click"]
    assert [tool["name"] for tool in params.api_params["tools"]] == ["click", "screenshot"]


@pytest.mark.parametrize("api_type", ["openai_chat_completion", "openai_responses"])
def test_installed_retry_boundary_marks_union_tools_non_strict_without_mutating_params(api_type: str) -> None:
    caller = _RetryBoundaryCaller(api_type=api_type)
    union_schema = {
        "type": "object",
        "oneOf": [
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            {
                "type": "object",
                "properties": {"duration_ms": {"type": "integer"}},
                "required": ["duration_ms"],
            },
        ],
    }
    function = {"name": "web_wait", "parameters": union_schema}
    raw_tool = {"type": "function", "function": function} if api_type == "openai_chat_completion" else function
    params = _retry_params(Message.user("wait"), api_type=api_type)
    params.api_params = {"tools": [raw_tool]}

    NexAUProviderAdapter().install_on_caller(cast(LLMCaller, caller))
    caller._call_with_retry(params)

    projected = cast(SimpleNamespace, caller.sync_params)
    assert projected is not params
    projected_tool = projected.api_params["tools"][0]
    projected_function = projected_tool["function"] if api_type == "openai_chat_completion" else projected_tool
    assert projected_function["strict"] is False
    assert projected_function["parameters"]["oneOf"] == union_schema["oneOf"]
    assert "strict" not in function


def test_installed_retry_boundary_normalizes_public_only_response_fields() -> None:
    caller = _RetryBoundaryCaller(api_type="openai_chat_completion")
    caller.response = ModelResponse.from_openai_message(
        {"role": "assistant", "content": "answer", "reasoning": "public-only fallback"}
    )
    params = _retry_params(Message.user("hello"), api_type="openai_chat_completion")

    NexAUProviderAdapter().install_on_caller(cast(LLMCaller, caller))
    response = caller._call_with_retry(params)

    assert response is caller.response
    assert response is not None
    assert response.reasoning_content is None


@pytest.mark.asyncio
async def test_real_llm_caller_async_middleware_path_preserves_private_call_shape() -> None:
    caller = LLMCaller(
        openai_client=object(),
        llm_config=_llm_config(),
        middleware_manager=MiddlewareManager([Middleware()]),
    )
    captured: dict[str, object] = {}

    async def fake_retry(params: object, sync_call_fn: object | None = None) -> None:
        captured["params"] = params
        captured["sync_call_fn"] = sync_call_fn

    caller._call_with_retry_async = fake_retry  # type: ignore[method-assign]
    NexAUProviderAdapter().install_on_caller(caller)

    result = await caller.call_llm_async(
        messages=[Message(role=Role.ASSISTANT, content=[ReasoningBlock(text="internal")])],
    )

    assert result is None
    params = cast(SimpleNamespace, captured["params"])
    assert isinstance(params.messages[0].content[0], TextBlock)
    assert callable(captured["sync_call_fn"])


class _InvalidThinkingSignatureMessages:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> None:
        self.requests.append(kwargs)
        request = httpx.Request("POST", "https://example.invalid/v1/messages")
        response = httpx.Response(400, request=request)
        raise anthropic.BadRequestError(
            "invalid `signature` in `thinking` block",
            response=response,
            body={"type": "error"},
        )


class _InvalidThinkingSignatureClient:
    def __init__(self) -> None:
        self.messages = _InvalidThinkingSignatureMessages()


class _AsyncInvalidThinkingSignatureMessages:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> None:
        self.requests.append(kwargs)
        request = httpx.Request("POST", "https://example.invalid/v1/messages")
        response = httpx.Response(400, request=request)
        raise anthropic.BadRequestError(
            "invalid `signature` in `thinking` block",
            response=response,
            body={"type": "error"},
        )


class _AsyncInvalidThinkingSignatureClient:
    def __init__(self) -> None:
        self.messages = _AsyncInvalidThinkingSignatureMessages()


def test_invalid_anthropic_signature_keeps_private_outer_retry_and_signed_payload() -> None:
    client = _InvalidThinkingSignatureClient()
    adapter = NexAUProviderAdapter()
    caller = adapter.create_llm_caller(
        openai_client=client,
        llm_config=_llm_config(),
        retry_attempts=2,
        retry_backoff_max_seconds=0,
    )
    messages = [
        Message(
            role=Role.ASSISTANT,
            content=[ReasoningBlock(text="internal", signature="stale-signature"), TextBlock(text="answer")],
        ),
        Message.user("continue"),
    ]

    with pytest.raises(anthropic.BadRequestError, match="invalid.*signature"):
        adapter.call_llm(
            caller,
            LegacyLLMCallRequest(messages=messages, max_tokens=32),
        )

    assert len(client.messages.requests) == 2
    for request in client.messages.requests:
        thinking = request["messages"][0]["content"][0]  # type: ignore[index]
        assert thinking == {"type": "thinking", "thinking": "internal", "signature": "stale-signature"}


@pytest.mark.asyncio
async def test_async_invalid_anthropic_signature_keeps_private_outer_retry() -> None:
    initial = _AsyncInvalidThinkingSignatureClient()
    replacement = _AsyncInvalidThinkingSignatureClient()
    adapter = NexAUProviderAdapter()
    caller = adapter.create_llm_caller(
        openai_client=object(),
        async_openai_client=initial,
        llm_config=_llm_config(),
        retry_attempts=2,
        retry_backoff_max_seconds=0,
    )
    messages = [
        Message(
            role=Role.ASSISTANT,
            content=[ReasoningBlock(text="internal", signature="stale-signature"), TextBlock(text="answer")],
        ),
        Message.user("continue"),
    ]
    caller.async_openai_client = replacement

    with pytest.raises(anthropic.BadRequestError, match="invalid.*signature"):
        await adapter.call_llm_async(
            caller,
            LegacyLLMCallRequest(messages=messages, max_tokens=32),
        )

    assert initial.messages.requests == []
    assert len(replacement.messages.requests) == 2
    for request in replacement.messages.requests:
        thinking = request["messages"][0]["content"][0]  # type: ignore[index]
        assert thinking["signature"] == "stale-signature"
