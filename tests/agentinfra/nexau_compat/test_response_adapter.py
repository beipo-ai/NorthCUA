from __future__ import annotations

from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.core.messages import ReasoningBlock, TextBlock
from nexau.core.usage import TokenUsage

from northcua.agentinfra.nexau_compat.adapters.responses import normalize_legacy_model_response


def test_openai_chat_bare_reasoning_matches_private_ignored_field() -> None:
    response = ModelResponse.from_openai_message(
        {"role": "assistant", "content": "answer", "reasoning": "public-only fallback"}
    )

    normalized = normalize_legacy_model_response("openai_chat_completion", response)

    assert normalized is response
    assert response.reasoning_content is None


def test_openai_chat_explicit_empty_reasoning_is_not_persisted_to_history() -> None:
    response = ModelResponse.from_openai_message({"role": "assistant", "content": "answer", "reasoning_content": ""})

    normalized = normalize_legacy_model_response("openai_chat_completion", response)

    assert normalized is response
    assert response.reasoning_content is None
    message = response.to_ump_message()
    assert [type(block) for block in message.content] == [TextBlock]
    assert not any(isinstance(block, ReasoningBlock) for block in message.content)


def test_openai_responses_empty_reasoning_matches_private_absence() -> None:
    response = ModelResponse.from_openai_response(
        {
            "output": [
                {
                    "id": "reasoning-1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": ""}],
                },
                {
                    "id": "message-1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            ]
        }
    )

    normalize_legacy_model_response("openai_responses", response)

    assert response.reasoning_content is None


def test_anthropic_empty_signature_matches_private_empty_string() -> None:
    response = ModelResponse.from_anthropic_message(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "internal", "signature": ""},
                {"type": "text", "text": "answer"},
            ],
        }
    )

    normalize_legacy_model_response("anthropic_chat_completion", response)

    assert response.reasoning_content == "internal"
    assert response.reasoning_signature == ""


def test_anthropic_valid_signature_is_unchanged() -> None:
    response = ModelResponse.from_anthropic_message(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "internal", "signature": "signed"},
                {"type": "text", "text": "answer"},
            ],
        }
    )

    normalize_legacy_model_response("anthropic_chat_completion", response)

    assert response.reasoning_signature == "signed"


def test_anthropic_empty_reasoning_is_not_persisted_to_history() -> None:
    response = ModelResponse.from_anthropic_message(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "", "signature": ""},
                {"type": "text", "text": "answer"},
            ],
        }
    )

    normalize_legacy_model_response("anthropic_chat_completion", response)

    assert response.reasoning_content is None
    message = response.to_ump_message()
    assert [type(block) for block in message.content] == [TextBlock]
    assert not any(isinstance(block, ReasoningBlock) for block in message.content)


def test_openai_responses_stream_items_match_private_schema() -> None:
    response = ModelResponse(
        response_items=[
            {
                "id": "reasoning-1",
                "type": "reasoning",
                "summary": [],
                "content": None,
                "status": "completed",
            },
            {
                "id": "call-1",
                "type": "function_call",
                "name": "lookup",
                "arguments": "{}",
                "namespace": None,
                "status": "completed",
            },
        ]
    )

    normalize_legacy_model_response("openai_responses", response, stream=True)

    assert response.response_items == [
        {"id": "reasoning-1", "type": "reasoning", "summary": []},
        {"id": "call-1", "type": "function_call", "name": "lookup", "arguments": "{}"},
    ]


def test_anthropic_stream_usage_matches_private_output_only_accounting() -> None:
    response = ModelResponse(
        usage=TokenUsage(
            input_tokens=8,
            completion_tokens=5,
            reasoning_tokens=2,
            total_tokens=15,
            cache_creation_tokens=2,
            cache_read_tokens=3,
            input_tokens_uncached=8,
        )
    )

    normalize_legacy_model_response("anthropic_chat_completion", response, stream=True)

    assert response.usage == TokenUsage(completion_tokens=5, reasoning_tokens=2, total_tokens=7)


def test_anthropic_non_stream_usage_is_unchanged() -> None:
    usage = TokenUsage(input_tokens=8, completion_tokens=5, total_tokens=13, input_tokens_uncached=8)
    response = ModelResponse(usage=usage)

    normalize_legacy_model_response("anthropic_chat_completion", response, stream=False)

    assert response.usage is usage
