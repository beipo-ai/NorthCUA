from __future__ import annotations

import base64
import io

import pytest
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution import llm_caller
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock, ToolUseBlock
from PIL import Image

from northcua.agentinfra.providers.provider_payload_strategies import ProviderPayloadStrategyRegistry
from northcua.agentinfra.providers.provider_payloads import (
    adapt_structured_tools_for_api_type,
    build_provider_request_payload,
    default_provider_payload_strategy_registry,
    default_tool_choice_for_api_type,
    prepare_runtime_messages_for_provider,
)
from northcua.agentinfra.providers.provider_profile_strategies import ProviderStrategyRegistry
from northcua.agentinfra.providers.provider_profiles import (
    ImageTransportPolicy,
    ProviderFamily,
    default_provider_strategy_registry,
    detect_gateway_name,
    resolve_provider_profile,
)


def test_detect_gateway_name_uses_explicit_profile_not_base_url() -> None:
    assert detect_gateway_name("https://gateway.example.invalid/v1") is None
    assert detect_gateway_name(explicit_profile="claude-compatible-gateway") == "claude-compatible-gateway"


def test_resolve_provider_profile_for_configured_anthropic_gateway_prefers_multimodal() -> None:
    profile = resolve_provider_profile(
        LLMConfig(
            model="claude-opus-4-6",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="anthropic_chat_completion",
            provider_profile="claude-compatible-gateway",
        )
    )

    assert profile.gateway_name == "claude-compatible-gateway"
    assert profile.provider_family == ProviderFamily.ANTHROPIC_MESSAGES
    assert profile.preferred_for_multimodal is True
    assert profile.image_transport == ImageTransportPolicy.ANTHROPIC_BASE64
    assert profile.transcode_images_to_jpeg is True


def test_resolve_provider_profile_for_generic_anthropic_defaults_to_jpeg_base64() -> None:
    profile = resolve_provider_profile(
        LLMConfig(
            model="claude-opus-4-6",
            base_url="https://anthropic.example.invalid/v1",
            api_key="key",
            api_type="anthropic_chat_completion",
        )
    )

    assert profile.gateway_name is None
    assert profile.provider_family == ProviderFamily.ANTHROPIC_MESSAGES
    assert profile.image_transport == ImageTransportPolicy.ANTHROPIC_BASE64
    assert profile.transcode_images_to_jpeg is True


def test_provider_strategy_registries_resolve_expected_families() -> None:
    capability_registry = default_provider_strategy_registry()
    payload_registry = default_provider_payload_strategy_registry()

    assert capability_registry.capability_strategy_for(ProviderFamily.ANTHROPIC_MESSAGES).provider_family == (
        ProviderFamily.ANTHROPIC_MESSAGES
    )
    assert payload_registry.strategy_for(ProviderFamily.OPENAI_RESPONSES).provider_family == (
        ProviderFamily.OPENAI_RESPONSES
    )


@pytest.mark.parametrize("api_type", ["openai_chat_completion", "openai_responses"])
def test_openai_tool_projection_marks_structural_unions_non_strict(api_type: str) -> None:
    union_schema = {
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"duration_ms": {"type": "integer"}},
                "required": ["duration_ms"],
                "additionalProperties": False,
            },
        ],
    }
    definitions = [
        {"name": "web_wait", "description": "Wait.", "input_schema": union_schema, "kind": "tool"},
        {
            "name": "web_open",
            "description": "Open.",
            "input_schema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            "kind": "tool",
        },
    ]

    tools = adapt_structured_tools_for_api_type(tool_definitions=definitions, api_type=api_type)
    wait = tools[0] if api_type == "openai_responses" else tools[0]["function"]
    open_tool = tools[1] if api_type == "openai_responses" else tools[1]["function"]

    assert wait["strict"] is False
    assert wait["parameters"]["oneOf"] == union_schema["oneOf"]
    assert "strict" not in open_tool


def test_empty_provider_strategy_registries_use_unknown_fallbacks() -> None:
    assert ProviderStrategyRegistry().capability_strategy_for(ProviderFamily.OPENAI_CHAT).provider_family == (
        ProviderFamily.UNKNOWN
    )
    assert ProviderPayloadStrategyRegistry().strategy_for(ProviderFamily.OPENAI_CHAT).provider_family == (
        ProviderFamily.UNKNOWN
    )


def test_resolve_provider_profile_for_anthropic_can_disable_jpeg_transcoding() -> None:
    profile = resolve_provider_profile(
        LLMConfig(
            model="claude-opus-4-6",
            base_url="https://anthropic.example.invalid/v1",
            api_key="key",
            api_type="anthropic_chat_completion",
            transcode_images_to_jpeg=False,
        )
    )

    assert profile.provider_family == ProviderFamily.ANTHROPIC_MESSAGES
    assert profile.transcode_images_to_jpeg is False


def test_resolve_provider_profile_for_openai_chat_can_disable_images_by_config() -> None:
    profile = resolve_provider_profile(
        LLMConfig(
            model="claude-opus-4-6",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_chat_completion",
            provider_profile="claude-compatible-gateway",
            supports_images=False,
        )
    )

    assert profile.gateway_name == "claude-compatible-gateway"
    assert profile.provider_family == ProviderFamily.OPENAI_CHAT
    assert profile.supports_images is False


def test_build_provider_request_payload_for_anthropic_uses_system_and_messages() -> None:
    llm_config = LLMConfig(
        model="claude-opus-4-6",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="anthropic_chat_completion",
        max_tokens=128,
    )

    prepared = build_provider_request_payload(
        llm_config=llm_config,
        messages=[
            Message(role=Role.SYSTEM, content=[TextBlock(text="System")]),
            Message(role=Role.USER, content=[TextBlock(text="Task")]),
        ],
    )

    assert prepared.provider_family == ProviderFamily.ANTHROPIC_MESSAGES
    assert prepared.request_payload["system"][0]["text"] == "System"
    assert prepared.request_payload["messages"][0]["role"] == "user"
    assert prepared.request_payload["messages"][0]["content"][0]["text"] == "Task"
    assert "provider_profile" not in prepared.request_payload


def test_build_provider_request_payload_coerces_mixed_message_inputs() -> None:
    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="gpt-5.5",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_chat_completion",
            max_tokens=128,
        ),
        messages=[
            Message(role=Role.SYSTEM, content=[TextBlock(text="System")]),
            {"role": "user", "content": "Task"},
        ],
    )

    assert prepared.request_payload["messages"][0]["role"] == "system"
    assert prepared.request_payload["messages"][0]["content"] == "System"
    assert prepared.request_payload["messages"][1]["role"] == "user"
    assert prepared.request_payload["messages"][1]["content"] == "Task"


def test_openai_chat_tool_result_image_is_injected_as_user_image_message() -> None:
    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="gpt-5.5",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_chat_completion",
            max_tokens=128,
            reasoning_effort="xhigh",
        ),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(id="call_1", name="capture_screen", input={}, raw_input="{}")],
            ),
            Message(
                role=Role.TOOL,
                content=[
                    ToolResultBlock(
                        tool_use_id="call_1",
                        content=[
                            TextBlock(text="Captured screenshot."),
                            ImageBlock(url="data:image/png;base64,AAAA"),
                        ],
                    )
                ],
            ),
        ],
    )

    messages = prepared.request_payload["messages"]

    assert prepared.request_payload["reasoning_effort"] == "xhigh"
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["id"] == "call_1"
    assert messages[1] == {"role": "tool", "tool_call_id": "call_1", "content": "Captured screenshot.<image>"}
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0] == {"type": "text", "text": "Images returned by tool call call_1:"}
    assert messages[2]["content"][1]["type"] == "image_url"
    assert messages[2]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_openai_chat_tool_result_image_can_embed_in_tool_message() -> None:
    llm_config = LLMConfig(
        model="gpt-5.5",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="openai_chat_completion",
        max_tokens=128,
        reasoning_effort="xhigh",
        tool_image_policy="embed_in_tool_message",
    )
    messages = [
        Message(
            role=Role.ASSISTANT,
            content=[ToolUseBlock(id="call_1", name="capture_screen", input={}, raw_input="{}")],
        ),
        Message(
            role=Role.TOOL,
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content=[
                        TextBlock(text="Captured screenshot."),
                        ImageBlock(url="data:image/png;base64,AAAA"),
                    ],
                )
            ],
        ),
    ]
    prepared = build_provider_request_payload(llm_config=llm_config, messages=messages)
    runtime_messages = prepare_runtime_messages_for_provider(llm_config=llm_config, messages=messages)
    runtime_payload = llm_caller.serialize_ump_to_openai_chat_payload(
        runtime_messages,
        tool_image_policy="inject_user_message",
    )

    payload_messages = prepared.request_payload["messages"]
    tool_message = payload_messages[1]

    assert "tool_image_policy" not in prepared.request_payload
    assert [message["role"] for message in payload_messages] == ["assistant", "tool"]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["content"][0] == {"type": "text", "text": "Captured screenshot."}
    assert tool_message["content"][1]["type"] == "image_url"
    assert tool_message["content"][1]["image_url"]["url"] == "data:image/png;base64,AAAA"
    assert runtime_payload == payload_messages


def test_openai_chat_reasoning_replay_policy_projects_response_summary() -> None:
    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="gpt-5.5",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_chat_completion",
            max_tokens=128,
            reasoning_replay_policy="chat_reasoning_content",
        ),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=[TextBlock(text="OK")],
                metadata={
                    "reasoning": [
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "Prior visual target was teal."}],
                            "encrypted_content": "enc-not-for-chat",
                        }
                    ]
                },
            )
        ],
    )

    assistant = prepared.request_payload["messages"][0]

    assert assistant["role"] == "assistant"
    assert assistant["content"] == "OK"
    assert assistant["reasoning_content"] == "Prior visual target was teal."
    assert "reasoning" not in assistant
    assert "reasoning_redacted_data" not in assistant


def test_build_provider_request_payload_strips_northcua_only_provider_flags() -> None:
    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="claude-opus-4-6",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="anthropic_chat_completion",
            provider_profile="claude-compatible-gateway",
            image_transport="anthropic_base64",
            transcode_images_to_jpeg=True,
            supports_images=True,
            supports_reasoning_content_replay=True,
            reasoning_replay_policy="chat_reasoning_content",
            tool_image_policy="embed_in_tool_message",
        ),
        messages=[Message(role=Role.USER, content=[TextBlock(text="Task")])],
    )

    assert "provider_profile" not in prepared.request_payload
    assert "image_transport" not in prepared.request_payload
    assert "transcode_images_to_jpeg" not in prepared.request_payload
    assert "supports_images" not in prepared.request_payload
    assert "supports_reasoning_content_replay" not in prepared.request_payload
    assert "reasoning_replay_policy" not in prepared.request_payload
    assert "tool_image_policy" not in prepared.request_payload


def test_build_provider_request_payload_transcodes_anthropic_png_image_to_jpeg() -> None:
    png_b64 = _png_b64()
    llm_config = LLMConfig(
        model="claude-opus-4-6",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="anthropic_chat_completion",
        max_tokens=128,
    )

    prepared = build_provider_request_payload(
        llm_config=llm_config,
        messages=[
            Message(
                role=Role.USER,
                content=[
                    TextBlock(text="Describe the image"),
                    ImageBlock(base64=png_b64, mime_type="image/png"),
                ],
            ),
        ],
    )

    image_block = prepared.request_payload["messages"][0]["content"][1]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/jpeg"
    assert image_block["source"]["data"] != png_b64


def test_prepare_runtime_messages_for_provider_transcodes_data_url_before_sdk_serialization() -> None:
    png_b64 = _png_b64()
    llm_config = LLMConfig(
        model="claude-opus-4-6",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="anthropic_chat_completion",
    )

    messages = prepare_runtime_messages_for_provider(
        llm_config=llm_config,
        messages=[
            Message(
                role=Role.USER,
                content=[
                    TextBlock(text="Describe the image"),
                    ImageBlock(url=f"data:image/png;base64,{png_b64}"),
                ],
            ),
        ],
    )

    image_block = messages[0].content[1]
    assert isinstance(image_block, ImageBlock)
    assert image_block.url is None
    assert image_block.mime_type == "image/jpeg"
    assert image_block.base64
    assert image_block.base64 != png_b64


def test_build_provider_request_payload_transcodes_tool_result_image_to_jpeg() -> None:
    png_b64 = _png_b64()
    llm_config = LLMConfig(
        model="claude-opus-4-6",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="anthropic_chat_completion",
        max_tokens=128,
    )

    prepared = build_provider_request_payload(
        llm_config=llm_config,
        messages=[
            Message(
                role=Role.TOOL,
                content=[
                    ToolResultBlock(
                        tool_use_id="toolu_1",
                        content=[
                            TextBlock(text="Screenshot"),
                            ImageBlock(base64=png_b64, mime_type="image/png"),
                        ],
                    )
                ],
            ),
        ],
    )

    tool_result = prepared.request_payload["messages"][0]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["content"][0]["type"] == "text"
    image_block = prepared.request_payload["messages"][0]["content"][1]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/jpeg"
    assert image_block["source"]["data"] != png_b64


def test_build_provider_request_payload_rejects_configured_openai_chat_image_inputs() -> None:
    llm_config = LLMConfig(
        model="claude-opus-4-6",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="openai_chat_completion",
        max_tokens=128,
        provider_profile="claude-compatible-gateway",
        supports_images=False,
    )

    with pytest.raises(ValueError, match="does not support image input"):
        build_provider_request_payload(
            llm_config=llm_config,
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        TextBlock(text="Describe the image"),
                        ImageBlock(url="data:image/png;base64,AAAA"),
                    ],
                ),
            ],
        )


def test_adapt_structured_tools_for_api_type_uses_anthropic_schema() -> None:
    tools = adapt_structured_tools_for_api_type(
        tool_definitions=[
            {
                "name": "ping_tool",
                "description": "Ping",
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }
        ],
        api_type="anthropic_chat_completion",
    )

    assert tools[0]["name"] == "ping_tool"
    assert "input_schema" in tools[0]
    assert "eager_input_streaming" in tools[0]


def test_default_tool_choice_for_api_type_matches_provider_family() -> None:
    assert default_tool_choice_for_api_type("openai_chat_completion", tools_present=True) == "auto"
    assert default_tool_choice_for_api_type("anthropic_chat_completion", tools_present=True) == {"type": "auto"}
    assert default_tool_choice_for_api_type("openai_responses", tools_present=True) == "auto"


def _png_b64() -> str:
    image = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
