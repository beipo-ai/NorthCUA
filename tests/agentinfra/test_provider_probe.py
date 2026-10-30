from __future__ import annotations

from nexau.archs.llm.llm_config import LLMConfig

from northcua.agentinfra.providers.provider_probe import _provider_headers, _provider_request_url
from northcua.agentinfra.providers.provider_probe_cases import default_probe_cases


def test_provider_request_url_for_openai_chat_uses_chat_completions() -> None:
    llm_config = LLMConfig(
        model="claude-opus-4-6",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="openai_chat_completion",
    )

    assert _provider_request_url(llm_config) == "https://gateway.example.invalid/v1/chat/completions"


def test_provider_request_url_for_anthropic_uses_messages_endpoint() -> None:
    llm_config = LLMConfig(
        model="claude-opus-4-6",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="anthropic_chat_completion",
    )

    assert _provider_request_url(llm_config) == "https://gateway.example.invalid/v1/messages"


def test_provider_headers_switch_to_anthropic_keys() -> None:
    llm_config = LLMConfig(
        model="claude-opus-4-6",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="anthropic_chat_completion",
    )

    headers = _provider_headers(llm_config)

    assert headers["x-api-key"] == "key"
    assert headers["anthropic-version"] == "2023-06-01"


def test_anthropic_tool_probe_uses_runtime_auto_tool_choice() -> None:
    tool_case = next(case for case in default_probe_cases("anthropic_chat_completion") if case.case_id == "tool_ping")

    assert tool_case.tool_choice == {"type": "auto"}


def test_openai_responses_tool_probe_uses_string_tool_choice() -> None:
    tool_case = next(case for case in default_probe_cases("openai_responses") if case.case_id == "tool_ping")

    assert tool_case.tool_choice == "required"
