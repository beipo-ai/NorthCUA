from __future__ import annotations

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig

from northcua.agentinfra.providers.llm_fallback import (
    agent_config_for_llm_candidate,
    attach_llm_fallbacks,
    conservative_context_budget_inputs,
    llm_fallback_candidates,
    parse_llm_fallbacks,
    runtime_llm_config,
)


def test_parse_llm_fallbacks_inherits_connection_and_provider_flags() -> None:
    primary = LLMConfig(
        model="claude-opus-4-6",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="anthropic_chat_completion",
        provider_profile="claude-compatible-gateway",
        image_transport="anthropic_base64",
        transcode_images_to_jpeg=True,
        reasoning={"effort": "high"},
    )

    fallbacks = parse_llm_fallbacks(
        [
            {
                "name": "sonnet",
                "model": "claude-sonnet-4-6",
            }
        ],
        primary_llm_config=primary,
    )

    assert len(fallbacks) == 1
    assert fallbacks[0].name == "sonnet"
    assert fallbacks[0].llm_config.model == "claude-sonnet-4-6"
    assert fallbacks[0].llm_config.base_url == "https://gateway.example.invalid/v1"
    assert fallbacks[0].llm_config.extra_params["provider_profile"] == "claude-compatible-gateway"
    assert fallbacks[0].llm_config.extra_params["transcode_images_to_jpeg"] is True
    assert "reasoning" not in fallbacks[0].llm_config.extra_params


def test_llm_fallback_candidates_return_primary_then_fallbacks() -> None:
    primary = LLMConfig(model="primary", base_url="https://gateway.example.invalid/v1", api_key="key")
    fallback = parse_llm_fallbacks([{"model": "backup"}], primary_llm_config=primary)
    agent_config = attach_llm_fallbacks(
        AgentConfig(name="agent", description="agent", llm_config=primary),
        fallback,
    )

    candidates = llm_fallback_candidates(agent_config)

    assert [candidate.name for candidate in candidates] == ["primary", "backup"]
    assert candidates[0].is_primary is True
    assert candidates[1].is_primary is False


def test_agent_config_for_llm_candidate_reuses_nexau_config_with_new_llm() -> None:
    primary = LLMConfig(model="primary", base_url="https://gateway.example.invalid/v1", api_key="key")
    fallback = parse_llm_fallbacks([{"model": "backup"}], primary_llm_config=primary)[0]
    agent_config = AgentConfig(
        name="agent",
        description="agent",
        llm_config=primary,
        global_storage={"tracer": object()},
    )

    candidate_config = agent_config_for_llm_candidate(agent_config, fallback)

    assert candidate_config is not agent_config
    assert candidate_config.llm_config.model == "backup"
    assert candidate_config.global_storage["tracer"] is agent_config.global_storage["tracer"]


def test_runtime_llm_config_strips_northcua_provider_flags_but_keeps_provider_params() -> None:
    llm_config = LLMConfig(
        model="claude-opus-4-6",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        api_type="anthropic_chat_completion",
        provider_profile="claude-compatible-gateway",
        image_transport="anthropic_base64",
        transcode_images_to_jpeg=True,
        reasoning={"effort": "high"},
    )

    runtime_config = runtime_llm_config(llm_config)

    assert runtime_config.model == "claude-opus-4-6"
    assert runtime_config.api_type == "anthropic_chat_completion"
    assert runtime_config.extra_params["reasoning"] == {"effort": "high"}
    assert "provider_profile" not in runtime_config.extra_params
    assert "image_transport" not in runtime_config.extra_params
    assert "transcode_images_to_jpeg" not in runtime_config.extra_params


def test_runtime_llm_config_drops_runtime_budget_provider_param() -> None:
    runtime_config = runtime_llm_config(
        LLMConfig(
            model="gpt-5.4",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_responses",
            max_context_tokens=200000,
        )
    )

    assert "max_context_tokens" not in runtime_config.extra_params


def test_runtime_llm_config_strips_chat_reasoning_replay_policy() -> None:
    runtime_config = runtime_llm_config(
        LLMConfig(
            model="qwen",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_chat_completion",
            reasoning_replay_policy="chat_reasoning_content",
        )
    )

    assert "reasoning_replay_policy" not in runtime_config.extra_params


def test_runtime_llm_config_strips_anthropic_v1_base_url_for_sdk_runtime() -> None:
    runtime_config = runtime_llm_config(
        LLMConfig(
            model="claude-opus-4-6",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="anthropic_chat_completion",
        )
    )

    assert runtime_config.base_url == "https://gateway.example.invalid"


def test_runtime_llm_config_keeps_openai_v1_base_url() -> None:
    runtime_config = runtime_llm_config(
        LLMConfig(
            model="gpt-5.4",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_responses",
        )
    )

    assert runtime_config.base_url == "https://gateway.example.invalid/v1"


def test_conservative_context_budget_uses_smallest_window_and_largest_output() -> None:
    primary = LLMConfig(
        model="primary",
        base_url="https://gateway.example.invalid/v1",
        api_key="key",
        max_tokens=2_000,
    )
    fallbacks = parse_llm_fallbacks(
        [
            {"model": "small-window", "max_context_tokens": 80_000, "max_tokens": 4_000},
            {"model": "large-window", "max_context_tokens": 200_000, "max_tokens": 1_000},
        ],
        primary_llm_config=primary,
    )
    agent_config = attach_llm_fallbacks(
        AgentConfig(llm_config=primary, max_context_tokens=128_000),
        fallbacks,
    )

    assert conservative_context_budget_inputs(agent_config, llm_fallback_candidates(agent_config)) == (80_000, 4_000)
