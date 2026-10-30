from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.core.messages import Message, Role, TextBlock

from northcua.agentinfra.context.adapters import LazySummaryClient, is_context_overflow_error
from northcua.agentinfra.context.bootstrap import prepare_context_messages
from northcua.agentinfra.context.bootstrap import runtime as context_runtime
from northcua.agentkit.model_step import _logical_agent_path, _task_epoch
from northcua.core.config import NorthCUAAppConfig


def _message(text: str) -> Message:
    return Message(role=Role.USER, content=[TextBlock(text=text)])


def test_context_overflow_classifier_walks_wrapped_provider_errors() -> None:
    try:
        try:
            raise RuntimeError("maximum context length is 128000 tokens")
        except RuntimeError as provider_error:
            raise ValueError("provider request failed") from provider_error
    except ValueError as wrapped:
        assert is_context_overflow_error(wrapped)

    assert not is_context_overflow_error(RuntimeError("429 rate limit"))
    assert not is_context_overflow_error(TimeoutError("request timed out"))


def test_task_epoch_is_stable_across_rounds_and_changes_across_tasks() -> None:
    first_round = _task_epoch({"user_query": "Book the verified flight", "round_num": 1})
    later_round = _task_epoch({"user_query": "Book the verified flight", "round_num": 99})
    other_task = _task_epoch({"user_query": "Cancel the reservation", "round_num": 1})

    assert first_round == later_round
    assert first_round != other_task
    assert _task_epoch({"evaluator_kind": "web_round_evaluator", "round_num": 2}) != _task_epoch(
        {"evaluator_kind": "web_round_evaluator", "round_num": 3}
    )


def test_logical_agent_path_separates_same_kind_instances() -> None:
    assert _logical_agent_path({"logical_agent_path": "root/researcher-1"}, agent_kind="generic") != (
        _logical_agent_path({"logical_agent_path": "root/researcher-2"}, agent_kind="generic")
    )
    assert _logical_agent_path({}, agent_kind="web") == "web"


@pytest.mark.asyncio
async def test_off_mode_returns_same_message_list_and_creates_no_artifacts(tmp_path: Path) -> None:
    app_config = NorthCUAAppConfig.model_validate(
        {
            "agent_config_path": "config/agent.example.yaml",
            "artifacts": {"root_dir": str(tmp_path)},
            "context_management": {"mode": "off"},
        }
    )
    messages = [_message("original")]
    llm_config = LLMConfig(
        model="gpt-4o",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        max_tokens=100,
    )

    result = await prepare_context_messages(
        app_config=app_config,
        agent_kind="web",
        user_id="user",
        session_id="session",
        task_epoch="task",
        logical_agent_path="web",
        logical_step_id="step",
        round_num=1,
        messages=messages,
        tools=[],
        agent_config=AgentConfig(llm_config=llm_config, max_context_tokens=1000),
        prepared_agent=cast(Agent, object()),
        primary_llm_config=llm_config,
    )

    assert result.messages is messages
    assert result.result is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_shadow_audits_full_candidate_but_returns_legacy_provider_messages(tmp_path: Path) -> None:
    app_config = NorthCUAAppConfig.model_validate(
        {
            "agent_config_path": "config/agent.example.yaml",
            "artifacts": {"root_dir": str(tmp_path)},
            "context_management": {
                "mode": "shadow",
                "budget": {"safety_margin_min_tokens": 0},
            },
        }
    )
    legacy = [_message("legacy provider prompt")]
    full_candidate = [_message("old history " * 10_000), *legacy]
    llm_config = LLMConfig(
        model="gpt-4o",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        max_tokens=100,
    )

    prepared = await prepare_context_messages(
        app_config=app_config,
        agent_kind="web",
        user_id="user",
        session_id="session",
        task_epoch="task",
        logical_agent_path="web",
        logical_step_id="step",
        round_num=1,
        messages=legacy,
        shadow_messages=full_candidate,
        tools=[],
        agent_config=AgentConfig(llm_config=llm_config, max_context_tokens=2_000),
        prepared_agent=cast(Agent, object()),
        primary_llm_config=llm_config,
    )

    assert prepared.messages == legacy
    assert prepared.messages[0] is legacy[0]
    assert prepared.result is not None
    assert prepared.result.estimate_before is not None
    assert prepared.result.estimate_before.total_tokens > 2_000


@pytest.mark.asyncio
async def test_shadow_projection_failure_returns_exact_legacy_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_config = NorthCUAAppConfig.model_validate(
        {
            "agent_config_path": "config/agent.example.yaml",
            "artifacts": {"root_dir": str(tmp_path)},
            "context_management": {"mode": "shadow"},
        }
    )
    legacy = [_message("legacy")]
    llm_config = LLMConfig(
        model="gpt-4o",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        max_tokens=100,
    )

    class BrokenProjector:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def project(self, messages: list[Message], *, round_num: int = 0) -> tuple[object, ...]:
            del messages, round_num
            raise RuntimeError("shadow projection failed")

    monkeypatch.setattr(context_runtime, "UMPContextProjector", BrokenProjector)
    prepared = await prepare_context_messages(
        app_config=app_config,
        agent_kind="web",
        user_id="user",
        session_id="session",
        task_epoch="task",
        logical_agent_path="web",
        logical_step_id="step",
        round_num=1,
        messages=legacy,
        shadow_messages=[_message("full")],
        tools=[],
        agent_config=AgentConfig(llm_config=llm_config, max_context_tokens=2_000),
        prepared_agent=cast(Agent, object()),
        primary_llm_config=llm_config,
    )

    assert prepared.messages is legacy
    assert prepared.result is None


@pytest.mark.asyncio
async def test_summary_client_close_failure_never_overrides_primary_outcome() -> None:
    class BrokenClose:
        async def close(self) -> None:
            raise RuntimeError("close failed")

    await context_runtime._close_summary_client_best_effort(cast(LazySummaryClient, BrokenClose()))
