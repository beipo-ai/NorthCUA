from __future__ import annotations

from uuid import UUID

import pytest
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.agent_context import GlobalStorage
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.archs.main_sub.execution.subagent_manager import SubAgentManager
from nexau.archs.tracer.core import BaseTracer, Span, SpanType

from northcua.agentinfra.nexau_compat.adapters.runtime import (
    BorrowedTracer,
    LegacyAgentErrorHandler,
    NexAURuntimeAdapter,
    NexAURuntimeCapabilityError,
)


class _History(list):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1


def test_legacy_error_handler_appends_and_reraises_without_user_handler() -> None:
    history = _History()
    agent = type("AgentDouble", (), {"history": history})()
    handler = LegacyAgentErrorHandler()

    with pytest.raises(RuntimeError, match="boom"):
        handler(RuntimeError("boom"), agent, {})  # type: ignore[arg-type]

    assert len(history) == 1
    assert history[0].get_text_content() == "Error: boom"
    assert history.flush_count == 1


def test_prepare_agent_config_wraps_root_and_subagent_once_on_public_candidate() -> None:
    child = type("ConfigDouble", (), {"error_handler": None, "sub_agents": {}})()
    root = type("ConfigDouble", (), {"error_handler": None, "sub_agents": {"child": child}})()
    adapter = NexAURuntimeAdapter()

    adapter.prepare_agent_config(root)
    first = root.error_handler
    adapter.prepare_agent_config(root)

    assert isinstance(first, LegacyAgentErrorHandler)
    assert root.error_handler is first
    assert isinstance(child.error_handler, LegacyAgentErrorHandler)


def test_prepare_agent_config_pins_provider_flags_for_root_and_subagent() -> None:
    child_llm = LLMConfig(
        model="child-model",
        base_url="https://example.invalid",
        api_key="fixture-key",
        api_type="anthropic_chat_completion",
    )
    root_llm = LLMConfig(
        model="root-model",
        base_url="https://example.invalid",
        api_key="fixture-key",
        api_type="anthropic_chat_completion",
    )
    child = type("ConfigDouble", (), {"error_handler": None, "llm_config": child_llm, "sub_agents": {}})()
    root = type(
        "ConfigDouble",
        (),
        {"error_handler": None, "llm_config": root_llm, "sub_agents": {"child": child}},
    )()

    NexAURuntimeAdapter().prepare_agent_config(root)

    assert root.llm_config is not root_llm
    assert child.llm_config is not child_llm
    if hasattr(root.llm_config, "tool_streaming"):
        assert root.llm_config.tool_streaming is True
        assert child.llm_config.tool_streaming is True
    if hasattr(root.llm_config, "allow_unsigned_thinking"):
        assert root.llm_config.allow_unsigned_thinking is False
        assert child.llm_config.allow_unsigned_thinking is False


class _Tracer(BaseTracer):
    def __init__(self) -> None:
        self.shutdown_count = 0

    def start_span(self, name, span_type, inputs=None, parent_span=None, attributes=None):
        return Span(id="span", name=name, type=span_type)

    def end_span(self, span, outputs=None, error=None, attributes=None) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_count += 1


class _Storage:
    def __init__(self, tracer: BaseTracer) -> None:
        self.values: dict[str, object] = {"tracer": tracer}

    def get(self, key: str, default: object | None = None) -> object | None:
        return self.values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self.values[key] = value


def test_borrowed_tracer_preserves_non_owning_private_cleanup_contract() -> None:
    delegate = _Tracer()
    borrowed = BorrowedTracer(delegate)

    span = borrowed.start_span("agent", SpanType.AGENT)
    borrowed.end_span(span)
    borrowed.shutdown()

    assert span.name == "agent"
    assert delegate.shutdown_count == 0


def test_finalize_agent_fails_closed_when_constructed_agent_contract_drifts() -> None:
    delegate = _Tracer()
    storage = _Storage(delegate)
    agent = type("AgentDouble", (), {"global_storage": storage})()
    adapter = NexAURuntimeAdapter()

    with pytest.raises(NexAURuntimeCapabilityError, match="_tool_registry"):
        adapter.finalize_created_agent(agent)  # type: ignore[arg-type]


def test_finalize_agent_installs_provider_compatibility_on_internal_caller() -> None:
    caller = LLMCaller(
        openai_client=object(),
        llm_config=LLMConfig(
            model="fixture-model",
            base_url="https://example.invalid",
            api_key="fixture-key",
            api_type="anthropic_chat_completion",
        ),
    )
    registry = type("RegistryDouble", (), {"get_all": lambda self: {}})()
    executor = type(
        "ExecutorDouble",
        (),
        {"llm_caller": caller, "update_structured_tools": lambda self, tools: None},
    )()
    agent = type(
        "AgentDouble",
        (),
        {
            "global_storage": _Storage(_Tracer()),
            "executor": executor,
            "_tool_registry": registry,
            "_build_tool_call_payload": lambda self: [],
        },
    )()

    NexAURuntimeAdapter().finalize_created_agent(agent)  # type: ignore[arg-type]

    assert getattr(caller, "__northcua_provider_compat_installed__", False) is True


class _ClosableClient:
    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_finalize_real_public_agent_contract_and_cleanup() -> None:
    storage = GlobalStorage()
    storage.set("tracer", _Tracer())
    agent = await Agent.create(
        config=AgentConfig(
            name="runtime-contract",
            description="runtime contract fixture",
            system_prompt="stable system",
            llm_config=LLMConfig(
                model="fixture-model",
                base_url="https://example.invalid",
                api_key="fixture-key",
                api_type="openai_responses",
            ),
        ),
        openai_client=_ClosableClient(),
        global_storage=storage,
        user_id="runtime-contract-user",
        session_id="runtime-contract-session",
    )
    try:
        result = NexAURuntimeAdapter().finalize_created_agent(agent)

        assert result is agent
        assert getattr(agent.executor.llm_caller, "__northcua_provider_compat_installed__", False) is True
        assert isinstance(agent.global_storage.get("tracer"), BorrowedTracer)
    finally:
        agent.sync_cleanup()


@pytest.mark.asyncio
async def test_creation_hook_finalizes_real_async_subagent_manager_child(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = NexAURuntimeAdapter()
    adapter.install_agent_creation_hook()
    child_config = AgentConfig(
        name="runtime-child",
        description="runtime child fixture",
        system_prompt="stable child system",
        llm_config=LLMConfig(
            model="fixture-model",
            base_url="https://example.invalid",
            api_key="fixture-key",
            api_type="openai_responses",
        ),
    )
    manager = SubAgentManager(
        agent_name="runtime-parent",
        sub_agents={"child": child_config},
        global_storage=GlobalStorage(),
        user_id="runtime-child-user",
        session_id="runtime-child-session",
    )
    observed: list[tuple[bool, bool]] = []

    async def fake_run_async(child: Agent, **_kwargs: object) -> str:
        observed.append(
            (
                bool(getattr(child, "__northcua_runtime_compat_finalized__", False)),
                bool(getattr(child.executor.llm_caller, "__northcua_provider_compat_installed__", False)),
            )
        )
        child.sync_cleanup()
        return "child-result"

    monkeypatch.setattr(Agent, "run_async", fake_run_async)

    result = await manager.call_sub_agent_async("child", "delegated task")

    assert observed == [(True, True)]
    assert "child-result" in result


def test_creation_hook_finalizes_real_sync_subagent_resume_child(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = NexAURuntimeAdapter()
    adapter.install_agent_creation_hook()
    child_config = AgentConfig(
        name="runtime-sync-child",
        description="runtime sync child fixture",
        system_prompt="stable child system",
        llm_config=LLMConfig(
            model="fixture-model",
            base_url="https://example.invalid",
            api_key="fixture-key",
            api_type="openai_responses",
        ),
    )
    manager = SubAgentManager(
        agent_name="runtime-parent",
        sub_agents={"child": child_config},
        global_storage=GlobalStorage(),
        user_id="runtime-sync-child-user",
        session_id="runtime-sync-child-session",
    )
    observed: list[tuple[str, bool, bool]] = []

    def fake_run(child: Agent, **_kwargs: object) -> str:
        observed.append(
            (
                child.agent_id,
                bool(getattr(child, "__northcua_runtime_compat_finalized__", False)),
                bool(getattr(child.executor.llm_caller, "__northcua_provider_compat_installed__", False)),
            )
        )
        child.sync_cleanup()
        return "resumed-child-result"

    monkeypatch.setattr(Agent, "run", fake_run)

    result = manager.call_sub_agent("child", "resume task", sub_agent_id="stable-child-id")

    assert observed == [("stable-child-id", True, True)]
    assert "resumed-child-result" in result
