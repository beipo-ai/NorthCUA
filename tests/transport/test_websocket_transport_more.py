from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from fastmcp import FastMCP
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.main_sub.execution.parse_structures import ParsedResponse, ToolCall
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.session import SessionManager
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock

import northcua.transport.ws.transport as transport_module
from northcua.agentinfra.providers.llm_fallback import LLMFallbackCandidate, attach_llm_fallbacks, parse_llm_fallbacks
from northcua.agentinfra.runtime.protocol import AgentTransportCapabilities
from northcua.agents.desktop.runtime.model import DesktopModelStepRunner
from northcua.core.enums import ResultStatus, RuntimeMode, WebSocketMessageType
from northcua.core.models import CommandExecutionResultRecord
from northcua.core.protocol import CommandRequestPayload, NorthCUAWebSocketEnvelope
from northcua.runtime.mcp.device_executor import NorthCUADeviceCommandExecutor
from northcua.runtime.mcp.invocation_context import current_mcp_invocation_context
from northcua.runtime.mcp.registry import MCPFactoryContext, NorthCUAMCPRegistry
from northcua.transport.ws.client import NorthCUAWebSocketClient
from northcua.transport.ws.config import NorthCUAWebSocketTransportConfig
from northcua.transport.ws.local_step import LocalStepModelResult
from northcua.transport.ws.local_step.single_model import ModelStepLLMTimeoutError, _model_step_timeout_seconds
from northcua.transport.ws.remote_tools import RemoteToolProxyFactory, use_remote_command_tool_executor
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult
from northcua.transport.ws.transport import NorthCUAWebSocketTransport

from tests.support.nexau_config import agent_config_with_runtime_mcp


class _ScriptedWebSocketConnection:
    """Small async websocket fake that feeds scripted envelopes to one client."""

    def __init__(self, *envelopes: NorthCUAWebSocketEnvelope) -> None:
        self._raw_messages = [envelope.model_dump_json() for envelope in envelopes]
        self.sent_messages: list[str] = []

    def __aiter__(self) -> "_ScriptedWebSocketConnection":
        return self

    async def __anext__(self) -> str:
        if not self._raw_messages:
            raise StopAsyncIteration
        return self._raw_messages.pop(0)

    async def send(self, payload: str) -> None:
        self.sent_messages.append(payload)


def _agent_config() -> AgentConfig:
    return agent_config_with_runtime_mcp(
        {
            "name": "NorthCUATestAgent",
            "description": "test",
            "system_prompt": "You are NorthCUA.",
            "llm_config": LLMConfig(
                model="gpt-5.4",
                base_url="http://localhost:3001/v1",
                api_key="key",
                api_type="openai_responses",
            ),
            "mcp_servers": [
                {"name": "DesktopNativeDataExecutor", "type": "stdio", "model_visible": False},
                {"name": "DesktopNativeActionExecutor", "type": "stdio"},
                {"name": "CommandLineExecutor", "type": "stdio"},
            ],
        }
    )


def _bypass_runtime_finalization_for_agent_creation_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep cache-key tests focused when ``Agent.create`` is replaced by a stub."""

    monkeypatch.setattr(
        transport_module._NEXAU_RUNTIME_ADAPTER,
        "install_agent_creation_hook",
        lambda: None,
    )
    monkeypatch.setattr(
        transport_module._NEXAU_RUNTIME_ADAPTER,
        "finalize_created_agent",
        lambda agent: agent,
    )


def test_prepare_agent_reuses_prompt_cache_key_within_session(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_keys: list[str | None] = []

    async def fake_create(**kwargs):
        llm_config = kwargs["config"].llm_config
        captured_keys.append(llm_config.get_param("prompt_cache_key") if llm_config else None)
        return SimpleNamespace(agent_id="agent", config=kwargs["config"])

    monkeypatch.setattr(transport_module.Agent, "create", staticmethod(fake_create))
    _bypass_runtime_finalization_for_agent_creation_stub(monkeypatch)
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    for _ in range(2):
        asyncio.run(
            transport.prepare_agent(
                user_id="user-1",
                session_id="session-stable",
                agent_config=_agent_config(),
                remote=True,
                transport_capabilities=AgentTransportCapabilities(
                    supports_remote_external_dispatch=True,
                    supports_stepwise_local_boundary=True,
                ),
            )
        )

    assert captured_keys[0]
    assert captured_keys == [captured_keys[0], captured_keys[0]]


def test_prepare_agent_prompt_cache_key_changes_across_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_keys: list[str | None] = []

    async def fake_create(**kwargs):
        llm_config = kwargs["config"].llm_config
        captured_keys.append(llm_config.get_param("prompt_cache_key") if llm_config else None)
        return SimpleNamespace(agent_id="agent", config=kwargs["config"])

    monkeypatch.setattr(transport_module.Agent, "create", staticmethod(fake_create))
    _bypass_runtime_finalization_for_agent_creation_stub(monkeypatch)
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    for session_id in ("session-a", "session-b"):
        asyncio.run(
            transport.prepare_agent(
                user_id="user-1",
                session_id=session_id,
                agent_config=_agent_config(),
                remote=True,
                transport_capabilities=AgentTransportCapabilities(supports_remote_external_dispatch=True),
            )
        )

    assert captured_keys[0]
    assert captured_keys[1]
    assert captured_keys[0] != captured_keys[1]


def test_prepare_agent_preserves_explicit_prompt_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_keys: list[str | None] = []

    async def fake_create(**kwargs):
        llm_config = kwargs["config"].llm_config
        captured_keys.append(llm_config.get_param("prompt_cache_key") if llm_config else None)
        return SimpleNamespace(agent_id="agent", config=kwargs["config"])

    monkeypatch.setattr(transport_module.Agent, "create", staticmethod(fake_create))
    _bypass_runtime_finalization_for_agent_creation_stub(monkeypatch)
    agent_config = _agent_config()
    assert agent_config.llm_config is not None
    agent_config.llm_config.set_param("prompt_cache_key", "user-provided-key")
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    asyncio.run(
        transport.prepare_agent(
            user_id="user-1",
            session_id="session-stable",
            agent_config=agent_config,
            remote=True,
            transport_capabilities=AgentTransportCapabilities(supports_remote_external_dispatch=True),
        )
    )

    assert captured_keys == ["user-provided-key"]


def test_remote_runtime_config_installs_proxy_tool_implementations() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=True,
            transport_capabilities=AgentTransportCapabilities(),
        )
    )

    click_tool = next(tool for tool in config.tools if tool.name == "click")
    assert callable(click_tool.implementation)


def test_transport_local_round_uses_stepwise_single_model_call() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    class _FakeMiddlewareManager:
        def run_before_agent(self, hook_input):
            return hook_input.messages

        def run_before_model(self, hook_input):
            return hook_input.messages

        def run_after_model(self, hook_input):
            return hook_input.parsed_response, hook_input.messages, False

    class _FakeLLMCaller:
        async def call_llm_async(self, *args, **kwargs):
            return ModelResponse(
                content="Clicking Calculator.",
                tool_calls=[
                    ModelToolCall(
                        call_id="call-1",
                        name="click",
                        arguments={"x": 1, "y": 2},
                        raw_arguments='{"x":1,"y":2}',
                    )
                ],
            )

    class _FakeResponseParser:
        def parse_response(self, response):
            return ParsedResponse(
                original_response=response.content or "",
                tool_calls=[
                    ToolCall(
                        tool_name="click",
                        parameters={"x": 1, "y": 2},
                        tool_call_id="call-1",
                        source="structured",
                    )
                ],
                model_response=response,
            )

    fake_agent = SimpleNamespace(
        agent_name="NorthCUATestAgent",
        agent_id="agent-test",
        config=_agent_config(),
        global_storage={},
        _tool_registry=SimpleNamespace(),
        sandbox_manager=None,
        _variables=None,
        _token_trace_session=None,
        skill_registry={},
        openai_client=None,
        executor=SimpleNamespace(
            middleware_manager=_FakeMiddlewareManager(),
            llm_caller=_FakeLLMCaller(),
            response_parser=_FakeResponseParser(),
            structured_tool_payload=[],
            use_structured_tool_calls=True,
            tool_call_mode="structured",
            shutdown_event=None,
            subagent_manager=None,
        ),
    )

    async def fake_prepare_agent(**kwargs):
        return fake_agent

    transport.prepare_agent = fake_prepare_agent  # type: ignore[method-assign]

    result, _agent = asyncio.run(
        transport.run_local_round(
            messages=[],
            user_id="user-1",
            session_id="sess-1",
            agent_config=_agent_config(),
        )
    )

    assert isinstance(result, tuple)
    assert result[0] == "Clicking Calculator."
    assert result[1]["stop_reason"] == "EXTERNAL_TOOL_CALL"
    assert result[1]["pending_tool_calls"] == [
        {"id": "call-1", "name": "click", "input": {"x": 1, "y": 2}},
    ]
    assert result[1]["agent_state"].agent_id == "agent-test"


def test_transport_local_single_model_step_records_llm_span() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )
    tracer = InMemoryTracer()

    class _FakeMiddlewareManager:
        def run_before_agent(self, hook_input):
            return hook_input.messages

        def run_before_model(self, hook_input):
            return hook_input.messages

        def run_after_model(self, hook_input):
            return hook_input.parsed_response, hook_input.messages, False

    class _FakeLLMCaller:
        async def call_llm_async(self, messages, *args, **kwargs):
            del messages, args, kwargs
            return ModelResponse(content='{"thought":"inspect"}', tool_calls=[])

    class _FakeResponseParser:
        def parse_response(self, response):
            return ParsedResponse(
                original_response=response.content or "",
                tool_calls=[],
                model_response=response,
            )

    class _FakePromptBuilder:
        def build_system_prompt(self, **kwargs):
            del kwargs
            return [SimpleNamespace(text="Rendered system prompt", cache=False)]

    fake_agent = SimpleNamespace(
        agent_name="NorthCUATestAgent",
        agent_id="agent-test",
        config=_agent_config(),
        global_storage={"tracer": tracer},
        _tool_registry=SimpleNamespace(compute_eager_tools=lambda: []),
        sandbox_manager=None,
        _variables=None,
        _token_trace_session=None,
        skill_registry={},
        openai_client=None,
        prompt_builder=_FakePromptBuilder(),
        use_structured_tool_calls=True,
        executor=SimpleNamespace(
            middleware_manager=_FakeMiddlewareManager(),
            llm_caller=_FakeLLMCaller(),
            response_parser=_FakeResponseParser(),
            structured_tool_payload=[],
            use_structured_tool_calls=True,
            tool_call_mode="structured",
            shutdown_event=None,
            subagent_manager=None,
        ),
    )

    asyncio.run(
        transport._run_local_single_model_step(
            messages=[Message(role=Role.USER, content=[TextBlock(text="user")])],
            agent=fake_agent,  # type: ignore[arg-type]
            context={"session_id": "sess-1", "round_num": 1},
        )
    )

    spans = list(tracer.spans.values())
    assert len(spans) == 1
    assert spans[0].type.value == "LLM"
    assert spans[0].name == "openai.responses.create"
    assert spans[0].inputs.get("model") == "gpt-5.4"
    assert spans[0].attributes["session_id"] == "sess-1"
    assert spans[0].outputs.get("assistant_text") == '{"thought":"inspect"}'


def test_transport_local_single_model_step_times_out_hung_llm_call(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="northcua.transport.ws.local_step.single_model")
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )
    agent_config = _agent_config()
    agent_config.llm_config.timeout = 0.01

    class _FakeLLMCaller:
        async def call_llm_async(self, messages, *args, **kwargs):
            del messages, args, kwargs
            await asyncio.sleep(1)
            return ModelResponse(content="too late", tool_calls=[])

    class _FakeResponseParser:
        def parse_response(self, response):
            return ParsedResponse(
                original_response=response.content or "",
                tool_calls=[],
                model_response=response,
            )

    fake_agent = SimpleNamespace(
        agent_name="NorthCUATestAgent",
        agent_id="agent-test",
        config=agent_config,
        global_storage={},
        _tool_registry=SimpleNamespace(compute_eager_tools=lambda: []),
        sandbox_manager=None,
        _variables=None,
        _token_trace_session=None,
        skill_registry={},
        openai_client=None,
        executor=SimpleNamespace(
            middleware_manager=None,
            llm_caller=_FakeLLMCaller(),
            response_parser=_FakeResponseParser(),
            structured_tool_payload=[],
            use_structured_tool_calls=True,
            tool_call_mode="structured",
            shutdown_event=None,
            subagent_manager=None,
        ),
    )

    with pytest.raises(ModelStepLLMTimeoutError):
        asyncio.run(
            transport._run_local_single_model_step(
                messages=[Message(role=Role.USER, content=[TextBlock(text="user")])],
                agent=fake_agent,  # type: ignore[arg-type]
                context={"session_id": "sess-1", "round_num": 1},
            )
        )

    assert any(record.__dict__.get("event") == "single_model_step_llm_call_started" for record in caplog.records)
    timeout_records = [
        record for record in caplog.records if record.__dict__.get("event") == "single_model_step_llm_call_timed_out"
    ]
    assert len(timeout_records) == 1
    assert timeout_records[0].__dict__["configured_timeout_seconds"] == 0.01
    assert timeout_records[0].__dict__["timeout_seconds"] == 0.04


def test_transport_local_single_model_step_logs_model_call_error(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="northcua.transport.ws.local_step.single_model")
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )
    agent_config = _agent_config()
    agent_config.llm_config.timeout = 30
    agent_config.llm_config.max_retries = 1

    class _FakeLLMCaller:
        async def call_llm_async(self, messages, *args, **kwargs):
            del messages, args, kwargs
            raise RuntimeError("provider request failed")

    class _FakeResponseParser:
        def parse_response(self, response):
            return ParsedResponse(
                original_response=response.content or "",
                tool_calls=[],
                model_response=response,
            )

    fake_agent = SimpleNamespace(
        agent_name="NorthCUATestAgent",
        agent_id="agent-test",
        config=agent_config,
        global_storage={},
        _tool_registry=SimpleNamespace(compute_eager_tools=lambda: []),
        sandbox_manager=None,
        _variables=None,
        _token_trace_session=None,
        skill_registry={},
        openai_client=None,
        executor=SimpleNamespace(
            middleware_manager=None,
            llm_caller=_FakeLLMCaller(),
            response_parser=_FakeResponseParser(),
            structured_tool_payload=[],
            use_structured_tool_calls=True,
            tool_call_mode="structured",
            shutdown_event=None,
            subagent_manager=None,
        ),
    )

    with pytest.raises(RuntimeError, match="provider request failed"):
        asyncio.run(
            transport._run_local_single_model_step(
                messages=[Message(role=Role.USER, content=[TextBlock(text="user")])],
                agent=fake_agent,  # type: ignore[arg-type]
                context={"session_id": "sess-1", "round_num": 1},
            )
        )

    failed_records = [
        record for record in caplog.records if record.__dict__.get("event") == "single_model_step_llm_call_failed"
    ]
    assert len(failed_records) == 1
    assert failed_records[0].__dict__["error_type"] == "RuntimeError"
    assert failed_records[0].__dict__["configured_timeout_seconds"] == 30
    assert failed_records[0].__dict__["configured_max_retries"] == 1


def test_transport_local_single_model_step_logs_cancelled_model_call(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="northcua.transport.ws.local_step.single_model")
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )
    agent_config = _agent_config()
    agent_config.llm_config.timeout = 30
    agent_config.llm_config.max_retries = 1

    class _FakeLLMCaller:
        async def call_llm_async(self, messages, *args, **kwargs):
            del messages, args, kwargs
            await asyncio.sleep(1)
            return ModelResponse(content="too late", tool_calls=[])

    class _FakeResponseParser:
        def parse_response(self, response):
            return ParsedResponse(
                original_response=response.content or "",
                tool_calls=[],
                model_response=response,
            )

    fake_agent = SimpleNamespace(
        agent_name="NorthCUATestAgent",
        agent_id="agent-test",
        config=agent_config,
        global_storage={},
        _tool_registry=SimpleNamespace(compute_eager_tools=lambda: []),
        sandbox_manager=None,
        _variables=None,
        _token_trace_session=None,
        skill_registry={},
        openai_client=None,
        executor=SimpleNamespace(
            middleware_manager=None,
            llm_caller=_FakeLLMCaller(),
            response_parser=_FakeResponseParser(),
            structured_tool_payload=[],
            use_structured_tool_calls=True,
            tool_call_mode="structured",
            shutdown_event=None,
            subagent_manager=None,
        ),
    )

    async def _run_and_cancel() -> None:
        task = asyncio.create_task(
            transport._run_local_single_model_step(
                messages=[Message(role=Role.USER, content=[TextBlock(text="user")])],
                agent=fake_agent,  # type: ignore[arg-type]
                context={"session_id": "sess-1", "round_num": 1},
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run_and_cancel())

    cancelled_records = [
        record for record in caplog.records if record.__dict__.get("event") == "single_model_step_llm_call_cancelled"
    ]
    assert len(cancelled_records) == 1
    assert cancelled_records[0].__dict__["configured_timeout_seconds"] == 30
    assert cancelled_records[0].__dict__["configured_max_retries"] == 1


def test_local_single_model_step_timeout_covers_llm_retry_budget() -> None:
    agent_config = _agent_config()
    agent_config.llm_config.timeout = 120
    agent_config.llm_config.max_retries = 6

    timeout_seconds = _model_step_timeout_seconds(SimpleNamespace(config=agent_config))  # type: ignore[arg-type]

    assert timeout_seconds == 840


def test_local_single_model_step_timeout_uses_current_candidate_retry_budget() -> None:
    agent_config = _agent_config()
    agent_config.llm_config.timeout = 120
    agent_config.llm_config.max_retries = 2
    fallbacks = parse_llm_fallbacks(
        [
            {
                "name": "backup",
                "timeout": 60,
                "max_retries": 3,
            }
        ],
        primary_llm_config=agent_config.llm_config,
    )
    agent_config = attach_llm_fallbacks(agent_config, fallbacks)

    timeout_seconds = _model_step_timeout_seconds(SimpleNamespace(config=agent_config))  # type: ignore[arg-type]

    assert timeout_seconds == 360


def test_local_single_model_step_timeout_does_not_sum_fallback_chain() -> None:
    agent_config = _agent_config()
    agent_config.llm_config.timeout = 300
    agent_config.llm_config.max_retries = 3
    fallbacks = parse_llm_fallbacks(
        [
            {
                "name": "backup",
                "timeout": 300,
                "max_retries": 3,
            }
        ],
        primary_llm_config=agent_config.llm_config,
    )
    agent_config = attach_llm_fallbacks(agent_config, fallbacks)

    timeout_seconds = _model_step_timeout_seconds(SimpleNamespace(config=agent_config))  # type: ignore[arg-type]

    assert timeout_seconds == 1200


def test_desktop_model_runner_tries_backup_after_candidate_timeout(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="northcua.agents.desktop.runtime.model")
    primary = LLMFallbackCandidate(
        name="primary",
        llm_config=LLMConfig(
            model="primary-model",
            base_url="http://localhost:3001/v1",
            api_key="key",
            api_type="openai_responses",
            timeout=300,
            max_retries=3,
        ),
        index=0,
        is_primary=True,
    )
    backup = LLMFallbackCandidate(
        name="backup",
        llm_config=LLMConfig(
            model="backup-model",
            base_url="http://localhost:3001/v1",
            api_key="key",
            api_type="openai_responses",
            timeout=300,
            max_retries=3,
        ),
        index=1,
    )
    attempts: list[str] = []
    backup_result = SimpleNamespace(llm_candidate_name="backup")

    async def _run_candidate(**kwargs):
        candidate = kwargs["candidate"]
        attempts.append(candidate.name)
        if candidate.is_primary:
            raise ModelStepLLMTimeoutError("primary timed out")
        return backup_result

    fake_runner = SimpleNamespace(
        app_config=SimpleNamespace(websocket=SimpleNamespace(runtime_mode=RuntimeMode.REMOTE)),
        agent_preparer=SimpleNamespace(llm_fallback_candidates=lambda: (primary, backup)),
        _run_candidate=_run_candidate,
    )

    result = asyncio.run(
        DesktopModelStepRunner.run(
            fake_runner,  # type: ignore[arg-type]
            task="do work",
            session_id="sess-1",
            round_num=1,
            round_step=1,
            session_step=1,
            step_budget=None,
            observation=SimpleNamespace(),
            history_turns=[],
            prompt_projector=SimpleNamespace(),
            trace_context_metadata={},
            remote_tool_dispatcher=None,
        )
    )

    assert result is backup_result
    assert attempts == ["primary", "backup"]
    failed_records = [
        record for record in caplog.records if record.__dict__.get("event") == "llm_fallback_attempt_failed"
    ]
    assert len(failed_records) == 1
    assert failed_records[0].__dict__["candidate_name"] == "primary"
    assert failed_records[0].__dict__["is_final"] is False


def test_desktop_model_runner_logs_final_candidate_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="northcua.agents.desktop.runtime.model")
    primary = LLMFallbackCandidate(
        name="primary",
        llm_config=LLMConfig(
            model="primary-model",
            base_url="http://localhost:3001/v1",
            api_key="key",
            api_type="openai_responses",
            timeout=300,
            max_retries=3,
        ),
        index=0,
        is_primary=True,
    )
    backup = LLMFallbackCandidate(
        name="backup",
        llm_config=LLMConfig(
            model="backup-model",
            base_url="http://localhost:3001/v1",
            api_key="key",
            api_type="openai_responses",
            timeout=300,
            max_retries=3,
        ),
        index=1,
    )
    attempts: list[str] = []

    async def _run_candidate(**kwargs):
        candidate = kwargs["candidate"]
        attempts.append(candidate.name)
        raise ModelStepLLMTimeoutError(f"{candidate.name} timed out")

    fake_runner = SimpleNamespace(
        app_config=SimpleNamespace(websocket=SimpleNamespace(runtime_mode=RuntimeMode.REMOTE)),
        agent_preparer=SimpleNamespace(llm_fallback_candidates=lambda: (primary, backup)),
        _run_candidate=_run_candidate,
    )

    with pytest.raises(ModelStepLLMTimeoutError, match="backup timed out"):
        asyncio.run(
            DesktopModelStepRunner.run(
                fake_runner,  # type: ignore[arg-type]
                task="do work",
                session_id="sess-1",
                round_num=1,
                round_step=1,
                session_step=1,
                step_budget=None,
                observation=SimpleNamespace(),
                history_turns=[],
                prompt_projector=SimpleNamespace(),
                trace_context_metadata={},
                remote_tool_dispatcher=None,
            )
        )

    assert attempts == ["primary", "backup"]
    failed_records = [
        record for record in caplog.records if record.__dict__.get("event") == "llm_fallback_attempt_failed"
    ]
    assert [record.__dict__["candidate_name"] for record in failed_records] == ["primary", "backup"]
    assert [record.__dict__["is_final"] for record in failed_records] == [False, True]
    assert failed_records[-1].__dict__["error_type"] == "ModelStepLLMTimeoutError"
