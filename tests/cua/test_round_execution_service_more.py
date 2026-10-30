"""Regression coverage for desktop round execution service persistence and fallback."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
from nexau.archs.tracer.core import SpanType
from nexau.core.messages import Role

from northcua.agentinfra.providers.llm_fallback import attach_llm_fallbacks, parse_llm_fallbacks
from northcua.agents.desktop.facade import NorthCUAAgentFacade
from northcua.agents.desktop.round_execution import RoundExecutionService
from northcua.agents.desktop.session import NorthCUARoundCoordinator
from northcua.core.config import ArtifactConfig, NorthCUAAppConfig
from northcua.core.enums import ResultStatus, TaskStatus
from northcua.core.models import (
    CommandExecutionResultRecord,
    DesktopStateSnapshot,
    LLMTraceRecord,
    RoundInput,
    StepLogRecord,
    WindowState,
)
from northcua.runtime.artifacts.store import NorthCUAArtifactStore


class _DummySessionManager:
    def __init__(self) -> None:
        self.setup_called = False
        self.state_updates: list[dict[str, object]] = []
        self.session = None

    async def setup_models(self) -> None:
        self.setup_called = True

    async def get_session(self, *, user_id: str, session_id: str) -> object | None:
        return self.session

    async def update_session_state(
        self,
        *,
        user_id: str,
        session_id: str,
        context: dict[str, object],
        storage: object,
    ) -> None:
        self.session = SimpleNamespace(context=context, storage=storage)
        self.state_updates.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "context": context,
                "storage": storage,
            }
        )


class _VerificationExecutor:
    def __init__(self, image_url: str = "data:image/png;base64,UVVJ") -> None:
        self.image_url = image_url
        self.payloads = []

    async def execute_command_request(self, payload):
        self.payloads.append(payload)
        return SimpleNamespace(
            results=[
                CommandExecutionResultRecord(
                    call_id=payload.commands[0].call_id,
                    status=ResultStatus.SUCCESS,
                    result=self.image_url,
                )
            ]
        )


def _write_agent_yaml(path: Path) -> None:
    path.write_text(
        """
name: NorthCUAAgent
description: Desktop agent
system_prompt: You are NorthCUA.
llm_config:
  model: gpt-5.4
  base_url: http://localhost:3001/v1
  api_key: key
  api_type: openai_responses
""".strip(),
        encoding="utf-8",
    )


def _default_internal_mcp_servers() -> list[dict[str, object]]:
    return [
        {
            "name": "DesktopNativeDataExecutor",
            "type": "stdio",
            "command": "python3",
            "model_visible": False,
        },
        {
            "name": "DesktopNativeA11yExecutor",
            "type": "stdio",
            "command": "python3",
            "model_visible": False,
        },
    ]


def test_round_execution_service_falls_back_once_to_second_llm_candidate_on_429(
    tmp_path, desktop_native_test_session
) -> None:
    import asyncio

    agent_yaml = tmp_path / "agent.yaml"
    _write_agent_yaml(agent_yaml)
    agent_config = AgentConfig.from_yaml(agent_yaml)
    fallbacks = parse_llm_fallbacks(
        [{"name": "backup", "model": "backup-model"}],
        primary_llm_config=agent_config.llm_config,
    )
    agent_config = attach_llm_fallbacks(agent_config, fallbacks)
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.agent_mcp_servers = _default_internal_mcp_servers()
    artifact_store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-fallback",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    rounds = NorthCUARoundCoordinator(session_id="sess-fallback")
    session_manager = _DummySessionManager()
    attempted_models: list[str] = []
    tracer = InMemoryTracer()

    class _DummyTransport:
        async def prepare_agent(
            self,
            *,
            user_id: str,
            session_id: str,
            agent_config: AgentConfig,
            remote: bool = False,
            externalize_tools: bool = False,
            transport_capabilities: object | None = None,
            variables: object | None = None,
        ) -> object:
            return _DummyAgent(agent_config)

        async def run_local_round(
            self,
            *,
            messages: list[object],
            user_id: str,
            session_id: str,
            agent_config: AgentConfig,
            agent: object | None = None,
            context: object | None = None,
            externalize_tools: bool = False,
            transport_capabilities: object | None = None,
        ) -> tuple[object, object]:
            attempted_models.append(agent_config.llm_config.model)
            span = tracer.start_span(
                "OpenAI Responses",
                SpanType.LLM,
                inputs={"model": agent_config.llm_config.model, "image": "data:image/png;base64,AAAA"},
            )
            if agent_config.llm_config.model == "gpt-5.4":
                tracer.end_span(span, outputs={"result": {"stop_reason": "error"}})
                raise RuntimeError("429 rate limit from primary provider")
            agent = agent or _DummyAgent(agent_config)
            await agent.run_async(messages)
            tracer.end_span(span, outputs={"result": {"stop_reason": "stop"}})
            return "Finished with fallback.", agent

    class _DummyAgent:
        def __init__(self, config: AgentConfig) -> None:
            self.config = config
            self.global_storage = {"tracer": tracer}
            self.tool_call_payload = []

        async def run_async(self, message: object, **_: object) -> None:
            self.config.before_model_hooks[0](
                SimpleNamespace(messages=message, agent_state=SimpleNamespace(_tool_registry=None))
            )
            self.config.after_model_hooks[0](
                SimpleNamespace(
                    model_response=SimpleNamespace(
                        content="Finished with fallback.",
                        raw_message={"stop_reason": "stop"},
                        reasoning_content="fallback reasoning",
                        tool_calls=[],
                    ),
                    original_response="",
                )
            )

    service = RoundExecutionService(
        app_config=app_config,
        agent_config=agent_config,
        artifact_store=artifact_store,
        round_coordinator=rounds,
        session_manager=session_manager,  # type: ignore[arg-type]
        execution_transport=_DummyTransport(),  # type: ignore[arg-type]
        user_id="test-user",
    )
    service.local_device_executor = _VerificationExecutor()  # type: ignore[assignment]
    snapshot = DesktopStateSnapshot(
        platform_name="macOS",
        current_window=WindowState(name="Calculator", type="window"),
        clean_screenshot_path="",
        clean_screenshot_url="",
        screenshot_metadata={"width": 1440, "height": 900},
    )

    step_log = asyncio.run(service.run_round(RoundInput(query="Open Calculator"), snapshot=snapshot))

    assert attempted_models == ["gpt-5.4", "backup-model"]
    assert tracer.spans == {}
    assert tracer.root_spans == []
    assert tracer.children == {}
    assert step_log.status == TaskStatus.FINISH
    assert step_log.result == "Finished with fallback."
    assert (
        json.loads(artifact_store.llm_trace_log_path.read_text(encoding="utf-8").splitlines()[-1])["llm_trace"][
            "request"
        ]["model"]
        == "backup-model"
    )
