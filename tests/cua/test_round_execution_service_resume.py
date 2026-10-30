"""Regression coverage for desktop round execution resume and external tool replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from nexau.archs.main_sub.config import AgentConfig
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


def test_round_execution_service_handles_multiple_external_tool_calls_in_one_round(
    tmp_path, desktop_native_test_session
) -> None:
    import asyncio

    agent_yaml = tmp_path / "agent.yaml"
    _write_agent_yaml(agent_yaml)
    agent_config = AgentConfig.from_yaml(agent_yaml)
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.agent_mcp_servers = _default_internal_mcp_servers()
    artifact_store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-3",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    rounds = NorthCUARoundCoordinator(session_id="sess-3")
    session_manager = _DummySessionManager()

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
            assert externalize_tools is False
            agent = agent or _DummyAgent(agent_config)
            await agent.run_async(messages)
            return (
                "",
                {
                    "stop_reason": "EXTERNAL_TOOL_CALL",
                    "pending_tool_calls": [
                        {"id": "call-1", "name": "click", "input": {"x": 1, "y": 2}},
                        {"id": "call-2", "name": "type", "input": {"text": "Calculator"}},
                    ],
                },
            ), agent

    class _DummyAgent:
        def __init__(self, config: AgentConfig) -> None:
            self.config = config
            self.global_storage = {}
            self.tool_call_payload = [
                {
                    "name": "click",
                    "description": "Click something",
                    "input_schema": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    },
                    "kind": "external",
                },
                {
                    "name": "type",
                    "description": "Type text",
                    "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
                    "kind": "external",
                },
            ]

        async def run_async(self, message: object, **_: object) -> None:
            self.config.before_model_hooks[0](
                SimpleNamespace(messages=message, agent_state=SimpleNamespace(_tool_registry=None))
            )
            self.config.after_model_hooks[0](
                SimpleNamespace(
                    model_response=SimpleNamespace(
                        content="I should focus and then type.",
                        raw_message={"output": []},
                        reasoning_content="reasoning",
                        tool_calls=[
                            SimpleNamespace(
                                call_id="call-1",
                                name="click",
                                raw_arguments='{"x":1,"y":2}',
                                arguments={"x": 1, "y": 2},
                                call_type="function",
                            ),
                            SimpleNamespace(
                                call_id="call-2",
                                name="type",
                                raw_arguments='{"text":"Calculator"}',
                                arguments={"text": "Calculator"},
                                call_type="function",
                            ),
                        ],
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
    service.local_device_executor = SimpleNamespace(
        execute_command_request=lambda payload: asyncio.sleep(
            0,
            result=SimpleNamespace(
                results=[
                    CommandExecutionResultRecord(
                        call_id="call-1",
                        status=ResultStatus.SUCCESS,
                        result={"text": "Clicked", "screenshot_path": ""},
                    ),
                    CommandExecutionResultRecord(
                        call_id="call-2",
                        status=ResultStatus.SUCCESS,
                        result={"text": "Typed", "screenshot_path": ""},
                    ),
                ]
            ),
        )
    )

    snapshot = DesktopStateSnapshot(
        platform_name="macOS",
        current_window=WindowState(name="Calculator", type="window"),
        clean_screenshot_path="",
        clean_screenshot_url="",
    )

    step_log = asyncio.run(service.run_round(RoundInput(query="Open Calculator"), snapshot=snapshot))

    assert step_log.status == TaskStatus.CONTINUE
    assert [call.name for call in step_log.assistant_tool_calls] == ["click", "type"]
    assert [result.name for result in step_log.tool_results] == ["click", "type"]
    assert [result.content for result in step_log.tool_results] == ["Clicked", "Typed"]


def test_desktop_facade_captures_step_limit_verification(tmp_path, desktop_native_test_session) -> None:
    import asyncio

    agent_yaml = tmp_path / "agent.yaml"
    _write_agent_yaml(agent_yaml)
    agent_config = AgentConfig.from_yaml(agent_yaml)
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.agent_mcp_servers = _default_internal_mcp_servers()
    artifact_store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-step-limit",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    session_manager = _DummySessionManager()
    facade = NorthCUAAgentFacade(
        app_config=app_config,
        agent_config=agent_config,
        artifact_store=artifact_store,
        session_manager=session_manager,  # type: ignore[arg-type]
        execution_transport=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )
    facade.round_execution_service.local_device_executor = _VerificationExecutor()  # type: ignore[assignment]
    step_log = StepLogRecord(
        session_id="sess-step-limit",
        session_step=2,
        round_step=2,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Open Calculator",
        result="",
        llm_trace=LLMTraceRecord(),
        current_window=WindowState(name="Calculator", type="window"),
    )
    facade.rounds.state.step_logs = [step_log]

    asyncio.run(facade.capture_step_limit_verification(step_log))

    assert step_log.status == TaskStatus.CONTINUE
    assert step_log.clean_screenshot_path.endswith("query_0001/action_step2_verify.png")
    assert Path(step_log.clean_screenshot_path).exists()
    assert step_log.observation_metadata["round_end_verification_status"] == "success"
    assert step_log.observation_metadata["round_end_verification_screenshot_path"].endswith(
        "query_0001/action_step2_verify.png"
    )
    assert step_log.observation_artifacts[-1].metadata["reason"] == "step_limit"

    step_lines = artifact_store.step_log_path.read_text(encoding="utf-8").strip().splitlines()
    persisted_step = json.loads(step_lines[0])
    assert persisted_step["status"] == "continue"
    assert persisted_step["clean_screenshot_path"].endswith("query_0001/action_step2_verify.png")
