"""Regression coverage for desktop round execution service persistence and fallback."""

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


def test_round_execution_service_writes_terminal_logs(monkeypatch, tmp_path, desktop_native_test_session) -> None:
    import asyncio

    agent_yaml = tmp_path / "agent.yaml"
    _write_agent_yaml(agent_yaml)
    agent_config = AgentConfig.from_yaml(agent_yaml)
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.execution.experiment_id = "exp-rounds"
    app_config.agent_mcp_servers = _default_internal_mcp_servers()
    artifact_store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-1",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    rounds = NorthCUARoundCoordinator(session_id="sess-1")
    session_manager = _DummySessionManager()
    trace_contexts: list[dict[str, object]] = []

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
            assert agent_config.max_iterations == app_config.execution.max_agent_iterations_per_round
            assert externalize_tools is False
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
            assert agent_config.max_iterations == app_config.execution.max_agent_iterations_per_round
            assert externalize_tools is False
            trace_contexts.append(context or {})
            agent = agent or _DummyAgent(agent_config)
            await agent.run_async(messages)
            return "Finished cleanly.", agent

    class _DummyAgent:
        def __init__(self, config: AgentConfig) -> None:
            self.config = config
            self.executor = SimpleNamespace(
                before_model_hooks=config.before_model_hooks,
                after_model_hooks=config.after_model_hooks,
                after_tool_hooks=config.after_tool_hooks,
            )
            self.global_storage = {}
            self.tool_call_payload = []

        async def run_async(self, message: object, **_: object) -> None:
            self.config.before_model_hooks[0](
                SimpleNamespace(messages=message, agent_state=SimpleNamespace(_tool_registry=None))
            )
            self.config.after_model_hooks[0](
                SimpleNamespace(
                    model_response=SimpleNamespace(
                        content="Finished cleanly.",
                        raw_message={"stop_reason": "stop"},
                        reasoning_content="Short reasoning",
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
    verification_executor = _VerificationExecutor()
    service.local_device_executor = verification_executor  # type: ignore[assignment]
    snapshot = DesktopStateSnapshot(
        platform_name="macOS",
        current_window=WindowState(name="Calculator", type="window"),
        clean_screenshot_path="",
        clean_screenshot_url="",
        screenshot_metadata={"width": 1440, "height": 900, "left": 0, "top": 0, "all_screens": False},
    )

    step_log = asyncio.run(service.run_round(RoundInput(query="Open Calculator"), snapshot=snapshot))

    assert step_log.status == TaskStatus.FINISH
    assert step_log.result == "Finished cleanly."
    assert step_log.clean_screenshot_path.endswith("query_0001/action_step1_verify.png")
    assert trace_contexts[-1]["experiment_id"] == "exp-rounds"
    assert trace_contexts[-1]["query_index"] == 1
    assert trace_contexts[-1]["round_num"] == 1
    assert trace_contexts[-1]["round_step"] == 1
    assert trace_contexts[-1]["session_step"] == 1
    assert str(trace_contexts[-1]["artifact_dir"]).endswith("query_0001")
    assert Path(step_log.clean_screenshot_path).exists()
    assert step_log.observation_metadata["round_end_verification_status"] == "success"
    assert step_log.observation_artifacts[-1].role == "round_end_verify"
    assert verification_executor.payloads[0].commands[0].tool_name == "capture_desktop_screenshot"
    assert step_log.screen_resolution.width == 1440
    assert step_log.screen_resolution.height == 900
    assert session_manager.setup_called is True
    assert session_manager.state_updates[-1]["context"]["step_logs"][0]["status"] == "finish"

    step_lines = artifact_store.step_log_path.read_text(encoding="utf-8").strip().splitlines()
    evaluation_lines = artifact_store.evaluation_log_path.read_text(encoding="utf-8").strip().splitlines()

    persisted_step = json.loads(step_lines[-1])
    assert persisted_step["result"] == "Finished cleanly."
    assert persisted_step["clean_screenshot_path"].endswith("query_0001/action_step1_verify.png")
    assert persisted_step["observation_metadata"]["round_end_verification_status"] == "success"
    assert persisted_step["llm_trace_path"].endswith("llm_trace.jsonl")
    assert persisted_step["llm_reasoning"] == "Short reasoning"
    assert "llm_trace" not in persisted_step
    assert persisted_step["screen_resolution"] == {
        "width": 1440,
        "height": 900,
        "left": 0,
        "top": 0,
        "all_screens": False,
    }
    llm_trace_lines = artifact_store.llm_trace_log_path.read_text(encoding="utf-8").strip().splitlines()
    llm_trace_payload = json.loads(llm_trace_lines[-1])
    assert llm_trace_payload["source"] == "desktop"
    assert llm_trace_payload["llm_trace"]["request"] != {}
    assert llm_trace_payload["llm_trace"]["response"] != {}
    assert llm_trace_payload["llm_trace"]["reasoning"] == "Short reasoning"
    assert step_log.llm_trace.request == {}
    assert step_log.llm_trace.response == {}
    assert rounds.state.step_logs[-1].llm_trace.request == {}
    assert rounds.state.step_logs[-1].llm_trace.response == {}
    assert json.loads(evaluation_lines[-1])["status"] == "finish"
