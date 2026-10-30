"""Regression coverage for desktop resume prompt rebuild and terminal persistence."""

from __future__ import annotations

import json

from tests.cua.test_round_execution_service_resume import (
    AgentConfig,
    ArtifactConfig,
    CommandExecutionResultRecord,
    DesktopStateSnapshot,
    LLMTraceRecord,
    NorthCUAAgentFacade,
    NorthCUAAppConfig,
    NorthCUAArtifactStore,
    NorthCUARoundCoordinator,
    ResultStatus,
    Role,
    RoundExecutionService,
    RoundInput,
    SimpleNamespace,
    StepLogRecord,
    TaskStatus,
    WindowState,
    _default_internal_mcp_servers,
    _DummySessionManager,
    _write_agent_yaml,
)


def test_round_execution_service_resume_round_rebuilds_prompt_envelope(tmp_path, desktop_native_test_session) -> None:
    import asyncio

    agent_yaml = tmp_path / "agent.yaml"
    _write_agent_yaml(agent_yaml)
    agent_config = AgentConfig.from_yaml(agent_yaml)
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.agent_mcp_servers = _default_internal_mcp_servers()
    artifact_store = NorthCUAArtifactStore.for_session(
        root_dir=tmp_path / "logs",
        session_id="sess-resume",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    rounds = NorthCUARoundCoordinator(session_id="sess-resume")
    session_manager = _DummySessionManager()

    class _DummyTransport:
        def __init__(self) -> None:
            self.run_calls = 0
            self.captured_second_messages = None

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
            return _DummyAgent(agent_config, self)

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
            self.run_calls += 1
            if self.run_calls == 2:
                self.captured_second_messages = messages
            agent = agent or _DummyAgent(agent_config, self)
            await agent.run_async(messages)
            if self.run_calls == 1:
                return (
                    "",
                    {
                        "stop_reason": "EXTERNAL_TOOL_CALL",
                        "pending_tool_calls": [
                            {"id": "call-1", "name": "click", "input": {"x": 1, "y": 2}},
                        ],
                    },
                ), agent
            return "Focused and ready.", agent

    class _DummyAgent:
        def __init__(self, config: AgentConfig, transport: _DummyTransport) -> None:
            self.config = config
            self.transport = transport
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
                }
            ]

        async def run_async(self, message: object, **_: object) -> None:
            self.config.before_model_hooks[0](
                SimpleNamespace(messages=message, agent_state=SimpleNamespace(_tool_registry=None))
            )
            if self.transport.run_calls == 1:
                model_response = SimpleNamespace(
                    content="Need to focus Calculator.",
                    raw_message={"output": []},
                    reasoning_content="I should focus the window before typing.",
                    tool_calls=[
                        SimpleNamespace(
                            call_id="call-1",
                            name="click",
                            raw_arguments='{"x":1,"y":2}',
                            arguments={"x": 1, "y": 2},
                            call_type="function",
                        )
                    ],
                )
            else:
                model_response = SimpleNamespace(
                    content="Focused and ready.",
                    raw_message={"stop_reason": "stop"},
                    reasoning_content="The window is focused now.",
                    tool_calls=[],
                )
            self.config.after_model_hooks[0](
                SimpleNamespace(
                    model_response=model_response,
                    original_response="",
                )
            )

    transport = _DummyTransport()
    service = RoundExecutionService(
        app_config=app_config,
        agent_config=agent_config,
        artifact_store=artifact_store,
        round_coordinator=rounds,
        session_manager=session_manager,  # type: ignore[arg-type]
        execution_transport=transport,  # type: ignore[arg-type]
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
                    )
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

    first = asyncio.run(service.run_round(RoundInput(query="Open Calculator"), snapshot=snapshot))
    second = asyncio.run(service.run_round(RoundInput(query="Open Calculator"), snapshot=snapshot))

    assert first.status == TaskStatus.CONTINUE
    assert second.status == TaskStatus.FINISH
    assert transport.captured_second_messages is not None
    assert transport.captured_second_messages[0].role == Role.SYSTEM
    assert any(message.role == Role.TOOL for message in transport.captured_second_messages)


def test_desktop_facade_persist_terminal_override_updates_session_and_artifacts(
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
        session_id="sess-override",
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
    step_log = StepLogRecord(
        session_id="sess-override",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.FAIL,
        request="Open Calculator",
        result="Task exceeded the local step budget before reaching a terminal state.",
        llm_trace=LLMTraceRecord(),
        current_window=WindowState(name="Calculator", type="window"),
    )
    facade.rounds.state.step_logs = [step_log]

    asyncio.run(facade.persist_terminal_override(request="Open Calculator", step_log=step_log))

    persisted_state = session_manager.state_updates[-1]["context"]
    assert persisted_state["step_logs"][0]["status"] == "fail"
    step_lines = artifact_store.step_log_path.read_text(encoding="utf-8").strip().splitlines()
    evaluation_lines = artifact_store.evaluation_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(step_lines) == 1
    assert json.loads(step_lines[0])["status"] == "fail"
    assert json.loads(evaluation_lines[0])["status"] == "fail"


def test_desktop_facade_persist_terminal_override_appends_pre_step_terminal_record(
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
        session_id="sess-pre-step-override",
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
    previous_step = StepLogRecord(
        session_id="sess-pre-step-override",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Open Calculator",
        result="",
    )
    terminal_step = StepLogRecord(
        session_id="sess-pre-step-override",
        session_step=2,
        round_step=2,
        round_num=1,
        status=TaskStatus.FAIL,
        request="Open Calculator",
        result="Task exceeded the session step budget before running another step.",
        terminal_decision={"action": "fail", "reason": "max_steps", "metadata": {"scope": "session"}},
    )
    facade.rounds.state.step_logs = [previous_step]
    facade.rounds.state.session_step = 2
    facade.rounds.state.round_step = 2

    asyncio.run(facade.persist_terminal_override(request="Open Calculator", step_log=terminal_step))

    persisted_state = session_manager.state_updates[-1]["context"]
    step_lines = artifact_store.step_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(persisted_state["step_logs"]) == 2
    assert persisted_state["session_step"] == 3
    assert len(step_lines) == 2
    assert json.loads(step_lines[-1])["terminal_decision"]["reason"] == "max_steps"
