"""Regression coverage for desktop round execution external tool resume persistence."""

from __future__ import annotations

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agents.desktop.round_execution.tool_dispatch import (
    LocalExternalToolDispatcher,
    RemoteExternalCommandExecutor,
)
from northcua.agents.desktop.runtime.model_execution import merge_local_tool_executions
from northcua.core.models import ToolExecutionRecord
from northcua.core.protocol import CommandRequestPayload, CommandResultPayload
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult
from tests.cua.test_round_execution_service import (
    AgentConfig,
    ArtifactConfig,
    CommandExecutionResultRecord,
    DesktopStateSnapshot,
    NorthCUAAppConfig,
    NorthCUAArtifactStore,
    NorthCUARoundCoordinator,
    ResultStatus,
    RoundExecutionService,
    RoundInput,
    SimpleNamespace,
    TaskStatus,
    WindowState,
    _default_internal_mcp_servers,
    _DummySessionManager,
    _write_agent_yaml,
)


def test_round_execution_service_local_external_step_persists_resume_results(
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
        session_id="sess-2",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    rounds = NorthCUARoundCoordinator(session_id="sess-2")
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
                }
            ]

        async def run_async(self, message: object, **_: object) -> None:
            self.config.before_model_hooks[0](
                SimpleNamespace(messages=message, agent_state=SimpleNamespace(_tool_registry=None))
            )
            self.config.after_model_hooks[0](
                SimpleNamespace(
                    model_response=SimpleNamespace(
                        content="Need to click.",
                        raw_message={"output": []},
                        reasoning_content="reasoning",
                        tool_calls=[
                            SimpleNamespace(
                                call_id="call-1",
                                name="click",
                                raw_arguments='{"x":1,"y":2}',
                                arguments={"x": 1, "y": 2},
                                call_type="function",
                            )
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

    step_log = asyncio.run(service.run_round(RoundInput(query="Open Calculator"), snapshot=snapshot))

    assert step_log.status == TaskStatus.CONTINUE
    assert step_log.assistant_tool_calls[0].name == "click"
    assert len(step_log.tool_results) == 1
    assert step_log.tool_results[0].content == "Clicked"
    assert rounds.state.pending_resume_results[0].call_id == "call-1"


def test_round_execution_merge_replaces_empty_hook_result_with_local_failure() -> None:
    trace_state = TraceCaptureState(
        tool_executions=[
            ToolExecutionRecord(
                tool_call_id="call-1",
                tool_name="click",
                tool_input={"x": "862, 345", "y": 345},
                tool_output=None,
                llm_tool_output=None,
                is_error=False,
            )
        ]
    )
    local_execution = ToolExecutionRecord(
        tool_call_id="call-1",
        tool_name="click",
        tool_input={"x": "862, 345", "y": 345},
        tool_output={
            "status": "error",
            "error": "Input should be a valid number",
            "error_type": "ValidationError",
        },
        llm_tool_output={
            "status": "error",
            "error": "Input should be a valid number",
            "error_type": "ValidationError",
        },
        is_error=True,
    )

    merge_local_tool_executions(trace_state, [local_execution])

    assert trace_state.tool_executions == [local_execution]


def test_local_external_tool_dispatcher_exposes_failure_error_to_model() -> None:
    output = LocalExternalToolDispatcher._execution_output(
        CommandExecutionResultRecord(
            call_id="call-1",
            status=ResultStatus.FAILURE,
            error="Input should be a valid number",
        )
    )

    assert output == {
        "status": "error",
        "error": "Input should be a valid number",
    }


def test_local_external_tool_dispatcher_preserves_pending_call_server_route() -> None:
    import asyncio

    class _RecordingExecutor:
        def __init__(self) -> None:
            self.payload: CommandRequestPayload | None = None

        async def execute_command_request(self, payload: CommandRequestPayload) -> CommandResultPayload:
            self.payload = payload
            return CommandResultPayload(
                session_id=payload.session_id,
                status=ResultStatus.SUCCESS,
                results=[
                    CommandExecutionResultRecord(
                        call_id=payload.commands[0].call_id,
                        status=ResultStatus.SUCCESS,
                        result={"text": "external exec ok"},
                    )
                ],
            )

    executor = _RecordingExecutor()
    dispatcher = LocalExternalToolDispatcher(device_executor=executor)

    results, executions = asyncio.run(
        dispatcher.execute(
            session_id="sess-route",
            round_num=1,
            round_step=0,
            pending_tool_calls=[
                {
                    "id": "call-exec",
                    "name": "exec",
                    "server_name": "UnmanagedServer",
                    "input": {"query": "pwd"},
                }
            ],
            agent=SimpleNamespace(executor=SimpleNamespace(middleware_manager=None)),
            agent_state=None,
        )
    )

    assert executor.payload is not None
    assert executor.payload.commands[0].tool_name == "exec"
    assert executor.payload.commands[0].server_name == "UnmanagedServer"
    assert executor.payload.commands[0].parameters == {"query": "pwd"}
    assert results[0].call_id == "call-exec"
    assert executions[0].tool_name == "exec"


def test_local_external_tool_dispatcher_normalizes_qwen_none_keypress_branch() -> None:
    import asyncio

    class _RecordingExecutor:
        def __init__(self) -> None:
            self.payload: CommandRequestPayload | None = None

        async def execute_command_request(self, payload: CommandRequestPayload) -> CommandResultPayload:
            self.payload = payload
            return CommandResultPayload(
                session_id=payload.session_id,
                status=ResultStatus.SUCCESS,
                results=[
                    CommandExecutionResultRecord(
                        call_id=payload.commands[0].call_id,
                        status=ResultStatus.SUCCESS,
                        result={"text": "keys pressed"},
                    )
                ],
            )

    executor = _RecordingExecutor()
    dispatcher = LocalExternalToolDispatcher(device_executor=executor)
    asyncio.run(
        dispatcher.execute(
            session_id="sess-keypress",
            round_num=1,
            round_step=1,
            pending_tool_calls=[
                {
                    "id": "call-keypress",
                    "name": "keypress",
                    "server_name": "DesktopNativeActionExecutor",
                    "input": {"key": "None", "keys": ["ctrl+home", "ctrl+c"]},
                }
            ],
            agent=SimpleNamespace(executor=SimpleNamespace(middleware_manager=None)),
            agent_state=None,
        )
    )

    assert executor.payload is not None
    assert executor.payload.commands[0].parameters == {"key": None, "keys": ["ctrl+home", "ctrl+c"]}


def test_remote_external_command_executor_preserves_server_route_for_remote_dispatch() -> None:
    import asyncio

    observed_pending: list[dict[str, object]] = []

    async def _dispatch_external_calls(pending_tool_calls):
        observed_pending.extend(pending_tool_calls)
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id="call-exec",
                    status=ResultStatus.SUCCESS,
                    result={"text": "external exec ok"},
                )
            ]
        )

    executor = RemoteExternalCommandExecutor(_dispatch_external_calls)
    result = asyncio.run(
        executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess-route",
                round_num=1,
                round_step=0,
                commands=[
                    {
                        "call_id": "call-exec",
                        "server_name": "UnmanagedServer",
                        "tool_name": "exec",
                        "parameters": {"query": "pwd"},
                    }
                ],
            )
        )
    )

    assert observed_pending == [
        {
            "id": "call-exec",
            "name": "exec",
            "server_name": "UnmanagedServer",
            "input": {"query": "pwd"},
        }
    ]
    assert result.status == ResultStatus.SUCCESS
    assert result.results[0].call_id == "call-exec"


def test_round_execution_service_all_tool_error_continues_then_retry_succeeds(
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
        session_id="sess-tool-error-retry",
        config=ArtifactConfig(root_dir=str(tmp_path / "logs")),
    )
    rounds = NorthCUARoundCoordinator(session_id="sess-tool-error-retry")
    session_manager = _DummySessionManager()

    class _RetryAgent:
        def __init__(self, config: AgentConfig) -> None:
            self.config = config
            self.global_storage = {}
            self.current_call_id = "call-1"
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
            self.config.after_model_hooks[0](
                SimpleNamespace(
                    model_response=SimpleNamespace(
                        content="I will use the corrected click target.",
                        raw_message={"output": []},
                        reasoning_content="reasoning",
                        tool_calls=[
                            SimpleNamespace(
                                call_id=self.current_call_id,
                                name="click",
                                raw_arguments='{"x":1,"y":2}',
                                arguments={"x": 1, "y": 2},
                                call_type="function",
                            )
                        ],
                    ),
                    original_response="",
                )
            )

    class _RetryTransport:
        def __init__(self) -> None:
            self.round_index = 0

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
            return _RetryAgent(agent_config)

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
            self.round_index += 1
            call_id = f"call-{self.round_index}"
            agent = agent or _RetryAgent(agent_config)
            agent.current_call_id = call_id
            await agent.run_async(messages)
            return (
                "",
                {
                    "stop_reason": "EXTERNAL_TOOL_CALL",
                    "pending_tool_calls": [
                        {"id": call_id, "name": "click", "input": {"x": 1, "y": 2}},
                    ],
                },
            ), agent

    async def _execute_command_request(payload):
        result = (
            CommandExecutionResultRecord(
                call_id=payload.commands[0].call_id,
                status=ResultStatus.FAILURE,
                result={"status": "error", "error": "x=999 is outside the screenshot width."},
            )
            if payload.commands[0].call_id == "call-1"
            else CommandExecutionResultRecord(
                call_id=payload.commands[0].call_id,
                status=ResultStatus.SUCCESS,
                result={"text": "Clicked", "screenshot_path": ""},
            )
        )
        return SimpleNamespace(results=[result])

    service = RoundExecutionService(
        app_config=app_config,
        agent_config=agent_config,
        artifact_store=artifact_store,
        round_coordinator=rounds,
        session_manager=session_manager,  # type: ignore[arg-type]
        execution_transport=_RetryTransport(),  # type: ignore[arg-type]
        user_id="test-user",
    )
    service.local_device_executor = SimpleNamespace(execute_command_request=_execute_command_request)
    snapshot = DesktopStateSnapshot(
        platform_name="macOS",
        current_window=WindowState(name="Calculator", type="window"),
        clean_screenshot_path="",
        clean_screenshot_url="",
    )

    first_step = asyncio.run(service.run_round(RoundInput(query="Click the target"), snapshot=snapshot))
    second_step = asyncio.run(service.run_round(RoundInput(query="Click the target"), snapshot=snapshot))

    assert first_step.status == TaskStatus.CONTINUE
    assert first_step.terminal_decision["reason"] == "needs_more_steps"
    assert first_step.tool_results[0].is_error is True
    assert "outside the screenshot width" in first_step.tool_results[0].content
    assert second_step.status == TaskStatus.CONTINUE
    assert second_step.tool_results[0].is_error is False
    assert [step.tool_results[0].is_error for step in rounds.state.step_logs] == [True, False]
