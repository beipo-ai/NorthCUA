from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from nexau.archs.tool import Tool

from northcua.agentkernel.tools import contract_to_nexau_tool
from northcua.agentkernel.tools.surface import ToolSurfaceService
from northcua.core.enums import ResultStatus
from northcua.core.errors import MCPRegistryError
from northcua.core.protocol import CommandRequestPayload
from northcua.runtime.mcp.contracts import register_contract_handlers
from northcua.runtime.mcp.device_executor import NorthCUADeviceCommandExecutor
from northcua.transport.ws.message_seed import SUPPRESS_RAW_AGENT_MCP_SERVERS_KEY
from northcua.transport.ws.tool_surface import (
    pending_tool_calls_from_parsed_response,
    route_by_tool_name_from_runtime_tools,
)
from northcua_protocol.capabilities import ToolRiskLevel
from northcua_runtime_shared.tool_contracts import (
    McpBinding,
    ToolContract,
    attach_tool_route_metadata,
    define_tool,
    fingerprint_json,
    tool_route_server_name,
)
from northcua_runtime_shared.tool_contracts.runtime_metadata import tool_contract_projection_id
from tests.transport.test_websocket_transport import (
    AgentTransportCapabilities,
    FastMCP,
    InMemoryDatabaseEngine,
    MCPFactoryContext,
    Message,
    ModelResponse,
    NorthCUAMCPRegistry,
    NorthCUAWebSocketTransport,
    NorthCUAWebSocketTransportConfig,
    ParsedResponse,
    Role,
    SessionManager,
    TextBlock,
    ToolCall,
    _agent_config,
    current_mcp_invocation_context,
)


def _echo_contract(server_name: str, *, max_payload_bytes: int = 1_000_000) -> ToolContract:
    return ToolContract(
        tool_id="echo_local",
        description="Echo a local message.",
        namespace="test.in_memory",
        json_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        side_effect=False,
        readonly=True,
        max_payload_bytes=max_payload_bytes,
        binding=McpBinding(server_name=server_name),
    )


def _register_echo_contract_tool(mcp: FastMCP, contract: ToolContract) -> None:
    def echo_local(message: str) -> dict[str, str]:
        return {"text": message}

    register_contract_handlers(mcp, [contract], {"echo_local": echo_local})


def _external_snapshot(name: str, input_schema: dict[str, object], **extra: object) -> dict[str, object]:
    return {
        "name": name,
        "input_schema": input_schema,
        "schema_fingerprint": fingerprint_json(input_schema),
        **extra,
    }


class _FakeDeviceServer:
    def __init__(self, result: object, *tool_names: str) -> None:
        self.result = result
        self.tool_names = tool_names
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def list_tools(self):
        return [type("_Tool", (), {"name": tool_name})() for tool_name in self.tool_names]

    async def call_tool(self, tool_name: str, parameters: dict[str, object]):
        self.calls.append((tool_name, parameters))
        return self.result


def test_transport_local_stepwise_path_prepends_runtime_system_prompt() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    captured_before_model_messages: list[Message] = []

    class _FakeMiddlewareManager:
        def run_before_agent(self, hook_input):
            return hook_input.messages

        def run_before_model(self, hook_input):
            captured_before_model_messages.extend(hook_input.messages)
            return hook_input.messages

        def run_after_model(self, hook_input):
            return hook_input.parsed_response, hook_input.messages, False

    captured_messages: list[Message] = []
    captured_tools: list[dict[str, object]] = []

    class _FakeLLMCaller:
        async def call_llm_async(self, messages, *args, **kwargs):
            captured_messages.extend(messages)
            captured_tools.extend(kwargs.get("tools") or [])
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
        global_storage={},
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
    fake_agent.tool_call_payload = [
        {"name": name, "description": name, "input_schema": {"type": "object", "properties": {}}}
        for name in ("click", "screenshot")
    ]
    fake_agent.executor.update_structured_tools = lambda definitions: None
    ToolSurfaceService().activate_for_agent(fake_agent, allowed_tool_names=("click",))

    asyncio.run(
        transport._run_local_single_model_step(
            messages=[Message(role=Role.USER, content=[TextBlock(text="user")])],
            agent=fake_agent,  # type: ignore[arg-type]
            context={"x": 1},
        )
    )

    assert captured_before_model_messages[0].role == Role.SYSTEM
    assert captured_before_model_messages[0].content[0].text == "Rendered system prompt"
    assert captured_before_model_messages[1].role == Role.USER
    assert captured_messages[0].role == Role.SYSTEM
    assert captured_messages[0].content[0].text == "Rendered system prompt"
    assert captured_messages[1].role == Role.USER
    assert [tool["name"] for tool in captured_tools] == ["click"]


def test_transport_local_round_with_prepared_agent_uses_exact_messages() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    class _FakeAgent:
        def __init__(self) -> None:
            self.history = []
            self.calls: list[list[Message]] = []
            self.tool_call_payload = [{"name": "web_hover"}]
            self.executor = SimpleNamespace()
            self.executor.structured_updates: list[list[dict[str, object]]] = []

            def _update_structured_tools(definitions):
                self.executor.structured_updates.append(list(definitions))

            self.executor.update_structured_tools = _update_structured_tools

        async def run_async(self, message, **kwargs):
            self.calls.append(list(message))
            return "done"

    agent = _FakeAgent()
    messages = [
        Message(role=Role.SYSTEM, content=[TextBlock(text="system")]),
        Message(role=Role.USER, content=[TextBlock(text="user")]),
    ]

    result, returned_agent = asyncio.run(
        transport.run_local_round(
            messages=messages,
            user_id="user-1",
            session_id="sess-1",
            agent_config=_agent_config(),
            agent=agent,  # type: ignore[arg-type]
        )
    )

    assert result == "done"
    assert returned_agent is agent
    assert agent.history == messages
    assert agent.calls == [[]]
    assert agent.executor.structured_updates == [[{"name": "web_hover"}]]


def test_transport_preserves_task_scoped_tool_surface_when_seeding_messages() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    class _FakeAgent:
        def __init__(self) -> None:
            self.history = []
            self.calls: list[list[Message]] = []
            self.config = SimpleNamespace(llm_config=SimpleNamespace(api_type="openai_chat_completion"))
            self.tool_call_payload = [
                {"name": name, "description": name, "input_schema": {"type": "object", "properties": {}}}
                for name in ("click", "screenshot")
            ]
            self.executor = SimpleNamespace(structured_updates=[])
            self.executor.update_structured_tools = lambda definitions: self.executor.structured_updates.append(
                list(definitions)
            )

        async def run_async(self, message, **kwargs):
            self.calls.append(list(message))
            return "done"

    agent = _FakeAgent()
    ToolSurfaceService().activate_for_agent(agent, allowed_tool_names=("click",))
    messages = [Message(role=Role.USER, content=[TextBlock(text="click once")])]

    result, returned_agent = asyncio.run(
        transport.run_local_round(
            messages=messages,
            user_id="user-allowlist",
            session_id="session-allowlist",
            agent_config=_agent_config(),
            agent=agent,  # type: ignore[arg-type]
        )
    )

    assert result == "done"
    assert returned_agent is agent
    assert [[tool["name"] for tool in update] for update in agent.executor.structured_updates] == [
        ["click"],
        ["click"],
    ]


def test_pending_tool_calls_reject_tool_outside_active_provider_surface() -> None:
    parsed = ParsedResponse(
        original_response="",
        tool_calls=[ToolCall(tool_name="screenshot", parameters={}, tool_call_id="call-1")],
        model_response=ModelResponse(content="", tool_calls=[]),
    )

    with pytest.raises(ValueError, match="outside the active provider surface"):
        pending_tool_calls_from_parsed_response(parsed, allowed_tool_names={"click"})


def test_transport_local_round_can_externalize_tools() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    observed: dict[str, object] = {}

    async def fake_build_runtime_agent_config(
        *, agent_config, remote=False, externalize_tools=False, transport_capabilities=None
    ):
        observed["remote"] = remote
        observed["externalize_tools"] = externalize_tools
        observed["uses_in_memory_mcp"] = getattr(transport_capabilities, "uses_in_memory_mcp", None)
        return agent_config

    transport.build_runtime_agent_config = fake_build_runtime_agent_config  # type: ignore[method-assign]

    class _FallbackAgent:
        def __init__(self) -> None:
            self.history = []
            self.calls: list[list[Message]] = []

        async def run_async(self, message, **kwargs):
            self.calls.append(list(message))
            return "done"

    result, _agent = asyncio.run(
        transport.run_local_round(
            messages=[Message(role=Role.USER, content=[TextBlock(text="user")])],
            user_id="user-1",
            session_id="sess-1",
            agent_config=_agent_config(),
            agent=_FallbackAgent(),  # type: ignore[arg-type]
            externalize_tools=True,
        )
    )

    assert result == "done"
    assert observed["externalize_tools"] is False
    assert observed["remote"] is False


def test_transport_runtime_config_projects_contract_tools_without_in_memory_capability() -> None:
    server_name = "TestInMemoryToolServerNoCapability"
    contract = _echo_contract(server_name)

    @NorthCUAMCPRegistry.register_decorator(server_name, tool_contracts=[contract])
    def _create_test_server_without_capability(context: MCPFactoryContext) -> FastMCP:
        del context
        mcp = FastMCP("Test In Memory Tool Server Without Capability")
        _register_echo_contract_tool(mcp, contract)
        return mcp

    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[{"name": server_name, "type": "in-memory", "model_visible": True}],
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=False,
            externalize_tools=False,
        )
    )

    assert [tool.name for tool in runtime_config.tools] == ["echo_local"]


def test_transport_runtime_config_filters_model_invisible_servers() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=False,
            externalize_tools=False,
        )
    )

    server_names = [server["name"] for server in runtime_config.mcp_servers]
    assert "DesktopNativeDataExecutor" not in server_names
    assert "DesktopNativeActionExecutor" not in server_names
    assert "CommandLineExecutor" not in server_names
    tool_names = {tool.name for tool in runtime_config.tools}
    keypress_schema = next(tool.input_schema for tool in runtime_config.tools if tool.name == "keypress")
    assert "click" in tool_names
    assert "exec" in tool_names
    assert "keypress" in tool_names
    assert "oneOf" not in keypress_schema


@pytest.mark.parametrize("remote", [False, True])
def test_transport_prompt_tool_schema_matches_executor_contract_surface(remote: bool) -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=remote,
            externalize_tools=False,
        )
    )

    keypress_tool = next(tool for tool in runtime_config.tools if tool.name == "keypress")
    keypress_contract = next(
        contract
        for contract in NorthCUAMCPRegistry.tool_contracts("DesktopNativeActionExecutor")
        if contract.tool_id == "keypress"
    )
    keypress_descriptor = next(
        descriptor
        for descriptor in NorthCUAMCPRegistry.wire_tool_descriptors("DesktopNativeActionExecutor")
        if descriptor.tool_id == "keypress"
    )

    assert keypress_tool.input_schema == keypress_contract.provider_parameters_schema
    assert keypress_descriptor.parameters_schema == keypress_contract.parameters_schema
    assert tool_route_server_name(keypress_tool) == "DesktopNativeActionExecutor"
    assert "oneOf" not in keypress_tool.input_schema
    assert "oneOf" in keypress_descriptor.parameters_schema


def test_transport_local_and_remote_direct_tool_surfaces_have_same_hash() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    local_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=False,
            externalize_tools=False,
        )
    )
    remote_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=True,
            externalize_tools=False,
        )
    )

    def _surface(config) -> list[dict[str, object]]:
        return sorted(
            (
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "server_name": tool_route_server_name(tool),
                }
                for tool in config.tools
            ),
            key=lambda item: (str(item["server_name"]), str(item["name"])),
        )

    assert local_config.mcp_servers == []
    assert remote_config.mcp_servers == []
    assert fingerprint_json(_surface(local_config)) == fingerprint_json(_surface(remote_config))


def test_model_tool_call_routes_to_contract_executor_without_route_metadata_leak() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=True,
            externalize_tools=False,
        )
    )
    pending = pending_tool_calls_from_parsed_response(
        ParsedResponse(
            original_response="",
            tool_calls=[
                ToolCall(
                    tool_name="click",
                    parameters={"x": 4, "y": 5},
                    tool_call_id="call-click",
                    source="structured",
                )
            ],
            model_response=ModelResponse(content="", tool_calls=[]),
        ),
        route_by_tool_name=route_by_tool_name_from_runtime_tools(runtime_config.tools),
    )

    assert pending == [
        {
            "id": "call-click",
            "name": "click",
            "input": {"x": 4, "y": 5},
            "server_name": "DesktopNativeActionExecutor",
        }
    ]

    fake_action_server = _FakeDeviceServer({"text": "clicked"}, "click")
    executor = NorthCUADeviceCommandExecutor.__new__(NorthCUADeviceCommandExecutor)
    executor._allowed_server_names = {"DesktopNativeActionExecutor"}
    executor._max_allowed_risk = ToolRiskLevel.HIGH
    executor._servers_by_registry_name = {"DesktopNativeActionExecutor": fake_action_server}
    executor._tool_server_by_name = None
    executor._descriptor_by_server_tool = {}
    executor._contract_by_server_tool = {}
    executor._server_name_by_tool_name = {}
    executor._ambiguous_tool_names = set()
    executor._index_registered_tool_contracts()

    result = asyncio.run(
        executor.execute_command_request(
            CommandRequestPayload(
                session_id="sess-route",
                round_num=1,
                round_step=0,
                commands=[
                    {
                        "call_id": str(pending[0]["id"]),
                        "server_name": str(pending[0]["server_name"]),
                        "tool_name": str(pending[0]["name"]),
                        "tool_type": "action",
                        "parameters": dict(pending[0]["input"]),
                    }
                ],
            )
        )
    )

    assert result.status == ResultStatus.SUCCESS
    assert fake_action_server.calls == [("click", {"x": 4.0, "y": 5.0, "button": "left"})]
    assert "server_name" not in fake_action_server.calls[0][1]


def test_transport_local_runtime_rejects_external_snapshot_without_remote_dispatch() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {
                "name": "UnmanagedServer",
                "type": "http",
                "url": "http://example.test/mcp",
                "external_mcp_snapshots": [
                    _external_snapshot(
                        "unmanaged_search",
                        {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        description="Search imported unmanaged data.",
                    )
                ],
            }
        ],
    )

    with pytest.raises(MCPRegistryError, match="remote/externalized tool dispatch"):
        asyncio.run(
            transport.build_runtime_agent_config(
                agent_config=_agent_config(),
                remote=False,
                externalize_tools=False,
            )
        )


def test_transport_remote_runtime_rejects_external_snapshot_without_explicit_externalization() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {
                "name": "UnmanagedServer",
                "type": "http",
                "url": "http://example.test/mcp",
                "external_mcp_snapshots": [
                    _external_snapshot(
                        "unmanaged_search",
                        {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        description="Search imported unmanaged data.",
                    )
                ],
            }
        ],
    )

    with pytest.raises(MCPRegistryError, match="standard remote/device command path"):
        asyncio.run(
            transport.build_runtime_agent_config(
                agent_config=_agent_config(),
                remote=True,
                externalize_tools=False,
            )
        )


def test_transport_explicit_externalized_snapshot_server_uses_proxy_tool() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {
                "name": "UnmanagedServer",
                "type": "http",
                "url": "http://example.test/mcp",
                "external_mcp_snapshots": [
                    _external_snapshot(
                        "unmanaged_search",
                        {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        description="Search imported unmanaged data.",
                    )
                ],
            }
        ],
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=False,
            externalize_tools=True,
        )
    )

    assert runtime_config.mcp_servers == []
    assert [tool.name for tool in runtime_config.tools] == ["unmanaged_search"]
    assert runtime_config.tools[0].input_schema["required"] == ["query"]
    assert callable(runtime_config.tools[0].implementation)
    assert tool_route_server_name(runtime_config.tools[0]) == "UnmanagedServer"


def test_transport_explicit_externalized_step_preserves_external_snapshot_server_route_on_pending_call() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {
                "name": "UnmanagedServer",
                "type": "http",
                "url": "http://example.test/mcp",
                "external_mcp_snapshots": [
                    _external_snapshot(
                        "exec",
                        {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        description="External command executor with a colliding provider name.",
                    )
                ],
            }
        ],
    )
    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=False,
            externalize_tools=True,
        )
    )

    assert [tool.name for tool in runtime_config.tools] == ["exec"]
    assert runtime_config.tools[0].source_name == "UnmanagedServer"
    assert runtime_config.tools[0].extra_kwargs == {}
    assert tool_route_server_name(runtime_config.tools[0]) == "UnmanagedServer"

    class _FakeMiddlewareManager:
        def run_before_agent(self, hook_input):
            return hook_input.messages

        def run_before_model(self, hook_input):
            return hook_input.messages

        def run_after_model(self, hook_input):
            return hook_input.parsed_response, hook_input.messages, False

    class _FakeLLMCaller:
        async def call_llm_async(self, *args, **kwargs):
            del args, kwargs
            return ModelResponse(content="Use external exec.", tool_calls=[])

    class _FakeResponseParser:
        def parse_response(self, response):
            return ParsedResponse(
                original_response=response.content or "",
                tool_calls=[
                    ToolCall(
                        tool_name="exec",
                        parameters={"query": "pwd"},
                        tool_call_id="call-exec",
                        source="structured",
                    )
                ],
                model_response=response,
            )

    class _FakePromptBuilder:
        def build_system_prompt(self, **kwargs):
            del kwargs
            return [SimpleNamespace(text="Rendered system prompt", cache=False)]

    fake_agent = SimpleNamespace(
        agent_name="NorthCUATestAgent",
        agent_id="agent-test",
        config=runtime_config,
        global_storage={},
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

    result = asyncio.run(
        transport._run_local_single_model_step(
            messages=[Message(role=Role.USER, content=[TextBlock(text="run pwd")])],
            agent=fake_agent,  # type: ignore[arg-type]
            context={},
        )
    )

    assert result.stop_reason == "EXTERNAL_TOOL_CALL"
    assert result.pending_tool_calls == [
        {
            "id": "call-exec",
            "name": "exec",
            "input": {"query": "pwd"},
            "server_name": "UnmanagedServer",
        }
    ]


def test_transport_runtime_config_rejects_inherited_routed_direct_tool_before_model_call() -> None:
    agent_config = _agent_config()
    base_tool = Tool(
        name="exec",
        description="Base direct exec.",
        input_schema={"type": "object", "properties": {}},
        implementation=None,
        source_name="BaseServer",
    )
    attach_tool_route_metadata(base_tool, server_name="BaseServer")
    agent_config.tools = [base_tool]
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {
                "name": "UnmanagedServer",
                "type": "http",
                "url": "http://example.test/mcp",
                "external_mcp_snapshots": [
                    _external_snapshot(
                        "exec",
                        {"type": "object", "properties": {}},
                        description="External exec.",
                    )
                ],
            }
        ],
    )

    with pytest.raises(MCPRegistryError, match="ToolContract-projected tools"):
        asyncio.run(
            transport.build_runtime_agent_config(
                agent_config=agent_config,
                remote=False,
                externalize_tools=True,
            )
        )


def test_transport_runtime_config_rejects_inherited_unrouted_direct_tool_before_model_call() -> None:
    agent_config = _agent_config()
    agent_config.tools = [
        Tool(
            name="exec",
            description="Plain direct exec without MCP route metadata.",
            input_schema={"type": "object", "properties": {}},
            implementation=None,
            source_name="NotRoute",
        )
    ]
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {
                "name": "UnmanagedServer",
                "type": "http",
                "url": "http://example.test/mcp",
                "external_mcp_snapshots": [
                    _external_snapshot(
                        "exec",
                        {"type": "object", "properties": {}},
                        description="External exec.",
                    )
                ],
            }
        ],
    )

    with pytest.raises(MCPRegistryError, match="ToolContract-projected tools"):
        asyncio.run(
            transport.build_runtime_agent_config(
                agent_config=agent_config,
                remote=False,
                externalize_tools=True,
            )
        )


def test_transport_runtime_config_preserves_contract_projected_base_tools() -> None:
    contract = define_tool(
        tool_id="test_preserved_contract_tool",
        public_name="preserved_contract_tool",
        description="A contract-projected tool injected by an internal runtime.",
        namespace="test.transport",
        json_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    projected_tool = contract_to_nexau_tool(contract)
    agent_config = _agent_config()
    agent_config.mcp_servers = []
    agent_config.tools = [projected_tool]
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=agent_config,
            remote=False,
            externalize_tools=False,
        )
    )

    assert len(runtime_config.tools) == 1
    assert runtime_config.tools[0].name == "preserved_contract_tool"
    assert runtime_config.tools[0].input_schema == contract.provider_parameters_schema
    assert tool_contract_projection_id(runtime_config.tools[0]) == contract.contract_id
    assert runtime_config.mcp_servers == []


def test_transport_runtime_config_rejects_stale_contract_projected_base_tool() -> None:
    contract = define_tool(
        tool_id="test_stale_contract_tool",
        public_name="stale_contract_tool",
        description="A contract-projected tool with mutated schema.",
        namespace="test.transport",
        json_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    projected_tool = contract_to_nexau_tool(contract)
    projected_tool.input_schema = {"type": "object", "properties": {}}
    agent_config = _agent_config()
    agent_config.mcp_servers = []
    agent_config.tools = [projected_tool]
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    with pytest.raises(MCPRegistryError, match="stale model-visible direct tools"):
        asyncio.run(
            transport.build_runtime_agent_config(
                agent_config=agent_config,
                remote=False,
                externalize_tools=False,
            )
        )


def test_transport_runtime_config_rejects_owned_direct_tool_external_snapshot_name_collision() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {"name": "DesktopNativeActionExecutor", "type": "stdio", "model_visible": True},
            {
                "name": "UnmanagedServer",
                "type": "http",
                "url": "http://example.test/mcp",
                "external_mcp_snapshots": [
                    _external_snapshot(
                        "click",
                        {"type": "object", "properties": {}},
                        description="External click with colliding provider name.",
                    )
                ],
            },
        ],
    )

    with pytest.raises(MCPRegistryError, match="collides"):
        asyncio.run(
            transport.build_runtime_agent_config(
                agent_config=_agent_config(),
                remote=False,
                externalize_tools=True,
            )
        )


def test_direct_tool_routes_require_explicit_server_metadata_not_source_name() -> None:
    routes = route_by_tool_name_from_runtime_tools(
        [
            Tool(
                name="exec",
                description="User-owned direct tool.",
                input_schema={"type": "object", "properties": {}},
                implementation=None,
                source_name="NotAMcpRoute",
            )
        ]
    )

    assert routes == {}


def test_direct_tool_routes_reject_duplicate_unrouted_provider_names() -> None:
    with pytest.raises(ValueError, match="Duplicate direct-tool provider name"):
        route_by_tool_name_from_runtime_tools(
            [
                Tool(
                    name="plain_exec",
                    description="First direct tool.",
                    input_schema={"type": "object", "properties": {}},
                    implementation=None,
                ),
                Tool(
                    name="plain_exec",
                    description="Second direct tool.",
                    input_schema={"type": "object", "properties": {}},
                    implementation=None,
                ),
            ]
        )


def test_direct_tool_routes_reject_duplicate_routed_provider_names_on_same_server() -> None:
    first = Tool(
        name="routed_exec",
        description="First routed tool.",
        input_schema={"type": "object", "properties": {}},
        implementation=None,
    )
    second = Tool(
        name="routed_exec",
        description="Second routed tool.",
        input_schema={"type": "object", "properties": {}},
        implementation=None,
    )
    attach_tool_route_metadata(first, server_name="RouteServer")
    attach_tool_route_metadata(second, server_name="RouteServer")

    with pytest.raises(ValueError, match="Duplicate direct-tool provider name"):
        route_by_tool_name_from_runtime_tools([first, second])


def test_transport_runtime_config_can_suppress_raw_app_mcp_servers() -> None:
    agent_config = _agent_config()
    agent_config.mcp_servers = []
    agent_config.global_storage = {SUPPRESS_RAW_AGENT_MCP_SERVERS_KEY: True}
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {"name": "DesktopNativeActionExecutor", "type": "stdio", "model_visible": True},
            {"name": "CommandLineExecutor", "type": "stdio", "model_visible": True},
        ],
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=agent_config,
            remote=False,
            externalize_tools=False,
        )
    )

    assert runtime_config.mcp_servers == []
    assert runtime_config.tools == []


def test_transport_runtime_config_externalizes_in_memory_servers_as_tools() -> None:
    server_name = "TestInMemoryToolServer"
    contract = _echo_contract(server_name)

    @NorthCUAMCPRegistry.register_decorator(server_name, tool_contracts=[contract])
    def _create_test_server(context: MCPFactoryContext) -> FastMCP:
        del context
        mcp = FastMCP("Test In Memory Tool Server")
        _register_echo_contract_tool(mcp, contract)
        return mcp

    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {"name": server_name, "type": "in-memory", "model_visible": True},
            {"name": "DesktopNativeDataExecutor", "type": "stdio", "model_visible": False},
        ],
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=False,
            externalize_tools=False,
            transport_capabilities=AgentTransportCapabilities(uses_in_memory_mcp=True),
        )
    )

    assert runtime_config.mcp_servers == []
    assert [tool.name for tool in runtime_config.tools] == ["echo_local"]


def test_transport_in_memory_contract_server_uses_contract_provider_schema_without_building_server() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {"name": "DesktopNativeActionExecutor", "type": "in-memory", "model_visible": True},
        ],
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=False,
            externalize_tools=False,
            transport_capabilities=AgentTransportCapabilities(uses_in_memory_mcp=True),
        )
    )

    tool_names = {tool.name for tool in runtime_config.tools}
    keypress_schema = next(tool.input_schema for tool in runtime_config.tools if tool.name == "keypress")

    assert "click" in tool_names
    assert "keypress" in tool_names
    assert "oneOf" not in keypress_schema
    assert runtime_config.mcp_servers == []


def test_transport_contract_direct_tool_validates_json_schema_before_runtime_call() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[
            {"name": "DesktopNativeActionExecutor", "type": "stdio", "model_visible": True},
        ],
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=False,
            externalize_tools=False,
        )
    )
    get_app_state = next(tool for tool in runtime_config.tools if tool.name == "keypress")

    with pytest.raises(ValueError, match="exactly one of"):
        asyncio.run(get_app_state.implementation(key="A", keys=["B"]))


def test_transport_in_memory_contract_tool_uses_dispatcher_payload_gate() -> None:
    server_name = "TestInMemoryToolServerPayloadGate"
    contract = _echo_contract(server_name, max_payload_bytes=24)

    @NorthCUAMCPRegistry.register_decorator(server_name, tool_contracts=[contract])
    def _create_payload_gate_server(context: MCPFactoryContext) -> FastMCP:
        del context
        mcp = FastMCP("Test In Memory Tool Server Payload Gate")
        _register_echo_contract_tool(mcp, contract)
        return mcp

    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[{"name": server_name, "type": "in-memory", "model_visible": True}],
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=False,
            externalize_tools=False,
            transport_capabilities=AgentTransportCapabilities(uses_in_memory_mcp=True),
        )
    )
    echo_tool = next(tool for tool in runtime_config.tools if tool.name == "echo_local")

    with pytest.raises(ValueError, match="payload exceeds"):
        asyncio.run(echo_tool.implementation(message="this payload is deliberately too large"))


def test_transport_in_memory_contract_tool_route_metadata_does_not_enter_runtime_params() -> None:
    server_name = "TestInMemoryToolServerRouteMetadata"
    contract = _echo_contract(server_name)

    @NorthCUAMCPRegistry.register_decorator(server_name, tool_contracts=[contract])
    def _create_route_metadata_server(context: MCPFactoryContext) -> FastMCP:
        del context
        mcp = FastMCP("Test In Memory Tool Server Route Metadata")
        _register_echo_contract_tool(mcp, contract)
        return mcp

    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
        raw_agent_mcp_servers=[{"name": server_name, "type": "in-memory", "model_visible": True}],
    )

    runtime_config = asyncio.run(
        transport.build_runtime_agent_config(
            agent_config=_agent_config(),
            remote=False,
            externalize_tools=False,
            transport_capabilities=AgentTransportCapabilities(uses_in_memory_mcp=True),
        )
    )
    echo_tool = next(tool for tool in runtime_config.tools if tool.name == "echo_local")

    result = asyncio.run(echo_tool.execute_async(message="hello"))

    assert echo_tool.extra_kwargs == {}
    assert tool_route_server_name(echo_tool) == server_name
    assert result == {"text": "hello"}


def test_transport_local_round_installs_invocation_context() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    observed: dict[str, object] = {}

    class _FallbackAgent:
        def __init__(self) -> None:
            self.history = []

        async def run_async(self, message, **kwargs):
            observed["session_id"] = current_mcp_invocation_context().session_id
            return "done"

    result, _agent = asyncio.run(
        transport.run_local_round(
            messages=[Message(role=Role.USER, content=[TextBlock(text="user")])],
            user_id="user-1",
            session_id="sess-invocation",
            agent_config=_agent_config(),
            agent=_FallbackAgent(),  # type: ignore[arg-type]
        )
    )

    assert result == "done"
    assert observed["session_id"] == "sess-invocation"
