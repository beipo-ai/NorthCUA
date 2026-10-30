from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastmcp import FastMCP
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.main_sub.execution.parse_structures import ParsedResponse, ToolCall
from nexau.archs.session import SessionManager
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock

from northcua.agentinfra.runtime.protocol import AgentTransportCapabilities
from northcua.core.enums import ResultStatus, WebSocketMessageType
from northcua.core.models import CommandExecutionResultRecord
from northcua.core.protocol import CommandRequestPayload, NorthCUAWebSocketEnvelope
from northcua.runtime.mcp.device_executor import NorthCUADeviceCommandExecutor
from northcua.runtime.mcp.invocation_context import current_mcp_invocation_context
from northcua.runtime.mcp.registry import MCPFactoryContext, NorthCUAMCPRegistry
from northcua.transport.ws.client import NorthCUAWebSocketClient
from northcua.transport.ws.config import NorthCUAWebSocketTransportConfig
from northcua.transport.ws.local_step import LocalStepModelResult
from northcua.transport.ws.remote_tools import (
    RemoteToolProxyFactory,
    use_remote_command_tool_executor,
    use_remote_tool_artifact_sink,
)
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult
from northcua.transport.ws.transport import NorthCUAWebSocketTransport
from tests.support.nexau_config import agent_config_with_runtime_mcp
from northcua_runtime_shared.tool_contracts import McpBinding, ToolContract


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


def test_transport_remote_round_returns_external_tool_pause_without_dispatching() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    model_messages: list[list[Message]] = []

    async def fake_single_model_step(**kwargs):
        model_messages.append(list(kwargs["messages"]))
        return LocalStepModelResult(
            response_text="",
            stop_reason="EXTERNAL_TOOL_CALL",
            pending_tool_calls=[
                {"id": "call-1", "name": "click", "input": {"x": 1, "y": 2}},
            ],
            agent_state=SimpleNamespace(),
        )

    async def fake_dispatch(pending):
        raise AssertionError(f"remote round must not dispatch inside the model step: {pending!r}")
        return [ToolResultBlock(tool_use_id="call-1", content=[TextBlock(text="Clicked")], is_error=False)]

    runtime_agent = SimpleNamespace()
    transport._run_local_single_model_step = fake_single_model_step  # type: ignore[method-assign]
    result, _agent = asyncio.run(
        transport.run_remote_round(
            messages=[Message(role=Role.USER, content=[TextBlock(text="Open Calculator")])],
            user_id="user-1",
            session_id="sess-1",
            agent_config=_agent_config(),
            dispatch_external_calls=fake_dispatch,
            agent=runtime_agent,
        )
    )

    assert len(model_messages) == 1
    assert model_messages[0][0].role == Role.USER
    assert model_messages[0][0].content[0].text == "Open Calculator"
    assert isinstance(result, tuple)
    response_text, meta = result
    assert response_text == ""
    assert meta["stop_reason"] == "EXTERNAL_TOOL_CALL"
    assert meta["pending_tool_calls"] == [{"id": "call-1", "name": "click", "input": {"x": 1, "y": 2}}]
    assert meta["agent_state"] is not None


def test_two_device_clients_execute_mcp_commands_without_cross_talk() -> None:
    """Simulate two sandbox clients receiving and executing independent MCP commands."""

    server_name = "WorkerEMultiClientActionServer"
    contract = ToolContract(
        tool_id="record_action",
        description="Record a fake action for websocket transport tests.",
        json_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        binding=McpBinding(server_name=server_name),
    )

    @NorthCUAMCPRegistry.register_decorator(server_name, tool_contracts=[contract])
    def _create_action_server(context: MCPFactoryContext) -> FastMCP:
        del context
        mcp = FastMCP("Worker E Multi Client Action Server")

        @mcp.tool
        def record_action(message: str) -> dict[str, str]:
            return {
                "message": message,
                "session_id": current_mcp_invocation_context().session_id,
            }

        return mcp

    def _command(session_id: str, call_id: str, message: str) -> NorthCUAWebSocketEnvelope:
        return NorthCUAWebSocketEnvelope.wrap(
            WebSocketMessageType.COMMAND_REQUEST,
            CommandRequestPayload(
                session_id=session_id,
                round_num=1,
                round_step=0,
                commands=[
                    {
                        "call_id": call_id,
                        "server_name": server_name,
                        "tool_name": "record_action",
                        "tool_type": "action",
                        "parameters": {"message": message},
                    }
                ],
            ),
        )

    client_a = NorthCUAWebSocketClient("ws://controller.invalid/a")
    client_b = NorthCUAWebSocketClient("ws://controller.invalid/b")
    connection_a = _ScriptedWebSocketConnection(_command("sandbox-a", "call-a", "from-a"))
    connection_b = _ScriptedWebSocketConnection(_command("sandbox-b", "call-b", "from-b"))
    client_a._connection = connection_a  # type: ignore[assignment]
    client_b._connection = connection_b  # type: ignore[assignment]
    executor_a = NorthCUADeviceCommandExecutor(allowed_server_names={server_name})
    executor_b = NorthCUADeviceCommandExecutor(allowed_server_names={server_name})

    async def _run_clients() -> None:
        await asyncio.gather(
            client_a.run_device_loop(executor_a),
            client_b.run_device_loop(executor_b),
        )

    asyncio.run(_run_clients())

    result_a = NorthCUAWebSocketEnvelope.model_validate_json(connection_a.sent_messages[0]).payload
    result_b = NorthCUAWebSocketEnvelope.model_validate_json(connection_b.sent_messages[0]).payload

    assert result_a.session_id == "sandbox-a"
    assert result_b.session_id == "sandbox-b"
    assert result_a.status == ResultStatus.SUCCESS
    assert result_b.status == ResultStatus.SUCCESS
    assert result_a.results[0].result == {"message": "from-a", "session_id": "sandbox-a"}
    assert result_b.results[0].result == {"message": "from-b", "session_id": "sandbox-b"}


def test_transport_remote_round_uses_shared_single_model_step_once() -> None:
    transport = NorthCUAWebSocketTransport(
        engine=InMemoryDatabaseEngine.get_shared_instance(),
        session_manager=SessionManager(engine=InMemoryDatabaseEngine.get_shared_instance()),
        config=NorthCUAWebSocketTransportConfig(),
        default_agent_config=_agent_config(),
    )

    initial_messages = [Message(role=Role.USER, content=[TextBlock(text="Open Calculator")])]
    step_messages: list[list[Message]] = []
    runtime_agent = SimpleNamespace(history=list(initial_messages))

    async def fake_single_model_step(**kwargs):
        step_messages.append(list(kwargs["messages"]))
        return LocalStepModelResult(response_text="done", stop_reason="stop", pending_tool_calls=[])

    async def fake_dispatch(_pending):
        raise AssertionError("text-only remote step should not dispatch tools")

    transport._run_local_single_model_step = fake_single_model_step  # type: ignore[method-assign]

    result, _agent = asyncio.run(
        transport.run_remote_round(
            messages=initial_messages,
            user_id="user-1",
            session_id="sess-1",
            agent_config=_agent_config(),
            dispatch_external_calls=fake_dispatch,
            agent=runtime_agent,
        )
    )

    assert result == "done"
    assert len(step_messages) == 1
    assert step_messages[0][0].role == Role.USER
    assert step_messages[0][0].content[0].text == "Open Calculator"


def test_remote_tool_proxy_dispatches_through_current_executor() -> None:
    dispatched_calls: list[dict[str, object]] = []
    proxy = RemoteToolProxyFactory().build(tool_name="click", server_name="DesktopNativeActionExecutor")

    async def fake_dispatch(pending):
        dispatched_calls.extend(pending)
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            tool_blocks=[ToolResultBlock(tool_use_id=call_id, content=[TextBlock(text="Clicked")], is_error=False)],
            command_results=[
                CommandExecutionResultRecord(
                    call_id=f"session::lease::{call_id}",
                    status=ResultStatus.SUCCESS,
                    result={"text": "Clicked", "screenshot_path": "/tmp/click.png"},
                )
            ],
        )

    async def _run_proxy() -> dict[str, object]:
        async with use_remote_command_tool_executor(fake_dispatch):
            return await proxy(x=7, y=8)

    result = asyncio.run(_run_proxy())

    assert result == {"text": "Clicked", "screenshot_path": "/tmp/click.png"}
    assert len(dispatched_calls) == 1
    assert dispatched_calls[0]["name"] == "click"
    assert dispatched_calls[0]["server_name"] == "DesktopNativeActionExecutor"
    assert dispatched_calls[0]["input"] == {"x": 7, "y": 8}
    assert str(dispatched_calls[0]["id"]).startswith("remote-proxy-")


def test_remote_runtime_tool_route_metadata_does_not_leak_into_command_input() -> None:
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
            remote=True,
            externalize_tools=False,
        )
    )
    click_tool = next(tool for tool in runtime_config.tools if tool.name == "click")
    dispatched_calls: list[dict[str, object]] = []

    async def fake_dispatch(pending):
        dispatched_calls.extend(pending)
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={"text": "Clicked", "screenshot_path": "/tmp/click.png"},
                )
            ],
        )

    async def _run_tool() -> dict[str, object]:
        async with use_remote_command_tool_executor(fake_dispatch):
            return await click_tool.execute_async(x=7, y=8)

    result = asyncio.run(_run_tool())

    assert result == {"text": "Clicked", "screenshot_path": "/tmp/click.png"}
    assert dispatched_calls[0]["server_name"] == "DesktopNativeActionExecutor"
    assert dispatched_calls[0]["input"] == {"x": 7.0, "y": 8.0, "button": "left"}
    assert "server_name" not in dispatched_calls[0]["input"]


def test_remote_tool_proxy_preserves_contract_validated_parameters_without_compat_normalization() -> None:
    dispatched_calls: list[dict[str, object]] = []
    proxy = RemoteToolProxyFactory().build(tool_name="keypress")

    async def fake_dispatch(pending):
        dispatched_calls.extend(pending)
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            tool_blocks=[ToolResultBlock(tool_use_id=call_id, content=[TextBlock(text="Pressed")], is_error=False)],
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={"text": "Pressed"},
                )
            ],
        )

    async def _run_proxy() -> dict[str, object]:
        async with use_remote_command_tool_executor(fake_dispatch):
            return await proxy(key="End", keys=["ctrl+end"])

    result = asyncio.run(_run_proxy())

    assert result == {"text": "Pressed"}
    assert len(dispatched_calls) == 1
    assert dispatched_calls[0]["name"] == "keypress"
    assert dispatched_calls[0]["input"] == {"key": "End", "keys": ["ctrl+end"]}


def test_remote_tool_proxy_does_not_return_inline_image_urls_to_model_history() -> None:
    proxy = RemoteToolProxyFactory().build(tool_name="capture_desktop_screenshot")

    async def fake_dispatch_with_raw_result(pending):
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            tool_blocks=[
                ToolResultBlock(
                    tool_use_id=call_id,
                    content=[ImageBlock(url="data:image/png;base64,AAAA"), TextBlock(text="captured")],
                    is_error=False,
                )
            ],
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={"text": "captured", "screenshot_url": "data:image/png;base64,AAAA"},
                )
            ],
        )

    async def _run_proxy() -> dict[str, object]:
        async with use_remote_command_tool_executor(fake_dispatch_with_raw_result):
            return await proxy()

    result = asyncio.run(_run_proxy())

    assert "screenshot_url" not in result
    assert result["text"] == "captured"
    assert result["image_omitted"] is True


def test_remote_tool_proxy_materializes_inline_images_when_sink_bound() -> None:
    proxy = RemoteToolProxyFactory().build(tool_name="web_screenshot")
    materialized: list[dict[str, object]] = []

    class _Sink:
        def materialize_image_data_url(
            self,
            *,
            tool_name: str,
            key: str,
            data_url: str,
            metadata: dict[str, object],
        ) -> str:
            materialized.append(
                {
                    "tool_name": tool_name,
                    "key": key,
                    "data_url": data_url,
                    "metadata_keys": sorted(metadata),
                }
            )
            return "/tmp/northcua-remote-tool.png"

    async def fake_dispatch_with_raw_result(pending):
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={"text": "captured", "screenshot_url": "data:image/png;base64,ZmFrZQ=="},
                )
            ],
        )

    async def _run_proxy() -> dict[str, object]:
        with use_remote_tool_artifact_sink(_Sink()):
            async with use_remote_command_tool_executor(fake_dispatch_with_raw_result):
                return await proxy()

    result = asyncio.run(_run_proxy())

    assert result["text"] == "captured"
    assert result["screenshot_path"] == "/tmp/northcua-remote-tool.png"
    assert result["image_materialized"] is True
    assert "screenshot_url" not in result
    assert "image_omitted" not in result
    assert materialized[0]["tool_name"] == "web_screenshot"
    assert materialized[0]["key"] == "screenshot_url"


def test_remote_tool_proxy_materializes_action_overlay_image_url_when_sink_bound() -> None:
    proxy = RemoteToolProxyFactory().build(tool_name="web_click_ref")
    materialized: list[dict[str, object]] = []

    class _Sink:
        def materialize_image_data_url(
            self,
            *,
            tool_name: str,
            key: str,
            data_url: str,
            metadata: dict[str, object],
        ) -> str:
            materialized.append(
                {
                    "tool_name": tool_name,
                    "key": key,
                    "data_url": data_url,
                    "metadata_keys": sorted(metadata),
                }
            )
            return "/tmp/northcua-action-overlay.png"

    async def fake_dispatch_with_action_overlay_result(pending):
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={
                        "text": "clicked",
                        "action_overlay_image_url": "data:image/png;base64,ZmFrZQ==",
                    },
                )
            ],
        )

    async def _run_proxy() -> dict[str, object]:
        with use_remote_tool_artifact_sink(_Sink()):
            async with use_remote_command_tool_executor(fake_dispatch_with_action_overlay_result):
                return await proxy()

    result = asyncio.run(_run_proxy())

    assert result["text"] == "clicked"
    assert result["action_overlay_image_path"] == "/tmp/northcua-action-overlay.png"
    assert result["image_materialized"] is True
    assert "action_overlay_image_url" not in result
    assert "image_omitted" not in result
    assert materialized[0]["tool_name"] == "web_click_ref"
    assert materialized[0]["key"] == "action_overlay_image_url"


def test_remote_tool_proxy_materializes_web_role_image_urls_when_sink_bound() -> None:
    proxy = RemoteToolProxyFactory().build(tool_name="web_click_ref")
    materialized: list[dict[str, object]] = []

    class _Sink:
        def materialize_image_data_url(
            self,
            *,
            tool_name: str,
            key: str,
            data_url: str,
            metadata: dict[str, object],
        ) -> str:
            materialized.append({"tool_name": tool_name, "key": key, "data_url": data_url})
            return f"/tmp/northcua-{key}.png"

    async def fake_dispatch_with_role_image_result(pending):
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={
                        "text": "clicked",
                        "raw_image_url": "data:image/png;base64,Q0xFQU4=",
                        "som_image_url": "data:image/png;base64,U09N",
                    },
                )
            ],
        )

    async def _run_proxy() -> dict[str, object]:
        with use_remote_tool_artifact_sink(_Sink()):
            async with use_remote_command_tool_executor(fake_dispatch_with_role_image_result):
                return await proxy()

    result = asyncio.run(_run_proxy())

    assert result["raw_image_path"] == "/tmp/northcua-raw_image_url.png"
    assert result["som_image_path"] == "/tmp/northcua-som_image_url.png"
    assert result["image_materialized"] is True
    assert "raw_image_url" not in result
    assert "som_image_url" not in result
    assert [entry["key"] for entry in materialized] == ["raw_image_url", "som_image_url"]


def test_remote_tool_proxy_sanitizes_nested_inline_image_urls() -> None:
    proxy = RemoteToolProxyFactory().build(tool_name="web_screenshot")
    materialized: list[dict[str, object]] = []

    class _Sink:
        def materialize_image_data_url(
            self,
            *,
            tool_name: str,
            key: str,
            data_url: str,
            metadata: dict[str, object],
        ) -> str:
            materialized.append(
                {
                    "tool_name": tool_name,
                    "key": key,
                    "data_url": data_url,
                    "metadata_keys": sorted(metadata),
                }
            )
            return f"/tmp/northcua-{key}-{len(materialized)}.png"

    async def fake_dispatch_with_nested_raw_result(pending):
        call_id = str(pending[0]["id"])
        return RemoteToolDispatchResult(
            command_results=[
                CommandExecutionResultRecord(
                    call_id=call_id,
                    status=ResultStatus.SUCCESS,
                    result={
                        "status": "ok",
                        "result": {
                            "nested": {"screenshot_url": "data:IMAGE/PNG;base64,ZmFrZQ=="},
                            "items": ["prefix", "data:Image/Png;base64,TkVTVEVEPQ=="],
                            "message": "prefix data:IMAGE/PNG;base64,VEVYVA== suffix",
                            "svg": "x data:image/svg+xml,<svg></svg> y",
                            "svg_item": "data:image/svg+xml,<svg></svg>",
                            "data:image/png;base64,S0VZ": "key value",
                        },
                    },
                )
            ],
        )

    async def _run_proxy() -> dict[str, object]:
        with use_remote_tool_artifact_sink(_Sink()):
            async with use_remote_command_tool_executor(fake_dispatch_with_nested_raw_result):
                return await proxy()

    result = asyncio.run(_run_proxy())

    assert "data:image" not in str(result).lower()
    assert result["result"]["nested"]["screenshot_path"].startswith("/tmp/northcua-screenshot_url-")
    assert result["result"]["items"][1] == "[image data omitted; materialized]"
    assert result["result"]["message"] == "prefix [image data omitted] suffix"
    assert result["result"]["svg"] == "x [image data omitted] y"
    assert result["result"]["svg_item"] == "[image data omitted]"
    assert result["result"]["[image data omitted]"] == "key value"
    assert result["image_materialized"] is True
    assert [entry["key"] for entry in materialized] == ["screenshot_url", "items"]


def test_remote_tool_proxy_strips_embedded_image_urls_from_fallback_text() -> None:
    proxy = RemoteToolProxyFactory().build(tool_name="capture_desktop_screenshot")

    async def fake_dispatch_without_raw_result(pending):
        call_id = str(pending[0]["id"])
        return [
            ToolResultBlock(
                tool_use_id=call_id,
                content=[
                    TextBlock(text="before data:IMAGE/PNG;base64,QUJD and data:image/svg+xml,<svg></svg> after"),
                    ImageBlock(url="data:image/png;base64,AAAA"),
                ],
                is_error=False,
            )
        ]

    async def _run_proxy() -> dict[str, object]:
        async with use_remote_command_tool_executor(fake_dispatch_without_raw_result):
            return await proxy()

    result = asyncio.run(_run_proxy())

    assert result["text"] == "before [image data omitted] and [image data omitted] after"
    assert result["image_omitted"] is True
    assert "data:image" not in str(result).lower()


def test_remote_tool_proxy_fallback_omits_image_blocks() -> None:
    proxy = RemoteToolProxyFactory().build(tool_name="capture_desktop_screenshot")

    async def fake_dispatch_without_raw_result(pending):
        call_id = str(pending[0]["id"])
        return [
            ToolResultBlock(
                tool_use_id=call_id,
                content=[ImageBlock(url="data:image/png;base64,AAAA"), TextBlock(text="captured")],
                is_error=False,
            )
        ]

    async def _run_proxy() -> dict[str, object]:
        async with use_remote_command_tool_executor(fake_dispatch_without_raw_result):
            return await proxy()

    result = asyncio.run(_run_proxy())

    assert "screenshot_url" not in result
    assert result["text"] == "captured"
    assert result["image_omitted"] is True
