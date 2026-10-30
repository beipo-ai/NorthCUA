from unittest.mock import AsyncMock

import pytest

from northcua.agentkit.device_adapter import AgentKitMCPRuntimeAdapter
from northcua.agents.web.backend import WebActionCommandResult
from northcua.agents.web.session import WebAgentSessionRegistry
from northcua.agents.web.session.agentkit_adapter import WebAgentDeviceSessionAdapter
from northcua_protocol.payloads import CommandRequest
from northcua_protocol.state import CommandState
from types import SimpleNamespace


def command():
    return CommandRequest(
        command_id="close",
        idempotency_key="close",
        session_id="cleanup",
        lease_id="lease",
        device_id="device",
        tool_id="web.__close_session",
        parameters={},
    )


@pytest.mark.asyncio
async def test_device_browser_close_error_is_not_reported_as_success(tmp_path):
    registry = WebAgentSessionRegistry()
    client = SimpleNamespace(
        close=AsyncMock(return_value=WebActionCommandResult(success=False, error="browser close failed"))
    )
    registry.register(SimpleNamespace(session_id="cleanup", client=client))
    adapter = WebAgentDeviceSessionAdapter(agent_kind="web")
    try:
        with pytest.raises(Exception, match="browser close failed"):
            await adapter.close_device_session(command=command(), artifact_root=tmp_path)
        with pytest.raises(LookupError):
            registry.get_registered_session("cleanup")
    finally:
        registry.unregister("cleanup")


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_runtime_discards_closed_session_adapter_even_on_error(tmp_path, fail):
    runtime = AgentKitMCPRuntimeAdapter(agent_kinds={"web"}, artifact_root=tmp_path)
    adapter = SimpleNamespace(close_device_session=AsyncMock(return_value={"status": "ok"}))
    if fail:
        adapter.close_device_session.side_effect = RuntimeError("close failed")
    runtime._session_adapters[("web", "cleanup")] = adapter
    if fail:
        with pytest.raises(RuntimeError):
            await runtime._close_session(command(), "web")
    else:
        result = await runtime._close_session(command(), "web")
        assert result.status == CommandState.SUCCEEDED
    assert not runtime._session_adapters
