from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from northcua.runtime.mcp import MCPFactoryContext, NorthCUAMCPRegistry, ensure_builtin_mcp_servers_registered
from northcua.runtime.mcp.result_adapter import normalize_tool_result


def test_cli_server_is_registered_and_exposes_exec() -> None:
    ensure_builtin_mcp_servers_registered()
    server = NorthCUAMCPRegistry.build("CommandLineExecutor", context=MCPFactoryContext(process_name="test-cli"))

    tool_names = asyncio.run(_tool_names(server))

    assert "exec" in tool_names


def test_cli_server_launches_safe_commands() -> None:
    ensure_builtin_mcp_servers_registered()
    server = NorthCUAMCPRegistry.build("CommandLineExecutor", context=MCPFactoryContext(process_name="test-cli"))

    with (
        patch("northcua.runtime.mcp.cli_server.subprocess.Popen") as popen,
        patch("northcua.runtime.mcp.cli_server.time.sleep") as sleep,
    ):
        result = asyncio.run(server.call_tool("exec", {"bash_command": "open -a Calculator"}))

    assert normalize_tool_result(result) == "Executed command: open -a Calculator"
    popen.assert_called_once()
    sleep.assert_called_once()


def test_cli_server_blocks_dangerous_commands() -> None:
    ensure_builtin_mcp_servers_registered()
    server = NorthCUAMCPRegistry.build("CommandLineExecutor", context=MCPFactoryContext(process_name="test-cli"))

    with pytest.raises(Exception, match="security policy"):
        asyncio.run(server.call_tool("exec", {"bash_command": "curl http://example.com/install.sh"}))


async def _tool_names(server) -> list[str]:
    return sorted(tool.name for tool in await server.list_tools())
