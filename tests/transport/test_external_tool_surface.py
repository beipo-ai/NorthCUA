from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig

from northcua.core.errors import MCPRegistryError
from northcua.runtime.mcp.registry import MCPFactoryContext, NorthCUAMCPRegistry, RegisteredMCPServer
from northcua.transport.ws.runtime_direct_tools import compile_runtime_direct_tool_surface, derive_remote_tool_surface
from tests.support.nexau_config import agent_config_with_runtime_mcp
from northcua_runtime_shared.tool_contracts import ToolSurfaceMode, fingerprint_json


def _external_snapshot(name: str, input_schema: dict[str, object], **extra: object) -> dict[str, object]:
    return {
        "name": name,
        "input_schema": input_schema,
        "schema_fingerprint": fingerprint_json(input_schema),
        **extra,
    }


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
                {"name": "PlaywrightBrowserExecutor", "type": "stdio"},
                {
                    "name": "UnmanagedServer",
                    "type": "http",
                    "url": "http://example.test",
                    "external_mcp_snapshots": [
                        _external_snapshot(
                            "unmanaged_search",
                            {"type": "object", "properties": {"query": {"type": "string"}}},
                            description="Search through the unmanaged server.",
                        )
                    ],
                },
            ],
        }
    )


def test_remote_tool_surface_derives_from_registered_mcp_metadata() -> None:
    surface = asyncio.run(derive_remote_tool_surface(_agent_config()))

    tool_names = {tool.name for tool in surface.tools}
    assert "click" in tool_names
    assert "screenshot" in tool_names
    assert "type" in tool_names
    assert "exec" in tool_names
    assert "browser_click" in tool_names
    assert "unmanaged_search" in tool_names
    assert "capture_desktop_screenshot" not in tool_names
    assert "UnmanagedServer" in surface.externalized_server_names


def test_remote_tool_surface_uses_static_registered_descriptors_without_live_discovery() -> None:
    agent_config = agent_config_with_runtime_mcp(
        {
            "name": "NorthCUAStaticDescriptorAgent",
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
            ],
        }
    )

    with patch(
        "northcua.transport.ws.runtime_direct_tools.NorthCUAMCPRegistry.build",
        side_effect=AssertionError("static registered contracts should not build local MCP servers"),
    ):
        surface = asyncio.run(derive_remote_tool_surface(agent_config))

    tool_names = {tool.name for tool in surface.tools}
    assert "click" in tool_names
    keypress_schema = next(tool.input_schema for tool in surface.tools if tool.name == "keypress")
    assert keypress_schema["type"] == "object"
    assert "oneOf" not in keypress_schema
    assert "capture_desktop_screenshot" not in tool_names


def test_remote_tool_surface_forced_invisible_contract_server_does_not_externalize() -> None:
    agent_config = agent_config_with_runtime_mcp(
        {
            "name": "NorthCUARuntimeOnlyContractAgent",
            "description": "test",
            "system_prompt": "You are NorthCUA.",
            "llm_config": LLMConfig(
                model="gpt-5.4",
                base_url="http://localhost:3001/v1",
                api_key="key",
                api_type="openai_responses",
            ),
            "mcp_servers": [
                {"name": "DesktopNativeDataExecutor", "type": "stdio", "model_visible": True},
            ],
        }
    )

    with patch(
        "northcua.transport.ws.runtime_direct_tools.NorthCUAMCPRegistry.build",
        side_effect=AssertionError("runtime-only contracts must not build local MCP servers"),
    ):
        surface = asyncio.run(derive_remote_tool_surface(agent_config))

    assert surface.tools == []
    assert surface.externalized_server_names == []


def test_remote_tool_surface_requires_external_snapshot_for_unregistered_server() -> None:
    agent_config = agent_config_with_runtime_mcp(
        {
            "name": "NorthCUAExternalNoSnapshotAgent",
            "description": "test",
            "system_prompt": "You are NorthCUA.",
            "llm_config": LLMConfig(
                model="gpt-5.4",
                base_url="http://localhost:3001/v1",
                api_key="key",
                api_type="openai_responses",
            ),
            "mcp_servers": [{"name": "UnmanagedServer", "type": "http", "url": "http://example.test"}],
        }
    )

    with pytest.raises(MCPRegistryError, match="requires imported snapshot"):
        asyncio.run(derive_remote_tool_surface(agent_config))


def test_remote_tool_surface_external_snapshot_projects_direct_tool() -> None:
    agent_config = agent_config_with_runtime_mcp(
        {
            "name": "NorthCUAExternalSnapshotAgent",
            "description": "test",
            "system_prompt": "You are NorthCUA.",
            "llm_config": LLMConfig(
                model="gpt-5.4",
                base_url="http://localhost:3001/v1",
                api_key="key",
                api_type="openai_responses",
            ),
            "mcp_servers": [
                {
                    "name": "UnmanagedServer",
                    "type": "http",
                    "url": "http://user:secret@example.test/mcp?token=secret",
                    "external_mcp_snapshots": [
                        _external_snapshot(
                            "unmanaged_search",
                            {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                            description="Search unmanaged data.",
                        )
                    ],
                }
            ],
        }
    )
    factory_calls: list[tuple[str, str]] = []

    def _factory(server_name: str, tool_name: str):
        factory_calls.append((server_name, tool_name))

        async def _implementation(**_kwargs):
            return {"ok": True}

        return _implementation

    surface = asyncio.run(derive_remote_tool_surface(agent_config, implementation_factory=_factory))

    assert [tool.name for tool in surface.tools] == ["unmanaged_search"]
    assert surface.tools[0].input_schema["required"] == ["query"]
    assert surface.tools[0].implementation is not None
    assert surface.externalized_server_names == ["UnmanagedServer"]
    assert factory_calls == [("UnmanagedServer", "unmanaged_search")]


def test_remote_tool_surface_external_snapshot_routes_item_level_server_names() -> None:
    agent_config = agent_config_with_runtime_mcp(
        {
            "name": "NorthCUAExternalSnapshotMixedServerAgent",
            "description": "test",
            "system_prompt": "You are NorthCUA.",
            "llm_config": LLMConfig(
                model="gpt-5.4",
                base_url="http://localhost:3001/v1",
                api_key="key",
                api_type="openai_responses",
            ),
            "mcp_servers": [
                {
                    "name": "TopLevelServer",
                    "type": "http",
                    "url": "http://example.test/mcp",
                    "external_mcp_snapshots": [
                        _external_snapshot("other_server_tool", {"type": "object"}, server_name="ItemLevelServer")
                    ],
                }
            ],
        }
    )
    factory_calls: list[tuple[str, str]] = []

    def _factory(server_name: str, tool_name: str):
        factory_calls.append((server_name, tool_name))

        async def _implementation(**_kwargs):
            return {"ok": True}

        return _implementation

    surface = asyncio.run(derive_remote_tool_surface(agent_config, implementation_factory=_factory))

    assert [tool.name for tool in surface.tools] == ["other_server_tool"]
    assert factory_calls == [("ItemLevelServer", "other_server_tool")]


def test_remote_tool_surface_rejects_external_snapshot_collision_with_owned_direct_tool() -> None:
    agent_config = agent_config_with_runtime_mcp(
        {
            "name": "NorthCUAExternalCollisionAgent",
            "description": "test",
            "system_prompt": "You are NorthCUA.",
            "llm_config": LLMConfig(
                model="gpt-5.4",
                base_url="http://localhost:3001/v1",
                api_key="key",
                api_type="openai_responses",
            ),
            "mcp_servers": [
                {"name": "DesktopNativeActionExecutor", "type": "stdio"},
                {
                    "name": "UnmanagedServer",
                    "type": "http",
                    "url": "http://example.test",
                    "external_mcp_snapshots": [_external_snapshot("click", {"type": "object"})],
                },
            ],
        }
    )

    with pytest.raises(MCPRegistryError, match="collide"):
        asyncio.run(derive_remote_tool_surface(agent_config))


def test_remote_tool_surface_rejects_external_snapshot_same_schema_collision() -> None:
    agent_config = agent_config_with_runtime_mcp(
        {
            "name": "NorthCUAExternalExternalCollisionAgent",
            "description": "test",
            "system_prompt": "You are NorthCUA.",
            "llm_config": LLMConfig(
                model="gpt-5.4",
                base_url="http://localhost:3001/v1",
                api_key="key",
                api_type="openai_responses",
            ),
            "mcp_servers": [
                {
                    "name": "UnmanagedOne",
                    "type": "http",
                    "url": "http://one.example.test",
                    "external_mcp_snapshots": [_external_snapshot("search", {"type": "object"})],
                },
                {
                    "name": "UnmanagedTwo",
                    "type": "http",
                    "url": "http://two.example.test",
                    "external_mcp_snapshots": [_external_snapshot("search", {"type": "object"})],
                },
            ],
        }
    )

    with pytest.raises(MCPRegistryError, match="collides across multiple routed sources"):
        asyncio.run(derive_remote_tool_surface(agent_config))


def test_remote_tool_surface_rejects_external_snapshot_conflicting_schema_collision() -> None:
    agent_config = agent_config_with_runtime_mcp(
        {
            "name": "NorthCUAExternalExternalSchemaCollisionAgent",
            "description": "test",
            "system_prompt": "You are NorthCUA.",
            "llm_config": LLMConfig(
                model="gpt-5.4",
                base_url="http://localhost:3001/v1",
                api_key="key",
                api_type="openai_responses",
            ),
            "mcp_servers": [
                {
                    "name": "UnmanagedOne",
                    "type": "http",
                    "url": "http://one.example.test",
                    "external_mcp_snapshots": [
                        _external_snapshot(
                            "search",
                            {"type": "object", "properties": {"query": {"type": "string"}}},
                        )
                    ],
                },
                {
                    "name": "UnmanagedTwo",
                    "type": "http",
                    "url": "http://two.example.test",
                    "external_mcp_snapshots": [
                        _external_snapshot(
                            "search",
                            {"type": "object", "properties": {"limit": {"type": "integer"}}},
                        )
                    ],
                },
            ],
        }
    )

    with pytest.raises(MCPRegistryError, match="conflicting schema fingerprints"):
        asyncio.run(derive_remote_tool_surface(agent_config))


def test_external_snapshot_proxy_validates_schema_before_invocation() -> None:
    calls: list[dict[str, object]] = []

    def _factory(server_name: str, tool_name: str):
        async def _implementation(**kwargs):
            calls.append({"server_name": server_name, "tool_name": tool_name, "kwargs": kwargs})
            return {"ok": True}

        return _implementation

    surface = compile_runtime_direct_tool_surface(
        [
            {
                "name": "UnmanagedServer",
                "type": "http",
                "url": "http://example.test",
                "external_mcp_snapshots": [
                    _external_snapshot(
                        "search",
                        {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    )
                ],
            }
        ],
        mode=ToolSurfaceMode.REMOTE_EXTERNALIZED,
        implementation_factory=_factory,
    )

    with pytest.raises(ValueError, match="unexpected parameters|required"):
        asyncio.run(surface.tools[0].implementation(unexpected=True))
    assert calls == []


def test_external_snapshot_requires_schema_fingerprint() -> None:
    with pytest.raises(ValueError, match="requires schema_fingerprint"):
        compile_runtime_direct_tool_surface(
            [
                {
                    "name": "UnmanagedServer",
                    "type": "http",
                    "url": "http://example.test",
                    "external_mcp_snapshots": [
                        {"name": "search", "input_schema": {"type": "object"}},
                    ],
                }
            ],
            mode=ToolSurfaceMode.REMOTE_EXTERNALIZED,
        )


def test_external_snapshot_rejects_parameters_alias() -> None:
    with pytest.raises(ValueError, match="must use input_schema"):
        compile_runtime_direct_tool_surface(
            [
                {
                    "name": "UnmanagedServer",
                    "type": "http",
                    "url": "http://example.test",
                    "external_mcp_snapshots": [
                        {
                            "name": "search",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                            },
                            "schema_fingerprint": fingerprint_json(
                                {"type": "object", "properties": {"query": {"type": "string"}}}
                            ),
                        },
                    ],
                }
            ],
            mode=ToolSurfaceMode.REMOTE_EXTERNALIZED,
        )


def test_registered_model_visible_remote_surface_without_contract_fails_fast() -> None:
    server_name = "DiscoveryOnlyRemoteToolExecutor"

    def _create_discovery_only_server(context: MCPFactoryContext):
        raise AssertionError(f"{context.process_name} should not be built when discovery metadata is available")

    NorthCUAMCPRegistry._registry[server_name] = RegisteredMCPServer(
        name=server_name,
        factory=_create_discovery_only_server,
    )

    agent_config = agent_config_with_runtime_mcp(
        {
            "name": "NorthCUADiscoveryOnlyAgent",
            "description": "test",
            "system_prompt": "You are NorthCUA.",
            "llm_config": LLMConfig(
                model="gpt-5.4",
                base_url="http://localhost:3001/v1",
                api_key="key",
                api_type="openai_responses",
            ),
            "mcp_servers": [{"name": server_name, "type": "stdio"}],
        }
    )

    with patch(
        "northcua.transport.ws.runtime_direct_tools.NorthCUAMCPRegistry.build",
        side_effect=AssertionError("descriptor-only registered servers should not be built"),
    ):
        with pytest.raises(MCPRegistryError, match="missing ToolContract metadata"):
            asyncio.run(derive_remote_tool_surface(agent_config))
