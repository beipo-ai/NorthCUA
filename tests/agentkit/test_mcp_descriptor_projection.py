"""Regression tests for projecting MCP descriptors into AgentKit specs."""

from __future__ import annotations

import asyncio

from northcua.agentkit.device_adapter import AgentKitMCPRuntimeAdapter
from northcua.agents.desktop.agentkit import desktop_agent_spec
from northcua.agents.web.agentkit import web_agent_spec
from northcua.runtime.mcp.registry import MCPFactoryContext, NorthCUAMCPRegistry


def _descriptor_schema_by_tool_id(server_name: str) -> dict[str, dict[str, object]]:
    return {
        descriptor.tool_id: descriptor.parameters_schema
        for descriptor in NorthCUAMCPRegistry.wire_tool_descriptors(server_name)
    }


def test_builtin_agentkit_specs_project_tool_schemas_from_mcp_registry() -> None:
    """Built-in AgentKit tools should mirror MCP descriptors, not re-declare schemas."""

    desktop_spec = desktop_agent_spec()
    web_spec = web_agent_spec()

    for spec in (desktop_spec, web_spec):
        for tool in spec.tools:
            assert tool.server_name
            descriptors = _descriptor_schema_by_tool_id(tool.server_name)
            assert tool.name in descriptors
            assert tool.parameters_schema == descriptors[tool.name]
            assert tool.contract is not None
            assert tool.parameters_schema == tool.contract.parameters_schema


def test_builtin_agentkit_probes_project_from_mcp_registry() -> None:
    """Probe specs use the same MCP descriptor metadata as executable tools."""

    web_spec = web_agent_spec()

    for probe in web_spec.probes:
        assert probe.server_name
        descriptors = _descriptor_schema_by_tool_id(probe.server_name)
        assert probe.name in descriptors
        assert probe.parameters_schema == descriptors[probe.name]
        assert probe.contract is not None
        assert probe.parameters_schema == probe.contract.parameters_schema


def test_agentkit_device_adapter_preserves_projected_mcp_schema() -> None:
    """Device-advertised AgentKit tools keep the MCP descriptor JSON schema."""

    adapter = AgentKitMCPRuntimeAdapter(agent_kinds={"web"})
    descriptors = {descriptor.tool_id: descriptor for descriptor in adapter.tools}
    web_open_schema = _descriptor_schema_by_tool_id("WebAgentActionExecutor")["web_open"]

    assert descriptors["web.web_open"].parameters_schema == web_open_schema
    assert descriptors["web_open"].parameters_schema == web_open_schema


def test_desktop_input_descriptors_carry_tool_specific_prompt_guidance() -> None:
    """Concrete input-tool semantics belong in descriptors rather than system prompt."""

    schemas = _descriptor_schema_by_tool_id("DesktopNativeActionExecutor")
    type_text = str(schemas["type"]["properties"]["text"]["description"])
    target_kind = schemas["type"]["properties"]["target_kind"]
    keypress_keys = str(schemas["keypress"]["properties"]["keys"]["description"])

    assert "intended field is focused" in type_text
    assert "large multi-line code" in type_text
    assert "Unsafe Paste confirmation prompts" in type_text
    assert "opening a file, writing the content there" in type_text
    assert target_kind["enum"] == ["auto", "editable", "terminal"]
    assert "terminal paste semantics" in str(target_kind["description"])
    assert schemas["keypress"]["oneOf"] == [
        {
            "properties": {
                "key": {"type": "string"},
                "keys": {"type": "null"},
            },
            "required": ["key"],
        },
        {
            "properties": {
                "key": {"type": "null"},
                "keys": {"items": {"type": "string"}, "type": "array"},
            },
            "required": ["keys"],
        },
    ]
    assert "Use this only when `keys` is omitted" in str(schemas["keypress"]["properties"]["key"]["description"])
    assert '["ctrl+l", "ctrl+l", "tab"]' in keypress_keys
    assert "`cmd` on macOS" in keypress_keys


def test_registered_web_mcp_tool_names_match_descriptor_source() -> None:
    """MCP implementations and descriptor metadata must stay in lockstep."""

    web_agent_spec()

    async def _run() -> None:
        for server_name in (
            "WebAgentActionExecutor",
            "WebAgentBackendExecutor",
            "WebAgentProbeExecutor",
        ):
            server = NorthCUAMCPRegistry.build(server_name, context=MCPFactoryContext(process_name="descriptor-test"))
            actual = {str(tool.name) for tool in await server.list_tools()}
            described = {descriptor.tool_id for descriptor in NorthCUAMCPRegistry.wire_tool_descriptors(server_name)}
            assert actual == described

    asyncio.run(_run())
