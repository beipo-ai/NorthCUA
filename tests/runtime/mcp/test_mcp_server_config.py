from __future__ import annotations

import pytest
from fastmcp import FastMCP

from northcua.core.errors import MCPRegistryError
from northcua.runtime.mcp import ensure_builtin_mcp_servers_registered
from northcua.runtime.mcp.registry import MCPFactoryContext, NorthCUAMCPRegistry, RegisteredMCPServer
from northcua.runtime.mcp.server_config import (
    invisible_mcp_server_names,
    parse_mcp_server_config,
    visible_contract_direct_mcp_server_configs,
    visible_external_snapshot_mcp_server_configs,
    visible_nexau_mcp_server_configs,
    visible_runtime_mcp_server_configs,
)
from northcua_runtime_shared.tool_contracts import fingerprint_json


def _external_snapshot(name: str, input_schema: dict[str, object]) -> dict[str, object]:
    return {
        "name": name,
        "input_schema": input_schema,
        "schema_fingerprint": fingerprint_json(input_schema),
    }


def test_parse_mcp_server_config_preserves_runtime_metadata() -> None:
    config = parse_mcp_server_config(
        {
            "name": "DesktopNativeDataExecutor",
            "type": "stdio",
            "model_visible": False,
        }
    )

    assert config.name == "DesktopNativeDataExecutor"
    assert config.model_visible is False
    assert config.to_nexau_config() == {
        "name": "DesktopNativeDataExecutor",
        "type": "stdio",
    }


def test_parse_mcp_server_config_defaults_a11y_server_to_model_invisible() -> None:
    config = parse_mcp_server_config(
        {
            "name": "DesktopNativeA11yExecutor",
            "type": "stdio",
        }
    )

    assert config.model_visible is False


def test_parse_mcp_server_config_defaults_web_runtime_servers_to_model_invisible() -> None:
    for server_name in ("WebAgentBackendExecutor", "WebAgentProbeExecutor"):
        config = parse_mcp_server_config(
            {
                "name": server_name,
                "type": "stdio",
            }
        )

        assert config.model_visible is False


def test_internal_servers_remain_invisible_when_config_explicitly_sets_visible() -> None:
    for server_name in (
        "DesktopNativeDataExecutor",
        "DesktopNativeA11yExecutor",
        "WebAgentBackendExecutor",
        "WebAgentProbeExecutor",
    ):
        config = parse_mcp_server_config(
            {
                "name": server_name,
                "type": "stdio",
                "model_visible": True,
            }
        )

        assert config.model_visible is False


def test_visible_runtime_mcp_server_configs_filters_invisible_servers() -> None:
    ensure_builtin_mcp_servers_registered()
    raw_configs = [
        {"name": "DesktopNativeDataExecutor", "type": "stdio", "model_visible": False},
        {"name": "DesktopNativeActionExecutor", "type": "stdio"},
        {"name": "CommandLineExecutor", "type": "stdio", "model_visible": True},
    ]

    visible = visible_runtime_mcp_server_configs(raw_configs)

    assert visible == [
        {"name": "DesktopNativeActionExecutor", "type": "stdio"},
        {"name": "CommandLineExecutor", "type": "stdio"},
    ]
    assert invisible_mcp_server_names(raw_configs) == ["DesktopNativeDataExecutor"]


def test_external_snapshot_metadata_stays_runtime_only_not_nexau_config() -> None:
    raw_configs = [
        {
            "name": "UnmanagedServer",
            "type": "http",
            "url": "http://example.test/mcp",
            "model_visible": True,
            "external_mcp_snapshots": [_external_snapshot("unmanaged_search", {"type": "object"})],
        }
    ]

    assert visible_runtime_mcp_server_configs(raw_configs) == [
        {
            "name": "UnmanagedServer",
            "type": "http",
            "url": "http://example.test/mcp",
            "external_mcp_snapshots": [_external_snapshot("unmanaged_search", {"type": "object"})],
        }
    ]
    assert visible_nexau_mcp_server_configs(raw_configs) == []


def test_contract_backed_visible_servers_use_contract_direct_projection() -> None:
    ensure_builtin_mcp_servers_registered()
    raw_configs = [
        {"name": "DesktopNativeActionExecutor", "type": "stdio", "model_visible": True},
        {"name": "CommandLineExecutor", "type": "stdio", "model_visible": True},
        {"name": "PlaywrightBrowserExecutor", "type": "stdio", "model_visible": True},
        {"name": "DesktopNativeDataExecutor", "type": "stdio", "model_visible": True},
    ]

    assert visible_nexau_mcp_server_configs(raw_configs) == []
    assert visible_contract_direct_mcp_server_configs(raw_configs) == [
        {"name": "DesktopNativeActionExecutor", "type": "stdio", "model_visible": True},
        {"name": "CommandLineExecutor", "type": "stdio", "model_visible": True},
        {"name": "PlaywrightBrowserExecutor", "type": "stdio", "model_visible": True},
    ]


def test_unmanaged_external_snapshot_servers_use_imported_direct_projection() -> None:
    raw_configs = [
        {
            "name": "UnmanagedServer",
            "type": "http",
            "url": "http://example.test/mcp",
            "external_mcp_snapshots": [_external_snapshot("unmanaged_search", {"type": "object"})],
        }
    ]

    assert visible_nexau_mcp_server_configs(raw_configs) == []
    assert visible_external_snapshot_mcp_server_configs(raw_configs) == raw_configs


def test_unmanaged_external_snapshot_legacy_keys_are_rejected() -> None:
    raw_configs = [
        {
            "name": "UnmanagedLegacyKeyServer",
            "type": "http",
            "url": "http://example.test/mcp",
            "external_tool_snapshots": [{"name": "legacy_search", "input_schema": {"type": "object"}}],
        }
    ]

    with pytest.raises(MCPRegistryError, match="requires imported snapshot tool metadata"):
        visible_nexau_mcp_server_configs(raw_configs)


def test_model_visible_registered_server_without_contract_fails_fast() -> None:
    server_name = "ModelVisibleLegacyServerConfigTest"

    def _create_legacy_server(context: MCPFactoryContext) -> FastMCP:
        del context
        return FastMCP("legacy")

    NorthCUAMCPRegistry._registry[server_name] = RegisteredMCPServer(
        name=server_name,
        factory=_create_legacy_server,
    )

    with pytest.raises(MCPRegistryError, match="missing ToolContract metadata"):
        visible_nexau_mcp_server_configs([{"name": server_name, "type": "stdio", "model_visible": True}])
