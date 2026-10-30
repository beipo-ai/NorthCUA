"""Fixtures for NorthCUA routing metadata outside NexAU's MCP wire schema."""

from __future__ import annotations

from typing import Any

from nexau.archs.main_sub.config import AgentConfig


def agent_config_with_runtime_mcp(payload: dict[str, Any]) -> AgentConfig:
    """Validate agent fields and attach raw metadata for routing-only tests.

    NexAU now validates MCP transports eagerly. NorthCUA's model visibility,
    imported snapshots and registered in-process server names are runtime
    metadata, not NexAU transport definitions. Production retains them in the
    app config and clears the NexAU MCP list before agent creation. These
    fixtures exercise the legacy routing boundary with the same raw metadata;
    they are never used to start NexAU MCP clients.
    """

    raw_servers = payload.get("mcp_servers", [])
    validated = AgentConfig.model_validate({**payload, "mcp_servers": []})
    return validated.model_copy(update={"mcp_servers": raw_servers})
