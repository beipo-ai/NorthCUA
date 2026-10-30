from __future__ import annotations

from pathlib import Path

import yaml

from northcua.agents.web.mcp.descriptors import (
    web_agent_action_contracts,
    web_agent_backend_contracts,
    web_agent_probe_contracts,
)


def test_controller_example_allowlists_all_web_action_and_probe_tools() -> None:
    config = yaml.safe_load(Path("config/controller.example.yaml").read_text(encoding="utf-8"))
    allowed_tools = set(config["security"]["allowed_tools"])
    expected_web_tools = {
        f"{contract.namespace}.{contract.tool_id}"
        for contract in (
            *web_agent_action_contracts(),
            *web_agent_probe_contracts(),
            *web_agent_backend_contracts(),
        )
    }
    expected_web_tools.add("web.web_close_session")

    assert expected_web_tools.issubset(allowed_tools)
    assert "shell.exec" not in allowed_tools
    assert "shell.exec" not in set(config["security"]["allowed_capabilities"])
