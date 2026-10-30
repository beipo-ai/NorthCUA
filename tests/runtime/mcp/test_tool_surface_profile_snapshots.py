from __future__ import annotations

import json
from pathlib import Path

from northcua.agentkit.evaluator.protocol import RoundEvaluationReport
from northcua.agentkit.evaluator.tool_contracts import ARTIFACT_TOOL_REFS, final_report_tool_contract
from northcua.agentkit.evaluator.tools.client_probe import standard_client_gui_tools, standard_client_probe_tools
from northcua.agents.web.mcp.descriptors import web_agent_action_contracts, web_agent_probe_contracts
from northcua.runtime.mcp import ensure_builtin_mcp_servers_registered
from northcua.sandboxes.tool_contracts import SANDBOX_PROVIDER_TOOL_CONTRACTS
from northcua_runtime_shared.desktop_surface import desktop_tool_contracts
from northcua_runtime_shared.tool_contracts import (
    ToolAudience,
    ToolContract,
    ToolProfile,
    ToolSurfaceConfig,
    ToolSurfaceResolver,
    ToolVisibility,
)

EXPECTED_PROFILE_SNAPSHOTS = json.loads(
    (Path(__file__).parents[2] / "fixtures/tool_surface_snapshots.json").read_text()
)


def test_builtin_tool_surface_profile_snapshots_are_pinned() -> None:
    ensure_builtin_mcp_servers_registered()
    actual = {
        name: {
            "tool_count": len(surface.contracts),
            "provider_count": len(surface.provider_tools),
            "fingerprints": surface.fingerprints,
        }
        for name, surface in _profile_surfaces().items()
    }

    assert actual == EXPECTED_PROFILE_SNAPSHOTS


def test_default_profiles_do_not_include_operator_or_provider_hidden_tools() -> None:
    surfaces = _profile_surfaces()
    default_tool_ids = {contract.tool_id for contract in surfaces["agent.desktop.default"].contracts}
    sandbox_surface = surfaces["env.sandbox.control_plane"]

    assert {"cua_replay_trajectory", "cua_move_cursor", "cua_set_config", "cua_page"}.isdisjoint(default_tool_ids)
    assert sandbox_surface.provider_tools == ()


def _profile_surfaces():
    return {
        "agent.desktop.default": ToolSurfaceResolver(_default_desktop_contracts()).resolve(
            ToolProfile(name="agent.desktop.default")
        ),
        "agent.web": ToolSurfaceResolver((*web_agent_action_contracts(), *web_agent_probe_contracts())).resolve(
            ToolProfile(name="agent.web")
        ),
        "evaluator.desktop.round": ToolSurfaceResolver(_evaluator_contracts()).resolve(
            ToolProfile(name="evaluator.desktop.round", audiences=frozenset({ToolAudience.EVALUATOR}))
        ),
        "env.sandbox.control_plane": ToolSurfaceResolver(SANDBOX_PROVIDER_TOOL_CONTRACTS).resolve(
            ToolProfile(
                name="env.sandbox.control_plane",
                audiences=frozenset({ToolAudience.ENV_BUILDER}),
                visibility=frozenset({ToolVisibility.OPERATOR_ONLY}),
            ),
            config=ToolSurfaceConfig(include_visibility=frozenset({ToolVisibility.OPERATOR_ONLY})),
        ),
    }


def _default_desktop_contracts() -> tuple[ToolContract, ...]:
    return tuple(
        contract
        for contract in desktop_tool_contracts()
        if contract.visibility in {ToolVisibility.MODEL_VISIBLE, ToolVisibility.RUNTIME_ONLY}
    )


def _evaluator_contracts() -> tuple[ToolContract, ...]:
    client_tools = (*standard_client_probe_tools(), *standard_client_gui_tools())
    return (
        *(spec.contract for spec in ARTIFACT_TOOL_REFS),
        final_report_tool_contract(RoundEvaluationReport),
        *(tool.contract for tool in client_tools if tool.contract is not None),
    )
