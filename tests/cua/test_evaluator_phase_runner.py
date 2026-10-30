from __future__ import annotations

from northcua.agents.desktop.evaluation.phase_runner import DesktopEvaluatorPhaseRunner
from northcua.transport.ws.message_seed import SUPPRESS_RAW_AGENT_MCP_SERVERS_KEY
from tests.transport.test_websocket_transport import _agent_config


def test_legacy_evaluator_phase_config_suppresses_raw_agent_mcp_servers() -> None:
    runner = DesktopEvaluatorPhaseRunner.__new__(DesktopEvaluatorPhaseRunner)
    runner.agent_config = _agent_config()

    prepared = runner.prepare_agent_config(
        tools=[],
        before_model_hooks=[],
        after_model_hooks=[],
        after_tool_hooks=[],
    )

    assert prepared.mcp_servers == []
    assert prepared.global_storage[SUPPRESS_RAW_AGENT_MCP_SERVERS_KEY] is True
