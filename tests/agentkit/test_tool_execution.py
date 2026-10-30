"""Tests for shared AgentKit pending tool-call execution."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("nexau.archs.tool.tool", reason="AgentKit tool execution tests require NexAU.")

from nexau.archs.tool.tool import Tool

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.tool_execution import AgentKitToolCallExecutor
from northcua.core.models import ToolExecutionRecord


def test_serial_tool_skips_later_mutations() -> None:
    """A terminal serial tool prevents later same-turn mutations from running."""

    calls: list[str] = []

    def _finalize() -> dict[str, str]:
        calls.append("finalize")
        return {"status": "success", "summary": "finalized"}

    def _write() -> dict[str, str]:
        calls.append("write")
        return {"status": "success", "summary": "written"}

    tools = (
        Tool(
            name="finalize_environment_plan",
            description="finalize",
            input_schema={},
            implementation=_finalize,
            disable_parallel=True,
        ),
        Tool(name="write_text_file", description="write", input_schema={}, implementation=_write),
    )
    trace_state = TraceCaptureState()

    executions = asyncio.run(
        AgentKitToolCallExecutor().execute(
            pending_tool_calls=[
                {"id": "final", "name": "finalize_environment_plan", "input": {}},
                {"id": "write", "name": "write_text_file", "input": {}},
            ],
            trace_state=trace_state,
            tools=tools,
        )
    )

    assert calls == ["finalize"]
    assert [execution.tool_name for execution in executions] == ["finalize_environment_plan", "write_text_file"]
    assert executions[1].is_error is False
    assert executions[1].tool_output == {
        "status": "skipped",
        "summary": "Skipped because finalize_environment_plan is a serial terminal tool.",
    }


def test_executor_does_not_redispatch_transport_executed_calls() -> None:
    """Calls already present in trace executions are not executed a second time."""

    calls: list[str] = []

    def _load_skill() -> dict[str, str]:
        calls.append("load-skill")
        return {"status": "success"}

    def _analyze() -> dict[str, str]:
        calls.append("analysis")
        return {"status": "success"}

    trace_state = TraceCaptureState()
    trace_state.tool_executions.append(
        ToolExecutionRecord(
            tool_call_id="call-load",
            tool_name="LoadSkill",
            tool_output={"status": "success"},
            llm_tool_output={"status": "success"},
        )
    )
    tools = (
        Tool(name="LoadSkill", description="load", input_schema={}, implementation=_load_skill),
        Tool(name="submit_environment_analysis", description="analysis", input_schema={}, implementation=_analyze),
    )

    executions = asyncio.run(
        AgentKitToolCallExecutor().execute(
            pending_tool_calls=[
                {"id": "call-load", "name": "LoadSkill", "input": {}},
                {"id": "call-analysis", "name": "submit_environment_analysis", "input": {}},
            ],
            trace_state=trace_state,
            tools=tools,
        )
    )

    assert calls == ["analysis"]
    assert [execution.tool_name for execution in executions] == ["submit_environment_analysis"]
    assert [execution.tool_name for execution in trace_state.tool_executions] == [
        "LoadSkill",
        "submit_environment_analysis",
    ]


def test_executor_passes_runtime_context_without_recording_it() -> None:
    """Runtime-only context is injected into tools without polluting tool arguments."""

    seen: list[object] = []
    agent_state = object()

    def _load_skill(*, skill_name: str, agent_state: object) -> dict[str, str]:
        seen.append(agent_state)
        return {"status": "success", "skill": skill_name}

    trace_state = TraceCaptureState()
    tools = (Tool(name="LoadSkill", description="load", input_schema={}, implementation=_load_skill),)

    executions = asyncio.run(
        AgentKitToolCallExecutor().execute(
            pending_tool_calls=[{"id": "call-load", "name": "LoadSkill", "input": {"skill_name": "minimax-xlsx"}}],
            trace_state=trace_state,
            tools=tools,
            tool_context={"agent_state": agent_state},
            tool_context_tool_names=frozenset({"LoadSkill"}),
        )
    )

    assert seen == [agent_state]
    assert executions[0].is_error is False
    assert executions[0].tool_input == {"skill_name": "minimax-xlsx"}
    assert "agent_state" not in executions[0].tool_input


def test_executor_passes_runtime_context_only_to_explicit_parameters() -> None:
    """Generic tools do not receive private runtime context by accident."""

    seen_load: list[object] = []
    seen_generic: list[dict[str, object]] = []
    agent_state = object()

    def _load_skill(*, skill_name: str, agent_state: object) -> dict[str, str]:
        seen_load.append(agent_state)
        return {"status": "success", "skill": skill_name}

    def _generic_tool(**payload: object) -> dict[str, str]:
        seen_generic.append(dict(payload))
        return {"status": "success"}

    trace_state = TraceCaptureState()
    tools = (
        Tool(name="LoadSkill", description="load", input_schema={}, implementation=_load_skill),
        Tool(name="submit_environment_analysis", description="analysis", input_schema={}, implementation=_generic_tool),
    )

    executions = asyncio.run(
        AgentKitToolCallExecutor().execute(
            pending_tool_calls=[
                {"id": "call-load", "name": "LoadSkill", "input": {"skill_name": "minimax-xlsx"}},
                {"id": "call-analysis", "name": "submit_environment_analysis", "input": {}},
            ],
            trace_state=trace_state,
            tools=tools,
            tool_context={"agent_state": agent_state},
            tool_context_tool_names=frozenset({"LoadSkill"}),
        )
    )

    assert seen_load == [agent_state]
    assert seen_generic == [{}]
    assert [execution.is_error for execution in executions] == [False, False]
