"""Shared AgentKit runtime hierarchy tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from northcua.agentkit.runtime import (
    AgentRoundRuntime,
    AgentRoundRuntimeSpec,
    AgentRuntimeCounterState,
    AgentRuntimePhase,
    AgentRuntimeStepContext,
    AgentRuntimeStepOutcome,
    AgentSessionRuntime,
    AgentStepRuntime,
    ModelStepPort,
    ObservationPort,
    PortStepPipelineExecutor,
    PromptPort,
    ToolDispatchPort,
)
from northcua.core.budget import AgentStepBudget
from northcua.core.terminal import AgentTerminalDecision, AgentTerminalReason


@dataclass(slots=True, frozen=True)
class _StepRecord:
    """Small runtime step record used by hierarchy unit tests."""

    round_step: int
    session_step: int
    phase: AgentRuntimePhase


def test_session_runtime_can_restart_resource_after_close() -> None:
    class _Resource:
        def __init__(self) -> None:
            self.started = 0
            self.closed = 0

        async def start(self) -> None:
            self.started += 1

        async def close(self) -> None:
            self.closed += 1

    resource = _Resource()
    runtime = AgentSessionRuntime(
        counters=AgentRuntimeCounterState(session_id="sess"),
        resource=resource,
    )

    asyncio.run(runtime.start())
    asyncio.run(runtime.close())
    asyncio.run(runtime.start())

    assert resource.started == 2
    assert resource.closed == 1


def test_port_step_pipeline_composes_low_code_runtime_ports() -> None:
    class _ObservationPort(ObservationPort[dict[str, object]]):
        async def collect(
            self,
            context: AgentRuntimeStepContext[dict[str, object], object],
        ) -> dict[str, object]:
            return {"query": context.query, "step": context.step_index}

    class _PromptPort(PromptPort[dict[str, object], list[str]]):
        def project(self, context: AgentRuntimeStepContext[dict[str, object], list[str]]) -> list[str]:
            return [str(context.observation), f"history={len(context.history)}"]

    class _ModelPort(ModelStepPort[list[str], str]):
        async def run_model_step(self, context: AgentRuntimeStepContext[object, list[str]]) -> str:
            return "|".join(context.prompt or [])

    class _ToolPort(ToolDispatchPort[str, dict[str, object]]):
        async def dispatch_tools(
            self,
            context: AgentRuntimeStepContext[object, object],
            model_result: str,
        ) -> dict[str, object]:
            return {"model_result": model_result, "session_step": context.session_step}

    executor = PortStepPipelineExecutor(
        observation_port=_ObservationPort(),
        prompt_port=_PromptPort(),
        model_step_port=_ModelPort(),
        tool_dispatch_port=_ToolPort(),
        result_builder=lambda context, model, tools: {
            "prompt": list(context.prompt or []),
            "model": model,
            "tools": tools,
        },
        terminal_decision_builder=lambda _context, _model, _tools: AgentTerminalDecision.finish(final_answer="done"),
    )
    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=AgentRuntimeCounterState(session_id="sess")),
        step_runtime=AgentStepRuntime(step_executor=executor),
    )

    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="inspect",
                budget=AgentStepBudget(max_round_steps=3),
                fallback_round_steps=3,
            )
        )
    )

    assert result.steps[0]["tools"]["session_step"] == 1
    assert result.terminal_decision.success


def test_step_runtime_preserves_domain_exception_type() -> None:
    class DomainStepError(RuntimeError):
        pass

    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        raise DomainStepError("domain failure")

    runtime = AgentStepRuntime(step_executor=_executor)

    try:
        asyncio.run(
            runtime.run_step(
                cursor=AgentRuntimeCounterState(session_id="sess").reserve_step(),
                query="fail",
                history=(),
            )
        )
    except DomainStepError:
        pass
    else:
        raise AssertionError("expected original domain exception")
