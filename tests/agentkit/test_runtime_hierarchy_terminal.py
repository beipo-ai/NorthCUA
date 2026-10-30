"""Shared AgentKit terminal projection and preflight budget tests."""

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
    AgentRuntimeTerminalContext,
    AgentRuntimeTerminalPoint,
    AgentRuntimeTerminalProjection,
    AgentSessionRuntime,
    AgentStepRuntime,
)
from northcua.core.budget import AgentStepBudget
from northcua.core.terminal import AgentTerminalDecision, AgentTerminalReason


@dataclass(slots=True, frozen=True)
class _StepRecord:
    """Small runtime step record used by terminal projection tests."""

    round_step: int
    session_step: int
    phase: AgentRuntimePhase
    label: str = "step"


class _Resource:
    def __init__(self) -> None:
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        pass


def test_round_runtime_preflights_budget_before_starting_resources() -> None:
    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        raise AssertionError("step executor should not run")

    resource = _Resource()
    counters = AgentRuntimeCounterState(session_id="sess", next_session_step=4)
    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=counters, resource=resource),
        step_runtime=AgentStepRuntime(step_executor=_executor),
    )

    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="do work",
                budget=AgentStepBudget(max_round_steps=5, max_session_steps=3),
                fallback_round_steps=5,
            )
        )
    )

    assert not resource.started
    assert result.steps == ()
    assert result.terminal_decision.reason == AgentTerminalReason.MAX_STEPS


def test_round_runtime_projects_pre_step_terminal_without_starting_resource() -> None:
    class _StateStore:
        def __init__(self) -> None:
            self.saved = 0

        async def hydrate(self) -> None:
            pass

        async def save(self) -> None:
            self.saved += 1

    class _Projection:
        async def project_terminal(
            self,
            context: AgentRuntimeTerminalContext[_StepRecord],
        ) -> AgentRuntimeTerminalProjection[_StepRecord] | None:
            assert context.point == AgentRuntimeTerminalPoint.PRE_STEP_LIMIT
            assert context.cursor is not None
            return AgentRuntimeTerminalProjection(
                result=_StepRecord(
                    round_step=context.cursor.round_step,
                    session_step=context.cursor.session_step,
                    phase=context.cursor.phase,
                    label="pre-limit",
                ),
                should_advance=True,
            )

    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        raise AssertionError("step executor should not run")

    resource = _Resource()
    store = _StateStore()
    counters = AgentRuntimeCounterState(session_id="sess", next_session_step=4)
    seen_steps: list[_StepRecord] = []
    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=counters, resource=resource, state_store=store),
        step_runtime=AgentStepRuntime(step_executor=_executor),
        terminal_projection=_Projection(),
    )

    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="do work",
                budget=AgentStepBudget(max_round_steps=5, max_session_steps=3),
                fallback_round_steps=5,
            ),
            on_step=seen_steps.append,
        )
    )

    assert not resource.started
    assert [step.label for step in result.steps] == ["pre-limit"]
    assert seen_steps == list(result.steps)
    assert counters.current_round_num == 1
    assert counters.next_session_step == 5
    assert store.saved == 1
    assert result.terminal_decision.reason == AgentTerminalReason.MAX_STEPS


def test_round_runtime_projects_post_step_terminal_before_step_callback() -> None:
    class _Projection:
        async def project_terminal(
            self,
            context: AgentRuntimeTerminalContext[_StepRecord],
        ) -> AgentRuntimeTerminalProjection[_StepRecord] | None:
            assert context.point == AgentRuntimeTerminalPoint.POST_STEP_LIMIT
            assert context.steps[-1].label == "model"
            assert context.cursor is not None
            return AgentRuntimeTerminalProjection(
                result=_StepRecord(
                    round_step=context.cursor.round_step,
                    session_step=context.cursor.session_step,
                    phase=context.cursor.phase,
                    label="post-limit",
                )
            )

    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        return AgentRuntimeStepOutcome(
            cursor=context.cursor,
            result=_StepRecord(
                round_step=context.cursor.round_step,
                session_step=context.cursor.session_step,
                phase=context.cursor.phase,
                label="model",
            ),
            terminal_decision=AgentTerminalDecision.continue_(summary="continue"),
        )

    seen_steps: list[_StepRecord] = []
    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=AgentRuntimeCounterState(session_id="sess")),
        step_runtime=AgentStepRuntime(step_executor=_executor),
        terminal_projection=_Projection(),
    )

    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="do work",
                budget=AgentStepBudget(max_round_steps=1),
                fallback_round_steps=5,
            ),
            on_step=seen_steps.append,
        )
    )

    assert [step.label for step in result.steps] == ["post-limit"]
    assert seen_steps == list(result.steps)
    assert result.terminal_decision.reason == AgentTerminalReason.MAX_STEPS


def test_round_runtime_projects_loop_exhaustion_terminal() -> None:
    class _Projection:
        async def project_terminal(
            self,
            context: AgentRuntimeTerminalContext[_StepRecord],
        ) -> AgentRuntimeTerminalProjection[_StepRecord] | None:
            assert context.point == AgentRuntimeTerminalPoint.LOOP_EXHAUSTED
            assert context.cursor is not None
            return AgentRuntimeTerminalProjection(
                result=_StepRecord(
                    round_step=context.cursor.round_step,
                    session_step=context.cursor.session_step,
                    phase=context.cursor.phase,
                    label="loop-exhausted",
                )
            )

    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        return AgentRuntimeStepOutcome(
            cursor=context.cursor,
            result=_StepRecord(
                round_step=context.cursor.round_step,
                session_step=context.cursor.session_step,
                phase=context.cursor.phase,
                label="model",
            ),
            terminal_decision=AgentTerminalDecision.continue_(summary="continue"),
        )

    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=AgentRuntimeCounterState(session_id="sess")),
        step_runtime=AgentStepRuntime(step_executor=_executor),
        terminal_projection=_Projection(),
    )

    seen_steps: list[_StepRecord] = []
    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="do work",
                budget=AgentStepBudget(),
                fallback_round_steps=1,
            ),
            on_step=seen_steps.append,
        )
    )

    assert [step.label for step in result.steps] == ["loop-exhausted"]
    assert [step.label for step in seen_steps] == ["loop-exhausted"]
    assert result.terminal_decision.reason == AgentTerminalReason.MAX_STEPS
