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
    label: str = "step"


def test_runtime_counter_tracks_main_and_evaluator_phases_separately() -> None:
    counters = AgentRuntimeCounterState(session_id="sess")
    assert counters.begin_round() == 1

    main_cursor = counters.reserve_step()
    counters.complete_step(main_cursor)
    evaluator_cursor = counters.reserve_step(AgentRuntimePhase.EVALUATOR)
    counters.complete_step(evaluator_cursor)

    assert main_cursor.round_step == 1
    assert main_cursor.session_step == 1
    assert evaluator_cursor.round_step == 2
    assert evaluator_cursor.session_step == 2
    assert evaluator_cursor.phase_step == 1
    assert counters.next_session_step == 2
    assert counters.next_round_step == 2
    assert counters.phase_steps[AgentRuntimePhase.EVALUATOR] == 2


def test_round_runtime_enforces_round_budget_with_standard_terminal_decision() -> None:
    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        record = _StepRecord(
            round_step=context.cursor.round_step,
            session_step=context.cursor.session_step,
            phase=context.cursor.phase,
        )
        return AgentRuntimeStepOutcome(
            cursor=context.cursor,
            result=record,
            terminal_decision=AgentTerminalDecision.continue_(summary="continue"),
        )

    counters = AgentRuntimeCounterState(session_id="sess")
    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=counters),
        step_runtime=AgentStepRuntime(step_executor=_executor),
    )

    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="do work",
                budget=AgentStepBudget(max_round_steps=2),
                fallback_round_steps=5,
            )
        )
    )

    assert [step.round_step for step in result.steps] == [1, 2]
    assert result.terminal_decision.reason == AgentTerminalReason.MAX_STEPS
    assert result.terminal_decision.metadata == {"scope": "round", "limit": 2, "current": 2}


def test_round_runtime_exposes_current_step_budget_metadata() -> None:
    seen_budgets: list[dict[str, object]] = []

    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        seen_budgets.append(dict(context.metadata["step_budget"]))
        return AgentRuntimeStepOutcome(
            cursor=context.cursor,
            result=_StepRecord(
                round_step=context.cursor.round_step,
                session_step=context.cursor.session_step,
                phase=context.cursor.phase,
            ),
            terminal_decision=AgentTerminalDecision.continue_(summary="continue"),
        )

    counters = AgentRuntimeCounterState(session_id="sess")
    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=counters),
        step_runtime=AgentStepRuntime(step_executor=_executor),
    )

    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="do work",
                budget=AgentStepBudget(max_round_steps=3, max_session_steps=10),
                fallback_round_steps=99,
            )
        )
    )

    assert result.terminal_decision.reason == AgentTerminalReason.MAX_STEPS
    assert seen_budgets == [
        {
            "phase": "main",
            "current_round_step": 1,
            "round_limit": 3,
            "remaining_round_steps": 3,
            "current_session_step": 1,
            "session_limit": 10,
        },
        {
            "phase": "main",
            "current_round_step": 2,
            "round_limit": 3,
            "remaining_round_steps": 2,
            "current_session_step": 2,
            "session_limit": 10,
        },
        {
            "phase": "main",
            "current_round_step": 3,
            "round_limit": 3,
            "remaining_round_steps": 1,
            "current_session_step": 3,
            "session_limit": 10,
        },
    ]


def test_round_runtime_can_continue_hydrated_round_without_resetting_counters() -> None:
    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        return AgentRuntimeStepOutcome(
            cursor=context.cursor,
            result=_StepRecord(
                round_step=context.cursor.round_step,
                session_step=context.cursor.session_step,
                phase=context.cursor.phase,
            ),
            terminal_decision=AgentTerminalDecision.finish(final_answer="done"),
        )

    counters = AgentRuntimeCounterState(
        session_id="sess",
        current_round_num=2,
        next_round_step=4,
        next_session_step=9,
    )
    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=counters),
        step_runtime=AgentStepRuntime(step_executor=_executor),
    )

    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="resume",
                budget=AgentStepBudget(max_round_steps=10),
                fallback_round_steps=10,
                begin_new_round=False,
            )
        )
    )

    assert result.steps[0].round_step == 4
    assert result.steps[0].session_step == 9
    assert counters.current_round_num == 2
    assert counters.next_round_step == 5
    assert counters.next_session_step == 10


def test_evaluator_phase_does_not_advance_main_session_step_budget() -> None:
    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        record = _StepRecord(
            round_step=context.cursor.round_step,
            session_step=context.cursor.session_step,
            phase=context.cursor.phase,
        )
        return AgentRuntimeStepOutcome(
            cursor=context.cursor,
            result=record,
            terminal_decision=AgentTerminalDecision.finish(final_answer="report ready"),
        )

    counters = AgentRuntimeCounterState(session_id="sess")
    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=counters),
        step_runtime=AgentStepRuntime(step_executor=_executor),
    )

    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="evaluate",
                budget=AgentStepBudget(max_round_steps=3),
                fallback_round_steps=3,
                phase=AgentRuntimePhase.EVALUATOR,
            )
        )
    )

    assert result.steps[0].phase == AgentRuntimePhase.EVALUATOR
    assert counters.next_session_step == 1
    assert counters.next_round_step == 1
    assert counters.phase_steps[AgentRuntimePhase.EVALUATOR] == 2


def test_non_main_phase_ignores_session_budget_and_uses_phase_step_index() -> None:
    seen_step_indices: list[int] = []

    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        seen_step_indices.append(context.step_index)
        return AgentRuntimeStepOutcome(
            cursor=context.cursor,
            result=_StepRecord(
                round_step=context.cursor.round_step,
                session_step=context.cursor.session_step,
                phase=context.cursor.phase,
            ),
            terminal_decision=AgentTerminalDecision.continue_(summary="continue"),
        )

    counters = AgentRuntimeCounterState(session_id="sess", next_session_step=10)
    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=counters),
        step_runtime=AgentStepRuntime(step_executor=_executor),
    )

    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="evaluate",
                budget=AgentStepBudget(max_round_steps=2, max_session_steps=1),
                fallback_round_steps=2,
                phase=AgentRuntimePhase.EVALUATOR,
            )
        )
    )

    assert seen_step_indices == [1, 2]
    assert len(result.steps) == 2
    assert result.terminal_decision.metadata == {"scope": "round", "limit": 2, "current": 2}
    assert counters.next_session_step == 10


def test_non_main_phase_budget_uses_phase_counter_when_round_counter_is_advanced() -> None:
    seen_step_indices: list[int] = []

    async def _executor(
        context: AgentRuntimeStepContext[object, object],
    ) -> AgentRuntimeStepOutcome[_StepRecord]:
        seen_step_indices.append(context.step_index)
        return AgentRuntimeStepOutcome(
            cursor=context.cursor,
            result=_StepRecord(
                round_step=context.cursor.round_step,
                session_step=context.cursor.session_step,
                phase=context.cursor.phase,
            ),
            terminal_decision=AgentTerminalDecision.continue_(summary="continue"),
        )

    counters = AgentRuntimeCounterState(
        session_id="sess",
        current_round_num=3,
        next_round_step=8,
        next_session_step=12,
    )
    runtime = AgentRoundRuntime(
        session_runtime=AgentSessionRuntime(counters=counters),
        step_runtime=AgentStepRuntime(step_executor=_executor),
    )

    result = asyncio.run(
        runtime.run(
            AgentRoundRuntimeSpec(
                query="evaluate",
                budget=AgentStepBudget(max_round_steps=2, max_session_steps=1),
                fallback_round_steps=2,
                phase=AgentRuntimePhase.EVALUATOR,
                begin_new_round=False,
            )
        )
    )

    assert seen_step_indices == [1, 2]
    assert [step.phase for step in result.steps] == [AgentRuntimePhase.EVALUATOR, AgentRuntimePhase.EVALUATOR]
    assert counters.next_round_step == 8
    assert counters.next_session_step == 12
