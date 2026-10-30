"""Shared terminal-contract coverage for built-in and low-code agents."""

from __future__ import annotations

import asyncio

from northcua.agentkit import (
    AgentBudgetPolicy,
    AgentBudgetScope,
    AgentCliSpec,
    AgentRunStatus,
    AgentRuntimeSpec,
    AgentSpec,
    AgentStepBudget,
    AgentTerminalAction,
    AgentTerminalDecision,
    AgentTerminalReason,
    StepwiseToolAgentTemplate,
    StepwiseToolStepContext,
    StepwiseToolStepResult,
    TerminalDecisionPolicy,
)
from northcua.core.enums import TaskStatus
from northcua.core.models import StepLogRecord
from northcua_protocol.capabilities import PlatformKind


def test_task_status_serializes_lowercase_values() -> None:
    assert TaskStatus.CONTINUE.value == "continue"
    assert TaskStatus.FINISH.value == "finish"
    assert TaskStatus.FAIL.value == "fail"
    assert TaskStatus.ERROR.value == "error"


def test_task_status_accepts_pre_migration_uppercase_values() -> None:
    assert TaskStatus("CONTINUE") == TaskStatus.CONTINUE
    assert TaskStatus("FINISH") == TaskStatus.FINISH
    assert TaskStatus("FAIL") == TaskStatus.FAIL
    assert TaskStatus("ERROR") == TaskStatus.ERROR

    record = StepLogRecord(
        session_id="sess-legacy",
        session_step=1,
        round_step=1,
        round_num=1,
        status="FINISH",
        request="legacy artifact",
    )
    assert record.status == TaskStatus.FINISH


def test_terminal_decision_serializes_domain_neutral_payload() -> None:
    decision = AgentTerminalDecision.finish(final_answer="Done.", summary="Completed cleanly.")

    assert decision.action == AgentTerminalAction.FINISH
    assert decision.reason == AgentTerminalReason.COMPLETED
    assert AgentRunStatus.from_terminal_decision(decision) == AgentRunStatus.SUCCESS
    assert decision.to_json() == {
        "action": "finish",
        "reason": "completed",
        "success": True,
        "retryable": False,
        "final_answer": "Done.",
        "summary": "Completed cleanly.",
        "metadata": {},
    }


def test_terminal_policy_detects_infeasible_environment_text() -> None:
    policy = TerminalDecisionPolicy()

    decision = policy.failure_decision_from_text(
        "The requested app is not installed in this sandbox.",
        summary="Model reported an environment gap.",
    )

    assert decision.action == AgentTerminalAction.FAIL
    assert decision.reason == AgentTerminalReason.INFEASIBLE


def test_budget_policy_emits_scoped_max_step_metadata() -> None:
    policy = AgentBudgetPolicy()
    budget = AgentStepBudget(max_round_steps=5, max_session_steps=3)

    session_decision = policy.terminal_decision_after_step(budget=budget, round_step=2, session_step=3)
    round_decision = policy.terminal_decision_after_step(
        budget=AgentStepBudget(max_round_steps=5),
        round_step=5,
        session_step=5,
    )

    assert session_decision is not None
    assert session_decision.reason == AgentTerminalReason.MAX_STEPS
    assert session_decision.metadata == {"scope": "session", "limit": 3, "current": 3}
    assert round_decision is not None
    assert round_decision.metadata == {"scope": "round", "limit": 5, "current": 5}


def test_budget_policy_builds_model_iteration_decision() -> None:
    decision = AgentBudgetPolicy.max_step_decision(
        scope=AgentBudgetScope.MODEL_ITERATION,
        limit=8,
        current=8,
    )

    assert decision.reason == AgentTerminalReason.MAX_STEPS
    assert decision.metadata == {"scope": "model_iteration", "limit": 8, "current": 8}


def test_stepwise_template_prefers_explicit_terminal_decision() -> None:
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="demo", display_name="Demo", platform=PlatformKind.LINUX),
        cli=AgentCliSpec(agent_name="demo", help="Demo agent"),
    )

    async def _executor(context: StepwiseToolStepContext) -> StepwiseToolStepResult:
        return StepwiseToolStepResult(
            status="blocked",
            summary="Display is unavailable.",
            terminal_decision=AgentTerminalDecision.infeasible(
                final_answer="Display is unavailable.",
                summary="The sandbox cannot expose a GUI display.",
            ),
        )

    result = asyncio.run(
        StepwiseToolAgentTemplate(spec=spec, step_executor=_executor).run(query="Open app", max_steps=3)
    )

    assert result.status == AgentRunStatus.FAIL
    assert result.terminal_decision.reason == AgentTerminalReason.INFEASIBLE
    assert result.to_terminal_decision().reason == AgentTerminalReason.INFEASIBLE
    assert len(result.steps) == 1
    assert result.steps[0].normalized_terminal_decision().reason == AgentTerminalReason.INFEASIBLE


def test_stepwise_template_preserves_legacy_done_semantics() -> None:
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="demo", display_name="Demo", platform=PlatformKind.LINUX),
        cli=AgentCliSpec(agent_name="demo", help="Demo agent"),
    )

    async def _executor(context: StepwiseToolStepContext) -> StepwiseToolStepResult:
        return StepwiseToolStepResult(status="ok", summary="Finished", done=True)

    result = asyncio.run(
        StepwiseToolAgentTemplate(spec=spec, step_executor=_executor).run(query="Do work", max_steps=3)
    )

    assert result.status == AgentRunStatus.SUCCESS
    assert result.terminal_decision.reason == AgentTerminalReason.COMPLETED
    assert result.steps[0].normalized_terminal_decision().reason == AgentTerminalReason.COMPLETED


def test_stepwise_template_marks_session_step_budget() -> None:
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="demo", display_name="Demo", platform=PlatformKind.LINUX),
        cli=AgentCliSpec(agent_name="demo", help="Demo agent"),
    )

    async def _executor(context: StepwiseToolStepContext) -> StepwiseToolStepResult:
        return StepwiseToolStepResult(status="ok", summary=f"step {context.step_index}")

    result = asyncio.run(
        StepwiseToolAgentTemplate(spec=spec, step_executor=_executor).run(
            query="Keep going",
            max_steps=5,
            max_session_steps=2,
        )
    )

    assert result.status == AgentRunStatus.MAX_STEPS
    assert len(result.steps) == 2
    assert result.steps[-1].normalized_terminal_decision().metadata["scope"] == "session"


def test_stepwise_template_respects_existing_session_step_offset() -> None:
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="demo", display_name="Demo", platform=PlatformKind.LINUX),
        cli=AgentCliSpec(agent_name="demo", help="Demo agent"),
    )
    seen_session_steps: list[int] = []

    async def _executor(context: StepwiseToolStepContext) -> StepwiseToolStepResult:
        seen_session_steps.append(context.session_step)
        return StepwiseToolStepResult(status="ok", summary=f"session step {context.session_step}")

    result = asyncio.run(
        StepwiseToolAgentTemplate(spec=spec, step_executor=_executor).run(
            query="Keep going",
            max_steps=5,
            max_session_steps=3,
            session_step_offset=2,
        )
    )

    assert seen_session_steps == [3]
    assert result.status == AgentRunStatus.MAX_STEPS
    assert result.steps[-1].normalized_terminal_decision().metadata == {
        "scope": "session",
        "limit": 3,
        "current": 3,
    }


def test_stepwise_template_stops_before_work_when_session_budget_is_exhausted() -> None:
    spec = AgentSpec(
        runtime=AgentRuntimeSpec(kind="demo", display_name="Demo", platform=PlatformKind.LINUX),
        cli=AgentCliSpec(agent_name="demo", help="Demo agent"),
    )
    calls = 0

    async def _executor(context: StepwiseToolStepContext) -> StepwiseToolStepResult:
        nonlocal calls
        calls += 1
        return StepwiseToolStepResult(status="ok", summary=f"step {context.step_index}")

    result = asyncio.run(
        StepwiseToolAgentTemplate(spec=spec, step_executor=_executor).run(
            query="Keep going",
            max_steps=5,
            max_session_steps=3,
            session_step_offset=3,
        )
    )

    assert calls == 0
    assert result.status == AgentRunStatus.MAX_STEPS
    assert result.terminal_decision.metadata == {
        "scope": "session",
        "limit": 3,
        "current": 3,
    }
    assert result.steps[0].normalized_terminal_decision().metadata == {
        "scope": "session",
        "limit": 3,
        "current": 3,
    }
