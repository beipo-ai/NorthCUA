"""Low-code AgentKit template coverage for the shared runtime hierarchy."""

from __future__ import annotations

import asyncio

from northcua.agentkit import AgentCliSpec, AgentRuntimeSpec, AgentSpec, SessionAdapter
from northcua.agentkit.runtime_templates import (
    StepwiseToolAgentTemplate,
    StepwiseToolStepContext,
    StepwiseToolStepResult,
)
from northcua_protocol.capabilities import PlatformKind


def _fake_spec() -> AgentSpec:
    return AgentSpec(
        runtime=AgentRuntimeSpec(
            kind="fake_runtime",
            display_name="Fake Runtime",
            platform=PlatformKind.LINUX,
        ),
        cli=AgentCliSpec(agent_name="fake-runtime", help="Fake runtime test agent."),
    )


def test_low_code_template_uses_session_runtime_counters_and_lifecycle() -> None:
    class _Session(SessionAdapter):
        def __init__(self) -> None:
            super().__init__(agent_kind="fake_runtime", session_id="lowcode-session")
            self.started = 0
            self.closed = 0

        async def start(self) -> None:
            self.started += 1

        async def close(self) -> None:
            self.closed += 1

    session = _Session()
    seen_session_steps: list[int] = []

    async def _executor(context: StepwiseToolStepContext) -> StepwiseToolStepResult:
        seen_session_steps.append(context.session_step)
        return StepwiseToolStepResult(
            step_index=context.step_index,
            summary=f"step {context.step_index}",
            done=context.step_index == 2,
        )

    result = asyncio.run(
        StepwiseToolAgentTemplate(
            spec=_fake_spec(),
            session=session,
            step_executor=_executor,
        ).run(
            query="run",
            max_steps=5,
            session_step_offset=4,
        )
    )

    assert session.started == 1
    assert session.closed == 1
    assert seen_session_steps == [5, 6]
    assert [step.step_index for step in result.steps] == [1, 2]
    assert result.terminal_decision.success


def test_low_code_template_preflights_session_budget_before_starting_resource() -> None:
    class _Session(SessionAdapter):
        def __init__(self) -> None:
            super().__init__(agent_kind="fake_runtime", session_id="lowcode-session")
            self.started = 0

        async def start(self) -> None:
            self.started += 1

    session = _Session()

    async def _executor(context: StepwiseToolStepContext) -> StepwiseToolStepResult:
        raise AssertionError("step executor should not run")

    result = asyncio.run(
        StepwiseToolAgentTemplate(
            spec=_fake_spec(),
            session=session,
            step_executor=_executor,
        ).run(
            query="run",
            max_steps=5,
            max_session_steps=2,
            session_step_offset=2,
        )
    )

    assert session.started == 0
    assert result.terminal_decision.metadata == {"scope": "session", "limit": 2, "current": 2}
