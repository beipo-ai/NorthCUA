"""Terminal-decision adapter coverage for the generic web agent."""

from __future__ import annotations

from northcua.agentkit import AgentTerminalAction, AgentTerminalReason
from northcua.agents.web.domain.results import WebAgentResult, WebRunStatus, WebTerminalStatus


def _web_result(
    *,
    status: WebRunStatus,
    terminal_status: WebTerminalStatus,
    final_answer: str = "",
    success: bool = False,
    tool_errors: list[str] | None = None,
) -> WebAgentResult:
    return WebAgentResult(
        status=status,
        terminal_status=terminal_status,
        session_id="sess-web",
        user_query="Inspect page",
        final_answer=final_answer,
        success=success,
        tool_errors=tool_errors or [],
    )


def test_web_success_maps_to_completed_finish() -> None:
    result = _web_result(
        status=WebRunStatus.SUCCESS,
        terminal_status=WebTerminalStatus.COMPLETED,
        final_answer="The heading is visible.",
        success=True,
    )

    decision = result.to_terminal_decision()

    assert decision.action == AgentTerminalAction.FINISH
    assert decision.reason == AgentTerminalReason.COMPLETED
    assert decision.success is True


def test_web_completed_failure_can_report_infeasible_environment() -> None:
    result = _web_result(
        status=WebRunStatus.FAILED,
        terminal_status=WebTerminalStatus.COMPLETED,
        final_answer="The file does not exist in this environment.",
        success=False,
    )

    decision = result.to_terminal_decision()

    assert decision.action == AgentTerminalAction.FAIL
    assert decision.reason == AgentTerminalReason.INFEASIBLE


def test_web_tool_error_is_retryable_tool_error() -> None:
    result = _web_result(
        status=WebRunStatus.FAILED,
        terminal_status=WebTerminalStatus.TOOL_ERROR,
        tool_errors=["click failed"],
    )

    decision = result.to_terminal_decision()

    assert decision.reason == AgentTerminalReason.TOOL_ERROR
    assert decision.retryable is True


def test_web_max_steps_maps_to_shared_max_steps_reason() -> None:
    result = _web_result(status=WebRunStatus.MAX_STEPS, terminal_status=WebTerminalStatus.MAX_STEPS)

    decision = result.to_terminal_decision()

    assert decision.action == AgentTerminalAction.FAIL
    assert decision.reason == AgentTerminalReason.MAX_STEPS
