"""Regression coverage for desktop stop-policy parsing and terminal decisions."""

from __future__ import annotations

from northcua.agents.desktop.services.stop_policy import StopPolicyService
from northcua.core.enums import ScreenshotSelectionKind, TaskStatus
from northcua.core.models import AssistantToolCallRecord, StepLogRecord, ToolResultArtifact, WindowState
from northcua.core.terminal import AgentTerminalReason
from northcua.telemetry.contracts import FailureCategory


def _step_with_single_action(*, name: str, arguments: str, image_path: str) -> StepLogRecord:
    return StepLogRecord(
        session_id="sess-1",
        session_step=0,
        round_step=0,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Open Calculator",
        current_window=WindowState(),
        assistant_tool_calls=[
            AssistantToolCallRecord(
                id="call-1",
                name=name,
                arguments=arguments,
            )
        ],
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-1",
                name=name,
                content="ok",
                status="success",
                image_path=image_path,
                selected_image_kind=ScreenshotSelectionKind.RAW,
            )
        ],
    )


def test_stop_policy_finishes_on_text_only_terminal_answer() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text="Calculator is open, and 12 + 34 = 46.",
        assistant_tool_calls=[],
        tool_results=[],
        step_logs=[],
    )

    assert decision.status == TaskStatus.FINISH
    assert decision.result == "Calculator is open, and 12 + 34 = 46."
    assert decision.terminal_decision.reason == AgentTerminalReason.COMPLETED


def test_stop_policy_requires_infeasible_tag_for_environment_failure() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text="[infeasible] I cannot open the application because access is blocked.",
        assistant_tool_calls=[],
        tool_results=[],
        step_logs=[],
    )

    assert decision.status == TaskStatus.FAIL
    assert "access is blocked" in decision.summary.lower()
    assert decision.terminal_decision.reason == AgentTerminalReason.INFEASIBLE


def test_stop_policy_plain_failure_text_is_model_reported_failure() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text="I cannot complete the task after the attempted actions failed.",
        assistant_tool_calls=[],
        tool_results=[],
        step_logs=[],
    )

    assert decision.status == TaskStatus.FAIL
    assert decision.terminal_decision.reason == AgentTerminalReason.MODEL_REPORTED_FAILURE


def test_stop_policy_recognizes_common_apology_failure_prefixes() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text="Sorry, I cannot complete the task because the file never appeared.",
        assistant_tool_calls=[],
        tool_results=[],
        step_logs=[],
    )

    assert decision.status == TaskStatus.FAIL
    assert decision.terminal_decision.reason == AgentTerminalReason.MODEL_REPORTED_FAILURE


def test_stop_policy_recognizes_not_possible_failure_prefix() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text="It is not possible to complete this task with the current permissions.",
        assistant_tool_calls=[],
        tool_results=[],
        step_logs=[],
    )

    assert decision.status == TaskStatus.FAIL
    assert decision.terminal_decision.reason == AgentTerminalReason.MODEL_REPORTED_FAILURE


def test_stop_policy_allows_domain_text_with_failure_words() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text=(
            "Driver License Eligibility Requirements found: a driving privilege cannot be suspended, "
            "and photocopied documents may be refused."
        ),
        assistant_tool_calls=[],
        tool_results=[],
        step_logs=[],
    )

    assert decision.status == TaskStatus.FINISH
    assert decision.terminal_decision.reason == AgentTerminalReason.COMPLETED


def test_stop_policy_untagged_environment_failure_is_not_success() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text="The requested app is not installed in this environment.",
        assistant_tool_calls=[],
        tool_results=[],
        step_logs=[],
    )

    assert decision.status == TaskStatus.FAIL
    assert decision.terminal_decision.reason == AgentTerminalReason.MODEL_REPORTED_FAILURE


def test_stop_policy_recognizes_legacy_environment_failure_phrases() -> None:
    service = StopPolicyService()

    for assistant_text in (
        "The environment lacks LibreOffice Calc, so I cannot edit the spreadsheet.",
        "Missing requirement: the image editor is unavailable.",
        "Not installed in this environment: the requested application.",
        "Could not complete the task because the required file is missing.",
    ):
        decision = service.decide(
            assistant_text=assistant_text,
            assistant_tool_calls=[],
            tool_results=[],
            step_logs=[],
        )

        assert decision.status == TaskStatus.FAIL
        assert decision.terminal_decision.reason == AgentTerminalReason.MODEL_REPORTED_FAILURE


def test_stop_policy_allows_source_specific_caveat_in_terminal_answer() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text=(
            "I inspected the files and found the requested summary. I cannot accurately cite "
            "methodology caveats from one mislabeled source, so I noted that caveat and cited the "
            "usable report pages instead."
        ),
        assistant_tool_calls=[],
        tool_results=[],
        step_logs=[],
    )

    assert decision.status == TaskStatus.FINISH
    assert decision.terminal_decision.reason == AgentTerminalReason.COMPLETED


def test_stop_policy_allows_blocked_as_domain_fact_in_terminal_answer() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text=(
            "I reviewed the spreadsheet and found the upload status is blocked by a pending license. "
            "The next action is to update the inventory with that project status."
        ),
        assistant_tool_calls=[],
        tool_results=[],
        step_logs=[],
    )

    assert decision.status == TaskStatus.FINISH
    assert decision.terminal_decision.reason == AgentTerminalReason.COMPLETED


def test_stop_policy_continues_when_tool_calls_are_present() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text="I am opening Calculator.",
        assistant_tool_calls=[
            AssistantToolCallRecord(id="call-1", name="exec", arguments='{"command":"open -a Calculator"}')
        ],
        tool_results=[ToolResultArtifact(tool_call_id="call-1", name="exec", content="Opened", status="success")],
        step_logs=[],
    )

    assert decision.status == TaskStatus.CONTINUE
    assert decision.terminal_decision.reason == AgentTerminalReason.NEEDS_MORE_STEPS


def test_stop_policy_continues_when_all_tools_error() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text="The click did not work.",
        assistant_tool_calls=[AssistantToolCallRecord(id="call-1", name="click", arguments='{"x":1,"y":2}')],
        tool_results=[
            ToolResultArtifact(tool_call_id="call-1", name="click", content="timed out", status="error", is_error=True)
        ],
        step_logs=[],
    )

    assert decision.status == TaskStatus.CONTINUE
    assert decision.terminal_decision.reason == AgentTerminalReason.NEEDS_MORE_STEPS
    assert "tool error" in decision.summary.lower()
    assert decision.terminal_decision.metadata["tool_error_count"] == 1
    assert decision.terminal_decision.metadata["tool_names"] == ["click"]
    assert decision.terminal_decision.metadata["tool_error_excerpt"] == "timed out"


def test_stop_policy_keeps_device_timeout_metadata_while_continuing() -> None:
    service = StopPolicyService()

    decision = service.decide(
        assistant_text="The scroll did not return.",
        assistant_tool_calls=[AssistantToolCallRecord(id="call-1", name="scroll", arguments='{"delta_y":300}')],
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-1",
                name="scroll",
                content="{'status': 'error', 'error': 'COMMAND_TIMED_OUT'}",
                status="error",
                is_error=True,
            )
        ],
        step_logs=[],
    )

    assert decision.status == TaskStatus.CONTINUE
    assert decision.terminal_decision.reason == AgentTerminalReason.NEEDS_MORE_STEPS
    assert decision.terminal_decision.metadata["failure_category"] == FailureCategory.DEVICE_COMMAND_TIMEOUT.value
    assert "COMMAND_TIMED_OUT" in str(decision.terminal_decision.metadata["tool_error_excerpt"])


def test_stop_policy_fails_on_repeated_ineffective_actions() -> None:
    service = StopPolicyService()
    previous_steps = [
        _step_with_single_action(name="click", arguments='{"x":100,"y":200}', image_path="/tmp/same.png"),
        _step_with_single_action(name="click", arguments='{"x":100,"y":200}', image_path="/tmp/same.png"),
    ]

    decision = service.decide(
        assistant_text="Trying the same click again.",
        assistant_tool_calls=[AssistantToolCallRecord(id="call-3", name="click", arguments='{"x":100,"y":200}')],
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-3",
                name="click",
                content="Clicked",
                status="success",
                image_path="/tmp/same.png",
                selected_image_kind=ScreenshotSelectionKind.RAW,
            )
        ],
        step_logs=previous_steps,
    )

    assert decision.status == TaskStatus.FAIL
    assert "Repeated ineffective actions" in decision.summary
    assert decision.terminal_decision.reason == AgentTerminalReason.REPEATED_INEFFECTIVE_ACTION
