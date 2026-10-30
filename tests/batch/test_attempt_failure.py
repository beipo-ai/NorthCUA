"""Batch attempt failure classification tests."""

from __future__ import annotations

from northcua.batch.contracts import BatchTaskState, BatchTaskStatus
from northcua.batch.runner.attempt_failure import _failure_category_for_attempt
from northcua.telemetry.contracts import FailureCategory


def test_readiness_message_overrides_starting_sandbox_phase_bucket() -> None:
    state = BatchTaskState(task_id="case", status=BatchTaskStatus.STARTING_SANDBOX)

    category = _failure_category_for_attempt(
        state=state,
        status=BatchTaskStatus.FAILED,
        error=RuntimeError("case recipe readiness probe timed out"),
    )

    assert category == FailureCategory.READINESS_TIMEOUT


def test_device_command_timeout_message_has_specific_bucket() -> None:
    state = BatchTaskState(task_id="case", status=BatchTaskStatus.RUNNING, controller_task_id="controller-task")

    category = _failure_category_for_attempt(
        state=state,
        status=BatchTaskStatus.FAILED,
        error=RuntimeError("COMMAND_TIMED_OUT while executing click"),
    )

    assert category == FailureCategory.DEVICE_COMMAND_TIMEOUT


def test_evaluator_timeout_message_has_evaluator_bucket() -> None:
    state = BatchTaskState(task_id="case", status=BatchTaskStatus.SUBMITTED, controller_task_id="controller-task")

    category = _failure_category_for_attempt(
        state=state,
        status=BatchTaskStatus.FAILED,
        error=RuntimeError("Evaluator round 1 timed out after 300.00s"),
    )

    assert category == FailureCategory.EVALUATOR
