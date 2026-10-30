"""Recovery policy decision tests."""

from __future__ import annotations

from northcua.batch.contracts import BatchManifest, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.experiments.contracts import (
    ExperimentAppProfile,
    ExperimentSpec,
    ExperimentTaskResult,
    ExperimentTaskSpec,
)
from northcua.experiments.recovery import DefaultFailurePolicy
from northcua.experiments.recovery.planner import ExperimentRecoveryPlanner
from northcua.reliability.contracts import FailureAction, RecoverablePhase
from northcua.telemetry.contracts import FailureCategory


def test_failure_policy_retries_failed_case_while_budget_remains() -> None:
    decision = DefaultFailurePolicy().decide(
        state=BatchTaskState(
            task_id="task",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            controller_task_id="controller",
        ),
        verification=None,
        max_attempts=2,
    )

    assert decision.action == FailureAction.RETRY_CASE
    assert decision.phase == RecoverablePhase.MAIN_AGENT
    assert decision.failure_category == FailureCategory.MAIN_AGENT
    assert decision.retry_budget_remaining == 1


def test_failure_policy_fails_fast_after_retry_budget_is_spent() -> None:
    decision = DefaultFailurePolicy().decide(
        state=BatchTaskState(task_id="task", status=BatchTaskStatus.TIMED_OUT, attempts=2),
        verification=None,
        max_attempts=2,
    )

    assert decision.action == FailureAction.FAIL_FAST
    assert decision.terminal is True
    assert decision.failure_category == FailureCategory.TIMEOUT


def test_failure_policy_preserves_existing_fail_fast_action() -> None:
    decision = DefaultFailurePolicy().decide(
        state=BatchTaskState(
            task_id="task",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            recovery_action=FailureAction.FAIL_FAST,
        ),
        verification=None,
        max_attempts=3,
    )

    assert decision.action == FailureAction.FAIL_FAST
    assert decision.terminal is True
    assert decision.retry_budget_remaining == 0


def test_failure_policy_preserves_non_retryable_terminal_decision() -> None:
    decision = DefaultFailurePolicy().decide(
        state=BatchTaskState(
            task_id="task",
            status=BatchTaskStatus.FAILED,
            attempts=1,
            terminal_decision={"action": "fail", "reason": "max_steps", "retryable": False},
        ),
        verification=None,
        max_attempts=3,
    )

    assert decision.action == FailureAction.FAIL_FAST
    assert decision.terminal is True
    assert decision.retry_budget_remaining == 0


def test_recovery_plan_preserves_non_retryable_terminal_decision(tmp_path) -> None:
    spec = ExperimentSpec(
        experiment_id="terminal-plan",
        tasks=[
            ExperimentTaskSpec(
                task_id="task",
                app_profile=ExperimentAppProfile.CHROME,
                query="Open the fixture",
            )
        ],
    )
    manifest = BatchManifest(
        batch_id="terminal-plan",
        tasks=[BatchTaskSpec(task_id="task", agent_kind="desktop", query="Open the fixture", max_retries=2)],
    )
    state = BatchTaskState(
        task_id="task",
        status=BatchTaskStatus.FAILED,
        attempts=1,
        terminal_decision={"action": "fail", "reason": "max_steps", "retryable": False},
    )

    plan = ExperimentRecoveryPlanner().plan(
        spec=spec,
        manifest=manifest,
        output_dir=tmp_path,
        states=[state],
    )

    assert plan.items[0].action == FailureAction.FAIL_FAST
    assert plan.items[0].runnable is False
    assert plan.items[0].terminal is True


def test_experiment_task_spec_defaults_main_agent_max_steps_to_100() -> None:
    task = ExperimentTaskSpec(task_id="task-default", app_profile=ExperimentAppProfile.CHROME, query="Inspect page")

    assert task.max_steps == 100


def test_failure_policy_marks_canceled_terminal() -> None:
    decision = DefaultFailurePolicy().decide(
        state=BatchTaskState(task_id="task", status=BatchTaskStatus.CANCELED, attempts=1),
        verification=None,
        max_attempts=3,
    )

    assert decision.action == FailureAction.MARK_CANCELED
    assert decision.terminal is True


def test_failure_policy_retries_evaluator_when_main_trace_exists() -> None:
    decision = DefaultFailurePolicy().decide(
        state=BatchTaskState(task_id="task", status=BatchTaskStatus.SUCCEEDED, attempts=1),
        verification=ExperimentTaskResult(
            task_id="task",
            app_profile=ExperimentAppProfile.CHROME,
            status="succeeded",
            main_steps_path="/tmp/steps.jsonl",
            main_llm_trace_path="/tmp/llm_trace.jsonl",
            errors=["evaluator llm_trace.jsonl missing or empty"],
        ),
        max_attempts=1,
    )

    assert decision.action == FailureAction.RETRY_EVALUATOR
    assert decision.evaluator_only is True
    assert decision.failure_category == FailureCategory.EVALUATOR


def test_failure_policy_rematerializes_artifacts_for_non_evaluator_missing_output() -> None:
    decision = DefaultFailurePolicy().decide(
        state=BatchTaskState(task_id="task", status=BatchTaskStatus.SUCCEEDED, attempts=1),
        verification=ExperimentTaskResult(
            task_id="task",
            app_profile=ExperimentAppProfile.CHROME,
            status="succeeded",
            errors=["main-agent screenshot missing"],
        ),
        max_attempts=1,
    )

    assert decision.action == FailureAction.REMATERIALIZE_ARTIFACTS
    assert decision.failure_category == FailureCategory.ARTIFACT_MISSING
