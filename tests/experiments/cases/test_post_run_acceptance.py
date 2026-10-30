from __future__ import annotations

import json
from pathlib import Path

import pytest

from northcua.batch.contracts import BatchManifest, BatchRunResult, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.post_run import PostRunHookResult, PostRunHookSpec, PostRunHookStatus
from northcua.experiments.cases.contracts import CaseSpec, CaseSuiteSpec, PostRunAcceptancePolicy
from northcua.experiments.cases.materialization import MaterializedCase, MaterializedCaseSuite
from northcua.experiments.cases.operations_support import (
    apply_post_run_acceptance_to_batch_result,
    rewrite_batch_summary_for_post_run_acceptance,
)
from northcua.sandboxes.contracts import SandboxProviderSpec


def _materialized_suite(
    tmp_path: Path,
    *,
    policy: PostRunAcceptancePolicy = PostRunAcceptancePolicy.REPORT_ONLY,
    include_hook: bool = True,
) -> MaterializedCaseSuite:
    hooks = [PostRunHookSpec(hook_id="primary-oracle", command="/tmp/oracle.py")] if include_hook else []
    case = CaseSpec(
        task_id="case-1",
        query="Do it",
        app_profile="browser.chrome",
        post_run_hooks=hooks,
    )
    sandbox = SandboxProviderSpec(provider="fake", template="template")
    return MaterializedCaseSuite(
        spec=CaseSuiteSpec(
            suite_id="post-run-acceptance",
            post_run_acceptance_policy=policy,
            tasks=[case],
        ),
        output_dir=tmp_path,
        fixture_dir=tmp_path / "fixtures",
        cases=(MaterializedCase(case=case, fixtures=(), startup_commands=(), sandbox=sandbox),),
        manifest=BatchManifest(
            batch_id="post-run-acceptance",
            tasks=[
                BatchTaskSpec(
                    task_id=case.task_id,
                    agent_kind="desktop",
                    query=case.query,
                    sandbox=sandbox,
                    post_run_hooks=hooks,
                )
            ],
        ),
    )


def _batch_result(tmp_path: Path, results: list[PostRunHookResult] | None = None) -> BatchRunResult:
    return BatchRunResult(
        batch_id="post-run-acceptance",
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[
            BatchTaskState(
                task_id="case-1",
                status=BatchTaskStatus.SUCCEEDED,
                post_run_results=results or [],
            )
        ],
    )


def test_post_run_acceptance_defaults_to_report_only_compatibility(tmp_path: Path) -> None:
    materialized = _materialized_suite(tmp_path)
    batch_result = _batch_result(
        tmp_path,
        [PostRunHookResult(hook_id="primary-oracle", status=PostRunHookStatus.FAILED, passed=False)],
    )

    projected = apply_post_run_acceptance_to_batch_result(
        materialized=materialized,
        batch_result=batch_result,
        dry_run=False,
    )

    assert materialized.spec.post_run_acceptance_policy == PostRunAcceptancePolicy.REPORT_ONLY
    assert projected is batch_result
    assert projected.succeeded == 1
    assert projected.failed == 0
    assert projected.task_states[0].status == BatchTaskStatus.SUCCEEDED


@pytest.mark.parametrize(
    ("include_hook", "results", "failure_code"),
    [
        (False, [], "primary_hook_missing"),
        (True, [], "primary_result_missing"),
        (
            True,
            [
                PostRunHookResult(
                    hook_id="primary-oracle",
                    status=PostRunHookStatus.ERROR,
                    error="oracle crashed",
                )
            ],
            "primary_hook_error",
        ),
        (
            True,
            [PostRunHookResult(hook_id="primary-oracle", status=PostRunHookStatus.FAILED, passed=False)],
            "primary_verdict_failed",
        ),
    ],
)
def test_strict_post_run_acceptance_fails_missing_error_or_false(
    tmp_path: Path,
    include_hook: bool,
    results: list[PostRunHookResult],
    failure_code: str,
) -> None:
    materialized = _materialized_suite(
        tmp_path,
        policy=PostRunAcceptancePolicy.REQUIRE_PRIMARY_PASS,
        include_hook=include_hook,
    )

    projected = apply_post_run_acceptance_to_batch_result(
        materialized=materialized,
        batch_result=_batch_result(tmp_path, results),
        dry_run=False,
    )

    state = projected.task_states[0]
    assert projected.succeeded == 0
    assert projected.failed == 1
    assert state.status == BatchTaskStatus.FAILED
    assert state.terminal_decision["post_run_acceptance"]["code"] == failure_code
    assert state.terminal_decision["post_run_acceptance"]["passed"] is False


def test_strict_post_run_acceptance_preserves_true_primary_pass(tmp_path: Path) -> None:
    materialized = _materialized_suite(
        tmp_path,
        policy=PostRunAcceptancePolicy.REQUIRE_PRIMARY_PASS,
    )
    batch_result = _batch_result(
        tmp_path,
        [PostRunHookResult(hook_id="primary-oracle", status=PostRunHookStatus.PASSED, passed=True)],
    )

    projected = apply_post_run_acceptance_to_batch_result(
        materialized=materialized,
        batch_result=batch_result,
        dry_run=False,
    )

    assert projected.succeeded == 1
    assert projected.failed == 0
    assert projected.task_states[0].status == BatchTaskStatus.SUCCEEDED


def test_strict_post_run_acceptance_skips_dry_run_oracles(tmp_path: Path) -> None:
    materialized = _materialized_suite(
        tmp_path,
        policy=PostRunAcceptancePolicy.REQUIRE_PRIMARY_PASS,
    )
    batch_result = _batch_result(tmp_path)

    projected = apply_post_run_acceptance_to_batch_result(
        materialized=materialized,
        batch_result=batch_result,
        dry_run=True,
    )

    assert projected is batch_result
    assert projected.succeeded == 1


def test_strict_post_run_acceptance_marks_batch_summary(tmp_path: Path) -> None:
    materialized = _materialized_suite(
        tmp_path,
        policy=PostRunAcceptancePolicy.REQUIRE_PRIMARY_PASS,
    )
    (tmp_path / "batch_summary.json").write_text('{"succeeded": 1, "failed": 0}', encoding="utf-8")
    projected = apply_post_run_acceptance_to_batch_result(
        materialized=materialized,
        batch_result=_batch_result(tmp_path),
        dry_run=False,
    )

    rewrite_batch_summary_for_post_run_acceptance(
        materialized=materialized,
        batch_result=projected,
        dry_run=False,
    )

    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["failed"] == 1
    assert summary["post_run_acceptance"]["passed"] is False
    assert summary["post_run_acceptance"]["failed_task_ids"] == ["case-1"]


def test_strict_primary_oracle_policy_accepts_cases_without_post_run_hooks() -> None:
    suite = CaseSuiteSpec(
        suite_id="strict-primary-oracle",
        post_run_acceptance_policy=PostRunAcceptancePolicy.REQUIRE_PRIMARY_PASS,
        tasks=[CaseSpec(task_id="case-without-hook", query="Do it", app_profile="browser.chrome")],
    )

    assert suite.post_run_acceptance_policy == PostRunAcceptancePolicy.REQUIRE_PRIMARY_PASS
    assert any(not case.post_run_hooks for case in suite.tasks)
