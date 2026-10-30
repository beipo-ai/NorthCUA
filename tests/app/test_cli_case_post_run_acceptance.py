from __future__ import annotations

from pathlib import Path

import pytest

from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
from northcua.cli import case_commands
from northcua.cli.parser import build_parser
from northcua.experiments.cases.contracts import CaseRunMode, CaseRunResult


@pytest.mark.asyncio
async def test_case_run_returns_nonzero_for_projected_post_run_acceptance_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_result = BatchRunResult(
        batch_id="strict-oracle-failure",
        total=1,
        succeeded=0,
        failed=1,
        canceled=0,
        output_dir=str(tmp_path),
        task_states=[BatchTaskState(task_id="case-1", status=BatchTaskStatus.FAILED)],
    )
    result = CaseRunResult(
        suite_id="strict-oracle-failure",
        run_id="strict-oracle-failure",
        run_mode=CaseRunMode.LIVE,
        total=1,
        succeeded=0,
        failed=1,
        canceled=0,
        output_dir=str(tmp_path),
        batch_result=batch_result,
        case_report_path=str(tmp_path / "case_report.json"),
        batch_manifest_path=str(tmp_path / "batch_manifest.json"),
    )

    class StubCaseOperations:
        async def run_file(self, **_: object) -> CaseRunResult:
            return result

    monkeypatch.setattr(case_commands, "CaseOperations", StubCaseOperations)
    args = build_parser().parse_args(["case", "run", "--spec", "suite.yaml"])

    assert await case_commands.run_case_command(args) == 1
