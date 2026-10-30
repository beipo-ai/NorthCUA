from __future__ import annotations

from pathlib import Path

import pytest

from northcua.evaluation.script import ScriptArtifactSpec, ScriptEvaluatorRunner, ScriptEvaluatorSpec
from northcua.sandboxes.contracts import SandboxCommandResult


class _Runtime:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.argv: tuple[str, ...] = ()

    async def run_argv(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: float,
        cwd: str,
        env: dict[str, str],
        allow_failure: bool,
    ) -> SandboxCommandResult:
        _ = timeout_seconds, cwd, env, allow_failure
        self.argv = argv
        return SandboxCommandResult(stdout='noise\n{"status":"passed","score":0.75}\n')

    async def download_file(self, remote_path: str, local_path: Path) -> None:
        _ = remote_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("artifact", encoding="utf-8")


@pytest.mark.asyncio
async def test_runtime_port_preserves_argv_and_reuses_parser_and_artifacts(tmp_path: Path) -> None:
    runtime = _Runtime(tmp_path)
    malicious = "value; touch /tmp/must-not-exist $(id) > output"
    hook = ScriptEvaluatorSpec(
        hook_id="reward",
        argv=["python3", "script with spaces.py", "", "你好", malicious],
        artifacts=[ScriptArtifactSpec(remote_path="/tmp/report.json", required=True)],
    )

    result = await ScriptEvaluatorRunner(output_dir=tmp_path).run_one_runtime(
        hook=hook,
        runtime=runtime,
        task_id="task-1",
        attempt=1,
    )

    assert runtime.argv == tuple(hook.argv)
    assert runtime.argv[-1] == malicious
    assert result.passed is True
    assert result.score == 0.75
    assert result.artifacts["/tmp/report.json"].endswith("report.json")


@pytest.mark.asyncio
async def test_runtime_port_requires_argv(tmp_path: Path) -> None:
    runner = ScriptEvaluatorRunner(output_dir=tmp_path)
    with pytest.raises(ValueError, match="requires argv"):
        await runner.run_one_runtime(
            hook=ScriptEvaluatorSpec(hook_id="legacy", command="echo legacy"),
            runtime=_Runtime(tmp_path),
            task_id="task-1",
            attempt=1,
        )


@pytest.mark.asyncio
async def test_runtime_port_redacts_provider_exception_from_result_and_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "private-provider-secret-sentinel"

    class _FailingRuntime(_Runtime):
        async def run_argv(
            self,
            argv: tuple[str, ...],
            *,
            timeout_seconds: float,
            cwd: str,
            env: dict[str, str],
            allow_failure: bool,
        ) -> SandboxCommandResult:
            del argv, timeout_seconds, cwd, env, allow_failure
            raise RuntimeError(sentinel)

    result = await ScriptEvaluatorRunner(output_dir=tmp_path).run_one_runtime(
        hook=ScriptEvaluatorSpec(hook_id="reward", argv=["python3", "evaluate.py"]),
        runtime=_FailingRuntime(tmp_path),
        task_id="task-1",
        attempt=1,
    )

    assert result.error == "script evaluator runtime failed [RuntimeError]"
    assert sentinel not in result.model_dump_json()
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_runtime_port_redacts_required_artifact_exception_from_result_and_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "private-artifact-provider-secret-sentinel"

    class _FailingDownloadRuntime(_Runtime):
        async def download_file(self, remote_path: str, local_path: Path) -> None:
            del remote_path, local_path
            raise RuntimeError(sentinel)

    result = await ScriptEvaluatorRunner(output_dir=tmp_path).run_one_runtime(
        hook=ScriptEvaluatorSpec(
            hook_id="reward",
            argv=["python3", "evaluate.py"],
            artifacts=[ScriptArtifactSpec(remote_path="/tmp/private.json", required=True)],
        ),
        runtime=_FailingDownloadRuntime(tmp_path),
        task_id="task-1",
        attempt=1,
    )

    assert result.error == "required post-run artifact download failed: /tmp/private.json"
    assert result.artifact_errors == {"/tmp/private.json": "artifact download failed [RuntimeError]"}
    assert sentinel not in result.model_dump_json()
    assert sentinel not in caplog.text
