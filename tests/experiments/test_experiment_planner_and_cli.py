"""Experiment planner and CLI parser tests."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from northcua.cli.case_commands import register_case_commands
from northcua.cli.controller_commands import register_controller_commands
from northcua.cli.experiment_commands import register_experiment_commands, run_experiment_command
from northcua.cli.parallelism_args import apply_batch_parallelism_defaults
from northcua.experiments.contracts import (
    ExperimentAppProfile,
    ExperimentControllerSpec,
    ExperimentSpec,
    ExperimentTaskSpec,
    TraceExpectationSpec,
)
from northcua.experiments.materialization import ExperimentMaterializer
from northcua.experiments.orchestration.controller import ManagedControllerProcess
from northcua.experiments.planner import ExperimentBatchPlanner
from tests.support.experiment_fixtures import golden_e2b_experiment_spec, run_fixture_experiment
from northcua_protocol.capabilities import Capability


def test_experiment_to_batch_manifest_projects_three_desktop_tasks(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(public_controller_url="ws://controller.example/ws", provider="fake")
    spec.tasks[0].max_session_steps = 11
    materialized = ExperimentMaterializer().materialize(spec, output_dir=tmp_path / "experiment")
    manifest = ExperimentBatchPlanner().plan(materialized, controller_url=spec.controller.public_url)

    assert manifest.batch_id == spec.experiment_id
    assert manifest.defaults.controller_url == "ws://controller.example/ws"
    assert manifest.defaults.concurrency == 3
    assert [task.agent_kind for task in manifest.tasks] == ["desktop", "desktop", "desktop"]
    assert manifest.tasks[0].max_session_steps == 11
    assert all(task.sandbox is not None and task.sandbox.file_uploads for task in manifest.tasks)


def test_experiment_planner_carries_controller_coordinate_policy_into_batch_tasks(tmp_path: Path) -> None:
    app_config = tmp_path / "app.yaml"
    app_config.write_text(
        """
agent_config_path: agent.yaml
desktop:
  coordinate_policy:
    input_space: normalized_1000
    clamp_to_bounds: true
""",
        encoding="utf-8",
    )
    spec = golden_e2b_experiment_spec(public_controller_url="ws://controller.example/ws", provider="fake")
    spec.controller.app_config_path = app_config
    materialized = ExperimentMaterializer().materialize(spec, output_dir=tmp_path / "experiment")

    manifest = ExperimentBatchPlanner().plan(materialized, controller_url=spec.controller.public_url)

    assert manifest.tasks[0].metadata["controller_app_config_path"] == str(app_config)
    assert manifest.tasks[0].metadata["northcua.desktop.coordinate_space"] == "normalized_1000"
    assert manifest.tasks[0].metadata["northcua.desktop.coordinate_clamp_to_bounds"] == "true"


def test_experiment_planner_adds_capabilities_for_configured_evaluator_live_tools(tmp_path: Path) -> None:
    app_config = tmp_path / "app.yaml"
    app_config.write_text(
        """
agent_config_path: agent.yaml
evaluation:
  tools:
    allow_client_live_probe: true
    allowed_placements:
      - server_artifact
      - client_probe
      - client_mutating
      - finalizer
    client_probe_tools:
      - client_bash_readonly
    client_gui_tools:
      - client_gui_click
""",
        encoding="utf-8",
    )
    spec = golden_e2b_experiment_spec(public_controller_url="ws://controller.example/ws", provider="fake")
    spec.controller.app_config_path = app_config
    materialized = ExperimentMaterializer().materialize(spec, output_dir=tmp_path / "experiment")

    manifest = ExperimentBatchPlanner().plan(materialized, controller_url=spec.controller.public_url)

    assert all(Capability.SHELL_EXEC in task.required_capabilities for task in manifest.tasks)
    assert all(Capability.DESKTOP_INPUT in task.required_capabilities for task in manifest.tasks)
    assert all(Capability.BROWSER_PAGE not in task.required_capabilities for task in manifest.tasks)


def test_experiment_planner_projects_web_url_and_browser_capability(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        experiment_id="web-e2b-three-test",
        controller=ExperimentControllerSpec(app_config_path=Path("config/app.web.e2b.yaml")),
        trace_expectations=TraceExpectationSpec(evaluator_kind="web_round_evaluator"),
        tasks=[
            ExperimentTaskSpec(
                task_id="web-info-lookup",
                app_profile=ExperimentAppProfile.CHROME,
                agent_kind="web",
                query="Find the code.",
                metadata={"url": " http://127.0.0.1:8761/index.html ", "web_case_kind": "info_lookup"},
            )
        ],
    )
    materialized = ExperimentMaterializer().materialize(spec, output_dir=tmp_path / "experiment")

    manifest = ExperimentBatchPlanner().plan(materialized, controller_url="ws://controller.example/ws")

    assert manifest.tasks[0].url == "http://127.0.0.1:8761/index.html"
    assert Capability.BROWSER_PAGE in manifest.tasks[0].required_capabilities
    assert Capability.DESKTOP_INPUT not in manifest.tasks[0].required_capabilities


def test_experiment_validate_parser_accepts_spec() -> None:
    args = _experiment_parser().parse_args(["experiment", "validate", "--spec", "experiment.yaml"])

    assert args.command_group == "experiment"
    assert args.experiment_command == "validate"
    assert args.spec == "experiment.yaml"


def test_experiment_parser_uses_public_controller_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_PUBLIC_CONTROLLER_URL", "wss://cloudflare.example/ws")

    args = _experiment_parser().parse_args(["experiment", "run", "--spec", "suite.yaml"])

    assert args.public_controller_url == "wss://cloudflare.example/ws"


def test_experiment_resume_preserves_persisted_url_unless_override_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_PUBLIC_CONTROLLER_URL", "wss://cloudflare.example/ws")

    args = _experiment_parser().parse_args(["experiment", "resume", "--experiment-id", "exp"])

    assert args.public_controller_url == ""


def test_case_parser_uses_public_controller_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_PUBLIC_CONTROLLER_URL", "wss://cloudflare.example/ws")

    args = _case_parser().parse_args(
        [
            "case",
            "run",
            "--spec",
            "suite.yaml",
            "--submit-concurrency",
            "20",
            "--submit-jitter-seconds",
            "2",
            "--device-readiness-timeout-seconds",
            "600",
            "--infra-retry-backoff-seconds",
            "30,90,180,300",
        ]
    )

    assert args.public_controller_url == "wss://cloudflare.example/ws"
    assert args.submit_concurrency == 20
    assert args.submit_jitter_seconds == 2
    assert args.device_readiness_timeout_seconds == 600
    assert args.infra_retry_backoff_seconds == "30,90,180,300"


def test_case_parallelism_fills_batch_and_prepared_defaults() -> None:
    args = _case_parser().parse_args(
        [
            "case",
            "run",
            "--spec",
            "suite.yaml",
            "--parallelism",
            "120",
            "--prepare-sandboxes",
        ]
    )

    apply_batch_parallelism_defaults(args)

    assert args.concurrency == 120
    assert args.create_concurrency == 24
    assert args.submit_concurrency == 36
    assert args.status_concurrency == 12
    assert args.prepare_buffer == 90
    assert args.prepare_concurrency == 36


def test_experiment_resume_and_recover_plan_parsers() -> None:
    parser = _experiment_parser()

    resume = parser.parse_args(
        [
            "experiment",
            "resume",
            "--experiment-id",
            "exp",
            "--parallelism",
            "120",
            "--prepare-sandboxes",
            "--public-controller-url",
            "wss://c",
        ]
    )
    apply_batch_parallelism_defaults(resume)
    plan = parser.parse_args(["experiment", "recover-plan", "--experiment-id", "exp", "--format", "md"])

    assert resume.experiment_command == "resume"
    assert resume.parallelism == 120
    assert resume.concurrency == 120
    assert resume.create_concurrency == 24
    assert resume.submit_concurrency == 36
    assert resume.status_concurrency == 12
    assert resume.prepare_buffer == 90
    assert resume.prepare_concurrency == 36
    assert resume.public_controller_url == "wss://c"
    assert plan.experiment_command == "recover-plan"
    assert plan.format == "md"


@pytest.mark.asyncio
async def test_experiment_resume_cli_forwards_parallelism_and_resume_throttles(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    class _FakeExperimentResult:
        batch_result = SimpleNamespace(failed=0, canceled=0)
        task_results = [SimpleNamespace(passed=True)]

        def model_dump(self, *, mode: str) -> dict[str, object]:
            return {"mode": mode, "ok": True}

    class _FakeExperimentOperations:
        async def resume(self, **kwargs: object) -> _FakeExperimentResult:
            captured.update(kwargs)
            return _FakeExperimentResult()

    monkeypatch.setattr("northcua.cli.experiment_commands.ExperimentOperations", _FakeExperimentOperations)
    args = _experiment_parser().parse_args(
        [
            "experiment",
            "resume",
            "--experiment-id",
            "exp",
            "--output-dir",
            "out",
            "--parallelism",
            "120",
            "--concurrency",
            "80",
            "--create-concurrency",
            "11",
            "--prepare-sandboxes",
            "--prepare-buffer",
            "70",
            "--submit-jitter-seconds",
            "2.5",
            "--device-readiness-timeout-seconds",
            "600",
            "--infra-retry-backoff-seconds",
            "30,90",
            "--public-controller-url",
            "wss://controller.example/ws",
        ]
    )

    exit_code = await run_experiment_command(args)

    assert exit_code == 0
    assert captured["experiment_id"] == "exp"
    assert captured["output_dir"] == Path("out")
    assert captured["concurrency"] == 80
    assert captured["sandbox_create_concurrency"] == 11
    assert captured["controller_submit_concurrency"] == 36
    assert captured["controller_status_concurrency"] == 12
    assert captured["controller_submit_jitter_seconds"] == 2.5
    assert captured["device_readiness_timeout_seconds"] == 600
    assert captured["infra_retry_backoff_seconds"] == [30.0, 90.0]
    assert captured["public_controller_url"] == "wss://controller.example/ws"
    policy = captured["prepared_sandbox_policy"]
    assert policy.enabled is True
    assert policy.buffer == 70
    assert policy.prepare_concurrency == 36
    assert '"ok": true' in capsys.readouterr().out


def test_experiment_pause_drain_cancel_parsers() -> None:
    parser = _experiment_parser()

    for command in ("pause", "drain", "cancel"):
        args = parser.parse_args(["experiment", command, "--experiment-id", "exp"])
        assert args.experiment_command == command
        assert args.experiment_id == "exp"


def test_controller_parser_accepts_artifact_root_dir() -> None:
    args = _controller_parser().parse_args(
        ["controller", "serve", "--artifact-root-dir", "logs/experiments/run/agent-artifacts"]
    )

    assert args.controller_command == "serve"
    assert args.artifact_root_dir == "logs/experiments/run/agent-artifacts"


def test_managed_controller_args_include_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "agent-artifacts"
    spec = ExperimentControllerSpec(
        local_port=8899,
        db_path=tmp_path / "controller.sqlite",
        blob_dir=tmp_path / "controller-blobs",
        artifact_root_dir=artifact_root,
    )

    args = ManagedControllerProcess().serve_args(spec)

    assert "--artifact-root-dir" in args
    assert args[args.index("--artifact-root-dir") + 1] == str(artifact_root)


def test_managed_controller_args_include_process_executor_options(tmp_path: Path) -> None:
    spec = ExperimentControllerSpec(
        local_port=8899,
        executor_mode="process",
        runner_workers=8,
        worker_queue_max=32,
        worker_start_timeout_seconds=20,
        worker_heartbeat_timeout_seconds=40,
        process_worker_module="tests.support.fake_worker_main",
    )

    args = ManagedControllerProcess().serve_args(spec)

    assert args[args.index("--executor-mode") + 1] == "process"
    assert args[args.index("--runner-workers") + 1] == "8"
    assert args[args.index("--worker-queue-max") + 1] == "32"
    assert args[args.index("--worker-start-timeout-seconds") + 1] == "20.0"
    assert args[args.index("--worker-heartbeat-timeout-seconds") + 1] == "40.0"
    assert args[args.index("--process-worker-module") + 1] == "tests.support.fake_worker_main"


def test_managed_controller_args_include_parallelism() -> None:
    spec = ExperimentControllerSpec(
        local_port=8899,
        executor_mode="process",
        parallelism=120,
    )

    args = ManagedControllerProcess().serve_args(spec)

    assert args[args.index("--parallelism") + 1] == "120"


@pytest.mark.asyncio
async def test_managed_controller_rejects_preexisting_listener(tmp_path: Path) -> None:
    server = await asyncio.start_server(lambda reader, writer: writer.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            await ManagedControllerProcess().start(
                ExperimentControllerSpec(local_port=port),
                output_dir=tmp_path,
            )
    finally:
        server.close()
        await server.wait_closed()


def _experiment_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_group", required=True)
    register_experiment_commands(subparsers)
    return parser


def _case_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_group", required=True)
    register_case_commands(subparsers)
    return parser


def _controller_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_group", required=True)
    register_controller_commands(subparsers)
    return parser
