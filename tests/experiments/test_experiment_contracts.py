"""Experiment spec validation and public URL policy tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from northcua.experiments.contracts import ExperimentRunConfig, ExperimentRuntimeMode, ExperimentSpec
from northcua.experiments.materialization import (
    ExperimentMaterializer,
    RuntimeWheelBundle,
    RuntimeWheelBundleBuilder,
    load_experiment_spec,
)
from northcua.experiments.orchestration import ExperimentOperations
from northcua.experiments.orchestration.preflight import ExperimentPreflight
from northcua.experiments.planner import ExperimentBatchPlanner
from tests.support.experiment_fixtures import golden_e2b_experiment_spec, run_fixture_experiment
from northcua.sandboxes.contracts import SandboxAllocationStrategy, SandboxCommandSpec, SandboxFileUploadSpec
from northcua_protocol.capabilities import Capability


def test_bundled_example_experiment_spec_loads() -> None:
    spec = load_experiment_spec("tests/fixtures/scenarios/experiments/e2b-golden-osworld.yaml")

    assert spec.experiment_id == "e2b-golden-osworld-demo"
    assert [task.app_profile.value for task in spec.tasks] == ["chrome", "calc", "gimp"]


def test_web_agent_e2b_smoke_spec_is_single_browser_case(tmp_path: Path) -> None:
    spec = load_experiment_spec("tests/fixtures/scenarios/experiments/e2b-web-agent-evaluator-smoke.yaml")

    assert spec.experiment_id == "web-agent-e2b-smoke"
    assert spec.concurrency == 1
    assert spec.controller.parallelism == 1
    assert spec.controller.runner_workers == 1
    assert spec.controller.worker_queue_max == 2
    assert spec.sandbox.provider == "e2b"
    assert spec.sandbox.template == "fixture-web-base"
    assert spec.runtime.mode == ExperimentRuntimeMode.BAKED_IN
    assert spec.trace_expectations.evaluator_kind == "web_round_evaluator"
    assert len(spec.tasks) == 1

    task = spec.tasks[0]
    assert task.task_id == "web-info-lookup-smoke"
    assert task.agent_kind == "web"
    assert task.app_profile.value == "chrome"
    assert task.metadata["web_case_kind"] == "info_lookup"
    assert task.metadata["smoke"] is True
    assert task.metadata["expected_answer"] == "PRISM-2479"
    assert task.metadata["url"] == "http://127.0.0.1:8761/index.html"
    assert [fixture.remote_path for fixture in task.fixtures] == [
        "/tmp/northcua-web-e2b-smoke/info/index.html",
        "/tmp/northcua-web-e2b-smoke/info/license.html",
    ]
    assert task.startup_commands[0].label == "start-web-info-smoke-fixture"

    materialized = ExperimentMaterializer().materialize(
        spec,
        output_dir=tmp_path / "experiment",
        spec_dir=Path("examples/experiments"),
    )
    manifest = ExperimentBatchPlanner().plan(materialized, controller_url="wss://controller.example/ws")

    assert manifest.batch_id == "web-agent-e2b-smoke"
    assert len(manifest.tasks) == 1
    assert manifest.defaults.concurrency == 1
    assert manifest.tasks[0].agent_kind == "web"
    assert manifest.tasks[0].url == "http://127.0.0.1:8761/index.html"
    assert Capability.BROWSER_PAGE in manifest.tasks[0].required_capabilities
    assert Capability.DESKTOP_INPUT not in manifest.tasks[0].required_capabilities


def test_web_agent_e2b_three_spec_materializes_all_fixtures(tmp_path: Path) -> None:
    spec = load_experiment_spec("tests/fixtures/scenarios/experiments/e2b-web-agent-evaluator-three.yaml")

    materialized = ExperimentMaterializer().materialize(
        spec,
        output_dir=tmp_path / "experiment",
        spec_dir=Path("examples/experiments"),
    )
    manifest = ExperimentBatchPlanner().plan(materialized, controller_url="wss://controller.example/ws")

    assert spec.experiment_id == "web-agent-e2b-three"
    assert len(spec.tasks) == 3
    assert spec.concurrency == 1
    assert spec.controller.parallelism == 1
    assert spec.controller.runner_workers == 1
    assert spec.sandbox.template == "fixture-web-candidate"
    assert spec.runtime.mode == ExperimentRuntimeMode.UPLOAD_WHEELS
    assert spec.trace_expectations.evaluator_kind == "web_round_evaluator"
    assert all(task.agent_kind == "web" for task in manifest.tasks)
    assert all(task.max_retries == 1 for task in manifest.tasks)
    assert all(Capability.BROWSER_PAGE in task.required_capabilities for task in manifest.tasks)
    assert all(upload.local_path.exists() for task in materialized.tasks for upload in task.uploads)
    assert [task.task_id for task in spec.tasks] == [
        "web-info-lookup",
        "web-form-state",
        "web-tabs-navigation",
    ]


def test_public_nexau_desktop_canary_uses_synthetic_candidate_runtime(
    tmp_path: Path,
) -> None:
    spec = load_experiment_spec("tests/fixtures/scenarios/experiments/e2b-desktop-synthetic-three-public-nexau.yaml")

    materialized = ExperimentMaterializer().materialize(
        spec,
        output_dir=tmp_path / "experiment",
        spec_dir=Path("examples/experiments"),
    )

    assert spec.experiment_id == "desktop-synthetic-three-public-nexau"
    assert spec.sandbox.template == "fixture-desktop-candidate"
    assert spec.runtime.mode == ExperimentRuntimeMode.UPLOAD_WHEELS
    assert spec.controller.app_config_path == Path("config/app.desktop.e2b.yaml")
    assert spec.trace_expectations.evaluator_kind == "desktop_round_completion"
    assert spec.metadata["excludes_osworld_dataset"] is True
    assert spec.metadata["excludes_webarena_verified"] is True
    assert [task.task_id for task in spec.tasks] == [
        "desktop-chrome-heading",
        "desktop-calc-total",
        "desktop-gimp-image",
    ]
    assert all(task.agent_kind == "desktop" for task in spec.tasks)
    assert all(task.max_retries == 1 for task in spec.tasks)
    assert all(task.metadata["synthetic_case"] is True for task in spec.tasks)
    assert all(upload.local_path.exists() for task in materialized.tasks for upload in task.uploads)
    chrome, calc, gimp = spec.tasks
    assert "--disable-component-update" in chrome.startup_commands[0].command
    assert "complete main heading" in chrome.query
    assert "Do not edit or save" in calc.query
    calc_startup = calc.startup_commands[0]
    assert calc_startup.label == "open-desktop-calc-native-fixture"
    assert "--headless --convert-to ods" in calc_startup.command
    assert "-env:UserInstallation=file:///tmp/northcua-public-nexau-calc-convert-profile" in calc_startup.command
    assert "-env:UserInstallation=file:///tmp/northcua-public-nexau-calc-profile" in calc_startup.command
    assert 'pgrep -af "soffice.*calc_fixture.ods"' in calc_startup.command
    assert calc_startup.timeout_seconds == 90
    assert "Do not edit the image" in gimp.query


def test_public_nexau_web_expansion_has_five_local_fixture_cases(tmp_path: Path) -> None:
    spec = load_experiment_spec("tests/fixtures/scenarios/experiments/e2b-web-agent-evaluator-five-public-nexau.yaml")
    materialized = ExperimentMaterializer().materialize(
        spec,
        output_dir=tmp_path / "experiment",
        spec_dir=Path("examples/experiments"),
    )

    assert spec.experiment_id == "web-agent-e2b-five-public-nexau"
    assert spec.sandbox.template == "fixture-web-candidate"
    assert spec.runtime.mode == ExperimentRuntimeMode.UPLOAD_WHEELS
    assert len(spec.tasks) == 5
    assert spec.metadata["excludes_webarena_verified"] is True
    assert {task.metadata["expected_answer"] for task in spec.tasks} == {
        "PRISM-2479",
        "WEB-FORM-314",
        "09:30 UTC",
        "OK",
        "Grace Hopper",
    }
    assert all(task.metadata["synthetic_case"] is True for task in spec.tasks)
    assert all(upload.local_path.exists() for task in materialized.tasks for upload in task.uploads)


@pytest.mark.parametrize(
    ("spec_path", "experiment_id", "template", "agent_kind"),
    [
        (
            "tests/fixtures/scenarios/experiments/e2b-web-process-smoke-public-nexau.yaml",
            "web-process-smoke-public-nexau",
            "fixture-web-candidate",
            "web",
        ),
        (
            "tests/fixtures/scenarios/experiments/e2b-desktop-process-smoke-public-nexau.yaml",
            "desktop-process-smoke-public-nexau",
            "fixture-desktop-candidate",
            "desktop",
        ),
    ],
)
def test_public_nexau_process_smoke_uses_real_worker_and_candidate_runtime(
    tmp_path: Path,
    spec_path: str,
    experiment_id: str,
    template: str,
    agent_kind: str,
) -> None:
    spec = load_experiment_spec(spec_path)
    materialized = ExperimentMaterializer().materialize(
        spec,
        output_dir=tmp_path / experiment_id,
        spec_dir=Path("examples/experiments"),
    )

    assert spec.experiment_id == experiment_id
    assert spec.controller.executor_mode == "process"
    assert spec.controller.runner_workers == 1
    assert spec.controller.worker_start_timeout_seconds == 120
    assert spec.controller.worker_heartbeat_timeout_seconds == 120
    assert spec.sandbox.template == template
    assert spec.runtime.mode == ExperimentRuntimeMode.UPLOAD_WHEELS
    assert len(spec.tasks) == 1
    assert spec.tasks[0].agent_kind == agent_kind
    assert spec.tasks[0].max_retries == 1
    assert spec.metadata["process_executor_smoke"] is True
    assert all(upload.local_path.exists() for task in materialized.tasks for upload in task.uploads)


def test_web_agent_e2b_smoke_scripts_are_shell_parseable() -> None:
    for script_root in (
        Path("scripts/web_agent_e2b_smoke"),
        Path("scripts/web_agent_e2b_three"),
        Path("scripts/webharbor_e2b_three"),
        Path("scripts/webharbor_e2b_five"),
    ):
        for script in sorted(script_root.glob("*.sh")):
            subprocess.run(["bash", "-n", str(script)], check=True)


def test_experiment_rejects_duplicate_task_ids() -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    raw = spec.model_dump(mode="json")
    raw["tasks"][1]["task_id"] = raw["tasks"][0]["task_id"]

    with pytest.raises(ValueError, match="unique"):
        ExperimentSpec.model_validate(raw)


def test_real_e2b_rejects_localhost_public_controller_url() -> None:
    spec = golden_e2b_experiment_spec(public_controller_url="wss://127.0.0.1:8885", provider="e2b")

    with pytest.raises(ValueError, match="localhost"):
        ExperimentPreflight().validate_public_controller_url(spec, public_url=spec.controller.public_url)


def test_real_e2b_rejects_cleartext_public_controller_url() -> None:
    spec = golden_e2b_experiment_spec(public_controller_url="ws://controller.example/ws", provider="e2b")

    with pytest.raises(ValueError, match="wss"):
        ExperimentPreflight().validate_public_controller_url(spec, public_url=spec.controller.public_url)


def test_real_e2b_requires_controller_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NORTHCUA_CONTROLLER_TOKEN", raising=False)
    spec = golden_e2b_experiment_spec(public_controller_url="wss://controller.example/ws", provider="e2b")

    with pytest.raises(ValueError, match="token"):
        ExperimentPreflight().validate_controller_auth(spec)


def test_real_e2b_dry_run_skips_live_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NORTHCUA_ALLOW_LIVE_SANDBOXES", raising=False)
    spec = golden_e2b_experiment_spec(public_controller_url="wss://controller.example/ws", provider="e2b")

    ExperimentPreflight().validate_live_gate(spec, require_real_e2b=False)


def test_real_e2b_live_run_requires_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NORTHCUA_ALLOW_LIVE_SANDBOXES", raising=False)
    spec = golden_e2b_experiment_spec(public_controller_url="wss://controller.example/ws", provider="e2b")

    with pytest.raises(RuntimeError, match="NORTHCUA_ALLOW_LIVE_SANDBOXES"):
        ExperimentPreflight().validate_live_gate(spec, require_real_e2b=True)


def test_experiment_run_config_requires_real_e2b_by_default(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(public_controller_url="wss://controller.example/ws", provider="e2b")

    config = ExperimentRunConfig(spec=spec, output_dir=tmp_path)

    assert config.require_real_e2b is True


def test_fake_provider_allows_local_controller_url() -> None:
    spec = golden_e2b_experiment_spec(public_controller_url="ws://127.0.0.1:8885", provider="fake")

    ExperimentPreflight().validate_public_controller_url(spec, public_url=spec.controller.public_url)


def test_materializer_generates_profile_fixtures(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    materialized = ExperimentMaterializer().materialize(spec, output_dir=tmp_path / "experiment")

    uploads = [upload for task in materialized.tasks for upload in task.uploads]
    assert len(uploads) == 3
    assert all(upload.local_path.exists() for upload in uploads)
    assert all(fixture.sha256 for task in materialized.spec.tasks for fixture in task.fixtures)
    assert all(not fixture.sha256 for task in spec.tasks for fixture in task.fixtures)
    assert {upload.remote_path for upload in uploads} == {
        "/tmp/northcua-golden/chrome_fixture.html",
        "/tmp/northcua-golden/calc_fixture.csv",
        "/tmp/northcua-golden/gimp_fixture.ppm",
    }
    gimp_fixture = next(upload.local_path for upload in uploads if upload.remote_path.endswith("gimp_fixture.ppm"))
    ppm_lines = gimp_fixture.read_text(encoding="ascii").splitlines()
    assert ppm_lines[:3] == ["P3", "128 128", "255"]


def test_golden_profile_uses_unique_experiment_ids() -> None:
    first = golden_e2b_experiment_spec(provider="fake")
    second = golden_e2b_experiment_spec(provider="fake")

    assert first.experiment_id != second.experiment_id


def test_golden_profile_uses_long_step_budget_and_hardened_chrome_startup() -> None:
    spec = golden_e2b_experiment_spec(provider="fake")
    chrome = spec.tasks[0]

    assert [task.max_steps for task in spec.tasks] == [100, 100, 100]
    chrome_command = chrome.startup_commands[0].command
    assert "--user-data-dir=/tmp/northcua-golden/chrome-profile" in chrome_command
    assert "--no-first-run" in chrome_command
    assert "--no-default-browser-check" in chrome_command
    assert "--disable-sync" in chrome_command
    assert "--disable-features=SigninIntercept,ChromeWhatsNewUI,Translate" in chrome_command


def test_runtime_bundle_is_attached_before_task_startup(tmp_path: Path) -> None:
    spec = golden_e2b_experiment_spec(provider="e2b")
    materialized = ExperimentMaterializer().materialize(spec, output_dir=tmp_path / "experiment")
    bundle = RuntimeWheelBundle(
        wheel_dir=tmp_path / "wheels",
        uploads=(
            SandboxFileUploadSpec(
                local_path=tmp_path / "northcua_device-0.1.0-py3-none-any.whl",
                remote_path="/tmp/northcua-runtime-wheels/northcua_device-0.1.0-py3-none-any.whl",
            ),
        ),
        startup_commands=(SandboxCommandSpec(command="python3 -m pip install northcua-device", timeout_seconds=300),),
    )
    bundle.uploads[0].local_path.write_bytes(b"wheel")

    updated = ExperimentOperations._attach_runtime_bundle(materialized, bundle)

    assert updated.spec.sandbox.file_uploads == list(bundle.uploads)
    assert all(task.task.startup_commands[0] == bundle.startup_commands[0] for task in updated.tasks)


def test_public_nexau_web_canary_attaches_candidate_runtime_wheels(tmp_path: Path) -> None:
    spec = load_experiment_spec("tests/fixtures/scenarios/experiments/e2b-web-agent-evaluator-three.yaml")
    materialized = ExperimentMaterializer().materialize(
        spec,
        output_dir=tmp_path / "experiment",
        spec_dir=Path("examples/experiments"),
    )

    updated = ExperimentOperations()._with_runtime_bundle(materialized, dry_run=False)

    assert updated.spec.runtime.mode == ExperimentRuntimeMode.UPLOAD_WHEELS
    assert any(
        upload.remote_path.startswith("/tmp/northcua-runtime-wheels") for upload in updated.spec.sandbox.file_uploads
    )
    assert any(
        command.label == "install-northcua-runtime-wheels"
        for task in updated.tasks
        for command in task.task.startup_commands
    )


def test_runtime_bundle_install_command_uses_candidate_files():
    command = RuntimeWheelBundleBuilder()._install_command(python_executable="python3").command
    assert "--force-reinstall" in command
    assert "/tmp/northcua-runtime-wheels/*.whl" in command
    assert "--no-deps" not in command
