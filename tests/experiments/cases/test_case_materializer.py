from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import pytest

from northcua.batch.contracts import BatchManifest, BatchRunResult, BatchTaskSpec, BatchTaskState, BatchTaskStatus
from northcua.batch.preparation import PreparedSandboxPolicy
from northcua.batch.state import SQLiteBatchStateRepository
from northcua.experiments.cases.catalog import CaseCatalog
from northcua.experiments.cases.contracts import (
    AppProfileId,
    CaseExpectedFeasibility,
    CaseRunMode,
    CaseRunResult,
    CaseSpec,
    CaseSuiteSpec,
    FixtureSpec,
    GeneratedFixtureSpec,
    MaterializedFixtureReference,
    RuntimeInjectionMode,
)
from northcua.experiments.cases.loader import load_case_file, load_case_suite
from northcua.experiments.cases.materialization import (
    CaseMaterializer,
    FixtureMaterializer,
    MaterializedCase,
    MaterializedCaseSuite,
)
from northcua.experiments.cases.operations_support import (
    CaseRunPersistence,
    RuntimeBundleAttacher,
    apply_expected_feasibility_to_batch_result,
    apply_trace_gate_to_batch_result,
    rewrite_batch_summary_for_expected_feasibility,
    rewrite_batch_summary_for_trace_gate,
)
from northcua.experiments.cases.reporting import CaseReportWriter
from northcua.experiments.cases.verification import CaseArtifactResult
from northcua.experiments.materialization import RuntimeWheelBundle
from northcua.sandboxes.contracts import (
    SandboxCommandSpec,
    SandboxFileDeliveryMode,
    SandboxFileUploadSpec,
    SandboxProviderSpec,
)
from northcua_protocol.capabilities import Capability
from tests.experiments.trace_verifier_test_utils import _TINY_PNG


@pytest.fixture(autouse=True)
def scenario_catalog(monkeypatch):
    # These regression cases own their app catalog; product defaults are minimal.
    path = Path(__file__).resolve().parents[2] / "fixtures/scenarios/catalogs/e2b-case-catalog.yaml"
    monkeypatch.setattr("northcua.experiments.cases.catalog.loader.default_case_catalog_path", lambda: path)


def test_materializer_projects_static_live_five_suite(tmp_path: Path) -> None:
    suite = load_case_suite("tests/fixtures/scenarios/experiments/e2b-live-five.yaml")

    materialized = CaseMaterializer().materialize(
        suite,
        output_dir=tmp_path,
        controller_url="ws://controller.example/ws",
    )

    assert materialized.manifest.batch_id == "e2b-live-five"
    assert len(materialized.cases) == 5
    templates = {task.task_id: task.sandbox.template for task in materialized.manifest.tasks if task.sandbox}
    assert templates["live-five-browser-eligibility"] == "northcua-e2b-office-web"
    assert templates["live-five-gimp-undo"] == "northcua-e2b-creative"
    assert templates["live-five-vscode-workspace"] == "northcua-e2b-dev"
    assert templates["live-five-contact-sheet"] == "northcua-e2b-desktop-full"
    calc = next(task for task in materialized.manifest.tasks if task.task_id == "live-five-calc-pivot")
    assert calc.sandbox is not None
    assert calc.sandbox.file_uploads[0].sha256
    assert calc.metadata["fixture_sha256"]
    assert "large_artifact_delivery_evidence" not in calc.metadata
    assert "intended_app" not in calc.sandbox.file_uploads[0].model_dump(mode="json")
    assert "host_verification" not in calc.sandbox.file_uploads[0].model_dump(mode="json")
    assert calc.metadata["controller_app_config_path"] == suite.controller.app_config_path
    assert "northcua.desktop.coordinate_space" not in calc.metadata
    assert "northcua.desktop.coordinate_clamp_to_bounds" not in calc.metadata
    assert calc.startup_commands
    assert calc.max_retries == 2


def test_case_run_state_counts_evaluation_pending_as_terminal_failure(tmp_path: Path) -> None:
    states = [
        BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED),
        BatchTaskState(task_id="case-2", status=BatchTaskStatus.EVALUATION_PENDING, evaluation_status="pending"),
    ]

    result = CaseRunPersistence.batch_result_from_states(batch_id="batch", batch_dir=tmp_path, states=states)

    assert result.total == 2
    assert result.succeeded == 1
    assert result.failed == 1
    assert result.incomplete == 0


def test_case_suite_loads_directory_file_and_jsonl_sources(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases"
    fixture_dir = case_dir / "fixtures"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "input.txt").write_text("hello", encoding="utf-8")
    (case_dir / "01-browser.case.yaml").write_text(
        """
task_id: source-browser
query: Open it.
app_profile: browser.chrome
template_profile: office-web
init:
  - recipe: open_url
    args:
      app: browser.chrome
      url: https://example.com
""".strip(),
        encoding="utf-8",
    )
    single = tmp_path / "single.case.yaml"
    single.write_text(
        """
version: v1
case:
  task_id: source-single-file
  query: Edit the file.
  app_profile: libreoffice.writer
  template_profile: office-web
  fixtures:
    - source: cases/fixtures/input.txt
      remote_path: /tmp/northcua-cases/source-single-file/input.txt
      role: input
metadata:
  owner: suite-author
""".strip(),
        encoding="utf-8",
    )
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "task_id": "source-jsonl",
                "query": "Open blank app.",
                "app_profile": "creative.gimp",
                "template_profile": "creative",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
version: v1
suite_id: source-suite
case_sources:
  - kind: directory
    path: cases
    glob: "*.case.yaml"
  - kind: file
    path: single.case.yaml
  - kind: jsonl
    path: cases.jsonl
""".strip(),
        encoding="utf-8",
    )

    suite = load_case_suite(suite_path)

    assert [task.task_id for task in suite.tasks] == ["source-browser", "source-single-file", "source-jsonl"]
    assert suite.tasks[1].fixtures[0].source_path == fixture_dir / "input.txt"
    assert suite.tasks[0].metadata["case_file"].endswith("01-browser.case.yaml")
    assert suite.tasks[1].metadata["owner"] == "suite-author"
    assert suite.tasks[2].metadata["case_jsonl_line"] == 1


def test_case_suite_id_defaults_to_spec_file_stem(tmp_path: Path) -> None:
    suite_path = tmp_path / "my suite.yaml"
    suite_path.write_text(
        """
version: v1
tasks:
  - task_id: browser
    query: Open it.
    app_profile: browser.chrome
""".strip(),
        encoding="utf-8",
    )

    first = load_case_suite(suite_path)
    second = load_case_suite(suite_path)

    assert first.suite_id == "my-suite"
    assert second.suite_id == first.suite_id


def test_case_file_resolves_dependency_injection_file_paths(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    injected = fixtures / "seed.txt"
    injected.write_text("seed", encoding="utf-8")
    case_path = tmp_path / "single.case.yaml"
    case_path.write_text(
        """
task_id: injected-file
query: Use injected file.
app_profile: desktop.text_editor
template_profile: desktop-full
dependency_injection:
  files:
    - source: fixtures/seed.txt
      remote_path: /tmp/northcua-cases/injected-file/seed.txt
""".strip(),
        encoding="utf-8",
    )

    case = load_case_file(case_path)

    assert case.dependency_injection.files[0].source_path == injected


def test_materializer_adds_evaluator_live_probe_capabilities(tmp_path: Path) -> None:
    app_config = tmp_path / "app.yaml"
    app_config.write_text(
        """
agent_config_path: agent.example.yaml
evaluation:
  enabled: true
  tools:
    allow_client_live_probe: true
    client_probe_tools:
      - client_bash_readonly
""",
        encoding="utf-8",
    )
    catalog = CaseCatalog.from_raw(
        {
            "templates": [{"profile_id": "minimal.desktop", "alias": "minimal-template", "provider": "fake"}],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "minimal.desktop",
                    "display_name": "Custom",
                    "launch_binary": "true",
                }
            ],
        }
    )
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "probe-caps",
            "controller": {"app_config_path": str(app_config)},
            "sandbox": {"provider": "fake"},
            "tasks": [
                {
                    "task_id": "probe",
                    "query": "Inspect the completed task.",
                    "app_profile": "custom.app",
                    "template_profile": "minimal.desktop",
                }
            ],
        }
    )

    materialized = CaseMaterializer(catalog=catalog).materialize(suite, output_dir=tmp_path)

    assert Capability.SHELL_EXEC in materialized.manifest.tasks[0].required_capabilities


def test_materializer_projects_pre_submit_commands(tmp_path: Path) -> None:
    catalog = CaseCatalog.from_raw(
        {
            "templates": [{"profile_id": "minimal.desktop", "alias": "minimal-template", "provider": "fake"}],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "minimal.desktop",
                    "display_name": "Custom",
                    "launch_binary": "true",
                }
            ],
        }
    )
    command = SandboxCommandSpec(command="/tmp/focus-browser", label="focus-browser", allow_failure=True)
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "pre-submit-suite",
            "tasks": [
                {
                    "task_id": "focus-case",
                    "query": "Use the web app.",
                    "app_profile": "custom.app",
                    "template_profile": "minimal.desktop",
                    "pre_submit_commands": [command.model_dump(mode="json")],
                }
            ],
        }
    )

    materialized = CaseMaterializer(catalog=catalog).materialize(suite, output_dir=tmp_path)
    task = materialized.manifest.tasks[0]

    assert materialized.cases[0].pre_submit_commands == (command,)
    assert task.pre_submit_commands == [command]
    assert [item.label for item in task.startup_commands] == ["desktop-session-readiness"]
    assert materialized.cases[0].case.desktop_session is not None
    assert materialized.cases[0].case.desktop_session.required


def test_materializer_derives_desktop_session_only_for_desktop_workflows(tmp_path: Path) -> None:
    catalog = CaseCatalog.from_raw(
        {
            "templates": [
                {
                    "profile_id": "session.desktop",
                    "alias": "session-template",
                    "provider": "fake",
                    "desktop_session": {"target_user": "catalog-user", "readiness_timeout_seconds": 25},
                }
            ],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "session.desktop",
                    "display_name": "Custom",
                    "launch_binary": "true",
                }
            ],
        }
    )
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "desktop-session-workflows",
            "tasks": [
                {"task_id": "desktop", "query": "Use desktop.", "app_profile": "custom.app"},
                {"task_id": "web", "query": "Use web.", "app_profile": "custom.app", "agent_kind": "web"},
                {
                    "task_id": "web-explicit",
                    "query": "Use web with a desktop dependency.",
                    "app_profile": "custom.app",
                    "agent_kind": "web",
                    "desktop_session": {"target_user": "case-user"},
                },
            ],
        }
    )

    materialized = CaseMaterializer(catalog=catalog).materialize(suite, output_dir=tmp_path)
    desktop, web, web_explicit = materialized.cases

    assert desktop.case.desktop_session is not None
    assert desktop.case.desktop_session.target_user == "catalog-user"
    assert [command.label for command in desktop.startup_commands] == ["desktop-session-readiness"]
    assert web.case.desktop_session is None
    assert web.startup_commands == ()
    assert web_explicit.case.desktop_session is not None
    assert web_explicit.case.desktop_session.target_user == "case-user"
    assert materialized.manifest.tasks[0].metadata["desktop_session"]["required"] is True
    assert "desktop_session" not in materialized.manifest.tasks[1].metadata


def test_materializer_infers_and_validates_desktop_session_launch_user(tmp_path: Path) -> None:
    catalog = CaseCatalog.from_raw(
        {
            "templates": [{"profile_id": "session.desktop", "alias": "session-template", "provider": "fake"}],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "session.desktop",
                    "display_name": "Custom",
                    "launch_binary": "true",
                    "launch_user": "desktop-user",
                }
            ],
        }
    )
    inferred = CaseSuiteSpec.model_validate(
        {"suite_id": "inferred", "tasks": [{"task_id": "desktop", "query": "Use it.", "app_profile": "custom.app"}]}
    )

    materialized = CaseMaterializer(catalog=catalog).materialize(inferred, output_dir=tmp_path)

    assert materialized.cases[0].case.desktop_session is not None
    assert materialized.cases[0].case.desktop_session.target_user == "desktop-user"

    conflicting = CaseSuiteSpec.model_validate(
        {
            "suite_id": "conflicting",
            "tasks": [
                {
                    "task_id": "desktop",
                    "query": "Use it.",
                    "app_profile": "custom.app",
                    "desktop_session": {"target_user": "other-user"},
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="conflicts with app launch_user"):
        CaseMaterializer(catalog=catalog).materialize(conflicting, output_dir=tmp_path)


def test_materializer_rejects_required_session_on_non_desktop_template(tmp_path: Path) -> None:
    catalog = CaseCatalog.from_raw(
        {
            "templates": [
                {"profile_id": "headless", "alias": "headless-template", "provider": "fake", "desktop": False}
            ],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "headless",
                    "display_name": "Custom",
                    "launch_binary": "true",
                }
            ],
        }
    )
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "headless-session",
            "tasks": [
                {
                    "task_id": "headless",
                    "query": "Use it.",
                    "app_profile": "custom.app",
                    "agent_kind": "web",
                    "desktop_session": {"required": True},
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="is not a desktop template"):
        CaseMaterializer(catalog=catalog).materialize(suite, output_dir=tmp_path)


def test_materializer_projects_legacy_guest_ownership_into_guarded_uploads(tmp_path: Path) -> None:
    fixture = tmp_path / "input.txt"
    fixture.write_text("hello", encoding="utf-8")
    hidden_fixture = tmp_path / "expected.txt"
    hidden_fixture.write_text("hidden", encoding="utf-8")
    explicit_fixture = tmp_path / "explicit.txt"
    explicit_fixture.write_text("explicit", encoding="utf-8")
    catalog = CaseCatalog.from_raw(
        {
            "templates": [{"profile_id": "minimal.desktop", "alias": "minimal-template", "provider": "fake"}],
            "apps": [
                {
                    "app_profile": "custom.editor",
                    "template_profile": "minimal.desktop",
                    "display_name": "Custom Editor",
                    "launch_binary": "custom-editor",
                    "launch_user": "user",
                    "window_hint": "Custom Editor",
                }
            ],
        }
    )
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "fixture-ownership",
            "tasks": [
                {
                    "task_id": "edit-input",
                    "query": "Edit the file.",
                    "app_profile": "custom.editor",
                    "template_profile": "minimal.desktop",
                    "fixtures": [
                        {
                            "source": str(fixture),
                            "remote_path": "/tmp/northcua-cases/edit-input/input.txt",
                            "role": "input",
                        },
                        {
                            "source": str(hidden_fixture),
                            "remote_path": "/tmp/northcua-cases/edit-input/expected.txt",
                            "role": "expected",
                            "visibility": "evaluator",
                        },
                        {
                            "source": str(explicit_fixture),
                            "remote_path": "/tmp/northcua-cases/edit-input/explicit.txt",
                            "owner": "root",
                            "group": "root",
                            "parent_owner": "root",
                            "parent_group": "root",
                        },
                    ],
                    "init": [
                        {
                            "recipe": "open_file",
                            "args": {
                                "app": "custom.editor",
                                "path": "/tmp/northcua-cases/edit-input/input.txt",
                            },
                        }
                    ],
                }
            ],
        }
    )

    materialized = CaseMaterializer(catalog=catalog).materialize(suite, output_dir=tmp_path)
    startup = materialized.manifest.tasks[0].startup_commands
    uploads = {upload.remote_path: upload for upload in materialized.cases[0].sandbox.file_uploads}

    assert set(uploads) == {
        "/tmp/northcua-cases/edit-input/input.txt",
        "/tmp/northcua-cases/edit-input/explicit.txt",
    }
    assert uploads["/tmp/northcua-cases/edit-input/input.txt"].owner == "user"
    assert uploads["/tmp/northcua-cases/edit-input/input.txt"].group == "user"
    assert uploads["/tmp/northcua-cases/edit-input/input.txt"].parent_owner == "user"
    assert uploads["/tmp/northcua-cases/edit-input/input.txt"].parent_group == "user"
    assert uploads["/tmp/northcua-cases/edit-input/explicit.txt"].owner == "root"
    assert uploads["/tmp/northcua-cases/edit-input/explicit.txt"].group == "root"
    assert uploads["/tmp/northcua-cases/edit-input/explicit.txt"].parent_owner == "root"
    assert uploads["/tmp/northcua-cases/edit-input/explicit.txt"].parent_group == "root"
    assert startup[0].label == "desktop-session-readiness"
    assert startup[1].label == "open-file-custom.editor"
    assert all(
        command.label != "prepare-user-fixtures"
        for phase in materialized.cases[0].setup_phases
        for command in phase.commands
    )


def test_case_suite_runtime_injection_mode_is_typed() -> None:
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "runtime-upload",
            "runtime": {"mode": "upload_wheels"},
            "tasks": [{"task_id": "browser", "query": "Open it", "app_profile": "browser.chrome"}],
        }
    )

    assert suite.runtime.mode == RuntimeInjectionMode.UPLOAD_WHEELS


def test_runtime_bundle_upload_injection_updates_each_task(tmp_path: Path) -> None:
    suite = load_case_suite("tests/fixtures/scenarios/experiments/e2b-live-five.yaml")
    materialized = CaseMaterializer().materialize(
        suite,
        output_dir=tmp_path,
        controller_url="wss://controller.example/ws",
    )
    wheel = tmp_path / "northcua-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    bundle = RuntimeWheelBundle(
        wheel_dir=tmp_path,
        uploads=(SandboxFileUploadSpec(local_path=wheel, remote_path="/tmp/northcua-runtime-wheels/runtime.whl"),),
        startup_commands=(
            SandboxCommandSpec(
                command="python3 -m pip install /tmp/northcua-runtime-wheels/runtime.whl", label="install"
            ),
        ),
    )

    injected = RuntimeBundleAttacher._attach_runtime_bundles(materialized, {"python3.12": bundle})

    for case, task in zip(injected.cases, injected.manifest.tasks, strict=True):
        assert case.sandbox.file_uploads[-1].remote_path == "/tmp/northcua-runtime-wheels/runtime.whl"
        assert task.sandbox is not None
        assert task.sandbox.file_uploads[-1].remote_path == "/tmp/northcua-runtime-wheels/runtime.whl"
        assert task.startup_commands[0].label == "install"


def test_materializer_rejects_app_not_installed_in_template(tmp_path: Path) -> None:
    catalog = CaseCatalog.from_raw(
        {
            "templates": [
                {
                    "profile_id": "minimal.desktop",
                    "alias": "minimal-template",
                    "provider": "fake",
                    "installed_apps": ["browser.chrome"],
                    "required_binaries": ["google-chrome"],
                }
            ],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "minimal.desktop",
                    "display_name": "Custom",
                    "launch_binary": "custom-app",
                    "required_binaries": ["custom-app"],
                }
            ],
        }
    )
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "catalog-mismatch",
            "sandbox": {"provider": "fake"},
            "tasks": [
                {
                    "task_id": "bad",
                    "query": "Open custom app.",
                    "app_profile": "custom.app",
                    "template_profile": "minimal.desktop",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="is not installed"):
        CaseMaterializer(catalog=catalog).materialize(suite, output_dir=tmp_path)


def test_materializer_only_projects_guest_visible_fixtures_to_uploads(tmp_path: Path) -> None:
    sources = {}
    for visibility in ("guest", "evaluator", "host"):
        source = tmp_path / f"{visibility}.txt"
        source.write_text(visibility, encoding="utf-8")
        sources[visibility] = source
    catalog = CaseCatalog.from_raw(
        {
            "templates": [{"profile_id": "minimal.desktop", "alias": "minimal-template", "provider": "fake"}],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "minimal.desktop",
                    "display_name": "Custom",
                    "launch_binary": "true",
                }
            ],
        }
    )
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "fixture-visibility",
            "tasks": [
                {
                    "task_id": "visible-fixtures",
                    "query": "Use the guest fixture.",
                    "app_profile": "custom.app",
                    "template_profile": "minimal.desktop",
                    "fixtures": [
                        {
                            "source": str(source),
                            "remote_path": f"/tmp/northcua-cases/visible-fixtures/{visibility}.txt",
                            "visibility": visibility,
                            "role": "expected" if visibility == "evaluator" else "input",
                        }
                        for visibility, source in sources.items()
                    ],
                }
            ],
        }
    )

    materialized = CaseMaterializer(catalog=catalog).materialize(suite, output_dir=tmp_path)
    case = materialized.cases[0]
    task = materialized.manifest.tasks[0]

    assert len(case.fixtures) == 3
    assert [upload.remote_path for upload in case.sandbox.file_uploads] == [
        "/tmp/northcua-cases/visible-fixtures/guest.txt"
    ]
    assert task.metadata["fixture_remote_paths"] == ["/tmp/northcua-cases/visible-fixtures/guest.txt"]
    references = [
        MaterializedFixtureReference.model_validate(item) for item in task.metadata["materialized_fixture_references"]
    ]
    assert [reference.visibility.value for reference in references] == ["evaluator", "host"]
    assert references[0].role.value == "expected"
    assert references[0].local_path == sources["evaluator"].resolve()
    assert all(reference.remote_path not in task.metadata["fixture_remote_paths"] for reference in references)
    assert case.consumer_fixture_references == tuple(references)


def test_materializer_resolves_task_setup_binding_and_all_app_requirements(tmp_path: Path) -> None:
    catalog = CaseCatalog.from_raw(
        {
            "templates": [
                {
                    "profile_id": "minimal.desktop",
                    "alias": "minimal-template",
                    "provider": "fake",
                    "required_capabilities": ["desktop.input"],
                    "required_guest_capabilities": ["desktop"],
                    "app_bindings": [
                        {
                            "app_profile": "custom.editor",
                            "provisioning_mode": "task_setup",
                            "first_run_variant": "osworld2",
                            "post_setup": {
                                "binary_probes": ["custom-editor"],
                                "version_probes": ["custom-editor --version"],
                            },
                            "required_guest_capabilities": ["shell"],
                        },
                        {"app_profile": "custom.helper", "provisioning_mode": "baked"},
                        {"app_profile": "custom.launcher", "provisioning_mode": "baked"},
                    ],
                }
            ],
            "apps": [
                {
                    "app_profile": "custom.editor",
                    "template_profile": "minimal.desktop",
                    "display_name": "Editor",
                    "launch_binary": "custom-editor",
                    "required_binaries": ["custom-editor"],
                    "required_capabilities": ["desktop.window"],
                    "required_guest_capabilities": ["network"],
                },
                {
                    "app_profile": "custom.helper",
                    "template_profile": "minimal.desktop",
                    "display_name": "Helper",
                    "launch_binary": "true",
                    "required_capabilities": ["a11y.tree"],
                    "required_guest_capabilities": ["desktop.audio"],
                },
                {
                    "app_profile": "custom.launcher",
                    "template_profile": "minimal.desktop",
                    "display_name": "Launcher",
                    "launch_binary": "true",
                    "required_capabilities": ["shell.exec"],
                    "required_guest_capabilities": ["desktop.clipboard"],
                },
            ],
        }
    )
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "task-setup-binding",
            "tasks": [
                {
                    "task_id": "task-setup",
                    "query": "Use all apps.",
                    "app_profile": "custom.editor",
                    "template_profile": "minimal.desktop",
                    "involved_app_profiles": ["custom.editor", "custom.helper"],
                    "required_capabilities": ["browser.page"],
                    "required_guest_capabilities": ["internet"],
                    "fixtures": [
                        {
                            "generated": {"generator": "setup.script", "params": {"commands": ["true"]}},
                            "remote_path": "/tmp/northcua-cases/task-setup/install.sh",
                            "mode": "0755",
                            "role": "setup_script",
                        }
                    ],
                    "dependency_injection": {
                        "setup_recipes": [
                            {
                                "recipe": "run_setup_script",
                                "args": {"path": "/tmp/northcua-cases/task-setup/install.sh"},
                            }
                        ]
                    },
                    "init": [{"recipe": "open_blank_app", "args": {"app": "custom.launcher"}}],
                }
            ],
        }
    )

    materialized = CaseMaterializer(catalog=catalog).materialize(suite, output_dir=tmp_path)
    case = materialized.cases[0]
    task = materialized.manifest.tasks[0]

    assert {capability.value for capability in case.case.required_guest_capabilities} == {
        "desktop",
        "desktop.audio",
        "desktop.clipboard",
        "internet",
        "network",
        "shell",
    }
    assert {
        Capability.DESKTOP_INPUT,
        Capability.DESKTOP_WINDOW,
        Capability.A11Y_TREE,
        Capability.SHELL_EXEC,
        Capability.BROWSER_PAGE,
    }.issubset(task.required_capabilities)
    assert [command.label for command in task.startup_commands[:6]] == [
        "guest-capability-gate",
        "run-setup-script",
        "post-setup-binary-custom.editor",
        "post-setup-version-custom.editor",
        "desktop-session-readiness",
        "open-blank-custom.launcher",
    ]
    assert task.metadata["template_app_bindings"]["custom.editor"] == {
        "provisioning_mode": "task_setup",
        "first_run_variant": "osworld2",
    }


def test_materializer_requires_binding_for_new_template_app_contract(tmp_path: Path) -> None:
    catalog = CaseCatalog.from_raw(
        {
            "templates": [
                {
                    "profile_id": "minimal.desktop",
                    "alias": "minimal-template",
                    "provider": "fake",
                    "app_bindings": [{"app_profile": "other.app", "provisioning_mode": "task_setup"}],
                }
            ],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "minimal.desktop",
                    "display_name": "Custom",
                    "launch_binary": "custom",
                },
                {
                    "app_profile": "other.app",
                    "template_profile": "minimal.desktop",
                    "display_name": "Other",
                    "launch_binary": "other",
                },
            ],
        }
    )
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "missing-binding",
            "tasks": [
                {
                    "task_id": "missing",
                    "query": "Open custom app.",
                    "app_profile": "custom.app",
                    "template_profile": "minimal.desktop",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="has no app binding"):
        CaseMaterializer(catalog=catalog).materialize(suite, output_dir=tmp_path)


def test_materializer_requires_restricted_setup_phase_for_task_setup_binding(tmp_path: Path) -> None:
    catalog = CaseCatalog.from_raw(
        {
            "templates": [
                {
                    "profile_id": "task.template",
                    "alias": "task-template",
                    "provider": "fake",
                    "app_bindings": [{"app_profile": "custom.app", "provisioning_mode": "task_setup"}],
                }
            ],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "task.template",
                    "display_name": "Custom",
                    "launch_binary": "custom",
                }
            ],
        }
    )
    base_task = {
        "task_id": "task-setup",
        "query": "Open custom.",
        "app_profile": "custom.app",
        "template_profile": "task.template",
    }

    no_setup = CaseSuiteSpec.model_validate({"suite_id": "no-setup", "tasks": [base_task]})
    with pytest.raises(ValueError, match="actual dependency_injection.setup_recipes phase"):
        CaseMaterializer(catalog=catalog).materialize(no_setup, output_dir=tmp_path)

    unrestricted = CaseSuiteSpec.model_validate(
        {
            "suite_id": "unrestricted-setup",
            "tasks": [
                {
                    **base_task,
                    "dependency_injection": {
                        "setup_recipes": [{"recipe": "open_blank_app", "args": {"app": "custom.app"}}]
                    },
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="only allows run_setup_script"):
        CaseMaterializer(catalog=catalog).materialize(unrestricted, output_dir=tmp_path)


def test_materializer_keeps_legacy_app_without_explicit_binding_compatible(tmp_path: Path) -> None:
    catalog = CaseCatalog.from_raw(
        {
            "templates": [{"profile_id": "legacy.template", "alias": "legacy", "provider": "fake"}],
            "apps": [
                {
                    "app_profile": "legacy.app",
                    "template_profile": "legacy.template",
                    "display_name": "Legacy",
                    "launch_binary": "true",
                }
            ],
        }
    )
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "legacy",
            "tasks": [
                {
                    "task_id": "legacy",
                    "query": "Open legacy.",
                    "app_profile": "legacy.app",
                    "template_profile": "legacy.template",
                }
            ],
        }
    )

    materialized = CaseMaterializer(catalog=catalog).materialize(suite, output_dir=tmp_path)

    assert materialized.manifest.tasks[0].metadata["template_app_bindings"] == {}


def test_materializer_uses_run_id_in_sandbox_metadata(tmp_path: Path) -> None:
    suite = load_case_suite("tests/fixtures/scenarios/experiments/e2b-live-five.yaml")

    materialized = CaseMaterializer().materialize(
        suite,
        output_dir=tmp_path,
        controller_url="ws://controller.example/ws",
        run_id="run-123",
    )

    assert materialized.manifest.batch_id == "run-123"
    sandbox = materialized.manifest.tasks[0].sandbox
    assert sandbox is not None
    assert sandbox.metadata["case_suite_id"] == "e2b-live-five"
    assert sandbox.metadata["batch_id"] == "run-123"


def test_preflight_effective_providers_ignores_unused_default_provider(tmp_path: Path) -> None:
    from northcua.experiments.cases.runtime import CaseRunPreflight

    suite = load_case_suite("tests/fixtures/scenarios/experiments/e2b-live-five.yaml")
    materialized = CaseMaterializer().materialize(suite, output_dir=tmp_path, run_id="providers")
    default_sandbox = materialized.manifest.defaults.sandbox.model_copy(update={"provider": "e2b"})
    manifest = materialized.manifest.model_copy(
        update={"defaults": materialized.manifest.defaults.model_copy(update={"sandbox": default_sandbox})}
    )
    fake_tasks = [
        task.model_copy(update={"sandbox": task.sandbox.model_copy(update={"provider": "fake"})})
        for task in materialized.manifest.tasks
        if task.sandbox is not None
    ]
    manifest = manifest.model_copy(update={"tasks": fake_tasks})

    assert CaseRunPreflight.effective_providers(manifest) == {"fake"}
    assert CaseRunPreflight.has_real_provider(manifest) is False


def test_materializer_persists_case_outputs(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    materialized = CaseOperations().materialize_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="ws://controller.example/ws",
    )

    assert (materialized.output_dir / "case_suite.json").is_file()
    assert (materialized.output_dir / "batch_manifest.json").is_file()
    assert '"token"' in (materialized.output_dir / "case_suite.json").read_text(encoding="utf-8")
    assert '"token": ""' in (materialized.output_dir / "batch_manifest.json").read_text(encoding="utf-8")


def test_case_trace_verifier_falls_back_to_actual_evaluator_kind(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
    from northcua.experiments.cases.verification import CaseTraceVerifier

    state = BatchTaskState(
        task_id="case-1",
        controller_task_id="controller-case-1",
        trace_id="trace-case-1",
        session_id="session-case-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    spec = CaseSuiteSpec.model_validate(
        {
            "suite_id": "trace-fallback",
            "trace_expectations": {"evaluator_kind": "desktop"},
            "tasks": [{"task_id": "case-1", "query": "Do it", "app_profile": "browser.chrome"}],
        }
    )
    session_root = tmp_path / "agent-artifacts" / state.session_id
    eval_root = session_root / "evaluations" / "desktop_round_completion" / "round_0001"
    (session_root / "query_0001").mkdir(parents=True)
    eval_root.mkdir(parents=True)
    (session_root / "steps.jsonl").write_text('{"session_step":1}\n', encoding="utf-8")
    (session_root / "steps.jsonl").write_text(
        json.dumps(
            {
                "session_step": 1,
                "observation_metadata": {
                    "round_end_verification_status": "success",
                    "round_end_verification_screenshot_path": str(
                        session_root / "query_0001" / "action_step1_verify.png"
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (session_root / "llm_trace.jsonl").write_text('{"trace":"main"}\n', encoding="utf-8")
    (session_root / "query_0001" / "action_step1_history_overlay_1.png").write_bytes(_TINY_PNG)
    (session_root / "query_0001" / "action_step1_verify.png").write_bytes(_TINY_PNG)
    (eval_root / "final_report.json").write_text('{"overall_verdict":"success"}', encoding="utf-8")
    (eval_root / "evidence_manifest.json").write_text('{"refs":[]}', encoding="utf-8")
    (eval_root / "llm_trace.jsonl").write_text('{"trace":"eval"}\n', encoding="utf-8")

    rows = CaseTraceVerifier().verify(
        output_dir=tmp_path,
        spec=spec,
        batch_result=BatchRunResult(
            batch_id="trace-fallback",
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(tmp_path),
            task_states=[state],
        ),
    )

    assert rows[0].passed


def test_case_trace_verifier_skips_unattempted_pending_artifact_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
    from northcua.experiments.cases.verification import CaseTraceVerifier

    state = BatchTaskState(task_id="case-1", status=BatchTaskStatus.PENDING, attempts=0)
    spec = CaseSuiteSpec.model_validate(
        {
            "suite_id": "trace-pending",
            "tasks": [{"task_id": "case-1", "query": "Do it", "app_profile": "browser.chrome"}],
        }
    )

    def _fail_scan(*args: object, **kwargs: object) -> list[Path]:
        raise AssertionError("pending unattempted case should not scan artifacts")

    monkeypatch.setattr(CaseTraceVerifier, "_candidates", staticmethod(_fail_scan))

    rows = CaseTraceVerifier().verify(
        output_dir=tmp_path,
        spec=spec,
        batch_result=BatchRunResult(
            batch_id="trace-pending",
            total=1,
            succeeded=0,
            failed=0,
            canceled=0,
            incomplete=1,
            output_dir=str(tmp_path),
            task_states=[state],
        ),
    )

    assert rows[0].status == "pending"
    assert rows[0].errors == ["batch task not attempted"]


def test_case_trace_verifier_requires_terminal_verification_screenshot(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
    from northcua.experiments.cases.verification import CaseTraceVerifier

    state = BatchTaskState(
        task_id="case-1",
        controller_task_id="controller-case-1",
        trace_id="trace-case-1",
        session_id="session-case-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    spec = CaseSuiteSpec.model_validate(
        {
            "suite_id": "trace-terminal-shot",
            "trace_expectations": {"evaluator_kind": "desktop"},
            "tasks": [{"task_id": "case-1", "query": "Do it", "app_profile": "browser.chrome"}],
        }
    )
    session_root = tmp_path / "agent-artifacts" / state.session_id
    eval_root = session_root / "evaluations" / "desktop_round_completion" / "round_0001"
    (session_root / "query_0001").mkdir(parents=True)
    eval_root.mkdir(parents=True)
    (session_root / "steps.jsonl").write_text('{"session_step":1}\n', encoding="utf-8")
    (session_root / "llm_trace.jsonl").write_text('{"trace":"main"}\n', encoding="utf-8")
    (session_root / "query_0001" / "action_step1_history_overlay_1.png").write_bytes(_TINY_PNG)
    (eval_root / "final_report.json").write_text('{"overall_verdict":"success"}', encoding="utf-8")
    (eval_root / "evidence_manifest.json").write_text('{"refs":[]}', encoding="utf-8")
    (eval_root / "llm_trace.jsonl").write_text('{"trace":"eval"}\n', encoding="utf-8")

    rows = CaseTraceVerifier().verify(
        output_dir=tmp_path,
        spec=spec,
        batch_result=BatchRunResult(
            batch_id="trace-terminal-shot",
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(tmp_path),
            task_states=[state],
        ),
    )

    assert not rows[0].passed
    assert "terminal verification screenshot missing" in rows[0].errors


def test_case_trace_verifier_accepts_relocated_terminal_verification_screenshot(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
    from northcua.experiments.cases.verification import CaseTraceVerifier

    state = BatchTaskState(
        task_id="case-1",
        controller_task_id="controller-case-1",
        trace_id="trace-case-1",
        session_id="session-case-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    spec = CaseSuiteSpec.model_validate(
        {
            "suite_id": "trace-relocated-terminal-shot",
            "trace_expectations": {"evaluator_kind": "desktop"},
            "tasks": [{"task_id": "case-1", "query": "Do it", "app_profile": "browser.chrome"}],
        }
    )
    session_root = tmp_path / "agent-artifacts" / state.session_id
    eval_root = session_root / "evaluations" / "desktop_round_completion" / "round_0001"
    (session_root / "query_0001").mkdir(parents=True)
    eval_root.mkdir(parents=True)
    local_verify = session_root / "query_0001" / "action_step7_verify.png"
    (session_root / "steps.jsonl").write_text(
        json.dumps(
            {
                "session_step": 7,
                "observation_metadata": {
                    "round_end_verification_status": "success",
                    "round_end_verification_screenshot_path": "/remote/session/query_0001/action_step7_verify.png",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (session_root / "llm_trace.jsonl").write_text('{"trace":"main"}\n', encoding="utf-8")
    (session_root / "query_0001" / "action_step7_history_overlay_1.png").write_bytes(_TINY_PNG)
    local_verify.write_bytes(_TINY_PNG)
    (eval_root / "final_report.json").write_text('{"overall_verdict":"success"}', encoding="utf-8")
    (eval_root / "evidence_manifest.json").write_text('{"refs":[]}', encoding="utf-8")
    (eval_root / "llm_trace.jsonl").write_text('{"trace":"eval"}\n', encoding="utf-8")

    rows = CaseTraceVerifier().verify(
        output_dir=tmp_path,
        spec=spec,
        batch_result=BatchRunResult(
            batch_id="trace-relocated-terminal-shot",
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(tmp_path),
            task_states=[state],
        ),
    )

    assert rows[0].passed
    assert str(local_verify) in rows[0].screenshot_paths
    assert rows[0].final_verify_screenshot_path == str(local_verify)


def test_case_report_includes_retry_breakdown_and_final_verify_path(tmp_path: Path) -> None:
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "report-retry-paths",
            "tasks": [{"task_id": "case-1", "query": "Do it", "app_profile": "browser.chrome"}],
        }
    )
    case = suite.tasks[0]
    sandbox = SandboxProviderSpec(template="template")
    materialized = MaterializedCaseSuite(
        spec=suite,
        output_dir=tmp_path,
        fixture_dir=tmp_path / "fixtures",
        cases=(MaterializedCase(case=case, fixtures=(), startup_commands=(), sandbox=sandbox),),
        manifest=BatchManifest(
            batch_id="report-retry-paths",
            tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", request="Do it", sandbox=sandbox)],
        ),
    )
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (batch_dir / "case_metrics.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "attempt_failure_category_breakdown": {"device_transport": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    verify_path = tmp_path / "agent-artifacts" / "session-1" / "query_0001" / "action_step3_verify.png"
    verify_path.parent.mkdir(parents=True)
    verify_path.write_bytes(_TINY_PNG)

    report_path = CaseReportWriter(tmp_path).write(
        materialized=materialized,
        batch_result=BatchRunResult(
            batch_id="report-retry-paths",
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(batch_dir),
            task_states=[
                BatchTaskState(
                    task_id="case-1",
                    status=BatchTaskStatus.SUCCEEDED,
                    attempts=2,
                    session_id="session-1",
                )
            ],
        ),
        artifact_results=[
            CaseArtifactResult(
                task_id="case-1",
                app_profile=case.app_profile,
                status="succeeded",
                session_id="session-1",
                screenshot_paths=[str(verify_path)],
                final_verify_screenshot_path=str(verify_path),
                passed=True,
            )
        ],
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    row = report["cases"][0]
    summary = json.loads((tmp_path / "case_report.summary.json").read_text(encoding="utf-8"))

    assert row["attempt_failure_category_breakdown"] == {"device_transport": 1}
    assert row["final_verify_screenshot_path"] == str(verify_path)
    assert report["trace_checks_passed"] == 1
    assert report["trace_checks_failed"] == 0
    assert report["evaluator_verdict_breakdown"] == {"missing": 1}
    assert summary["trace_checks_passed"] == 1
    assert summary["trace_checks_failed"] == 0
    assert summary["evaluator_verdict_breakdown"] == {"missing": 1}


def test_case_report_classifies_evaluator_quality_gate_failure(tmp_path: Path) -> None:
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "report-evaluator-failure",
            "tasks": [{"task_id": "case-1", "query": "Do it", "app_profile": "browser.chrome"}],
        }
    )
    case = suite.tasks[0]
    sandbox = SandboxProviderSpec(template="template")
    materialized = MaterializedCaseSuite(
        spec=suite,
        output_dir=tmp_path,
        fixture_dir=tmp_path / "fixtures",
        cases=(MaterializedCase(case=case, fixtures=(), startup_commands=(), sandbox=sandbox),),
        manifest=BatchManifest(
            batch_id="report-evaluator-failure",
            tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", request="Do it", sandbox=sandbox)],
        ),
    )
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()

    report_path = CaseReportWriter(tmp_path).write(
        materialized=materialized,
        batch_result=BatchRunResult(
            batch_id="report-evaluator-failure",
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(batch_dir),
            task_states=[BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED)],
        ),
        artifact_results=[
            CaseArtifactResult(
                task_id="case-1",
                app_profile=case.app_profile,
                status="succeeded",
                evaluator_verdict="failure",
                passed=False,
                errors=["evaluator overall_verdict is not success: failure"],
            )
        ],
    )

    row = json.loads(report_path.read_text(encoding="utf-8"))["cases"][0]
    assert row["quality_gate_failure_category"] == "evaluator"


def test_case_trace_verifier_rejects_stale_verify_screenshot(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
    from northcua.experiments.cases.verification import CaseTraceVerifier

    state = BatchTaskState(
        task_id="case-1",
        controller_task_id="controller-case-1",
        trace_id="trace-case-1",
        session_id="session-case-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    spec = CaseSuiteSpec.model_validate(
        {
            "suite_id": "trace-stale-shot",
            "trace_expectations": {"evaluator_kind": "desktop"},
            "tasks": [{"task_id": "case-1", "query": "Do it", "app_profile": "browser.chrome"}],
        }
    )
    session_root = tmp_path / "agent-artifacts" / state.session_id
    eval_root = session_root / "evaluations" / "desktop_round_completion" / "round_0001"
    (session_root / "query_0001").mkdir(parents=True)
    eval_root.mkdir(parents=True)
    (session_root / "steps.jsonl").write_text(
        '{"session_step":1}\n{"session_step":2}\n',
        encoding="utf-8",
    )
    (session_root / "llm_trace.jsonl").write_text('{"trace":"main"}\n', encoding="utf-8")
    (session_root / "query_0001" / "action_step1_history_overlay_1.png").write_bytes(_TINY_PNG)
    (session_root / "query_0001" / "action_step1_verify.png").write_bytes(_TINY_PNG)
    (eval_root / "final_report.json").write_text('{"overall_verdict":"success"}', encoding="utf-8")
    (eval_root / "evidence_manifest.json").write_text('{"refs":[]}', encoding="utf-8")
    (eval_root / "llm_trace.jsonl").write_text('{"trace":"eval"}\n', encoding="utf-8")

    rows = CaseTraceVerifier().verify(
        output_dir=tmp_path,
        spec=spec,
        batch_result=BatchRunResult(
            batch_id="trace-stale-shot",
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(tmp_path),
            task_states=[state],
        ),
    )

    assert not rows[0].passed
    assert "terminal verification screenshot missing" in rows[0].errors


def test_live_trace_gate_marks_success_without_artifacts_incomplete() -> None:
    batch_result = BatchRunResult(
        batch_id="trace-gate",
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        incomplete=0,
        output_dir="/tmp/trace-gate",
        task_states=[BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED)],
    )

    gated = apply_trace_gate_to_batch_result(
        batch_result=batch_result,
        artifact_results=[
            CaseArtifactResult(
                task_id="case-1",
                app_profile="browser.chrome",
                status="succeeded",
                passed=False,
                errors=["main-agent llm_trace.jsonl missing or empty"],
            )
        ],
        dry_run=False,
    )

    assert gated.incomplete == 1
    assert gated.succeeded == 0
    assert gated.task_states[0].status == BatchTaskStatus.INCOMPLETE
    assert gated.task_states[0].terminal_decision["trace_gate"]["passed"] is False
    assert batch_result.incomplete == 0


def test_expected_feasibility_projection_preserves_trace_gate_incomplete(tmp_path: Path) -> None:
    case = CaseSpec(
        task_id="case-infeasible",
        query="Use the unavailable account.",
        app_profile="browser.chrome",
        expected_feasibility=CaseExpectedFeasibility.INFEASIBLE,
    )
    sandbox = SandboxProviderSpec()
    materialized = MaterializedCaseSuite(
        spec=CaseSuiteSpec(suite_id="trace-gated-infeasible", tasks=[case]),
        output_dir=tmp_path,
        fixture_dir=tmp_path / "fixtures",
        cases=(MaterializedCase(case=case, fixtures=(), startup_commands=(), sandbox=sandbox),),
        manifest=BatchManifest(
            batch_id="trace-gated-infeasible",
            tasks=[BatchTaskSpec(task_id=case.task_id, agent_kind="desktop", request=case.query)],
        ),
    )
    raw = BatchRunResult(
        batch_id="trace-gated-infeasible",
        total=1,
        succeeded=0,
        failed=0,
        canceled=0,
        incomplete=1,
        output_dir=str(tmp_path),
        task_states=[
            BatchTaskState(
                task_id=case.task_id,
                status=BatchTaskStatus.INCOMPLETE,
                terminal_decision={"reason": "infeasible", "trace_gate": {"passed": False}},
            )
        ],
    )

    projected = apply_expected_feasibility_to_batch_result(materialized=materialized, batch_result=raw)

    assert projected.succeeded == 0
    assert projected.incomplete == 1
    assert projected.task_states[0].status == BatchTaskStatus.INCOMPLETE


def test_trace_gate_rewrites_batch_summary_counts(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    (batch_dir / "batch_summary.json").write_text(
        json.dumps(
            {
                "batch_id": "trace-gate",
                "total": 1,
                "succeeded": 1,
                "failed": 0,
                "canceled": 0,
                "incomplete": 0,
            }
        ),
        encoding="utf-8",
    )
    batch_result = BatchRunResult(
        batch_id="trace-gate",
        total=1,
        succeeded=0,
        failed=0,
        canceled=0,
        incomplete=1,
        output_dir=str(batch_dir),
        task_states=[BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED)],
    )

    rewrite_batch_summary_for_trace_gate(
        batch_dir=batch_dir,
        batch_result=batch_result,
        artifact_results=[
            CaseArtifactResult(
                task_id="case-1",
                app_profile="browser.chrome",
                status="succeeded",
                passed=False,
            )
        ],
        dry_run=False,
    )

    summary = json.loads((batch_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["succeeded"] == 0
    assert summary["incomplete"] == 1
    assert summary["trace_gate"]["missing_artifact_success_task_ids"] == ["case-1"]


def test_dry_run_trace_gate_does_not_require_live_artifacts() -> None:
    batch_result = BatchRunResult(
        batch_id="trace-gate-dry",
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        incomplete=0,
        output_dir="/tmp/trace-gate-dry",
        task_states=[BatchTaskState(task_id="case-1", status=BatchTaskStatus.SUCCEEDED)],
    )

    gated = apply_trace_gate_to_batch_result(
        batch_result=batch_result,
        artifact_results=[
            CaseArtifactResult(
                task_id="case-1",
                app_profile="browser.chrome",
                status="succeeded",
                passed=False,
            )
        ],
        dry_run=True,
    )

    assert gated.incomplete == 0


def test_expected_infeasible_projection_counts_terminal_infeasible_as_success(tmp_path: Path) -> None:
    case = CaseSpec(
        task_id="case-infeasible",
        query="Use the unavailable account.",
        app_profile="browser.chrome",
        expected_feasibility=CaseExpectedFeasibility.INFEASIBLE,
    )
    materialized = MaterializedCaseSuite(
        spec=CaseSuiteSpec(suite_id="expected-infeasible", tasks=[case]),
        output_dir=tmp_path,
        fixture_dir=tmp_path / "fixtures",
        cases=(
            MaterializedCase(
                case=case,
                fixtures=(),
                startup_commands=(),
                sandbox=SandboxProviderSpec(),
            ),
        ),
        manifest=BatchManifest(
            batch_id="expected-infeasible",
            tasks=[BatchTaskSpec(task_id="case-infeasible", agent_kind="desktop", request=case.query)],
        ),
    )
    raw_result = BatchRunResult(
        batch_id="expected-infeasible",
        total=1,
        succeeded=0,
        failed=1,
        canceled=0,
        incomplete=0,
        output_dir=str(tmp_path),
        task_states=[
            BatchTaskState(
                task_id="case-infeasible",
                status=BatchTaskStatus.FAILED,
                error="[INFEASIBLE] Missing account",
                last_failure_category="main_agent",
                terminal_decision={"reason": "infeasible", "final_answer": "[INFEASIBLE] Missing account"},
            )
        ],
    )

    projected = apply_expected_feasibility_to_batch_result(
        materialized=materialized,
        batch_result=raw_result,
    )

    assert projected.succeeded == 1
    assert projected.failed == 0
    assert projected.task_states[0].status == BatchTaskStatus.SUCCEEDED
    assert projected.task_states[0].error == ""
    assert raw_result.failed == 1


def test_expected_infeasible_projection_rewrites_batch_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "batch_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "batch_id": "expected-infeasible",
                "total": 1,
                "succeeded": 0,
                "failed": 1,
                "canceled": 0,
                "incomplete": 0,
            }
        ),
        encoding="utf-8",
    )
    projected = BatchRunResult(
        batch_id="expected-infeasible",
        total=1,
        succeeded=1,
        failed=0,
        canceled=0,
        incomplete=0,
        output_dir=str(tmp_path),
        task_states=[
            BatchTaskState(
                task_id="case-infeasible",
                status=BatchTaskStatus.SUCCEEDED,
                terminal_decision={"reason": "infeasible", "expected_feasibility_projection": True},
            )
        ],
    )

    rewrite_batch_summary_for_expected_feasibility(batch_dir=tmp_path, batch_result=projected)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["succeeded"] == 1
    assert summary["failed"] == 0
    assert summary["expected_feasibility_projection"]["expected_infeasible_success_task_ids"] == ["case-infeasible"]


def test_case_trace_verifier_rejects_early_success_when_latest_verify_failed(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
    from northcua.experiments.cases.verification import CaseTraceVerifier

    state = BatchTaskState(
        task_id="case-1",
        controller_task_id="controller-case-1",
        trace_id="trace-case-1",
        session_id="session-case-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    spec = CaseSuiteSpec.model_validate(
        {
            "suite_id": "trace-latest-shot-failed",
            "trace_expectations": {"evaluator_kind": "desktop"},
            "tasks": [{"task_id": "case-1", "query": "Do it", "app_profile": "browser.chrome"}],
        }
    )
    session_root = tmp_path / "agent-artifacts" / state.session_id
    eval_root = session_root / "evaluations" / "desktop_round_completion" / "round_0001"
    (session_root / "query_0001").mkdir(parents=True)
    eval_root.mkdir(parents=True)
    early_verify = session_root / "query_0001" / "action_step1_verify.png"
    (session_root / "steps.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "session_step": 1,
                        "observation_metadata": {
                            "round_end_verification_status": "success",
                            "round_end_verification_screenshot_path": str(early_verify),
                        },
                    }
                ),
                json.dumps(
                    {
                        "session_step": 2,
                        "observation_metadata": {
                            "round_end_verification_status": "failure",
                            "round_end_verification_error": "capture failed",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (session_root / "llm_trace.jsonl").write_text('{"trace":"main"}\n', encoding="utf-8")
    early_verify.write_bytes(_TINY_PNG)
    (eval_root / "final_report.json").write_text('{"overall_verdict":"success"}', encoding="utf-8")
    (eval_root / "evidence_manifest.json").write_text('{"refs":[]}', encoding="utf-8")
    (eval_root / "llm_trace.jsonl").write_text('{"trace":"eval"}\n', encoding="utf-8")

    rows = CaseTraceVerifier().verify(
        output_dir=tmp_path,
        spec=spec,
        batch_result=BatchRunResult(
            batch_id="trace-latest-shot-failed",
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(tmp_path),
            task_states=[state],
        ),
    )

    assert not rows[0].passed
    assert "terminal verification screenshot missing" in rows[0].errors


def test_case_trace_verifier_rejects_evaluator_kind_substring_collision(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
    from northcua.experiments.cases.verification import CaseTraceVerifier

    state = BatchTaskState(
        task_id="case-1",
        controller_task_id="controller-case-1",
        trace_id="trace-case-1",
        session_id="session-case-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    spec = CaseSuiteSpec.model_validate(
        {
            "suite_id": "trace-kind-collision",
            "trace_expectations": {"evaluator_kind": "desktop"},
            "tasks": [{"task_id": "case-1", "query": "Do it", "app_profile": "browser.chrome"}],
        }
    )
    session_root = tmp_path / "agent-artifacts" / state.session_id
    eval_root = session_root / "evaluations" / "not_desktop_round_completion" / "round_0001"
    (session_root / "query_0001").mkdir(parents=True)
    eval_root.mkdir(parents=True)
    verify_path = session_root / "query_0001" / "action_step1_verify.png"
    (session_root / "steps.jsonl").write_text(
        json.dumps(
            {
                "session_step": 1,
                "observation_metadata": {
                    "round_end_verification_status": "success",
                    "round_end_verification_screenshot_path": str(verify_path),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (session_root / "llm_trace.jsonl").write_text('{"trace":"main"}\n', encoding="utf-8")
    verify_path.write_bytes(_TINY_PNG)
    (eval_root / "final_report.json").write_text('{"overall_verdict":"success"}', encoding="utf-8")
    (eval_root / "evidence_manifest.json").write_text('{"refs":[]}', encoding="utf-8")
    (eval_root / "llm_trace.jsonl").write_text('{"trace":"eval"}\n', encoding="utf-8")

    rows = CaseTraceVerifier().verify(
        output_dir=tmp_path,
        spec=spec,
        batch_result=BatchRunResult(
            batch_id="trace-kind-collision",
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(tmp_path),
            task_states=[state],
        ),
    )

    assert not rows[0].passed
    assert "evaluator final_report.json missing or empty" in rows[0].errors


def test_case_trace_verifier_rejects_orphan_verify_filename_without_success_metadata(tmp_path: Path) -> None:
    from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
    from northcua.experiments.cases.verification import CaseTraceVerifier

    state = BatchTaskState(
        task_id="case-1",
        controller_task_id="controller-case-1",
        trace_id="trace-case-1",
        session_id="session-case-1",
        status=BatchTaskStatus.SUCCEEDED,
        attempts=1,
    )
    spec = CaseSuiteSpec.model_validate(
        {
            "suite_id": "trace-orphan-verify",
            "tasks": [{"task_id": "case-1", "query": "Do it", "app_profile": "browser.chrome"}],
        }
    )
    session_root = tmp_path / "agent-artifacts" / state.session_id
    eval_root = session_root / "evaluations" / "desktop_round_completion" / "round_0001"
    query_root = session_root / "query_0001"
    query_root.mkdir(parents=True)
    eval_root.mkdir(parents=True)
    verify_path = query_root / "action_step2_verify.png"
    (session_root / "steps.jsonl").write_text(
        json.dumps({"session_step": 2, "observation_metadata": {"round_end_verification_status": "failed"}}) + "\n",
        encoding="utf-8",
    )
    (session_root / "llm_trace.jsonl").write_text('{"trace":"main"}\n', encoding="utf-8")
    verify_path.write_bytes(_TINY_PNG)
    (eval_root / "final_report.json").write_text('{"overall_verdict":"success"}', encoding="utf-8")
    (eval_root / "evidence_manifest.json").write_text('{"refs":[]}', encoding="utf-8")
    (eval_root / "llm_trace.jsonl").write_text('{"trace":"eval"}\n', encoding="utf-8")

    rows = CaseTraceVerifier().verify(
        output_dir=tmp_path,
        spec=spec,
        batch_result=BatchRunResult(
            batch_id="trace-orphan-verify",
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(tmp_path),
            task_states=[state],
        ),
    )

    assert not rows[0].passed
    assert "terminal verification screenshot missing" in rows[0].errors


@pytest.mark.asyncio
async def test_case_run_dry_run_writes_report_without_controller_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTHCUA_CONTROLLER_TOKEN", "secret-token")
    from northcua.experiments.cases.operations import CaseOperations

    operations = CaseOperations()
    result = await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
    )

    assert result.succeeded == 5
    report_path = Path(result.case_report_path)
    assert report_path.is_file()
    assert "secret-token" not in (Path(result.output_dir) / "case_suite.json").read_text(encoding="utf-8")
    assert "secret-token" not in (Path(result.output_dir) / "batch_manifest.json").read_text(encoding="utf-8")
    assert '"run_mode": "dry_run"' in report_path.read_text(encoding="utf-8")
    assert "sandbox_log_dir" in report_path.read_text(encoding="utf-8")
    initial_report = json.loads(report_path.read_text(encoding="utf-8"))
    initial_calc = next(case for case in initial_report["cases"] if case["task_id"] == "live-five-calc-pivot")
    assert all(initial_calc["fixture_sha256"].values())

    rebuilt_report = operations.report(
        suite_id=result.suite_id,
        run_id=result.run_id,
        output_dir=tmp_path,
    )
    rebuilt_calc = next(case for case in rebuilt_report["cases"] if case["task_id"] == "live-five-calc-pivot")
    assert rebuilt_calc["fixture_sha256"] == initial_calc["fixture_sha256"]


@pytest.mark.asyncio
async def test_case_run_parallelism_overrides_spec_and_controller(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    result = await CaseOperations().run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        concurrency=4,
        controller_parallelism=4,
    )

    output_dir = Path(result.output_dir)
    persisted_suite = json.loads((output_dir / "case_suite.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "batch_manifest.json").read_text(encoding="utf-8"))

    assert persisted_suite["concurrency"] == 4
    assert persisted_suite["controller"]["parallelism"] == 4
    assert persisted_suite["controller"]["runner_workers"] == 4
    assert persisted_suite["controller"]["worker_queue_max"] == 4
    assert manifest["defaults"]["concurrency"] == 4


@pytest.mark.asyncio
async def test_case_run_requires_explicit_resume_for_existing_run(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    operations = CaseOperations()
    await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
    )

    with pytest.raises(FileExistsError, match="--resume or --force"):
        await operations.run_file(
            spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
            output_dir=tmp_path,
            public_controller_url="wss://controller.example/ws",
            dry_run=True,
            run_id="fixed-run",
        )


@pytest.mark.asyncio
async def test_case_run_resume_uses_persisted_manifest_not_current_spec(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
version: v1
suite_id: resume-source-of-truth
tasks:
  - task_id: browser
    query: Original query.
    app_profile: browser.chrome
    template_profile: office-web
""".strip(),
        encoding="utf-8",
    )
    operations = CaseOperations()
    first = await operations.run_file(
        spec_path=suite_path,
        output_dir=tmp_path / "runs",
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
    )
    suite_path.write_text(
        """
version: v1
suite_id: resume-source-of-truth
tasks:
  - task_id: browser
    query: Mutated query that must not be used on resume.
    app_profile: creative.gimp
    template_profile: creative
""".strip(),
        encoding="utf-8",
    )

    resumed = await operations.run_file(
        spec_path=suite_path,
        output_dir=tmp_path / "runs",
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
        resume=True,
    )

    persisted_manifest = (Path(resumed.output_dir) / "batch_manifest.json").read_text(encoding="utf-8")
    assert resumed.run_id == first.run_id
    assert "Original query." in persisted_manifest
    assert "Mutated query" not in persisted_manifest


@pytest.mark.asyncio
async def test_case_run_resume_does_not_expand_current_case_sources_for_default_suite_id(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    case_path = case_dir / "browser.case.yaml"
    case_path.write_text(
        """
version: v1
case:
  task_id: browser-from-source
  query: Original source query.
  app_profile: browser.chrome
  template_profile: office-web
""".strip(),
        encoding="utf-8",
    )
    suite_path = tmp_path / "source-suite.yaml"
    suite_path.write_text(
        """
version: v1
case_sources:
  - kind: directory
    path: cases
""".strip(),
        encoding="utf-8",
    )
    operations = CaseOperations()
    first = await operations.run_file(
        spec_path=suite_path,
        output_dir=tmp_path / "runs",
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
    )
    case_path.unlink()

    resumed = await operations.run_file(
        spec_path=suite_path,
        output_dir=tmp_path / "runs",
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
        resume=True,
    )

    persisted_manifest = (Path(resumed.output_dir) / "batch_manifest.json").read_text(encoding="utf-8")
    assert resumed.run_id == first.run_id
    assert "Original source query." in persisted_manifest


@pytest.mark.asyncio
async def test_case_run_resume_clears_pause_and_drain_flags(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    operations = CaseOperations()
    first = await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
    )
    sqlite_path = Path(first.output_dir) / "batch" / first.run_id / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(sqlite_path)
    repository.set_control_flag("paused", True)
    repository.set_control_flag("drain_requested", True)

    await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
        resume=True,
    )

    flags = {
        row["name"]: row["value"] for row in repository._connection.execute("SELECT name, value FROM batch_control")
    }
    assert flags["paused"] == "0"
    assert flags["drain_requested"] == "0"


@pytest.mark.asyncio
async def test_case_run_live_resume_runs_pre_clean_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from northcua.experiments.cases import operations as case_operations_module
    from northcua.experiments.cases.operations import CaseOperations

    monkeypatch.setenv("NORTHCUA_CONTROLLER_TOKEN", "resume-token")
    monkeypatch.setenv("NORTHCUA_RUN_E2B_BATCH", "1")
    operations = CaseOperations()
    first = await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
    )
    events: list[str] = []

    async def fake_cleanup(**kwargs: object) -> None:
        events.append("cleanup")
        batch_dir = Path(kwargs["output_dir"]) / str(kwargs["batch_id"])
        now = time.time()
        (batch_dir / "pre_resume_cleanup_summary.json").write_text(
            json.dumps(
                {
                    "started_at_unix": now,
                    "finished_at_unix": now,
                    "passed": True,
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )

    async def fake_execute(**kwargs: object) -> CaseRunResult:
        events.append("runner")
        batch_result = BatchRunResult(
            batch_id=first.run_id,
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(Path(first.output_dir) / "batch" / first.run_id),
            task_states=[],
        )
        return CaseRunResult(
            suite_id=first.suite_id,
            run_id=first.run_id,
            run_mode=CaseRunMode.LIVE,
            dry_run=False,
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            incomplete=0,
            output_dir=first.output_dir,
            batch_result=batch_result,
            case_report_path=str(Path(first.output_dir) / "case_report.json"),
            batch_manifest_path=str(Path(first.output_dir) / "batch_manifest.json"),
        )

    monkeypatch.setattr(case_operations_module, "_run_pre_resume_cleanup", fake_cleanup)
    monkeypatch.setattr(CaseOperations, "_validate_resume_runtime_config", staticmethod(lambda **_: None))
    monkeypatch.setattr(operations, "_execute_materialized_run", fake_execute)

    await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=False,
        run_id="fixed-run",
        resume=True,
    )

    assert events == ["cleanup", "runner"]


@pytest.mark.asyncio
async def test_case_run_live_resume_pre_clean_tolerates_missing_controller_db_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from northcua.experiments.cases import operations as case_operations_module
    from northcua.experiments.cases.operations import CaseOperations

    monkeypatch.setenv("NORTHCUA_CONTROLLER_TOKEN", "resume-token")
    monkeypatch.setenv("NORTHCUA_RUN_E2B_BATCH", "1")
    operations = CaseOperations()
    first = await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
    )
    case_suite_path = Path(first.output_dir) / "case_suite.json"
    persisted_suite = json.loads(case_suite_path.read_text(encoding="utf-8"))
    persisted_suite["controller"]["managed_local"] = False
    persisted_suite["controller"]["db_path"] = None
    case_suite_path.write_text(json.dumps(persisted_suite), encoding="utf-8")
    seen: dict[str, object] = {}

    async def fake_cleanup(**kwargs: object) -> None:
        seen.update(kwargs)
        batch_dir = Path(kwargs["output_dir"]) / str(kwargs["batch_id"])
        now = time.time()
        (batch_dir / "pre_resume_cleanup_summary.json").write_text(
            json.dumps(
                {
                    "started_at_unix": now,
                    "finished_at_unix": now,
                    "passed": True,
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )

    async def fake_execute(**kwargs: object) -> CaseRunResult:
        batch_result = BatchRunResult(
            batch_id=first.run_id,
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(Path(first.output_dir) / "batch" / first.run_id),
            task_states=[],
        )
        return CaseRunResult(
            suite_id=first.suite_id,
            run_id=first.run_id,
            run_mode=CaseRunMode.LIVE,
            dry_run=False,
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            incomplete=0,
            output_dir=first.output_dir,
            batch_result=batch_result,
            case_report_path=str(Path(first.output_dir) / "case_report.json"),
            batch_manifest_path=str(Path(first.output_dir) / "batch_manifest.json"),
        )

    monkeypatch.setattr(case_operations_module, "_run_pre_resume_cleanup", fake_cleanup)
    monkeypatch.setattr(CaseOperations, "_validate_resume_runtime_config", staticmethod(lambda **_: None))
    monkeypatch.setattr(operations, "_execute_materialized_run", fake_execute)

    await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=False,
        run_id="fixed-run",
        resume=True,
    )

    assert seen["controller_db"] == Path(first.output_dir) / "controller.sqlite"


@pytest.mark.asyncio
async def test_case_run_live_resume_aborts_when_pre_clean_leaves_active_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from northcua.experiments.cases import operations as case_operations_module
    from northcua.experiments.cases.operations import CaseOperations

    monkeypatch.setenv("NORTHCUA_CONTROLLER_TOKEN", "resume-token")
    monkeypatch.setenv("NORTHCUA_RUN_E2B_BATCH", "1")
    operations = CaseOperations()
    first = await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
    )
    sqlite_path = Path(first.output_dir) / "batch" / first.run_id / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(sqlite_path)
    repository.set_control_flag("paused", True)
    repository.upsert_state(
        BatchTaskState(
            task_id="live-five-calc-pivot",
            status=BatchTaskStatus.SUBMITTED,
            attempts=1,
            controller_task_id="batch-stale-task",
        )
    )
    events: list[str] = []

    async def fake_cleanup(**kwargs: object) -> None:
        events.append("cleanup")
        batch_dir = Path(kwargs["output_dir"]) / str(kwargs["batch_id"])
        now = time.time()
        (batch_dir / "pre_resume_cleanup_summary.json").write_text(
            json.dumps(
                {
                    "started_at_unix": now,
                    "finished_at_unix": now,
                    "passed": True,
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )

    async def fake_execute(**_: object) -> CaseRunResult:
        raise AssertionError("runner must not start when pre-clean leaves active rows")

    monkeypatch.setattr(case_operations_module, "_run_pre_resume_cleanup", fake_cleanup)
    monkeypatch.setattr(CaseOperations, "_validate_resume_runtime_config", staticmethod(lambda **_: None))
    monkeypatch.setattr(operations, "_execute_materialized_run", fake_execute)

    with pytest.raises(RuntimeError, match="left active batch rows"):
        await operations.run_file(
            spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
            output_dir=tmp_path,
            public_controller_url="wss://controller.example/ws",
            dry_run=False,
            run_id="fixed-run",
            resume=True,
        )

    assert events == ["cleanup"]
    flags = {
        row["name"]: row["value"] for row in repository._connection.execute("SELECT name, value FROM batch_control")
    }
    assert flags["paused"] == "1"


def test_case_resume_barrier_blocks_mixed_leased_and_unleased_submitted_rows(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    sqlite_path = tmp_path / "batch_state.sqlite"
    manifest = BatchManifest(
        batch_id="mixed-residue",
        tasks=[
            BatchTaskSpec(task_id="old-leased", agent_kind="desktop", query="old"),
            BatchTaskSpec(task_id="new-unleased", agent_kind="desktop", query="new"),
        ],
    )
    repository = SQLiteBatchStateRepository(sqlite_path)
    repository.initialize_manifest(manifest)
    repository.upsert_state(
        BatchTaskState(
            task_id="old-leased",
            status=BatchTaskStatus.SUBMITTED,
            attempts=3,
            controller_task_id="batch-old",
        )
    )
    repository.upsert_state(
        BatchTaskState(
            task_id="new-unleased",
            status=BatchTaskStatus.SUBMITTED,
            attempts=4,
            controller_task_id="batch-new",
        )
    )
    with sqlite3.connect(str(sqlite_path)) as connection:
        connection.execute(
            """
            UPDATE batch_tasks
            SET lease_owner='old-runner', lease_token='active-token', lease_expires_at=?
            WHERE task_id='old-leased'
            """,
            (time.time() + 3600,),
        )

    with pytest.raises(RuntimeError, match="left active batch rows") as exc_info:
        CaseOperations._assert_clean_case_resume_barrier(sqlite_path)

    message = str(exc_info.value)
    assert "'count': 2" in message
    assert "'lease_count': 1" in message


@pytest.mark.asyncio
async def test_case_run_failed_resume_keeps_pause_on_prepared_policy_mismatch(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    operations = CaseOperations()
    first = await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
        prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1),
    )
    sqlite_path = Path(first.output_dir) / "batch" / first.run_id / "batch_state.sqlite"
    repository = SQLiteBatchStateRepository(sqlite_path)
    repository.set_control_flag("paused", True)

    with pytest.raises(ValueError, match="prepared_sandbox_policy"):
        await operations.run_file(
            spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
            output_dir=tmp_path,
            public_controller_url="wss://controller.example/ws",
            dry_run=True,
            run_id="fixed-run",
            resume=True,
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=False),
        )

    assert SQLiteBatchStateRepository(sqlite_path).control_flags()["paused"]


@pytest.mark.asyncio
async def test_case_run_resume_reuses_persisted_cli_prepared_policy(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    operations = CaseOperations()
    first = await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
        prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True, buffer=1),
    )

    resumed = await operations.run_file(
        spec_path="tests/fixtures/scenarios/experiments/e2b-live-five.yaml",
        output_dir=tmp_path,
        public_controller_url="wss://controller.example/ws",
        dry_run=True,
        run_id="fixed-run",
        resume=True,
    )

    assert resumed.run_id == first.run_id


@pytest.mark.asyncio
async def test_case_run_invalid_prepared_allocation_does_not_persist_run(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
version: v1
suite_id: invalid-prepared-case
sandbox:
  provider: fake
  template: desktop
  allocation_strategy: connect_existing
  sandbox_id: existing
tasks:
  - task_id: browser
    query: Original query.
    app_profile: browser.chrome
    template_profile: office-web
""".strip(),
        encoding="utf-8",
    )
    run_root = tmp_path / "runs" / "invalid-prepared-case" / "fixed-run"

    with pytest.raises(ValueError, match="connect_existing"):
        await CaseOperations().run_file(
            spec_path=suite_path,
            output_dir=tmp_path / "runs",
            public_controller_url="wss://controller.example/ws",
            dry_run=True,
            run_id="fixed-run",
            prepared_sandbox_policy=PreparedSandboxPolicy(enabled=True),
        )

    assert not (run_root / "case_suite.json").exists()
    assert not (run_root / "batch").exists()


@pytest.mark.asyncio
async def test_case_run_force_keeps_existing_run_when_preflight_fails(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
version: v1
suite_id: force-preflight
tasks:
  - task_id: browser
    query: Original query.
    app_profile: browser.chrome
    template_profile: office-web
""".strip(),
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "force-preflight" / "fixed-run"
    run_dir.mkdir(parents=True)
    marker = run_dir / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        await CaseOperations().run_file(
            spec_path=suite_path,
            catalog_path=tmp_path / "missing-catalog.yaml",
            output_dir=tmp_path / "runs",
            public_controller_url="wss://controller.example/ws",
            dry_run=True,
            run_id="fixed-run",
            force=True,
        )

    assert marker.read_text(encoding="utf-8") == "keep me"


@pytest.mark.asyncio
async def test_case_run_force_rejects_active_sqlite_run(tmp_path: Path) -> None:
    from northcua.experiments.cases.operations import CaseOperations

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
version: v1
suite_id: force-active
tasks:
  - task_id: browser
    query: Original query.
    app_profile: browser.chrome
    template_profile: office-web
""".strip(),
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "force-active" / "fixed-run"
    batch_dir = run_dir / "batch" / "fixed-run"
    batch_dir.mkdir(parents=True)
    marker = run_dir / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")
    manifest = BatchManifest(
        batch_id="fixed-run",
        tasks=[BatchTaskSpec(task_id="browser", agent_kind="desktop", query="Original query.")],
    )
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    assert repository.upsert_state(BatchTaskState(task_id="browser", status=BatchTaskStatus.RUNNING, attempts=1))

    with pytest.raises(RuntimeError, match="refusing to force-delete active case run"):
        await CaseOperations().run_file(
            spec_path=suite_path,
            output_dir=tmp_path / "runs",
            public_controller_url="wss://controller.example/ws",
            dry_run=True,
            run_id="fixed-run",
            force=True,
        )

    assert marker.read_text(encoding="utf-8") == "keep me"


@pytest.mark.asyncio
async def test_case_run_force_rejects_leased_canceled_sqlite_run(tmp_path: Path) -> None:
    from northcua.batch.state import LeasePolicy
    from northcua.experiments.cases.operations import CaseOperations

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
version: v1
suite_id: force-leased-canceled
tasks:
  - task_id: browser
    query: Original query.
    app_profile: browser.chrome
    template_profile: office-web
""".strip(),
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs" / "force-leased-canceled" / "fixed-run"
    batch_dir = run_dir / "batch" / "fixed-run"
    batch_dir.mkdir(parents=True)
    marker = run_dir / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")
    manifest = BatchManifest(
        batch_id="fixed-run",
        tasks=[BatchTaskSpec(task_id="browser", agent_kind="desktop", query="Original query.")],
    )
    repository = SQLiteBatchStateRepository(batch_dir / "batch_state.sqlite")
    repository.initialize_manifest(manifest)
    claim = repository.claim_next(
        lease_owner="runner-a",
        policy=LeasePolicy(lease_timeout_seconds=60, heartbeat_interval_seconds=20),
    )
    assert claim is not None
    repository.mark_cancel_requested()

    with pytest.raises(RuntimeError, match="active case run"):
        await CaseOperations().run_file(
            spec_path=suite_path,
            output_dir=tmp_path / "runs",
            public_controller_url="wss://controller.example/ws",
            dry_run=True,
            run_id="fixed-run",
            force=True,
        )

    assert marker.read_text(encoding="utf-8") == "keep me"


def test_materializer_rejects_duplicate_remote_paths(tmp_path: Path) -> None:
    case = CaseSpec(
        task_id="dups",
        query="Do it",
        app_profile="browser.chrome",
        fixtures=[
            FixtureSpec(generated=GeneratedFixtureSpec(generator="text.file"), remote_path="/tmp/northcua-cases/a.txt"),
            FixtureSpec(generated=GeneratedFixtureSpec(generator="text.file"), remote_path="/tmp/northcua-cases/a.txt"),
        ],
    )
    suite = CaseSuiteSpec(suite_id="dups", tasks=[case])

    with pytest.raises(ValueError, match="remote_path"):
        CaseMaterializer().materialize(suite, output_dir=tmp_path)


def test_generated_fixture_staging_uses_full_remote_path(tmp_path: Path) -> None:
    materializer = FixtureMaterializer()
    left = materializer.materialize(
        FixtureSpec(
            generated=GeneratedFixtureSpec(generator="text.file"), remote_path="/tmp/northcua-cases/a/data.txt"
        ),
        task_id="same-name",
        fixture_dir=tmp_path,
        spec_dir=None,
    )
    right = materializer.materialize(
        FixtureSpec(
            generated=GeneratedFixtureSpec(generator="text.file"), remote_path="/tmp/northcua-cases/b/data.txt"
        ),
        task_id="same-name",
        fixture_dir=tmp_path,
        spec_dir=None,
    )

    assert left.upload.local_path != right.upload.local_path


def test_large_artifact_guest_download_is_not_materialized_through_generic_url_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from northcua.experiments.cases.materialization.fixtures import _DownloadEndpoint

    monkeypatch.setattr(
        FixtureMaterializer,
        "_download",
        lambda *args, **kwargs: pytest.fail("large guest downloads must not use host URL materialization"),
    )
    monkeypatch.setattr(
        FixtureMaterializer,
        "_download_endpoint",
        staticmethod(
            lambda url: _DownloadEndpoint(
                url=url,
                scheme="https",
                host="artifacts.example.test",
                port=443,
                address="8.8.8.8",
                target="/blobs/large",
            )
        ),
    )
    monkeypatch.setattr(
        FixtureMaterializer,
        "_preflight_guest_download_source",
        staticmethod(lambda endpoint, *, expected_size_bytes: "endpoint_and_size_verified"),
    )
    fixture = FixtureSpec.model_validate(
        {
            "url": "https://artifacts.example.test/blobs/large?signature=hidden",
            "remote_path": "/tmp/northcua-cases/large.bin",
            "artifact": {
                "sha256": "b" * 64,
                "size_bytes": 100 * 1024 * 1024,
                "delivery": "guest_download",
            },
        }
    )

    materialized = FixtureMaterializer().materialize(
        fixture,
        task_id="large-download",
        fixture_dir=tmp_path,
        spec_dir=None,
        app_profile=AppProfileId.VLC,
    )

    assert materialized.upload.delivery == SandboxFileDeliveryMode.GUEST_DOWNLOAD
    assert materialized.upload.local_path is None
    assert materialized.upload.size_bytes == 100 * 1024 * 1024
    assert materialized.upload.source_address == "8.8.8.8"
    assert materialized.upload.intended_app == "media.vlc"
    assert materialized.large_artifact_evidence is not None
    assert materialized.large_artifact_evidence.fixture_remote_path == "/tmp/northcua-cases/large.bin"
    assert materialized.large_artifact_evidence.host_verification == "endpoint_and_size_verified"
    assert materialized.large_artifact_evidence.source == "https://artifacts.example.test/blobs/large"


def test_large_artifact_chunked_upload_verifies_local_content_address(tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    source.write_bytes(b"chunked-payload")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    fixture = FixtureSpec.model_validate(
        {
            "source": str(source),
            "remote_path": "/tmp/northcua-cases/large.bin",
            "artifact": {
                "sha256": checksum,
                "size_bytes": source.stat().st_size,
                "delivery": "chunked_upload",
                "chunk_size_bytes": 4,
            },
        }
    )

    materialized = FixtureMaterializer().materialize(
        fixture,
        task_id="large-chunked",
        fixture_dir=tmp_path,
        spec_dir=None,
        app_profile=AppProfileId.VLC,
    )

    assert materialized.upload.delivery == SandboxFileDeliveryMode.CHUNKED_UPLOAD
    assert materialized.upload.sha256 == checksum
    assert materialized.reference is not None
    assert materialized.reference.size_bytes == source.stat().st_size
    assert materialized.large_artifact_evidence is not None
    assert materialized.large_artifact_evidence.app_profile == AppProfileId.VLC
    assert materialized.large_artifact_evidence.host_verification == "sha256_and_size_verified"


def test_large_artifact_local_manifest_fails_closed_before_allocation(tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    source.write_bytes(b"actual-payload")
    fixture = FixtureSpec.model_validate(
        {
            "source": str(source),
            "remote_path": "/tmp/northcua-cases/large.bin",
            "artifact": {
                "sha256": "a" * 64,
                "size_bytes": source.stat().st_size,
                "delivery": "chunked_upload",
            },
        }
    )

    with pytest.raises(ValueError, match="content address mismatch"):
        FixtureMaterializer().materialize(
            fixture,
            task_id="large-mismatch",
            fixture_dir=tmp_path / "fixtures",
            spec_dir=None,
            app_profile=AppProfileId.VLC,
        )


def test_large_artifact_guest_download_preflight_checks_declared_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from northcua.experiments.cases.materialization.fixtures import _DownloadEndpoint

    class HeadResponse:
        status = 200

        def getheader(self, name: str) -> str:
            return "1024" if name == "Content-Length" else ""

    class HeadConnection:
        requests: list[tuple[str, str]] = []

        def __init__(self, endpoint: _DownloadEndpoint, *, timeout: float) -> None:
            _ = endpoint, timeout

        def request(self, method: str, target: str, **kwargs: object) -> None:
            _ = kwargs
            self.requests.append((method, target))

        def getresponse(self) -> HeadResponse:
            return HeadResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "northcua.experiments.cases.materialization.fixtures._BoundHTTPSConnection",
        HeadConnection,
    )
    endpoint = _DownloadEndpoint(
        url="https://artifacts.example.test/blob?signature=hidden",
        scheme="https",
        host="artifacts.example.test",
        port=443,
        address="8.8.8.8",
        target="/blob?signature=hidden",
    )

    status = FixtureMaterializer._preflight_guest_download_source(endpoint, expected_size_bytes=1024)

    assert status == "endpoint_and_size_verified"
    assert HeadConnection.requests == [("HEAD", "/blob?signature=hidden")]
    with pytest.raises(ValueError, match="source size mismatch"):
        FixtureMaterializer._preflight_guest_download_source(endpoint, expected_size_bytes=2048)


def test_case_loader_materializes_large_artifact_manifest_evidence(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    source = case_dir / "large.bin"
    source.write_bytes(b"content-addressed-media")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    case_path = case_dir / "large.case.yaml"
    case_path.write_text(
        f"""
task_id: large-manifest
query: Open the supplied media file.
app_profile: media.vlc
template_profile: desktop-full
fixtures:
  - source: large.bin
    remote_path: /tmp/northcua-cases/large-manifest/media.bin
    artifact:
      sha256: {checksum}
      size_bytes: {source.stat().st_size}
      delivery: chunked_upload
      chunk_size_bytes: 4
""".strip(),
        encoding="utf-8",
    )
    case = load_case_file(case_path)
    suite = CaseSuiteSpec(suite_id="large-manifest-suite", tasks=[case])

    materialized = CaseMaterializer().materialize(suite, output_dir=tmp_path / "output")

    assert case.fixtures[0].source_path == source
    task = materialized.manifest.tasks[0]
    evidence = task.metadata["large_artifact_delivery_evidence"]
    assert evidence == [
        {
            "task_id": "large-manifest",
            "app_profile": "media.vlc",
            "fixture_remote_path": "/tmp/northcua-cases/large-manifest/media.bin",
            "source_kind": "local_path",
            "source": str(source),
            "sha256": checksum,
            "size_bytes": source.stat().st_size,
            "delivery": "chunked_upload",
            "chunk_size_bytes": 4,
            "host_verification": "sha256_and_size_verified",
            "guest_verification": "sha256_and_size_before_init",
        }
    ]
    assert task.sandbox is not None
    assert [upload.remote_path for upload in task.sandbox.file_uploads] == [
        "/tmp/northcua-cases/large-manifest/media.bin"
    ]
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    report_path = CaseReportWriter(materialized.output_dir).write(
        materialized=materialized,
        batch_result=BatchRunResult(
            batch_id="large-manifest-suite",
            total=1,
            succeeded=0,
            failed=0,
            canceled=0,
            output_dir=str(batch_dir),
            task_states=[],
        ),
        dry_run=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["cases"][0]["large_artifact_delivery_evidence"] == evidence


def test_generic_url_fixture_keeps_fifty_mib_download_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from northcua.experiments.cases.materialization.fixtures import _DownloadEndpoint

    class OversizedResponse:
        status = 200

        def getheader(self, name: str) -> str:
            return str(50 * 1024 * 1024 + 1) if name == "Content-Length" else ""

    class OversizedConnection:
        def __init__(self, endpoint: _DownloadEndpoint, *, timeout: float) -> None:
            _ = endpoint, timeout

        def request(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

        def getresponse(self) -> OversizedResponse:
            return OversizedResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "northcua.experiments.cases.materialization.fixtures._BoundHTTPSConnection",
        OversizedConnection,
    )
    endpoint = _DownloadEndpoint(
        url="https://example.com/file.bin",
        scheme="https",
        host="example.com",
        port=443,
        address="93.184.216.34",
        target="/file.bin",
    )

    with pytest.raises(ValueError, match=str(50 * 1024 * 1024)):
        FixtureMaterializer._download("https://example.com/file.bin", tmp_path / "file.bin", endpoint=endpoint)


def test_url_fixture_download_rejects_redirects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from northcua.experiments.cases.materialization.fixtures import _DownloadEndpoint

    class RedirectResponse:
        status = 302

        def getheader(self, name: str) -> str:
            _ = name
            return ""

        def read(self, size: int) -> bytes:
            _ = size
            return b""

    class RedirectConnection:
        def __init__(self, endpoint: _DownloadEndpoint, *, timeout: float) -> None:
            _ = endpoint, timeout

        def request(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

        def getresponse(self) -> RedirectResponse:
            return RedirectResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "northcua.experiments.cases.materialization.fixtures._BoundHTTPSConnection",
        RedirectConnection,
    )
    endpoint = _DownloadEndpoint(
        url="https://example.com/file.txt",
        scheme="https",
        host="example.com",
        port=443,
        address="93.184.216.34",
        target="/file.txt",
    )

    with pytest.raises(ValueError, match="redirects"):
        FixtureMaterializer._download("https://example.com/file.txt", tmp_path / "file.txt", endpoint=endpoint)


def test_url_fixture_endpoint_rejects_non_global_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(
        *args: object, **kwargs: object
    ) -> list[tuple[object, object, object, object, tuple[str, int]]]:
        _ = args, kwargs
        return [(None, None, None, None, ("100.64.0.1", 443))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)

    with pytest.raises(ValueError, match="disallowed address"):
        FixtureMaterializer._download_endpoint("https://example.com/file.txt")


def test_url_fixture_endpoint_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="must not embed credentials"):
        FixtureMaterializer._download_endpoint("https://user:secret@example.com/file.txt")


@pytest.mark.asyncio
async def test_case_run_live_preflight_uses_effective_catalog_provider(tmp_path: Path) -> None:
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: fake-root
sandbox:
  provider: fake
tasks:
  - task_id: browser
    query: Open it
    app_profile: browser.chrome
""",
        encoding="utf-8",
    )
    from northcua.experiments.cases.operations import CaseOperations

    with pytest.raises(ValueError, match="public-controller-url"):
        await CaseOperations().run_file(spec_path=spec_path, output_dir=tmp_path, dry_run=False)
