from __future__ import annotations

import json
from pathlib import Path

import pytest

from northcua.batch.contracts import BatchRunResult, BatchTaskState, BatchTaskStatus
from northcua.experiments.cases.catalog import CaseCatalog
from northcua.experiments.cases.contracts import CaseSetupPhase, CaseSpec, CaseSuiteSpec
from northcua.experiments.cases.materialization import CaseMaterializer
from northcua.experiments.cases.reporting import CaseReportWriter


def test_typed_setup_projects_legacy_fields_and_round_trips() -> None:
    case = CaseSpec.model_validate(_case_payload(typed=True))

    assert case.setup is not None
    assert [phase.phase for phase in case.setup.phases] == [CaseSetupPhase.SETUP, CaseSetupPhase.INIT]
    assert [recipe.recipe.value for recipe in case.dependency_injection.setup_recipes] == ["run_setup_script"]
    assert [recipe.recipe.value for recipe in case.init] == ["open_blank_app"]

    serialized = case.model_dump(mode="json")
    assert serialized["setup"]["phases"][0]["phase"] == "setup"
    assert serialized["dependency_injection"]["setup_recipes"] == serialized["setup"]["phases"][0]["recipes"]
    assert serialized["init"] == serialized["setup"]["phases"][1]["recipes"]
    assert CaseSpec.model_validate(serialized) == case


def test_legacy_init_serialization_and_phase_order_remain_unchanged() -> None:
    payload = _case_payload(typed=False)
    case = CaseSpec.model_validate(payload)

    serialized = case.model_dump(mode="json")
    assert "setup" not in serialized
    assert serialized["init"][0]["recipe"] == "open_blank_app"
    assert [phase.phase for phase in case.effective_setup_phases()] == [
        CaseSetupPhase.SETUP,
        CaseSetupPhase.INIT,
    ]


def test_typed_setup_rejects_conflicting_legacy_recipes() -> None:
    payload = _case_payload(typed=True)
    payload["init"] = [{"recipe": "open_url", "args": {"app": "typed.setup.app", "url": "https://example.com"}}]

    with pytest.raises(ValueError, match="typed init phase conflicts"):
        CaseSpec.model_validate(payload)


def test_typed_setup_rejects_duplicate_or_reversed_phases() -> None:
    payload = _case_payload(typed=True)
    setup = payload["setup"]
    assert isinstance(setup, dict)
    phases = setup["phases"]
    assert isinstance(phases, list)

    setup["phases"] = [phases[0], phases[0]]
    with pytest.raises(ValueError, match="phases must be unique"):
        CaseSpec.model_validate(payload)

    setup["phases"] = list(reversed(phases))
    with pytest.raises(ValueError, match="ordered as setup then init"):
        CaseSpec.model_validate(payload)


def test_setup_script_contract_renders_explicit_user_and_result_ownership(tmp_path: Path) -> None:
    payload = _case_payload(typed=True)
    setup = payload["setup"]
    assert isinstance(setup, dict)
    phases = setup["phases"]
    assert isinstance(phases, list)
    setup_phase = phases[0]
    assert isinstance(setup_phase, dict)
    recipe = setup_phase["recipes"][0]
    recipe["args"].update(
        {
            "run_as_user": "user",
            "result_owner": "user",
            "result_paths": ["/home/user/Projects/typed-setup/output"],
        }
    )

    materialized = CaseMaterializer(catalog=_catalog()).materialize(
        CaseSuiteSpec.model_validate({"suite_id": "typed-owner", "tasks": [payload]}),
        output_dir=tmp_path,
    )
    command = materialized.cases[0].startup_commands[0].command

    assert 'runuser -u "$northcua_setup_user"' in command
    assert 'USER="$northcua_setup_user"' in command
    assert "/home/user/Projects/typed-setup/output" in command
    assert 'find "$northcua_result_path" -xdev' in command


def test_setup_script_contract_rejects_unpaired_or_unsafe_result_paths() -> None:
    payload = _case_payload(typed=True)
    setup = payload["setup"]
    assert isinstance(setup, dict)
    recipe = setup["phases"][0]["recipes"][0]
    recipe["args"]["result_paths"] = ["/home/user/Projects/typed-setup/output"]
    with pytest.raises(ValueError, match="result_paths and result_owner"):
        CaseSpec.model_validate(payload)

    recipe["args"]["result_owner"] = "user"
    recipe["args"]["result_paths"] = ["/etc/passwd"]
    with pytest.raises(ValueError, match="must be under /home/user"):
        CaseSpec.model_validate(payload)


def test_typed_and_legacy_task_setup_materialize_identical_commands_and_report_phases(tmp_path: Path) -> None:
    materializer = CaseMaterializer(catalog=_catalog())
    typed = materializer.materialize(
        CaseSuiteSpec.model_validate({"suite_id": "typed-suite", "tasks": [_case_payload(typed=True)]}),
        output_dir=tmp_path / "typed",
    )
    legacy = materializer.materialize(
        CaseSuiteSpec.model_validate(
            {
                "suite_id": "legacy-suite",
                "tasks": [_case_payload(typed=False, legacy_setup_in_init=True)],
            }
        ),
        output_dir=tmp_path / "legacy",
    )

    typed_case = typed.cases[0]
    legacy_case = legacy.cases[0]
    assert [command.model_dump(mode="json") for command in typed_case.startup_commands] == [
        command.model_dump(mode="json") for command in legacy_case.startup_commands
    ]
    assert [command.label for command in typed_case.startup_commands] == [
        "run-setup-script",
        "desktop-session-readiness",
        "open-blank-typed.setup.app",
    ]
    task_metadata = typed.manifest.tasks[0].metadata
    assert task_metadata["setup_contract"] == "typed"
    assert [phase["phase"] for phase in task_metadata["setup_phases"]] == ["setup", "init"]
    legacy_metadata = legacy.manifest.tasks[0].metadata
    assert legacy_metadata["setup_contract"] == "legacy"
    assert [phase["phase"] for phase in legacy_metadata["setup_phases"]] == ["setup", "init"]
    assert legacy_metadata["template_app_bindings"]["typed.setup.app"]["provisioning_mode"] == "task_setup"

    batch_dir = tmp_path / "batch"
    report_dir = tmp_path / "report"
    batch_dir.mkdir()
    report_dir.mkdir()
    report_path = CaseReportWriter(report_dir).write(
        materialized=typed,
        batch_result=BatchRunResult(
            batch_id="typed-suite",
            total=1,
            succeeded=1,
            failed=0,
            canceled=0,
            output_dir=str(batch_dir),
            task_states=[BatchTaskState(task_id="typed-setup", status=BatchTaskStatus.SUCCEEDED)],
        ),
    )
    row = json.loads(report_path.read_text(encoding="utf-8"))["cases"][0]
    assert row["setup_contract"] == "typed"
    assert row["setup_phases"] == task_metadata["setup_phases"]


def _catalog() -> CaseCatalog:
    return CaseCatalog.from_raw(
        {
            "templates": [
                {
                    "profile_id": "typed.setup.template",
                    "alias": "typed-setup-template",
                    "provider": "fake",
                    "app_bindings": [
                        {
                            "app_profile": "typed.setup.app",
                            "provisioning_mode": "task_setup",
                        }
                    ],
                }
            ],
            "apps": [
                {
                    "app_profile": "typed.setup.app",
                    "template_profile": "typed.setup.template",
                    "display_name": "Typed Setup App",
                    "launch_binary": "true",
                }
            ],
        }
    )


def _case_payload(*, typed: bool, legacy_setup_in_init: bool = False) -> dict[str, object]:
    setup_recipe = {
        "recipe": "run_setup_script",
        "args": {"path": "/tmp/northcua-cases/typed-setup/install.sh"},
    }
    init_recipe = {"recipe": "open_blank_app", "args": {"app": "typed.setup.app"}}
    payload: dict[str, object] = {
        "task_id": "typed-setup",
        "query": "Open the app after setup.",
        "app_profile": "typed.setup.app",
        "template_profile": "typed.setup.template",
        "fixtures": [
            {
                "generated": {"generator": "setup.script", "params": {"commands": ["true"]}},
                "remote_path": "/tmp/northcua-cases/typed-setup/install.sh",
                "mode": "0755",
                "role": "setup_script",
            }
        ],
    }
    if typed:
        payload["setup"] = {
            "phases": [
                {"phase": "setup", "recipes": [setup_recipe]},
                {"phase": "init", "recipes": [init_recipe]},
            ]
        }
    else:
        if legacy_setup_in_init:
            payload["init"] = [setup_recipe, init_recipe]
        else:
            payload["dependency_injection"] = {"setup_recipes": [setup_recipe]}
            payload["init"] = [init_recipe]
    return payload
