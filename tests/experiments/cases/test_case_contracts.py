from __future__ import annotations

import pytest

from northcua.experiments.cases.contracts import (
    AppProfileId,
    AppProvisioningMode,
    CaseExpectedFeasibility,
    CaseSourceKind,
    CaseSourceSpec,
    CaseSpec,
    CaseSuiteSpec,
    DesktopSessionSpec,
    FixtureGeneratorId,
    FixtureRole,
    FixtureSpec,
    FixtureVisibility,
    GeneratedFixtureSpec,
    GuestCapability,
    InitRecipeSpec,
    LargeArtifactDeliveryMode,
    LargeArtifactFixtureSpec,
    RecipeKind,
    SpreadsheetRowsParams,
    TemplateAppBindingSpec,
    TemplateProfileSpec,
)
from northcua_protocol.capabilities import Capability


def test_case_suite_rejects_duplicate_task_ids() -> None:
    task = CaseSpec(task_id="dup", query="Do it", app_profile=AppProfileId.BROWSER_CHROME)

    with pytest.raises(ValueError, match="unique"):
        CaseSuiteSpec(tasks=[task, task])


def test_fixture_requires_absolute_remote_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        FixtureSpec(source="fixture.txt", remote_path="tmp/fixture.txt")


def test_fixture_source_kind_is_normalized_from_generated() -> None:
    fixture = FixtureSpec(
        generated=GeneratedFixtureSpec(generator=FixtureGeneratorId.TEXT_FILE),
        remote_path="/tmp/northcua-cases/fixture.txt",
    )

    assert fixture.source_kind.value == "generated"
    assert fixture.generated is not None
    assert fixture.generated.params.text


def test_generated_fixture_params_are_typed_by_generator() -> None:
    generated = GeneratedFixtureSpec(
        generator=FixtureGeneratorId.SPREADSHEET_XLSX,
        params={"rows": [["Invoice No.", "Amount"], ["INV-001", 120]]},
    )

    assert isinstance(generated.params, SpreadsheetRowsParams)
    assert generated.params.rows[1][1] == 120


def test_generated_fixture_rejects_unknown_params() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        GeneratedFixtureSpec(generator=FixtureGeneratorId.TEXT_FILE, params={"unknown": "drift"})


def test_init_recipe_coerces_args_by_recipe_kind() -> None:
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_URL,
        args={"app": "browser.chrome", "url": "https://example.com"},
    )

    assert recipe.args.url == "https://example.com"


def test_case_contract_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        CaseSpec(task_id="strict", query="Do it", app_profile=AppProfileId.BROWSER_CHROME, typo=True)


def test_case_contract_defaults_involved_apps_and_feasibility() -> None:
    case = CaseSpec(task_id="task", query="Do it", app_profile=AppProfileId.BROWSER_CHROME)

    assert case.involved_app_profiles == [AppProfileId.BROWSER_CHROME]
    assert case.involved_app_count == 1
    assert case.expected_feasibility == CaseExpectedFeasibility.FEASIBLE


def test_desktop_session_contract_is_typed_and_accepts_migration_aliases() -> None:
    session = DesktopSessionSpec.model_validate(
        {"user": "desktop-user", "target_uid": 1200, "readiness_timeout_seconds": 45}
    )
    case = CaseSpec(
        task_id="desktop-session",
        query="Use the desktop",
        app_profile=AppProfileId.DESKTOP_GENERIC,
        desktop_session=session,
    )

    assert session.required
    assert session.target_user == "desktop-user"
    assert session.target_uid == 1200
    assert case.model_dump(mode="json")["desktop_session"]["target_user"] == "desktop-user"
    with pytest.raises(ValueError, match="valid Linux user"):
        DesktopSessionSpec(target_user="../user")
    with pytest.raises(ValueError, match="less than or equal to 300"):
        DesktopSessionSpec(readiness_timeout_seconds=301)


def test_case_contract_validates_involved_app_count() -> None:
    with pytest.raises(ValueError, match="involved_app_count"):
        CaseSpec(
            task_id="task",
            query="Do it",
            app_profile=AppProfileId.BROWSER_CHROME,
            involved_app_profiles=[AppProfileId.BROWSER_CHROME, AppProfileId.TEXT_EDITOR],
            involved_app_count=1,
        )


def test_case_contract_rejects_unsafe_slugs_and_limits() -> None:
    with pytest.raises(ValueError, match="path-safe"):
        CaseSpec(task_id="../escape", query="Do it", app_profile=AppProfileId.BROWSER_CHROME)
    with pytest.raises(ValueError):
        CaseSuiteSpec(suite_id="../escape", tasks=[CaseSpec(task_id="ok", query="Do it", app_profile="browser.chrome")])
    with pytest.raises(ValueError):
        CaseSpec(task_id="bad-steps", query="Do it", app_profile=AppProfileId.BROWSER_CHROME, max_steps=0)


def test_url_fixture_requires_checksum_and_http_scheme() -> None:
    with pytest.raises(ValueError, match="sha256"):
        FixtureSpec(url="https://example.com/file.txt", remote_path="/tmp/northcua-cases/file.txt")
    with pytest.raises(ValueError, match="http"):
        FixtureSpec(url="file:///etc/passwd", remote_path="/tmp/northcua-cases/passwd", sha256="0" * 64)


def test_fixture_rejects_unsafe_remote_path_prefix() -> None:
    with pytest.raises(ValueError, match="remote_path must be under"):
        FixtureSpec(
            generated=GeneratedFixtureSpec(generator=FixtureGeneratorId.TEXT_FILE),
            remote_path="/etc/profile.d/escape.sh",
        )
    with pytest.raises(ValueError, match="remote_path must be under"):
        FixtureSpec(
            generated=GeneratedFixtureSpec(generator=FixtureGeneratorId.TEXT_FILE),
            remote_path="/home/user/.bashrc",
        )


def test_fixture_accepts_safe_remote_path_prefix_root() -> None:
    fixture = FixtureSpec(
        generated=GeneratedFixtureSpec(generator=FixtureGeneratorId.TEXT_FILE),
        remote_path="/home/user/project",
    )

    assert fixture.remote_path == "/home/user/project"


def test_fixture_contract_types_guest_ownership_and_visibility() -> None:
    fixture = FixtureSpec(
        source="fixture.txt",
        remote_path="/tmp/northcua-cases/fixture.txt",
        mode="0640",
        owner="user",
        group="users",
        parent_mode="0750",
        parent_owner="root",
        parent_group="users",
        visibility="evaluator",
    )

    assert fixture.visibility == FixtureVisibility.EVALUATOR
    assert fixture.owner == "user"
    assert fixture.parent_mode == "0750"
    with pytest.raises(ValueError, match="owner/group"):
        FixtureSpec(
            source="fixture.txt",
            remote_path="/tmp/northcua-cases/fixture.txt",
            owner="user:root",
        )
    with pytest.raises(ValueError, match="shared system directory"):
        FixtureSpec(
            source="fixture.txt",
            remote_path="/tmp/northcua-fixture.txt",
            parent_owner="user",
        )


def test_expected_fixture_requires_non_guest_visibility() -> None:
    with pytest.raises(ValueError, match="expected fixtures.*evaluator or host"):
        FixtureSpec(
            source="expected.json",
            remote_path="/tmp/northcua-cases/expected.json",
            role=FixtureRole.EXPECTED,
        )

    fixture = FixtureSpec(
        source="expected.json",
        remote_path="/tmp/northcua-cases/expected.json",
        role=FixtureRole.EXPECTED,
        visibility=FixtureVisibility.EVALUATOR,
    )

    assert fixture.visibility == FixtureVisibility.EVALUATOR


def test_large_artifact_contract_requires_content_addressed_delivery_source() -> None:
    artifact = LargeArtifactFixtureSpec(
        sha256="a" * 64,
        size_bytes=100 * 1024 * 1024,
        delivery=LargeArtifactDeliveryMode.GUEST_DOWNLOAD,
    )
    fixture = FixtureSpec(
        url="https://artifacts.example.test/blobs/a",
        remote_path="/tmp/northcua-cases/large.bin",
        large_artifact=artifact,
    )

    assert fixture.sha256 == "a" * 64
    assert fixture.large_artifact == artifact
    with pytest.raises(ValueError, match="guest_download.*URL"):
        FixtureSpec(
            source="large.bin",
            remote_path="/tmp/northcua-cases/large.bin",
            large_artifact=artifact,
        )
    with pytest.raises(ValueError, match="source_path"):
        FixtureSpec(
            url="https://artifacts.example.test/blobs/a",
            remote_path="/tmp/northcua-cases/large.bin",
            artifact={
                "sha256": "a" * 64,
                "size_bytes": 100 * 1024 * 1024,
                "delivery": "chunked_upload",
            },
        )
    with pytest.raises(ValueError, match="less than or equal to"):
        LargeArtifactFixtureSpec(
            sha256="a" * 64,
            size_bytes=100 * 1024 * 1024,
            delivery=LargeArtifactDeliveryMode.CHUNKED_UPLOAD,
            chunk_size_bytes=50 * 1024 * 1024 + 1,
        )


def test_template_app_binding_supports_nested_and_flattened_post_setup_probes() -> None:
    nested = TemplateAppBindingSpec.model_validate(
        {
            "app_profile": "custom.editor",
            "provisioning_mode": "task_setup",
            "first_run_variant": "osworld2",
            "post_setup": {
                "binary_probes": ["custom-editor"],
                "version_probes": ["custom-editor --version"],
            },
        }
    )
    flattened = TemplateAppBindingSpec.model_validate(
        {
            "app_profile": "custom.editor",
            "post_setup_binary_probes": "custom-editor",
            "post_setup_version_probes": ["custom-editor --version"],
        }
    )

    assert nested.provisioning_mode == AppProvisioningMode.TASK_SETUP
    assert nested.post_setup.version_probes == ["custom-editor --version"]
    assert flattened.post_setup_binary_probes == ["custom-editor"]
    mapped_template = TemplateProfileSpec.model_validate(
        {
            "profile_id": "mapped.template",
            "alias": "mapped-template",
            "app_bindings": {"custom.editor": {"provisioning_mode": "task_setup"}},
        }
    )
    assert mapped_template.app_bindings[0].app_profile == AppProfileId("custom.editor")
    with pytest.raises(ValueError, match="unique"):
        TemplateProfileSpec(
            profile_id="custom.template",
            alias="custom-template",
            app_bindings=[nested, nested],
        )
    with pytest.raises(ValueError, match="shell operators"):
        TemplateAppBindingSpec.model_validate(
            {
                "app_profile": "custom.editor",
                "post_setup": {"version_probes": ["custom-editor --version; touch /tmp/unsafe"]},
            }
        )


def test_guest_capability_is_distinct_from_protocol_capability() -> None:
    case = CaseSpec(
        task_id="guest-capability",
        query="Use the guest app.",
        app_profile="custom.editor",
        required_guest_capabilities={"shell", "custom.service"},
        required_capabilities={Capability.SHELL_EXEC},
    )

    assert GuestCapability.SHELL in case.required_guest_capabilities
    assert GuestCapability("custom.service") in case.required_guest_capabilities
    assert all(not isinstance(capability, Capability) for capability in case.required_guest_capabilities)


def test_case_suite_allows_custom_provider_ids_for_registered_extensions() -> None:
    task = CaseSpec(task_id="task", query="Do it", app_profile="browser.chrome")

    suite = CaseSuiteSpec.model_validate({"suite_id": "suite", "sandbox": {"provider": "custom"}, "tasks": [task]})

    assert suite.sandbox.provider == "custom"


def test_case_source_contract_accepts_typed_directory_and_string_file_refs() -> None:
    file_source = CaseSourceSpec.model_validate("cases/browser.case.yaml")
    dir_source = CaseSourceSpec(kind=CaseSourceKind.DIRECTORY, path="cases", glob="*.case.yaml")

    assert file_source.kind == CaseSourceKind.FILE
    assert dir_source.kind == CaseSourceKind.DIRECTORY
    with pytest.raises(ValueError, match="glob"):
        CaseSourceSpec(kind=CaseSourceKind.JSONL, path="cases.jsonl", glob="*.jsonl")
