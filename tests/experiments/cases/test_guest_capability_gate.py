from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from northcua.batch.contracts import (
    BatchManifest,
    BatchRunResult,
    BatchTaskSpec,
    BatchTaskState,
    BatchTaskStatus,
)
from northcua.batch.preparation.preparer import _recipe_failure_details
from northcua.batch.preparation.startup import _is_fixture_setup_command
from northcua.experiments.cases.catalog import CaseCatalog
from northcua.experiments.cases.contracts import (
    AppProfileId,
    CaseSpec,
    CaseSuiteSpec,
    DesktopSessionSpec,
    GuestCapability,
    TemplateProfileId,
)
from northcua.experiments.cases.guest_capabilities import (
    GuestCapabilityGateRenderer,
    parse_guest_capability_evidence,
)
from northcua.experiments.cases.materialization import (
    CaseMaterializer,
    MaterializedCase,
    MaterializedCaseSuite,
)
from northcua.experiments.cases.reporting import CaseReportWriter
from northcua.sandboxes.contracts import (
    SandboxCommandResult,
    SandboxCommandSpec,
    SandboxHandle,
    SandboxProviderSpec,
)
from northcua.sandboxes.guest_capability_evidence import (
    GUEST_CAPABILITY_EVIDENCE_PREFIX,
    GUEST_CAPABILITY_GATE_LABEL,
    guest_capability_evidence_payloads,
)
from northcua.sandboxes.lifecycle.manager import SandboxLifecycleManager
from northcua.sandboxes.provider.fake import FakeSandboxProvider

_TEMPLATE_ID = TemplateProfileId("probe.desktop")
_APP_ID = AppProfileId("custom.app")


def _catalog(*, template_capabilities: list[str] | None = None) -> CaseCatalog:
    return CaseCatalog.from_raw(
        {
            "templates": [
                {
                    "profile_id": "probe.desktop",
                    "alias": "probe-template",
                    "provider": "fake",
                    "required_guest_capabilities": template_capabilities or [],
                    "app_bindings": [
                        {
                            "app_profile": "custom.app",
                            "required_guest_capabilities": ["linux.snap"],
                        }
                    ],
                }
            ],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "probe.desktop",
                    "display_name": "Custom",
                    "launch_binary": "true",
                    "required_guest_capabilities": ["kernel.fanotify"],
                }
            ],
        }
    )


def test_renderer_covers_every_builtin_with_runtime_operations() -> None:
    renderer = GuestCapabilityGateRenderer()
    capabilities = set(GuestCapability)

    command = renderer.render(
        capabilities,
        desktop_session=DesktopSessionSpec(target_user="user", readiness_timeout_seconds=1),
        python_executable="python3.12",
    )

    assert command is not None
    assert command.label == GUEST_CAPABILITY_GATE_LABEL
    assert all(capability.value in command.command for capability in capabilities)
    for actual_operation in (
        "--appimage-mount",
        "fusermount",
        "XGetSelectionOwner",
        "snap list",
        "snapd.socket",
        "/snap/audacity/1212",
        "snap run --shell audacity",
        "org.freedesktop.portal.Desktop",
        "docker run --rm",
        "docker compose",
        "cgroup.controllers",
        "systemctl list-units",
        "fanotify_init",
        "base64.b64decode",
        'export WINEPREFIX="$probe_root/prefix"',
        "DXGIDeclareAdapterRemovalSupport",
        "glxinfo -B",
        "vulkaninfo --summary",
        "CapEff",
        "mount --bind",
        "1500000",
        "1800000",
        "findmnt --noheadings --output TARGET,SOURCE,FSTYPE --target /home/user",
        "org.freedesktop.DBus.ListNames",
    ):
        assert actual_operation in command.command
    assert "snap refresh" not in command.command
    assert "clock_settime" not in command.command
    assert "XSetSelectionOwner" not in command.command
    assert "xclip -selection clipboard" not in command.command
    assert all(inventory_probe not in command.command for inventory_probe in ("apt-cache", "dpkg-query", "rpm -q"))
    rendered_script = shlex.split(command.command)[2]
    syntax = subprocess.run(["bash", "-n", "-c", rendered_script], text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr


def test_unpinned_snap_systemd_probe_fails_closed_after_daemon_checks() -> None:
    command = GuestCapabilityGateRenderer().render(
        {GuestCapability.SNAP_SYSTEMD},
        desktop_session=None,
        python_executable="python3",
    )

    assert command is not None
    assert "snap-systemd-unpinned-fail-closed" in command.command
    assert "cat /proc/1/comm" in command.command
    assert "LoadState" in command.command
    assert "snapd.service" in command.command
    assert "systemctl is-active --quiet snapd.socket" in command.command
    assert "timeout 20s snap list" in command.command
    assert "requires an app-and-revision-specific capability id" in command.command
    assert "false" in command.command
    assert "snap refresh" not in command.command


def test_pinned_snap_probes_check_revision_mount_and_launch_without_refresh() -> None:
    command = GuestCapabilityGateRenderer().render(
        {
            GuestCapability.SNAP_SYSTEMD_FIREFOX_4793,
            GuestCapability.SNAP_SYSTEMD_AUDACITY_1212,
        },
        desktop_session=None,
        python_executable="python3",
    )

    assert command is not None
    assert "snap-audacity-1212-pinned-mount-launch" in command.command
    assert "snap list audacity" in command.command
    assert "snap_mount=/snap/audacity/1212" in command.command
    assert "snap-firefox-4793-pinned-mount-launch" in command.command
    assert "snap list firefox" in command.command
    assert "snap_mount=/snap/firefox/4793" in command.command
    assert 'mountpoint -q "$snap_mount"' in command.command
    assert "FSTYPE" in command.command
    assert "squashfs" in command.command
    assert "snap run --shell audacity" in command.command
    assert "snap run --shell firefox" in command.command
    assert 'HOME="$probe_root/home"' in command.command
    assert "snap refresh" not in command.command


def test_appimage_and_disk_probes_exercise_runtime_not_package_inventory() -> None:
    command = GuestCapabilityGateRenderer().render(
        {
            GuestCapability.APPIMAGE_FUSE,
            GuestCapability.DISK_FREE_1500_MB,
            GuestCapability.DISK_FREE_1800_MB,
        },
        desktop_session=DesktopSessionSpec(target_user="user", readiness_timeout_seconds=1),
        python_executable="python3.12",
    )

    assert command is not None
    assert "appimage-fuse-user-mount-unmount" in command.command
    assert "--appimage-mount" in command.command
    assert "fusermount3" in command.command
    assert 'mountpoint -q "$mount_path"' in command.command
    assert '"$unmount_helper" -u "$mount_path"' in command.command
    assert "disk-free-home-filesystem-1500mb" in command.command
    assert "disk-free-home-filesystem-1800mb" in command.command
    assert "findmnt --noheadings --output TARGET,SOURCE,FSTYPE --target /home/user" in command.command
    assert "os.statvfs" in command.command
    assert "df -Pk" not in command.command
    assert "os.open('/dev/fuse'" not in command.command
    assert all(name not in command.command for name in ("apt-cache", "dpkg-query", "rpm -q"))


def test_clipboard_probe_never_writes_or_claims_the_selection() -> None:
    command = GuestCapabilityGateRenderer().render(
        {GuestCapability.CLIPBOARD},
        desktop_session=DesktopSessionSpec(target_user="user", readiness_timeout_seconds=1),
        python_executable="python3.12",
    )

    assert command is not None
    assert "x11-clipboard-selection-query" in command.command
    assert "XGetSelectionOwner" in command.command
    assert "clipboard=x11.XInternAtom" in command.command
    assert "CLIPBOARD" in command.command
    assert ",1)" in command.command
    assert "XSetSelectionOwner" not in command.command
    assert "xclip" not in command.command
    assert "xsel" not in command.command
    assert "sentinel" not in command.command


def test_cap_sys_time_probe_uses_effective_capability_evidence_only() -> None:
    command = GuestCapabilityGateRenderer().render(
        {GuestCapability.CAP_SYS_TIME},
        desktop_session=None,
        python_executable="python3.12",
    )

    assert command is not None
    assert "effective-cap-sys-time" in command.command
    assert "CapEff" in command.command
    assert "Seccomp" in command.command
    assert "1 << 25" in command.command
    assert "clock_settime" not in command.command
    assert "clock_gettime" not in command.command


def test_wine_dxvk_probe_isolates_home_and_prefix_and_cleans_up() -> None:
    command = GuestCapabilityGateRenderer().render(
        {GuestCapability.WINE_DXVK},
        desktop_session=DesktopSessionSpec(target_user="user", readiness_timeout_seconds=1),
        python_executable="python3.12",
    )

    assert command is not None
    assert "wine-dxvk-isolated-prefix" in command.command
    assert "probe_root=$(mktemp -d)" in command.command
    assert 'export HOME="$probe_root/home"' in command.command
    assert 'export WINEPREFIX="$probe_root/prefix"' in command.command
    assert 'rm -rf "$probe_root"' in command.command
    assert "wineboot -u" in command.command
    assert "setup_dxvk.sh" in command.command
    assert "${WINEPREFIX:-$HOME/.wine}" not in command.command


def test_docker_capabilities_are_separate_supported_runtime_probes() -> None:
    command = GuestCapabilityGateRenderer().render(
        {
            GuestCapability.SYSTEMD,
            GuestCapability.CGROUPS,
            GuestCapability.NESTED_DOCKER,
            GuestCapability.DOCKER_COMPOSE,
        },
        desktop_session=None,
        python_executable="python3",
    )

    assert command is not None
    assert "systemd-manager-call" in command.command
    assert "cgroup-filesystem" in command.command
    assert "nested-docker-container" in command.command
    assert "docker-compose-container" in command.command
    assert "docker run --rm --pull=never hello-world" in command.command
    assert "docker compose" in command.command
    assert "pull_policy: never" in command.command
    assert "down --remove-orphans" in command.command


def test_docker_case_materializes_after_only_evidence_blocker_is_removed(tmp_path: Path) -> None:
    case = CaseSpec.model_validate(
        {
            "task_id": "docker-capability-probe",
            "query": "Verify the synthetic Docker workflow.",
            "app_profile": "custom.app",
            "template_profile": "probe.desktop",
            "required_guest_capabilities": [
                "nested.docker",
                "systemd",
                "cgroups",
                "docker.compose",
            ],
            "fixtures": [
                {
                    "generated": {"generator": "setup.script", "params": {"commands": ["true"]}},
                    "remote_path": "/tmp/northcua-cases/docker-capability-probe/task-setup.sh",
                    "mode": "0755",
                    "owner": "user",
                    "group": "user",
                    "parent_mode": "0755",
                    "parent_owner": "user",
                    "parent_group": "user",
                    "visibility": "guest",
                    "role": "setup_script",
                }
            ],
            "dependency_injection": {
                "setup_recipes": [
                    {
                        "recipe": "run_setup_script",
                        "args": {"path": "/tmp/northcua-cases/docker-capability-probe/task-setup.sh"},
                    }
                ]
            },
        }
    )
    suite = CaseSuiteSpec(suite_id="docker-capability-probe", tasks=[case])

    materialized = CaseMaterializer(catalog=_catalog()).materialize(
        suite,
        output_dir=tmp_path,
        spec_dir=tmp_path,
    )

    assert len(materialized.manifest.tasks) == 1
    assert {
        "nested.docker",
        "systemd",
        "cgroups",
        "docker.compose",
    }.issubset({capability.value for capability in materialized.cases[0].case.required_guest_capabilities})


def test_shell_probe_executes_and_emits_available_evidence() -> None:
    command = GuestCapabilityGateRenderer().render(
        {GuestCapability.SHELL},
        desktop_session=None,
        python_executable="python3",
    )
    assert command is not None

    result = subprocess.run(
        command.command,
        shell=True,
        executable="/bin/bash",
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert [item.model_dump(mode="json") for item in parse_guest_capability_evidence(result.stdout)] == [
        {
            "capability": "shell",
            "status": "available",
            "probe": "bash-execution",
            "exit_code": 0,
        }
    ]


def test_materializer_unions_requirements_and_prepends_gate(tmp_path: Path) -> None:
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "guest-capability-union",
            "tasks": [
                {
                    "task_id": "probe-case",
                    "query": "Use the app.",
                    "app_profile": "custom.app",
                    "required_guest_capabilities": ["shell"],
                }
            ],
        }
    )

    materialized = CaseMaterializer(catalog=_catalog(template_capabilities=["linux.fuse"])).materialize(
        suite,
        output_dir=tmp_path,
    )

    case = materialized.cases[0]
    task = materialized.manifest.tasks[0]
    expected = {"kernel.fanotify", "linux.fuse", "linux.snap", "shell"}
    assert {capability.value for capability in case.case.required_guest_capabilities} == expected
    assert task.metadata["required_guest_capabilities"] == sorted(expected)
    assert task.sandbox is not None
    assert task.sandbox.metadata["required_guest_capabilities"] == sorted(expected)
    assert task.startup_commands[0].label == GUEST_CAPABILITY_GATE_LABEL
    assert task.startup_commands[1].label == "desktop-session-readiness"


def test_materializer_rejects_unprobeable_requirement_before_output(tmp_path: Path) -> None:
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "unsupported-guest-capability",
            "tasks": [
                {
                    "task_id": "probe-case",
                    "query": "Use the app.",
                    "app_profile": "custom.app",
                    "required_guest_capabilities": ["custom.unprobeable"],
                }
            ],
        }
    )
    output_dir = tmp_path / "materialized"

    with pytest.raises(ValueError, match="no runtime probe.*custom.unprobeable"):
        CaseMaterializer(catalog=_catalog()).materialize(suite, output_dir=output_dir)

    assert not output_dir.exists()


def test_materializer_rejects_desktop_probe_with_disabled_session_before_output(tmp_path: Path) -> None:
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "disabled-desktop-session",
            "tasks": [
                {
                    "task_id": "probe-case",
                    "query": "Use the app.",
                    "app_profile": "custom.app",
                    "desktop_session": {"required": False},
                    "required_guest_capabilities": ["desktop.clipboard"],
                }
            ],
        }
    )
    output_dir = tmp_path / "materialized"

    with pytest.raises(ValueError, match="require a live desktop session.*desktop.clipboard"):
        CaseMaterializer(catalog=_catalog()).materialize(suite, output_dir=output_dir)

    assert not output_dir.exists()


def test_legacy_case_without_requirements_has_no_capability_gate(tmp_path: Path) -> None:
    catalog = _catalog()
    raw = catalog.templates[_TEMPLATE_ID].model_dump(mode="json")
    raw["app_bindings"][0]["required_guest_capabilities"] = []
    app = catalog.apps[_APP_ID].model_dump(mode="json")
    app["required_guest_capabilities"] = []
    legacy_catalog = CaseCatalog.from_raw({"templates": [raw], "apps": [app]})
    suite = CaseSuiteSpec.model_validate(
        {
            "suite_id": "legacy-no-gate",
            "tasks": [{"task_id": "legacy", "query": "Use it.", "app_profile": "custom.app"}],
        }
    )

    materialized = CaseMaterializer(catalog=legacy_catalog).materialize(suite, output_dir=tmp_path)

    assert [command.label for command in materialized.cases[0].startup_commands] == ["desktop-session-readiness"]


class _RejectingGateProvider(FakeSandboxProvider):
    async def run_command(self, handle: SandboxHandle, command: SandboxCommandSpec) -> SandboxCommandResult:
        self.commands.append((handle.sandbox_id, command))
        if command.label == GUEST_CAPABILITY_GATE_LABEL:
            available = (
                f'{GUEST_CAPABILITY_EVIDENCE_PREFIX}{{"capability":"shell","status":"available",'
                '"probe":"bash-execution","exit_code":0}'
            )
            unavailable = (
                f'{GUEST_CAPABILITY_EVIDENCE_PREFIX}{{"capability":"linux.fuse","status":"unavailable",'
                '"probe":"fuse-appimage-mount-unmount","exit_code":1}'
            )
            return SandboxCommandResult(exit_code=1, stdout=available, stderr=unavailable)
        return SandboxCommandResult(stdout="unexpected")


@pytest.mark.asyncio
async def test_lifecycle_stops_before_init_and_reports_structured_evidence() -> None:
    provider = _RejectingGateProvider()
    gate = SandboxCommandSpec(command="probe", label=GUEST_CAPABILITY_GATE_LABEL)
    init = SandboxCommandSpec(command="init", label="case-init")

    with pytest.raises(RuntimeError, match="failure_kind=guest_capability_unavailable") as exc_info:
        await SandboxLifecycleManager(provider).acquire(
            SandboxProviderSpec(provider="fake", template="desktop"),
            startup_commands=[gate, init],
        )

    assert [command.label for _, command in provider.commands] == [GUEST_CAPABILITY_GATE_LABEL]
    capability_evidence = guest_capability_evidence_payloads(str(exc_info.value))
    assert {item["capability"]: item for item in capability_evidence} == {
        "shell": {
            "capability": "shell",
            "status": "available",
            "probe": "bash-execution",
            "exit_code": 0,
        },
        "linux.fuse": {
            "capability": "linux.fuse",
            "status": "unavailable",
            "probe": "fuse-appimage-mount-unmount",
            "exit_code": 1,
        },
    }
    assert _recipe_failure_details(str(exc_info.value))["guest_capability_evidence"] == (capability_evidence)


def test_prepared_sandbox_treats_gate_as_bootstrap_command() -> None:
    command = SandboxCommandSpec(command="probe", label=GUEST_CAPABILITY_GATE_LABEL)

    assert _is_fixture_setup_command(command)


def test_case_report_projects_structured_capability_failure(tmp_path: Path) -> None:
    case = CaseSpec(task_id="case-1", query="Use it.", app_profile="custom.app")
    sandbox = SandboxProviderSpec(provider="fake", template="desktop")
    materialized = MaterializedCaseSuite(
        spec=CaseSuiteSpec(suite_id="capability-report", tasks=[case]),
        output_dir=tmp_path,
        fixture_dir=tmp_path / "fixtures",
        cases=(MaterializedCase(case=case, fixtures=(), startup_commands=(), sandbox=sandbox),),
        manifest=BatchManifest(
            batch_id="capability-report",
            tasks=[BatchTaskSpec(task_id="case-1", agent_kind="desktop", query="Use it.", sandbox=sandbox)],
        ),
    )
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    evidence = (
        f'{GUEST_CAPABILITY_EVIDENCE_PREFIX}{{"capability":"linux.fuse","status":"unavailable",'
        '"probe":"fuse-appimage-mount-unmount","exit_code":1}'
    )

    report_path = CaseReportWriter(tmp_path).write(
        materialized=materialized,
        batch_result=BatchRunResult(
            batch_id="capability-report",
            total=1,
            succeeded=0,
            failed=1,
            canceled=0,
            output_dir=str(batch_dir),
            task_states=[
                BatchTaskState(
                    task_id="case-1",
                    status=BatchTaskStatus.FAILED,
                    error=f"failure_kind=guest_capability_unavailable {evidence}",
                )
            ],
        ),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["cases"][0]["guest_capability_evidence"] == [
        {
            "capability": "linux.fuse",
            "status": "unavailable",
            "probe": "fuse-appimage-mount-unmount",
            "exit_code": 1,
        }
    ]
