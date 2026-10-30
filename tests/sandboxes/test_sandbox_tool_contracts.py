from __future__ import annotations

from pathlib import Path

import pytest

from northcua.sandboxes.contracts import (
    SandboxCommandSpec,
    SandboxDeviceBinding,
    SandboxFileUploadSpec,
    SandboxHandle,
    SandboxLease,
    SandboxLifecycleState,
    SandboxProviderSpec,
)
from northcua.sandboxes.device.bootstrap import SandboxDeviceBootstrapper
from northcua.sandboxes.lifecycle.manager import SandboxLifecycleManager
from northcua.sandboxes.lifecycle.task_lease import SandboxTaskLeaseManager
from northcua.sandboxes.provider.fake import FakeSandboxProvider
from northcua.sandboxes.tool_contracts import (
    SANDBOX_PROVIDER_CONTRACT_BY_METHOD,
    SANDBOX_PROVIDER_TOOL_CONTRACTS,
    prepare_sandbox_provider_dispatch,
)
from northcua_runtime_shared.tool_contracts import (
    ToolAudience,
    ToolDispatcher,
    ToolDispatchPolicy,
    ToolProfile,
    ToolSurfaceConfig,
    ToolSurfaceResolver,
    ToolVisibility,
)


def test_sandbox_provider_contracts_are_operator_only_control_plane_tools() -> None:
    assert SANDBOX_PROVIDER_TOOL_CONTRACTS

    for contract in SANDBOX_PROVIDER_TOOL_CONTRACTS:
        assert contract.audience == ToolAudience.ENV_BUILDER
        assert contract.visibility == ToolVisibility.OPERATOR_ONLY
        assert contract.model_visible is False
        assert contract.binding is not None
        assert contract.binding.kind.value == "sandbox_provider"


def test_sandbox_provider_contracts_cover_control_plane_provider_methods() -> None:
    assert set(SANDBOX_PROVIDER_CONTRACT_BY_METHOD) == {
        "preflight",
        "create",
        "connect",
        "run_command",
        "start_process",
        "upload",
        "download",
        "collect_logs",
        "terminate",
        "extend_timeout",
        "build_template",
        "probe",
        "capacity_probe",
    }


def test_sandbox_provider_contracts_do_not_project_to_model_surface() -> None:
    surface = ToolSurfaceResolver(SANDBOX_PROVIDER_TOOL_CONTRACTS).resolve(
        ToolProfile(
            name="env.sandbox.control_plane",
            audiences=frozenset({ToolAudience.ENV_BUILDER}),
            visibility=frozenset({ToolVisibility.OPERATOR_ONLY}),
        ),
        config=ToolSurfaceConfig(include_visibility=frozenset({ToolVisibility.OPERATOR_ONLY})),
    )

    assert len(surface.validation_descriptors) == len(SANDBOX_PROVIDER_TOOL_CONTRACTS)
    assert surface.provider_tools == ()
    assert surface.fingerprints["contract_summary_hash"]


def test_sandbox_provider_dispatch_requires_operator_policy() -> None:
    contract = next(
        contract for contract in SANDBOX_PROVIDER_TOOL_CONTRACTS if contract.tool_id == "sandbox_run_command"
    )
    payload = {
        "handle_id": "sandbox-1",
        "command": SandboxCommandSpec(command="echo ok").model_dump(mode="json"),
    }

    with pytest.raises(PermissionError, match="operator-only"):
        ToolDispatcher().prepare(contract, payload)

    envelope = ToolDispatcher().prepare(
        contract,
        payload,
        policy=ToolDispatchPolicy(allow_operator=True),
    )

    assert envelope.route["provider_method"] == "run_command"
    assert envelope.audit["visibility"] == ToolVisibility.OPERATOR_ONLY.value


def test_sandbox_provider_dispatch_rejects_unknown_provider_method() -> None:
    with pytest.raises(LookupError, match="has no ToolContract"):
        prepare_sandbox_provider_dispatch("missing_method", {})


@pytest.mark.asyncio
async def test_sandbox_task_lease_manager_validates_extend_timeout_before_provider_call() -> None:
    provider = FakeSandboxProvider()
    manager = SandboxTaskLeaseManager(provider)
    lease = SandboxLease(
        lease_id="lease-1",
        handle=SandboxHandle(
            sandbox_id="sandbox-1",
            provider="fake",
            template="ubuntu",
            state=SandboxLifecycleState.CREATED,
        ),
        device_id="device-1",
    )

    with pytest.raises(ValueError, match="greater than 0"):
        await manager.extend_timeout(lease, timeout_seconds=0)

    assert provider.extended_timeouts == []


@pytest.mark.asyncio
async def test_sandbox_lifecycle_manager_prepares_provider_dispatch_for_acquire_and_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = FakeSandboxProvider()
    manager = SandboxLifecycleManager(provider)
    upload_source = tmp_path / "payload.txt"
    upload_source.write_text("payload", encoding="utf-8")
    prepared_methods: list[str] = []

    def _record_prepare(provider_method: str, parameters, *, policy=None):
        prepared_methods.append(provider_method)
        return prepare_sandbox_provider_dispatch(provider_method, parameters, policy=policy)

    monkeypatch.setattr("northcua.sandboxes.tool_contracts.prepare_sandbox_provider_dispatch", _record_prepare)

    handle = await manager.acquire(
        SandboxProviderSpec(
            provider="fake",
            template="ubuntu",
            file_uploads=[
                SandboxFileUploadSpec(
                    local_path=upload_source,
                    remote_path="/tmp/payload.txt",
                    sha256="0" * 64,
                    mode="0644",
                )
            ],
        ),
        startup_commands=[SandboxCommandSpec(command="echo startup", label="startup")],
    )
    await manager.release(
        handle,
        strategy=SandboxProviderSpec().allocation_strategy,
        teardown_commands=[SandboxCommandSpec(command="echo teardown", label="teardown", allow_failure=True)],
    )

    assert prepared_methods == [
        "preflight",
        "create",
        "run_command",
        "run_command",
        "run_command",
        "upload",
        "run_command",
        "run_command",
        "run_command",
        "run_command",
        "terminate",
    ]


@pytest.mark.asyncio
async def test_sandbox_device_bootstrapper_prepares_start_process_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeSandboxProvider()
    bootstrapper = SandboxDeviceBootstrapper(provider)
    handle = SandboxHandle(
        sandbox_id="sandbox-1",
        provider="fake",
        template="ubuntu",
        state=SandboxLifecycleState.CREATED,
    )
    prepared_methods: list[str] = []

    def _record_prepare(provider_method: str, parameters, *, policy=None):
        prepared_methods.append(provider_method)
        return prepare_sandbox_provider_dispatch(provider_method, parameters, policy=policy)

    monkeypatch.setattr("northcua.sandboxes.tool_contracts.prepare_sandbox_provider_dispatch", _record_prepare)

    await bootstrapper.start_worker(
        handle,
        SandboxDeviceBinding(
            device_id="device-1",
            display_name="Sandbox Device",
            server_url="ws://controller.example.test/ws",
        ),
    )

    assert prepared_methods == ["start_process"]
    assert provider.commands
