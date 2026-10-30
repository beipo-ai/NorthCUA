from __future__ import annotations

import pytest

from northcua.sandboxes.contracts import SandboxDeviceBinding, SandboxReadinessState
from northcua.sandboxes.device import (
    ExternalDeviceBootstrapReceipt,
    ExternalDeviceBootstrapService,
    build_external_device_bootstrap_plan,
)
from northcua.sandboxes.device.readiness import DelayedDeviceReadinessProbe, NoopDeviceReadinessProbe
from northcua_protocol.capabilities import Capability, PlatformKind


class _Adapter:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.calls = 0
        self.mismatch = mismatch

    async def ensure_started(self, plan):  # type: ignore[no-untyped-def]
        self.calls += 1
        return ExternalDeviceBootstrapReceipt(
            operation_id=plan.operation_id,
            runtime_session_ref=plan.runtime_session_ref,
            service_key="wrong" if self.mismatch else plan.service_key,
            device_id=plan.device_id,
            plan_sha256=plan.plan_sha256,
            state="started" if self.calls == 1 else "already_started",
        )


def _binding() -> SandboxDeviceBinding:
    return SandboxDeviceBinding(
        device_id="device-1",
        display_name="desktop member",
        server_url="wss://controller.test/device",
        platform=PlatformKind.LINUX,
        capabilities={Capability.DESKTOP_INPUT, Capability.DESKTOP_SCREENSHOT},
        agentkit_agent_kinds={"desktop"},
        allowed_tools={"desktop.screenshot"},
        environment={"PUBLIC_SETTING": "enabled"},
    )


def test_plan_is_deterministic_and_never_serializes_device_credential() -> None:
    kwargs = {
        "operation_id": "bootstrap-1",
        "runtime_session_ref": "runtime://session-1",
        "binding": _binding(),
        "device_credential_ref": "env:NORTHCUA_DEVICE_TOKEN",
    }
    first = build_external_device_bootstrap_plan(**kwargs)
    second = build_external_device_bootstrap_plan(**kwargs)
    encoded = first.model_dump_json()

    assert first == second
    assert first.plan_sha256 == second.plan_sha256
    assert "--token" not in first.argv
    assert "env:NORTHCUA_DEVICE_TOKEN" in encoded
    assert "NORTHCUA_DEVICE_TOKEN" not in first.environment
    assert first.secret_environment_refs == {"NORTHCUA_DEVICE_TOKEN": "env:NORTHCUA_DEVICE_TOKEN"}
    assert first.environment_inheritance == "desktop_session"


def test_non_desktop_plan_does_not_request_graphical_session_environment() -> None:
    plan = build_external_device_bootstrap_plan(
        operation_id="bootstrap-web",
        runtime_session_ref="runtime://session-web",
        binding=_binding().model_copy(
            update={
                "platform": PlatformKind.BROWSER,
                "capabilities": {Capability.BROWSER_PAGE},
                "agentkit_agent_kinds": {"web"},
            }
        ),
        device_credential_ref="env:NORTHCUA_DEVICE_TOKEN",
    )

    assert plan.environment_inheritance == "none"


def test_plan_can_inject_secret_controller_route_without_serializing_url() -> None:
    sentinel = "wss://gateway.test/proxy/route/?apikey=secret-route-key"
    plan = build_external_device_bootstrap_plan(
        operation_id="bootstrap-1",
        runtime_session_ref="runtime://session-1",
        binding=_binding().model_copy(update={"server_url": "wss://gateway.test/proxy/route/"}),
        device_credential_ref="env:NORTHCUA_DEVICE_TOKEN",
        server_url_credential_ref="file-secret:///run/secrets/controller-public-url",
    )

    assert plan.secret_environment_refs == {
        "NORTHCUA_DEVICE_SERVER": "file-secret:///run/secrets/controller-public-url",
        "NORTHCUA_DEVICE_TOKEN": "env:NORTHCUA_DEVICE_TOKEN",
    }
    assert plan.argv[plan.argv.index("--server") + 1] == ""
    assert sentinel not in plan.model_dump_json()


def test_plan_rejects_inline_device_token() -> None:
    with pytest.raises(ValueError, match="secret reference"):
        build_external_device_bootstrap_plan(
            operation_id="bootstrap-1",
            runtime_session_ref="runtime://session-1",
            binding=_binding().model_copy(update={"token": "inline-secret"}),
            device_credential_ref="secret://device/member-1",
        )


@pytest.mark.parametrize("credential_ref", ["inline-secret", "", "https://secret.test/value"])
def test_plan_rejects_inline_or_unapproved_device_credential_ref(credential_ref: str) -> None:
    sentinel = "device-secret-sentinel"
    with pytest.raises(ValueError, match="approved secret reference") as error:
        build_external_device_bootstrap_plan(
            operation_id="bootstrap-1",
            runtime_session_ref="runtime://session-1",
            binding=_binding(),
            device_credential_ref=credential_ref or sentinel,
        )
    assert sentinel not in str(error.value)


@pytest.mark.parametrize("key", ["E2B_API_KEY", "CUSTOM_TOKEN", "PRIVATE_CREDENTIAL", "DB_PASSWORD"])
def test_plan_rejects_sensitive_public_environment(key: str) -> None:
    sentinel = "provider-owner-secret-sentinel"
    binding = _binding().model_copy(update={"environment": {key: sentinel}})

    with pytest.raises(ValueError, match="sensitive key"):
        build_external_device_bootstrap_plan(
            operation_id="bootstrap-1",
            runtime_session_ref="runtime://session-1",
            binding=binding,
            device_credential_ref="secret://device/member-1",
        )

    assert sentinel not in repr(binding.model_dump(exclude={"environment"}))


@pytest.mark.asyncio
async def test_bootstrap_service_is_idempotent_and_requires_real_registration() -> None:
    plan = build_external_device_bootstrap_plan(
        operation_id="bootstrap-1",
        runtime_session_ref="runtime://session-1",
        binding=_binding(),
        device_credential_ref="env:NORTHCUA_DEVICE_TOKEN",
    )
    adapter = _Adapter()
    service = ExternalDeviceBootstrapService(
        adapter=adapter,
        readiness_probe=DelayedDeviceReadinessProbe(SandboxReadinessState.READY),
    )

    first = await service.ensure_ready(plan)
    second = await service.ensure_ready(plan)

    assert first.state == "started"
    assert second.state == "already_started"
    assert adapter.calls == 2


def test_bootstrap_service_rejects_noop_probe() -> None:
    with pytest.raises(ValueError, match="real readiness"):
        ExternalDeviceBootstrapService(adapter=_Adapter(), readiness_probe=NoopDeviceReadinessProbe())


@pytest.mark.asyncio
async def test_bootstrap_service_rejects_mismatched_receipt_and_timeout() -> None:
    plan = build_external_device_bootstrap_plan(
        operation_id="bootstrap-1",
        runtime_session_ref="runtime://session-1",
        binding=_binding(),
        device_credential_ref="env:NORTHCUA_DEVICE_TOKEN",
        readiness_timeout_seconds=0.001,
    )
    with pytest.raises(RuntimeError, match="does not match"):
        await ExternalDeviceBootstrapService(
            adapter=_Adapter(mismatch=True),
            readiness_probe=DelayedDeviceReadinessProbe(),
        ).ensure_ready(plan)
    with pytest.raises(TimeoutError, match="did not become ready"):
        await ExternalDeviceBootstrapService(
            adapter=_Adapter(),
            readiness_probe=DelayedDeviceReadinessProbe(SandboxReadinessState.TIMED_OUT),
        ).ensure_ready(plan)
