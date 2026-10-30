"""Sandbox provider registry and fake provider tests."""

from __future__ import annotations

import pytest

from northcua.sandboxes.contracts import (
    SandboxCapacityProbeRequest,
    SandboxCommandSpec,
    SandboxImageSpec,
    SandboxProviderFeature,
    SandboxProviderSpec,
)
from northcua.sandboxes.provider.fake import FakeSandboxProvider
from northcua.sandboxes.provider.registry import SandboxProviderRegistry


def test_provider_registry_rejects_duplicate_ids() -> None:
    registry = SandboxProviderRegistry()
    registry.register("fake", FakeSandboxProvider)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("fake", FakeSandboxProvider)


@pytest.mark.asyncio
async def test_fake_provider_records_commands_and_termination() -> None:
    provider = FakeSandboxProvider()
    spec = SandboxProviderSpec(provider="fake", template="ubuntu", desktop=False)

    handle = await provider.create(spec)
    result = await provider.run_command(handle, SandboxCommandSpec(command="echo ok"))
    await provider.terminate(handle)

    assert result.ok
    assert provider.commands[0][0] == handle.sandbox_id
    assert provider.terminated == [handle.sandbox_id]


@pytest.mark.asyncio
async def test_fake_provider_redacts_command_output() -> None:
    provider = FakeSandboxProvider()
    handle = await provider.create(SandboxProviderSpec(provider="fake", template="ubuntu", desktop=False))

    result = await provider.start_process(
        handle, SandboxCommandSpec(command="echo secret-token", redact=["secret-token"])
    )

    assert "secret-token" not in result.stdout
    assert "[REDACTED]" in result.stdout


def test_sandbox_image_spec_is_chainable_pure_data() -> None:
    base = SandboxImageSpec()
    image = base.apt_install("curl").pip_install("pytest").env(NORTHCUA="1").expose(8765)

    assert base.layers == []
    assert [layer.kind for layer in image.layers] == ["apt", "pip", "env", "expose"]
    assert image.layers[0].packages == ["curl"]


@pytest.mark.asyncio
async def test_fake_provider_file_and_log_capabilities(tmp_path) -> None:
    provider = FakeSandboxProvider()
    handle = await provider.create(SandboxProviderSpec(provider="fake", template="ubuntu", desktop=False))
    local = tmp_path / "input.txt"
    local.write_text("hello", encoding="utf-8")
    downloaded = tmp_path / "out" / "download.txt"

    await provider.upload(handle, local, "/tmp/input.txt")
    await provider.download(handle, "/tmp/remote.txt", downloaded)
    logs = await provider.collect_logs(handle, tmp_path / "logs")

    assert provider.capabilities().supports(SandboxProviderFeature.UPLOAD)
    assert downloaded.read_text(encoding="utf-8").startswith("fake-download")
    assert logs[0].local_path.exists()


@pytest.mark.asyncio
async def test_fake_provider_capacity_probe_counts_active_handles() -> None:
    provider = FakeSandboxProvider()
    spec = SandboxProviderSpec(provider="fake", template="ubuntu", metadata={"northcua": "true"})
    first = await provider.create(spec)
    second = await provider.create(SandboxProviderSpec(provider="fake", template="other"))
    await provider.terminate(second)

    result = await provider.capacity_probe(
        SandboxCapacityProbeRequest(
            provider="fake",
            template="ubuntu",
            requested_concurrency=4,
            max_active_sandboxes=3,
            reserve_sandboxes=1,
            include_items=True,
        )
    )

    assert result.total_active == 1
    assert result.northcua_active == 1
    assert result.matching_template_active == 1
    assert result.recommended_concurrency == 1
    assert result.items[0].sandbox_id == first.sandbox_id
