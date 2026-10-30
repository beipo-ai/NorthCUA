"""Sandbox health helper tests."""

from __future__ import annotations

import pytest

from northcua.sandboxes.contracts import (
    SandboxCapacityProbeRequest,
    SandboxCommandResult,
    SandboxHandle,
    SandboxProviderSpec,
)
from northcua.sandboxes.health import CapacityGuard, SandboxHealthProbe, SandboxJanitor
from northcua.sandboxes.provider.fake import FakeSandboxProvider
from northcua.sandboxes.provider.registry import SandboxProviderRegistry


class _HealthyFakeProvider(FakeSandboxProvider):
    async def run_command(self, handle: SandboxHandle, command) -> SandboxCommandResult:
        return SandboxCommandResult(stdout="northcua-health-ok\n")


@pytest.mark.asyncio
async def test_sandbox_health_probe_passes_for_marker_output() -> None:
    provider = _HealthyFakeProvider()
    handle = await provider.create(SandboxProviderSpec(provider="fake"))

    report = await SandboxHealthProbe().probe(provider=provider, handle=handle)

    assert report.healthy is True
    assert report.checks == {"python": True}


@pytest.mark.asyncio
async def test_capacity_guard_uses_passive_probe_recommendation() -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    guard = CapacityGuard(registry)

    decision = await guard.decide(
        SandboxCapacityProbeRequest(provider="fake", requested_concurrency=4, max_active_sandboxes=2),
        auto_clamp=True,
    )

    assert decision.effective_concurrency <= 2
    assert decision.auto_clamped is True


@pytest.mark.asyncio
async def test_janitor_reports_northcua_owned_candidates() -> None:
    provider = FakeSandboxProvider()
    registry = SandboxProviderRegistry()
    registry.register("fake", lambda: provider)
    handle = await provider.create(SandboxProviderSpec(provider="fake", metadata={"northcua": "true"}))

    report = await SandboxJanitor(registry).scan(SandboxCapacityProbeRequest(provider="fake"))

    assert report.matched == 1
    assert report.skipped == [handle.sandbox_id]
