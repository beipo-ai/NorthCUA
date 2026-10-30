"""Tests for typed platform runtime extension points."""

from __future__ import annotations

from typing import Sequence

import pytest

from northcua.core.enums import ResultStatus
from northcua.core.models import CommandExecutionResultRecord, DesktopStateSnapshot, ObservationContext
from northcua.core.protocol import CommandRequestPayload, CommandResultPayload
from northcua.platforms import (
    PlatformActionRequest,
    PlatformActionResult,
    PlatformActionStatus,
    PlatformArtifactKind,
    PlatformArtifactRef,
    PlatformArtifactRole,
    PlatformHealthStatus,
    PlatformKind,
    PlatformObservation,
    PlatformRuntime,
    PlatformRuntimeRegistry,
    default_platform_runtime_registry,
)
from northcua.platforms.browser import BrowserPlatformRuntime
from northcua.platforms.desktop import DesktopPlatformRuntime
from northcua.platforms.dispatch import PlatformCommandExecutor
from northcua.runtime.browser.backend import BrowserSnapshotResult, BrowserTabInfo, PlaywrightBrowserBackend


class _FakeRuntime(PlatformRuntime):
    """Small runtime implementation used to exercise registry contracts."""

    platform_kind = PlatformKind.REMOTE_VM

    async def healthcheck(self) -> PlatformHealthStatus:
        return PlatformHealthStatus(platform_kind=self.platform_kind, ok=True)

    async def capture_observation(
        self,
        *,
        session_step: int,
        previous_steps: list[object] | None = None,
    ) -> PlatformObservation:
        return PlatformObservation(platform_kind=self.platform_kind, metadata={"session_step": session_step})

    async def execute_action(self, request: PlatformActionRequest) -> PlatformActionResult:
        return PlatformActionResult(call_id=request.call_id, status=PlatformActionStatus.SUCCESS)


class _FakeDesktopObservationService:
    """Minimal observation service used by the desktop platform adapter test."""

    async def collect(self, session_step: int, previous_steps: list[object]) -> DesktopStateSnapshot:
        return DesktopStateSnapshot(
            platform_name="test-desktop",
            clean_screenshot_path=f"/tmp/step-{session_step}.png",
            screenshot_metadata={"previous_step_count": len(previous_steps)},
        )

    async def collect_context(self, session_step: int, previous_steps: list[object]) -> ObservationContext:
        snapshot = await self.collect(session_step, previous_steps)
        return ObservationContext(
            platform_name=snapshot.platform_name,
            current_clean_screenshot_path=snapshot.clean_screenshot_path,
            observation_metadata=dict(snapshot.screenshot_metadata),
        )

    @staticmethod
    def snapshot_from_context(observation_context: ObservationContext) -> DesktopStateSnapshot:
        return DesktopStateSnapshot(
            platform_name=observation_context.platform_name,
            clean_screenshot_path=observation_context.current_clean_screenshot_path,
            screenshot_metadata=dict(observation_context.observation_metadata),
        )


class _FakeDesktopCommandExecutor:
    """Minimal command executor used by the desktop platform adapter test."""

    async def execute_command_request(self, payload) -> CommandResultPayload:
        return CommandResultPayload(
            session_id=payload.session_id,
            status=ResultStatus.SUCCESS,
            results=[
                CommandExecutionResultRecord(
                    call_id=payload.commands[0].call_id,
                    status=ResultStatus.SUCCESS,
                    result={"ok": True},
                )
            ],
        )


class _FakeBrowserBackend(PlaywrightBrowserBackend):
    """Small browser backend used to test the browser platform adapter."""

    def __init__(self) -> None:
        self.clicked_selector = ""

    def list_tabs(self) -> list[BrowserTabInfo]:
        return []

    def new_tab(self) -> BrowserTabInfo:
        return BrowserTabInfo(index=0, title="", url="about:blank", is_active=True)

    def close_tab(self, index: int | None = None) -> None:
        return None

    def select_tab(self, index: int) -> BrowserTabInfo:
        return BrowserTabInfo(index=index, title="", url="about:blank", is_active=True)

    def navigate(self, url: str) -> str:
        return url

    def navigate_back(self) -> None:
        return None

    def navigate_forward(self) -> None:
        return None

    def reload(self) -> None:
        return None

    def capture_snapshot(self, selector: str | None = None, depth: int | None = None) -> BrowserSnapshotResult:
        return BrowserSnapshotResult(title="Fixture", url="https://example.test")

    def click(
        self,
        selector: str,
        *,
        button: str | None = None,
        double_click: bool = False,
        modifiers: Sequence[str] | None = None,
    ) -> None:
        self.clicked_selector = selector

    def drag(self, start_selector: str, end_selector: str) -> None:
        return None

    def hover(self, selector: str) -> None:
        return None

    def select_option(self, selector: str, values: Sequence[str]) -> None:
        return None

    def set_checked(self, selector: str, checked: bool) -> None:
        return None

    def stop(self) -> None:
        return None


def test_platform_runtime_registry_creates_registered_runtime() -> None:
    registry = PlatformRuntimeRegistry()
    registry.register(PlatformKind.REMOTE_VM, _FakeRuntime)

    runtime = registry.create(PlatformKind.REMOTE_VM)

    assert isinstance(runtime, _FakeRuntime)
    assert registry.registered_kinds() == (PlatformKind.REMOTE_VM,)


def test_default_platform_runtime_registry_resolves_builtin_descriptor_runtimes() -> None:
    registry = default_platform_runtime_registry()

    assert isinstance(registry.create(PlatformKind.DESKTOP), DesktopPlatformRuntime)
    assert isinstance(registry.create(PlatformKind.BROWSER), BrowserPlatformRuntime)


@pytest.mark.asyncio
async def test_descriptor_only_platform_runtimes_advertise_capabilities() -> None:
    desktop_health = await DesktopPlatformRuntime().healthcheck()
    browser_health = await BrowserPlatformRuntime().healthcheck()

    assert desktop_health.platform_kind == PlatformKind.DESKTOP
    assert browser_health.platform_kind == PlatformKind.BROWSER
    assert {capability.name for capability in desktop_health.capabilities} == {"screenshot", "action"}
    assert {capability.name for capability in browser_health.capabilities} == {"snapshot", "action"}


@pytest.mark.asyncio
async def test_descriptor_only_platform_runtimes_fail_before_live_backends() -> None:
    with pytest.raises(RuntimeError):
        await DesktopPlatformRuntime().capture_observation(session_step=1)
    with pytest.raises(RuntimeError):
        await BrowserPlatformRuntime().capture_observation(session_step=1)


@pytest.mark.asyncio
async def test_desktop_platform_runtime_wraps_observation_and_action_services() -> None:
    runtime = DesktopPlatformRuntime(
        observation_service=_FakeDesktopObservationService(),
        action_executor=_FakeDesktopCommandExecutor(),
    )

    observation = await runtime.capture_observation(session_step=3, previous_steps=[object(), object()])
    action = await runtime.execute_action(
        PlatformActionRequest(
            name="click",
            call_id="call-1",
            parameters={"x": 1, "y": 2},
            metadata={"session_id": "sess", "round_num": 1, "round_step": 1},
        )
    )

    assert observation.platform_kind == PlatformKind.DESKTOP
    assert observation.artifacts[0].path == "/tmp/step-3.png"
    assert action.status == PlatformActionStatus.SUCCESS
    assert action.result == {"ok": True}


@pytest.mark.asyncio
async def test_browser_platform_runtime_wraps_snapshot_and_actions() -> None:
    backend = _FakeBrowserBackend()
    runtime = BrowserPlatformRuntime(browser_backend=backend)

    observation = await runtime.capture_observation(session_step=1)
    action = await runtime.execute_action(PlatformActionRequest(name="click", parameters={"selector": "#submit"}))

    assert observation.platform_kind == PlatformKind.BROWSER
    assert observation.metadata["title"] == "Fixture"
    assert action.status == PlatformActionStatus.SUCCESS
    assert backend.clicked_selector == "#submit"


@pytest.mark.asyncio
async def test_platform_command_executor_projects_command_batches() -> None:
    executor = PlatformCommandExecutor(_FakeRuntime())

    result = await executor.execute_command_request(
        CommandRequestPayload(
            session_id="sess",
            round_num=1,
            round_step=2,
            commands=[
                {
                    "call_id": "call-1",
                    "tool_name": "noop",
                    "parameters": {},
                    "tool_type": "action",
                }
            ],
        )
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.results[0].call_id == "call-1"
