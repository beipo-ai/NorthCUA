"""Regression coverage for legacy desktop observation service projection helpers."""

from __future__ import annotations

import asyncio

from PIL import Image

from northcua.core.config import ArtifactConfig
from northcua.core.models import (
    DesktopStateSnapshot,
    ObservationContext,
    StepLogRecord,
)
from northcua.runtime.artifacts.store import NorthCUAArtifactStore
from northcua.runtime.desktop import DesktopObservationService


def test_observation_service_with_coordinator_does_not_build_local_mcp(tmp_path, monkeypatch) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-remote", ArtifactConfig())

    class _Coordinator:
        async def collect(self, session_step: int, previous_steps: list[StepLogRecord]) -> ObservationContext:
            assert session_step == 1
            assert previous_steps == []
            return ObservationContext(
                platform_name="remote",
                current_clean_screenshot_path="/tmp/remote.png",
                current_clean_screenshot_url="data:image/png;base64,AAAA",
            )

    def _fail_build(*args, **kwargs):
        del args, kwargs
        raise AssertionError("local desktop MCP should not be built for coordinator-backed observation")

    monkeypatch.setattr("northcua.runtime.mcp.registry.NorthCUAMCPRegistry.build", _fail_build)
    service = DesktopObservationService(
        coordinator=_Coordinator(),
    )

    snapshot = asyncio.run(service.collect(1, []))

    assert snapshot.platform_name == "remote"
    assert snapshot.clean_screenshot_path == "/tmp/remote.png"


def test_snapshot_from_context_preserves_observation_metadata() -> None:
    snapshot = DesktopObservationService.snapshot_from_context(
        ObservationContext(
            platform_name="macos",
            current_clean_screenshot_path="/tmp/raw.png",
            current_clean_screenshot_url="data:image/png;base64,AAAA",
            selected_history_image_path="/tmp/raw.png",
            selected_history_image_url="data:image/png;base64,AAAA",
            observation_metadata={
                "current_state_source": "latest_tool_result:raw",
                "selected_image_kind": "raw",
                "source_image_path": "/tmp/raw.png",
            },
        )
    )

    assert snapshot.screenshot_metadata["selected_image_kind"] == "raw"
    assert snapshot.screenshot_metadata["source_image_path"] == "/tmp/raw.png"


def test_snapshot_from_context_fills_missing_image_dimensions(tmp_path) -> None:
    screenshot = tmp_path / "desktop.png"
    Image.new("RGB", (1920, 1080)).save(screenshot)

    snapshot = DesktopObservationService.snapshot_from_context(
        ObservationContext(
            platform_name="linux",
            current_clean_screenshot_path=str(screenshot),
            observation_metadata={"all_screens": False},
        )
    )

    assert snapshot.screenshot_metadata == {"all_screens": False, "width": 1920, "height": 1080}


def test_context_from_snapshot_uses_path_for_logged_image_artifact() -> None:
    context = DesktopObservationService.context_from_snapshot(
        DesktopStateSnapshot(
            platform_name="macos",
            clean_screenshot_path="/tmp/raw.png",
            clean_screenshot_url="data:image/png;base64,AAAA",
        )
    )

    assert context.current_artifacts[0].path == "/tmp/raw.png"
    assert context.current_artifacts[0].url == ""
