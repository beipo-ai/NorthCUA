"""Observation coordinator accessibility capability cache tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tests.runtime.accessibility.test_accessibility_runtime import (
    AccessibilityCapabilities,
    AccessibilityCaptureStatus,
    AccessibilityDiffMode,
    AccessibilityObservationConfig,
    AccessibilityRawFormat,
    AccessibilityScope,
    AccessibilitySnapshot,
    AccessibilitySourceKind,
    AccessibilitySummaryBuilder,
    ArtifactConfig,
    NorthCUAAppConfig,
    NorthCUAArtifactStore,
    ObservationConfig,
    ObservationCoordinator,
    ProbeExecutionRecord,
    _node,
    _png_data_url,
)


def test_accessibility_capabilities_are_cached_per_session(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-cache", ArtifactConfig())
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.yaml",
        observation=ObservationConfig(
            a11y=AccessibilityObservationConfig(enabled=True, diff_mode=AccessibilityDiffMode.BASIC, fail_open=True),
        ),
    )
    app_config.agent_mcp_servers = [
        {"name": "DesktopNativeDataExecutor", "type": "stdio", "model_visible": False},
        {"name": "DesktopNativeA11yExecutor", "type": "stdio", "model_visible": False},
    ]

    class _DummyProbeExecutor:
        def __init__(self) -> None:
            self.capability_calls = 0

        async def execute_chain(self, chain_name, *, session_step, steps):
            if chain_name == "__a11y_capabilities__":
                self.capability_calls += 1
                return [
                    ProbeExecutionRecord(
                        chain_name=chain_name,
                        step_index=0,
                        server_name="DesktopNativeA11yExecutor",
                        tool_name="get_accessibility_capabilities",
                        arguments={},
                        output=AccessibilityCapabilities(
                            platform_name="darwin",
                            source=AccessibilitySourceKind.AX,
                            supported_scopes=[AccessibilityScope.DESKTOP],
                            requires_permission=True,
                            permission_granted=True,
                            raw_formats=[AccessibilityRawFormat.JSON],
                        ).model_dump(mode="json"),
                        source="mcp:DesktopNativeA11yExecutor.get_accessibility_capabilities",
                    )
                ]
            if chain_name == "desktop_a11y_default":
                root = _node("desktop-root", role="desktop", name="desktop")
                return [
                    ProbeExecutionRecord(
                        chain_name=chain_name,
                        step_index=0,
                        server_name="DesktopNativeA11yExecutor",
                        tool_name="capture_accessibility_tree",
                        arguments=steps[0].arguments,
                        output=AccessibilitySnapshot(
                            status=AccessibilityCaptureStatus.SUCCESS,
                            platform_name="darwin",
                            source=AccessibilitySourceKind.AX,
                            scope=AccessibilityScope.DESKTOP,
                            tree=root,
                            summary=AccessibilitySummaryBuilder().build(root),
                        ).model_dump(mode="json"),
                        source="mcp:DesktopNativeA11yExecutor.capture_accessibility_tree",
                    )
                ]
            return [
                ProbeExecutionRecord(
                    chain_name=chain_name,
                    step_index=0,
                    server_name="DesktopNativeDataExecutor",
                    tool_name="capture_desktop_screenshot",
                    arguments={"all_screens": False},
                    output=_png_data_url(),
                    source="mcp:DesktopNativeDataExecutor.capture_desktop_screenshot",
                ),
                ProbeExecutionRecord(
                    chain_name=chain_name,
                    step_index=1,
                    server_name="DesktopNativeDataExecutor",
                    tool_name="get_app_window_info",
                    arguments={"field_list": []},
                    output={"control_text": "Feishu", "control_type": "window", "control_rect": [0, 0, 10, 10]},
                    source="mcp:DesktopNativeDataExecutor.get_app_window_info",
                ),
                ProbeExecutionRecord(
                    chain_name=chain_name,
                    step_index=2,
                    server_name="DesktopNativeDataExecutor",
                    tool_name="get_app_window_controls_target_info",
                    arguments={"field_list": []},
                    output=[],
                    source="mcp:DesktopNativeDataExecutor.get_app_window_controls_target_info",
                ),
            ]

    executor = _DummyProbeExecutor()
    coordinator_one = ObservationCoordinator(
        app_config=app_config,
        agent_config=SimpleNamespace(mcp_servers=[]),
        artifact_store=store,
        probe_executor=executor,
    )
    coordinator_two = ObservationCoordinator(
        app_config=app_config,
        agent_config=SimpleNamespace(mcp_servers=[]),
        artifact_store=store,
        probe_executor=executor,
    )

    asyncio.run(coordinator_one.collect(session_step=1, previous_steps=[]))
    asyncio.run(coordinator_two.collect(session_step=2, previous_steps=[]))

    assert executor.capability_calls == 1
