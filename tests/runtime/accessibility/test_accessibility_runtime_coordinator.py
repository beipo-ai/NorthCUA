"""Observation coordinator accessibility diff and permission tests."""

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
    StepLogRecord,
    _node,
    _png_data_url,
)


def test_observation_coordinator_collects_accessibility_and_diff(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-1", ArtifactConfig())
    app_config = NorthCUAAppConfig(
        agent_config_path="agent.yaml",
        observation=ObservationConfig(
            a11y=AccessibilityObservationConfig(enabled=True, diff_mode=AccessibilityDiffMode.BASIC),
        ),
    )
    app_config.agent_mcp_servers = [
        {"name": "DesktopNativeDataExecutor", "type": "stdio", "model_visible": False},
        {"name": "DesktopNativeA11yExecutor", "type": "stdio", "model_visible": False},
    ]

    summary_builder = AccessibilitySummaryBuilder()

    class _DummyProbeExecutor:
        async def execute_chain(self, chain_name, *, session_step, steps):
            if chain_name == "__a11y_capabilities__":
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
                root = _node(
                    "desktop-root",
                    role="desktop",
                    name="desktop",
                    children=[
                        _node(
                            "desktop/0",
                            role="window",
                            name="Feishu",
                            children=[
                                _node(
                                    "desktop/0/0" if session_step == 1 else "desktop/0/1",
                                    role="text_field",
                                    name="Search" if session_step == 1 else "Message",
                                    states=["focused"],
                                    actions=["set_value"],
                                )
                            ],
                        )
                    ],
                )
                snapshot = AccessibilitySnapshot(
                    status=AccessibilityCaptureStatus.SUCCESS,
                    platform_name="darwin",
                    source=AccessibilitySourceKind.AX,
                    scope=AccessibilityScope.DESKTOP,
                    tree=root,
                    summary=summary_builder.build(root),
                ).model_dump(mode="json")
                return [
                    ProbeExecutionRecord(
                        chain_name=chain_name,
                        step_index=0,
                        server_name="DesktopNativeA11yExecutor",
                        tool_name="capture_accessibility_tree",
                        arguments=steps[0].arguments,
                        output=snapshot,
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

    coordinator = ObservationCoordinator(
        app_config=app_config,
        agent_config=SimpleNamespace(mcp_servers=[]),
        artifact_store=store,
        probe_executor=_DummyProbeExecutor(),
    )

    step1_context = asyncio.run(coordinator.collect(session_step=1, previous_steps=[]))
    step1 = StepLogRecord(
        session_id="sess-1",
        session_step=1,
        round_step=1,
        round_num=1,
        status="continue",
        request="Open Feishu",
        accessibility=step1_context.accessibility,
        observation_artifacts=step1_context.current_artifacts,
    )

    step2_context = asyncio.run(coordinator.collect(session_step=2, previous_steps=[step1]))

    assert step1_context.accessibility is not None
    assert step1_context.accessibility.artifact_path.endswith("a11y_step1.json")
    assert step2_context.accessibility is not None
    assert step2_context.accessibility.diff is not None
    assert step2_context.accessibility.diff.focus_changed is True
    assert any(artifact.kind.value == "accessibility_diff" for artifact in step2_context.current_artifacts)


def test_observation_coordinator_short_circuits_when_capability_denied(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-denied", ArtifactConfig())
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
            self.capture_calls = 0

        async def execute_chain(self, chain_name, *, session_step, steps):
            del session_step
            if chain_name == "__a11y_capabilities__":
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
                            permission_granted=False,
                            raw_formats=[],
                            message="Accessibility permission not granted",
                        ).model_dump(mode="json"),
                        source="mcp:DesktopNativeA11yExecutor.get_accessibility_capabilities",
                    )
                ]
            if chain_name == "desktop_snapshot_default":
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
            self.capture_calls += 1
            raise AssertionError("capture_accessibility_tree should not run when capability denies capture")

    executor = _DummyProbeExecutor()
    coordinator = ObservationCoordinator(
        app_config=app_config,
        agent_config=SimpleNamespace(mcp_servers=[]),
        artifact_store=store,
        probe_executor=executor,
    )

    context = asyncio.run(coordinator.collect(session_step=1, previous_steps=[]))

    assert context.accessibility is not None
    assert context.accessibility.status == AccessibilityCaptureStatus.PERMISSION_DENIED
    assert executor.capture_calls == 0
