from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from northcua.agents.desktop.observation import (
    ObservationCoordinator,
    ProbeExecutionRecord,
    RemoteInternalProbeExecutor,
)
from northcua.core.config import (
    AccessibilityObservationConfig,
    ArtifactConfig,
    NorthCUAAppConfig,
    ObservationConfig,
)
from northcua.core.enums import (
    AccessibilityCaptureStatus,
    AccessibilityDiffMode,
    AccessibilityRawFormat,
    AccessibilityScope,
    AccessibilitySourceKind,
    ResultStatus,
)
from northcua.core.models import CommandExecutionResultRecord, StepLogRecord
from northcua.runtime.accessibility import (
    AccessibilityArtifactPersister,
    AccessibilityBounds,
    AccessibilityCapabilities,
    AccessibilityCaptureRequest,
    AccessibilityCollector,
    AccessibilityDiffBuilder,
    AccessibilityNode,
    AccessibilitySnapshot,
    AccessibilitySummaryBuilder,
)
from northcua.runtime.accessibility.linux import LinuxAccessibilityBackend
from northcua.runtime.accessibility.macos import MacOSAccessibilityBackend
from northcua.runtime.accessibility.windows import WindowsAccessibilityBackend
from northcua.runtime.artifacts.store import NorthCUAArtifactStore
from northcua.runtime.mcp.accessibility_server import create_desktop_native_a11y_mcp_server
from northcua.runtime.mcp.registry import MCPFactoryContext
from northcua.runtime.mcp.result_adapter import normalize_tool_result


def _png_data_url() -> str:
    image = Image.new("RGB", (2, 3), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _node(
    node_id: str,
    *,
    role: str,
    name: str,
    states: list[str] | None = None,
    actions: list[str] | None = None,
    children: list[AccessibilityNode] | None = None,
) -> AccessibilityNode:
    return AccessibilityNode(
        node_id=node_id,
        role=role,
        name=name,
        states=list(states or []),
        actions=list(actions or []),
        children=list(children or []),
    )


def test_observation_coordinator_fail_open_handles_invalid_snapshot_and_skips_failed_diff(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-invalid", ArtifactConfig())
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
                if session_step == 1:
                    output = {"status": "definitely-not-valid", "platform_name": "darwin"}
                else:
                    root = _node("desktop-root", role="desktop", name="desktop")
                    output = AccessibilitySnapshot(
                        status=AccessibilityCaptureStatus.SUCCESS,
                        platform_name="darwin",
                        source=AccessibilitySourceKind.AX,
                        scope=AccessibilityScope.DESKTOP,
                        tree=root,
                        summary=AccessibilitySummaryBuilder().build(root),
                    ).model_dump(mode="json")
                return [
                    ProbeExecutionRecord(
                        chain_name=chain_name,
                        step_index=0,
                        server_name="DesktopNativeA11yExecutor",
                        tool_name="capture_accessibility_tree",
                        arguments=steps[0].arguments,
                        output=output,
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
    assert step1_context.accessibility is not None
    assert step1_context.accessibility.status == AccessibilityCaptureStatus.ERROR

    step1 = StepLogRecord(
        session_id="sess-invalid",
        session_step=1,
        round_step=1,
        round_num=1,
        status="continue",
        request="Open Feishu",
        accessibility=step1_context.accessibility,
    )
    step2_context = asyncio.run(coordinator.collect(session_step=2, previous_steps=[step1]))

    assert step2_context.accessibility is not None
    assert step2_context.accessibility.status == AccessibilityCaptureStatus.SUCCESS
    assert step2_context.accessibility.diff is None


def test_remote_probe_executor_transports_accessibility_snapshot() -> None:
    async def _send_payload(_envelope):
        return None

    snapshot = AccessibilitySnapshot(
        status=AccessibilityCaptureStatus.SUCCESS,
        platform_name="darwin",
        source=AccessibilitySourceKind.AX,
        scope=AccessibilityScope.DESKTOP,
        tree=_node("desktop-root", role="desktop", name="desktop"),
    ).model_dump(mode="json")

    async def _await_results(call_ids):
        return [
            CommandExecutionResultRecord(
                call_id=call_ids[0],
                status=ResultStatus.SUCCESS,
                result=snapshot,
            )
        ]

    executor = RemoteInternalProbeExecutor(
        session_id="sess-1",
        round_num=1,
        round_step=1,
        send_payload=_send_payload,
        await_results=_await_results,
    )

    records = asyncio.run(
        executor.execute_chain(
            "remote_a11y",
            session_step=1,
            steps=[
                SimpleNamespace(
                    server="DesktopNativeA11yExecutor",
                    tool="capture_accessibility_tree",
                    arguments={"scope": "desktop"},
                    adapter="accessibility_tree",
                )
            ],
        )
    )

    assert records[0].output["status"] == "success"
    assert records[0].output["source"] == "ax"


def test_macos_backend_ax_bounds_prefers_axframe_when_available() -> None:
    class _FakeAXElement:
        def __init__(self, attrs: dict[str, object]) -> None:
            self.attrs = attrs

    class _FakeAXValue:
        def __init__(self, decoded: object) -> None:
            self.decoded = decoded

        def __repr__(self) -> str:
            return "<FakeAXValue>"

    frame_value = _FakeAXValue(
        SimpleNamespace(
            origin=SimpleNamespace(x=11.0, y=22.0),
            size=SimpleNamespace(width=333.0, height=444.0),
        )
    )
    element = _FakeAXElement(
        {
            "AXFrame": frame_value,
            "AXPosition": SimpleNamespace(x=1, y=2),
            "AXSize": SimpleNamespace(width=3, height=4),
        }
    )
    fake_application_services = SimpleNamespace(
        AXUIElementCopyAttributeValue=lambda target, attr, _unused=None: (
            (0, target.attrs[attr]) if attr in target.attrs else (1, None)
        ),
        AXValueGetValue=lambda value, _expected_type, _unused=None: value.decoded,
        kAXValueCGRectType=3,
        kAXValueCGPointType=1,
        kAXValueCGSizeType=2,
    )

    bounds = MacOSAccessibilityBackend._ax_bounds(fake_application_services, element)

    assert bounds is not None
    assert bounds.model_dump() == {"x": 11, "y": 22, "width": 333, "height": 444}
