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


def test_accessibility_summary_diff_and_persistence(tmp_path) -> None:
    root1 = _node(
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
                        "desktop/0/0",
                        role="text_field",
                        name="Search",
                        states=["focused"],
                        actions=["set_value"],
                    )
                ],
            )
        ],
    )
    root2 = _node(
        "desktop-root",
        role="desktop",
        name="desktop",
        children=[
            _node(
                "desktop/0",
                role="window",
                name="Feishu",
                children=[
                    _node("desktop/0/0", role="button", name="Send", actions=["press"]),
                    _node(
                        "desktop/0/1",
                        role="text_field",
                        name="Message",
                        states=["focused"],
                        actions=["set_value"],
                    ),
                ],
            )
        ],
    )
    summary_builder = AccessibilitySummaryBuilder()
    snapshot1 = AccessibilitySnapshot(
        status=AccessibilityCaptureStatus.SUCCESS,
        platform_name="darwin",
        source=AccessibilitySourceKind.AX,
        scope=AccessibilityScope.DESKTOP,
        tree=root1,
        summary=summary_builder.build(root1),
        raw_format=AccessibilityRawFormat.JSON,
        raw_payload='{"native": true}',
    )
    snapshot2 = AccessibilitySnapshot(
        status=AccessibilityCaptureStatus.SUCCESS,
        platform_name="darwin",
        source=AccessibilitySourceKind.AX,
        scope=AccessibilityScope.DESKTOP,
        tree=root2,
        summary=summary_builder.build(root2),
    )

    diff = AccessibilityDiffBuilder().build(snapshot1, snapshot2)
    store = NorthCUAArtifactStore.for_session(tmp_path, "sess-1", ArtifactConfig())
    persister = AccessibilityArtifactPersister(store)
    artifact_path, raw_path = persister.persist_snapshot(snapshot1, step_index=1, persist_raw_artifact=True)
    diff_path = persister.persist_diff(diff, step_index=2) if diff is not None else ""
    record = persister.build_step_record(
        snapshot=snapshot2,
        artifact_path=str(store.a11y_snapshot_path(2)),
        raw_artifact_path="",
        diff_mode=AccessibilityDiffMode.BASIC,
        diff=diff,
    )

    assert diff is not None
    assert diff.added_nodes == 2
    assert diff.removed_nodes == 1
    assert diff.changed_nodes == 0
    assert diff.focus_changed is True
    assert artifact_path.endswith("a11y_step1.json")
    assert raw_path.endswith("a11y_step1.raw.json")
    assert diff_path.endswith("a11y_step2_diff.json")
    persisted = json.loads(store.a11y_snapshot_path(1).read_text(encoding="utf-8"))
    assert persisted["raw_payload"] == ""
    assert json.loads(store.a11y_raw_path(1, AccessibilityRawFormat.JSON).read_text(encoding="utf-8")) == {
        "native": True
    }
    assert record.focused_name == "Message"
    assert record.diff is not None


def test_accessibility_diff_is_stable_under_sibling_reordering() -> None:
    previous = AccessibilitySnapshot(
        status=AccessibilityCaptureStatus.SUCCESS,
        platform_name="darwin",
        source=AccessibilitySourceKind.AX,
        scope=AccessibilityScope.DESKTOP,
        tree=_node(
            "desktop-root",
            role="desktop",
            name="desktop",
            children=[
                _node("desktop/0", role="button", name="Send"),
                _node("desktop/1", role="button", name="Cancel"),
            ],
        ),
    )
    current = AccessibilitySnapshot(
        status=AccessibilityCaptureStatus.SUCCESS,
        platform_name="darwin",
        source=AccessibilitySourceKind.AX,
        scope=AccessibilityScope.DESKTOP,
        tree=_node(
            "desktop-root",
            role="desktop",
            name="desktop",
            children=[
                _node("desktop/1", role="button", name="Cancel"),
                _node("desktop/0", role="button", name="Send"),
            ],
        ),
    )

    diff = AccessibilityDiffBuilder().build(previous, current)

    assert diff is not None
    assert diff.added_nodes == 0
    assert diff.removed_nodes == 0


def test_accessibility_collector_timeout_is_bounded() -> None:
    class _SlowBackend:
        source_kind = AccessibilitySourceKind.AX

        def capabilities(self):
            return AccessibilityCapabilities(
                platform_name="darwin",
                source=AccessibilitySourceKind.AX,
                supported_scopes=[AccessibilityScope.DESKTOP],
            )

        def capture(self, request):
            del request
            time.sleep(0.3)
            return AccessibilitySnapshot(
                status=AccessibilityCaptureStatus.SUCCESS,
                platform_name="darwin",
                source=AccessibilitySourceKind.AX,
                scope=AccessibilityScope.DESKTOP,
                tree=_node("desktop-root", role="desktop", name="desktop"),
            )

    collector = AccessibilityCollector(backend=_SlowBackend(), platform_name="darwin")
    started = time.perf_counter()
    snapshot = collector.capture(
        AccessibilityCaptureRequest(scope=AccessibilityScope.DESKTOP, capture_timeout_seconds=0.05)
    )
    elapsed = time.perf_counter() - started

    assert snapshot.status == AccessibilityCaptureStatus.TIMEOUT
    assert elapsed < 0.2


def test_accessibility_collector_does_not_synthesize_raw_payload_from_normalized_tree() -> None:
    class _BackendWithoutNativeRaw:
        source_kind = AccessibilitySourceKind.AX

        def capabilities(self):
            return AccessibilityCapabilities(
                platform_name="darwin",
                source=AccessibilitySourceKind.AX,
                supported_scopes=[AccessibilityScope.DESKTOP],
            )

        def capture(self, request):
            del request
            return AccessibilitySnapshot(
                status=AccessibilityCaptureStatus.SUCCESS,
                platform_name="darwin",
                source=AccessibilitySourceKind.AX,
                scope=AccessibilityScope.DESKTOP,
                tree=_node("desktop-root", role="desktop", name="desktop"),
            )

    snapshot = AccessibilityCollector(
        backend=_BackendWithoutNativeRaw(),
        platform_name="darwin",
    ).capture(
        AccessibilityCaptureRequest(
            scope=AccessibilityScope.DESKTOP,
            include_raw_payload=True,
        )
    )

    assert snapshot.raw_payload == ""
    assert snapshot.raw_format == AccessibilityRawFormat.NONE


def test_accessibility_mcp_server_tools_return_normalized_payload(monkeypatch) -> None:
    class _FakeCollector:
        def capabilities(self):
            return AccessibilityCapabilities(
                platform_name="darwin",
                source=AccessibilitySourceKind.AX,
                supported_scopes=[AccessibilityScope.DESKTOP, AccessibilityScope.ACTIVE_WINDOW],
                requires_permission=True,
                permission_granted=True,
                raw_formats=[AccessibilityRawFormat.JSON],
            )

        def capture(self, request):
            assert request.scope == AccessibilityScope.DESKTOP
            assert request.max_depth == 12
            return AccessibilitySnapshot(
                status=AccessibilityCaptureStatus.SUCCESS,
                platform_name="darwin",
                source=AccessibilitySourceKind.AX,
                scope=request.scope,
                tree=_node("desktop-root", role="desktop", name="desktop"),
            )

    monkeypatch.setattr(
        "northcua.runtime.mcp.accessibility_server.AccessibilityCollector",
        lambda platform_name=None: _FakeCollector(),
    )
    server = create_desktop_native_a11y_mcp_server(MCPFactoryContext(process_name="test-a11y"))

    capabilities = normalize_tool_result(asyncio.run(server.call_tool("get_accessibility_capabilities", {})))
    snapshot = normalize_tool_result(
        asyncio.run(
            server.call_tool(
                "capture_accessibility_tree",
                {
                    "scope": "desktop",
                    "max_depth": 12,
                    "max_children_per_node": 20,
                    "max_nodes": 100,
                    "include_raw_payload": False,
                    "capture_timeout_seconds": 2.0,
                },
            )
        )
    )

    assert capabilities["source"] == "ax"
    assert capabilities["supported_scopes"] == ["desktop", "active_window"]
    assert snapshot["status"] == "success"
    assert snapshot["tree"]["node_id"] == "desktop-root"
