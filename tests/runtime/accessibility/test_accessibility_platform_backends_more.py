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


def test_macos_backend_ax_bounds_falls_back_to_repr_parsing() -> None:
    class _FakeAXElement:
        def __init__(self, attrs: dict[str, object]) -> None:
            self.attrs = attrs

    class _FakeAXValue:
        def __init__(self, text: str) -> None:
            self.text = text

        def __repr__(self) -> str:
            return self.text

    element = _FakeAXElement(
        {
            "AXPosition": _FakeAXValue("<CGPoint x:12 y:34>"),
            "AXSize": _FakeAXValue("<CGSize width:56 height:78>"),
        }
    )
    fake_application_services = SimpleNamespace(
        AXUIElementCopyAttributeValue=lambda target, attr, _unused=None: (
            (0, target.attrs[attr]) if attr in target.attrs else (1, None)
        ),
        AXValueGetValue=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("decode failed")),
        kAXValueCGRectType=3,
        kAXValueCGPointType=1,
        kAXValueCGSizeType=2,
    )

    bounds = MacOSAccessibilityBackend._ax_bounds(fake_application_services, element)

    assert bounds is not None
    assert bounds.model_dump() == {"x": 12, "y": 34, "width": 56, "height": 78}


def test_linux_backend_bounds_prefers_xy_screen_and_filters_invalid_geometry() -> None:
    class _StateSet:
        def __init__(self, *values) -> None:
            self._values = set(values)

        def contains(self, value):
            return value in self._values

    class _Component:
        def __init__(self, extents, sink: list[object]) -> None:
            self._extents = extents
            self._sink = sink

        def getExtents(self, coords):
            self._sink.append(coords)
            return self._extents

    class _Node:
        def __init__(self, *, extents, states, sink: list[object]) -> None:
            self._extents = extents
            self._states = states
            self._sink = sink

        def getState(self):
            return _StateSet(*self._states)

        def queryComponent(self):
            return _Component(self._extents, self._sink)

    pyatspi = SimpleNamespace(
        XY_SCREEN=99,
        DESKTOP_COORDS=7,
        STATE_VISIBLE="visible",
        STATE_SHOWING="showing",
    )
    coord_calls: list[object] = []
    visible_node = _Node(
        extents=SimpleNamespace(x=10, y=20, width=30, height=40),
        states=("visible", "showing"),
        sink=coord_calls,
    )
    hidden_node = _Node(
        extents=SimpleNamespace(x=10, y=20, width=30, height=40),
        states=("visible",),
        sink=coord_calls,
    )
    zero_size_node = _Node(
        extents=SimpleNamespace(x=0, y=0, width=0, height=0),
        states=("visible", "showing"),
        sink=coord_calls,
    )

    visible_bounds = LinuxAccessibilityBackend._bounds(pyatspi, visible_node)
    hidden_bounds = LinuxAccessibilityBackend._bounds(pyatspi, hidden_node)
    zero_size_bounds = LinuxAccessibilityBackend._bounds(pyatspi, zero_size_node)

    assert visible_bounds is not None
    assert visible_bounds.model_dump() == {"x": 10, "y": 20, "width": 30, "height": 40}
    assert hidden_bounds is None
    assert zero_size_bounds is None
    assert coord_calls == [99, 99]
