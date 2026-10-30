"""Regression coverage for native accessibility backend adapters."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from northcua.runtime.accessibility import linux as linux_backend
from tests.runtime.accessibility.test_accessibility_platform_backends import (
    AccessibilityCaptureRequest,
    AccessibilityCaptureStatus,
    AccessibilityCollector,
    AccessibilityScope,
    LinuxAccessibilityBackend,
    MacOSAccessibilityBackend,
    WindowsAccessibilityBackend,
)


def test_macos_backend_traverses_object_reference_attributes_beyond_axchildren(monkeypatch) -> None:
    class _FakeAXElement:
        def __init__(
            self, signature: str, attrs: dict[str, object] | None = None, actions: list[str] | None = None
        ) -> None:
            self.signature = signature
            self.attrs = dict(attrs or {})
            self.actions = list(actions or [])

        def __repr__(self) -> str:
            return f"<FakeAXElement {self.signature}>"

    message_field = _FakeAXElement(
        "message-field",
        attrs={
            "AXRole": "AXTextField",
            "AXTitle": "Message",
            "AXValue": "hello",
            "AXFocused": True,
            "AXEnabled": True,
            "AXPosition": SimpleNamespace(x=24, y=44),
            "AXSize": SimpleNamespace(width=220, height=36),
        },
    )
    send_button = _FakeAXElement(
        "send-button",
        attrs={
            "AXRole": "AXButton",
            "AXTitle": "Send",
            "AXEnabled": True,
            "AXPosition": SimpleNamespace(x=252, y=44),
            "AXSize": SimpleNamespace(width=64, height=36),
        },
        actions=["AXPress"],
    )
    content_group = _FakeAXElement(
        "content-group",
        attrs={
            "AXRole": "AXGroup",
            "AXDescription": "Conversation",
            "AXEnabled": True,
            "AXPosition": SimpleNamespace(x=12, y=24),
            "AXSize": SimpleNamespace(width=320, height=80),
            "AXContents": [message_field, send_button],
        },
    )
    window_element = _FakeAXElement(
        "window",
        attrs={
            "AXRole": "AXWindow",
            "AXTitle": "Feishu",
            "AXDescription": "Chat window",
            "AXEnabled": True,
            "AXPosition": SimpleNamespace(x=10, y=20),
            "AXSize": SimpleNamespace(width=360, height=200),
            "AXContents": [content_group],
        },
    )
    app_element = _FakeAXElement("app", attrs={"AXWindows": [window_element]})
    fake_apps = {11: app_element}

    fake_application_services = SimpleNamespace(
        AXUIElementRef=_FakeAXElement,
        AXUIElementCreateApplication=lambda pid: fake_apps[pid],
        AXUIElementCreateSystemWide=lambda: _FakeAXElement("system"),
        AXUIElementCopyAttributeNames=lambda element, _unused=None: (0, list(element.attrs.keys())),
        AXUIElementCopyAttributeValue=lambda element, attr, _unused=None: (
            (0, element.attrs[attr]) if attr in element.attrs else (1, None)
        ),
        AXUIElementCopyActionNames=lambda element, _unused=None: (0, list(element.actions)),
    )
    fake_quartz = SimpleNamespace(
        kCGWindowListOptionOnScreenOnly=1,
        kCGWindowListExcludeDesktopElements=2,
        kCGNullWindowID=0,
        CGWindowListCopyWindowInfo=lambda *_args: [
            {
                "kCGWindowLayer": 0,
                "kCGWindowOwnerName": "Feishu",
                "kCGWindowName": "Feishu",
                "kCGWindowOwnerPID": 11,
                "kCGWindowBounds": {"X": 10, "Y": 20, "Width": 360, "Height": 200},
            }
        ],
    )
    monkeypatch.setitem(sys.modules, "ApplicationServices", fake_application_services)
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    backend = MacOSAccessibilityBackend()
    monkeypatch.setattr(backend, "_framework_state", lambda: (True, ""))
    monkeypatch.setattr(backend, "_permission_state", lambda: (True, ""))

    snapshot = AccessibilityCollector(backend=backend, platform_name="darwin").capture(
        AccessibilityCaptureRequest(scope=AccessibilityScope.DESKTOP)
    )

    assert snapshot.status == AccessibilityCaptureStatus.SUCCESS
    assert snapshot.stats.max_depth >= 3
    assert snapshot.summary is not None
    assert snapshot.summary.focused_name == "Message"
    assert snapshot.summary.actionable_count >= 2
    assert snapshot.tree is not None
    top_window = snapshot.tree.children[0]
    assert top_window.role == "window"
    assert top_window.children[0].attributes["ax_relation"] == "contents"
    assert top_window.children[0].children[0].role == "text_field"
    assert top_window.children[0].children[1].actions == ["press"]


def test_platform_backends_do_not_advertise_raw_native_payload_without_native_serializer(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pywinauto",
        SimpleNamespace(Desktop=lambda backend="uia": SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "pyatspi", SimpleNamespace())
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")

    windows_capabilities = WindowsAccessibilityBackend().capabilities()
    linux_capabilities = LinuxAccessibilityBackend().capabilities()

    assert windows_capabilities.raw_formats == []
    assert linux_capabilities.raw_formats == []


def test_linux_backend_falls_back_to_system_dist_packages_for_pyatspi(tmp_path: Path, monkeypatch) -> None:
    dist_packages = tmp_path / "dist-packages"
    dist_packages.mkdir()
    (dist_packages / "pyatspi.py").write_text("NAME = 'fake-pyatspi'\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "pyatspi", raising=False)
    monkeypatch.setattr(
        linux_backend,
        "Path",
        lambda raw: dist_packages if raw == "/usr/lib/python3/dist-packages" else Path(raw),
    )
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(dist_packages)])

    module = linux_backend._import_pyatspi()

    assert module.NAME == "fake-pyatspi"
    assert str(dist_packages) in sys.path


def test_linux_backend_prefers_system_gi_bindings_after_partial_import_failure(tmp_path: Path, monkeypatch) -> None:
    """A bad pip/uv GI module must not poison the pyatspi retry path."""

    bad_packages = tmp_path / "bad-packages"
    dist_packages = tmp_path / "dist-packages"
    bad_packages.mkdir()
    dist_packages.mkdir()
    (bad_packages / "pyatspi.py").write_text("import gi\n", encoding="utf-8")
    (bad_packages / "gi.py").write_text("raise ImportError('bad gi')\n", encoding="utf-8")
    (dist_packages / "pyatspi.py").write_text("NAME = 'system-pyatspi'\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "pyatspi", raising=False)
    monkeypatch.delitem(sys.modules, "gi", raising=False)
    monkeypatch.setattr(
        linux_backend,
        "Path",
        lambda raw: (
            dist_packages
            if raw == "/usr/lib/python3/dist-packages"
            else Path("/missing")
            if raw == "/usr/local/lib/python3/dist-packages"
            else Path(raw)
        ),
    )
    monkeypatch.setattr(sys, "path", [str(bad_packages)])

    module = linux_backend._import_pyatspi()

    assert module.NAME == "system-pyatspi"
    assert sys.path[0] == str(dist_packages)


def test_linux_backend_uses_system_python_atspi_capture_fallback(monkeypatch) -> None:
    """E2B py3.12 can miss GI while distro Python still has AT-SPI bindings."""

    payload = {
        "status": "success",
        "platform_name": "linux",
        "source": "atspi",
        "scope": "desktop",
        "tree": {"node_id": "desktop-root", "role": "desktop", "name": "desktop", "children": []},
        "stats": {"node_count": 1, "max_depth": 0, "truncated": False, "capture_ms": 1.0},
        "raw_format": "none",
        "raw_payload": "",
    }

    class _Completed:
        returncode = 0
        stdout = linux_backend.json.dumps(payload)
        stderr = ""

    def _raise_import_error() -> object:
        raise ImportError("bad gi")

    def _run(*_args, **_kwargs):
        return _Completed()

    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setattr(linux_backend, "_import_pyatspi", _raise_import_error)
    monkeypatch.setattr(linux_backend, "_system_python_candidates", lambda: ["/usr/bin/python3"])
    monkeypatch.setattr(linux_backend.subprocess, "run", _run)

    snapshot = LinuxAccessibilityBackend().capture(AccessibilityCaptureRequest(scope=AccessibilityScope.DESKTOP))

    assert snapshot.status == AccessibilityCaptureStatus.SUCCESS
    assert snapshot.tree is not None
    assert snapshot.tree.node_id == "desktop-root"


def test_windows_backend_avoids_recursive_cycles(monkeypatch) -> None:
    class _Rect:
        def __init__(self, left, top, right, bottom) -> None:
            self.left = left
            self.top = top
            self.right = right
            self.bottom = bottom

    class _ElementInfo:
        def __init__(self, *, name: str, control_type: str, handle: int) -> None:
            self.name = name
            self.control_type = control_type
            self.handle = handle
            self.process_id = 42
            self.rectangle = _Rect(0, 0, 40, 20)
            self.automation_id = ""
            self.class_name = "Button"

    class _Wrapper:
        def __init__(self, name: str, control_type: str, handle: int) -> None:
            self.element_info = _ElementInfo(name=name, control_type=control_type, handle=handle)
            self._children = []

        def window_text(self):
            return self.element_info.name

        def rectangle(self):
            return self.element_info.rectangle

        def children(self):
            return list(self._children)

        def is_enabled(self):
            return True

        def is_visible(self):
            return True

        def has_keyboard_focus(self):
            return False

    root = _Wrapper("Root", "Window", 1)
    child = _Wrapper("Child", "Button", 2)
    root._children = [child]
    child._children = [root]

    class _Desktop:
        def __init__(self, backend="uia") -> None:
            assert backend == "uia"

        def top_window(self):
            return root

        def windows(self):
            return [root]

    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(Desktop=_Desktop))

    snapshot = AccessibilityCollector(
        backend=WindowsAccessibilityBackend(),
        platform_name="windows",
    ).capture(AccessibilityCaptureRequest(scope=AccessibilityScope.ACTIVE_WINDOW, max_depth=10))

    assert snapshot.status == AccessibilityCaptureStatus.SUCCESS
    assert snapshot.tree is not None
    assert snapshot.stats.max_depth == 1
    assert len(snapshot.tree.children) == 1
    assert snapshot.tree.children[0].name == "Child"


def test_linux_backend_extracts_text_value_and_attributes(monkeypatch) -> None:
    class _StateSet:
        def __init__(self, *values) -> None:
            self._values = set(values)

        def contains(self, value):
            return value in self._values

    class _Text:
        def __init__(self, text: str) -> None:
            self._text = text
            self.characterCount = len(text)

        def getText(self, start: int, end: int):
            return self._text[start:end]

    class _Value:
        def __init__(self, current: str) -> None:
            self.currentValue = current

    class _Component:
        def getExtents(self, _coords):
            return SimpleNamespace(x=5, y=6, width=70, height=18)

    class _Node:
        def __init__(self, name: str, role: str, *, children=None) -> None:
            self.name = name
            self.description = f"{name} description"
            self._role = role
            self._children = list(children or [])
            self.childCount = len(self._children)
            self.parent = None
            for child in self._children:
                child.parent = self

        def getRoleName(self):
            return self._role

        def getChildAtIndex(self, index: int):
            return self._children[index]

        def getState(self):
            return _StateSet("focused", "enabled", "visible")

        def queryAction(self):
            return SimpleNamespace(nActions=1, getName=lambda index: "press")

        def queryComponent(self):
            return _Component()

        def queryText(self):
            return _Text("hello world")

        def queryValue(self):
            return _Value("7")

        def get_attributes(self):
            return {"class": "entry", "id": "search"}

    desktop_root = _Node(
        "DesktopRoot", "desktop", children=[_Node("App", "application", children=[_Node("Search", "text")])]
    )

    pyatspi = SimpleNamespace(
        Registry=SimpleNamespace(getDesktop=lambda index: desktop_root),
        DESKTOP_COORDS=0,
        STATE_FOCUSED="focused",
        STATE_ACTIVE="active",
        STATE_ENABLED="enabled",
        STATE_VISIBLE="visible",
        STATE_SHOWING="showing",
    )
    monkeypatch.setitem(sys.modules, "pyatspi", pyatspi)
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")

    snapshot = LinuxAccessibilityBackend().capture(
        AccessibilityCaptureRequest(scope=AccessibilityScope.DESKTOP, max_depth=10)
    )

    assert snapshot.status == AccessibilityCaptureStatus.SUCCESS
    assert snapshot.tree is not None
    leaf = snapshot.tree.children[0].children[0]
    assert leaf.text == "hello world"
    assert leaf.value == "7"
    assert leaf.attributes["class"] == "entry"
    assert leaf.actions == ["press"]
