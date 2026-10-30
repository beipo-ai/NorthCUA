from __future__ import annotations

import asyncio
import base64
import os
import subprocess
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from northcua.core.coordinates import CoordinatePolicy, CoordinateTarget, coordinate_policy_context_values
from northcua.core.enums import CoordinateSpace
from northcua.runtime.mcp import MCPFactoryContext, NorthCUAMCPRegistry, ensure_builtin_mcp_servers_registered
from northcua.runtime.mcp.desktop_native.image_encoding import encode_desktop_image
from northcua.runtime.mcp.desktop_native_server import (
    DesktopCoordinateFrame,
    DesktopNativeServerState,
    PyAutoGuiDesktopBackend,
)
from northcua.runtime.mcp.result_adapter import normalize_tool_result
from northcua_runtime_shared.desktop_surface.models import TextTargetKind


def _png_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(32, 96, 160))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_data_url() -> str:
    return f"data:image/png;base64,{base64.b64encode(_png_bytes()).decode('ascii')}"


def test_desktop_screenshot_encoding_defaults_to_png(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NORTHCUA_DESKTOP_SCREENSHOT_FORMAT", raising=False)

    encoded = encode_desktop_image(Image.new("RGB", (4, 4), color=(32, 96, 160)))

    assert encoded.data_url.startswith("data:image/png;base64,")
    assert encoded.mime_type == "image/png"
    assert encoded.extension == ".png"


def test_desktop_screenshot_encoding_can_use_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORTHCUA_DESKTOP_SCREENSHOT_FORMAT", "jpeg")
    monkeypatch.setenv("NORTHCUA_DESKTOP_SCREENSHOT_JPEG_QUALITY", "75")

    encoded = encode_desktop_image(Image.new("RGB", (4, 4), color=(32, 96, 160)))

    assert encoded.data_url.startswith("data:image/jpeg;base64,")
    assert encoded.mime_type == "image/jpeg"
    assert encoded.extension == ".jpg"


class _FakeDesktopBackend:
    def __init__(self, *, frame: DesktopCoordinateFrame | None = None) -> None:
        self.frame = frame or DesktopCoordinateFrame(left=0, top=0, width=2, height=2)
        self.screenshot_calls = 0
        self.typed: list[tuple[str, TextTargetKind]] = []

    def capture_desktop_screenshot(self, all_screens: bool = True):
        self.screenshot_calls += 1
        return _png_data_url(), self.frame

    def capture_active_window_screenshot(self):
        return _png_data_url()

    def get_open_windows(self, remove_empty: bool = True, refresh_app_windows: bool = True):
        return []

    def get_active_window_info(self, field_list=None):
        return {}

    def get_active_window_controls_target_info(self, field_list=None):
        return []

    def click(self, x: float, y: float, button: str = "left") -> str:
        return "Clicked."

    def double_click(self, x: float, y: float, button: str = "left") -> str:
        return "Double-clicked."

    def drag(self, path, button: str = "left") -> str:
        return "Dragged."

    def mouse_down(self, button: str = "left") -> str:
        return "Mouse down."

    def mouse_up(self, button: str = "left") -> str:
        return "Mouse up."

    def move(self, x: float, y: float) -> str:
        return "Moved."

    def scroll(self, x=None, y=None, delta_x: float = 0, delta_y: float = 0) -> str:
        return "Scrolled."

    def type_text(self, text: str, target_kind: TextTargetKind = TextTargetKind.AUTO) -> str:
        self.typed.append((text, TextTargetKind.coerce(target_kind)))
        return "Typed."

    def key_down(self, key: str) -> str:
        return "Key down."

    def key_up(self, key: str) -> str:
        return "Key up."

    def keypress(self, key=None, keys=None) -> str:
        return "Pressed."


class _RecordingDesktopBackend(_FakeDesktopBackend):
    def __init__(self, *, frame: DesktopCoordinateFrame) -> None:
        super().__init__(frame=frame)
        self.clicked: list[tuple[float, float, str]] = []
        self.mouse_events: list[tuple[str, str]] = []
        self.key_events: list[tuple[str, str]] = []

    def click(self, x: float, y: float, button: str = "left") -> str:
        self.clicked.append((x, y, button))
        return "Clicked."

    def mouse_down(self, button: str = "left") -> str:
        self.mouse_events.append(("down", button))
        return "Mouse down."

    def mouse_up(self, button: str = "left") -> str:
        self.mouse_events.append(("up", button))
        return "Mouse up."

    def key_down(self, key: str) -> str:
        self.key_events.append(("down", key))
        return "Key down."

    def key_up(self, key: str) -> str:
        self.key_events.append(("up", key))
        return "Key up."


class _FakePyAutoGui:
    def __init__(self) -> None:
        self.pressed: list[str] = []
        self.key_downs: list[str] = []
        self.key_ups: list[str] = []
        self.hotkeys: list[tuple[str, ...]] = []
        self.hotkey_intervals: list[float | None] = []
        self.written: list[str] = []
        self.mouse_downs: list[str] = []
        self.mouse_ups: list[str] = []
        self.screenshot_calls = 0

    def press(self, key: str) -> None:
        self.pressed.append(key)

    def keyDown(self, key: str) -> None:
        self.key_downs.append(key)

    def keyUp(self, key: str) -> None:
        self.key_ups.append(key)

    def hotkey(self, *keys: str, interval: float | None = None) -> None:
        self.hotkeys.append(tuple(keys))
        self.hotkey_intervals.append(interval)

    def write(self, text: str, interval: float = 0.01) -> None:
        del interval
        self.written.append(text)

    def mouseDown(self, button: str = "left") -> None:
        self.mouse_downs.append(button)

    def mouseUp(self, button: str = "left") -> None:
        self.mouse_ups.append(button)

    def size(self) -> tuple[int, int]:
        return (2, 3)

    def screenshot(self) -> Image.Image:
        self.screenshot_calls += 1
        return Image.new("RGB", (2, 3), color=(32, 96, 160))


class _FakeClipboardStdin:
    def __init__(self) -> None:
        self.written = ""
        self.closed = False

    def write(self, text: str) -> int:
        self.written += text
        return len(text)

    def close(self) -> None:
        self.closed = True


class _FakeClipboardProcess:
    def __init__(self) -> None:
        self.stdin = _FakeClipboardStdin()
        self.pid = 123
        self.waits = 0
        self.terminated = False
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waits += 1
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


async def _tool_names(server) -> list[str]:
    return sorted(tool.name for tool in await server.list_tools())


def test_desktop_action_server_exposes_screenshot_tool(desktop_native_test_session) -> None:
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    server = NorthCUAMCPRegistry.build(
        "DesktopNativeActionExecutor",
        context=MCPFactoryContext(process_name="test-desktop-action"),
    )

    tool_names = asyncio.run(_tool_names(server))
    described = {
        descriptor.tool_id for descriptor in NorthCUAMCPRegistry.wire_tool_descriptors("DesktopNativeActionExecutor")
    }

    assert "note" in tool_names
    assert "screenshot" in tool_names
    assert "note" in described
    assert set(tool_names) == described
    assert "mouse_down" not in tool_names
    assert "mouse_up" not in tool_names
    assert "key_down" not in tool_names
    assert "key_up" not in tool_names


def test_desktop_action_note_tool_preserves_prompt_visible_text() -> None:
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    backend = _FakeDesktopBackend()

    with patch(
        "northcua.runtime.mcp.desktop_native_server.create_desktop_native_backend",
        return_value=backend,
    ):
        server = NorthCUAMCPRegistry.build(
            "DesktopNativeActionExecutor",
            context=MCPFactoryContext(process_name="test-desktop-action"),
        )
        result = asyncio.run(
            server.call_tool(
                "note",
                {
                    "text": "Ada email is ada.lovelace@northcua.test.",
                    "reason": "Need the exact value when filling the contact sheet.",
                    "kind": "fact",
                },
            )
        )

    payload = normalize_tool_result(result)

    assert payload == {
        "text": (
            "[fact] Ada email is ada.lovelace@northcua.test.\n"
            "Reason: Need the exact value when filling the contact sheet."
        ),
        "status": "recorded",
        "kind": "fact",
        "note_text": "Ada email is ada.lovelace@northcua.test.",
        "reason": "Need the exact value when filling the contact sheet.",
        "screenshot_policy": "none",
    }
    assert "screenshot_path" not in payload
    assert backend.screenshot_calls == 0
    DesktopNativeServerState().reset()


def test_desktop_type_tool_forwards_text_target_kind() -> None:
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    backend = _FakeDesktopBackend()

    with patch(
        "northcua.runtime.mcp.desktop_native_server.create_desktop_native_backend",
        return_value=backend,
    ):
        server = NorthCUAMCPRegistry.build(
            "DesktopNativeActionExecutor",
            context=MCPFactoryContext(process_name="test-desktop-action"),
        )
        asyncio.run(server.call_tool("type", {"text": "printf hi", "target_kind": "terminal"}))
        asyncio.run(server.call_tool("type", {"text": "Ada"}))

    assert backend.typed == [
        ("printf hi", TextTargetKind.TERMINAL),
        ("Ada", TextTargetKind.AUTO),
    ]
    DesktopNativeServerState().reset()


def test_desktop_type_tool_runtime_schema_exposes_target_kind_enum(desktop_native_test_session) -> None:
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    server = NorthCUAMCPRegistry.build(
        "DesktopNativeActionExecutor",
        context=MCPFactoryContext(process_name="test-desktop-action"),
    )

    tools = asyncio.run(server.list_tools())
    type_tool = next(tool for tool in tools if tool.name == "type")
    target_kind = type_tool.parameters["properties"]["target_kind"]

    assert target_kind["enum"] == ["auto", "editable", "terminal"]
    assert target_kind["default"] == "auto"
    assert "Ctrl+Shift+V" in target_kind["description"]
    DesktopNativeServerState().reset()


def test_desktop_keypress_runtime_schema_matches_unit_contract(desktop_native_test_session) -> None:
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    server = NorthCUAMCPRegistry.build(
        "DesktopNativeActionExecutor",
        context=MCPFactoryContext(process_name="test-desktop-action"),
    )

    tools = asyncio.run(server.list_tools())
    keypress_tool = next(tool for tool in tools if tool.name == "keypress")
    schema = keypress_tool.parameters

    assert "Provide exactly one of `key` or `keys`" in keypress_tool.description
    assert "Use this only when `keys` is omitted" in schema["properties"]["key"]["description"]
    assert "Each array item is one independent press unit" in schema["properties"]["keys"]["description"]
    DesktopNativeServerState().reset()


def test_desktop_keypress_tool_rejects_key_and_keys_together() -> None:
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    backend = _FakeDesktopBackend()

    with patch(
        "northcua.runtime.mcp.desktop_native_server.create_desktop_native_backend",
        return_value=backend,
    ):
        server = NorthCUAMCPRegistry.build(
            "DesktopNativeActionExecutor",
            context=MCPFactoryContext(process_name="test-desktop-action"),
        )
        with pytest.raises(Exception, match="exactly one of `key` or `keys`"):
            asyncio.run(server.call_tool("keypress", {"key": "enter", "keys": ["ctrl+l"]}))

    DesktopNativeServerState().reset()


def test_desktop_action_screenshot_tool_returns_action_style_image_payload() -> None:
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()

    with patch(
        "northcua.runtime.mcp.desktop_native_server.create_desktop_native_backend",
        return_value=_FakeDesktopBackend(),
    ):
        server = NorthCUAMCPRegistry.build(
            "DesktopNativeActionExecutor",
            context=MCPFactoryContext(process_name="test-desktop-action"),
        )
        result = asyncio.run(server.call_tool("screenshot", {"all_screens": False}))

    payload = normalize_tool_result(result)

    assert payload["text"] == "Captured a fresh desktop screenshot."
    assert payload["screenshot_format"] == "local_path"
    assert os.path.exists(payload["screenshot_path"])

    os.remove(payload["screenshot_path"])
    DesktopNativeServerState().reset()


def test_keypress_normalizes_meta_family_to_command_on_macos() -> None:
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()

    with patch("northcua.runtime.mcp.desktop_native._pyautogui_keys.platform.system", return_value="Darwin"):
        assert backend._normalize_keys(key=None, keys=["meta", "space"]) == ["command", "space"]
        assert backend._normalize_keys(key=None, keys=["cmd+space"]) == ["command", "space"]
        assert backend._normalize_keys(key=None, keys=["super", "opt", "escape"]) == [
            "command",
            "alt",
            "esc",
        ]


def test_clipboard_paste_shortcut_uses_terminal_variant_on_linux() -> None:
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    fake_pyautogui = _FakePyAutoGui()

    with (
        patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui),
        patch("northcua.runtime.mcp.desktop_native._pyautogui_clipboard.platform.system", return_value="Linux"),
    ):
        backend._paste_clipboard_shortcut(target_kind=TextTargetKind.TERMINAL)
        backend._paste_clipboard_shortcut(target_kind=TextTargetKind.EDITABLE)

    assert fake_pyautogui.hotkeys == [("ctrl", "shift", "v"), ("ctrl", "v")]


def test_explicit_text_target_requires_clipboard_success() -> None:
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()

    with patch.object(backend, "_type_text_via_clipboard", return_value=False):
        with pytest.raises(Exception, match="requires clipboard paste support"):
            backend.type_text("echo hi", target_kind=TextTargetKind.TERMINAL)


def test_clipboard_type_continues_when_previous_clipboard_is_unreadable() -> None:
    """An empty Linux clipboard should not prevent writing and pasting new text."""

    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    fake_pyautogui = _FakePyAutoGui()
    written: list[str] = []

    with (
        patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui),
        patch.object(backend, "_get_clipboard_text", side_effect=RuntimeError("empty clipboard")),
        patch.object(backend, "_set_clipboard_text", side_effect=lambda text: written.append(text)),
        patch("northcua.runtime.mcp.desktop_native._pyautogui_clipboard.platform.system", return_value="Linux"),
    ):
        assert backend._type_text_via_clipboard("Driver License", target_kind=TextTargetKind.EDITABLE)

    assert written == ["Driver License"]
    assert fake_pyautogui.hotkeys == [("ctrl", "v")]


def test_linux_xclip_write_is_non_blocking_until_paste() -> None:
    """xclip owns the clipboard from a live process, so typing must paste before waiting."""

    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    fake_pyautogui = _FakePyAutoGui()
    fake_process = _FakeClipboardProcess()
    popen_calls: list[list[str]] = []

    def _popen(command, **kwargs):
        del kwargs
        popen_calls.append(command)
        return fake_process

    def _which(binary: str) -> str | None:
        return "/usr/bin/xclip" if binary == "xclip" else None

    with (
        patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui),
        patch.object(backend, "_get_clipboard_text", return_value=""),
        patch("northcua.runtime.mcp.desktop_native._pyautogui_clipboard.platform.system", return_value="Linux"),
        patch("northcua.runtime.mcp.desktop_native._pyautogui_clipboard.shutil.which", side_effect=_which),
        patch("northcua.runtime.mcp.desktop_native._pyautogui_clipboard.subprocess.Popen", side_effect=_popen),
    ):
        assert backend._type_text_via_clipboard("Driver License", target_kind=TextTargetKind.EDITABLE)

    assert popen_calls == [["xclip", "-selection", "clipboard", "-loops", "1"]]
    assert fake_process.stdin.written == "Driver License"
    assert fake_process.stdin.closed
    assert fake_pyautogui.hotkeys == [("ctrl", "v")]
    assert fake_process.waits == 1
    assert not fake_process.terminated


def test_default_type_falls_back_when_clipboard_write_times_out() -> None:
    """A blocked Linux clipboard command should not degrade default typing."""

    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    fake_pyautogui = _FakePyAutoGui()

    with (
        patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui),
        patch.object(backend, "_get_clipboard_text", return_value=""),
        patch.object(
            backend,
            "_set_clipboard_text",
            side_effect=subprocess.TimeoutExpired(cmd=["xclip"], timeout=2.0),
        ),
        patch("northcua.runtime.mcp.desktop_native._pyautogui_clipboard.platform.system", return_value="Linux"),
    ):
        assert backend.type_text("Driver License", target_kind=TextTargetKind.AUTO) == "Typed 14 character(s)."

    assert fake_pyautogui.hotkeys == []
    assert fake_pyautogui.written == ["Driver License"]


def test_desktop_screenshot_falls_back_when_mss_has_no_monitors() -> None:
    class _EmptyMSS:
        monitors: list[dict[str, int]] = []

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    class _MSSModule:
        @staticmethod
        def mss() -> _EmptyMSS:
            return _EmptyMSS()

    fake_pyautogui = _FakePyAutoGui()
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with (
        patch.object(backend, "_import_mss", return_value=_MSSModule()),
        patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui),
    ):
        screenshot, frame = backend.capture_desktop_screenshot()

    assert screenshot.startswith("data:image/png;base64,")
    assert frame == DesktopCoordinateFrame(left=0, top=0, width=2, height=3)
    assert fake_pyautogui.screenshot_calls == 1


def test_desktop_screenshot_uses_aggregate_monitor_when_only_one_mss_monitor_exists() -> None:
    class _Screenshot:
        width = 2
        height = 2
        rgb = b"\x20\x60\xa0" * 4

    class _SingleMonitorMSS:
        monitors = [{"left": 5, "top": 7, "width": 2, "height": 2}]

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def grab(self, monitor):
            assert monitor == self.monitors[0]
            return _Screenshot()

    class _MSSModule:
        @staticmethod
        def mss() -> _SingleMonitorMSS:
            return _SingleMonitorMSS()

    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with patch.object(backend, "_import_mss", return_value=_MSSModule()):
        screenshot, frame = backend.capture_desktop_screenshot(all_screens=False)

    assert screenshot.startswith("data:image/png;base64,")
    assert frame == DesktopCoordinateFrame(left=5, top=7, width=2, height=2)


def test_desktop_screenshot_falls_back_when_mss_grab_fails() -> None:
    class _FailingMSS:
        monitors = [{"left": 5, "top": 7, "width": 2, "height": 2}]

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def grab(self, monitor):
            del monitor
            raise RuntimeError("CoreGraphics capture failed")

    class _MSSModule:
        @staticmethod
        def mss() -> _FailingMSS:
            return _FailingMSS()

    fake_pyautogui = _FakePyAutoGui()
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with (
        patch.object(backend, "_import_mss", return_value=_MSSModule()),
        patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui),
    ):
        screenshot, frame = backend.capture_desktop_screenshot(all_screens=False)

    assert screenshot.startswith("data:image/png;base64,")
    assert frame == DesktopCoordinateFrame(left=0, top=0, width=2, height=3)
    assert fake_pyautogui.screenshot_calls == 1
