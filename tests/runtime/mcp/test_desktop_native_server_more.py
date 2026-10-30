from __future__ import annotations

import asyncio
import base64
import os
from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image

from northcua.core.coordinates import CoordinatePolicy, CoordinateTarget, coordinate_policy_context_values
from northcua.core.enums import CoordinateSpace
from northcua.runtime.mcp import MCPFactoryContext, NorthCUAMCPRegistry, ensure_builtin_mcp_servers_registered
from northcua.runtime.mcp.desktop_native._models import DesktopNativeEnvironmentError
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


class _FakeDesktopBackend:
    def __init__(self, *, frame: DesktopCoordinateFrame | None = None) -> None:
        self.frame = frame or DesktopCoordinateFrame(left=0, top=0, width=2, height=2)

    def capture_desktop_screenshot(self, all_screens: bool = True):
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
        del target_kind
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


def test_desktop_screenshot_retries_transient_mss_grab_failure() -> None:
    class _Screenshot:
        width = 2
        height = 2
        rgb = b"\x20\x60\xa0" * 4

    class _FlakyMSS:
        monitors = [{"left": 5, "top": 7, "width": 2, "height": 2}]
        grab_attempts = 0

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def grab(self, monitor):
            del monitor
            type(self).grab_attempts += 1
            if type(self).grab_attempts == 1:
                raise RuntimeError("transient CoreGraphics capture failed")
            return _Screenshot()

    class _MSSModule:
        @staticmethod
        def mss() -> _FlakyMSS:
            return _FlakyMSS()

    fake_pyautogui = _FakePyAutoGui()
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with (
        patch.object(backend, "_import_mss", return_value=_MSSModule()),
        patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui),
    ):
        screenshot, frame = backend.capture_desktop_screenshot(all_screens=False)

    assert screenshot.startswith("data:image/png;base64,")
    assert frame == DesktopCoordinateFrame(left=5, top=7, width=2, height=2)
    assert _FlakyMSS.grab_attempts == 2
    assert fake_pyautogui.screenshot_calls == 0


def test_desktop_screenshot_rejects_black_pyautogui_fallback() -> None:
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

    class _BlackPyAutoGui(_FakePyAutoGui):
        def screenshot(self) -> Image.Image:
            self.screenshot_calls += 1
            return Image.new("RGB", (2, 3), color=(0, 0, 0))

    fake_pyautogui = _BlackPyAutoGui()
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with (
        patch.object(backend, "_import_mss", return_value=_MSSModule()),
        patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui),
    ):
        with pytest.raises(Exception, match="fallback image was all black"):
            backend.capture_desktop_screenshot(all_screens=False)

    assert fake_pyautogui.screenshot_calls == 1


def test_keypress_normalizes_meta_family_to_win_off_macos() -> None:
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()

    with patch("northcua.runtime.mcp.desktop_native._pyautogui_keys.platform.system", return_value="Windows"):
        assert backend._normalize_keys(key=None, keys=["meta", "space"]) == ["win", "space"]
        assert backend._normalize_keys(key=None, keys=["windows+tab"]) == ["win", "tab"]


def test_keypress_treats_modifier_plus_key_as_hotkey_on_macos() -> None:
    fake_pyautogui = _FakePyAutoGui()
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with (
        patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui),
        patch("northcua.runtime.mcp.desktop_native._pyautogui_keys.platform.system", return_value="Darwin"),
    ):
        backend.keypress(keys=["cmd", "space"])

    assert fake_pyautogui.hotkeys == [("command", "space")]
    assert fake_pyautogui.hotkey_intervals == [0.1]
    assert fake_pyautogui.written == []
    assert fake_pyautogui.pressed == []


@pytest.mark.parametrize(
    ("system_name", "environment", "error"),
    (
        ("Linux", {}, "requires a GUI X11 session"),
        ("Linux", {"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"}, "Wayland is not supported"),
        ("Plan9", {}, "Unsupported desktop platform"),
    ),
)
def test_desktop_backend_rejects_unsupported_session(
    monkeypatch: pytest.MonkeyPatch,
    system_name: str,
    environment: dict[str, str],
    error: str,
) -> None:
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "northcua.runtime.mcp.desktop_native._pyautogui_support.platform.system",
        lambda: system_name,
    )

    with pytest.raises(DesktopNativeEnvironmentError, match=error):
        PyAutoGuiDesktopBackend()


def test_desktop_backend_accepts_linux_x11(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setattr(
        "northcua.runtime.mcp.desktop_native._pyautogui_support.platform.system",
        lambda: "Linux",
    )

    PyAutoGuiDesktopBackend()


def test_keypress_rejects_key_and_keys_together() -> None:
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()

    with pytest.raises(Exception, match="exactly one of `key` or `keys`"):
        backend.keypress(key="enter", keys=["ctrl+l"])


def test_keypress_repeated_chord_units_are_dispatched_independently() -> None:
    fake_pyautogui = _FakePyAutoGui()
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with (
        patch("northcua.runtime.mcp.desktop_native._pyautogui_keys.platform.system", return_value="Linux"),
        patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui),
    ):
        backend.keypress(keys=["ctrl+l", "ctrl+l", "tab", "ctrl+s"])

    assert fake_pyautogui.hotkeys == [("ctrl", "l"), ("ctrl", "l"), ("ctrl", "s")]
    assert fake_pyautogui.hotkey_intervals == [0.1, 0.1, 0.1]
    assert fake_pyautogui.pressed == ["tab"]
    assert fake_pyautogui.written == []


def test_keypress_plus_unit_is_always_simultaneous_hotkey() -> None:
    fake_pyautogui = _FakePyAutoGui()
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui):
        backend.keypress(keys=["a+b"])

    assert fake_pyautogui.hotkeys == [("a", "b")]
    assert fake_pyautogui.hotkey_intervals == [0.1]
    assert fake_pyautogui.written == []
    assert fake_pyautogui.pressed == []


def test_keypress_treats_multi_key_text_input_as_sequence() -> None:
    fake_pyautogui = _FakePyAutoGui()
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui):
        backend.keypress(keys=["1", "2", "+", "3", "4", "enter"])

    assert fake_pyautogui.hotkeys == []
    assert fake_pyautogui.written == ["12+34"]
    assert fake_pyautogui.pressed == ["enter"]


def test_mouse_hold_primitives_normalize_button_without_model_tool_exposure() -> None:
    fake_pyautogui = _FakePyAutoGui()
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui):
        assert backend.mouse_down(button="wheel") == "Pressed and held middle mouse button."
        assert backend.mouse_up(button="bad-button") == "Released left mouse button."

    assert fake_pyautogui.mouse_downs == ["middle"]
    assert fake_pyautogui.mouse_ups == ["left"]


def test_key_hold_primitives_use_single_normalized_key_without_model_tool_exposure() -> None:
    fake_pyautogui = _FakePyAutoGui()
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()
    with patch("northcua.runtime.mcp.desktop_native._pyautogui_keys.platform.system", return_value="Darwin"):
        with patch.object(backend, "_import_pyautogui", return_value=fake_pyautogui):
            assert backend.key_down("cmd") == "Pressed and held key: command."
            assert backend.key_up("escape") == "Released key: esc."

    assert fake_pyautogui.key_downs == ["command"]
    assert fake_pyautogui.key_ups == ["esc"]


def test_key_hold_rejects_multi_key_input() -> None:
    with patch.object(PyAutoGuiDesktopBackend, "_ensure_supported_environment", return_value=None):
        backend = PyAutoGuiDesktopBackend()

    with pytest.raises(Exception, match="exactly one key"):
        backend.key_down("cmd+shift")


def test_desktop_action_server_maps_normalized_coordinates_via_shared_policy() -> None:
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    backend = _RecordingDesktopBackend(frame=DesktopCoordinateFrame(left=100, top=200, width=400, height=300))
    context = MCPFactoryContext(
        process_name="test-desktop-action",
        extra=coordinate_policy_context_values(
            CoordinateTarget.DESKTOP,
            CoordinatePolicy(input_space=CoordinateSpace.NORMALIZED_1000),
        ),
    )

    with patch(
        "northcua.runtime.mcp.desktop_native_server.create_desktop_native_backend",
        return_value=backend,
    ):
        server = NorthCUAMCPRegistry.build("DesktopNativeActionExecutor", context=context)
        asyncio.run(server.call_tool("screenshot", {"all_screens": False}))
        result = normalize_tool_result(asyncio.run(server.call_tool("click", {"x": 500, "y": 500})))

    assert backend.clicked == [(300, 350, "left")]
    assert isinstance(result, dict)
    assert result["tool_input_model"] == {"x": 500, "y": 500, "button": "left"}
    assert result["tool_input_pixels"] == {"x": 200, "y": 150, "button": "left"}
    assert result["tool_input_runtime_pixels"] == {"x": 300, "y": 350, "button": "left"}
    assert result["coordinate_policy"] == {"input_space": "normalized_1000", "clamp_to_bounds": False}
    DesktopNativeServerState().reset()


def test_desktop_action_server_accepts_absolute_pixels_without_frame_offset_translation() -> None:
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    backend = _RecordingDesktopBackend(frame=DesktopCoordinateFrame(left=100, top=200, width=400, height=300))
    context = MCPFactoryContext(
        process_name="test-desktop-action",
        extra=coordinate_policy_context_values(
            CoordinateTarget.DESKTOP,
            CoordinatePolicy(input_space=CoordinateSpace.ABSOLUTE_PIXELS),
        ),
    )

    with patch(
        "northcua.runtime.mcp.desktop_native_server.create_desktop_native_backend",
        return_value=backend,
    ):
        server = NorthCUAMCPRegistry.build("DesktopNativeActionExecutor", context=context)
        asyncio.run(server.call_tool("screenshot", {"all_screens": False}))
        asyncio.run(server.call_tool("click", {"x": 350, "y": 420}))

    assert backend.clicked == [(350, 420, "left")]
    DesktopNativeServerState().reset()
