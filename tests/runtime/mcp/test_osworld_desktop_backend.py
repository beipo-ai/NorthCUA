from __future__ import annotations

import asyncio
from io import BytesIO

import pytest
from PIL import Image

from northcua.core.coordinates import CoordinatePolicy, CoordinateTarget, coordinate_policy_context_values
from northcua.core.enums import CoordinateSpace
from northcua.runtime.mcp import MCPFactoryContext, NorthCUAMCPRegistry, ensure_builtin_mcp_servers_registered
from northcua.runtime.mcp.desktop_native.backend import create_desktop_native_backend
from northcua.runtime.mcp.desktop_native.osworld_backend import OSWorldDesktopBackend
from northcua.runtime.mcp.desktop_native_server import DesktopNativeServerState
from northcua.runtime.mcp.result_adapter import normalize_tool_result
from northcua_runtime_shared.desktop_surface.models import TextTargetKind


def _png_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(32, 96, 160))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeOSWorldResponse:
    def __init__(self, *, content: bytes = b"", payload: dict[str, object] | None = None) -> None:
        self.content = content
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeOSWorldSession:
    def __init__(self) -> None:
        self.trust_env = True
        self.requests: list[tuple[str, str, dict[str, object] | None, float]] = []

    def request(self, method: str, url: str, *, json=None, timeout: float):
        self.requests.append((method, url, json, timeout))
        if url.endswith("/screenshot"):
            return _FakeOSWorldResponse(content=_png_bytes())
        return _FakeOSWorldResponse(payload={"status": "success", "output": "{}"})


def test_desktop_backend_factory_selects_osworld_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_DESKTOP_BACKEND", "osworld")
    monkeypatch.setenv("NORTHCUA_OSWORLD_RETRY_INTERVAL_SECONDS", "0")

    backend = create_desktop_native_backend()

    assert isinstance(backend, OSWorldDesktopBackend)


def test_osworld_action_server_note_is_text_only_and_never_calls_vm(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_DESKTOP_BACKEND", "osworld")
    monkeypatch.setenv("NORTHCUA_OSWORLD_RETRY_INTERVAL_SECONDS", "0")

    def _unexpected_request(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("note should not call the OSWorld VM")

    monkeypatch.setattr(OSWorldDesktopBackend, "_request", _unexpected_request)
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    server = NorthCUAMCPRegistry.build(
        "DesktopNativeActionExecutor",
        context=MCPFactoryContext(process_name="test-osworld-action"),
    )

    result = asyncio.run(
        server.call_tool(
            "note",
            {
                "text": "Retry by clearing the field before typing.",
                "reason": "All-select did not replace the existing text.",
                "kind": "precaution",
            },
        )
    )

    payload = normalize_tool_result(result)
    assert payload["text"] == (
        "[precaution] Retry by clearing the field before typing.\nReason: All-select did not replace the existing text."
    )
    assert payload["screenshot_policy"] == "none"
    assert "screenshot_path" not in payload
    DesktopNativeServerState().reset()


def test_osworld_backend_captures_screenshot_and_dispatches_action(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_OSWORLD_VM_IP", "10.0.0.20")
    monkeypatch.setenv("NORTHCUA_OSWORLD_SERVER_PORT", "7000")
    monkeypatch.setenv("NORTHCUA_OSWORLD_RETRY_INTERVAL_SECONDS", "0")
    backend = OSWorldDesktopBackend()
    fake_session = _FakeOSWorldSession()
    backend._session = fake_session

    screenshot, frame = backend.capture_desktop_screenshot()
    result = backend.click(12.4, 56.6, button="right")

    assert screenshot.startswith("data:image/png;base64,")
    assert frame.width == 2
    assert frame.height == 2
    assert result == "Clicked at (12, 57) with right button."
    assert fake_session.requests[0][0] == "GET"
    assert fake_session.requests[0][1] == "http://10.0.0.20:7000/screenshot"
    assert fake_session.requests[1][0] == "POST"
    assert fake_session.requests[1][1] == "http://10.0.0.20:7000/run_python"
    assert "pyautogui.click(12, 57, button='right')" in str(fake_session.requests[1][2])


def test_osworld_action_server_maps_normalized_1000_before_pyautogui(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_DESKTOP_BACKEND", "osworld")
    monkeypatch.setenv("NORTHCUA_OSWORLD_RETRY_INTERVAL_SECONDS", "0")
    ensure_builtin_mcp_servers_registered()
    DesktopNativeServerState().reset()
    backend = OSWorldDesktopBackend()
    fake_session = _FakeOSWorldSession()
    backend._session = fake_session
    context = MCPFactoryContext(
        process_name="test-osworld-normalized",
        extra=coordinate_policy_context_values(
            CoordinateTarget.DESKTOP,
            CoordinatePolicy(input_space=CoordinateSpace.NORMALIZED_1000),
        ),
    )

    monkeypatch.setattr("northcua.runtime.mcp.desktop_native_server.create_desktop_native_backend", lambda: backend)
    server = NorthCUAMCPRegistry.build("DesktopNativeActionExecutor", context=context)
    asyncio.run(server.call_tool("screenshot", {"all_screens": False}))
    result = normalize_tool_result(asyncio.run(server.call_tool("click", {"x": 500, "y": 500})))

    assert isinstance(result, dict)
    assert result["tool_input_pixels"] == {"x": 1, "y": 1, "button": "left"}
    assert "pyautogui.click(1, 1, button='left')" in str(fake_session.requests[1][2])
    DesktopNativeServerState().reset()


def test_osworld_type_target_kind_uses_clipboard_paste_script(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_OSWORLD_RETRY_INTERVAL_SECONDS", "0")
    backend = OSWorldDesktopBackend()
    fake_session = _FakeOSWorldSession()
    backend._session = fake_session

    result = backend.type_text("echo hello", target_kind=TextTargetKind.TERMINAL)

    assert result == "Typed 10 character(s)."
    payload = fake_session.requests[0][2]
    assert payload is not None
    code = str(payload["code"])
    assert '"target_kind": "terminal"' in code
    assert "xclip" in code
    assert 'pyautogui.hotkey("ctrl", "shift", "v", interval=0.05)' in code


def test_osworld_type_auto_preserves_legacy_pyautogui_write(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_OSWORLD_RETRY_INTERVAL_SECONDS", "0")
    backend = OSWorldDesktopBackend()
    fake_session = _FakeOSWorldSession()
    backend._session = fake_session

    result = backend.type_text("Ada", target_kind=TextTargetKind.AUTO)

    assert result == "Typed 3 character(s)."
    payload = fake_session.requests[0][2]
    assert payload is not None
    code = str(payload["code"])
    assert "pyautogui.write('Ada', interval=0.01)" in code
    assert "xclip" not in code


def test_osworld_keypress_batches_repeated_chord_units(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_OSWORLD_RETRY_INTERVAL_SECONDS", "0")
    backend = OSWorldDesktopBackend()
    fake_session = _FakeOSWorldSession()
    backend._session = fake_session

    result = backend.keypress(keys=["ctrl+l", "ctrl+l", "tab", "ctrl+s"])

    assert result == "Pressed keys: ctrl+l+ctrl+l+tab+ctrl+s."
    assert len(fake_session.requests) == 1
    payload = fake_session.requests[0][2]
    assert payload is not None
    code = str(payload["code"])
    assert 'json.loads(\'[["ctrl", "l"], ["ctrl", "l"], ["tab"], ["ctrl", "s"]]\')' in code
    assert "pyautogui.hotkey(*unit, interval=0.05)" in code


def test_osworld_keypress_rejects_key_and_keys_together(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_OSWORLD_RETRY_INTERVAL_SECONDS", "0")
    backend = OSWorldDesktopBackend()
    fake_session = _FakeOSWorldSession()
    backend._session = fake_session

    with pytest.raises(Exception, match="exactly one of `key` or `keys`"):
        backend.keypress(key="enter", keys=["ctrl+l"])
    assert fake_session.requests == []


def test_osworld_keypress_uses_linux_aliases_on_macos_host(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_OSWORLD_RETRY_INTERVAL_SECONDS", "0")
    backend = OSWorldDesktopBackend()
    fake_session = _FakeOSWorldSession()
    backend._session = fake_session

    result = backend.keypress(keys=["meta", "space"])

    assert result == "Pressed keys: win+space."
    payload = fake_session.requests[0][2]
    assert payload is not None
    assert "pyautogui.hotkey('win', 'space', interval=0.1)" in str(payload["code"])
