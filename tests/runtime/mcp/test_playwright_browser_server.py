from __future__ import annotations

import asyncio

import pytest

from northcua.core.enums import PlaywrightBrowserEngine
from northcua.runtime.browser.backend import PythonPlaywrightBrowserBackend, resolve_playwright_browser_name
from northcua.runtime.mcp.invocation_context import MCPInvocationContext, use_mcp_invocation_context
from northcua.runtime.mcp.playwright_browser_server import (
    BrowserElementInfo,
    BrowserSnapshotResult,
    BrowserTabInfo,
    MCPFactoryContext,
    PlaywrightBrowserBackend,
    create_playwright_browser_mcp_server,
)
from northcua.runtime.mcp.result_adapter import normalize_tool_result


class _FakePlaywrightBackend(PlaywrightBrowserBackend):
    def __init__(self) -> None:
        self.click_calls: list[tuple[str, str | None, bool, list[str] | None]] = []

    def list_tabs(self) -> list[BrowserTabInfo]:
        return [BrowserTabInfo(index=0, title="Example", url="https://example.test", is_active=True)]

    def new_tab(self) -> BrowserTabInfo:
        return BrowserTabInfo(index=1, title="", url="", is_active=True)

    def close_tab(self, index: int | None = None) -> None:
        return None

    def select_tab(self, index: int) -> BrowserTabInfo:
        return BrowserTabInfo(index=index, title="Example", url="https://example.test", is_active=True)

    def navigate(self, url: str) -> str:
        return url

    def navigate_back(self) -> None:
        return None

    def navigate_forward(self) -> None:
        return None

    def reload(self) -> None:
        return None

    def capture_snapshot(self, selector: str | None = None, depth: int | None = None) -> BrowserSnapshotResult:
        del selector, depth
        return BrowserSnapshotResult(
            title="Example",
            url="https://example.test",
            elements=[
                BrowserElementInfo(
                    selector="#submit",
                    role="button",
                    name="Submit",
                    tag_name="button",
                    depth=0,
                )
            ],
        )

    def click(
        self,
        selector: str,
        *,
        button: str | None = None,
        double_click: bool = False,
        modifiers: list[str] | None = None,
    ) -> None:
        self.click_calls.append((selector, button, double_click, modifiers))

    def drag(self, start_selector: str, end_selector: str) -> None:
        return None

    def hover(self, selector: str) -> None:
        return None

    def select_option(self, selector: str, values: list[str]) -> None:
        return None

    def set_checked(self, selector: str, checked: bool) -> None:
        return None

    def stop(self) -> None:
        return None


def test_playwright_browser_server_exposes_expected_tools() -> None:
    server = create_playwright_browser_mcp_server(
        MCPFactoryContext(process_name="test-browser"),
        backend_factory=_FakePlaywrightBackend,
    )

    tool_names = asyncio.run(_tool_names(server))

    assert {
        "browser_tabs",
        "browser_navigate",
        "browser_navigate_back",
        "browser_navigate_forward",
        "browser_reload",
        "browser_snapshot",
        "browser_click",
        "browser_drag",
        "browser_hover",
        "browser_select_option",
        "browser_generate_locator",
        "browser_check",
        "browser_uncheck",
    }.issubset(set(tool_names))


def test_playwright_browser_server_resolves_snapshot_refs() -> None:
    backend = _FakePlaywrightBackend()
    server = create_playwright_browser_mcp_server(
        MCPFactoryContext(process_name="test-browser"),
        backend_factory=lambda: backend,
    )

    snapshot = asyncio.run(server.call_tool("browser_snapshot", {}))
    locator = asyncio.run(server.call_tool("browser_generate_locator", {"ref": "e1"}))
    click = asyncio.run(server.call_tool("browser_click", {"ref": "e1"}))

    assert '[e1] button "Submit" <button>' in normalize_tool_result(snapshot)
    assert normalize_tool_result(locator) == "#submit"
    assert normalize_tool_result(click) == "Successfully clicked e1."
    assert backend.click_calls == [("#submit", None, False, None)]


def test_playwright_browser_server_keeps_snapshot_refs_session_scoped() -> None:
    backend = _FakePlaywrightBackend()
    server = create_playwright_browser_mcp_server(
        MCPFactoryContext(process_name="test-browser"),
        backend_factory=lambda: backend,
    )

    async def _run() -> None:
        with use_mcp_invocation_context(MCPInvocationContext(session_id="sess-a")):
            await server.call_tool("browser_snapshot", {})
        with use_mcp_invocation_context(MCPInvocationContext(session_id="sess-b")):
            await server.call_tool("browser_click", {"ref": "e1"})

    with pytest.raises(Exception, match="Unknown ref 'e1'"):
        asyncio.run(_run())


def test_resolve_playwright_browser_name_accepts_enum_value() -> None:
    assert resolve_playwright_browser_name(PlaywrightBrowserEngine.CHROMIUM) == "chromium"
    assert resolve_playwright_browser_name("PlaywrightBrowserEngine.CHROMIUM") == "chromium"


def test_browser_settings_keep_legacy_fallback_and_prefer_canonical_names(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ("NORTHCUA", "NEXCUA"):
        for setting in ("BROWSER", "HEADLESS", "VIEWPORT"):
            monkeypatch.delenv(f"{prefix}_PLAYWRIGHT_{setting}", raising=False)
    monkeypatch.setenv("NEXCUA_PLAYWRIGHT_BROWSER", "firefox")
    monkeypatch.setenv("NEXCUA_PLAYWRIGHT_HEADLESS", "0")
    monkeypatch.setenv("NEXCUA_PLAYWRIGHT_VIEWPORT", "800x600")
    legacy = PythonPlaywrightBrowserBackend()
    assert legacy._browser_name == "firefox"
    assert legacy._headless is False
    assert legacy._viewport == {"width": 800, "height": 600}

    monkeypatch.setenv("NORTHCUA_PLAYWRIGHT_BROWSER", "webkit")
    monkeypatch.setenv("NORTHCUA_PLAYWRIGHT_HEADLESS", "1")
    monkeypatch.setenv("NORTHCUA_PLAYWRIGHT_VIEWPORT", "1024x768")
    canonical = PythonPlaywrightBrowserBackend()
    assert canonical._browser_name == "webkit"
    assert canonical._headless is True
    assert canonical._viewport == {"width": 1024, "height": 768}
    assert resolve_playwright_browser_name("chromium") == "chromium"


async def _tool_names(server) -> list[str]:
    return sorted(tool.name for tool in await server.list_tools())
