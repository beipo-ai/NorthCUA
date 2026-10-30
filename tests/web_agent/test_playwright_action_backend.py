from __future__ import annotations

import asyncio
import base64

import pytest
from pydantic import ValidationError

from northcua.agents.web.backend import (
    PlaywrightWebActionBackend,
    WebActionBackendUnavailableError,
    WebActionCommandError,
    build_playwright_web_backend,
)
from northcua.agents.web.domain.actions import WebMouseButton, WebWaitParams
from northcua.agents.web.mcp.descriptors import web_agent_action_contracts
from northcua.core.config import WebAgentConfig


class _FakeMouse:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def click(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("click", args, kwargs))

    async def move(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("move", args, kwargs))

    async def down(self) -> None:
        self.calls.append(("down", (), {}))

    async def up(self) -> None:
        self.calls.append(("up", (), {}))

    async def wheel(self, *args: object) -> None:
        self.calls.append(("wheel", args, {}))


class _FakeKeyboard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def type(self, text: str) -> None:
        self.calls.append(("type", text))

    async def press(self, key: str) -> None:
        self.calls.append(("press", key))


class _FakeLocator:
    def __init__(
        self,
        *,
        text: str = "Hello",
        visible: bool = True,
        enabled: bool = True,
        tag: str = "button",
        input_type: str = "",
        role: str = "",
        contenteditable: str = "",
        hidden: bool = False,
        disabled: bool = False,
        inert: bool = False,
        pointer_events: str = "auto",
        target: str = "",
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self.text = text
        self.visible = visible
        self.enabled = enabled
        self.tag = tag
        self.input_type = input_type
        self.role = role
        self.contenteditable = contenteditable
        self.hidden = hidden
        self.disabled = disabled
        self.inert = inert
        self.pointer_events = pointer_events
        self.target = target
        self.first = self

    async def count(self) -> int:
        return 1

    async def click(self, **kwargs: object) -> None:
        self.calls.append(("click", kwargs))

    async def hover(self, **kwargs: object) -> None:
        self.calls.append(("hover", kwargs))

    async def fill(self, text: str, **kwargs: object) -> None:
        self.calls.append(("fill", text))

    async def select_option(self, values: list[str], **kwargs: object) -> list[str]:
        self.calls.append(("select_option", list(values)))
        return values

    async def check(self, **kwargs: object) -> None:
        self.calls.append(("check", kwargs))

    async def uncheck(self, **kwargs: object) -> None:
        self.calls.append(("uncheck", kwargs))

    async def drag_to(self, target: object, **kwargs: object) -> None:
        self.calls.append(("drag_to", target))

    async def inner_text(self, **kwargs: object) -> str:
        self.calls.append(("inner_text", kwargs))
        return self.text

    async def bounding_box(self, **kwargs: object) -> dict[str, float]:
        self.calls.append(("bounding_box", kwargs))
        return {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}

    async def is_visible(self, **kwargs: object) -> bool:
        self.calls.append(("is_visible", kwargs))
        return self.visible

    async def is_enabled(self, **kwargs: object) -> bool:
        self.calls.append(("is_enabled", kwargs))
        return self.enabled

    async def wait_for(self, **kwargs: object) -> None:
        self.calls.append(("wait_for", kwargs))

    async def get_attribute(self, name: str, **kwargs: object) -> str | None:
        self.calls.append(("get_attribute", {"name": name, **kwargs}))
        return self.target if name == "target" and self.target else None

    async def evaluate(self, _script: str, kind: str | None = None, **kwargs: object) -> bool | dict[str, bool | str]:
        self.calls.append(("evaluate", {"kind": kind, **kwargs}))
        if kind is None:
            if "pointerEvents" in _script:
                return {
                    "blocked": self.pointer_events == "none",
                    "reason": "target has pointer-events:none" if self.pointer_events == "none" else "",
                }
            return {"hidden": self.hidden, "disabled": self.disabled or self.inert}
        if kind == "select":
            return self.tag == "select"
        if kind == "checkable":
            return self.tag == "input" and self.input_type in {"checkbox", "radio"}
        if kind == "fillable":
            return (
                self.tag in {"input", "textarea"} or self.contenteditable in {"true", "empty"} or self.role == "textbox"
            )
        return True


class _PopupLocator(_FakeLocator):
    def __init__(self, context: "_FakeContext", **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.context = context

    async def click(self, **kwargs: object) -> None:
        await super().click(**kwargs)
        if kwargs.get("trial") is True:
            return
        await self.context.new_page()


class _DelayedPopupLocator(_FakeLocator):
    def __init__(self, context: "_FakeContext", **kwargs: object) -> None:
        super().__init__(target="_blank", **kwargs)
        self.context = context

    async def click(self, **kwargs: object) -> None:
        await super().click(**kwargs)
        if kwargs.get("trial") is True:
            return

        async def _open_later() -> None:
            await asyncio.sleep(0.02)
            await self.context.new_page()

        asyncio.create_task(_open_later())


class _FakePage:
    def __init__(self) -> None:
        self.url = "https://example.com"
        self.viewport_size = {"width": 800, "height": 600}
        self.mouse = _FakeMouse()
        self.keyboard = _FakeKeyboard()
        self.locators = {
            "[data-northcua-web-agent-ref='e1']": _FakeLocator(text="Button"),
            "[data-northcua-web-agent-ref='e2']": _FakeLocator(text="Input", tag="input"),
        }
        self.calls: list[tuple[str, object]] = []
        self.closed = False
        self.owner: _FakeContext | None = None
        self.frames = [self]

    async def evaluate(self, script: str, arg: object | None = None) -> object:
        self.calls.append(("evaluate", arg))
        if isinstance(arg, dict):
            return {
                "url": self.url,
                "title": "Example",
                "origin": "https://example.com",
                "snapshot": 'button "Submit" [ref=@e1]\ntextbox "Email" [ref=@e2]',
                "refs": {
                    "e1": {"role": "button", "name": "Submit", "text": "Submit", "tag": "button"},
                    "e2": {"role": "textbox", "name": "Email", "text": "", "tag": "input"},
                },
                "visibleText": "Submit Email Static repository row climate-viz 12.8k stars",
                "visibleTextChars": 59,
                "visibleTextTruncated": False,
            }
        if "devicePixelRatio" in script:
            return 2
        return {"width": 800, "height": 1200}

    def locator(self, selector: str) -> _FakeLocator:
        return self.locators.get(selector, _MissingLocator())

    async def screenshot(self, **kwargs: object) -> bytes:
        self.calls.append(("screenshot", kwargs))
        return b"fake-png"

    async def goto(self, url: str, **kwargs: object) -> None:
        self.url = url
        self.calls.append(("goto", kwargs))

    async def bring_to_front(self) -> None:
        self.calls.append(("bring_to_front", None))

    async def title(self) -> str:
        return "Example"

    async def wait_for_load_state(self, state: str, **kwargs: object) -> None:
        self.calls.append(("wait_for_load_state", state))

    def get_by_text(self, text: str) -> _FakeLocator:
        self.calls.append(("get_by_text", text))
        return _FakeLocator(text=text)

    async def wait_for_url(self, pattern: str, **kwargs: object) -> None:
        self.calls.append(("wait_for_url", pattern))

    async def wait_for_selector(self, selector: str, **kwargs: object) -> None:
        self.calls.append(("wait_for_selector", selector))

    async def close(self) -> None:
        self.closed = True
        if self.owner is not None and self in self.owner.pages:
            self.owner.pages.remove(self)


class _MissingLocator(_FakeLocator):
    async def count(self) -> int:
        return 0


class _FakeContext:
    def __init__(self, *pages: _FakePage) -> None:
        self.pages = list(pages) or [_FakePage()]
        for page in self.pages:
            page.owner = self

    async def new_page(self) -> _FakePage:
        page = _FakePage()
        page.owner = self
        self.pages.append(page)
        return page


class _FakeCdpSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []
        self.detached = False

    async def send(self, method: str, params: object | None = None) -> dict[str, object]:
        self.calls.append((method, params))
        if method == "Page.getLayoutMetrics":
            return {"contentSize": {"width": 800, "height": 1200}}
        if method == "Page.captureScreenshot":
            return {"data": base64.b64encode(b"cdp-png").decode("ascii")}
        return {}

    async def detach(self) -> None:
        self.detached = True


class _FakeCdpContext(_FakeContext):
    def __init__(self, *pages: _FakePage) -> None:
        super().__init__(*pages)
        self.cdp_session = _FakeCdpSession()

    async def new_cdp_session(self, page: _FakePage) -> _FakeCdpSession:
        assert page in self.pages
        return self.cdp_session


class _TimeoutScreenshotPage(_FakePage):
    async def screenshot(self, **kwargs: object) -> bytes:
        self.calls.append(("screenshot", kwargs))
        raise TimeoutError("fonts never settled")


class _ProbePage(_FakePage):
    async def evaluate(self, script: str, arg: object | None = None) -> object:
        if isinstance(arg, dict) and "refs" in arg:
            refs = [f"@{ref}" for ref in arg["refs"]]  # type: ignore[index]
            return {
                "boxes": {ref: {"x": 10, "y": 20, "width": 30, "height": 40} for ref in refs},
                "states": {ref: {"visible": True, "enabled": True} for ref in refs},
                "errors": {},
            }
        return await super().evaluate(script, arg)


def _backend_with_context(context: _FakeContext) -> PlaywrightWebActionBackend:
    backend = PlaywrightWebActionBackend(config=WebAgentConfig(), session_id="sess")

    async def _context() -> _FakeContext:
        return context

    backend._browser_context = _context  # type: ignore[method-assign]
    return backend


def test_web_agent_config_uses_playwright_browser_defaults() -> None:
    assert WebAgentConfig().playwright_browser == "chromium"


def test_web_agent_config_rejects_backend_selector() -> None:
    with pytest.raises(ValidationError, match="action_backend"):
        WebAgentConfig.model_validate({"action_backend": "playwright"})


def test_playwright_web_backend_factory_creates_playwright_backend() -> None:
    default_backend = build_playwright_web_backend(config=WebAgentConfig(), session_id="sess")

    assert isinstance(default_backend, PlaywrightWebActionBackend)


def test_playwright_backend_press_normalizes_common_key_aliases() -> None:
    page = _FakePage()
    backend = _backend_with_context(_FakeContext(page))

    async def _run() -> None:
        result = await backend.press("Left")
        assert result.require_success().data == {"key": "ArrowLeft"}

    asyncio.run(_run())
    assert ("press", "ArrowLeft") in page.keyboard.calls


def test_playwright_backend_does_not_change_model_visible_web_tool_surface() -> None:
    tool_ids = {contract.tool_id for contract in web_agent_action_contracts()}

    assert tool_ids
    assert all(tool_id.startswith("web_") for tool_id in tool_ids)
    assert not any(tool_id.startswith("browser_") for tool_id in tool_ids)


def test_playwright_backend_ref_actions_use_latest_snapshot_ref_map_only() -> None:
    page = _FakePage()
    backend = _backend_with_context(_FakeContext(page))

    async def _run() -> None:
        snapshot = await backend.snapshot()
        assert [element.model_ref for element in snapshot.elements] == ["@e1", "@e2"]

        click_result = await backend.click_ref("@e1")
        assert click_result.require_success().data == {"ref": "@e1"}
        click_calls = [call for call in page.locators["[data-northcua-web-agent-ref='e1']"].calls if call[0] == "click"]
        assert click_calls[-1][1].get("trial") is not True

        missing_result = await backend.click_ref("@missing")
        assert missing_result.success is False
        with pytest.raises(WebActionCommandError, match="unknown ref"):
            missing_result.require_success()

    asyncio.run(_run())


def test_playwright_backend_waits_for_delayed_target_blank_page() -> None:
    page = _FakePage()
    context = _FakeContext(page)
    page.locators["[data-northcua-web-agent-ref='e1']"] = _DelayedPopupLocator(context, text="Open details")
    backend = _backend_with_context(context)

    async def _run() -> None:
        await backend.snapshot()
        result = await backend.click_ref("@e1")
        assert result.require_success().data == {"ref": "@e1"}
        assert len(context.pages) == 2
        assert backend._active_page_index == 1

    asyncio.run(_run())


def test_playwright_backend_ref_action_preflight_fails_fast_for_disabled_target() -> None:
    page = _FakePage()
    page.locators["[data-northcua-web-agent-ref='e1']"] = _FakeLocator(text="Disabled", disabled=True)
    backend = _backend_with_context(_FakeContext(page))

    async def _run() -> None:
        await backend.snapshot()
        result = await backend.click_ref("@e1")
        assert result.success is False
        assert result.code == "element_disabled"
        assert result.reason_code == "element_disabled"
        assert result.feedback_source == "preflight"
        assert "disabled" in result.hint
        with pytest.raises(WebActionCommandError) as exc_info:
            result.require_success()
        assert exc_info.value.code == "element_disabled"

    asyncio.run(_run())
    assert page.locators["[data-northcua-web-agent-ref='e1']"].calls == [
        ("evaluate", {"kind": None, "timeout": WebAgentConfig().action_preflight_timeout_ms}),
    ]


def test_playwright_backend_ref_action_preflight_allows_disabled_hover() -> None:
    page = _FakePage()
    page.locators["[data-northcua-web-agent-ref='e1']"] = _FakeLocator(text="Disabled tooltip", disabled=True)
    backend = _backend_with_context(_FakeContext(page))

    async def _run() -> None:
        await backend.snapshot()
        result = await backend.hover_ref("@e1")
        assert result.success is True
        assert result.require_success().data == {"ref": "@e1"}

    asyncio.run(_run())
    assert page.locators["[data-northcua-web-agent-ref='e1']"].calls == [
        ("evaluate", {"kind": None, "timeout": WebAgentConfig().action_preflight_timeout_ms}),
        ("evaluate", {"kind": None, "timeout": WebAgentConfig().action_preflight_timeout_ms}),
        ("hover", {"timeout": WebAgentConfig().command_timeout_seconds * 1000}),
    ]


def test_playwright_backend_ref_action_preflight_rejects_wrong_action_for_element() -> None:
    page = _FakePage()
    page.locators["[data-northcua-web-agent-ref='e1']"] = _FakeLocator(text="Button", tag="button")
    backend = _backend_with_context(_FakeContext(page))

    async def _run() -> None:
        await backend.snapshot()
        result = await backend.fill_ref("@e1", "hello")
        assert result.success is False
        assert result.code == "wrong_action_for_element"
        assert "matches the element role" in result.hint
        assert not any(call[0] == "fill" for call in page.locators["[data-northcua-web-agent-ref='e1']"].calls)

    asyncio.run(_run())


def test_playwright_backend_ref_action_preflight_fails_fast_for_pointer_events_none() -> None:
    page = _FakePage()
    page.locators["[data-northcua-web-agent-ref='e1']"] = _FakeLocator(
        text="Overlay",
        pointer_events="none",
    )
    backend = _backend_with_context(_FakeContext(page))

    async def _run() -> None:
        await backend.snapshot()
        result = await backend.click_ref("@e1")
        assert result.success is False
        assert result.code == "element_not_receiving_events"
        assert result.reason_code == "element_not_receiving_events"
        assert result.feedback_source == "preflight"
        assert "covered" in result.hint
        assert not any(call[0] == "click" for call in page.locators["[data-northcua-web-agent-ref='e1']"].calls)

    asyncio.run(_run())


def test_playwright_backend_ref_action_preflight_allows_enabled_target() -> None:
    page = _FakePage()
    backend = _backend_with_context(_FakeContext(page))

    async def _run() -> None:
        await backend.snapshot()
        result = await backend.click_ref("@e1")
        assert result.require_success().data == {"ref": "@e1"}

    asyncio.run(_run())
    calls = page.locators["[data-northcua-web-agent-ref='e1']"].calls
    assert calls[0] == ("evaluate", {"kind": None, "timeout": WebAgentConfig().action_preflight_timeout_ms})
    assert calls[1] == ("evaluate", {"kind": None, "timeout": WebAgentConfig().action_preflight_timeout_ms})
    assert calls[2] == (
        "get_attribute",
        {"name": "target", "timeout": WebAgentConfig().action_preflight_timeout_ms},
    )
    assert calls[3][0] == "click"
    assert calls[3][1].get("trial") is not True


def test_playwright_backend_snapshot_preserves_visible_noninteractive_text() -> None:
    page = _FakePage()
    backend = _backend_with_context(_FakeContext(page))

    async def _run() -> None:
        snapshot = await backend.snapshot()
        assert snapshot.raw["visible_text_frames"][0]["text"] == (
            "Submit Email Static repository row climate-viz 12.8k stars"
        )
        assert snapshot.raw["visible_text_frames"][0]["truncated"] is False
        snapshot_args = [
            call[1] for call in page.calls if isinstance(call[1], dict) and "maxVisibleTextChars" in call[1]
        ]
        assert snapshot_args[-1]["maxVisibleTextChars"] == WebAgentConfig().snapshot_visible_text_max_chars

    asyncio.run(_run())


def test_playwright_backend_unavailable_does_not_silently_fallback() -> None:
    backend = PlaywrightWebActionBackend(config=WebAgentConfig(), session_id="sess")

    async def _fail() -> object:
        raise WebActionBackendUnavailableError("launch failed")

    backend._browser_context = _fail  # type: ignore[method-assign]

    async def _run() -> None:
        with pytest.raises(WebActionBackendUnavailableError, match="launch failed"):
            await backend.open("https://example.com")

    asyncio.run(_run())


def test_playwright_backend_screenshot_returns_inline_clean_image_without_path() -> None:
    backend = _backend_with_context(_FakeContext(_FakePage()))

    async def _run() -> None:
        screenshot = await backend.screenshot(annotate=False)
        assert screenshot.path == ""
        assert screenshot.data_url.startswith("data:image/png;base64,")
        assert screenshot.width == 800
        assert screenshot.height == 600

    asyncio.run(_run())


def test_playwright_backend_screenshot_uses_cdp_fallback_after_timeout() -> None:
    page = _TimeoutScreenshotPage()
    context = _FakeCdpContext(page)
    backend = _backend_with_context(context)

    async def _run() -> None:
        screenshot = await backend.screenshot(annotate=False)
        assert screenshot.path == ""
        assert screenshot.data_url == f"data:image/png;base64,{base64.b64encode(b'cdp-png').decode('ascii')}"
        assert screenshot.raw["fallback"] == "chromium_cdp_capture_screenshot"
        assert context.cdp_session.calls[-1][0] == "Page.captureScreenshot"
        assert context.cdp_session.calls[-1][1] == {
            "format": "png",
            "fromSurface": True,
            "captureBeyondViewport": False,
        }
        assert context.cdp_session.detached is True

    asyncio.run(_run())


def test_playwright_backend_wait_executes_exact_target() -> None:
    page = _FakePage()
    backend = _backend_with_context(_FakeContext(page))

    async def _run() -> None:
        waited = await backend.wait(WebWaitParams(duration_ms=50), timeout_seconds=0.001)
        assert waited.require_success().data == {"waited_ms": 1, "requested_ms": 50, "clamped": True}
        await backend.wait(WebWaitParams(load_state="domcontentloaded"))
        await backend.wait(WebWaitParams(text="Ready"))
        await backend.wait(WebWaitParams(url_pattern="**/done"))
        await backend.wait(WebWaitParams(selector="#app"))

    asyncio.run(_run())

    wait_calls = [call for call in page.calls if call[0].startswith("wait_for") or call[0] == "get_by_text"]
    assert wait_calls == [
        ("wait_for_load_state", "domcontentloaded"),
        ("get_by_text", "Ready"),
        ("wait_for_url", "**/done"),
        ("wait_for_selector", "#app"),
    ]


def test_playwright_backend_tabs_are_session_scoped_and_label_addressable() -> None:
    context = _FakeContext(_FakePage())
    backend = _backend_with_context(context)

    async def _run() -> None:
        created = await backend.new_tab(url="https://docs.example", label="docs")
        assert created.require_success().data["tab"]["label"] == "docs"  # type: ignore[index]
        listed = await backend.list_tabs()
        tabs = listed.require_success().data["tabs"]  # type: ignore[index]
        assert [tab["ref"] for tab in tabs] == ["t1", "t2"]

        focused = await backend.focus_tab("docs")
        assert focused.require_success().data["tab"]["ref"] == "t2"  # type: ignore[index]

        closed = await backend.close_tab("docs")
        assert closed.require_success().data["active"] == "t1"  # type: ignore[index]
        assert len(context.pages) == 1

    asyncio.run(_run())


def test_playwright_backend_ref_probe_batches_main_frame_boxes_and_states() -> None:
    page = _ProbePage()
    backend = _backend_with_context(_FakeContext(page))

    async def _run() -> None:
        await backend.snapshot()
        boxes = await backend.get_boxes(["@e1", "@e2"])
        states = await backend.get_ref_states(["@e1", "@e2"])
        assert set(boxes.require_success().data["boxes"]) == {"@e1", "@e2"}  # type: ignore[index]
        assert states.require_success().data["states"]["@e1"] == {"visible": True, "enabled": True}  # type: ignore[index]

    asyncio.run(_run())

    assert not page.locators["[data-northcua-web-agent-ref='e1']"].calls


def test_playwright_backend_click_ref_activates_new_popup_page() -> None:
    page = _FakePage()
    context = _FakeContext(page)
    page.locators["[data-northcua-web-agent-ref='e1']"] = _PopupLocator(context, text="Open")
    backend = _backend_with_context(context)

    async def _run() -> None:
        await backend.snapshot()
        result = await backend.click_ref("@e1")
        assert result.require_success().data == {"ref": "@e1"}
        listed = await backend.list_tabs()
        assert listed.require_success().data["active"] == "t2"  # type: ignore[index]

    asyncio.run(_run())
