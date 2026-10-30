"""Run real Playwright backend web-agent E2E smoke cases.

The script exercises public no-login pages through the same backend and
observation collector used by the model-visible ``web_*`` tools. It writes a
path-free JSON report plus a JSONL trace summary. Raw screenshot bytes and
local screenshot paths are intentionally excluded from both files.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import http.server
import json
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from northcua.agents.web.artifacts import WebArtifactStore
from northcua.agents.web.backend import PlaywrightWebActionBackend, WebActionBackend
from northcua.agents.web.domain.actions import WebAgentActionType, WebScrollDirection, WebWaitParams
from northcua.agents.web.domain.observations import WebElementRef, WebObservationContext
from northcua.agents.web.evaluation.service import build_web_round_evaluator_spec
from northcua.agents.web.evaluation.tools import WebBrowserLiveProbeDispatcher
from northcua.agents.web.mcp.action_service import WebAgentMCPActionService
from northcua.agents.web.observation.projector import WebObservationProjector
from northcua.agents.web.session.models import RegisteredWebAgentSession
from northcua.core.config import ArtifactConfig, WebAgentConfig


@dataclass(slots=True)
class E2EStep:
    name: str
    status: str
    elapsed_ms: float
    details: dict[str, object] = field(default_factory=dict)
    error: str = ""


@dataclass(slots=True)
class E2ECase:
    name: str
    url: str
    runner: Callable[["E2ERunContext"], Awaitable[None]]


class E2ERunContext:
    """Small helper that records browser actions and observation QA."""

    def __init__(
        self,
        *,
        registered: RegisteredWebAgentSession,
        service: WebAgentMCPActionService,
        target_url: str,
    ) -> None:
        self.registered = registered
        self.client = registered.client
        self.service = service
        self.target_url = target_url
        self.projector = WebObservationProjector()
        self.steps: list[E2EStep] = []
        self.observations: list[dict[str, object]] = []
        self.live_probe_checks: list[dict[str, object]] = []

    async def record(self, name: str, operation: Callable[[], Awaitable[object]]) -> object:
        started = time.perf_counter()
        try:
            result = await operation()
        except Exception as exc:
            self.steps.append(
                E2EStep(
                    name=name,
                    status="error",
                    elapsed_ms=_elapsed_ms(started),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        details = _summarize_result(result)
        if hasattr(result, "success") and not bool(getattr(result, "success")):
            error = str(getattr(result, "error", ""))[:240]
            self.steps.append(
                E2EStep(name=name, status="error", elapsed_ms=_elapsed_ms(started), details=details, error=error)
            )
            raise RuntimeError(error or f"{name} failed")
        self.steps.append(E2EStep(name=name, status="ok", elapsed_ms=_elapsed_ms(started), details=details))
        return result

    async def record_expected_error(
        self,
        name: str,
        operation: Callable[[], Awaitable[object]],
        *,
        expected_code: str,
        expected_hint_contains: str = "",
    ) -> object:
        """Record an action that must fail with a specific agent-friendly code."""

        started = time.perf_counter()
        result = await operation()
        details = _summarize_result(result)
        code = str(details.get("code") or "")
        hint = str(details.get("hint") or "")
        if not hasattr(result, "success") or bool(getattr(result, "success")):
            self.steps.append(
                E2EStep(
                    name=name,
                    status="error",
                    elapsed_ms=_elapsed_ms(started),
                    details=details,
                    error=f"expected failure code {expected_code}, got success",
                )
            )
            raise RuntimeError(f"{name} unexpectedly succeeded")
        if code != expected_code:
            self.steps.append(
                E2EStep(
                    name=name,
                    status="error",
                    elapsed_ms=_elapsed_ms(started),
                    details=details,
                    error=f"expected code {expected_code}, got {code}",
                )
            )
            raise RuntimeError(f"{name} expected {expected_code}, got {code}")
        if expected_hint_contains and expected_hint_contains.lower() not in hint.lower():
            self.steps.append(
                E2EStep(
                    name=name,
                    status="error",
                    elapsed_ms=_elapsed_ms(started),
                    details=details,
                    error=f"expected hint containing {expected_hint_contains!r}, got {hint!r}",
                )
            )
            raise RuntimeError(f"{name} missing expected hint")
        self.steps.append(
            E2EStep(
                name=name,
                status="ok",
                elapsed_ms=_elapsed_ms(started),
                details={**details, "expected_error_code": expected_code},
            )
        )
        return result

    async def observe(self, name: str) -> WebObservationContext:
        started = time.perf_counter()
        try:
            observation = await self.service.collect_current_observation(
                self.registered,
                mode="som",
                observation_id=name,
                validate_url=True,
            )
        except Exception as exc:
            self.steps.append(
                E2EStep(
                    name=f"observe:{name}",
                    status="error",
                    elapsed_ms=_elapsed_ms(started),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        payload = self.projector.bundle_dict(observation)
        summary = _observation_summary(observation, payload)
        self.observations.append(summary)
        self.steps.append(
            E2EStep(name=f"observe:{name}", status="ok", elapsed_ms=_elapsed_ms(started), details=summary)
        )
        return observation

    async def live_probe(
        self,
        name: str,
        *,
        expected_url_contains: str = "",
        expected_snapshot_contains: str = "",
        expected_tab_contains: tuple[str, ...] = (),
    ) -> None:
        """Verify evaluator client-browser wrappers see the same live session."""

        started = time.perf_counter()
        dispatcher = WebBrowserLiveProbeDispatcher(registered_session=self.registered)
        spec = build_web_round_evaluator_spec(enable_live_probe=True)
        tools = {tool.name: tool for tool in spec.tools}
        checks: dict[str, object] = {
            "name": name,
            "expected_url_contains": expected_url_contains,
            "expected_snapshot_contains": expected_snapshot_contains,
            "expected_tab_contains": list(expected_tab_contains),
        }
        try:
            backend_url_result = await self.client.get_url()
            backend_url = ""
            if backend_url_result.success and isinstance(backend_url_result.data, dict):
                backend_url = str(backend_url_result.data.get("url") or "")
            url_probe = await dispatcher.dispatch(tool=tools["client_browser_get_url"], parameters={}, remote=False)
            snapshot_probe = await dispatcher.dispatch(
                tool=tools["client_browser_snapshot"], parameters={}, remote=False
            )
            tabs_probe = await dispatcher.dispatch(tool=tools["client_browser_tabs"], parameters={}, remote=False)
            screenshot_probe = await dispatcher.dispatch(
                tool=tools["client_browser_screenshot"],
                parameters={"annotate": True},
                remote=False,
            )
            probe_url = _nested_result(url_probe).get("url")
            snapshot_text = _json_text(snapshot_probe)
            tab_labels = _tab_labels(_nested_result(tabs_probe))
            checks.update(
                {
                    "backend_url_host": _safe_host(backend_url),
                    "probe_url_host": _safe_host(str(probe_url or "")),
                    "same_url": bool(backend_url) and probe_url == backend_url,
                    "snapshot_status": snapshot_probe.get("status"),
                    "snapshot_contains_expected": (
                        expected_snapshot_contains in snapshot_text if expected_snapshot_contains else None
                    ),
                    "tabs_status": tabs_probe.get("status"),
                    "tab_labels": tab_labels,
                    "tabs_contain_expected": (
                        all(expected in tab_labels for expected in expected_tab_contains)
                        if expected_tab_contains
                        else None
                    ),
                    "screenshot_status": screenshot_probe.get("status"),
                    "screenshot_has_som": bool(
                        _nested_result(screenshot_probe).get("annotation", {}).get("marks")
                        if isinstance(_nested_result(screenshot_probe).get("annotation"), dict)
                        else False
                    ),
                    "path_free_payload": _path_free(
                        {
                            "url": url_probe,
                            "snapshot": snapshot_probe,
                            "tabs": tabs_probe,
                            "screenshot": screenshot_probe,
                        }
                    ),
                    "no_data_url_payload": "data:image"
                    not in json.dumps(
                        {
                            "url": url_probe,
                            "snapshot": snapshot_probe,
                            "tabs": tabs_probe,
                            "screenshot": screenshot_probe,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            if expected_url_contains and expected_url_contains not in backend_url:
                raise AssertionError(f"expected {expected_url_contains!r} in backend URL {backend_url!r}")
            if not checks["same_url"]:
                raise AssertionError(f"live probe saw {probe_url!r}, backend saw {backend_url!r}")
            if snapshot_probe.get("status") != "success":
                raise AssertionError(f"live probe snapshot failed: {snapshot_probe}")
            if tabs_probe.get("status") != "success":
                raise AssertionError(f"live probe tabs failed: {tabs_probe}")
            if screenshot_probe.get("status") != "success":
                raise AssertionError(f"live probe screenshot failed: {screenshot_probe}")
            if expected_snapshot_contains and not checks["snapshot_contains_expected"]:
                raise AssertionError(
                    f"live probe snapshot did not contain expected state {expected_snapshot_contains!r}"
                )
            if expected_tab_contains and not checks["tabs_contain_expected"]:
                raise AssertionError(f"live probe tabs did not contain expected labels {expected_tab_contains!r}")
            if not checks["path_free_payload"] or not checks["no_data_url_payload"]:
                raise AssertionError("live probe payload leaked path or raw data URL")
            self.live_probe_checks.append({"status": "ok", **checks})
            self.steps.append(
                E2EStep(
                    name=f"live_probe:{name}",
                    status="ok",
                    elapsed_ms=_elapsed_ms(started),
                    details=checks,
                )
            )
        except Exception as exc:
            self.live_probe_checks.append({"status": "error", **checks, "error": f"{type(exc).__name__}: {exc}"})
            self.steps.append(
                E2EStep(
                    name=f"live_probe:{name}",
                    status="error",
                    elapsed_ms=_elapsed_ms(started),
                    details=checks,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _summarize_result(result: object) -> dict[str, object]:
    if hasattr(result, "success") and hasattr(result, "data"):
        data = getattr(result, "data")
        summary: dict[str, object] = {
            "success": bool(getattr(result, "success")),
            "data_keys": sorted(str(key) for key in data) if isinstance(data, dict) else [],
            "error": str(getattr(result, "error", ""))[:240],
        }
        for field_name in ("code", "reason_code", "hint", "concise_log_summary", "feedback_source"):
            value = getattr(result, field_name, None)
            if value:
                summary[field_name] = str(value)
        return summary
    return {"type": type(result).__name__}


def _observation_summary(observation: WebObservationContext, payload: dict[str, object]) -> dict[str, object]:
    screenshot_payload = payload.get("screenshot") if isinstance(payload.get("screenshot"), dict) else {}
    annotation = screenshot_payload.get("annotation") if isinstance(screenshot_payload, dict) else {}
    marks = annotation.get("marks") if isinstance(annotation, dict) else []
    screenshot = observation.screenshot
    return {
        "observation_id": observation.observation_id,
        "settled": observation.settled,
        "settle_reason": observation.settle_reason,
        "title": observation.page_title,
        "url_host": _safe_host(observation.page_url),
        "ref_count": len(observation.snapshot.elements),
        "snapshot_chars": len(observation.snapshot.snapshot),
        "screenshot_kind": screenshot_payload.get("kind") if isinstance(screenshot_payload, dict) else "",
        "screenshot_width": screenshot.width if screenshot else None,
        "screenshot_height": screenshot.height if screenshot else None,
        "som_mark_count": len(marks) if isinstance(marks, list) else 0,
        "som_truncated": bool(annotation.get("truncated")) if isinstance(annotation, dict) else False,
        "som_omitted_count": int(annotation.get("omitted_count") or 0) if isinstance(annotation, dict) else 0,
        "path_free_payload": _path_free(payload),
        "no_data_url_payload": "data:image" not in json.dumps(payload, ensure_ascii=False),
    }


def _safe_host(url: str) -> str:
    if "://" not in url:
        return url
    return url.split("://", 1)[1].split("/", 1)[0]


def _nested_result(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    result = payload.get("result")
    if not isinstance(result, dict):
        return {}
    nested = result.get("result")
    return nested if isinstance(nested, dict) else {}


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _tab_labels(tabs_payload: dict[str, object]) -> list[str]:
    tabs = tabs_payload.get("tabs")
    if not isinstance(tabs, list):
        return []
    labels: list[str] = []
    for tab in tabs:
        if isinstance(tab, dict):
            labels.append(str(tab.get("label") or ""))
    return labels


def _path_free(payload: object) -> bool:
    encoded = json.dumps(payload, ensure_ascii=False)
    forbidden = ("data:image", ".png", "/screenshots/", "\\screenshots\\")
    return not any(token in encoded for token in forbidden)


def _find_ref(elements: list[WebElementRef], *needles: str, role: str = "", tag: str = "") -> str:
    lowered_needles = [needle.lower() for needle in needles if needle]
    for element in elements:
        if role and element.role.lower() != role.lower():
            continue
        if tag and element.tag.lower() != tag.lower():
            continue
        haystack = f"{element.role} {element.name} {element.text} {element.tag}".lower()
        if all(needle in haystack for needle in lowered_needles):
            return element.model_ref
    samples = [
        {
            "ref": element.model_ref,
            "role": element.role,
            "name": element.name[:80],
            "text": element.text[:80],
            "tag": element.tag,
        }
        for element in elements[:16]
    ]
    raise RuntimeError(f"ref not found for needles={needles!r}, role={role!r}, tag={tag!r}; samples={samples!r}")


async def _hacker_news_case(ctx: E2ERunContext) -> None:
    await ctx.record("open", lambda: ctx.client.open(ctx.target_url))
    obs = await ctx.observe("hn-initial")
    assert obs.snapshot.elements, "HN snapshot did not expose refs"
    ref = _find_ref(obs.snapshot.elements, "new", role="link")
    await ctx.record("click_ref:new", lambda: ctx.client.click_ref(ref))
    await ctx.record(
        "wait_load", lambda: ctx.client.wait(WebWaitParams(load_state="domcontentloaded"), timeout_seconds=8)
    )
    after = await ctx.observe("hn-after-click")
    assert "news.ycombinator.com" in after.page_url
    await ctx.record("scroll", lambda: ctx.client.scroll(WebScrollDirection.DOWN, 500))
    scrolled = await ctx.observe("hn-after-scroll")
    assert scrolled.screenshot and scrolled.screenshot.annotation
    await ctx.live_probe("hn-after-scroll", expected_url_contains="news.ycombinator.com")


async def _python_docs_case(ctx: E2ERunContext) -> None:
    await ctx.record("open", lambda: ctx.client.open(ctx.target_url))
    obs = await ctx.observe("python-docs-initial")
    ref = _find_ref(obs.snapshot.elements, "library", role="link")
    box = await ctx.record("get_box:library", lambda: ctx.client.get_box(ref))
    assert getattr(box, "success", False), "library ref box lookup failed"
    await ctx.record("click_ref:library", lambda: ctx.client.click_ref(ref))
    await ctx.record(
        "wait_text:library",
        lambda: ctx.client.wait(WebWaitParams(text="The Python Standard Library"), timeout_seconds=10),
    )
    after = await ctx.observe("python-docs-after-library")
    assert "docs.python.org" in after.page_url
    await ctx.record("screenshot_viewport", lambda: ctx.client.screenshot(annotate=False))
    await ctx.live_probe("python-docs-after-library", expected_url_contains="docs.python.org")


async def _wikipedia_case(ctx: E2ERunContext) -> None:
    await ctx.record("open", lambda: ctx.client.open(ctx.target_url))
    obs = await ctx.observe("wikipedia-initial")
    ref = _find_ref(obs.snapshot.elements, "english")
    await ctx.record("click_ref:english", lambda: ctx.client.click_ref(ref))
    await ctx.record(
        "wait_url:enwiki",
        lambda: ctx.client.wait(WebWaitParams(url_pattern="**en.wikipedia.org/**"), timeout_seconds=10),
    )
    after = await ctx.observe("wikipedia-after-english")
    assert "wikipedia.org" in after.page_url
    await ctx.live_probe("wikipedia-after-english", expected_url_contains="wikipedia.org")


async def _tab_workflow_case(ctx: E2ERunContext) -> None:
    await ctx.record("open", lambda: ctx.client.open(ctx.target_url))
    await ctx.observe("tabs-first")
    await ctx.record(
        "new_tab:iana", lambda: ctx.client.new_tab(url="https://www.iana.org/domains/reserved", label="iana")
    )
    await ctx.observe("tabs-second")
    await ctx.record("list_tabs", ctx.client.list_tabs)
    await ctx.record("focus_tab:t1", lambda: ctx.client.focus_tab("t1"))
    await ctx.observe("tabs-refocused")
    await ctx.record("close_tab:iana", lambda: ctx.client.close_tab("iana"))
    await ctx.record("list_tabs:final", ctx.client.list_tabs)
    await ctx.live_probe("tabs-refocused", expected_url_contains="example.com")


async def _local_info_case(ctx: E2ERunContext) -> None:
    await ctx.record("open", lambda: ctx.client.open(ctx.target_url))
    obs = await ctx.observe("local-info-initial")
    ref = _find_ref(obs.snapshot.elements, "renewal", role="link")
    await ctx.record("click_ref:renewal", lambda: ctx.client.click_ref(ref))
    await ctx.record(
        "wait_url:license", lambda: ctx.client.wait(WebWaitParams(url_pattern="**/license.html"), timeout_seconds=5)
    )
    after = await ctx.observe("local-info-after-renewal")
    assert after.page_url.endswith("/license.html"), f"unexpected info URL: {after.page_url}"
    assert after.page_title == "Renewal Details", f"unexpected info title: {after.page_title}"
    await ctx.live_probe(
        "local-info-after-renewal",
        expected_url_contains="/license.html",
    )


async def _local_form_case(ctx: E2ERunContext) -> None:
    await ctx.record("open", lambda: ctx.client.open(ctx.target_url))
    obs = await ctx.observe("local-form-initial")
    name_ref = _find_ref(obs.snapshot.elements, "full", "name", role="textbox", tag="input")
    tier_ref = _find_ref(obs.snapshot.elements, "plan", role="combobox", tag="select")
    terms_ref = _find_ref(obs.snapshot.elements, "agree", role="checkbox", tag="input")
    submit_ref = _find_ref(obs.snapshot.elements, "submit", role="button")
    await ctx.record("fill_ref:name", lambda: ctx.client.fill_ref(name_ref, "Ada Lovelace"))
    await ctx.record("select_option:tier", lambda: ctx.client.select_option_ref(tier_ref, ["enterprise"]))
    await ctx.record("check_ref:terms", lambda: ctx.client.check_ref(terms_ref))
    await ctx.record("click_ref:submit", lambda: ctx.client.click_ref(submit_ref))
    await ctx.record(
        "wait_text:confirmation", lambda: ctx.client.wait(WebWaitParams(text="WEB-FORM-314"), timeout_seconds=5)
    )
    await ctx.observe("local-form-after-submit")
    await ctx.live_probe(
        "local-form-after-submit",
        expected_url_contains="/form/index.html",
        expected_snapshot_contains="Ada Lovelace",
    )


async def _local_tabs_case(ctx: E2ERunContext) -> None:
    await ctx.record("open", lambda: ctx.client.open(ctx.target_url))
    obs = await ctx.observe("local-tabs-initial")
    _find_ref(obs.snapshot.elements, "details", role="link")
    await ctx.record(
        "new_tab:details",
        lambda: ctx.client.new_tab(url=ctx.target_url.rsplit("/", 1)[0] + "/details.html", label="details"),
    )
    await ctx.record("wait_text:launch", lambda: ctx.client.wait(WebWaitParams(text="09:30 UTC"), timeout_seconds=5))
    await ctx.observe("local-tabs-details")
    await ctx.record("list_tabs:with-details", ctx.client.list_tabs)
    await ctx.record("focus_tab:first", lambda: ctx.client.focus_tab("t1"))
    await ctx.observe("local-tabs-refocused-initial")
    await ctx.record("focus_tab:details", lambda: ctx.client.focus_tab("details"))
    await ctx.record("list_tabs:final", ctx.client.list_tabs)
    await ctx.live_probe(
        "local-tabs-final",
        expected_url_contains="/details.html",
        expected_tab_contains=("details",),
    )


CASES: tuple[E2ECase, ...] = (
    E2ECase(name="hacker_news_navigation_scroll_som", url="https://news.ycombinator.com/", runner=_hacker_news_case),
    E2ECase(name="python_docs_ref_bbox_click", url="https://docs.python.org/3/", runner=_python_docs_case),
    E2ECase(name="wikipedia_public_navigation", url="https://www.wikipedia.org/", runner=_wikipedia_case),
    E2ECase(name="tab_workflow_public_pages", url="https://example.com/", runner=_tab_workflow_case),
)


async def _local_actionability_case(ctx: E2ERunContext) -> None:
    await ctx.record("open", lambda: ctx.client.open(ctx.target_url))
    obs = await ctx.observe("local-actionability-initial")
    ok_ref = _find_ref(obs.snapshot.elements, "ok", role="button")
    disabled_ref = _find_ref(obs.snapshot.elements, "disabled", role="button")
    overlay_ref = _find_ref(obs.snapshot.elements, "notifications", role="button")
    await ctx.record("click_ref:ok", lambda: ctx.client.click_ref(ok_ref))
    await ctx.record_expected_error(
        "click_ref:disabled",
        lambda: ctx.client.click_ref(disabled_ref),
        expected_code="element_disabled",
        expected_hint_contains="disabled",
    )
    await ctx.record_expected_error(
        "fill_ref:button",
        lambda: ctx.client.fill_ref(ok_ref, "nope"),
        expected_code="wrong_action_for_element",
        expected_hint_contains="matches the element role",
    )
    await ctx.record_expected_error(
        "click_ref:pass_through_overlay",
        lambda: ctx.client.click_ref(overlay_ref),
        expected_code="element_not_receiving_events",
        expected_hint_contains="covered",
    )


def _local_fixture_cases(base_url: str) -> tuple[E2ECase, ...]:
    return (
        E2ECase(name="local_info_lookup_live_probe", url=f"{base_url}/info/index.html", runner=_local_info_case),
        E2ECase(name="local_form_state_live_probe", url=f"{base_url}/form/index.html", runner=_local_form_case),
        E2ECase(name="local_tabs_navigation_live_probe", url=f"{base_url}/tabs/index.html", runner=_local_tabs_case),
        E2ECase(
            name="local_actionability_feedback_negative",
            url=f"{base_url}/actionability/index.html",
            runner=_local_actionability_case,
        ),
    )


class _QuietFixtureRequestHandler(http.server.SimpleHTTPRequestHandler):
    _ROOT_REWRITES = {
        "/license.html": "/info/license.html",
        "/details.html": "/tabs/details.html",
    }

    def translate_path(self, path: str) -> str:
        return super().translate_path(self._ROOT_REWRITES.get(path.split("?", 1)[0], path))

    def log_message(self, format: str, *args: object) -> None:
        return


class LocalFixtureServer:
    """Serve deterministic Web E2E fixtures from the repository tree."""

    def __init__(self, fixture_root: Path) -> None:
        self.fixture_root = fixture_root
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("fixture server has not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "LocalFixtureServer":
        if not self.fixture_root.exists():
            raise FileNotFoundError(f"fixture root does not exist: {self.fixture_root}")
        handler = functools.partial(_QuietFixtureRequestHandler, directory=str(self.fixture_root))
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="web-e2e-fixtures", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


async def _run_case(
    case: E2ECase,
    *,
    root_dir: Path,
    headed: bool,
    browser_channel: str | None = None,
    browser_executable_path: str | None = None,
) -> dict[str, object]:
    session_id = f"web-e2e-{case.name}-{time.time_ns()}"
    config = WebAgentConfig(
        headed=headed,
        browser_channel=browser_channel,
        browser_executable_path=browser_executable_path,
        command_timeout_seconds=45,
        screenshot_timeout_seconds=6,
        snapshot_depth=8,
        post_action_wait_ms=0,
        allow_private_network_urls=True,
    )
    store = WebArtifactStore.for_session(root_dir, session_id, ArtifactConfig())
    backend: WebActionBackend = PlaywrightWebActionBackend(config=config, session_id=session_id)
    registered = RegisteredWebAgentSession(
        session_id=session_id,
        client=backend,
        artifact_store=store,
        settings=config,
    )
    service = WebAgentMCPActionService()
    ctx = E2ERunContext(registered=registered, service=service, target_url=case.url)
    started = time.perf_counter()
    status = "ok"
    error = ""
    try:
        await case.runner(ctx)
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        close_started = time.perf_counter()
        close_result = await backend.close()
        if not close_result.success:
            status = "error"
            error = close_result.error or "Browser cleanup failed"
        ctx.steps.append(
            E2EStep(
                name="close_session",
                status="ok" if close_result.success else "error",
                elapsed_ms=_elapsed_ms(close_started),
                details=_summarize_result(close_result),
                error=close_result.error,
            )
        )
    case_report = {
        "case": case.name,
        "target_host": _safe_host(case.url),
        "status": status,
        "error": error,
        "elapsed_ms": _elapsed_ms(started),
        "artifact_session": store.session_id,
        "steps": [asdict(step) for step in ctx.steps],
        "observations": ctx.observations,
        "live_probe_checks": ctx.live_probe_checks,
        "live_probe_success_count": sum(1 for check in ctx.live_probe_checks if check.get("status") == "ok"),
        "live_probe_failure_count": sum(1 for check in ctx.live_probe_checks if check.get("status") != "ok"),
    }
    trace_path = store.root / "e2e_trace_summary.jsonl"
    for step in ctx.steps:
        store.append_jsonl(
            trace_path,
            {
                "case": case.name,
                "step": step.name,
                "status": step.status,
                "elapsed_ms": step.elapsed_ms,
                "details": step.details,
                "error": step.error,
            },
        )
    case_report["trace_summary_file"] = trace_path.name
    return case_report


def _redaction_scan(report: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(report, ensure_ascii=False)
    findings = []
    for token in ("data:image", "/screenshots/", "\\screenshots\\", "/Users/", ".png"):
        if token in encoded:
            findings.append(token)
    return {"ok": not findings, "findings": findings}


async def run_e2e(
    *,
    headed: bool,
    browser_channel: str | None = None,
    browser_executable_path: str | None = None,
    output: Path,
    artifact_root: Path,
    local_fixtures: bool = False,
    fixture_root: Path | None = None,
) -> dict[str, object]:
    fixture_root = fixture_root or Path("examples/fixtures/web-e2b-three")
    cases = []
    fixture_server = LocalFixtureServer(fixture_root) if local_fixtures else contextlib.nullcontext(None)
    with fixture_server as server:
        selected_cases = _local_fixture_cases(server.base_url) if isinstance(server, LocalFixtureServer) else CASES
        for case in selected_cases:
            cases.append(
                await _run_case(
                    case,
                    root_dir=artifact_root,
                    headed=headed,
                    browser_channel=browser_channel,
                    browser_executable_path=browser_executable_path,
                )
            )
    live_probe_success_count = sum(int(case.get("live_probe_success_count") or 0) for case in cases)
    live_probe_failure_count = sum(int(case.get("live_probe_failure_count") or 0) for case in cases)
    report: dict[str, object] = {
        "experiment": "web_agent_playwright_real_e2e",
        "headed": headed,
        "fixture_mode": "local" if local_fixtures else "public",
        "case_count": len(cases),
        "success_count": sum(1 for case in cases if case["status"] == "ok"),
        "failure_count": sum(1 for case in cases if case["status"] != "ok"),
        "live_probe_success_count": live_probe_success_count,
        "live_probe_failure_count": live_probe_failure_count,
        "cases": cases,
    }
    report["redaction_scan"] = _redaction_scan(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headed", action="store_true", help="Run visible browser windows.")
    parser.add_argument(
        "--local-fixtures",
        action="store_true",
        help="Serve deterministic repo fixtures locally instead of visiting public websites.",
    )
    parser.add_argument(
        "--fixture-root",
        default="examples/fixtures/web-e2b-three",
        help="Fixture root used with --local-fixtures.",
    )
    parser.add_argument("--browser-executable-path", default=None)
    parser.add_argument("--browser-channel", choices=["chrome", "msedge"], default=None)
    parser.add_argument("--output", default="logs/web_agent_playwright_e2e_latest.json")
    parser.add_argument("--artifact-root", default="logs/web_agent_playwright_e2e_artifacts")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = asyncio.run(
        run_e2e(
            headed=bool(args.headed),
            browser_channel=args.browser_channel,
            browser_executable_path=args.browser_executable_path,
            output=Path(args.output),
            artifact_root=Path(args.artifact_root),
            local_fixtures=bool(args.local_fixtures),
            fixture_root=Path(args.fixture_root),
        )
    )
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2))
    if args.fail_on_error and (
        report["failure_count"] or report["live_probe_failure_count"] or not report["redaction_scan"]["ok"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
