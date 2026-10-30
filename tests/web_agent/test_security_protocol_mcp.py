from __future__ import annotations

import asyncio
import base64
import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image

from northcua.agentinfra.runtime.artifacts import ArtifactStoreFactory, sanitize_session_id
from northcua.agents.web.artifacts import WebArtifactStore
from northcua.agents.web.backend import WebActionCommandError, WebActionCommandResult
from northcua.agents.web.domain.key_names import canonical_web_key_name
from northcua.agents.web.domain.observations import WebElementRef, WebScreenshotObservation, WebSnapshotObservation
from northcua.agents.web.mcp.action_service import WebAgentMCPActionService
from northcua.agents.web.mcp.registration import ensure_web_agent_mcp_servers_registered
from northcua.agents.web.observation.ref_alias import WebObservationRefAliases, WebRefAlias
from northcua.agents.web.observation.som import InteractionFirstSomSelector, WebSomSelectionConfig
from northcua.agents.web.security import WebNavigationPolicy
from northcua.agents.web.server import WebAgentWebSocketServer
from northcua.agents.web.session import RegisteredWebAgentSession, WebAgentSessionRegistry
from northcua.core.config import ArtifactConfig, WebAgentConfig, WebPostActionSettleConfig
from northcua.core.enums import WebSocketMessageType
from northcua.core.protocol import NorthCUAWebSocketEnvelope, WebTaskRequestPayload
from northcua.runtime.artifacts.store import NorthCUAArtifactStore
from northcua.runtime.mcp.invocation_context import MCPInvocationContext, use_mcp_invocation_context
from northcua.runtime.mcp.registry import MCPFactoryContext, NorthCUAMCPRegistry
from northcua.runtime.mcp.result_adapter import normalize_tool_result


def test_web_press_key_name_canonicalization() -> None:
    assert canonical_web_key_name("Left") == "ArrowLeft"
    assert canonical_web_key_name("right") == "ArrowRight"
    assert canonical_web_key_name("Return") == "Enter"
    assert canonical_web_key_name("Esc") == "Escape"
    assert canonical_web_key_name("ctrl+left") == "Control+ArrowLeft"
    assert canonical_web_key_name("Cmd+A") == "Meta+A"


class _FakeMCPClient:
    def __init__(self, *, screenshot_path: str = "") -> None:
        self.opened: list[str] = []
        self.waits = 0
        self.snapshot_calls = 0
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.box_batch_calls: list[list[str]] = []
        self.screenshot_path = screenshot_path

    async def open(self, url: str) -> WebActionCommandResult:
        self.opened.append(url)
        return WebActionCommandResult(success=True, data={"url": url})

    async def wait(self, params, *, timeout_seconds: float | None = None) -> WebActionCommandResult:
        del timeout_seconds
        self.waits += 1
        return WebActionCommandResult(success=True, data={"waited": True})

    async def snapshot(self) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        return WebSnapshotObservation(
            url="https://example.com",
            snapshot=(
                '- button "Submit" [ref=@e1]\n'
                '- textbox "Email" [ref=@e2]\n'
                '- button "Hidden" [ref=@e3]\n'
                '- button "Disabled" [ref=@e4]'
            ),
            elements=[
                WebElementRef(ref_id="e1", role="button", name="Submit"),
                WebElementRef(ref_id="e2", role="textbox", name="Email"),
                WebElementRef(ref_id="e3", role="button", name="Hidden", attributes={"aria-hidden": "true"}),
                WebElementRef(ref_id="e4", role="button", name="Disabled", attributes={"disabled": ""}),
            ],
        )

    async def click_ref(self, ref: str) -> WebActionCommandResult:
        if ref == "@stale":
            raise WebActionCommandError("stale ref")
        self.calls.append(("click_ref", {"ref": ref}))
        return WebActionCommandResult(success=True, data={"clicked": ref})

    async def hover_ref(self, ref: str) -> WebActionCommandResult:
        self.calls.append(("hover_ref", {"ref": ref}))
        return WebActionCommandResult(success=True, data={"hovered": ref})

    async def fill_ref(self, ref: str, text: str) -> WebActionCommandResult:
        self.calls.append(("fill_ref", {"ref": ref, "text": text}))
        return WebActionCommandResult(success=True, data={"filled": ref})

    async def select_option_ref(self, ref: str, values: list[str]) -> WebActionCommandResult:
        self.calls.append(("select_option_ref", {"ref": ref, "values": values}))
        return WebActionCommandResult(success=True, data={"selected": values})

    async def check_ref(self, ref: str) -> WebActionCommandResult:
        self.calls.append(("check_ref", {"ref": ref}))
        return WebActionCommandResult(success=True, data={"checked": ref})

    async def uncheck_ref(self, ref: str) -> WebActionCommandResult:
        self.calls.append(("uncheck_ref", {"ref": ref}))
        return WebActionCommandResult(success=True, data={"unchecked": ref})

    async def drag_ref(self, from_ref: str, to_ref: str) -> WebActionCommandResult:
        self.calls.append(("drag_ref", {"from_ref": from_ref, "to_ref": to_ref}))
        return WebActionCommandResult(success=True, data={"dragged": True})

    async def click_point(self, x: float, y: float, button) -> WebActionCommandResult:
        self.calls.append(("click_point", {"x": x, "y": y, "button": button.value}))
        return WebActionCommandResult(success=True, data={"clicked": [x, y]})

    async def double_click_point(self, x: float, y: float, button) -> WebActionCommandResult:
        self.calls.append(("double_click_point", {"x": x, "y": y, "button": button.value}))
        return WebActionCommandResult(success=True, data={"double_clicked": [x, y]})

    async def drag_point(self, from_x: float, from_y: float, to_x: float, to_y: float) -> WebActionCommandResult:
        self.calls.append(("drag_point", {"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y}))
        return WebActionCommandResult(success=True, data={"dragged": True})

    async def type_text(self, text: str) -> WebActionCommandResult:
        self.calls.append(("type_text", {"text": text}))
        return WebActionCommandResult(success=True, data={"typed": True})

    async def press(self, key: str) -> WebActionCommandResult:
        self.calls.append(("press", {"key": key}))
        return WebActionCommandResult(success=True, data={"pressed": key})

    async def scroll(self, direction, amount: int) -> WebActionCommandResult:
        self.calls.append(("scroll", {"direction": direction.value, "amount": amount}))
        return WebActionCommandResult(success=True, data={"scrolled": direction.value, "amount": amount})

    async def get_text(self, ref: str) -> WebActionCommandResult:
        self.calls.append(("get_text", {"ref": ref}))
        return WebActionCommandResult(success=True, data={"text": f"Text for {ref}"})

    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        buffer = io.BytesIO()
        Image.new("RGB", (640, 360), "white").save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return WebScreenshotObservation(path=self.screenshot_path, data_url=f"data:image/png;base64,{data}")

    async def get_box(self, ref: str) -> WebActionCommandResult:
        boxes = {
            "@e1": {"x": 48, "y": 40, "width": 120, "height": 34},
            "@e2": {"x": 48, "y": 96, "width": 220, "height": 30},
            "@e3": {"x": 48, "y": 140, "width": 120, "height": 34},
            "@e4": {"x": 48, "y": 184, "width": 120, "height": 34},
        }
        return WebActionCommandResult(success=True, data=boxes.get(ref, {"x": 0, "y": 0, "width": 1, "height": 1}))

    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        self.box_batch_calls.append(list(refs))
        boxes = {
            "@e1": {"x": 48, "y": 40, "width": 120, "height": 34},
            "@e2": {"x": 48, "y": 96, "width": 220, "height": 30},
            "@e3": {"x": 48, "y": 140, "width": 120, "height": 34},
            "@e4": {"x": 48, "y": 184, "width": 120, "height": 34},
        }
        return WebActionCommandResult(
            success=True,
            data={"boxes": {ref: boxes[ref] for ref in refs if ref in boxes}, "errors": {}},
        )

    async def get_ref_states(self, refs: list[str]) -> WebActionCommandResult:
        states = {
            ref: {
                "visible": ref != "@e3",
                "enabled": ref != "@e4",
            }
            for ref in refs
        }
        return WebActionCommandResult(success=True, data={"states": states, "errors": {}})

    async def get_viewport_size(self) -> WebActionCommandResult:
        return WebActionCommandResult(
            success=True,
            data={"result": {"width": 640, "height": 360, "devicePixelRatio": 1}},
        )

    async def get_url(self) -> WebActionCommandResult:
        url = self.opened[-1] if self.opened else "https://example.com"
        return WebActionCommandResult(success=True, data={"url": url})

    async def list_tabs(self) -> WebActionCommandResult:
        return WebActionCommandResult(success=True, data={"tabs": [{"id": "t1", "url": "https://example.com"}]})

    async def new_tab(self, *, url: str | None = None, label: str | None = None) -> WebActionCommandResult:
        self.calls.append(("new_tab", {"url": url or "", "label": label or ""}))
        return WebActionCommandResult(success=True, data={"id": "t2", "url": url, "label": label})

    async def close_tab(self, tab_ref: str | None = None) -> WebActionCommandResult:
        if tab_ref == "ghost":
            return WebActionCommandResult(
                success=False,
                data={"status": "error", "error": "No tab with label ghost"},
                error="No tab with label ghost",
            )
        self.calls.append(("close_tab", {"tab_ref": tab_ref or ""}))
        return WebActionCommandResult(success=True, data={"closed": tab_ref})

    async def focus_tab(self, tab_ref: str) -> WebActionCommandResult:
        if tab_ref == "ghost":
            return WebActionCommandResult(
                success=False,
                data={"status": "error", "error": "No tab with label ghost"},
                error="No tab with label ghost",
            )
        self.calls.append(("focus_tab", {"tab_ref": tab_ref}))
        return WebActionCommandResult(success=True, data={"focused": tab_ref})


class _PostObservationFailClient(_FakeMCPClient):
    async def snapshot(self) -> WebSnapshotObservation:
        raise WebActionCommandError("snapshot backend unavailable")


class _PostObservationTimeoutClient(_FakeMCPClient):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_cancelled = False

    async def snapshot(self) -> WebSnapshotObservation:
        try:
            await asyncio.sleep(3600)
        finally:
            self.snapshot_cancelled = True
        raise AssertionError("unreachable")


def test_web_task_payload_round_trips_through_websocket_envelope() -> None:
    envelope = NorthCUAWebSocketEnvelope.wrap(
        WebSocketMessageType.TASK_REQUEST,
        WebTaskRequestPayload(session_id="sess", request="Find pricing", url="https://example.com"),
    )

    parsed = NorthCUAWebSocketEnvelope.model_validate_json(envelope.model_dump_json())

    assert isinstance(parsed.payload, WebTaskRequestPayload)
    assert parsed.payload.agent_kind.value == "web"
    assert parsed.payload.url == "https://example.com"


def test_session_ids_are_sanitized_for_artifact_paths(tmp_path) -> None:
    store = ArtifactStoreFactory.create_session_store(
        WebArtifactStore,
        root_dir=tmp_path,
        session_id="../outside",
        config=ArtifactConfig(),
    )

    expected = f"~outside-{hashlib.sha256(b'../outside').hexdigest()}"
    assert sanitize_session_id("../outside") == expected
    assert store.root.parent == tmp_path
    assert store.root.name == expected
    assert store.storage_key == expected
    assert store.session_id == "../outside"


def test_sanitized_session_artifact_namespaces_do_not_collide(tmp_path) -> None:
    assert sanitize_session_id("ordinary-session.1") == "ordinary-session.1"

    first = ArtifactStoreFactory.create_session_store(
        WebArtifactStore,
        root_dir=tmp_path,
        session_id="tenant/a",
        config=ArtifactConfig(),
    )
    second = ArtifactStoreFactory.create_session_store(
        WebArtifactStore,
        root_dir=tmp_path,
        session_id="tenant:a",
        config=ArtifactConfig(),
    )
    long_first = sanitize_session_id("x" * 128 + "a")
    long_second = sanitize_session_id("x" * 128 + "b")

    assert first.root != second.root
    assert first.context_summary_llm_trace_log_path != second.context_summary_llm_trace_log_path
    resumed = ArtifactStoreFactory.create_session_store(
        WebArtifactStore,
        root_dir=tmp_path,
        session_id="tenant/a",
        config=ArtifactConfig(),
    )
    transformed = sanitize_session_id("tenant/a")
    assert sanitize_session_id(transformed) != transformed
    assert first.root == resumed.root
    assert first.session_id == resumed.session_id == "tenant/a"
    assert first.storage_key == resumed.storage_key == transformed
    desktop = ArtifactStoreFactory.create_session_store(
        NorthCUAArtifactStore,
        root_dir=tmp_path / "desktop",
        session_id="tenant/a",
        config=ArtifactConfig(),
    )
    assert desktop.session_id == "tenant/a"
    assert desktop.storage_key == transformed
    assert sanitize_session_id("tenant") != sanitize_session_id(" tenant ")
    assert long_first != long_second
    assert len(long_first) == 128
    assert len(long_second) == 128


def test_web_server_missing_session_fallback_preserves_logical_id(tmp_path) -> None:
    store = ArtifactStoreFactory.create_session_store(
        WebArtifactStore,
        root_dir=tmp_path,
        session_id="tenant/a",
        config=ArtifactConfig(),
    )

    assert WebAgentWebSocketServer._resolve_session_id("", store.session_id) == "tenant/a"
    assert WebAgentWebSocketServer._resolve_session_id("tenant/a", "different") == "tenant/a"


def test_navigation_policy_blocks_sensitive_schemes_and_private_networks(monkeypatch: pytest.MonkeyPatch) -> None:
    def _public_getaddrinfo(
        *args: object, **kwargs: object
    ) -> list[tuple[object, object, object, str, tuple[str, int]]]:
        del args, kwargs
        return [(None, None, None, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("northcua.agents.web.security.socket.getaddrinfo", _public_getaddrinfo)
    policy = WebNavigationPolicy(WebAgentConfig())

    assert policy.validate_url("https://example.com/path") == "https://example.com/path"
    assert policy.validate_url("http://localhost:3000") == "http://localhost:3000"
    with pytest.raises(ValueError):
        policy.validate_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        policy.validate_url("http://169.254.169.254/latest/meta-data")
    with pytest.raises(ValueError):
        policy.validate_url("chrome://settings")


def test_command_payload_projection_maps_source_refs_without_touching_urls(tmp_path) -> None:
    store = ArtifactStoreFactory.create_session_store(
        WebArtifactStore,
        root_dir=tmp_path,
        session_id="projection",
        config=ArtifactConfig(),
    )
    registered = RegisteredWebAgentSession(
        session_id="projection",
        client=_FakeMCPClient(),
        artifact_store=store,
        settings=WebAgentConfig(),
    )
    service = WebAgentMCPActionService()
    observation = service._snapshot_observation_context(
        WebSnapshotObservation(
            url="https://example.com/path?ref=@e1",
            elements=[
                WebElementRef(ref_id="e1", role="button", name="Submit"),
                WebElementRef(ref_id="e2", role="link", name="Docs"),
            ],
        ),
        observation_id="test",
    )
    service.update_ref_aliases_from_observation(registered, observation)

    payload = service.project_command_payload_refs(
        registered,
        {
            "clicked": "@e1",
            "nested": {"refs": ["@e2", "https://example.com/@e1", "ordinary"]},
            "text": "Clicked @e1 in human text.",
        },
    )

    assert payload == {
        "clicked": "1",
        "nested": {"refs": ["2", "https://example.com/@e1", "ordinary"]},
        "text": "Clicked @e1 in human text.",
    }


def test_mcp_error_payload_preserves_actionability_feedback() -> None:
    from northcua.agents.web.backend.errors import WebActionCommandError
    from northcua.agents.web.domain.actions import WebAgentActionType

    exc = WebActionCommandError(
        "click_ref preflight failed for @e3: element is not enabled",
        code="element_disabled",
        reason_code="element_disabled",
        hint="The target is disabled. Do not retry the same ref.",
        concise_log_summary="element is not enabled",
        feedback_source="preflight",
    )

    payload = WebAgentMCPActionService().error_payload(WebAgentActionType.CLICK_REF, exc)

    assert payload["status"] == "error"
    assert payload["code"] == "element_disabled"
    assert payload["reason_code"] == "element_disabled"
    assert payload["hint"] == "The target is disabled. Do not retry the same ref."
    assert payload["concise_log_summary"] == "element is not enabled"
    assert payload["feedback_source"] == "preflight"


def test_registered_session_resolves_numeric_refs_only_through_latest_aliases(tmp_path) -> None:
    from northcua.agents.web.observation.ref_alias import WebObservationRefAliases, WebRefAlias

    store = ArtifactStoreFactory.create_session_store(
        WebArtifactStore,
        root_dir=tmp_path,
        session_id="alias-resolution",
        config=ArtifactConfig(),
    )
    registered = RegisteredWebAgentSession(
        session_id="alias-resolution",
        client=_FakeMCPClient(),
        artifact_store=store,
        settings=WebAgentConfig(),
    )
    registered.update_ref_aliases(
        WebObservationRefAliases(
            observation_id="obs-components",
            aliases=[WebRefAlias(ref="26", source_ref="@e37")],
        )
    )

    assert registered.resolve_model_ref("26") == "@e37"
    assert registered.resolve_model_ref(26) == "@e37"
    with pytest.raises(ValueError, match="internal-only"):
        registered.resolve_model_ref("@e37")
    with pytest.raises(ValueError, match="internal-only"):
        registered.resolve_model_ref("e37")
    with pytest.raises(ValueError, match="Unknown or stale web ref"):
        registered.resolve_model_ref("99")

    registered.update_ref_aliases(WebObservationRefAliases())
    with pytest.raises(ValueError, match="Unknown or stale web ref"):
        registered.resolve_model_ref("26")
    registered.update_ref_aliases(
        WebObservationRefAliases(observation_id="obs-next", aliases=[WebRefAlias(ref="1", source_ref="@e2")])
    )
    registered.clear_ref_aliases(observation_id="post-click-failed")
    with pytest.raises(ValueError, match="Unknown or stale web ref"):
        registered.resolve_model_ref("1")


@pytest.mark.parametrize("address", ["10.0.0.5", "169.254.169.254", "224.0.0.1", "240.0.0.1", "0.0.0.0"])
def test_navigation_policy_blocks_dns_rebinding_to_private_network(
    monkeypatch: pytest.MonkeyPatch, address: str
) -> None:
    def _private_getaddrinfo(
        *args: object, **kwargs: object
    ) -> list[tuple[object, object, object, str, tuple[str, int]]]:
        del args, kwargs
        return [(None, None, None, "", (address, 443))]

    monkeypatch.setattr("northcua.agents.web.security.socket.getaddrinfo", _private_getaddrinfo)

    with pytest.raises(ValueError, match="Private/internal network URLs are disabled"):
        WebNavigationPolicy(WebAgentConfig()).validate_url("https://internal.example.test")


def test_web_agent_mcp_tools_validate_session_url_and_ref_errors(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _public_getaddrinfo(
        *args: object, **kwargs: object
    ) -> list[tuple[object, object, object, str, tuple[str, int]]]:
        del args, kwargs
        return [(None, None, None, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("northcua.agents.web.security.socket.getaddrinfo", _public_getaddrinfo)
    ensure_web_agent_mcp_servers_registered()
    registry = WebAgentSessionRegistry()
    registry.clear()
    client = _FakeMCPClient()
    store = WebArtifactStore.for_session(tmp_path, "mcp-sess", ArtifactConfig())
    registry.register(
        RegisteredWebAgentSession(
            session_id="mcp-sess",
            client=client,
            artifact_store=store,
            settings=WebAgentConfig(post_action_wait_ms=0),
        )
    )
    server = NorthCUAMCPRegistry.build("WebAgentActionExecutor", context=MCPFactoryContext(process_name="test"))

    async def _run() -> None:
        tools = await server.list_tools()
        tool_names = {tool.name for tool in tools}
        assert {
            "web_note",
            "web_open",
            "web_click_ref",
            "web_hover_ref",
            "web_select_option_ref",
            "web_check_ref",
            "web_uncheck_ref",
            "web_drag_ref",
            "web_click_point",
            "web_double_click_point",
            "web_drag_point",
            "web_list_tabs",
            "web_new_tab",
            "web_close_tab",
            "web_focus_tab",
            "web_screenshot",
        }.issubset(tool_names)
        assert "web_done" not in tool_names
        no_context = normalize_tool_result(await server.call_tool("web_open", {"url": "https://example.com"}))
        assert isinstance(no_context, dict)
        assert no_context["status"] == "error"
        note = normalize_tool_result(
            await server.call_tool(
                "web_note",
                {
                    "text": "Ada email is ada.lovelace@northcua.test.",
                    "reason": "Need this value for a later form field.",
                    "kind": "fact",
                },
            )
        )
        assert note == {
            "text": "[fact] Ada email is ada.lovelace@northcua.test.\nReason: Need this value for a later form field.",
            "status": "recorded",
            "action": "note",
            "kind": "fact",
            "note_text": "Ada email is ada.lovelace@northcua.test.",
            "reason": "Need this value for a later form field.",
            "screenshot_policy": "none",
        }
        assert "screenshot_url" not in note
        assert "result" not in note
        assert client.calls == []
        assert client.waits == 0
        with use_mcp_invocation_context(MCPInvocationContext(session_id="mcp-sess")):

            def assert_fresh_refs(payload: dict[str, object]) -> None:
                assert payload["status"] == "ok"
                observation = payload["observation"]
                assert isinstance(observation, dict)
                snapshot = observation["snapshot"]
                assert isinstance(snapshot, dict)
                assert snapshot["ref_count"] == 2
                refs = snapshot["refs"]
                assert isinstance(refs, list)
                assert [ref["ref"] for ref in refs] == ["1", "2"]
                marks = observation["screenshot"]["annotation"]["marks"]
                assert [mark["ref"] for mark in marks] == [ref["ref"] for ref in refs]
                assert [mark["label"] for mark in marks] == [ref["ref"] for ref in refs]
                assert "screenshot_path" not in payload
                assert str(payload["screenshot_url"]).startswith("data:image/png;base64,")

            opened = normalize_tool_result(await server.call_tool("web_open", {"url": "https://example.com"}))
            assert isinstance(opened, dict)
            assert_fresh_refs(opened)
            assert opened["observation"]["settled"] is True
            assert opened["observation"]["screenshot"]["kind"] == "set_of_mark"
            assert opened["observation"]["screenshot"]["image_attached"] is True
            assert "screenshot_path" not in opened
            assert str(opened["screenshot_url"]).startswith("data:image/png;base64,")
            blocked = normalize_tool_result(await server.call_tool("web_open", {"url": "file:///etc/passwd"}))
            assert isinstance(blocked, dict)
            assert blocked["status"] == "error"
            assert blocked["retryable"] is False
            assert blocked["code"] == "policy_or_artifact_error"
            stale = normalize_tool_result(await server.call_tool("web_click_ref", {"ref": "@stale"}))
            assert isinstance(stale, dict)
            assert stale["status"] == "error"
            assert stale["retryable"] is True
            assert stale["code"] == "invalid_request"
            assert "Revise the request" in stale["hint"]
            snapshot = normalize_tool_result(await server.call_tool("web_snapshot", {}))
            assert isinstance(snapshot, dict)
            assert snapshot["status"] == "ok"
            snapshot_result = snapshot["result"]
            assert snapshot_result["ref_count"] == 2
            assert [ref["ref"] for ref in snapshot_result["refs"]] == ["1", "2"]
            assert [ref["label"] for ref in snapshot_result["refs"]] == ["Submit", "Email"]
            registered = registry.get_registered_session("mcp-sess")
            assert registered.resolve_model_ref("1") == "@e1"
            assert registered.resolve_model_ref("2") == "@e2"
            with pytest.raises(ValueError, match="Unknown or stale web ref"):
                registered.resolve_model_ref("3")
            waits_before_invalid_calls = client.waits
            invalid_wait = normalize_tool_result(await server.call_tool("web_wait", {}))
            assert isinstance(invalid_wait, dict)
            assert invalid_wait["status"] == "error"
            assert invalid_wait["code"] == "invalid_wait_target_combination"
            assert invalid_wait["reason_code"] == "invalid_wait_target_combination"
            assert invalid_wait["retryable"] is True
            assert invalid_wait["received_targets"] == []
            assert invalid_wait["feedback_source"] == "contract_validator"
            assert "received no wait target" in invalid_wait["hint"]
            assert client.waits == waits_before_invalid_calls
            conflicting_wait = normalize_tool_result(
                await server.call_tool(
                    "web_wait",
                    {
                        "duration_ms": 1000,
                        "load_state": "domcontentloaded",
                        "text": "Ready",
                    },
                )
            )
            assert isinstance(conflicting_wait, dict)
            assert conflicting_wait["status"] == "error"
            assert conflicting_wait["code"] == "invalid_wait_target_combination"
            assert conflicting_wait["reason_code"] == "invalid_wait_target_combination"
            assert conflicting_wait["retryable"] is True
            assert conflicting_wait["received_targets"] == ["duration_ms", "load_state", "text"]
            assert conflicting_wait["feedback_source"] == "contract_validator"
            assert "duration_ms, load_state, text" in conflicting_wait["hint"]
            assert "No target takes priority" in conflicting_wait["hint"]
            assert client.waits == waits_before_invalid_calls
            hovered = normalize_tool_result(await server.call_tool("web_hover_ref", {"ref": "2"}))
            assert isinstance(hovered, dict)
            assert_fresh_refs(hovered)
            assert hovered["params"] == {"ref": "2"}
            assert hovered["observation"]["page"]["url"] == "https://example.com"
            selected = normalize_tool_result(
                await server.call_tool("web_select_option_ref", {"ref": "1", "values": [" US "]})
            )
            assert isinstance(selected, dict)
            assert_fresh_refs(selected)
            assert selected["params"] == {"ref": "1", "values": [" US "]}
            checked = normalize_tool_result(await server.call_tool("web_check_ref", {"ref": "1"}))
            assert isinstance(checked, dict)
            assert_fresh_refs(checked)
            assert checked["params"] == {"ref": "1"}
            unchecked = normalize_tool_result(await server.call_tool("web_uncheck_ref", {"ref": "1"}))
            assert isinstance(unchecked, dict)
            assert_fresh_refs(unchecked)
            assert unchecked["params"] == {"ref": "1"}
            dragged_ref = normalize_tool_result(
                await server.call_tool("web_drag_ref", {"from_ref": "1", "to_ref": "2"})
            )
            assert isinstance(dragged_ref, dict)
            assert_fresh_refs(dragged_ref)
            assert dragged_ref["params"] == {"from_ref": "1", "to_ref": "2"}
            point = normalize_tool_result(
                await server.call_tool("web_click_point", {"x": 10, "y": 20, "button": "right"})
            )
            assert isinstance(point, dict)
            assert_fresh_refs(point)
            assert point["params"] == {"x": 10.0, "y": 20.0, "button": "right"}
            double_point = normalize_tool_result(await server.call_tool("web_double_click_point", {"x": 11, "y": 21}))
            assert isinstance(double_point, dict)
            assert_fresh_refs(double_point)
            assert double_point["params"] == {"x": 11.0, "y": 21.0, "button": "left"}
            dragged_point = normalize_tool_result(
                await server.call_tool("web_drag_point", {"from_x": 1, "from_y": 2, "to_x": 3, "to_y": 4})
            )
            assert isinstance(dragged_point, dict)
            assert_fresh_refs(dragged_point)
            assert dragged_point["params"] == {"from_x": 1.0, "from_y": 2.0, "to_x": 3.0, "to_y": 4.0}
            typed = normalize_tool_result(await server.call_tool("web_type_text", {"text": "hello"}))
            assert isinstance(typed, dict)
            assert_fresh_refs(typed)
            pressed = normalize_tool_result(await server.call_tool("web_press", {"key": "Left"}))
            assert isinstance(pressed, dict)
            assert_fresh_refs(pressed)
            assert pressed["text"] == "Pressed ArrowLeft."
            assert pressed["params"] == {"key": "ArrowLeft"}
            assert ("press", {"key": "ArrowLeft"}) in client.calls
            scrolled = normalize_tool_result(await server.call_tool("web_scroll", {"direction": "down", "amount": 120}))
            assert isinstance(scrolled, dict)
            assert_fresh_refs(scrolled)
            waited = normalize_tool_result(await server.call_tool("web_wait", {"duration_ms": 1}))
            assert isinstance(waited, dict)
            assert_fresh_refs(waited)
            text = normalize_tool_result(await server.call_tool("web_get_text", {"ref": "1"}))
            assert isinstance(text, dict)
            assert text["status"] == "ok"
            assert "observation" not in text
            tabs = normalize_tool_result(await server.call_tool("web_list_tabs", {}))
            assert isinstance(tabs, dict)
            assert tabs["status"] == "ok"
            assert "observation" not in tabs
            new_tab = normalize_tool_result(
                await server.call_tool("web_new_tab", {"url": "https://example.com/docs", "label": "docs"})
            )
            assert isinstance(new_tab, dict)
            assert_fresh_refs(new_tab)
            assert new_tab["params"] == {"url": "https://example.com/docs", "label": "docs"}
            focused = normalize_tool_result(await server.call_tool("web_focus_tab", {"tab_ref": "docs"}))
            assert isinstance(focused, dict)
            assert_fresh_refs(focused)
            assert focused["params"] == {"tab_ref": "docs"}
            missing_tab = normalize_tool_result(await server.call_tool("web_focus_tab", {"tab_ref": "ghost"}))
            assert isinstance(missing_tab, dict)
            assert missing_tab["status"] == "error"
            assert missing_tab["retryable"] is True
            assert missing_tab["code"] == "tab_not_found"
            closed = normalize_tool_result(await server.call_tool("web_close_tab", {"tab_ref": "t2"}))
            assert isinstance(closed, dict)
            assert_fresh_refs(closed)
            assert closed["params"] == {"tab_ref": "t2"}
            screenshot = normalize_tool_result(await server.call_tool("web_screenshot", {}))
            assert isinstance(screenshot, dict)
            assert screenshot["status"] == "ok"
            assert isinstance(screenshot["screenshot_url"], str)
            assert screenshot["screenshot_url"].startswith("data:image/png;base64,")
            annotated = normalize_tool_result(await server.call_tool("web_screenshot", {"annotate": True}))
            assert isinstance(annotated, dict)
            assert annotated["status"] == "ok"
            assert annotated["text"] == "Captured annotated viewport screenshot."
            assert_fresh_refs(annotated)
            assert "screenshot_path" not in annotated
            assert str(annotated["screenshot_url"]).startswith("data:image/png;base64,")
            annotation = annotated["result"]["annotation"]
            assert annotation["kind"] == "set_of_mark"
            assert annotation["coordinate_frame"] == "viewport_css"
            assert annotation["marks"][0]["ref"] == "1"
            assert annotation["marks"][0]["label"] == "1"
            assert annotation["marks"][0]["bbox_css"] == {"x": 48.0, "y": 40.0, "width": 120.0, "height": 34.0}
            assert annotation["marks"][1]["ref"] == "2"
            assert annotation["marks"][1]["label"] == "2"
            assert {mark["ref"] for mark in annotation["marks"]} == {"1", "2"}
            assert "source_ref" not in annotation["marks"][0]
            assert annotation["marked_count"] == 2
            assert annotation["omitted_reason_counts"] == {"attribute_disabled": 1, "attribute_hidden": 1}
            assert "source_screenshot_path" not in annotation
            assert "overlay_screenshot_path" not in annotation
            assert any(
                path.name.startswith("tool_") and path.name.endswith(".png") and "_som" not in path.name
                for path in store.screenshots_dir.iterdir()
            )
            assert any(path.name.endswith("_som.png") for path in store.screenshots_dir.iterdir())
            assert any(path.name.endswith(".annotation.json") for path in store.screenshots_dir.iterdir())
            assert ["@e1", "@e2", "@e3", "@e4"] in client.box_batch_calls

    asyncio.run(_run())
    registry.clear()


def test_mutating_action_success_survives_post_observation_failure(tmp_path) -> None:
    ensure_web_agent_mcp_servers_registered()
    registry = WebAgentSessionRegistry()
    registry.clear()
    client = _PostObservationFailClient()
    store = WebArtifactStore.for_session(tmp_path, "post-observation-fail", ArtifactConfig())
    registered = RegisteredWebAgentSession(
        session_id="post-observation-fail",
        client=client,
        artifact_store=store,
        settings=WebAgentConfig(
            post_action_wait_ms=0,
            allow_private_network_urls=True,
            post_action_settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=0),
        ),
    )
    registered.update_ref_aliases(
        WebObservationRefAliases(
            observation_id="unit-test",
            aliases=[WebRefAlias(ref="1", source_ref="@e1")],
        )
    )
    registry.register(registered)
    server = NorthCUAMCPRegistry.build("WebAgentActionExecutor", context=MCPFactoryContext(process_name="test"))

    async def _run() -> None:
        with use_mcp_invocation_context(MCPInvocationContext(session_id="post-observation-fail")):
            clicked = normalize_tool_result(await server.call_tool("web_click_ref", {"ref": "1"}))
            assert isinstance(clicked, dict)
            assert clicked["status"] == "ok"
            assert clicked["result"] == {"clicked": "1"}
            assert "observation" not in clicked
            assert clicked["observation_error"]["code"] == "post_action_observation_failed"
            assert clicked["observation_error"]["retryable"] is True
            assert client.calls == [("click_ref", {"ref": "@e1"})]

    asyncio.run(_run())
    registry.clear()


def test_mutating_action_success_survives_post_observation_timeout(tmp_path) -> None:
    ensure_web_agent_mcp_servers_registered()
    registry = WebAgentSessionRegistry()
    registry.clear()
    client = _PostObservationTimeoutClient()
    store = WebArtifactStore.for_session(tmp_path, "post-observation-timeout", ArtifactConfig())
    registered = RegisteredWebAgentSession(
        session_id="post-observation-timeout",
        client=client,
        artifact_store=store,
        settings=WebAgentConfig(
            post_action_wait_ms=0,
            post_action_observation_timeout_seconds=0.01,
            allow_private_network_urls=True,
            post_action_settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=3000),
        ),
    )
    registered.update_ref_aliases(
        WebObservationRefAliases(
            observation_id="unit-test",
            aliases=[WebRefAlias(ref="1", source_ref="@e1")],
        )
    )
    registry.register(registered)
    server = NorthCUAMCPRegistry.build("WebAgentActionExecutor", context=MCPFactoryContext(process_name="test"))

    async def _run() -> None:
        with use_mcp_invocation_context(MCPInvocationContext(session_id="post-observation-timeout")):
            clicked = normalize_tool_result(await server.call_tool("web_click_ref", {"ref": "1"}))
            assert isinstance(clicked, dict)
            assert clicked["status"] == "ok"
            assert clicked["result"] == {"clicked": "1"}
            assert "observation" not in clicked
            assert clicked["observation_error"]["code"] == "post_action_observation_failed"
            assert clicked["observation_error"]["error_type"] == "TimeoutError"
            assert clicked["observation_error"]["error"] == "post-action observation exceeded 0.01 seconds"
            assert clicked["observation_error"]["retryable"] is True
            assert client.calls == [("click_ref", {"ref": "@e1"})]

    asyncio.run(_run())
    assert client.snapshot_cancelled is True
    registry.clear()


def test_pre_action_overlay_reuses_actionable_observation_without_refreshing_refs(tmp_path) -> None:
    """Overlay capture must not rewrite backend refs before a ref action executes."""

    class _SnapshotSensitiveClient(_FakeMCPClient):
        def __init__(self) -> None:
            super().__init__()
            self.snapshot_calls_at_click: list[int] = []

        async def click_ref(self, ref: str) -> WebActionCommandResult:
            self.snapshot_calls_at_click.append(self.snapshot_calls)
            return await super().click_ref(ref)

    ensure_web_agent_mcp_servers_registered()
    registry = WebAgentSessionRegistry()
    registry.clear()
    client = _SnapshotSensitiveClient()
    store = WebArtifactStore.for_session(tmp_path, "pre-action-ref-stability", ArtifactConfig())
    registered = RegisteredWebAgentSession(
        session_id="pre-action-ref-stability",
        client=client,
        artifact_store=store,
        settings=WebAgentConfig(
            post_action_wait_ms=0,
            allow_private_network_urls=True,
            post_action_settle=WebPostActionSettleConfig(
                debounce_ms=0,
                sample_interval_ms=0,
                max_wait_ms=0,
                final_bbox_verify=False,
            ),
        ),
    )
    registry.register(registered)
    service = WebAgentMCPActionService(registry=registry)
    server = NorthCUAMCPRegistry.build("WebAgentActionExecutor", context=MCPFactoryContext(process_name="test"))

    async def _run() -> None:
        initial = await service.collect_current_observation(
            registered,
            mode="som",
            observation_id="initial-actionable",
        )
        assert registered.current_actionable_observation is initial
        assert registered.resolve_model_ref("1") == "@e1"
        assert client.snapshot_calls == 1

        with use_mcp_invocation_context(MCPInvocationContext(session_id="pre-action-ref-stability")):
            clicked = normalize_tool_result(await server.call_tool("web_click_ref", {"ref": "1"}))

        assert isinstance(clicked, dict)
        assert clicked["status"] == "ok"
        assert client.calls[0] == ("click_ref", {"ref": "@e1"})
        # The action executes against the one initial snapshot. Post-action
        # settling may capture more state, but no pre-action refresh may occur.
        assert client.snapshot_calls_at_click == [1]
        assert any(path.name.endswith("_action_overlay.png") for path in store.screenshots_dir.iterdir())

    asyncio.run(_run())
    registry.clear()


def test_web_agent_runtime_only_bbox_and_viewport_probes(tmp_path) -> None:
    ensure_web_agent_mcp_servers_registered()
    registry = WebAgentSessionRegistry()
    registry.clear()
    client = _FakeMCPClient()
    store = WebArtifactStore.for_session(tmp_path, "probe-sess", ArtifactConfig())
    registry.register(
        RegisteredWebAgentSession(
            session_id="probe-sess",
            client=client,
            artifact_store=store,
            settings=WebAgentConfig(post_action_wait_ms=0),
        )
    )
    server = NorthCUAMCPRegistry.build("WebAgentProbeExecutor", context=MCPFactoryContext(process_name="test"))

    async def _run() -> None:
        with use_mcp_invocation_context(MCPInvocationContext(session_id="probe-sess")):
            box = normalize_tool_result(await server.call_tool("web_agent_get_box", {"ref": "e1"}))
            assert box == {"x": 48, "y": 40, "width": 120, "height": 34}
            boxes = normalize_tool_result(await server.call_tool("web_agent_get_boxes", {"refs": ["e1", "@e2", "e1"]}))
            assert boxes == {
                "boxes": {
                    "@e1": {"x": 48, "y": 40, "width": 120, "height": 34},
                    "@e2": {"x": 48, "y": 96, "width": 220, "height": 30},
                },
                "errors": {},
            }
            states = normalize_tool_result(
                await server.call_tool("web_agent_get_ref_states", {"refs": ["e1", "@e3", "e4", "e1"]})
            )
            assert states == {
                "states": {
                    "@e1": {"visible": True, "enabled": True},
                    "@e3": {"visible": False, "enabled": True},
                    "@e4": {"visible": True, "enabled": False},
                },
                "errors": {},
            }
            viewport = normalize_tool_result(await server.call_tool("web_agent_viewport", {}))
            assert viewport == {"width": 640, "height": 360, "devicePixelRatio": 1}

    asyncio.run(_run())
    registry.clear()


def test_interaction_first_som_keeps_visible_domain_metadata_links() -> None:
    selection = InteractionFirstSomSelector(WebSomSelectionConfig(max_marks=10, max_box_lookup=10)).select(
        elements=[
            WebElementRef(ref_id="e1", role="link", name="example.com"),
            WebElementRef(ref_id="e2", role="link", name="12 comments"),
        ],
        viewport_width=640,
        viewport_height=360,
        box_payloads={
            "@e1": {"x": 120, "y": 180, "width": 90, "height": 14},
            "@e2": {"x": 220, "y": 180, "width": 90, "height": 14},
        },
        state_payloads={
            "@e1": {"visible": True, "enabled": True},
            "@e2": {"visible": True, "enabled": True},
        },
    )

    assert [mark.element.model_ref for mark in selection.marks] == ["@e1", "@e2"]
    assert selection.omitted_reason_counts == {}


def test_annotated_screenshot_materializes_remote_path_data_url(tmp_path) -> None:
    ensure_web_agent_mcp_servers_registered()
    registry = WebAgentSessionRegistry()
    registry.clear()
    client = _FakeMCPClient(screenshot_path="/remote/device/tool.png")
    store = WebArtifactStore.for_session(tmp_path, "remote-shot-sess", ArtifactConfig())
    registry.register(
        RegisteredWebAgentSession(
            session_id="remote-shot-sess",
            client=client,
            artifact_store=store,
            settings=WebAgentConfig(post_action_wait_ms=0, allow_private_network_urls=True),
        )
    )
    server = NorthCUAMCPRegistry.build("WebAgentActionExecutor", context=MCPFactoryContext(process_name="test"))

    async def _run() -> None:
        with use_mcp_invocation_context(MCPInvocationContext(session_id="remote-shot-sess")):
            annotated = normalize_tool_result(await server.call_tool("web_screenshot", {"annotate": True}))
            assert isinstance(annotated, dict)
            assert annotated["status"] == "ok"
            assert "screenshot_path" not in annotated
            annotation = annotated["result"]["annotation"]
            assert "source_screenshot_path" not in annotation
            assert "overlay_screenshot_path" not in annotation
            assert any(path.name.endswith("_som.png") for path in store.screenshots_dir.iterdir())
            assert any(path.name.endswith(".annotation.json") for path in store.screenshots_dir.iterdir())

    asyncio.run(_run())
    registry.clear()
