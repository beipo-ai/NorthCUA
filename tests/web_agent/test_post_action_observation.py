from __future__ import annotations

import asyncio
import base64
from io import BytesIO

import pytest
from PIL import Image

from northcua.agents.web.artifacts import WebArtifactStore
from northcua.agents.web.backend import WebActionBackendTimeoutError, WebActionCommandResult
from northcua.agents.web.domain.actions import WebAgentActionType
from northcua.agents.web.domain.observations import WebElementRef, WebScreenshotObservation, WebSnapshotObservation
from northcua.agents.web.mcp.action_service import WebAgentMCPActionService
from northcua.agents.web.observation.collector import WebObservationCollector, WebObservationQualityError
from northcua.agents.web.observation.projector import WebObservationProjector
from northcua.agents.web.observation.settle import WebPostActionSettleService, WebSettleResult
from northcua.agents.web.session import RegisteredWebAgentSession
from northcua.core.config import ArtifactConfig, WebAgentConfig, WebPostActionSettleConfig


class _StableClient:
    def __init__(self) -> None:
        self.snapshot_calls = 0
        self.box_calls = 0
        self.state_calls = 0
        self.screenshot_calls = 0
        self.viewport_calls = 0
        self.waits: list[object] = []
        self.wait_timeouts: list[float | None] = []

    async def wait(self, params, *, timeout_seconds: float | None = None) -> WebActionCommandResult:
        self.waits.append(params)
        self.wait_timeouts.append(timeout_seconds)
        return WebActionCommandResult(success=True, data={"waited": True})

    async def snapshot(self) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        return WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            snapshot='button "Submit" [ref=@e1]',
            elements=[WebElementRef(ref_id="e1", role="button", name="Submit")],
        )

    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        self.box_calls += 1
        return WebActionCommandResult(
            success=True,
            data={"boxes": {ref: {"x": 10, "y": 20, "width": 100, "height": 30} for ref in refs}},
        )

    async def get_ref_states(self, refs: list[str]) -> WebActionCommandResult:
        self.state_calls += 1
        return WebActionCommandResult(
            success=True,
            data={"states": {ref: {"visible": True, "enabled": True} for ref in refs}},
        )

    async def get_viewport_size(self) -> WebActionCommandResult:
        self.viewport_calls += 1
        return WebActionCommandResult(success=True, data={"width": 640, "height": 360, "devicePixelRatio": 1})

    async def get_url(self) -> WebActionCommandResult:
        return WebActionCommandResult(success=True, data={"url": "https://example.com"})

    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        del kwargs
        self.screenshot_calls += 1
        return WebScreenshotObservation(data_url=_png_data_url(640, 360), width=640, height=360)


class _JavascriptHrefCurrentUrlClient(_StableClient):
    async def get_url(self) -> WebActionCommandResult:
        return WebActionCommandResult(success=True, data={"url": "javascript:;"})


class _ForbiddenUrlObservationFailClient(_StableClient):
    async def get_url(self) -> WebActionCommandResult:
        return WebActionCommandResult(success=True, data={"url": "file:///etc/passwd"})

    async def snapshot(self) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        raise RuntimeError("snapshot backend unavailable")


class _ForbiddenObservedUrlClient(_StableClient):
    async def snapshot(self) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        return WebSnapshotObservation(
            url="file:///etc/passwd",
            title="Forbidden",
            snapshot='button "Forbidden" [ref=@e1]',
            elements=[WebElementRef(ref_id="e1", role="button", name="Forbidden")],
        )


class _HangingSnapshotClient(_StableClient):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_cancelled = False

    async def snapshot(self) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        try:
            await asyncio.sleep(3600)
        finally:
            self.snapshot_cancelled = True
        raise AssertionError("unreachable")


class _ChangingClient(_StableClient):
    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        self.box_calls += 1
        return WebActionCommandResult(
            success=True,
            data={"boxes": {ref: {"x": 10 + self.box_calls, "y": 20, "width": 100, "height": 30} for ref in refs}},
        )

    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        del kwargs
        data = base64.b64encode(b"fake").decode("ascii")
        return WebScreenshotObservation(data_url=f"data:image/png;base64,{data}", width=640, height=360)


class _FailingBoxClient(_StableClient):
    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        self.box_calls += 1
        del refs
        return WebActionCommandResult(success=False, error="bbox backend unavailable")


class _PartialBoxErrorClient(_StableClient):
    async def snapshot(self) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        return WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            snapshot='button "One" [ref=@e1]\nbutton "Two" [ref=@e2]',
            elements=[
                WebElementRef(ref_id="e1", role="button", name="One"),
                WebElementRef(ref_id="e2", role="button", name="Two"),
            ],
        )

    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        self.box_calls += 1
        return WebActionCommandResult(
            success=True,
            data={
                "boxes": {"@e1": {"x": 10, "y": 20, "width": 100, "height": 30}},
                "errors": {"@e2": "element detached"},
            },
        )


class _ScreenshotTimeoutClient(_StableClient):
    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        del kwargs
        self.screenshot_calls += 1
        raise WebActionBackendTimeoutError("playwright command timed out after 3s: screenshot")


class _EmptySomClient(_StableClient):
    async def snapshot(self) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        return WebSnapshotObservation(
            url="https://example.com/rollerblade",
            title="Rollerblade",
            snapshot="",
            elements=[],
        )


class _UnparsedRefSnapshotClient(_EmptySomClient):
    async def snapshot(self) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        return WebSnapshotObservation(
            url="https://example.com/parser-drift",
            title="Parser Drift",
            snapshot='button "Submit" [ref=@e1]',
            elements=[],
        )


class _AllRefsOmittedSomClient(_StableClient):
    async def get_ref_states(self, refs: list[str]) -> WebActionCommandResult:
        self.state_calls += 1
        return WebActionCommandResult(
            success=True, data={"states": {ref: {"visible": False, "enabled": True} for ref in refs}}
        )


class _MissingBoxesSomClient(_StableClient):
    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        self.box_calls += 1
        del refs
        return WebActionCommandResult(success=True, data={"boxes": {}})


class _LargeMissingBoxesSomClient(_MissingBoxesSomClient):
    async def snapshot(self) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        elements = [WebElementRef(ref_id=f"e{index}", role="button", name=f"Button {index}") for index in range(1, 8)]
        snapshot_text = "\n".join(f'button "Button {index}" [ref=@e{index}]' for index in range(1, 8))
        return WebSnapshotObservation(
            url="https://example.com/large",
            title="Large",
            snapshot=snapshot_text,
            elements=elements,
        )


class _DistinctBoxClient(_StableClient):
    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        self.box_calls += 1
        boxes = {}
        for ref in refs:
            boxes[ref] = {"x": 10, "y": 72 if ref == "@e2" else 20, "width": 100, "height": 30}
        return WebActionCommandResult(success=True, data={"boxes": boxes})


class _PartiallyOutsideViewportClient(_StableClient):
    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        self.box_calls += 1
        return WebActionCommandResult(
            success=True,
            data={"boxes": {ref: {"x": 5, "y": 350, "width": 90, "height": 30} for ref in refs}},
        )


class _RetrySnapshotClient(_StableClient):
    def __init__(self) -> None:
        super().__init__()
        self.annotated_snapshot_refs: list[list[str]] = []

    async def snapshot(self) -> WebSnapshotObservation:
        self.snapshot_calls += 1
        if self.snapshot_calls == 1:
            return WebSnapshotObservation(
                url="https://example.com",
                title="Example",
                snapshot='button "Old" [ref=@e1]',
                elements=[WebElementRef(ref_id="e1", role="button", name="Old")],
            )
        return WebSnapshotObservation(
            url="https://example.com",
            title="Example",
            snapshot='button "New" [ref=@e2]',
            elements=[WebElementRef(ref_id="e2", role="button", name="New")],
        )

    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        self.box_calls += 1
        offset = 1 if self.box_calls == 1 else 2 if self.box_calls == 2 else 20
        return WebActionCommandResult(
            success=True,
            data={"boxes": {ref: {"x": offset, "y": 20, "width": 100, "height": 30} for ref in refs}},
        )

    async def screenshot(self, **kwargs) -> WebScreenshotObservation:
        del kwargs
        data = base64.b64encode(b"fake").decode("ascii")
        return WebScreenshotObservation(data_url=f"data:image/png;base64,{data}", width=640, height=360)


class _AsyncProbeGate:
    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.started: list[str] = []
        self.released = False
        self.event = asyncio.Event()

    async def arrive(self, name: str) -> None:
        self.started.append(name)
        if len(self.started) >= self.expected:
            self.released = True
            self.event.set()
        await asyncio.wait_for(self.event.wait(), timeout=1.0)


class _ConcurrentBboxClient(_StableClient):
    def __init__(self) -> None:
        super().__init__()
        self._gate: _AsyncProbeGate | None = None
        self.probe_waves: list[list[str]] = []

    def _next_gate(self) -> _AsyncProbeGate:
        if self._gate is None or self._gate.released:
            self._gate = _AsyncProbeGate(expected=2)
            self.probe_waves.append(self._gate.started)
        return self._gate

    async def get_boxes(self, refs: list[str]) -> WebActionCommandResult:
        gate = self._next_gate()
        await gate.arrive("boxes")
        return await super().get_boxes(refs)

    async def get_ref_states(self, refs: list[str]) -> WebActionCommandResult:
        gate = self._next_gate()
        await gate.arrive("states")
        return await super().get_ref_states(refs)


def _png_data_url(width: int, height: int) -> str:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _registered(
    tmp_path,
    client,
    *,
    settle: WebPostActionSettleConfig,
    config: WebAgentConfig | None = None,
) -> RegisteredWebAgentSession:
    settings = config or WebAgentConfig()
    settings.allow_private_network_urls = True
    settings.post_action_settle = settle
    settings.post_action_observation_mode = "som"
    return RegisteredWebAgentSession(
        session_id="settle-test",
        client=client,
        artifact_store=WebArtifactStore.for_session(tmp_path, "settle-test", ArtifactConfig()),
        settings=settings,
    )


def test_attach_observation_payload_preserves_action_overlay_image_url() -> None:
    overlay_url = _png_data_url(8, 8)

    payload = WebAgentMCPActionService().attach_observation_payload(
        {"status": "ok", "text": "clicked"},
        observation=None,
        image_urls={"action_overlay_image_url": overlay_url},
    )

    assert payload["action_overlay_image_url"] == overlay_url


def test_post_action_settle_returns_stable_after_required_samples(tmp_path) -> None:
    client = _StableClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )

    async def _run() -> None:
        result = await WebPostActionSettleService().settle_after_action(
            registered=registered,
            action=WebAgentActionType.CLICK_REF,
        )
        assert result.settled is True
        assert result.reason == "stable_2_samples"

    asyncio.run(_run())
    assert client.snapshot_calls == 2
    assert client.box_calls == 2
    assert client.state_calls == 2


def test_post_action_settle_reads_boxes_and_states_concurrently(tmp_path) -> None:
    client = _ConcurrentBboxClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )
    snapshot = WebSnapshotObservation(
        url="https://example.com",
        title="Example",
        snapshot='button "Submit" [ref=@e1]',
        elements=[WebElementRef(ref_id="e1", role="button", name="Submit")],
    )

    async def _run() -> None:
        bbox_hash = await WebPostActionSettleService().bbox_hash(registered, snapshot.elements)
        assert bbox_hash

    asyncio.run(_run())
    assert len(client.probe_waves) == 1
    assert set(client.probe_waves[0]) == {"boxes", "states"}
    assert client.box_calls == 1
    assert client.state_calls == 1


def test_navigation_settle_load_wait_uses_short_timeout(tmp_path) -> None:
    client = _StableClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(
            debounce_ms=0,
            sample_interval_ms=0,
            max_wait_ms=0,
            load_wait_timeout_ms=1200,
        ),
    )

    async def _run() -> None:
        await WebPostActionSettleService().settle_after_action(
            registered=registered,
            action=WebAgentActionType.OPEN,
        )

    asyncio.run(_run())
    assert client.wait_timeouts == [1.2]


def test_best_effort_post_action_validates_collected_observation_url(tmp_path) -> None:
    client = _JavascriptHrefCurrentUrlClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )

    async def _run() -> None:
        (
            observation,
            image_urls,
            observation_error,
        ) = await WebAgentMCPActionService().best_effort_post_action_observation_payload(
            registered,
            action=WebAgentActionType.CLICK_REF,
        )
        assert observation_error is None
        assert observation is not None
        assert observation["page"]["url"] == "https://example.com"
        assert str(image_urls["screenshot_url"]).startswith("data:image/png;base64,")
        assert str(image_urls["raw_image_url"]).startswith("data:image/png;base64,")
        assert str(image_urls["som_image_url"]).startswith("data:image/png;base64,")

    asyncio.run(_run())


def test_best_effort_post_action_observation_failure_still_validates_current_url(tmp_path) -> None:
    client = _ForbiddenUrlObservationFailClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )

    async def _run() -> None:
        with pytest.raises(Exception, match="file"):
            await WebAgentMCPActionService().best_effort_post_action_observation_payload(
                registered,
                action=WebAgentActionType.CLICK_REF,
            )

    asyncio.run(_run())


def test_best_effort_post_action_observation_timeout_includes_settle(tmp_path) -> None:
    client = _HangingSnapshotClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=3000),
        config=WebAgentConfig(post_action_observation_timeout_seconds=0.01),
    )

    async def _run() -> None:
        observation, image_urls, observation_error = await asyncio.wait_for(
            WebAgentMCPActionService().best_effort_post_action_observation_payload(
                registered,
                action=WebAgentActionType.CLICK_REF,
            ),
            timeout=0.5,
        )

        assert observation is None
        assert image_urls == {}
        assert observation_error is not None
        assert observation_error["code"] == "post_action_observation_failed"
        assert observation_error["error_type"] == "TimeoutError"
        assert observation_error["error"] == "post-action observation exceeded 0.01 seconds"
        assert observation_error["retryable"] is True

    asyncio.run(_run())
    assert client.snapshot_calls == 1
    assert client.snapshot_cancelled is True


def test_best_effort_post_action_observation_propagates_caller_cancellation(tmp_path) -> None:
    client = _HangingSnapshotClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=3000),
        config=WebAgentConfig(post_action_observation_timeout_seconds=60),
    )

    async def _run() -> None:
        task = asyncio.create_task(
            WebAgentMCPActionService().best_effort_post_action_observation_payload(
                registered,
                action=WebAgentActionType.CLICK_REF,
            )
        )
        while client.snapshot_calls == 0:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    assert client.snapshot_cancelled is True


def test_best_effort_post_action_rejects_disallowed_observed_url_before_alias_update(tmp_path) -> None:
    client = _ForbiddenObservedUrlClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )

    async def _run() -> None:
        with pytest.raises(Exception, match="file"):
            await WebAgentMCPActionService().best_effort_post_action_observation_payload(
                registered,
                action=WebAgentActionType.CLICK_REF,
            )

    asyncio.run(_run())
    assert registered.current_ref_aliases.aliases == []


def test_post_action_settle_timeout_is_best_effort(tmp_path) -> None:
    client = _ChangingClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=0),
    )

    async def _run() -> None:
        result = await WebPostActionSettleService().settle_after_action(
            registered=registered,
            action=WebAgentActionType.SCROLL,
        )
        assert result.settled is False
        assert result.reason.startswith("timeout_best_effort")

    asyncio.run(_run())


def test_post_action_settle_bbox_probe_failure_is_not_reported_stable(tmp_path) -> None:
    client = _FailingBoxClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=0),
    )

    async def _run() -> None:
        result = await WebPostActionSettleService().settle_after_action(
            registered=registered,
            action=WebAgentActionType.CLICK_REF,
        )
        assert result.settled is False
        assert result.reason.startswith("timeout_best_effort")
        assert "box probe failed" in result.reason

    asyncio.run(_run())
    assert client.box_calls == 1


def test_post_action_settle_partial_bbox_probe_errors_are_not_reported_stable(tmp_path) -> None:
    client = _PartialBoxErrorClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=0),
    )

    async def _run() -> None:
        result = await WebPostActionSettleService().settle_after_action(
            registered=registered,
            action=WebAgentActionType.CLICK_REF,
        )
        assert result.settled is False
        assert result.reason.startswith("timeout_best_effort")
        assert "per-ref box errors" in result.reason
        assert "element detached" in result.reason

    asyncio.run(_run())
    assert client.box_calls == 1


def test_observation_collector_marks_bbox_mismatch_best_effort(tmp_path) -> None:
    client = _ChangingClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )

    async def _annotate(_registered, screenshot, _snapshot, **_kwargs):
        return screenshot

    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=lambda _registered, screenshot: screenshot,
        annotate_screenshot=_annotate,
    )

    async def _run() -> None:
        observation = await collector.collect(
            registered=registered,
            mode="som",
            settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
        )
        assert observation.settled is False
        assert "bbox_changed_during_capture_best_effort" in observation.settle_reason
        assert observation.screenshot is not None

    asyncio.run(_run())
    assert client.box_calls == 4


def test_observation_collector_degrades_to_snapshot_when_screenshot_times_out(tmp_path) -> None:
    client = _ScreenshotTimeoutClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )

    async def _annotate(_registered, screenshot, _snapshot, **_kwargs):
        raise AssertionError("screenshot timeout should skip annotation")

    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=lambda _registered, screenshot: screenshot,
        annotate_screenshot=_annotate,
    )

    async def _run() -> None:
        observation = await collector.collect(
            registered=registered,
            mode="som",
            settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
        )
        assert observation.settled is False
        assert "screenshot_capture_failed" in observation.settle_reason
        assert observation.snapshot.elements[0].model_ref == "@e1"
        assert observation.screenshot is not None
        assert "WebActionBackendTimeoutError" in str(observation.screenshot.raw.get("capture_error"))
        bundle = WebObservationProjector().bundle_dict(observation)
        assert bundle["snapshot"]["ref_count"] == 1
        assert bundle["screenshot"]["image_attached"] is False
        assert "screenshot_capture_failed" in bundle["screenshot"]["image_omitted_reason"]

    asyncio.run(_run())
    assert client.screenshot_calls == 1
    assert client.box_calls == 1


def test_observation_collector_returns_retry_snapshot_when_bbox_retry_succeeds(tmp_path) -> None:
    client = _RetrySnapshotClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )

    async def _annotate(_registered, screenshot, snapshot, **_kwargs):
        client.annotated_snapshot_refs.append([element.model_ref for element in snapshot.elements])
        return screenshot

    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=lambda _registered, screenshot: screenshot,
        annotate_screenshot=_annotate,
    )

    async def _run() -> None:
        observation = await collector.collect(
            registered=registered,
            mode="som",
            settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
        )
        assert observation.settled is True
        assert observation.snapshot.elements[0].model_ref == "@e2"
        assert "New" in observation.snapshot.snapshot
        assert client.annotated_snapshot_refs[-1] == ["@e2"]

    asyncio.run(_run())


def test_observation_collector_reuses_final_probe_for_som_annotation(tmp_path) -> None:
    client = _StableClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )
    service = WebAgentMCPActionService()
    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=service.persist_screenshot_if_needed,
        annotate_screenshot=service.annotate_screenshot_with_marks,
    )

    async def _run() -> None:
        observation = await collector.collect(
            registered=registered,
            mode="som",
            settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
        )
        assert observation.settled is True
        assert observation.screenshot is not None
        assert observation.screenshot.annotation is not None
        assert [mark.ref for mark in observation.screenshot.annotation.marks] == ["1"]
        assert [mark.label for mark in observation.screenshot.annotation.marks] == ["1"]
        assert [mark.source_ref for mark in observation.screenshot.annotation.marks] == ["@e1"]

    asyncio.run(_run())
    assert client.box_calls == 2
    assert client.state_calls == 2
    assert client.viewport_calls == 1
    assert client.screenshot_calls == 1


def test_som_annotation_fallback_reads_boxes_and_states_concurrently(tmp_path) -> None:
    client = _ConcurrentBboxClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )
    service = WebAgentMCPActionService()
    screenshot = service.persist_screenshot_if_needed(
        registered,
        WebScreenshotObservation(data_url=_png_data_url(640, 360), width=640, height=360),
    )

    async def _run() -> None:
        snapshot = await client.snapshot()
        annotated = await service.annotate_screenshot_with_marks(registered, screenshot, snapshot)
        assert annotated.annotation is not None
        assert [mark.ref for mark in annotated.annotation.marks] == ["1"]
        assert [mark.label for mark in annotated.annotation.marks] == ["1"]
        assert [mark.source_ref for mark in annotated.annotation.marks] == ["@e1"]

    asyncio.run(_run())
    assert len(client.probe_waves) == 1
    assert set(client.probe_waves[0]) == {"boxes", "states"}
    assert client.box_calls == 1
    assert client.state_calls == 1


def test_som_annotation_tops_up_missing_probe_payloads(tmp_path) -> None:
    client = _DistinctBoxClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )
    service = WebAgentMCPActionService()
    screenshot = service.persist_screenshot_if_needed(
        registered,
        WebScreenshotObservation(data_url=_png_data_url(640, 360), width=640, height=360),
    )
    snapshot = WebSnapshotObservation(
        url="https://example.com",
        title="Example",
        snapshot='button "One" [ref=@e1]\nbutton "Two" [ref=@e2]',
        elements=[
            WebElementRef(ref_id="e1", role="button", name="One"),
            WebElementRef(ref_id="e2", role="button", name="Two"),
        ],
    )

    async def _run() -> None:
        annotated = await service.annotate_screenshot_with_marks(
            registered,
            screenshot,
            snapshot,
            viewport_payload={"width": 640, "height": 360},
            box_payloads={"@e1": {"x": 10, "y": 20, "width": 100, "height": 30}},
            state_payloads={"@e1": {"visible": True, "enabled": True}},
        )
        assert annotated.annotation is not None
        assert {mark.ref for mark in annotated.annotation.marks} == {"1", "2"}
        assert {mark.source_ref for mark in annotated.annotation.marks} == {"@e1", "@e2"}

    asyncio.run(_run())
    assert client.box_calls == 1
    assert client.state_calls == 1


def test_observation_collector_allows_empty_som_on_content_only_page(tmp_path) -> None:
    client = _EmptySomClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(
            debounce_ms=0,
            sample_interval_ms=0,
            max_wait_ms=100,
            final_bbox_verify=False,
        ),
    )
    registered.settings.som_empty_ref_retry_attempts = 1
    service = WebAgentMCPActionService()
    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=service.persist_screenshot_if_needed,
        annotate_screenshot=service.annotate_screenshot_with_marks,
    )

    async def _run() -> None:
        observation = await collector.collect(
            registered=registered,
            mode="som",
            settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
        )
        assert observation.screenshot is not None
        assert observation.screenshot.annotation is not None
        assert observation.screenshot.annotation.marks == []
        assert observation.screenshot.annotation.visible_interactive_count == 0
        assert observation.screenshot.annotation.omitted_reason_counts == {}

    asyncio.run(_run())
    assert client.snapshot_calls == 1
    assert client.screenshot_calls == 1


def test_observation_collector_retries_and_fails_ref_text_without_parsed_elements(tmp_path) -> None:
    client = _UnparsedRefSnapshotClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(
            debounce_ms=0,
            sample_interval_ms=0,
            max_wait_ms=100,
            final_bbox_verify=False,
        ),
    )
    registered.settings.som_empty_ref_retry_attempts = 1
    service = WebAgentMCPActionService()
    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=service.persist_screenshot_if_needed,
        annotate_screenshot=service.annotate_screenshot_with_marks,
    )

    async def _run() -> None:
        with pytest.raises(WebObservationQualityError, match="model-visible refs and SoM marks were empty"):
            await collector.collect(
                registered=registered,
                mode="som",
                settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
            )

    asyncio.run(_run())
    assert client.snapshot_calls == 2
    assert client.screenshot_calls == 2


def test_observation_collector_allows_empty_som_when_refs_are_hidden(tmp_path) -> None:
    client = _AllRefsOmittedSomClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(
            debounce_ms=0,
            sample_interval_ms=0,
            max_wait_ms=100,
            final_bbox_verify=False,
        ),
    )
    registered.settings.som_empty_ref_retry_attempts = 1
    service = WebAgentMCPActionService()
    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=service.persist_screenshot_if_needed,
        annotate_screenshot=service.annotate_screenshot_with_marks,
    )

    async def _run() -> None:
        observation = await collector.collect(
            registered=registered,
            mode="som",
            settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
        )
        assert observation.screenshot is not None
        assert observation.screenshot.annotation is not None
        assert observation.screenshot.annotation.marks == []
        assert observation.screenshot.annotation.visible_interactive_count == 0
        assert observation.screenshot.annotation.omitted_reason_counts == {"state_hidden": 1}

    asyncio.run(_run())
    assert client.snapshot_calls == 1
    assert client.screenshot_calls == 1
    assert client.box_calls == 1
    assert client.state_calls == 1


def test_observation_collector_retries_and_fails_when_all_refs_missing_bbox(tmp_path) -> None:
    client = _MissingBoxesSomClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(
            debounce_ms=0,
            sample_interval_ms=0,
            max_wait_ms=100,
            final_bbox_verify=False,
        ),
    )
    registered.settings.som_empty_ref_retry_attempts = 1
    service = WebAgentMCPActionService()
    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=service.persist_screenshot_if_needed,
        annotate_screenshot=service.annotate_screenshot_with_marks,
    )

    async def _run() -> None:
        with pytest.raises(WebObservationQualityError, match="model-visible refs and SoM marks were empty"):
            await collector.collect(
                registered=registered,
                mode="som",
                settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
            )

    asyncio.run(_run())
    assert client.snapshot_calls == 2
    assert client.screenshot_calls == 2
    assert client.box_calls == 4
    assert client.state_calls == 4


def test_observation_collector_default_bbox_verify_still_fails_missing_bbox_quality(tmp_path) -> None:
    client = _MissingBoxesSomClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(
            debounce_ms=0,
            sample_interval_ms=0,
            max_wait_ms=100,
            final_bbox_verify=True,
        ),
    )
    registered.settings.som_empty_ref_retry_attempts = 1
    service = WebAgentMCPActionService()
    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=service.persist_screenshot_if_needed,
        annotate_screenshot=service.annotate_screenshot_with_marks,
    )

    async def _run() -> None:
        with pytest.raises(WebObservationQualityError, match="model-visible refs and SoM marks were empty"):
            await collector.collect(
                registered=registered,
                mode="som",
                settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
            )

    asyncio.run(_run())
    assert client.snapshot_calls >= 4
    assert client.screenshot_calls >= 4
    assert client.box_calls >= 8
    assert client.state_calls >= 8


def test_observation_collector_uses_probe_window_for_missing_bbox_failure(tmp_path) -> None:
    client = _LargeMissingBoxesSomClient()
    registered = _registered(
        tmp_path,
        client,
        config=WebAgentConfig(som_max_box_lookup=3),
        settle=WebPostActionSettleConfig(
            debounce_ms=0,
            sample_interval_ms=0,
            max_wait_ms=100,
            final_bbox_verify=False,
        ),
    )
    registered.settings.som_empty_ref_retry_attempts = 1
    service = WebAgentMCPActionService()
    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=service.persist_screenshot_if_needed,
        annotate_screenshot=service.annotate_screenshot_with_marks,
    )

    async def _run() -> None:
        with pytest.raises(WebObservationQualityError, match="model-visible refs and SoM marks were empty"):
            await collector.collect(
                registered=registered,
                mode="som",
                settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
            )

    asyncio.run(_run())
    assert client.snapshot_calls == 2
    assert client.screenshot_calls == 2
    assert client.box_calls == 4
    assert client.state_calls == 4


def test_observation_collector_retries_and_fails_annotation_error_with_refs(tmp_path) -> None:
    client = _StableClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(
            debounce_ms=0,
            sample_interval_ms=0,
            max_wait_ms=100,
            final_bbox_verify=False,
        ),
    )
    registered.settings.som_empty_ref_retry_attempts = 1
    service = WebAgentMCPActionService()
    annotation_calls = 0

    async def _failing_annotator(*args, **kwargs) -> WebScreenshotObservation:
        nonlocal annotation_calls
        annotation_calls += 1
        raise RuntimeError("annotation backend unavailable")

    collector = WebObservationCollector(
        settle_service=WebPostActionSettleService(),
        persist_screenshot=service.persist_screenshot_if_needed,
        annotate_screenshot=_failing_annotator,
    )

    async def _run() -> None:
        with pytest.raises(WebObservationQualityError, match="model-visible refs and SoM marks were empty"):
            await collector.collect(
                registered=registered,
                mode="som",
                settle_result=WebSettleResult(settled=True, reason="stable_2_samples", sample_count=2),
            )

    asyncio.run(_run())
    assert annotation_calls == 2
    assert client.snapshot_calls == 2
    assert client.screenshot_calls == 2


def test_som_annotation_clips_partially_visible_boxes_to_viewport(tmp_path) -> None:
    client = _PartiallyOutsideViewportClient()
    registered = _registered(
        tmp_path,
        client,
        settle=WebPostActionSettleConfig(debounce_ms=0, sample_interval_ms=0, max_wait_ms=100),
    )
    service = WebAgentMCPActionService()
    screenshot = service.persist_screenshot_if_needed(
        registered,
        WebScreenshotObservation(data_url=_png_data_url(640, 360), width=640, height=360),
    )

    async def _run() -> None:
        snapshot = await client.snapshot()
        annotated = await service.annotate_screenshot_with_marks(
            registered,
            screenshot,
            snapshot,
            viewport_payload={"width": 640, "height": 360},
        )
        assert annotated.annotation is not None
        mark = annotated.annotation.marks[0]
        assert mark.bbox_css.y == 350
        assert mark.bbox_css.height == 10
        assert mark.bbox_image.y == 350
        assert mark.bbox_image.height == 10

    asyncio.run(_run())
