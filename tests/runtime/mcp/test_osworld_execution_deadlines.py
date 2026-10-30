"""CPU fault injection for OSWorld transport; never contacts a real desktop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from northcua.runtime.mcp.desktop_native import osworld_http
from northcua.runtime.mcp.desktop_native._models import DesktopNativeError
from northcua.runtime.mcp.desktop_native.osworld_backend import OSWorldDesktopBackend
from northcua_protocol.state import CommandState
from northcua_runtime_shared import execution
from northcua_runtime_shared.execution import RuntimeExecutionError, execution_scope


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(osworld_http, "time", clock)
    monkeypatch.setattr(execution, "time", clock)
    return clock


def response(payload=None):
    return SimpleNamespace(raise_for_status=lambda: None, close=lambda: None, json=lambda: payload)


def test_read_retries_share_budget_and_backoff(clock):
    backend = OSWorldDesktopBackend()
    backend.retry_times = 3
    backend.retry_interval_seconds = 1
    budgets = []

    def request(*args, timeout, **kwargs):
        budgets.append(timeout.total)
        clock.now += 1.5
        raise requests.ReadTimeout("sensitive endpoint omitted from public failure")

    backend._session = SimpleNamespace(request=request)
    with execution_scope(4), pytest.raises(RuntimeExecutionError) as exc:
        backend._request("GET", "/screenshot", timeout=10)
    assert budgets == [4, 1.5]
    assert clock.sleeps == [1]
    assert clock.now == 4
    assert exc.value.status == CommandState.TIMED_OUT


def test_post_response_loss_never_reexecutes_action(clock, caplog):
    backend = OSWorldDesktopBackend()
    effects = []

    def request(*args, **kwargs):
        effects.append("clicked")
        raise requests.ReadTimeout("credential-token-must-not-appear")

    backend._session = SimpleNamespace(request=request)
    with execution_scope(300) as context, pytest.raises(RuntimeExecutionError) as exc:
        backend.click(1, 2)
    assert effects == ["clicked"]
    assert clock.sleeps == []
    assert exc.value.code == "COMMAND_DELIVERY_UNKNOWN"
    assert context.failure is exc.value
    assert "credential-token-must-not-appear" not in caplog.text


def test_read_retries_connection_failure_and_returns_with_remaining_budget(clock):
    backend = OSWorldDesktopBackend()
    backend.retry_interval_seconds = 1
    budgets = []

    def request(*args, timeout, **kwargs):
        budgets.append(timeout.total)
        if len(budgets) == 1:
            clock.now += 2
            raise requests.ConnectionError("not connected")
        return response()

    backend._session = SimpleNamespace(request=request)
    with execution_scope(10):
        backend._request("GET", "/screenshot", timeout=10)
    assert budgets == [10, 7]


def test_expired_budget_does_not_send_any_request(clock):
    backend = OSWorldDesktopBackend()
    backend._session = SimpleNamespace(request=lambda *args, **kwargs: pytest.fail("must not send"))
    with execution_scope(0), pytest.raises(RuntimeExecutionError, match="COMMAND_TIMED_OUT"):
        backend.click(1, 2)


def test_late_success_is_not_delivered_as_confirmed_post_success(clock):
    backend = OSWorldDesktopBackend()
    calls = []

    def request(*args, **kwargs):
        calls.append("action")
        clock.now = 11
        return response()

    backend._session = SimpleNamespace(request=request)
    with execution_scope(10), pytest.raises(RuntimeExecutionError, match="COMMAND_DELIVERY_UNKNOWN"):
        backend.click(1, 2)
    assert calls == ["action"]


def test_backend_business_error_remains_an_ordinary_tool_error(clock):
    backend = OSWorldDesktopBackend()
    backend._session = SimpleNamespace(
        request=lambda *args, **kwargs: response({"status": "error", "error": "bad key"})
    )
    with execution_scope(10) as context, pytest.raises(DesktopNativeError, match="bad key"):
        backend.click(1, 2)
    assert context.failure is None


def test_read_exhaustion_is_typed_without_exception_text(clock):
    backend = OSWorldDesktopBackend()
    backend.retry_interval_seconds = 0
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        raise requests.ConnectionError("private diagnostic")

    backend._session = SimpleNamespace(request=request)
    with pytest.raises(RuntimeExecutionError, match="^COMMAND_TRANSPORT_FAILED$"):
        backend._request("GET", "/screenshot", timeout=10)
    assert len(calls) == 3


def test_per_request_ceiling_does_not_shorten_healthy_screenshot_retries(clock):
    backend = OSWorldDesktopBackend()
    budgets = []

    def request(*args, timeout, **kwargs):
        budgets.append(timeout.total)
        clock.now += 9
        if len(budgets) == 1:
            raise requests.ReadTimeout("transient screenshot timeout")
        return response()

    backend._session = SimpleNamespace(request=request)
    with execution_scope(300):
        result = backend._request("GET", "/screenshot", timeout=10)
    assert result is not None
    assert budgets == [10, 10]
    assert clock.now == 23


@pytest.mark.parametrize("status", [400, 422])
def test_http_argument_rejection_is_ordinary_and_post_is_not_retried(clock, status):
    backend = OSWorldDesktopBackend()
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        reply = requests.Response()
        reply.status_code = status
        reply._content = b'{"error":"invalid argument"}'
        reply._content_consumed = True
        raise requests.HTTPError(response=reply)

    backend._session = SimpleNamespace(request=request)
    with execution_scope(300) as context, pytest.raises(DesktopNativeError, match=str(status)):
        backend.click(1, 2)
    assert context.failure is None
    assert calls == [1]


@pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
def test_http_service_rejection_is_infra_without_replaying_post(clock, status):
    backend = OSWorldDesktopBackend()
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        reply = requests.Response()
        reply.status_code = status
        reply._content = b"private diagnostics"
        reply._content_consumed = True
        raise requests.HTTPError(response=reply)

    backend._session = SimpleNamespace(request=request)
    with execution_scope(300) as context, pytest.raises(RuntimeExecutionError, match="COMMAND_TRANSPORT_FAILED"):
        backend.click(1, 2)
    assert context.failure is not None
    assert calls == [1]
