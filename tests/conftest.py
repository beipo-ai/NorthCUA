"""Shared pytest fixtures for platform-independent runtime contract tests."""

from __future__ import annotations

import pytest

from northcua.runtime.mcp.desktop_native.backend import PyAutoGuiDesktopBackend
from northcua.runtime.mcp.desktop_native.state import DesktopNativeServerState


@pytest.fixture
def desktop_native_test_session(monkeypatch: pytest.MonkeyPatch):
    """Build desktop MCP surfaces without requiring the test host to run X11."""

    monkeypatch.setattr(PyAutoGuiDesktopBackend, "_ensure_supported_environment", lambda _self: None)
    state = DesktopNativeServerState()
    state.reset()
    yield
    state.reset()
