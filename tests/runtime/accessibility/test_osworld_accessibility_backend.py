from __future__ import annotations

from northcua.core.enums import AccessibilityCaptureStatus, AccessibilityScope
from northcua.runtime.accessibility import AccessibilityCaptureRequest
from northcua.runtime.accessibility.osworld import OSWorldAccessibilityBackend


class _FakeAccessibilityResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"AT": "<root><button name='OK'/></root>"}


class _FakeAccessibilitySession:
    def __init__(self) -> None:
        self.trust_env = True
        self.calls: list[tuple[str, float]] = []

    def get(self, url: str, *, timeout: float):
        self.calls.append((url, timeout))
        return _FakeAccessibilityResponse()


def test_osworld_accessibility_backend_fetches_raw_payload(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_OSWORLD_VM_IP", "10.0.0.30")
    monkeypatch.setenv("NORTHCUA_OSWORLD_SERVER_PORT", "7100")
    monkeypatch.setenv("NORTHCUA_OSWORLD_COMMAND_TIMEOUT_SECONDS", "9")
    backend = OSWorldAccessibilityBackend()
    fake_session = _FakeAccessibilitySession()
    backend.session = fake_session

    snapshot = backend.capture(AccessibilityCaptureRequest(scope=AccessibilityScope.DESKTOP, include_raw_payload=True))

    assert snapshot.status == AccessibilityCaptureStatus.SUCCESS
    assert snapshot.raw_payload == "<root><button name='OK'/></root>"
    assert fake_session.calls == [("http://10.0.0.30:7100/accessibility", 9.0)]


def test_osworld_accessibility_backend_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("NORTHCUA_OSWORLD_ENABLE_A11Y", "false")

    backend = OSWorldAccessibilityBackend()
    capabilities = backend.capabilities()
    snapshot = backend.capture(AccessibilityCaptureRequest(scope=AccessibilityScope.DESKTOP))

    assert capabilities.supported_scopes == []
    assert snapshot.status == AccessibilityCaptureStatus.DISABLED
