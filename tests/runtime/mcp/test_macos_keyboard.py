"""macOS event-state regression checks without posting physical input."""

from types import SimpleNamespace

import pytest

from northcua.runtime.mcp.desktop_native._macos_keyboard import MacOSKeyboardAdapter


class QuartzFixture:
    kCGEventFlagMaskCommand = 1
    kCGEventFlagMaskControl = 2
    kCGEventFlagMaskAlternate = 4
    kCGEventFlagMaskShift = 8
    kCGHIDEventTap = 0

    def __init__(self):
        self.events = []

    def CGEventCreateKeyboardEvent(self, source, code, down):
        return {"code": code, "down": down, "flags": 999, "text": "stale"}

    def CGEventSetFlags(self, event, flags):
        event["flags"] = flags

    def CGEventKeyboardSetUnicodeString(self, event, length, text):
        event["text"] = text

    def CGEventPost(self, tap, event):
        self.events.append(event.copy())


@pytest.fixture
def keyboard(monkeypatch):
    monkeypatch.setattr("northcua.runtime.mcp.desktop_native._macos_keyboard.time.sleep", lambda _: None)
    q = QuartzFixture()
    module = SimpleNamespace(
        PAUSE=0, platformModule=SimpleNamespace(keyboardMapping={"command": 55, "v": 9, "enter": 36})
    )
    return MacOSKeyboardAdapter(module, q), q


def test_paste_carries_command_flag_on_both_letter_events(keyboard):
    adapter, q = keyboard
    adapter.hotkey("command", "v")
    assert q.events == [
        {"code": 55, "down": True, "flags": 1, "text": ""},
        {"code": 9, "down": True, "flags": 1, "text": "v"},
        {"code": 9, "down": False, "flags": 1, "text": "v"},
        {"code": 55, "down": False, "flags": 0, "text": ""},
    ]


def test_return_clears_inherited_flags_and_character_payload(keyboard):
    adapter, q = keyboard
    adapter.press("enter")
    assert q.events == [
        {"code": 36, "down": True, "flags": 0, "text": "\r"},
        {"code": 36, "down": False, "flags": 0, "text": "\r"},
    ]


def test_failed_chord_releases_already_pressed_modifier(keyboard):
    adapter, q = keyboard
    with pytest.raises(ValueError, match="Unsupported"):
        adapter.hotkey("command", "invalid")
    assert q.events[-1] == {"code": 55, "down": False, "flags": 0, "text": ""}


def test_explicit_modifier_hold_survives_a_separate_press(keyboard):
    adapter, q = keyboard
    adapter.keyDown("command")
    adapter.press("v")
    adapter.keyUp("command")
    assert [event["flags"] for event in q.events] == [1, 1, 1, 0]
