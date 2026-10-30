"""History overlay scroll semantics regressions."""

from __future__ import annotations

from northcua.agentkernel.overlay.history_overlay.badges import OverlayBadgeTextFormatter
from northcua.agentkernel.overlay.history_overlay.scroll import HistoryActionOverlayScrollDrawingHelper


def test_scroll_badge_matches_pyautogui_vertical_semantics() -> None:
    formatter = OverlayBadgeTextFormatter(badges_enabled=True, max_badge_text=80)

    assert formatter.scroll_badge_text(delta_x=0, delta_y=3) == "Scroll Up"
    assert formatter.scroll_badge_text(delta_x=0, delta_y=-3) == "Scroll Down"


def test_scroll_overlay_direction_matches_pyautogui_vertical_semantics() -> None:
    assert HistoryActionOverlayScrollDrawingHelper._scroll_directions(delta_x=0, delta_y=3) == ["up"]
    assert HistoryActionOverlayScrollDrawingHelper._scroll_directions(delta_x=0, delta_y=-3) == ["down"]
