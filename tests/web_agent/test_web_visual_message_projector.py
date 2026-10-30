"""Prompt message projection tests for Web/Desktop visual-window parity."""

from __future__ import annotations

from pathlib import Path

from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock
from PIL import Image

from northcua.agents.web.domain.evaluation_artifacts import WebToolResultArtifact
from northcua.agents.web.prompt.visual_window import WebPromptVisualHistoryFactory, WebPromptVisualMessageProjector
from northcua.core.config import PromptPolicyConfig
from northcua.core.image_payloads import image_data_url_from_path


def test_w3_projects_previous_action_overlay_latest_clean_and_latest_som(tmp_path: Path) -> None:
    image_paths = {
        name: _write_png(tmp_path / f"{name}.png", color)
        for name, color in {
            "raw1": "white",
            "overlay1": "red",
            "som1": "blue",
            "raw2": "green",
            "overlay2": "yellow",
            "som2": "purple",
        }.items()
    }
    history_messages = [
        _tool_result_message("call-1", old_image_url=image_data_url_from_path(str(image_paths["som1"]))),
        _tool_result_message("call-2", old_image_url=image_data_url_from_path(str(image_paths["som2"]))),
    ]
    visual_history = [
        WebPromptVisualHistoryFactory().build_turn(
            session_step=1,
            round_num=1,
            tool_artifacts=[
                _artifact(
                    call_id="call-1",
                    raw=image_paths["raw1"],
                    overlay=image_paths["overlay1"],
                    som=image_paths["som1"],
                )
            ],
        ),
        WebPromptVisualHistoryFactory().build_turn(
            session_step=2,
            round_num=1,
            tool_artifacts=[
                _artifact(
                    call_id="call-2",
                    raw=image_paths["raw2"],
                    overlay=image_paths["overlay2"],
                    som=image_paths["som2"],
                )
            ],
        ),
    ]

    result = WebPromptVisualMessageProjector(PromptPolicyConfig(screenshot_window_images=3)).apply(
        history_messages=history_messages,
        visual_history_turns=[turn for turn in visual_history if turn is not None],
        current_image_url="",
        allow_current_image=False,
        current_image_omitted_reason="",
    )

    call1_images = _tool_result_image_urls(result.history_messages[0])
    call2_images = _tool_result_image_urls(result.history_messages[1])
    assert call1_images == [image_data_url_from_path(str(image_paths["overlay2"]))]
    assert call2_images == [
        image_data_url_from_path(str(image_paths["raw2"])),
        image_data_url_from_path(str(image_paths["som2"])),
    ]
    assert "Action-overlay history image" in _tool_result_text(result.history_messages[0])
    assert "not current page state" in _tool_result_text(result.history_messages[0])
    assert "Clean page-state image" in _tool_result_text(result.history_messages[1])
    assert "Set-of-Mark" in _tool_result_text(result.history_messages[1])


def test_w3_replaces_initial_image_with_first_action_overlay(tmp_path: Path) -> None:
    image_paths = {
        name: _write_png(tmp_path / f"{name}.png", color)
        for name, color in {
            "initial": "gray",
            "raw1": "white",
            "overlay1": "red",
            "som1": "blue",
        }.items()
    }
    history_messages = [
        Message(
            role=Role.USER,
            content=[
                TextBlock(text="Task:\nOpen the page.\n\nInitial page observation:\n..."),
                ImageBlock(url=image_data_url_from_path(str(image_paths["initial"]))),
            ],
        ),
        _tool_result_message("call-1", old_image_url=image_data_url_from_path(str(image_paths["som1"]))),
    ]
    visual_turn = WebPromptVisualHistoryFactory().build_turn(
        session_step=1,
        round_num=1,
        tool_artifacts=[
            _artifact(
                call_id="call-1",
                raw=image_paths["raw1"],
                overlay=image_paths["overlay1"],
                som=image_paths["som1"],
            )
        ],
    )

    result = WebPromptVisualMessageProjector(PromptPolicyConfig(screenshot_window_images=3)).apply(
        history_messages=history_messages,
        visual_history_turns=[visual_turn] if visual_turn is not None else [],
        current_image_url="",
        allow_current_image=False,
        current_image_omitted_reason="",
    )

    assert _top_level_image_urls(result.history_messages[0]) == [image_data_url_from_path(str(image_paths["overlay1"]))]
    assert "Action-overlay history image" in _top_level_text(result.history_messages[0])
    assert "refs in it are expired" in _top_level_text(result.history_messages[0])
    assert _tool_result_image_urls(result.history_messages[1]) == [
        image_data_url_from_path(str(image_paths["raw1"])),
        image_data_url_from_path(str(image_paths["som1"])),
    ]


def test_current_observation_image_fills_latest_clean_when_history_lacks_raw(tmp_path: Path) -> None:
    image_paths = {
        name: _write_png(tmp_path / f"{name}.png", color)
        for name, color in {
            "overlay1": "red",
            "som1": "blue",
            "current": "green",
        }.items()
    }
    history_messages = [
        Message(role=Role.USER, content=[TextBlock(text="Task:\nRefresh after stale observation.")]),
        _tool_result_message("call-1", old_image_url=image_data_url_from_path(str(image_paths["som1"]))),
    ]
    visual_turn = WebPromptVisualHistoryFactory().build_turn(
        session_step=1,
        round_num=1,
        tool_artifacts=[
            _artifact(
                call_id="call-1",
                raw=None,
                overlay=image_paths["overlay1"],
                som=image_paths["som1"],
            )
        ],
    )

    result = WebPromptVisualMessageProjector(PromptPolicyConfig(screenshot_window_images=3)).apply(
        history_messages=history_messages,
        visual_history_turns=[visual_turn] if visual_turn is not None else [],
        current_image_url=image_data_url_from_path(str(image_paths["current"])),
        allow_current_image=True,
        current_image_omitted_reason="",
    )

    assert _top_level_image_urls(result.history_messages[0]) == [
        image_data_url_from_path(str(image_paths["overlay1"])),
        image_data_url_from_path(str(image_paths["current"])),
    ]
    assert _tool_result_image_urls(result.history_messages[1]) == [image_data_url_from_path(str(image_paths["som1"]))]


def test_current_observation_image_does_not_duplicate_latest_raw(tmp_path: Path) -> None:
    image_paths = {
        name: _write_png(tmp_path / f"{name}.png", color)
        for name, color in {
            "raw1": "white",
            "overlay1": "red",
            "som1": "blue",
            "current": "green",
        }.items()
    }
    history_messages = [
        Message(role=Role.USER, content=[TextBlock(text="Task:\nUse latest raw.")]),
        _tool_result_message("call-1", old_image_url=image_data_url_from_path(str(image_paths["som1"]))),
    ]
    visual_turn = WebPromptVisualHistoryFactory().build_turn(
        session_step=1,
        round_num=1,
        tool_artifacts=[
            _artifact(
                call_id="call-1",
                raw=image_paths["raw1"],
                overlay=image_paths["overlay1"],
                som=image_paths["som1"],
            )
        ],
    )

    result = WebPromptVisualMessageProjector(PromptPolicyConfig(screenshot_window_images=3)).apply(
        history_messages=history_messages,
        visual_history_turns=[visual_turn] if visual_turn is not None else [],
        current_image_url=image_data_url_from_path(str(image_paths["current"])),
        allow_current_image=True,
        current_image_omitted_reason="",
    )

    assert _top_level_image_urls(result.history_messages[0]) == [image_data_url_from_path(str(image_paths["overlay1"]))]
    assert image_data_url_from_path(str(image_paths["current"])) not in _all_image_urls(result.history_messages)


def test_failed_tool_result_does_not_pin_latest_som(tmp_path: Path) -> None:
    som_path = _write_png(tmp_path / "failed_som.png", "blue")
    history_messages = [_tool_result_message("call-1", old_image_url=image_data_url_from_path(str(som_path)))]
    visual_turn = WebPromptVisualHistoryFactory().build_turn(
        session_step=1,
        round_num=1,
        tool_artifacts=[
            _artifact(
                call_id="call-1",
                raw=None,
                overlay=None,
                som=som_path,
                status="error",
                is_error=True,
            )
        ],
    )

    result = WebPromptVisualMessageProjector(PromptPolicyConfig(screenshot_window_images=2)).apply(
        history_messages=history_messages,
        visual_history_turns=[visual_turn] if visual_turn is not None else [],
        current_image_url="",
        allow_current_image=False,
        current_image_omitted_reason="",
    )

    assert _all_image_urls(result.history_messages) == []


def test_recovered_error_projects_latest_clean_and_som(tmp_path: Path) -> None:
    image_paths = {
        name: _write_png(tmp_path / f"{name}.png", color)
        for name, color in {
            "raw1": "white",
            "som1": "blue",
        }.items()
    }
    history_messages = [
        _tool_result_message("call-1", old_image_url=image_data_url_from_path(str(image_paths["som1"])))
    ]
    artifact = _artifact(
        call_id="call-1",
        raw=image_paths["raw1"],
        overlay=None,
        som=image_paths["som1"],
        status="error",
        is_error=True,
    )
    artifact.metadata["observation_recovered"] = True
    visual_turn = WebPromptVisualHistoryFactory().build_turn(
        session_step=1,
        round_num=1,
        tool_artifacts=[artifact],
    )

    result = WebPromptVisualMessageProjector(PromptPolicyConfig(screenshot_window_images=2)).apply(
        history_messages=history_messages,
        visual_history_turns=[visual_turn] if visual_turn is not None else [],
        current_image_url="",
        allow_current_image=False,
        current_image_omitted_reason="",
    )

    assert _tool_result_image_urls(result.history_messages[0]) == [
        image_data_url_from_path(str(image_paths["raw1"])),
        image_data_url_from_path(str(image_paths["som1"])),
    ]
    assert "Clean page-state image" in _tool_result_text(result.history_messages[0])
    assert "Set-of-Mark" in _tool_result_text(result.history_messages[0])


def test_refs_disabled_does_not_append_actionable_som(tmp_path: Path) -> None:
    image_paths = {
        name: _write_png(tmp_path / f"{name}.png", color)
        for name, color in {
            "raw1": "white",
            "overlay1": "red",
            "som1": "blue",
        }.items()
    }
    history_messages = [
        Message(role=Role.USER, content=[TextBlock(text="Task:\nRefs disabled.")]),
        _tool_result_message("call-1", old_image_url=image_data_url_from_path(str(image_paths["som1"]))),
    ]
    visual_turn = WebPromptVisualHistoryFactory().build_turn(
        session_step=1,
        round_num=1,
        tool_artifacts=[
            _artifact(
                call_id="call-1",
                raw=image_paths["raw1"],
                overlay=image_paths["overlay1"],
                som=image_paths["som1"],
            )
        ],
    )

    result = WebPromptVisualMessageProjector(
        PromptPolicyConfig(screenshot_window_images=2, ref_snapshot_window_observations=0)
    ).apply(
        history_messages=history_messages,
        visual_history_turns=[visual_turn] if visual_turn is not None else [],
        current_image_url="",
        allow_current_image=False,
        current_image_omitted_reason="",
    )

    assert _tool_result_image_urls(result.history_messages[1]) == [image_data_url_from_path(str(image_paths["raw1"]))]
    assert image_data_url_from_path(str(image_paths["som1"])) not in _all_image_urls(result.history_messages)


def _artifact(
    *,
    call_id: str,
    raw: Path | None,
    overlay: Path | None,
    som: Path | None,
    status: str = "ok",
    is_error: bool = False,
) -> WebToolResultArtifact:
    return WebToolResultArtifact(
        tool_name="web_click_ref",
        tool_call_id=call_id,
        action="click_ref",
        status=status,
        is_error=is_error,
        text="Clicked.",
        raw_image_path=str(raw) if raw is not None else "",
        action_overlay_image_path=str(overlay) if overlay is not None else "",
        som_image_path=str(som) if som is not None else "",
        selected_image_path=str(som or overlay or raw or ""),
    )


def _tool_result_message(call_id: str, *, old_image_url: str) -> Message:
    return Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id=call_id,
                content=[TextBlock(text=f"Tool result {call_id}"), ImageBlock(url=old_image_url)],
            )
        ],
    )


def _tool_result_image_urls(message: Message) -> list[str]:
    block = message.content[0]
    assert isinstance(block, ToolResultBlock)
    assert isinstance(block.content, list)
    return [str(item.url) for item in block.content if isinstance(item, ImageBlock)]


def _tool_result_text(message: Message) -> str:
    block = message.content[0]
    assert isinstance(block, ToolResultBlock)
    assert isinstance(block.content, list)
    return "\n".join(str(item.text) for item in block.content if isinstance(item, TextBlock))


def _top_level_image_urls(message: Message) -> list[str]:
    return [str(item.url) for item in message.content if isinstance(item, ImageBlock)]


def _top_level_text(message: Message) -> str:
    return "\n".join(str(item.text) for item in message.content if isinstance(item, TextBlock))


def _all_image_urls(messages: list[Message]) -> list[str]:
    urls: list[str] = []
    for message in messages:
        for block in message.content:
            if isinstance(block, ImageBlock):
                urls.append(str(block.url))
            if isinstance(block, ToolResultBlock) and isinstance(block.content, list):
                urls.extend(str(item.url) for item in block.content if isinstance(item, ImageBlock))
    return urls


def _write_png(path: Path, color: str) -> Path:
    Image.new("RGB", (24, 16), color=color).save(path)
    return path


import pytest


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("has_raw", [False, True])
def test_production_projector_total_image_budget(tmp_path, budget, has_raw):
    raw = _write_png(tmp_path / "raw.png", "white")
    overlay = _write_png(tmp_path / "overlay.png", "red")
    som = _write_png(tmp_path / "som.png", "blue")
    turn = WebPromptVisualHistoryFactory().build_turn(
        session_step=1,
        round_num=1,
        tool_artifacts=[_artifact(call_id="call", raw=raw if has_raw else None, overlay=overlay, som=som)],
    )
    result = WebPromptVisualMessageProjector(PromptPolicyConfig(screenshot_window_images=budget)).apply(
        history_messages=[
            Message(role=Role.USER, content=[TextBlock(text="Task: test")]),
            _tool_result_message("call", old_image_url="old"),
        ],
        visual_history_turns=[turn],
        current_image_url=image_data_url_from_path(str(raw)),
        allow_current_image=True,
        current_image_omitted_reason="",
    )
    images = [url for message in result.history_messages for url in _top_level_image_urls(message)]
    images += [
        url
        for message in result.history_messages
        if message.content and isinstance(message.content[0], ToolResultBlock)
        for url in _tool_result_image_urls(message)
    ]
    assert len(images) <= budget
    if budget:
        assert image_data_url_from_path(str(raw)) in images
    assert (image_data_url_from_path(str(som)) in images) == (budget >= 2)
