"""Regression coverage for desktop prompt envelope construction."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from nexau.archs.llm.llm_config import LLMConfig
from PIL import Image

from northcua.agentinfra.providers.provider_payloads import build_provider_request_payload
from northcua.agents.desktop.prompt import HistoryTurn, PromptEnvelopeBuilder
from northcua.core.config import PromptPolicyConfig
from northcua.core.models import ObservationArtifact, ObservationContext, WindowState


def _png_data_url() -> str:
    image = Image.new("RGB", (2, 3), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _png_data_url_with_color(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (2, 3), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def test_prompt_builder_sliding_overlay_clean_budget_matrix() -> None:
    def _turn(index: int) -> HistoryTurn:
        return HistoryTurn(
            assistant_text=f"Round {index}",
            tool_results=[
                {
                    "tool_call_id": f"call_{index}",
                    "name": "click",
                    "content": "Clicked.",
                    "raw_image_path": f"/tmp/raw-{index}.png",
                    "raw_image_url": f"raw-{index}",
                    "overlay_image_path": f"/tmp/overlay-{index}.png",
                    "overlay_image_url": f"overlay-{index}",
                    "image_path": f"/tmp/selected-{index}.png",
                    "image_url": f"selected-{index}",
                },
            ],
        )

    def _selected_images(*, budget: int, history_count: int) -> tuple[list[str], list[str | None]]:
        builder = PromptEnvelopeBuilder(
            template_root=Path("northcua/agents/desktop/prompt/templates"),
            policy=PromptPolicyConfig(screenshot_window_images=budget, history_window_rounds=10),
        )
        envelope = builder.build(
            task="Open Calculator",
            platform_name="macos",
            current_window=WindowState(),
            history_turns=[_turn(index) for index in range(1, history_count + 1)],
            current_clean_screenshot_url="current",
            tools=[],
            tool_choice="auto",
        )
        history_images: list[str | None] = []
        for message in envelope.history_messages:
            if message.role.value != "tool":
                continue
            block = message.content[0]
            if isinstance(block.content, list) and len(block.content) > 1:
                history_images.append(block.content[1].url)
            else:
                history_images.append(None)
        return envelope.observation_images, history_images

    expected = {
        (0, 0): ([], []),
        (0, 1): ([], [None]),
        (0, 2): ([], [None, None]),
        (0, 3): ([], [None, None, None]),
        (1, 0): (["current"], []),
        (1, 1): ([], ["raw-1"]),
        (1, 2): ([], [None, "raw-2"]),
        (1, 3): ([], [None, None, "raw-3"]),
        (2, 0): (["current"], []),
        (2, 1): (["overlay-1"], ["raw-1"]),
        (2, 2): ([], ["overlay-2", "raw-2"]),
        (2, 3): ([], [None, "overlay-3", "raw-3"]),
        (3, 0): (["current"], []),
        (3, 1): (["overlay-1"], ["raw-1"]),
        (3, 2): (["overlay-1"], ["overlay-2", "raw-2"]),
        (3, 3): ([], ["overlay-2", "overlay-3", "raw-3"]),
        (4, 0): (["current"], []),
        (4, 1): (["overlay-1"], ["raw-1"]),
        (4, 2): (["overlay-1"], ["overlay-2", "raw-2"]),
        (4, 3): (["overlay-1"], ["overlay-2", "overlay-3", "raw-3"]),
    }
    for (budget, history_count), images in expected.items():
        assert _selected_images(budget=budget, history_count=history_count) == images


def test_prompt_builder_transition_overlay_targets_visual_tool_result() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2, history_window_rounds=10),
    )
    history = [
        HistoryTurn(
            assistant_text="Round 1",
            tool_results=[
                {
                    "tool_call_id": "call_1_visual",
                    "name": "click",
                    "content": "Clicked.",
                    "raw_image_path": "/tmp/raw-1.png",
                    "raw_image_url": "raw-1",
                    "overlay_image_path": "/tmp/overlay-1.png",
                    "overlay_image_url": "overlay-1",
                    "image_path": "/tmp/selected-1.png",
                    "image_url": "selected-1",
                },
                {
                    "tool_call_id": "call_1_text",
                    "name": "get_window_info",
                    "content": "Window info.",
                },
            ],
        ),
        HistoryTurn(
            assistant_text="Round 2",
            tool_results=[
                {
                    "tool_call_id": "call_2",
                    "name": "type",
                    "content": "Typed.",
                    "raw_image_path": "/tmp/raw-2.png",
                    "raw_image_url": "raw-2",
                    "overlay_image_path": "/tmp/overlay-2.png",
                    "overlay_image_url": "overlay-2",
                    "image_path": "/tmp/selected-2.png",
                    "image_url": "selected-2",
                },
            ],
        ),
    ]

    envelope = builder.build(
        task="Open Calculator",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=history,
        current_clean_screenshot_url="current",
        tools=[],
        tool_choice="auto",
    )

    image_urls: list[str] = []
    text_only_result_count = 0
    for message in envelope.history_messages:
        if message.role.value != "tool":
            continue
        block = message.content[0]
        if isinstance(block.content, list) and len(block.content) > 1:
            image_urls.append(block.content[1].url)
        else:
            text_only_result_count += 1
    assert image_urls == ["overlay-2", "raw-2"]
    assert text_only_result_count == 1


def test_prompt_builder_note_turn_does_not_reset_visual_sliding_window() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2, history_window_rounds=10),
    )
    history = [
        HistoryTurn(
            assistant_text="Round 1",
            tool_results=[
                {
                    "tool_call_id": "call_1",
                    "name": "click",
                    "content": "Clicked.",
                    "raw_image_path": "/tmp/raw-1.png",
                    "raw_image_url": "raw-1",
                    "overlay_image_path": "/tmp/overlay-1.png",
                    "overlay_image_url": "overlay-1",
                    "image_path": "/tmp/selected-1.png",
                    "image_url": "selected-1",
                }
            ],
        ),
        HistoryTurn(
            assistant_text="Round 2",
            tool_results=[
                {
                    "tool_call_id": "call_note",
                    "name": "note",
                    "content": "[fact] Ada email is ada.lovelace@northcua.test.",
                    "raw_image_path": "",
                    "raw_image_url": "",
                    "overlay_image_path": "",
                    "overlay_image_url": "",
                    "image_path": "",
                    "image_url": "",
                }
            ],
        ),
    ]

    envelope = builder.build(
        task="Fill the contact sheet",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=history,
        current_clean_screenshot_url="current",
        tools=[],
        tool_choice="auto",
    )

    tool_image_urls: dict[str, str | None] = {}
    note_text = ""
    for message in envelope.history_messages:
        if message.role.value != "tool":
            continue
        block = message.content[0]
        content = block.content
        if isinstance(content, list) and len(content) > 1:
            tool_image_urls[block.tool_use_id] = content[1].url
        else:
            tool_image_urls[block.tool_use_id] = None
            if block.tool_use_id == "call_note":
                note_text = str(content)

    assert envelope.observation_images == ["overlay-1"]
    assert tool_image_urls == {"call_1": "raw-1", "call_note": None}
    assert "ada.lovelace@northcua.test" in note_text


def test_prompt_builder_note_turn_does_not_consume_image_budget() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=1, history_window_rounds=10),
    )
    history = [
        HistoryTurn(
            assistant_text="Round 1",
            tool_results=[
                {
                    "tool_call_id": "call_1",
                    "name": "click",
                    "content": "Clicked.",
                    "raw_image_path": "/tmp/raw-1.png",
                    "raw_image_url": "raw-1",
                    "overlay_image_path": "/tmp/overlay-1.png",
                    "overlay_image_url": "overlay-1",
                    "image_path": "/tmp/selected-1.png",
                    "image_url": "selected-1",
                }
            ],
        ),
        HistoryTurn(
            assistant_text="Round 2",
            tool_results=[
                {
                    "tool_call_id": "call_note",
                    "name": "note",
                    "content": "[fact] The email domain is northcua.test.",
                    "raw_image_path": "",
                    "raw_image_url": "",
                    "overlay_image_path": "",
                    "overlay_image_url": "",
                    "image_path": "",
                    "image_url": "",
                }
            ],
        ),
    ]

    envelope = builder.build(
        task="Fill the contact sheet",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=history,
        current_clean_screenshot_url="current",
        tools=[],
        tool_choice="auto",
    )

    image_urls: list[str] = []
    note_content = ""
    for message in envelope.history_messages:
        if message.role.value != "tool":
            continue
        block = message.content[0]
        content = block.content
        if isinstance(content, list) and len(content) > 1:
            image_urls.append(content[1].url)
        elif block.tool_use_id == "call_note":
            note_content = str(content)

    assert envelope.observation_images == []
    assert image_urls == ["raw-1"]
    assert "northcua.test" in note_content


def test_prompt_builder_transition_overlay_skips_intervening_note_turn() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2, history_window_rounds=10),
    )
    history = [
        HistoryTurn(
            assistant_text="Round 1",
            tool_results=[
                {
                    "tool_call_id": "call_1",
                    "name": "click",
                    "content": "Clicked.",
                    "raw_image_path": "/tmp/raw-1.png",
                    "raw_image_url": "raw-1",
                    "overlay_image_path": "/tmp/overlay-1.png",
                    "overlay_image_url": "overlay-1",
                    "image_path": "/tmp/selected-1.png",
                    "image_url": "selected-1",
                }
            ],
        ),
        HistoryTurn(
            assistant_text="Round 2",
            tool_results=[
                {
                    "tool_call_id": "call_note",
                    "name": "note",
                    "content": "[precaution] Keep the exact domain northcua.test.",
                    "raw_image_path": "",
                    "raw_image_url": "",
                    "overlay_image_path": "",
                    "overlay_image_url": "",
                    "image_path": "",
                    "image_url": "",
                }
            ],
        ),
        HistoryTurn(
            assistant_text="Round 3",
            tool_results=[
                {
                    "tool_call_id": "call_3",
                    "name": "type",
                    "content": "Typed.",
                    "raw_image_path": "/tmp/raw-3.png",
                    "raw_image_url": "raw-3",
                    "overlay_image_path": "/tmp/overlay-3.png",
                    "overlay_image_url": "overlay-3",
                    "image_path": "/tmp/selected-3.png",
                    "image_url": "selected-3",
                }
            ],
        ),
    ]

    envelope = builder.build(
        task="Fill the contact sheet",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=history,
        current_clean_screenshot_url="current",
        tools=[],
        tool_choice="auto",
    )

    tool_image_urls: dict[str, str | None] = {}
    for message in envelope.history_messages:
        if message.role.value != "tool":
            continue
        block = message.content[0]
        content = block.content
        tool_image_urls[block.tool_use_id] = content[1].url if isinstance(content, list) and len(content) > 1 else None

    assert envelope.observation_images == []
    assert tool_image_urls == {"call_1": "overlay-3", "call_note": None, "call_3": "raw-3"}


def test_prompt_builder_tool_error_turn_does_not_consume_visual_window() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2, history_window_rounds=10),
    )
    history = [
        HistoryTurn(
            assistant_text="I clicked the first field.",
            tool_results=[
                {
                    "tool_call_id": "call_1",
                    "name": "click",
                    "content": "Clicked.",
                    "raw_image_path": "/tmp/raw-1.png",
                    "raw_image_url": "raw-1",
                    "overlay_image_path": "/tmp/overlay-1.png",
                    "overlay_image_url": "overlay-1",
                    "image_path": "/tmp/selected-1.png",
                    "image_url": "selected-1",
                }
            ],
        ),
        HistoryTurn(
            assistant_text="The retry target was probably too far right.",
            tool_results=[
                {
                    "tool_call_id": "call_2_error",
                    "name": "click",
                    "content": "click failed: x=1800 is outside the screenshot width.",
                    "status": "error",
                    "is_error": True,
                    "raw_image_path": "/tmp/raw-error.png",
                    "raw_image_url": "raw-error",
                    "overlay_image_path": "/tmp/overlay-error.png",
                    "overlay_image_url": "overlay-error",
                    "image_path": "/tmp/selected-error.png",
                    "image_url": "selected-error",
                }
            ],
        ),
        HistoryTurn(
            assistant_text="I clicked the corrected target.",
            tool_results=[
                {
                    "tool_call_id": "call_3",
                    "name": "click",
                    "content": "Clicked.",
                    "raw_image_path": "/tmp/raw-3.png",
                    "raw_image_url": "raw-3",
                    "overlay_image_path": "/tmp/overlay-3.png",
                    "overlay_image_url": "overlay-3",
                    "image_path": "/tmp/selected-3.png",
                    "image_url": "selected-3",
                }
            ],
        ),
    ]

    envelope = builder.build(
        task="Fill the contact sheet",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=history,
        current_clean_screenshot_url="current",
        tools=[],
        tool_choice="auto",
    )

    tool_image_urls: dict[str, str | None] = {}
    tool_error_flags: dict[str, bool] = {}
    for message in envelope.history_messages:
        if message.role.value != "tool":
            continue
        block = message.content[0]
        content = block.content
        tool_image_urls[block.tool_use_id] = content[1].url if isinstance(content, list) and len(content) > 1 else None
        tool_error_flags[block.tool_use_id] = bool(block.is_error)

    assert envelope.observation_images == []
    assert tool_image_urls == {"call_1": "overlay-3", "call_2_error": None, "call_3": "raw-3"}
    assert tool_error_flags == {"call_1": False, "call_2_error": True, "call_3": False}


def test_prompt_builder_combines_user_text_with_separator_and_history_guidance() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2),
    )
    round1_raw = _png_data_url_with_color((0, 255, 0))
    round1_overlay = _png_data_url_with_color((255, 0, 0))
    round2_raw = _png_data_url_with_color((0, 0, 255))
    round2_overlay = _png_data_url_with_color((255, 255, 0))
    history = [
        HistoryTurn(
            assistant_text="Round 1",
            tool_results=[
                {
                    "tool_call_id": "call_1",
                    "name": "click",
                    "content": "Clicked.",
                    "raw_image_path": "/tmp/raw-1.png",
                    "raw_image_url": round1_raw,
                    "overlay_image_path": "/tmp/overlay-1.png",
                    "overlay_image_url": round1_overlay,
                    "image_path": "/tmp/overlay-1.png",
                    "image_url": round1_overlay,
                },
            ],
        ),
        HistoryTurn(
            assistant_text="Round 2",
            tool_results=[
                {
                    "tool_call_id": "call_2",
                    "name": "type",
                    "content": "Typed.",
                    "raw_image_path": "/tmp/raw-2.png",
                    "raw_image_url": round2_raw,
                    "overlay_image_path": "/tmp/overlay-2.png",
                    "overlay_image_url": round2_overlay,
                    "image_path": "/tmp/overlay-2.png",
                    "image_url": round2_overlay,
                },
            ],
        ),
    ]

    envelope = builder.build(
        task="Could you make the background of this image transparent for me?",
        platform_name="darwin",
        current_window=WindowState(),
        history_turns=history,
        current_clean_screenshot_url=round2_raw,
        tools=[],
        tool_choice="auto",
    )

    user_message = envelope.to_messages()[1]
    user_text = user_message.content[0].text
    assert "clearly blocked." in user_text
    assert "starting with `[INFEASIBLE]`" in user_text
    assert "\n\nCurrent observation:" in user_text
    assert "Use the most recent tool-result screenshot as the freshest visual state." in user_text
    assert "No current screenshot is available in this turn." not in user_text


def test_text_only_note_keeps_initial_grounding_until_a_visual_tool_returns() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=1),
    )
    current = _png_data_url_with_color((12, 34, 56))
    note = HistoryTurn(
        assistant_text="Remember", tool_results=[{"tool_call_id": "note-1", "name": "note", "content": "Saved."}]
    )
    envelope = builder.build(
        task="Fill the window",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=[note],
        current_clean_screenshot_url=current,
        tools=[],
        tool_choice="auto",
    )
    assert envelope.observation_images == [current]
    assert note.tool_results == [{"tool_call_id": "note-1", "name": "note", "content": "Saved."}]
