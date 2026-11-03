"""Regression coverage for desktop prompt envelope construction."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from nexau.archs.llm.llm_config import LLMConfig
from PIL import Image

from northcua.agentinfra.providers.provider_payloads import build_provider_request_payload
from northcua.agents.desktop.prompt import HistoryTurn, PromptEnvelopeBuilder
from northcua.agents.desktop.services.history_replay import HistoryReplayService
from northcua.core.config import PromptPolicyConfig
from northcua.core.coordinates import CoordinatePolicy
from northcua.core.enums import CoordinateSpace, TaskStatus
from northcua.core.image_payloads import image_data_url_from_path
from northcua.core.models import (
    AssistantToolCallRecord,
    ObservationArtifact,
    ObservationContext,
    StepLogRecord,
    ToolResultArtifact,
    WindowState,
)


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


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (2, 3), color=color).save(path)


def test_history_turn_preserves_openai_function_call_arguments() -> None:
    turn = HistoryTurn(
        assistant_text="I will click.",
        assistant_tool_calls=[
            {
                "id": "call_1",
                "type": "function_call",
                "function": {"name": "click", "arguments": '{"x": 10, "y": 20}'},
            }
        ],
    )

    assistant_message = turn.to_messages()[0]
    tool_use = assistant_message.content[1]

    assert tool_use.name == "click"
    assert tool_use.input == {"x": 10, "y": 20}
    assert tool_use.raw_input == '{"x": 10, "y": 20}'


def test_history_replay_rehydrates_selected_images_from_paths_after_payload_release(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.png"
    overlay_path = tmp_path / "overlay.png"
    _write_png(raw_path, (0, 255, 0))
    _write_png(overlay_path, (255, 0, 0))
    step = StepLogRecord(
        session_id="sess-paths",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Open Calculator",
        assistant_text="I clicked the launcher.",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call_1",
                name="click",
                content="Clicked.",
                status="success",
                raw_image_path=str(raw_path),
                raw_image_url=_png_data_url_with_color((0, 255, 0)),
                overlay_image_path=str(overlay_path),
                overlay_image_url=_png_data_url_with_color((255, 0, 0)),
                image_path=str(overlay_path),
                image_url=_png_data_url_with_color((255, 0, 0)),
            )
        ],
    )
    step.release_retained_payloads()

    history = HistoryReplayService().build_turns([step], history_window_rounds=5)
    assert history[0].tool_results[0]["raw_image_url"] == ""
    assert history[0].tool_results[0]["overlay_image_url"] == ""

    envelope = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2),
    ).build(
        task="Open Calculator",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=history,
        current_clean_screenshot_url="",
        tools=[],
        tool_choice="auto",
    )

    assert envelope.observation_images[0].startswith("data:image/png;base64,")
    tool_messages = [message for message in envelope.history_messages if message.role.value == "tool"]
    block = tool_messages[0].content[0]
    assert block.content[1].url.startswith("data:image/png;base64,")


def test_history_replay_rehydrates_released_images_without_assigning_them_to_note(tmp_path: Path) -> None:
    raw_1 = tmp_path / "raw-1.png"
    overlay_1 = tmp_path / "overlay-1.png"
    raw_3 = tmp_path / "raw-3.png"
    overlay_3 = tmp_path / "overlay-3.png"
    _write_png(raw_1, (0, 255, 0))
    _write_png(overlay_1, (255, 0, 0))
    _write_png(raw_3, (0, 0, 255))
    _write_png(overlay_3, (255, 255, 0))
    steps = [
        StepLogRecord(
            session_id="sess-note-paths",
            session_step=1,
            round_step=1,
            round_num=1,
            status=TaskStatus.CONTINUE,
            request="Fill the contact sheet",
            assistant_text="I clicked the first field.",
            tool_results=[
                ToolResultArtifact(
                    tool_call_id="call_1",
                    name="click",
                    content="Clicked.",
                    status="success",
                    raw_image_path=str(raw_1),
                    raw_image_url=image_data_url_from_path(str(raw_1)),
                    overlay_image_path=str(overlay_1),
                    overlay_image_url=image_data_url_from_path(str(overlay_1)),
                    image_path=str(overlay_1),
                    image_url=image_data_url_from_path(str(overlay_1)),
                )
            ],
        ),
        StepLogRecord(
            session_id="sess-note-paths",
            session_step=2,
            round_step=2,
            round_num=1,
            status=TaskStatus.CONTINUE,
            request="Fill the contact sheet",
            assistant_text="I should preserve the domain before typing.",
            tool_results=[
                ToolResultArtifact(
                    tool_call_id="call_note",
                    name="note",
                    content="[fact] The email domain is northcua.test.",
                    status="recorded",
                )
            ],
        ),
        StepLogRecord(
            session_id="sess-note-paths",
            session_step=3,
            round_step=3,
            round_num=1,
            status=TaskStatus.CONTINUE,
            request="Fill the contact sheet",
            assistant_text="I typed the domain.",
            tool_results=[
                ToolResultArtifact(
                    tool_call_id="call_3",
                    name="type",
                    content="Typed.",
                    status="success",
                    raw_image_path=str(raw_3),
                    raw_image_url=image_data_url_from_path(str(raw_3)),
                    overlay_image_path=str(overlay_3),
                    overlay_image_url=image_data_url_from_path(str(overlay_3)),
                    image_path=str(overlay_3),
                    image_url=image_data_url_from_path(str(overlay_3)),
                )
            ],
        ),
    ]
    for step in steps:
        step.release_retained_payloads()

    history = HistoryReplayService().build_turns(steps, history_window_rounds=10)
    assert all(not result.get("image_url") for turn in history for result in turn.tool_results)

    envelope = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2),
    ).build(
        task="Fill the contact sheet",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=history,
        current_clean_screenshot_url="",
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
    assert tool_image_urls == {
        "call_1": image_data_url_from_path(str(overlay_3)),
        "call_note": None,
        "call_3": image_data_url_from_path(str(raw_3)),
    }


def test_history_turn_mentions_omitted_path_only_tool_image() -> None:
    turn = HistoryTurn(
        assistant_text="I clicked.",
        tool_results=[
            {
                "tool_call_id": "call_1",
                "content": "Clicked.",
                "raw_image_path": "/tmp/raw.png",
                "raw_image_url": "",
                "image_path": "",
                "image_url": "",
            }
        ],
    )

    tool_message = turn.to_messages()[1]
    content = tool_message.content[0].content

    assert isinstance(content, str)
    assert "Clicked." in content
    assert "image omitted from this replay turn" in content


def test_history_turn_replays_responses_reasoning_before_assistant_item() -> None:
    turn = HistoryTurn(
        assistant_text="I will click.",
        reasoning="The target is centered.",
        reasoning_encrypted_content="enc-reasoning",
        assistant_tool_calls=[
            {
                "id": "call_1",
                "type": "function_call",
                "function": {"name": "click", "arguments": '{"x": 10, "y": 20}'},
            }
        ],
    )

    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="gpt-5.5",
            base_url="https://api.openai.com/v1",
            api_key="key",
            api_type="openai_responses",
            max_tokens=128,
        ),
        messages=turn.to_messages(),
    )

    items = prepared.request_payload["input"]
    assert items[0] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "The target is centered."}],
        "encrypted_content": "enc-reasoning",
    }
    assert items[1]["type"] == "message"
    assert items[1]["role"] == "assistant"
    assert items[2]["type"] == "function_call"


def test_history_turn_replays_chat_reasoning_content_when_policy_enabled() -> None:
    turn = HistoryTurn(
        assistant_text="I will click.",
        reasoning="The target is centered.",
        reasoning_encrypted_content="enc-reasoning",
        assistant_tool_calls=[
            {
                "id": "call_1",
                "type": "function_call",
                "function": {"name": "click", "arguments": '{"x": 10, "y": 20}'},
            }
        ],
    )

    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="gpt-5.5",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="openai_chat_completion",
            max_tokens=128,
            reasoning_replay_policy="chat_reasoning_content",
        ),
        messages=turn.to_messages(),
    )

    assistant = prepared.request_payload["messages"][0]

    assert assistant["role"] == "assistant"
    assert assistant["content"] == "I will click."
    assert assistant["reasoning_content"] == "The target is centered."
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert "reasoning" not in assistant


def test_history_replay_service_carries_reasoning_for_responses_replay() -> None:
    service = HistoryReplayService()
    step = StepLogRecord(
        session_id="session-a",
        session_step=1,
        round_num=1,
        round_step=1,
        status=TaskStatus.CONTINUE,
        request="Click center",
        llm_reasoning="Need to click center.",
        llm_reasoning_encrypted_content="enc-123",
        assistant_text="I will click.",
        assistant_tool_calls=[AssistantToolCallRecord(id="call_1", name="click", arguments='{"x": 10, "y": 20}')],
    )

    turn = service.build_turns([step], history_window_rounds=1)[0]

    assert turn.reasoning == "Need to click center."
    assert turn.reasoning_encrypted_content == "enc-123"


def test_history_turn_mentions_when_prompt_image_budget_omits_tool_image() -> None:
    turn = HistoryTurn(
        assistant_text="I clicked.",
        tool_results=[
            {
                "tool_call_id": "call_1",
                "content": "Clicked.",
                "raw_image_path": "/tmp/raw.png",
                "raw_image_url": "data:image/png;base64,AAAA",
                "image_path": "",
                "image_url": "",
            }
        ],
    )

    tool_message = turn.to_messages()[1]
    content = tool_message.content[0].content

    assert isinstance(content, str)
    assert "Clicked." in content
    assert "image omitted from this replay turn" in content


def test_history_turn_does_not_add_image_omission_notice_to_text_only_tools() -> None:
    turn = HistoryTurn(
        assistant_text="I will note.",
        tool_results=[{"tool_call_id": "call_note", "content": "Remember this."}],
    )

    tool_message = turn.to_messages()[1]
    content = tool_message.content[0].content

    assert content == "Remember this."


def test_history_turn_replays_claude_tool_use_arguments_in_anthropic_payload() -> None:
    turn = HistoryTurn(
        assistant_text="I will click.",
        assistant_tool_calls=[
            {
                "id": "toolu_1",
                "type": "tool_use",
                "name": "click",
                "arguments": '{"x": 42, "y": 99}',
            }
        ],
        tool_results=[{"tool_call_id": "toolu_1", "content": "Clicked at (42, 99)."}],
    )

    messages = turn.to_messages()
    tool_use = messages[0].content[1]
    prepared = build_provider_request_payload(
        llm_config=LLMConfig(
            model="claude-opus-4-6",
            base_url="https://gateway.example.invalid/v1",
            api_key="key",
            api_type="anthropic_chat_completion",
            max_tokens=128,
        ),
        messages=messages,
    )
    tool_use_payload = prepared.request_payload["messages"][0]["content"][1]

    assert tool_use.name == "click"
    assert tool_use.input == {"x": 42, "y": 99}
    assert tool_use.raw_input == '{"x": 42, "y": 99}'
    assert tool_use_payload == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "click",
        "input": {"x": 42, "y": 99},
    }


def test_prompt_builder_keeps_initial_image_only_on_first_round() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2),
    )

    envelope = builder.build(
        task="Open Calculator",
        platform_name="macos",
        current_window=WindowState(name="Calculator", type="window", rect=[0, 0, 100, 100]),
        history_turns=[],
        current_clean_screenshot_url=_png_data_url(),
        tools=[],
        tool_choice="auto",
    )

    assert len(envelope.observation_images) == 1
    assert len(envelope.to_messages()) == 2
    assert "Before sending focused input" in envelope.system_prompt


def test_prompt_builder_injects_normalized_1000_coordinate_policy() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2),
        coordinate_policy=CoordinatePolicy(input_space=CoordinateSpace.NORMALIZED_1000),
    )

    envelope = builder.build(
        task="Click the center",
        platform_name="macos",
        current_window=WindowState(name="Desktop", type="window", rect=[0, 0, 100, 100]),
        history_turns=[],
        current_clean_screenshot_url=_png_data_url(),
        tools=[],
        tool_choice="auto",
    )

    assert "Return x,y normalized to [0,1000]" in envelope.system_prompt
    assert "(1000, 1000)" in envelope.system_prompt
    assert "Scroll `delta_x` and `delta_y` are wheel deltas" in envelope.system_prompt
    assert "Record the observed failure and a prevention reminder" in envelope.system_prompt
    assert "If terminal use is required, be careful with large direct pastes" in envelope.system_prompt
    assert "Unsafe Paste prompts" in envelope.system_prompt
    assert "opening a file, writing the content there" in envelope.system_prompt
    assert "GUI operations are more transparent" in envelope.system_prompt
    assert "not to replace normal GUI interaction" in envelope.system_prompt
    assert "you may emit multiple tool calls in the same response" in envelope.system_prompt
    assert "CUA Driver background tools" not in envelope.system_prompt
    assert "cua_get_window_state" not in envelope.system_prompt
    assert "On macOS prefer `cmd`" not in envelope.system_prompt
    assert "`type`" not in envelope.system_prompt
    assert "`keypress`" not in envelope.system_prompt
    assert "Do not return tool calls alone" in envelope.system_prompt
    assert "Coordinate origin is the top-left corner" in envelope.system_prompt
    assert "OSWorld benchmark environment:" not in envelope.system_prompt
    assert "Current window:" not in envelope.observation_user_text
    assert "Desktop environment: macos" in envelope.system_prompt
    assert "Home directory:" not in envelope.system_prompt


def test_prompt_builder_marks_remote_home_directory_unavailable() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=0),
    )

    envelope = builder.build(
        task="Open Calculator",
        platform_name="linux",
        current_window=WindowState(),
        history_turns=[],
        observation_context=ObservationContext(
            platform_name="remote",
        ),
        tools=[],
        tool_choice="auto",
    )

    assert "Desktop environment: remote" in envelope.system_prompt
    assert "Home directory: unavailable" in envelope.system_prompt
    assert (
        builder._home_directory(
            ObservationContext(
                platform_name="remote-controller",
            )
        )
        == "unavailable"
    )


def test_prompt_builder_uses_backend_metadata_without_benchmark_assumptions() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=0),
    )

    generic = builder.build(
        task="Open Chrome",
        platform_name="linux",
        current_window=WindowState(),
        history_turns=[],
        observation_context=ObservationContext(platform_name="linux"),
        tools=[],
        tool_choice="auto",
    )
    osworld = builder.build(
        task="Open Chrome",
        platform_name="osworld",
        current_window=WindowState(),
        history_turns=[],
        observation_context=ObservationContext(platform_name="osworld"),
        tools=[],
        tool_choice="auto",
    )

    assert "OSWorld benchmark environment:" not in generic.system_prompt
    assert "OSWorld benchmark environment:" not in generic.observation_user_text
    assert "OSWorld benchmark environment:" not in osworld.system_prompt
    assert "OSWorld benchmark environment:" not in osworld.observation_user_text
    assert "Ubuntu 22.04" not in osworld.system_prompt
    assert "Home directory:" not in osworld.system_prompt
    assert "Desktop environment: osworld" in osworld.system_prompt
    assert "Do not ask the user for clarification." in generic.system_prompt
    assert "Do not ask the user for clarification." in osworld.system_prompt
    assert "emit `[INFEASIBLE]` and stop" in osworld.system_prompt


def test_prompt_builder_keeps_cua_target_metadata_out_of_user_text() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=0),
    )

    envelope = builder.build(
        task="Read Calculator",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=[],
        observation_context=ObservationContext(
            platform_name="macos",
            observation_metadata={"cua_target": {"pid": 123, "window_id": 456}},
        ),
        tools=[],
        tool_choice="auto",
    )

    assert "CUA background target" not in envelope.observation_user_text
    assert "pid=123" not in envelope.observation_user_text
    assert "window_id=456" not in envelope.observation_user_text


def test_prompt_builder_does_not_inject_instruction_pack_outside_cua_context() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=0),
    )

    envelope = builder.build(
        task="Use Calculator",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=[],
        observation_context=ObservationContext(
            platform_name="macos",
            observation_metadata={"app_name": "Calculator"},
        ),
        tools=[],
        tool_choice="auto",
    )

    assert "App instruction pack:" not in envelope.observation_user_text


def test_prompt_builder_round_two_uses_overlay_in_user_and_latest_raw_in_history() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2),
    )
    raw_url = _png_data_url_with_color((0, 255, 0))
    overlay_url = _png_data_url_with_color((255, 0, 0))
    history = [
        HistoryTurn(
            assistant_text="I clicked the launcher.",
            tool_results=[
                {
                    "tool_call_id": "call_1",
                    "name": "click",
                    "content": "Clicked.",
                    "raw_image_path": "/tmp/raw-1.png",
                    "raw_image_url": raw_url,
                    "overlay_image_path": "/tmp/overlay-1.png",
                    "overlay_image_url": overlay_url,
                    "image_path": "/tmp/overlay-1.png",
                    "image_url": overlay_url,
                },
            ],
        )
    ]

    envelope = builder.build(
        task="Open Calculator",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=history,
        current_clean_screenshot_url=_png_data_url(),
        tools=[],
        tool_choice="auto",
    )

    assert len(envelope.observation_images) == 1
    assert envelope.observation_images[0] == overlay_url
    tool_messages = [message for message in envelope.history_messages if message.role.value == "tool"]
    assert len(tool_messages) == 1
    block = tool_messages[0].content[0]
    assert block.content[1].url == raw_url
    assert "not the current clean desktop state" in envelope.observation_user_text
    assert "freshest clean post-action desktop state" in envelope.observation_user_text


def test_prompt_builder_round_three_uses_transition_overlay_and_latest_raw() -> None:
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
        task="Open Calculator",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=history,
        current_clean_screenshot_url=round2_raw,
        tools=[],
        tool_choice="auto",
    )

    assert envelope.observation_images == []
    assert "Current screenshot resolution: 2 x 3 pixels." in envelope.system_prompt
    tool_messages = [message for message in envelope.history_messages if message.role.value == "tool"]
    assert len(tool_messages) == 2
    first_block = tool_messages[0].content[0]
    second_block = tool_messages[1].content[0]
    assert first_block.content[1].url == round2_overlay
    assert second_block.content[1].url == round2_raw


def test_prompt_builder_keeps_tool_returned_state_and_text_notes() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2),
    )
    current_url = _png_data_url_with_color((12, 34, 56))
    raw_url = _png_data_url_with_color((0, 255, 0))
    history = [
        HistoryTurn(
            assistant_text="Round 1",
            tool_results=[
                {
                    "tool_call_id": "call_1",
                    "name": "click",
                    "content": "Clicked.",
                    "raw_image_path": "/tmp/raw-1.png",
                    "raw_image_url": raw_url,
                    "overlay_image_path": "/tmp/overlay-1.png",
                    "overlay_image_url": _png_data_url_with_color((255, 0, 0)),
                    "image_path": "/tmp/overlay-1.png",
                    "image_url": _png_data_url_with_color((255, 0, 0)),
                },
            ],
        )
    ]

    envelope = builder.build(
        task="Open Calculator",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=history,
        observation_context=ObservationContext(
            platform_name="macos",
            current_clean_screenshot_url=current_url,
            current_artifacts=[
                ObservationArtifact(
                    kind="text",
                    role="prompt_note",
                    source="mcp:DesktopNativeDataExecutor.get_app_window_info",
                    text="Calculator is focused.",
                )
            ],
        ),
        tools=[],
        tool_choice="auto",
    )

    assert envelope.observation_images == [_png_data_url_with_color((255, 0, 0))]
    assert current_url not in envelope.observation_images
    assert "Calculator is focused." in envelope.observation_user_text
    tool_messages = [message for message in envelope.history_messages if message.role.value == "tool"]
    assert len(tool_messages) == 1
    block = tool_messages[0].content[0]
    assert block.content[1].url == raw_url


def test_prompt_builder_does_not_depend_on_window_or_controls_metadata() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=2),
    )
    current_url = _png_data_url_with_color((12, 34, 56))
    base_context = ObservationContext(
        platform_name="macos",
        current_clean_screenshot_url=current_url,
        observation_metadata={"desktop_environment": "linux", "home_directory": "/home/oai"},
    )
    window_context = base_context.model_copy(
        update={
            "current_window": WindowState(name="Calculator", type="window", rect=[0, 0, 100, 100]),
            "current_controls": [{"name": "Equals", "rect": [1, 2, 3, 4]}],
        },
        deep=True,
    )

    base_envelope = builder.build(
        task="Open Calculator",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=[],
        observation_context=base_context,
        tools=[],
        tool_choice="auto",
    )
    window_envelope = builder.build(
        task="Open Calculator",
        platform_name="macos",
        current_window=WindowState(),
        history_turns=[],
        observation_context=window_context,
        tools=[],
        tool_choice="auto",
    )

    assert window_envelope.system_prompt == base_envelope.system_prompt
    assert window_envelope.persistent_user_text == base_envelope.persistent_user_text
    assert window_envelope.observation_user_text == base_envelope.observation_user_text
    assert window_envelope.observation_images == base_envelope.observation_images
    assert window_envelope.history_messages == base_envelope.history_messages


def test_desktop_system_prompt_requires_verified_task_completion_contract() -> None:
    envelope = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=1),
    ).build(
        task="Create the chart and save the spreadsheet.",
        platform_name="linux",
        current_window=WindowState(),
        history_turns=[],
        tools=[],
        tool_choice="auto",
    )

    system_prompt = envelope.system_prompt
    assert "Task completion contract:" in system_prompt
    assert "answer_task" in system_prompt
    assert "state_task" in system_prompt
    assert "artifact_task" in system_prompt
    assert "Do not treat a natural-language answer as completion" in system_prompt
    assert "verifier-observable desktop, browser, application, or file state" in system_prompt
    assert "read-only bash/file/status inspection tool" in system_prompt
    assert "representative cells, formulas, copied ranges, formatting, charts" in system_prompt
    assert "based on verified state" in system_prompt


def test_desktop_system_prompt_includes_runtime_step_budget_when_available() -> None:
    builder = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=1),
    )
    step_7_envelope = builder.build(
        task="Complete the spreadsheet.",
        platform_name="linux",
        current_window=WindowState(),
        history_turns=[],
        step_budget={
            "current_round_step": 7,
            "round_limit": 20,
            "remaining_round_steps": 14,
            "current_session_step": 9,
            "session_limit": 50,
        },
        tools=[],
        tool_choice="auto",
    )
    step_1_envelope = builder.build(
        task="Complete the spreadsheet.",
        platform_name="linux",
        current_window=WindowState(),
        history_turns=[],
        step_budget={
            "current_round_step": 1,
            "round_limit": 20,
            "remaining_round_steps": 20,
            "current_session_step": 1,
            "session_limit": 50,
        },
        tools=[],
        tool_choice="auto",
    )

    system_prompt = step_7_envelope.system_prompt
    assert step_1_envelope.system_prompt == system_prompt
    assert "This task has a finite step budget of 20 model turns." in system_prompt
    assert "this is step" not in system_prompt
    assert "step(s), including this one, remain" not in system_prompt
    assert "Session budget:" not in system_prompt
    assert "Step budget is finite. Use it deliberately" in system_prompt
    assert "When the remaining budget is low, stop exploring" in system_prompt


def test_desktop_system_prompt_omits_step_budget_when_unavailable() -> None:
    envelope = PromptEnvelopeBuilder(
        template_root=Path("northcua/agents/desktop/prompt/templates"),
        policy=PromptPolicyConfig(screenshot_window_images=1),
    ).build(
        task="Open Calculator.",
        platform_name="linux",
        current_window=WindowState(),
        history_turns=[],
        tools=[],
        tool_choice="auto",
    )

    assert "finite step budget of" not in envelope.system_prompt
