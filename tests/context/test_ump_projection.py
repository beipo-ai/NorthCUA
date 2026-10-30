from __future__ import annotations

from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock, ToolUseBlock

from northcua.agentinfra.context.adapters import NexAUTokenEstimator, UMPContextProjector
from northcua.agentkernel.context.domain.enums import ContextSegmentKind, RetentionClass
from northcua.agentkernel.context.domain.presentation import (
    ContextPresentationNotice,
    ContextPresentationNoticeCatalog,
)


def _user(text: str) -> Message:
    return Message(role=Role.USER, content=[TextBlock(text=text)])


def _assistant_tool(call_id: str) -> Message:
    return Message(
        role=Role.ASSISTANT,
        content=[ToolUseBlock(id=call_id, name="click", input={"x": 100, "y": 200})],
    )


def _tool(call_id: str, text: str) -> Message:
    return Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id=call_id, content=text)],
    )


def test_projector_pins_task_and_latest_append_observation() -> None:
    messages = [
        _user("Task: book a flight"),
        _user("old observation"),
        _assistant_tool("call-1"),
        _tool("call-1", "old result"),
        _user("latest observation"),
    ]

    segments = UMPContextProjector(policy_profile="web", recent_iterations=1).project(messages)

    assert segments[0].kind == ContextSegmentKind.TASK
    assert segments[0].retention == RetentionClass.PINNED
    assert segments[-1].kind == ContextSegmentKind.CURRENT_OBSERVATION
    assert segments[-1].retention == RetentionClass.PINNED
    assert len(segments[1].messages) == 3
    assert segments[1].retention == RetentionClass.RECENT
    assert "100" not in segments[1].summary_text
    assert "volatile value omitted" in segments[1].summary_text


def test_projector_keeps_tool_use_and_result_atomic() -> None:
    messages = [_user("task/current observation"), _assistant_tool("call-1"), _tool("call-1", "done")]

    segments = UMPContextProjector(policy_profile="desktop", recent_iterations=1).project(messages)

    assert len(segments) == 2
    assert segments[1].kind == ContextSegmentKind.ASSISTANT_TOOL_TURN
    assert tuple(message.role for message in segments[1].messages) == (Role.ASSISTANT, Role.TOOL)


def test_projector_pins_unclosed_tool_exchange() -> None:
    messages = [_user("task"), _assistant_tool("call-missing")]

    segments = UMPContextProjector(policy_profile="desktop", recent_iterations=1).project(messages)

    assert segments[-1].kind == ContextSegmentKind.OPEN_TOOL_EXCHANGE
    assert segments[-1].retention == RetentionClass.OPEN_EXCHANGE


def test_projector_drops_image_data_and_encrypted_reasoning_from_summary_input() -> None:
    message = Message(
        role=Role.USER,
        content=[TextBlock(text="data:image/png;base64,TOPSECRET encrypted_content")],
        metadata={"encrypted_content": "TOPSECRET"},
    )

    segment = UMPContextProjector(policy_profile="generic", recent_iterations=1).project([message])[0]

    assert "TOPSECRET" not in segment.summary_text
    assert "data:image" not in segment.summary_text


def test_projector_redacts_credentials_embedded_in_tool_text() -> None:
    messages = [
        _user("task"),
        _assistant_tool("call-1"),
        _tool("call-1", 'api_key=TOPSECRET awsSecretAccessKey="abc123" Bearer abc.def'),
    ]

    segment = UMPContextProjector(policy_profile="generic", recent_iterations=1).project(messages)[1]

    assert "TOPSECRET" not in segment.summary_text
    assert "abc.def" not in segment.summary_text
    assert "abc123" not in segment.summary_text
    assert "credential omitted" in segment.summary_text


def test_desktop_history_only_pins_latest_gui_tool_result_as_current_observation() -> None:
    messages = [
        _user("task"),
        _assistant_tool("call-old"),
        _tool("call-old", "screenshot_path=/tmp/old.png"),
        _assistant_tool("call-current"),
        _tool("call-current", "desktop observation screenshot_path=/tmp/current.png"),
    ]

    segments = UMPContextProjector(policy_profile="desktop", recent_iterations=1).project(messages)

    assert segments[-1].kind == ContextSegmentKind.CURRENT_OBSERVATION
    assert segments[-1].retention == RetentionClass.PINNED
    assert tuple(message.role for message in segments[-1].messages) == (Role.ASSISTANT, Role.TOOL)


def test_pre_round_user_observation_takes_precedence_over_old_gui_result() -> None:
    messages = [
        _user("task"),
        _assistant_tool("call-old"),
        _tool("call-old", "screenshot_path=/tmp/old.png"),
        _user("current user observation"),
    ]

    segments = UMPContextProjector(policy_profile="web", recent_iterations=1).project(messages)

    assert segments[-1].kind == ContextSegmentKind.CURRENT_OBSERVATION
    assert segments[-1].messages == (messages[-1],)
    assert segments[1].kind == ContextSegmentKind.ASSISTANT_TOOL_TURN


def test_token_estimator_does_not_count_base64_bytes_as_text_tokens() -> None:
    small = Message(role=Role.USER, content=[ImageBlock(base64="A" * 1_000, mime_type="image/png")])
    large = Message(role=Role.USER, content=[ImageBlock(base64="A" * 2_000_000, mime_type="image/png")])
    estimator = NexAUTokenEstimator(model="gpt-4o")

    small_estimate = estimator.estimate((small,), ())
    large_estimate = estimator.estimate((large,), ())

    assert large_estimate.total_tokens < 10_000
    assert large_estimate.total_tokens - small_estimate.total_tokens < 2_000


def test_segment_identity_is_append_stable_and_checkpoint_tail_is_namespaced() -> None:
    projector = UMPContextProjector(policy_profile="generic", recent_iterations=1)
    original = [_user("task"), _user("repeat"), Message(role=Role.ASSISTANT, content=[TextBlock(text="same")])]
    first = projector.project(original)
    appended = projector.project(
        [*original, _user("repeat"), Message(role=Role.ASSISTANT, content=[TextBlock(text="same")])]
    )
    marker = Message(
        role=Role.FRAMEWORK,
        content=[TextBlock(text="checkpoint")],
        metadata={"northcua_context_checkpoint": True, "checkpoint_id": "checkpoint-1"},
    )
    resumed = projector.project(
        [marker, _user("repeat"), Message(role=Role.ASSISTANT, content=[TextBlock(text="same")])]
    )

    assert appended[1].segment_id == first[1].segment_id
    assert appended[-1].segment_id != first[-1].segment_id
    assert resumed[-1].segment_id not in {segment.segment_id for segment in first}


def test_segment_identity_ignores_visual_window_overlay_aging() -> None:
    projector = UMPContextProjector(policy_profile="web", recent_iterations=1)
    assistant = _assistant_tool("call-1")
    with_overlay = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call-1",
                content=[
                    TextBlock(text="verified click result"),
                    TextBlock(
                        text=(
                            "Action-overlay history image: shows where a previous action was attempted. "
                            "It is not current page state, and refs in it are expired."
                        )
                    ),
                    ImageBlock(url="data:image/png;base64,overlay"),
                ],
            )
        ],
    )
    aged = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call-1",
                content=[TextBlock(text="verified click result")],
            )
        ],
    )

    current_segment = projector.project([_user("task"), assistant, with_overlay])[1]
    aged_segment = projector.project([_user("task"), assistant, aged])[1]

    assert current_segment.stable_digest == aged_segment.stable_digest
    assert current_segment.segment_id == aged_segment.segment_id


def test_desktop_replay_image_omission_notice_does_not_change_semantic_identity() -> None:
    projector = UMPContextProjector(policy_profile="desktop", recent_iterations=1)
    assistant = _assistant_tool("call-1")
    original = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call-1",
                content=[
                    TextBlock(text="Clicked at (988, 798) with left button."),
                    ImageBlock(url="data:image/png;base64,current"),
                ],
            )
        ],
    )
    replayed = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="renamed-1",
                content=(
                    "Clicked at (988, 798) with left button.\n\n"
                    + ContextPresentationNotice.DESKTOP_HISTORY_IMAGE_OMITTED.value
                ),
            )
        ],
    )
    renamed_assistant = _assistant_tool("renamed-1")

    current_segment = projector.project([_user("task"), assistant, original])[1]
    replayed_segment = projector.project([_user("task"), renamed_assistant, replayed])[1]

    assert current_segment.stable_digest == replayed_segment.stable_digest
    assert current_segment.segment_id == replayed_segment.segment_id
    assert current_segment.summary_text == replayed_segment.summary_text


def test_presentation_notice_catalog_only_strips_trusted_tool_result_suffixes() -> None:
    notice = ContextPresentationNotice.DESKTOP_HISTORY_IMAGE_OMITTED.value

    assert ContextPresentationNoticeCatalog.strip_appended_suffixes(f"result\n\n{notice}") == "result"
    assert ContextPresentationNoticeCatalog.strip_appended_suffixes(notice) == ""
    assert ContextPresentationNoticeCatalog.strip_appended_suffixes(f"result{notice}") == f"result{notice}"
    assert ContextPresentationNoticeCatalog.strip_appended_suffixes(f"result\n{notice}") == f"result\n{notice}"
    assert ContextPresentationNoticeCatalog.strip_appended_suffixes(f"{notice}\nresult") == f"{notice}\nresult"
    assert (
        ContextPresentationNoticeCatalog.strip_appended_suffixes(f"result {notice} extra") == f"result {notice} extra"
    )


def test_segment_identity_preserves_user_text_that_starts_like_visual_notice() -> None:
    projector = UMPContextProjector(policy_profile="web", recent_iterations=1)
    old = projector.project([_user("task"), _user("Visual history image. old critical instruction")])[-1]
    changed = projector.project([_user("task"), _user("Visual history image. new different instruction")])[-1]

    assert old.stable_digest != changed.stable_digest
    assert old.segment_id != changed.segment_id


def test_segment_identity_preserves_tool_pair_topology_while_aliasing_provider_ids() -> None:
    projector = UMPContextProjector(policy_profile="web", recent_iterations=1)
    matched = projector.project([_user("task"), _assistant_tool("call-a"), _tool("call-a", "done")])[1]
    renamed = projector.project([_user("task"), _assistant_tool("call-x"), _tool("call-x", "done")])[1]
    unmatched = projector.project([_user("task"), _assistant_tool("call-a"), _tool("call-b", "done")])[1]

    assert matched.stable_digest == renamed.stable_digest
    assert matched.segment_id == renamed.segment_id
    assert unmatched.stable_digest != matched.stable_digest
    assert unmatched.kind == ContextSegmentKind.OPEN_TOOL_EXCHANGE
    assert unmatched.retention == RetentionClass.OPEN_EXCHANGE
