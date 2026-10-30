from __future__ import annotations

import json
from uuid import UUID

import pytest
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock

from tests.support.providers.golden_serialization import (
    NexAUGoldenSerializationAdapter,
    NexAUGoldenSerializationPolicy,
    PublicMessageExtensionPolicy,
)


def test_legacy_message_projection_is_stable_and_keeps_nexau_type() -> None:
    message_id = UUID("11111111-1111-4111-8111-111111111111")
    message = Message(
        id=message_id,
        role=Role.ASSISTANT,
        content=[TextBlock(text="完成")],
        metadata={"z": 2, "a": {"id": message_id}},
    )

    adapter = NexAUGoldenSerializationAdapter()
    payload = adapter.message_payload(message)

    assert isinstance(message, Message)
    assert payload == {
        "id": str(message_id),
        "role": "assistant",
        "content": [{"type": "text", "text": "完成"}],
        "metadata": {"z": 2, "a": {"id": str(message_id)}},
    }
    assert json.loads(adapter.canonical_json([message])) == [payload]
    assert adapter.stable_digest([message]) == adapter.stable_digest([message.model_copy()])


def test_message_load_returns_installed_nexau_model() -> None:
    payload = {
        "id": "22222222-2222-4222-8222-222222222222",
        "role": "user",
        "content": [{"type": "text", "text": "hello"}],
        "metadata": {},
    }

    loaded = NexAUGoldenSerializationAdapter.load_message(payload)

    assert type(loaded) is Message
    assert str(loaded.id) == payload["id"]


def test_public_message_load_accepts_external_non_uuid_identifier() -> None:
    if Message.model_fields["id"].annotation is not str:
        pytest.skip("installed private baseline still requires UUID message ids")
    payload = {
        "id": "msg-external-history-17",
        "role": "user",
        "content": [{"type": "text", "text": "hello"}],
        "metadata": {},
    }

    loaded = NexAUGoldenSerializationAdapter.load_message(payload)

    assert type(loaded) is Message
    assert loaded.id == "msg-external-history-17"


def test_public_message_extensions_are_opt_in() -> None:
    if "id" not in TextBlock.model_fields or "raw_output" not in ToolResultBlock.model_fields:
        pytest.skip("installed private baseline does not expose public message extensions")

    text = TextBlock(id="stream-text-1", text="answer")
    tool_result = ToolResultBlock(
        tool_use_id="call-1",
        content="ok",
        raw_output={"exit_code": 0},
    )
    message = Message(role=Role.ASSISTANT, content=[text, tool_result])

    legacy = NexAUGoldenSerializationAdapter().message_payload(message)
    public = NexAUGoldenSerializationAdapter(
        NexAUGoldenSerializationPolicy(public_extensions=PublicMessageExtensionPolicy.PRESERVE)
    ).message_payload(message)

    assert legacy["content"] == [
        {"type": "text", "text": "answer"},
        {"type": "tool_result", "tool_use_id": "call-1", "content": "ok", "is_error": False},
    ]
    assert public["content"] == [
        {"type": "text", "id": "stream-text-1", "text": "answer"},
        {
            "type": "tool_result",
            "tool_use_id": "call-1",
            "content": "ok",
            "is_error": False,
            "raw_output": {"exit_code": 0},
        },
    ]
