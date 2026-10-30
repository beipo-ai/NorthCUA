"""Typed loading for synthetic NexAU provider-stream fixtures."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import cast

from anthropic.types import RawMessageStreamEvent
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter


class StreamProvider(StrEnum):
    """Provider streaming boundaries exercised by the migration harness."""

    OPENAI_RESPONSES = "openai_responses"
    OPENAI_CHAT = "openai_chat"
    ANTHROPIC = "anthropic"
    GEMINI_REST = "gemini_rest"

    @property
    def api_type(self) -> str:
        return {
            StreamProvider.OPENAI_RESPONSES: "openai_responses",
            StreamProvider.OPENAI_CHAT: "openai_chat_completion",
            StreamProvider.ANTHROPIC: "anthropic_chat_completion",
            StreamProvider.GEMINI_REST: "gemini_rest",
        }[self]

    @property
    def async_boundary(self) -> str:
        if self is StreamProvider.GEMINI_REST:
            return "nexau_httpx_transport"
        if self is StreamProvider.ANTHROPIC:
            return "anthropic_sdk_stream"
        return "openai_sdk_stream"

    @property
    def disconnect_boundary(self) -> str:
        if self is StreamProvider.GEMINI_REST:
            return "nexau_requests_transport"
        if self is StreamProvider.ANTHROPIC:
            return "anthropic_sdk_stream"
        return "openai_sdk_stream"


def load_stream_events(provider: StreamProvider, path: Path) -> tuple[object, ...]:
    """Load one fixture with the same SDK models consumed by NexAU."""

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("stream fixture root must be a non-empty list")

    events = cast(list[object], raw)
    if provider is StreamProvider.GEMINI_REST:
        if not all(isinstance(item, Mapping) for item in events):
            raise ValueError("Gemini stream fixture entries must be mappings")
        return tuple(cast(Mapping[str, object], item) for item in events)

    event_adapter: TypeAdapter[object]
    if provider is StreamProvider.OPENAI_RESPONSES:
        event_adapter = cast(TypeAdapter[object], TypeAdapter(ResponseStreamEvent))
    elif provider is StreamProvider.OPENAI_CHAT:
        event_adapter = cast(TypeAdapter[object], TypeAdapter(ChatCompletionChunk))
    else:
        event_adapter = cast(TypeAdapter[object], TypeAdapter(RawMessageStreamEvent))
    return tuple(event_adapter.validate_python(item) for item in events)
