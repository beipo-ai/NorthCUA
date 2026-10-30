"""Deterministic SDK and HTTP stream transports for compatibility replay."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

from tests.support.providers.stream_fixtures import StreamProvider


class SyncEventStream:
    """SDK-compatible context-managed iterator over recorded events."""

    def __init__(self, events: Sequence[object], *, disconnect_after: int | None = None) -> None:
        self._events = tuple(events)
        self._disconnect_after = disconnect_after
        self._index = 0

    def __enter__(self) -> SyncEventStream:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self) -> Iterator[object]:
        return self

    def __next__(self) -> object:
        if self._disconnect_after is not None and self._index == self._disconnect_after:
            raise ConnectionError("recorded stream disconnected")
        if self._index >= len(self._events):
            raise StopIteration
        event = self._events[self._index]
        self._index += 1
        return event


class BlockingAsyncEventStream:
    """Yield one event, then block until the surrounding task is cancelled."""

    def __init__(self, first_event: object, started: asyncio.Event) -> None:
        self._first_event = first_event
        self._started = started
        self._yielded = False

    async def __aenter__(self) -> BlockingAsyncEventStream:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def __aiter__(self) -> BlockingAsyncEventStream:
        return self

    async def __anext__(self) -> object:
        if not self._yielded:
            self._yielded = True
            return self._first_event
        self._started.set()
        await asyncio.sleep(60)
        raise StopAsyncIteration


class _SyncResponsesResource:
    def __init__(self, stream: SyncEventStream) -> None:
        self._stream = stream

    def stream(self, **_kwargs: object) -> SyncEventStream:
        return self._stream

    def create(self, *, stream: bool, **_kwargs: object) -> SyncEventStream:
        assert stream is True
        return self._stream


class _AsyncResponsesResource:
    def __init__(self, stream: BlockingAsyncEventStream) -> None:
        self._stream = stream

    def stream(self, **_kwargs: object) -> BlockingAsyncEventStream:
        return self._stream

    async def create(self, *, stream: bool, **_kwargs: object) -> BlockingAsyncEventStream:
        assert stream is True
        return self._stream


class _SyncChatCompletionsResource:
    def __init__(self, stream: SyncEventStream) -> None:
        self._stream = stream

    def create(self, **_kwargs: object) -> SyncEventStream:
        return self._stream


class _AsyncChatCompletionsResource:
    def __init__(self, stream: BlockingAsyncEventStream) -> None:
        self._stream = stream

    async def create(self, **_kwargs: object) -> BlockingAsyncEventStream:
        return self._stream


class _SyncAnthropicMessagesResource:
    def __init__(self, stream: SyncEventStream) -> None:
        self._stream = stream

    def create(self, **_kwargs: object) -> SyncEventStream:
        return self._stream


class _AsyncAnthropicMessagesResource:
    def __init__(self, stream: BlockingAsyncEventStream) -> None:
        self._stream = stream

    def stream(self, **_kwargs: object) -> BlockingAsyncEventStream:
        return self._stream


def sync_sdk_client(
    provider: StreamProvider,
    events: Sequence[object],
    *,
    disconnect_after: int | None = None,
) -> object:
    """Build the provider SDK surface consumed by synchronous NexAU calls."""

    stream = SyncEventStream(events, disconnect_after=disconnect_after)
    if provider is StreamProvider.OPENAI_RESPONSES:
        return SimpleNamespace(responses=_SyncResponsesResource(stream))
    if provider is StreamProvider.OPENAI_CHAT:
        return SimpleNamespace(chat=SimpleNamespace(completions=_SyncChatCompletionsResource(stream)))
    if provider is StreamProvider.ANTHROPIC:
        return SimpleNamespace(messages=_SyncAnthropicMessagesResource(stream))
    return object()


def async_sdk_client(
    provider: StreamProvider,
    first_event: object,
    started: asyncio.Event,
) -> object | None:
    """Build the provider SDK surface consumed by asynchronous NexAU calls."""

    stream = BlockingAsyncEventStream(first_event, started)
    if provider is StreamProvider.OPENAI_RESPONSES:
        return SimpleNamespace(responses=_AsyncResponsesResource(stream))
    if provider is StreamProvider.OPENAI_CHAT:
        return SimpleNamespace(chat=SimpleNamespace(completions=_AsyncChatCompletionsResource(stream)))
    if provider is StreamProvider.ANTHROPIC:
        return SimpleNamespace(messages=_AsyncAnthropicMessagesResource(stream))
    return None


class _GeminiSyncResponse:
    def __init__(self, events: Sequence[object], *, disconnect_after: int | None) -> None:
        self._events = events
        self._disconnect_after = disconnect_after

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self) -> Iterator[bytes]:
        for index, event in enumerate(self._events):
            if self._disconnect_after is not None and index == self._disconnect_after:
                raise ConnectionError("recorded stream disconnected")
            yield b"data: " + json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()


class _GeminiAsyncResponse:
    def __init__(self, first_event: object, started: asyncio.Event) -> None:
        self._first_event = first_event
        self._started = started

    async def __aenter__(self) -> _GeminiAsyncResponse:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        yield "data: " + json.dumps(self._first_event, ensure_ascii=False, separators=(",", ":"))
        self._started.set()
        await asyncio.sleep(60)


class _GeminiAsyncClient:
    def __init__(self, first_event: object, started: asyncio.Event, **_kwargs: object) -> None:
        self._response = _GeminiAsyncResponse(first_event, started)

    async def __aenter__(self) -> _GeminiAsyncClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def stream(self, *_args: object, **_kwargs: object) -> _GeminiAsyncResponse:
        return self._response


@contextmanager
def patched_gemini_sync_transport(
    events: Sequence[object],
    *,
    disconnect_after: int | None = None,
) -> Iterator[None]:
    """Replace only NexAU's internal requests transport for one replay."""

    from nexau.archs.main_sub.execution import llm_caller as llm_caller_module

    original_post = llm_caller_module.requests.post

    def _post(*_args: object, **_kwargs: object) -> _GeminiSyncResponse:
        return _GeminiSyncResponse(events, disconnect_after=disconnect_after)

    llm_caller_module.requests.post = _post
    try:
        yield
    finally:
        llm_caller_module.requests.post = original_post


@contextmanager
def patched_gemini_async_transport(first_event: object, started: asyncio.Event) -> Iterator[None]:
    """Replace only NexAU's internal httpx transport for cancellation replay."""

    from nexau.archs.main_sub.execution import llm_caller as llm_caller_module

    original_client = llm_caller_module.httpx.AsyncClient

    def _client(**kwargs: object) -> _GeminiAsyncClient:
        return _GeminiAsyncClient(first_event, started, **kwargs)

    llm_caller_module.httpx.AsyncClient = cast(object, _client)
    try:
        yield
    finally:
        llm_caller_module.httpx.AsyncClient = original_client
