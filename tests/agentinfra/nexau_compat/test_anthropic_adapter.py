from __future__ import annotations

from typing import cast

import anthropic
import httpx
import pytest

from northcua.agentinfra.nexau_compat.adapters.anthropic import (
    LegacyAnthropicSignatureError,
    wrap_anthropic_client,
)


def _signature_error() -> anthropic.BadRequestError:
    request = httpx.Request("POST", "https://example.invalid/v1/messages")
    response = httpx.Response(400, request=request)
    return anthropic.BadRequestError(
        "invalid `signature` in `thinking` block",
        response=response,
        body={"type": "error"},
    )


class _SyncFailingStream:
    def __enter__(self) -> "_SyncFailingStream":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self) -> "_SyncFailingStream":
        return self

    def __next__(self) -> object:
        raise _signature_error()


class _AsyncFailingStream:
    async def __aenter__(self) -> "_AsyncFailingStream":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def __aiter__(self) -> "_AsyncFailingStream":
        return self

    async def __anext__(self) -> object:
        raise _signature_error()


class _SyncIteratorStartFailingStream:
    def __iter__(self):
        raise _signature_error()


class _AsyncIteratorStartFailingStream:
    def __aiter__(self):
        raise _signature_error()


class _Messages:
    def __init__(self, stream: object) -> None:
        self._stream = stream

    def stream(self) -> object:
        return self._stream


class _Client:
    def __init__(self, stream: object) -> None:
        self.messages = _Messages(stream)


def test_sync_stream_signature_error_is_promoted_before_public_internal_repair() -> None:
    client = cast(_Client, wrap_anthropic_client(_Client(_SyncFailingStream())))

    with pytest.raises(LegacyAnthropicSignatureError) as raised:
        with client.messages.stream() as stream:  # type: ignore[attr-defined]
            next(iter(stream))

    assert isinstance(raised.value.original, anthropic.BadRequestError)


@pytest.mark.asyncio
async def test_async_stream_signature_error_is_promoted_before_public_internal_repair() -> None:
    client = cast(_Client, wrap_anthropic_client(_Client(_AsyncFailingStream())))

    with pytest.raises(LegacyAnthropicSignatureError) as raised:
        async with client.messages.stream() as stream:  # type: ignore[attr-defined]
            await stream.__anext__()

    assert isinstance(raised.value.original, anthropic.BadRequestError)


def test_sync_stream_promotes_signature_error_from_iterator_construction() -> None:
    client = cast(_Client, wrap_anthropic_client(_Client(_SyncIteratorStartFailingStream())))

    with pytest.raises(LegacyAnthropicSignatureError) as raised:
        iter(client.messages.stream())  # type: ignore[attr-defined]

    assert isinstance(raised.value.original, anthropic.BadRequestError)


def test_async_stream_promotes_signature_error_from_iterator_construction() -> None:
    client = cast(_Client, wrap_anthropic_client(_Client(_AsyncIteratorStartFailingStream())))

    with pytest.raises(LegacyAnthropicSignatureError) as raised:
        client.messages.stream().__aiter__()  # type: ignore[attr-defined]

    assert isinstance(raised.value.original, anthropic.BadRequestError)
