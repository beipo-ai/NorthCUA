"""Replay recorded provider streams through private and public NexAU.

The probe executes each provider boundary to completion and freezes the
normalized ``ModelResponse`` plus UMP projection. It also verifies transport
disconnect propagation and native async cancellation without network access.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.core.messages import Message

from tests.support.providers.stream_fixtures import StreamProvider, load_stream_events
from tests.support.providers.stream_transports import (
    async_sdk_client,
    patched_gemini_async_transport,
    patched_gemini_sync_transport,
    sync_sdk_client,
)

ProviderMode = Literal["public"]


@dataclass(frozen=True, slots=True)
class StreamFailureOutcome:
    """Stable exception contract at one explicitly named stream boundary."""

    type: str
    message: str
    boundary: str


@dataclass(frozen=True, slots=True)
class AsyncCancellationOutcome:
    """Cancellation support and the provider boundary actually exercised."""

    supported: bool
    outcome: str
    boundary: str


@dataclass(frozen=True, slots=True)
class StreamReplayOutcomeV2:
    """Deterministic outcomes from the three stream lifecycle scenarios."""

    schema_version: int
    provider: str
    completed: dict[str, object]
    disconnect: StreamFailureOutcome
    async_cancel: AsyncCancellationOutcome


def _config(provider: StreamProvider) -> LLMConfig:
    return LLMConfig(
        model="fixture-model",
        base_url="https://example.invalid",
        api_key="fixture-key",
        api_type=provider.api_type,
        max_tokens=64,
        stream=True,
    )


def _caller(
    mode: ProviderMode,
    provider: StreamProvider,
    *,
    sync_client: object,
    async_client: object | None = None,
) -> tuple[LLMCaller, object | None]:
    from northcua.agentinfra.nexau_compat.adapters.providers import NexAUProviderAdapter

    adapter = NexAUProviderAdapter()
    return (
        adapter.create_llm_caller(
            openai_client=sync_client,
            async_openai_client=async_client,
            llm_config=_config(provider),
            retry_attempts=1,
            retry_backoff_max_seconds=0,
        ),
        adapter,
    )


def _sync_call(caller: LLMCaller, adapter: object | None) -> ModelResponse:
    messages = [Message.user("recorded stream fixture")]
    if adapter is None:
        return _require_response(caller.call_llm(messages=messages, max_tokens=64))
    from northcua.agentinfra.nexau_compat.adapters.providers import LegacyLLMCallRequest, NexAUProviderAdapter

    typed_adapter = cast(NexAUProviderAdapter, adapter)
    return _require_response(typed_adapter.call_llm(caller, LegacyLLMCallRequest(messages=messages, max_tokens=64)))


async def _async_call(caller: LLMCaller, adapter: object | None) -> ModelResponse:
    messages = [Message.user("recorded stream fixture")]
    if adapter is None:
        return _require_response(await caller.call_llm_async(messages=messages, max_tokens=64))
    from northcua.agentinfra.nexau_compat.adapters.providers import LegacyLLMCallRequest, NexAUProviderAdapter

    typed_adapter = cast(NexAUProviderAdapter, adapter)
    return _require_response(
        await typed_adapter.call_llm_async(caller, LegacyLLMCallRequest(messages=messages, max_tokens=64))
    )


def _require_response(response: ModelResponse | None) -> ModelResponse:
    if response is None:
        raise RuntimeError("recorded stream produced no ModelResponse")
    return response


def _normalize_response(response: ModelResponse) -> dict[str, object]:
    """Serialize every stable semantic ModelResponse field and its UMP view."""

    ump_message = response.to_ump_message()
    ump_content: list[dict[str, object]] = []
    for block in ump_message.content:
        payload = cast(dict[str, object], block.model_dump(mode="json", exclude_none=True))
        # These fields do not exist in the private pin and are already removed
        # by NorthCUA's persistence adapter. Message ids are generated locally
        # and therefore intentionally excluded from this deterministic golden.
        if payload.get("type") in {"text", "image", "reasoning"}:
            payload.pop("id", None)
        payload.pop("raw_output", None)
        ump_content.append(payload)
    role = str(getattr(response.role, "value", response.role))
    return {
        "content": response.content,
        "output_token_ids": response.output_token_ids,
        "reasoning_content": response.reasoning_content,
        "reasoning_details": response.reasoning_details,
        "reasoning_redacted_data": response.reasoning_redacted_data,
        "reasoning_signature": response.reasoning_signature,
        "response_items": response.response_items,
        "role": role,
        "thought_signature": response.thought_signature,
        "tool_calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
                "raw_arguments": call.raw_arguments,
            }
            for call in response.tool_calls
        ],
        "usage": asdict(response.usage),
        "ump": {"role": ump_message.role.value, "content": ump_content},
    }


def _sync_transport(
    provider: StreamProvider,
    events: tuple[object, ...],
    *,
    disconnect_after: int | None = None,
) -> AbstractContextManager[None]:
    if provider is StreamProvider.GEMINI_REST:
        return patched_gemini_sync_transport(events, disconnect_after=disconnect_after)
    return nullcontext()


async def _cancel_outcome(mode: ProviderMode, provider: StreamProvider, first_event: object) -> str:
    started = asyncio.Event()
    async_client = async_sdk_client(provider, first_event, started)
    caller, adapter = _caller(mode, provider, sync_client=object(), async_client=async_client)
    transport: AbstractContextManager[None] = (
        patched_gemini_async_transport(first_event, started)
        if provider is StreamProvider.GEMINI_REST
        else nullcontext()
    )
    with transport:
        task = asyncio.create_task(_async_call(caller, adapter))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "CancelledError"
    raise AssertionError(f"async {provider.value} stream ignored task cancellation")


def collect_stream_replay(
    mode: ProviderMode,
    provider: StreamProvider,
    events_path: Path,
) -> StreamReplayOutcomeV2:
    """Execute completion, disconnect, and cancellation scenarios."""

    previous_logging_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        events = load_stream_events(provider, events_path)
        completed_caller, completed_adapter = _caller(
            mode,
            provider,
            sync_client=sync_sdk_client(provider, events),
        )
        with _sync_transport(provider, events):
            completed = _normalize_response(_sync_call(completed_caller, completed_adapter))

        disconnect_caller, disconnect_adapter = _caller(
            mode,
            provider,
            sync_client=sync_sdk_client(provider, events, disconnect_after=2),
        )
        try:
            with _sync_transport(provider, events, disconnect_after=2):
                _sync_call(disconnect_caller, disconnect_adapter)
        except Exception as exc:
            disconnect = StreamFailureOutcome(
                type=type(exc).__name__,
                message=str(exc),
                boundary=provider.disconnect_boundary,
            )
        else:
            raise AssertionError("recorded stream disconnect was swallowed")

        return StreamReplayOutcomeV2(
            schema_version=2,
            provider=provider.value,
            completed=completed,
            disconnect=disconnect,
            async_cancel=AsyncCancellationOutcome(
                supported=True,
                outcome=asyncio.run(_cancel_outcome(mode, provider, events[0])),
                boundary=provider.async_boundary,
            ),
        )
    finally:
        logging.disable(previous_logging_disable)


def _encode(outcome: StreamReplayOutcomeV2) -> bytes:
    return (json.dumps(asdict(outcome), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "verify"))
    parser.add_argument("--mode", choices=("public",), required=True)
    parser.add_argument("--provider", choices=tuple(provider.value for provider in StreamProvider), required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provider = StreamProvider(args.provider)
    actual = _encode(collect_stream_replay(args.mode, provider, args.events))
    if args.command == "collect":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(actual)
        return 0
    expected = args.output.read_bytes()
    if actual != expected:
        raise SystemExit(
            "stream replay differs from recorded contract: "
            f"expected={hashlib.sha256(expected).hexdigest()} actual={hashlib.sha256(actual).hexdigest()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
