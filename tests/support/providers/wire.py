"""Capture and compare real NexAU ``LLMCaller`` provider-boundary payloads.

The private baseline intentionally runs the raw private caller.  The public
candidate runs the real public caller through NorthCUA's compatibility facade.
Each provider client is replaced only at its final network method, so this
harness exercises caller parameter assembly, retry projection, serializers,
and provider-specific request construction rather than comparing the facade to
itself.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.core.messages import Message, ReasoningBlock, Role, TextBlock, ToolResultBlock, ToolUseBlock

ProviderMode = Literal["public"]
TransportMode = Literal["sync", "async"]
ProviderOperation = Literal["create", "stream", "post"]
WireScenario = Literal[
    "complete_exchange",
    "missing_call_id",
    "duplicate_call_id",
    "long_call_id",
]

_PROVIDERS = (
    "openai_chat_completion",
    "openai_responses",
    "anthropic_chat_completion",
    "gemini_rest",
)
_LONG_CALL_ID = "call-" + ("L" * 256)


class _WireCaptured(RuntimeError):
    """Stop one request immediately after its final provider boundary."""


@dataclass(frozen=True, slots=True)
class WireRequestV1:
    """One deterministic provider-boundary request."""

    provider: str
    scenario: WireScenario
    transport: TransportMode
    streaming: bool
    operation: ProviderOperation
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class WireContractV1:
    """Versioned wire capture emitted by an isolated environment."""

    schema_version: int
    requests: tuple[WireRequestV1, ...]


class _Recorder:
    def __init__(self) -> None:
        self._requests: list[WireRequestV1] = []
        self.scenario: WireScenario = "complete_exchange"

    def capture(
        self,
        provider: str,
        transport: TransportMode,
        streaming: bool,
        operation: ProviderOperation,
        payload: Mapping[str, object],
    ) -> None:
        self._requests.append(
            WireRequestV1(
                provider=provider,
                scenario=self.scenario,
                transport=transport,
                streaming=streaming,
                operation=operation,
                payload={str(key): _json_value(value) for key, value in payload.items()},
            )
        )
        raise _WireCaptured

    def document(self) -> WireContractV1:
        return WireContractV1(schema_version=1, requests=tuple(self._requests))


class _SyncCreateBoundary:
    def __init__(self, recorder: _Recorder, provider: str, *, streaming: bool) -> None:
        self._recorder = recorder
        self._provider = provider
        self._streaming = streaming

    def create(self, **kwargs: object) -> None:
        self._recorder.capture(self._provider, "sync", self._streaming, "create", kwargs)

    def stream(self, **kwargs: object) -> None:
        self._recorder.capture(self._provider, "sync", self._streaming, "stream", kwargs)


class _AsyncCreateBoundary:
    def __init__(self, recorder: _Recorder, provider: str, *, streaming: bool) -> None:
        self._recorder = recorder
        self._provider = provider
        self._streaming = streaming

    async def create(self, **kwargs: object) -> None:
        self._recorder.capture(self._provider, "async", self._streaming, "create", kwargs)

    def stream(self, **kwargs: object) -> None:
        self._recorder.capture(self._provider, "async", self._streaming, "stream", kwargs)


class _SyncProviderClient:
    def __init__(self, recorder: _Recorder, provider: str, *, streaming: bool) -> None:
        boundary = _SyncCreateBoundary(recorder, provider, streaming=streaming)
        self.responses = boundary
        self.messages = boundary
        self.chat = SimpleNamespace(completions=boundary)


class _AsyncProviderClient:
    def __init__(self, recorder: _Recorder, provider: str, *, streaming: bool) -> None:
        boundary = _AsyncCreateBoundary(recorder, provider, streaming=streaming)
        self.responses = boundary
        self.messages = boundary
        self.chat = SimpleNamespace(completions=boundary)


class _AsyncGeminiClient:
    recorder: _Recorder
    streaming: bool

    def __init__(self, **_kwargs: object) -> None:
        pass

    async def __aenter__(self) -> _AsyncGeminiClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(self, url: str, **kwargs: object) -> None:
        self.recorder.capture(
            "gemini_rest",
            "async",
            self.streaming,
            "post",
            {"url": _redact_url(url), **kwargs},
        )

    def stream(self, method: str, url: str, **kwargs: object) -> None:
        self.recorder.capture(
            "gemini_rest",
            "async",
            self.streaming,
            "stream",
            {"method": method, "url": _redact_url(url), **kwargs},
        )


def _messages(scenario: WireScenario = "complete_exchange") -> list[Message]:
    """Return histories covering normal, missing, and duplicate call IDs."""

    prefix = [
        Message(role=Role.SYSTEM, content=[TextBlock(text="stable system")]),
        Message.user("look up x"),
        Message(
            role=Role.ASSISTANT,
            content=[ReasoningBlock(text="verified internal"), TextBlock(text="working")],
        ),
        Message.user("use the tool"),
    ]
    if scenario == "missing_call_id":
        exchange = [
            Message(role=Role.ASSISTANT, content=[ToolUseBlock(id="", name="lookup", input={"q": "x"})]),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_use_id="", content="result")],
                metadata={"tool_name": "lookup"},
            ),
        ]
    elif scenario == "duplicate_call_id":
        exchange = [
            Message(
                role=Role.ASSISTANT,
                content=[
                    ToolUseBlock(id="call-duplicate", name="lookup", input={"q": "x"}),
                    ToolUseBlock(id="call-duplicate", name="lookup", input={"q": "y"}),
                ],
            ),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_use_id="call-duplicate", content="first")],
                metadata={"tool_name": "lookup"},
            ),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_use_id="call-duplicate", content="second")],
                metadata={"tool_name": "lookup"},
            ),
        ]
    elif scenario == "long_call_id":
        exchange = [
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(id=_LONG_CALL_ID, name="lookup", input={"q": "x"})],
            ),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_use_id=_LONG_CALL_ID, content="result")],
                metadata={"tool_name": "lookup"},
            ),
        ]
    else:
        exchange = [
            Message(
                role=Role.ASSISTANT,
                content=[ReasoningBlock(text=""), ToolUseBlock(id="call-7", name="lookup", input={"q": "x"})],
            ),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_use_id="call-7", content="result")],
                metadata={"tool_name": "lookup"},
            ),
        ]
    return [*prefix, *exchange, Message.user("continue")]


def _tools() -> Sequence[Mapping[str, object]]:
    return (
        {
            "name": "lookup",
            "description": "Look up a fixture value.",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
                "additionalProperties": False,
            },
        },
    )


def _config(provider: str, *, streaming: bool) -> LLMConfig:
    return LLMConfig(
        model="fixture-model",
        base_url="https://example.invalid",
        api_key="fixture-key",
        api_type=provider,
        temperature=0,
        max_tokens=64,
        stream=streaming,
    )


def _build_caller(
    mode: ProviderMode,
    provider: str,
    streaming: bool,
    sync_client: object,
    async_client: object,
) -> tuple[LLMCaller, object | None]:
    config = _config(provider, streaming=streaming)
    from northcua.agentinfra.nexau_compat.adapters.providers import NexAUProviderAdapter

    adapter = NexAUProviderAdapter()
    return (
        adapter.create_llm_caller(
            openai_client=sync_client,
            async_openai_client=async_client,
            llm_config=config,
            retry_attempts=1,
            retry_backoff_max_seconds=0,
        ),
        adapter,
    )


def _call_sync(
    caller: LLMCaller,
    adapter: object | None,
    *,
    messages: list[Message] | None = None,
) -> None:
    request_messages = messages or _messages()
    if adapter is None:
        caller.call_llm(
            messages=request_messages,
            max_tokens=64,
            tool_call_mode="structured",
            tools=_tools(),
        )
        return

    from northcua.agentinfra.nexau_compat.adapters.providers import LegacyLLMCallRequest, NexAUProviderAdapter

    assert isinstance(adapter, NexAUProviderAdapter)
    adapter.call_llm(
        caller,
        LegacyLLMCallRequest(
            messages=request_messages,
            max_tokens=64,
            tool_call_mode="structured",
            tools=_tools(),
        ),
    )


async def _call_async(
    caller: LLMCaller,
    adapter: object | None,
    *,
    messages: list[Message] | None = None,
) -> None:
    request_messages = messages or _messages()
    if adapter is None:
        await caller.call_llm_async(
            messages=request_messages,
            max_tokens=64,
            tool_call_mode="structured",
            tools=_tools(),
        )
        return

    from northcua.agentinfra.nexau_compat.adapters.providers import LegacyLLMCallRequest, NexAUProviderAdapter

    assert isinstance(adapter, NexAUProviderAdapter)
    await adapter.call_llm_async(
        caller,
        LegacyLLMCallRequest(
            messages=request_messages,
            max_tokens=64,
            tool_call_mode="structured",
            tools=_tools(),
        ),
    )


def collect_wire_contract(mode: ProviderMode) -> WireContractV1:
    """Exercise sync/async and streaming/non-streaming paths for all providers."""

    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        recorder = _Recorder()
        _AsyncGeminiClient.recorder = recorder
        for provider in _PROVIDERS:
            for streaming in (False, True):
                sync_client = _SyncProviderClient(recorder, provider, streaming=streaming)
                async_client = _AsyncProviderClient(recorder, provider, streaming=streaming)
                caller, adapter = _build_caller(
                    mode,
                    provider,
                    streaming,
                    sync_client,
                    async_client,
                )
                try:
                    if provider == "gemini_rest":

                        def gemini_post(url: str, **kwargs: object) -> None:
                            recorder.capture(
                                provider,
                                "sync",
                                streaming,
                                "post",
                                {"url": _redact_url(url), **kwargs},
                            )

                        with patch(
                            "nexau.archs.main_sub.execution.llm_caller.requests.post",
                            side_effect=gemini_post,
                        ):
                            _call_sync(caller, adapter)
                    else:
                        _call_sync(caller, adapter)
                except _WireCaptured:
                    pass

                try:
                    if provider == "gemini_rest":
                        _AsyncGeminiClient.streaming = streaming
                        with patch(
                            "nexau.archs.main_sub.execution.llm_caller.httpx.AsyncClient",
                            _AsyncGeminiClient,
                        ):
                            asyncio.run(_call_async(caller, adapter))
                    else:
                        asyncio.run(_call_async(caller, adapter))
                except _WireCaptured:
                    pass
                finally:
                    shutdown = getattr(caller, "shutdown_thread_pool", None)
                    if callable(shutdown):
                        shutdown()
        for scenario in ("missing_call_id", "duplicate_call_id", "long_call_id"):
            recorder.scenario = scenario
            for provider in _PROVIDERS:
                for streaming in (False, True):
                    sync_client = _SyncProviderClient(recorder, provider, streaming=streaming)
                    async_client = _AsyncProviderClient(recorder, provider, streaming=streaming)
                    caller, adapter = _build_caller(
                        mode,
                        provider,
                        streaming,
                        sync_client,
                        async_client,
                    )
                    scenario_messages = _messages(scenario)
                    try:
                        if provider == "gemini_rest":

                            def gemini_edge_post(url: str, **kwargs: object) -> None:
                                recorder.capture(
                                    provider,
                                    "sync",
                                    streaming,
                                    "post",
                                    {"url": _redact_url(url), **kwargs},
                                )

                            with patch(
                                "nexau.archs.main_sub.execution.llm_caller.requests.post",
                                side_effect=gemini_edge_post,
                            ):
                                _call_sync(caller, adapter, messages=scenario_messages)
                        else:
                            _call_sync(caller, adapter, messages=scenario_messages)
                    except _WireCaptured:
                        pass

                    try:
                        if provider == "gemini_rest":
                            _AsyncGeminiClient.streaming = streaming
                            with patch(
                                "nexau.archs.main_sub.execution.llm_caller.httpx.AsyncClient",
                                _AsyncGeminiClient,
                            ):
                                asyncio.run(_call_async(caller, adapter, messages=scenario_messages))
                        else:
                            asyncio.run(_call_async(caller, adapter, messages=scenario_messages))
                    except _WireCaptured:
                        pass
                    finally:
                        shutdown = getattr(caller, "shutdown_thread_pool", None)
                        if callable(shutdown):
                            shutdown()
        return recorder.document()
    finally:
        logging.disable(previous_logging_disable)


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(key, "<redacted>") if key.lower() == "key" else (key, value) for key, value in parse_qsl(parts.query)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json", exclude_none=True))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _encode(document: WireContractV1) -> bytes:
    return (json.dumps(asdict(document), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _collect_command(mode: ProviderMode, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_encode(collect_wire_contract(mode)))


def _verify_command(mode: ProviderMode, golden: Path) -> None:
    actual = _encode(collect_wire_contract(mode))
    expected = golden.read_bytes()
    if actual != expected:
        raise SystemExit(
            "provider wire contract differs from recorded contract: "
            f"expected={hashlib.sha256(expected).hexdigest()} actual={hashlib.sha256(actual).hexdigest()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("collect", "verify"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--mode", choices=("public",), required=True)
        command_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mode = args.mode
    if mode != "public":
        raise SystemExit(f"unsupported mode: {mode}")
    if args.command == "collect":
        _collect_command(mode, args.output)
    else:
        _verify_command(mode, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
