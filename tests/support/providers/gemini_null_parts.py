"""Replay malformed Gemini responses through the public NexAU retry boundary."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast

ProviderMode = Literal["public"]

_RETRY_ATTEMPTS = 3
_TERMINAL_OUTCOME = "malformed_provider_response_after_retry_exhaustion"
_NULL_PARTS_RESPONSE = {"candidates": [{"content": {"parts": None, "role": "model"}}]}


@dataclass(frozen=True, slots=True)
class GeminiNullPartsOutcomeV1:
    """Stable behavior observed after the real provider retry loop exits."""

    schema_version: int
    request_payload_sha256: str
    attempt_count: int
    retry_callback_count: int
    terminal_outcome: str
    exception_type: str
    exception_message: str
    cause_type: str | None
    cause_message: str | None
    context_type: str | None
    context_message: str | None


class _NullPartsResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return cast(dict[str, object], _NULL_PARTS_RESPONSE)


@contextmanager
def _recording_transport(request_payloads: list[dict[str, object]]) -> Iterator[None]:
    """Replace only NexAU's Gemini HTTP send and retain canonical request bodies."""

    from nexau.archs.main_sub.execution import llm_caller as llm_caller_module

    original_post = llm_caller_module.requests.post

    def post(_url: str, **kwargs: object) -> _NullPartsResponse:
        body = kwargs.get("json")
        if not isinstance(body, Mapping):
            raise AssertionError("Gemini replay expected a JSON request body")
        request_payloads.append(_json_object(body))
        return _NullPartsResponse()

    llm_caller_module.requests.post = post
    try:
        yield
    finally:
        llm_caller_module.requests.post = original_post


def _json_object(value: Mapping[object, object]) -> dict[str, object]:
    """Round-trip a request body into a detached JSON-only mapping."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("Gemini request body did not serialize to an object")
    return cast(dict[str, object], decoded)


def _payload_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalized_exception_message(error: BaseException | None) -> str | None:
    if error is None:
        return None
    return " ".join(str(error).split())


def _terminal_outcome(error: BaseException) -> str:
    reviewed = {
        ("TypeError", "'NoneType' object is not iterable"),
        ("RuntimeError", "No response content or tool calls"),
    }
    observed = (type(error).__name__, _normalized_exception_message(error))
    if observed in reviewed:
        return _TERMINAL_OUTCOME
    return "unreviewed_terminal_exception"


def _build_caller(mode: ProviderMode, on_retry: object) -> tuple[object, object | None]:
    from nexau.archs.llm.llm_config import LLMConfig

    config = LLMConfig(
        model="fixture-model",
        base_url="https://example.invalid",
        api_key="fixture-key",
        api_type="gemini_rest",
        max_tokens=64,
        stream=False,
    )
    kwargs = {
        "openai_client": object(),
        "llm_config": config,
        "retry_attempts": _RETRY_ATTEMPTS,
        "retry_backoff_max_seconds": 0,
        "on_retry": on_retry,
    }
    from northcua.agentinfra.nexau_compat.adapters.providers import NexAUProviderAdapter

    adapter = NexAUProviderAdapter()
    return adapter.create_llm_caller(**kwargs), adapter


def _call(caller: object, adapter: object | None) -> object:
    from nexau.archs.main_sub.execution.llm_caller import LLMCaller
    from nexau.core.messages import Message

    typed_caller = cast(LLMCaller, caller)
    messages = [Message.user("Gemini null-parts malformed-response fixture")]
    if adapter is None:
        return typed_caller.call_llm(messages=messages, max_tokens=64)

    from northcua.agentinfra.nexau_compat.adapters.providers import LegacyLLMCallRequest, NexAUProviderAdapter

    return cast(NexAUProviderAdapter, adapter).call_llm(
        typed_caller,
        LegacyLLMCallRequest(messages=messages, max_tokens=64),
    )


def collect_gemini_null_parts(mode: ProviderMode) -> GeminiNullPartsOutcomeV1:
    """Run the malformed response through all retries and return typed evidence."""

    retry_callbacks: list[tuple[int, int]] = []

    def on_retry(attempt: int, max_attempts: int, _backoff: float, _message: str) -> None:
        retry_callbacks.append((attempt, max_attempts))

    caller, adapter = _build_caller(mode, on_retry)
    request_payloads: list[dict[str, object]] = []
    previous_logging_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with _recording_transport(request_payloads):
            try:
                _call(caller, adapter)
            except Exception as exc:
                terminal_exception = exc
            else:
                raise AssertionError("Gemini null-parts response unexpectedly produced a model response")
    finally:
        logging.disable(previous_logging_disable)
        shutdown = getattr(caller, "shutdown_thread_pool", None)
        if callable(shutdown):
            shutdown()

    if len(request_payloads) != _RETRY_ATTEMPTS:
        raise AssertionError(
            f"Gemini null-parts replay expected {_RETRY_ATTEMPTS} HTTP attempts, found {len(request_payloads)}"
        )
    payload_digests = {_payload_digest(payload) for payload in request_payloads}
    if len(payload_digests) != 1:
        raise AssertionError("Gemini null-parts request payload changed between retries")
    expected_callbacks = tuple((attempt, _RETRY_ATTEMPTS) for attempt in range(1, _RETRY_ATTEMPTS))
    if tuple(retry_callbacks) != expected_callbacks:
        raise AssertionError(
            f"Gemini null-parts retry callbacks differ: expected={expected_callbacks}, found={tuple(retry_callbacks)}"
        )

    return GeminiNullPartsOutcomeV1(
        schema_version=1,
        request_payload_sha256=next(iter(payload_digests)),
        attempt_count=len(request_payloads),
        retry_callback_count=len(retry_callbacks),
        terminal_outcome=_terminal_outcome(terminal_exception),
        exception_type=type(terminal_exception).__name__,
        exception_message=_normalized_exception_message(terminal_exception) or "",
        cause_type=type(terminal_exception.__cause__).__name__ if terminal_exception.__cause__ is not None else None,
        cause_message=_normalized_exception_message(terminal_exception.__cause__),
        context_type=(
            type(terminal_exception.__context__).__name__ if terminal_exception.__context__ is not None else None
        ),
        context_message=_normalized_exception_message(terminal_exception.__context__),
    )
