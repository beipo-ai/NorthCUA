from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.providers.stream_fixtures import StreamProvider

_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _ROOT / "tests" / "fixtures" / "nexau_compat"


_CASES = {
    StreamProvider.OPENAI_RESPONSES: (
        _FIXTURES / "openai_responses_stream_events_v1.json",
        _FIXTURES / "openai_responses_stream_outcome_v2.json",
    ),
    StreamProvider.OPENAI_CHAT: (
        _FIXTURES / "openai_chat_stream_events_v1.json",
        _FIXTURES / "openai_chat_stream_outcome_v2.json",
    ),
    StreamProvider.ANTHROPIC: (
        _FIXTURES / "anthropic_stream_events_v1.json",
        _FIXTURES / "anthropic_stream_outcome_v2.json",
    ),
    StreamProvider.GEMINI_REST: (
        _FIXTURES / "gemini_rest_stream_events_v1.json",
        _FIXTURES / "gemini_rest_stream_outcome_v2.json",
    ),
}


@pytest.mark.parametrize("provider", tuple(StreamProvider))
def test_public_stream_lifecycle_matches_recorded_contract(provider: StreamProvider) -> None:
    events, golden = _CASES[provider]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.support.providers.stream_replay",
            "verify",
            "--mode",
            "public",
            "--provider",
            provider.value,
            "--events",
            str(events),
            "--output",
            str(golden),
        ],
        cwd=_ROOT,
        check=True,
    )


@pytest.mark.parametrize("provider", tuple(StreamProvider))
def test_stream_golden_covers_completed_disconnect_and_async_cancel(provider: StreamProvider) -> None:
    _, golden = _CASES[provider]
    outcome = json.loads(golden.read_text(encoding="utf-8"))
    completed = outcome["completed"]

    assert outcome["schema_version"] == 2
    assert outcome["provider"] == provider.value
    assert completed["role"] == "assistant"
    assert completed["ump"]["role"] == "assistant"
    assert completed["ump"]["content"]
    assert outcome["disconnect"] == {
        "boundary": provider.disconnect_boundary,
        "message": "recorded stream disconnected",
        "type": "ConnectionError",
    }
    assert outcome["async_cancel"] == {
        "boundary": provider.async_boundary,
        "outcome": "CancelledError",
        "supported": True,
    }


def test_responses_golden_preserves_reasoning_parallel_calls_and_ump_order() -> None:
    _, golden = _CASES[StreamProvider.OPENAI_RESPONSES]
    completed = json.loads(golden.read_text(encoding="utf-8"))["completed"]

    assert completed["reasoning_content"] == "verify tools"
    assert [call["call_id"] for call in completed["tool_calls"]] == [
        "call-short",
        "call-LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
    ]
    assert [block["type"] for block in completed["ump"]["content"]] == [
        "reasoning",
        "text",
        "tool_use",
        "tool_use",
    ]


def test_provider_goldens_preserve_reasoning_tool_ids_and_signatures() -> None:
    outcomes = {
        provider: json.loads(golden.read_text(encoding="utf-8"))["completed"]
        for provider, (_, golden) in _CASES.items()
    }

    assert outcomes[StreamProvider.OPENAI_CHAT]["reasoning_content"] == "plan lookup"
    assert outcomes[StreamProvider.OPENAI_CHAT]["tool_calls"][0]["call_id"] == "call-chat"
    assert outcomes[StreamProvider.ANTHROPIC]["reasoning_signature"] == "signature-fixture"
    assert outcomes[StreamProvider.ANTHROPIC]["tool_calls"][0]["call_id"] == "call-anthropic"
    assert outcomes[StreamProvider.GEMINI_REST]["thought_signature"] == "thought-fixture"
    assert outcomes[StreamProvider.GEMINI_REST]["tool_calls"][0]["call_id"] == "gemini_tc_0"
