from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_GOLDEN = _REPOSITORY_ROOT / "tests" / "fixtures" / "nexau_compat" / "provider_wire_v1.json"
_PROVIDERS = (
    "openai_chat_completion",
    "openai_responses",
    "anthropic_chat_completion",
    "gemini_rest",
)
_EDGE_SCENARIOS = ("missing_call_id", "duplicate_call_id", "long_call_id")


def test_public_real_llm_caller_wire_matches_recorded_contract() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.support.providers.wire",
            "verify",
            "--mode",
            "public",
            "--output",
            str(_GOLDEN),
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
    )


def test_wire_golden_covers_every_provider_transport_stream_and_edge_scenario() -> None:
    document = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    requests = document["requests"]

    normal = [request for request in requests if request["scenario"] == "complete_exchange"]
    edges = [request for request in requests if request["scenario"] != "complete_exchange"]

    assert len(requests) == 64
    assert {(request["provider"], request["transport"], request["streaming"]) for request in normal} == {
        (provider, transport, streaming)
        for provider in _PROVIDERS
        for transport in ("sync", "async")
        for streaming in (False, True)
    }
    assert {
        (request["provider"], request["transport"], request["streaming"], request["operation"]) for request in normal
    } == {
        ("openai_chat_completion", transport, streaming, "create")
        for transport in ("sync", "async")
        for streaming in (False, True)
    } | {
        ("openai_responses", transport, streaming, "create")
        for transport in ("sync", "async")
        for streaming in (False, True)
    } | {
        (
            "anthropic_chat_completion",
            transport,
            streaming,
            "stream" if transport == "async" and streaming else "create",
        )
        for transport in ("sync", "async")
        for streaming in (False, True)
    } | {
        (
            "gemini_rest",
            transport,
            streaming,
            "stream" if transport == "async" and streaming else "post",
        )
        for transport in ("sync", "async")
        for streaming in (False, True)
    }
    assert {
        (request["scenario"], request["provider"], request["transport"], request["streaming"]) for request in edges
    } == {
        (scenario, provider, transport, streaming)
        for scenario in _EDGE_SCENARIOS
        for provider in _PROVIDERS
        for transport in ("sync", "async")
        for streaming in (False, True)
    }

    normal_operations = {
        (request["provider"], request["transport"], request["streaming"]): request["operation"] for request in normal
    }
    assert all(
        request["operation"] == normal_operations[(request["provider"], request["transport"], request["streaming"])]
        for request in edges
    )


def test_wire_edge_goldens_preserve_provider_specific_call_id_contracts() -> None:
    document = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    edge_requests = {
        (request["scenario"], request["provider"], request["transport"], request["streaming"]): request
        for request in document["requests"]
        if request["scenario"] in _EDGE_SCENARIOS
    }

    for transport in ("sync", "async"):
        for streaming in (False, True):
            chat_payload = edge_requests[("long_call_id", "openai_chat_completion", transport, streaming)]["payload"]
            responses_payload = edge_requests[("long_call_id", "openai_responses", transport, streaming)]["payload"]
            anthropic_payload = edge_requests[("long_call_id", "anthropic_chat_completion", transport, streaming)][
                "payload"
            ]

            chat_call_id = chat_payload["messages"][-3]["tool_calls"][0]["id"]
            responses_call_id = responses_payload["input"][-3]["call_id"]
            anthropic_call_id = anthropic_payload["messages"][-3]["content"][0]["id"]

            assert chat_call_id == responses_call_id == anthropic_call_id
            assert chat_call_id.startswith("call-")
            assert len(chat_call_id) == 261

            gemini_payload = edge_requests[("long_call_id", "gemini_rest", transport, streaming)]["payload"]
            assert gemini_payload["json"]["contents"][-3]["parts"] == [
                {"functionCall": {"args": {"q": "x"}, "name": "lookup"}}
            ]


def test_responses_wire_drops_only_public_empty_reasoning_reconstruction() -> None:
    document = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    responses = [
        request
        for request in document["requests"]
        if request["provider"] == "openai_responses" and request["scenario"] == "complete_exchange"
    ]

    for request in responses:
        reasoning_items = [item for item in request["payload"]["input"] if item.get("type") == "reasoning"]
        assert reasoning_items == [
            {
                "summary": [{"text": "verified internal", "type": "summary_text"}],
                "type": "reasoning",
            }
        ]
