from __future__ import annotations

import json

import pytest
from nexau.archs.llm.llm_config import LLMConfig

import northcua.agentinfra.context.adapters.summary as summary_module
from northcua.agentinfra.context.adapters.summary import NexAUSummaryClient
from northcua.agentinfra.context.summary.validation import validate_and_expand_source_references
from northcua.agentkernel.context.domain.config import ContextSummaryConfig
from northcua.agentkernel.context.domain.enums import ContextSegmentKind
from northcua.agentkernel.context.domain.models import (
    ContextOwnerKey,
    ContextSummaryDocumentV1,
    SummaryInvocationRequest,
    SummaryItem,
    SummarySourceSegment,
)
from northcua.agentkernel.context.domain.policy import ContextSummaryPolicy
from northcua.agentkernel.context.strategies import ContextSummaryError


class _Usage:
    def to_dict(self) -> dict[str, int]:
        return {"input_tokens": 20, "completion_tokens": 5, "total_tokens": 25}


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage = _Usage()


class _FakeCaller:
    instances: list["_FakeCaller"] = []
    responses: list[str] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.__class__.instances.append(self)

    async def call_llm_async(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        return _Response(self.__class__.responses.pop(0))

    def shutdown_thread_pool(self) -> None:
        self.closed = True


def _patch_fake_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    def create(_adapter: object, **kwargs: object) -> _FakeCaller:
        return _FakeCaller(**kwargs)

    monkeypatch.setattr(summary_module.NexAUProviderAdapter, "create_llm_caller", create)


def _llm_config() -> LLMConfig:
    return LLMConfig(
        model="summary-model",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        max_tokens=4096,
        stream=True,
        reasoning={"effort": "high"},
    )


def test_summary_config_strips_all_northcua_only_provider_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeCaller.instances.clear()
    _patch_fake_caller(monkeypatch)
    primary = LLMConfig(
        model="summary-model",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        max_context_tokens=200_000,
        provider_profile="gateway",
        supports_images=True,
        tool_image_policy="artifact",
        prompt_cache_key="main-session-key",
        reasoning={"effort": "high"},
    )

    NexAUSummaryClient(
        primary_llm_config=primary,
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
    )

    llm_config = _FakeCaller.instances[0].kwargs["llm_config"]
    params = llm_config.to_openai_params()
    for key in (
        "max_context_tokens",
        "provider_profile",
        "supports_images",
        "tool_image_policy",
        "reasoning",
        "prompt_cache_key",
    ):
        assert key not in params


def test_summary_config_explicitly_disables_declared_openai_responses_reasoning() -> None:
    primary = LLMConfig(
        model="gpt-5.5",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        api_type="openai_responses",
        reasoning={"effort": "high", "summary": "detailed"},
    )

    derived = summary_module._derive_summary_llm_config(
        primary,
        ContextSummaryConfig(max_output_tokens=512),
    )

    assert derived.get_param("reasoning") == {"effort": "none"}
    assert summary_module.summary_reasoning_fingerprint_mode(primary) == "explicit_none"


def _request() -> SummaryInvocationRequest:
    return SummaryInvocationRequest(
        invocation_id="invocation",
        compaction_event_id="event",
        checkpoint_id="checkpoint",
        owner=ContextOwnerKey(
            user_id="user",
            session_id="session",
            task_epoch="task",
            logical_agent_path="web",
        ),
        policy_profile="web",
        prompt_fingerprint="prompt",
        source_digest="digest",
        segments=(
            SummarySourceSegment(
                segment_id="segment-1",
                kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                text="verified historical action " * 200,
            ),
        ),
        max_output_tokens=512,
    )


@pytest.mark.asyncio
async def test_summary_client_isolated_no_tools_no_reasoning_and_validates_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = [json.dumps({"schema_version": 1, "goal": "Finish the verified task"})]
    _patch_fake_caller(monkeypatch)
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=object(),
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512),
        policy=ContextSummaryPolicy(profile="web", rules=("Keep verified URLs.",)),
    )

    result = await client.summarize(_request())
    await client.close()

    caller = _FakeCaller.instances[0]
    assert caller.kwargs["global_storage"] is None
    assert caller.kwargs["llm_config"].stream is False
    assert caller.kwargs["llm_config"].get_param("reasoning") is None
    assert caller.calls[0]["tools"] is None
    assert caller.calls[0]["tool_call_mode"] == "structured"
    assert result.document.goal == "Finish the verified task"
    assert result.request_payload["purpose"] == "northcua_context_summary"
    assert len(result.call_records) == 1
    assert result.call_records[0].purpose == "context_summary"
    assert caller.closed


@pytest.mark.asyncio
async def test_summary_client_repairs_invalid_json_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = ["not json", json.dumps({"schema_version": 1, "goal": "Repaired"})]
    _patch_fake_caller(monkeypatch)
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512, repair_attempts=1),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
    )

    result = await client.summarize(_request())
    await client.close()

    assert result.document.goal == "Repaired"
    assert len(_FakeCaller.instances[0].calls) == 2
    assert [record.purpose for record in result.call_records] == ["context_summary", "context_summary_repair"]
    repair_prompt = _FakeCaller.instances[0].calls[1]["messages"][-1].content[0].text
    assert "Hard rendered checkpoint limit:" in repair_prompt
    assert "Hard JSON limit:" in repair_prompt
    assert "strictly shorter" in repair_prompt


def test_previous_summary_payload_uses_compact_canonical_item_provenance() -> None:
    previous = ContextSummaryDocumentV1(
        current_state=(SummaryItem(text="visible"),),
        verified_facts=(SummaryItem(text="supported", source_segment_ids=("segment-1",)),),
    )
    contract = summary_module._compression_contract(
        previous_summary=previous,
        segments=_request().segments,
        minimum_reduction_ratio=0.05,
    )

    payload = summary_module._summary_payload(
        previous_summary=previous,
        segments=_request().segments,
        contract=contract,
    )

    assert payload["previous_summary"]["current_state"] == [{"text": "visible"}]
    assert payload["previous_summary"]["verified_facts"] == [{"text": "supported", "source_segment_ids": ["segment-1"]}]


def test_summary_validation_drops_unknown_raw_segment_sources_only() -> None:
    document = ContextSummaryDocumentV1(
        goal="Continue",
        verified_facts=(
            SummaryItem(text="kept", source_segment_ids=("segment-1",)),
            SummaryItem(text="raw reference", source_segment_ids=("segment-raw-2",)),
        ),
    )

    validated = validate_and_expand_source_references(
        document,
        previous_summary=None,
        allowed_source_ids=frozenset({"segment-1"}),
        source_aliases=None,
    )

    assert validated.verified_facts[0].source_segment_ids == ("segment-1",)
    assert validated.verified_facts[1].source_segment_ids == ()


def test_summary_validation_still_rejects_unknown_non_raw_sources() -> None:
    document = ContextSummaryDocumentV1(
        goal="Continue",
        verified_facts=(SummaryItem(text="bad reference", source_segment_ids=("segment-forged",)),),
    )

    with pytest.raises(ContextSummaryError, match="unknown source segment ids"):
        validate_and_expand_source_references(
            document,
            previous_summary=None,
            allowed_source_ids=frozenset({"segment-1"}),
            source_aliases=None,
        )


@pytest.mark.asyncio
async def test_summary_client_enforces_dynamic_compression_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = [
        json.dumps({"schema_version": 1, "goal": "repeated state " * 100}),
        json.dumps(
            {
                "schema_version": 1,
                "goal": "",
                "current_state": [{"text": "Verified action completed", "source_segment_ids": ["segment-1"]}],
            }
        ),
    ]
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="verified action completed " * 48,
                ),
            )
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(
            max_output_tokens=512,
            repair_attempts=1,
            minimum_reduction_ratio=0.25,
        ),
        policy=ContextSummaryPolicy(profile="web", rules=("Do not persist stale browser refs.",)),
    )

    result = await client.summarize(request)

    caller = _FakeCaller.instances[0]
    assert len(caller.calls) == 2
    assert result.document.goal == ""
    assert [record.purpose for record in result.call_records] == [
        "context_summary",
        "context_summary_repair",
    ]
    messages = caller.calls[0]["messages"]
    system_prompt = messages[0].content[0].text
    user_prompt = messages[1].content[0].text
    assert "Compression is a hard contract" in system_prompt
    assert "never write placeholders" in system_prompt
    assert "Never copy or paraphrase generic policy" in system_prompt
    assert "below 90%" in system_prompt
    assert "unsuccessful action" in system_prompt
    assert '"maximum_canonical_characters"' in user_prompt
    assert '"minimum_reduction_ratio":0.25' in user_prompt


@pytest.mark.asyncio
async def test_summary_client_rejects_impossible_short_source_before_llm_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = []
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="short source",
                ),
            )
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
    )

    with pytest.raises(ContextSummaryError, match="too small"):
        await client.summarize(request)

    assert _FakeCaller.instances[0].calls == []


@pytest.mark.asyncio
async def test_previous_checkpoint_is_retained_baseline_not_recompressed_each_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCaller.instances.clear()
    previous = ContextSummaryDocumentV1(goal="p" * 1000)
    _FakeCaller.responses = [previous.model_dump_json()]
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "previous_summary": previous,
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="new state " * 10,
                ),
            ),
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512, minimum_reduction_ratio=0.25),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
    )

    result = await client.summarize(request)

    assert result.document.goal == previous.goal
    payload_text = _FakeCaller.instances[0].calls[0]["messages"][1].content[0].text
    assert '"retained_baseline_characters":' in payload_text
    assert '"new_source_characters":100' in payload_text


@pytest.mark.asyncio
async def test_repair_budget_is_global_across_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = [
        "invalid chunk one",
        json.dumps({"schema_version": 1, "goal": "Repaired first chunk"}),
        "invalid chunk two",
    ]
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="a" * 70_000,
                ),
            )
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512, max_chunks=2, max_calls=4, repair_attempts=1),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
    )

    with pytest.raises(Exception, match="does not contain a JSON object") as raised:
        await client.summarize(request)

    records = getattr(raised.value, "northcua_summary_call_records")
    assert [record.purpose for record in records] == [
        "context_summary_chunk_1",
        "context_summary_chunk_1_repair",
        "context_summary_chunk_2",
    ]


@pytest.mark.asyncio
async def test_multi_chunk_merge_expands_partial_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = [
        json.dumps(
            {
                "schema_version": 1,
                "goal": "Chunk one",
                "verified_facts": [{"text": "first", "source_segment_ids": ["segment-1"]}],
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "goal": "Chunk two",
                "verified_facts": [{"text": "second", "source_segment_ids": ["segment-1"]}],
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "goal": "Merged",
                "verified_facts": [{"text": "merged fact", "source_segment_ids": ["partial-1"]}],
            }
        ),
    ]
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="a" * 70_000,
                ),
            )
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512, max_chunks=2, max_calls=4),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
    )

    result = await client.summarize(request)

    assert [record.purpose for record in result.call_records] == [
        "context_summary_chunk_1",
        "context_summary_chunk_2",
        "context_summary_merge",
    ]
    assert result.document.verified_facts[0].source_segment_ids == ("segment-1",)


@pytest.mark.asyncio
async def test_multi_chunk_short_tail_is_rebalanced_before_any_paid_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = [
        json.dumps({"schema_version": 1, "goal": "."}),
        json.dumps({"schema_version": 1, "goal": "."}),
        json.dumps({"schema_version": 1, "goal": "."}),
    ]
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="a" * 2750,
                ),
            ),
            "max_calls": 4,
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(
            max_output_tokens=512,
            max_chunks=2,
            max_calls=4,
            minimum_reduction_ratio=0.25,
        ),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
        context_limit_tokens=4096,
    )

    result = await client.summarize(request)

    assert result.document.goal == "."
    assert [record.purpose for record in result.call_records] == [
        "context_summary_chunk_1",
        "context_summary_chunk_2",
        "context_summary_merge",
    ]


@pytest.mark.asyncio
async def test_multi_segment_chunking_fills_short_leading_chunk_before_rebalance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = [
        json.dumps({"schema_version": 1, "goal": "."}),
        json.dumps({"schema_version": 1, "goal": "."}),
        json.dumps({"schema_version": 1, "goal": "."}),
    ]
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="a" * 100,
                ),
                SummarySourceSegment(
                    segment_id="segment-2",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="b" * 2750,
                ),
            ),
            "max_calls": 5,
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512, max_chunks=3, max_calls=5),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
        context_limit_tokens=4096,
    )

    result = await client.summarize(request)

    assert result.document.goal == "."
    assert len(result.call_records) == 3


@pytest.mark.asyncio
async def test_multi_chunk_plan_rejects_insufficient_remaining_calls_before_leaf_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = []
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "max_calls": 2,
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="a" * 70_000,
                ),
            ),
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512, max_chunks=2, max_calls=4),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
    )

    with pytest.raises(ContextSummaryError, match="cannot complete"):
        await client.summarize(request)

    assert _FakeCaller.instances[0].calls == []


@pytest.mark.asyncio
async def test_multi_chunk_plan_reserves_global_repair_before_any_leaf_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = []
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "max_calls": 3,
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="a" * 70_000,
                ),
            ),
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512, max_chunks=2, max_calls=4),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
    )

    with pytest.raises(ContextSummaryError, match="cannot complete"):
        await client.summarize(request)

    assert _FakeCaller.instances[0].calls == []


@pytest.mark.asyncio
async def test_bounded_partial_merge_accounts_for_nested_json_escaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = [
        json.dumps({"schema_version": 1, "goal": "\\" * 100}),
        json.dumps({"schema_version": 1, "goal": "\\" * 100}),
        json.dumps({"schema_version": 1, "goal": "Merged"}),
    ]
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="a" * 12_000,
                ),
            ),
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=2048, max_chunks=2, max_calls=4),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
        context_limit_tokens=8192,
    )

    result = await client.summarize(request)

    assert result.document.goal == "Merged"
    assert len(result.call_records) == 3


@pytest.mark.asyncio
async def test_advertised_128k_source_capacity_is_executable_with_previous_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy byte contract must fit the real tokenizer/chunker call graph."""

    _FakeCaller.instances.clear()
    _FakeCaller.responses = [json.dumps({"schema_version": 1, "goal": f"partial-{index}"}) for index in range(6)]
    _patch_fake_caller(monkeypatch)
    config = ContextSummaryConfig(max_output_tokens=2048, max_chunks=4, max_calls=6)
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=config,
        policy=ContextSummaryPolicy(profile="generic", rules=()),
        context_limit_tokens=128_000,
    )
    previous = ContextSummaryDocumentV1(goal="p" * 11_000)
    assert len(previous.canonical_prompt_text().encode("utf-8")) <= 11_587
    request = _request().model_copy(
        update={
            "previous_summary": previous,
            # NUL is the observed worst case for the actual JSON+tiktoken path.
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="\x00" * client.source_capacity_bytes,
                ),
            ),
            "max_calls": config.max_calls,
        }
    )

    result = await client.summarize(request)

    assert result.document.goal.startswith("partial-")
    assert 1 <= len(_FakeCaller.instances[0].calls) <= config.max_calls


@pytest.mark.asyncio
async def test_bounded_source_is_explicit_in_checkpoint_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = [json.dumps({"schema_version": 1, "goal": "Continue"})]
    _patch_fake_caller(monkeypatch)
    request = _request().model_copy(
        update={
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="head\n...[bounded payload sha256=" + "a" * 64 + "]...\ntail" * 100,
                ),
            )
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
    )

    result = await client.summarize(request)

    assert result.document.artifacts
    assert "omitted-middle sha256=" in result.document.artifacts[0].text
    assert result.document.artifacts[0].source_segment_ids == ("segment-1",)


@pytest.mark.asyncio
async def test_bounded_marker_reserve_repairs_before_final_size_and_compression_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = [
        json.dumps({"schema_version": 1, "goal": "x" * 250}),
        json.dumps({"schema_version": 1, "goal": "."}),
    ]
    _patch_fake_caller(monkeypatch)
    marker = "[bounded payload sha256=" + "a" * 64 + "]"
    request = _request().model_copy(
        update={
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="z" * 450 + marker + "z" * 450,
                ),
            )
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(max_output_tokens=512),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
        context_limit_tokens=8192,
    )

    result = await client.summarize(request)

    assert [record.purpose for record in result.call_records] == [
        "context_summary",
        "context_summary_repair",
    ]
    assert len(result.document.canonical_prompt_text().encode("utf-8")) <= 665
    assert "omitted-middle sha256=" in result.document.artifacts[0].text


@pytest.mark.asyncio
async def test_multi_chunk_final_merge_applies_global_marker_reserve_without_double_subtraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCaller.instances.clear()
    _FakeCaller.responses = [
        json.dumps({"schema_version": 1, "goal": "."}),
        json.dumps({"schema_version": 1, "goal": "."}),
        json.dumps({"schema_version": 1, "goal": "x" * 250}),
        json.dumps({"schema_version": 1, "goal": "."}),
    ]
    _patch_fake_caller(monkeypatch)
    marker = "[bounded payload sha256=" + "a" * 64 + "]"
    request = _request().model_copy(
        update={
            "segments": (
                SummarySourceSegment(
                    segment_id="segment-1",
                    kind=ContextSegmentKind.ASSISTANT_TOOL_TURN,
                    text="z" * 5000 + marker + "z" * 5000,
                ),
            ),
            "max_calls": 4,
        }
    )
    client = NexAUSummaryClient(
        primary_llm_config=_llm_config(),
        openai_client=object(),
        async_openai_client=None,
        session_id="session",
        config=ContextSummaryConfig(
            max_output_tokens=2048,
            max_chunks=2,
            max_calls=4,
            repair_attempts=1,
        ),
        policy=ContextSummaryPolicy(profile="generic", rules=()),
        context_limit_tokens=8192,
    )

    result = await client.summarize(request)

    assert [record.purpose for record in result.call_records] == [
        "context_summary_chunk_1",
        "context_summary_chunk_2",
        "context_summary_merge",
        "context_summary_merge_repair",
    ]
    assert len(result.document.canonical_prompt_text().encode("utf-8")) <= 512
