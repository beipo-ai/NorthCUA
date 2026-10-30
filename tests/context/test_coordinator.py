from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pytest

from northcua.agentinfra.context.persistence import ArtifactCheckpointRepository
from northcua.agentkernel.context.application import CheckpointHistoryAssembler, ContextCoordinator
from northcua.agentkernel.context.domain.config import ContextManagementConfig
from northcua.agentkernel.context.domain.enums import (
    CompactionOutcome,
    ContextManagementMode,
    ContextSegmentKind,
    RetentionClass,
)
from northcua.agentkernel.context.domain.models import (
    ContextCheckpoint,
    ContextOwnerKey,
    ContextSourceRef,
    ContextSummaryDocumentV1,
    ContextTokenEstimate,
    SummaryInvocationCallRecord,
    SummaryInvocationRequest,
    SummaryInvocationResult,
    SummaryItem,
)
from northcua.agentkernel.context.domain.runtime import ContextPreparationRequest, ContextSegment
from northcua.agentkernel.context.strategies import ContextBudgetExceededError


class FakeEstimator:
    def estimate(self, messages: tuple[str, ...], tools: tuple[str, ...]) -> ContextTokenEstimate:
        total = sum(int(message.split(":", 1)[0]) for message in messages) + len(tools)
        return ContextTokenEstimate(
            message_tokens=total - len(tools), tool_tokens=len(tools), total_tokens=total, estimator="fake"
        )


class FakeMessageFactory:
    def build(self, summary: ContextSummaryDocumentV1, checkpoint_id: str) -> str:
        del checkpoint_id
        return f"5:{summary.goal}"


class MemoryRepository:
    def __init__(self) -> None:
        self.checkpoint: ContextCheckpoint | None = None
        self.lease_entries = 0

    async def load(self, owner: ContextOwnerKey) -> ContextCheckpoint | None:
        return self.checkpoint if self.checkpoint is not None and self.checkpoint.owner == owner else None

    async def compare_and_swap(
        self,
        owner: ContextOwnerKey,
        *,
        expected_revision: int,
        checkpoint: ContextCheckpoint,
    ) -> bool:
        current = self.checkpoint.revision if self.checkpoint else 0
        if current != expected_revision or checkpoint.owner != owner:
            return False
        self.checkpoint = checkpoint
        return True

    @asynccontextmanager
    async def lease(self, owner: ContextOwnerKey, source_digest: str) -> AsyncIterator[None]:
        del owner, source_digest
        self.lease_entries += 1
        yield


class FakeSummaryClient:
    def __init__(self, *, fail: bool = False, source_capacity_chars: int | None = None) -> None:
        self.calls = 0
        self.fail = fail
        self.source_capacity_chars = source_capacity_chars

    async def summarize(self, request: SummaryInvocationRequest) -> SummaryInvocationResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("summary unavailable")
        source_ids = tuple(segment.segment_id for segment in request.segments)
        document = ContextSummaryDocumentV1(
            goal="continue task",
            verified_facts=(SummaryItem(text="old work summarized", source_segment_ids=source_ids),),
        )
        return SummaryInvocationResult(
            invocation_id=request.invocation_id,
            document=document,
            model="fake",
            provider="fake",
            usage={"input_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            call_records=(
                SummaryInvocationCallRecord(
                    call_index=1,
                    purpose="context_summary",
                    request={"messages": []},
                    response={"content": "typed"},
                    usage={"input_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    status="succeeded",
                ),
            ),
        )

    async def close(self) -> None:
        return None


def _policy(mode: ContextManagementMode):
    return ContextManagementConfig.model_validate(
        {
            "mode": mode.value,
            "budget": {
                "soft_trigger_ratio": 0.7,
                "target_ratio": 0.5,
                "safety_margin_ratio": 0,
                "safety_margin_min_tokens": 0,
                "recent_iterations": 1,
            },
        }
    ).resolve("generic")


def _segment(index: int, tokens: int, retention: RetentionClass) -> ContextSegment[str]:
    digest = f"{index:064x}"
    return ContextSegment(
        segment_id=f"segment-{index}",
        kind=ContextSegmentKind.TASK if retention == RetentionClass.PINNED else ContextSegmentKind.ASSISTANT_TOOL_TURN,
        retention=retention,
        source_refs=(ContextSourceRef(sequence=index, stable_digest=digest),),
        messages=(f"{tokens}:message-{index}",),
        summary_text=f"message {index}",
        stable_digest=digest,
    )


def _request(
    mode: ContextManagementMode, segments: tuple[ContextSegment[str], ...]
) -> ContextPreparationRequest[str, str]:
    policy = _policy(mode)
    return ContextPreparationRequest(
        logical_step_id="step-1",
        owner=ContextOwnerKey(user_id="user", session_id="session", task_epoch="task", logical_agent_path="agent"),
        agent_kind="generic",
        policy=policy,
        original_messages=tuple(message for segment in segments for message in segment.messages),
        segments=segments,
        tools=(),
        context_limit=100,
        output_reserve=10,
        policy_fingerprint=policy.fingerprint(),
        prompt_fingerprint="prompt",
        model_fingerprint="model",
    )


@pytest.mark.asyncio
async def test_off_is_exact_noop_and_does_not_call_summary() -> None:
    client = FakeSummaryClient()
    request = _request(ContextManagementMode.OFF, (_segment(0, 90, RetentionClass.PINNED),))
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=MemoryRepository(),
        summary_client=client,
    )

    result = await coordinator.prepare(request)

    assert result.messages is request.original_messages
    assert result.outcome == CompactionOutcome.UNCHANGED
    assert client.calls == 0


@pytest.mark.asyncio
async def test_enforce_commits_once_and_reuses_checkpoint() -> None:
    repository = MemoryRepository()
    client = FakeSummaryClient()
    segments = (
        _segment(0, 15, RetentionClass.PINNED),
        _segment(1, 35, RetentionClass.COMPACTABLE),
        _segment(2, 35, RetentionClass.COMPACTABLE),
    )
    request = _request(ContextManagementMode.ENFORCE, segments)
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=repository,
        summary_client=client,
    )

    first = await coordinator.prepare(request)
    second = await coordinator.prepare(request)

    assert first.outcome == CompactionOutcome.CHECKPOINT_COMMITTED
    assert second.outcome == CompactionOutcome.CHECKPOINT_REUSED
    assert client.calls == 1
    assert first.messages == second.messages
    assert repository.checkpoint is not None
    assert repository.checkpoint.covered_segment_ids == ("segment-1", "segment-2")


@pytest.mark.asyncio
async def test_long_history_progressively_commits_until_target_without_repaying_prefix() -> None:
    repository = MemoryRepository()
    client = FakeSummaryClient(source_capacity_chars=10)
    segments = (
        _segment(0, 15, RetentionClass.PINNED),
        _segment(1, 30, RetentionClass.COMPACTABLE),
        _segment(2, 30, RetentionClass.COMPACTABLE),
        _segment(3, 30, RetentionClass.COMPACTABLE),
        _segment(4, 30, RetentionClass.COMPACTABLE),
    )
    request = _request(ContextManagementMode.ENFORCE, segments)
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=repository,
        summary_client=client,
    )

    first = await coordinator.prepare(request)
    calls_after_first = client.calls
    second = await coordinator.prepare(request)

    assert first.outcome == CompactionOutcome.CHECKPOINT_COMMITTED
    assert first.estimate_after is not None and first.estimate_after.total_tokens <= 45
    assert repository.checkpoint is not None
    assert len(repository.checkpoint.covered_segment_ids) >= 3
    assert calls_after_first >= 2
    assert client.calls == calls_after_first
    assert second.outcome == CompactionOutcome.CHECKPOINT_REUSED
    assert repository.lease_entries == 1


@pytest.mark.asyncio
async def test_soft_summary_failure_continues_when_below_hard_budget() -> None:
    segments = (
        _segment(0, 20, RetentionClass.PINNED),
        _segment(1, 45, RetentionClass.COMPACTABLE),
    )
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=MemoryRepository(),
        summary_client=FakeSummaryClient(fail=True),
    )

    result = await coordinator.prepare(_request(ContextManagementMode.ENFORCE, segments))

    assert result.outcome == CompactionOutcome.SUMMARY_FAILED
    assert result.messages == ("20:message-0", "45:message-1")


@pytest.mark.asyncio
async def test_hard_guard_trims_atomic_segment_after_summary_failure() -> None:
    segments = (
        _segment(0, 30, RetentionClass.PINNED),
        _segment(1, 65, RetentionClass.COMPACTABLE),
    )
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=MemoryRepository(),
        summary_client=FakeSummaryClient(fail=True),
    )

    result = await coordinator.prepare(_request(ContextManagementMode.ENFORCE, segments))

    assert result.outcome == CompactionOutcome.DETERMINISTIC_TRIM
    assert result.messages == ("30:message-0",)
    assert result.removed_segment_ids == ("segment-1",)


@pytest.mark.asyncio
async def test_hard_guard_rejects_pinned_context_before_provider() -> None:
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=MemoryRepository(),
        summary_client=FakeSummaryClient(),
    )

    with pytest.raises(ContextBudgetExceededError, match="pinned/current"):
        await coordinator.prepare(_request(ContextManagementMode.ENFORCE, (_segment(0, 95, RetentionClass.PINNED),)))


@pytest.mark.asyncio
async def test_provider_overflow_recovery_trims_to_target_budget() -> None:
    segments = (
        _segment(0, 20, RetentionClass.PINNED),
        _segment(1, 40, RetentionClass.COMPACTABLE),
        _segment(2, 30, RetentionClass.RECENT),
    )
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=MemoryRepository(),
        summary_client=FakeSummaryClient(),
    )

    result = await coordinator.recover_overflow(_request(ContextManagementMode.ENFORCE, segments))

    assert result.outcome == CompactionOutcome.DETERMINISTIC_TRIM
    assert result.estimate_after is not None
    assert result.budget is not None
    assert result.estimate_after.total_tokens <= result.budget.target_tokens
    assert result.messages == ("20:message-0",)


@pytest.mark.asyncio
async def test_provider_overflow_recovery_allows_irreducible_context_below_hard_budget() -> None:
    segments = (
        _segment(0, 55, RetentionClass.PINNED),
        _segment(1, 35, RetentionClass.COMPACTABLE),
    )
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=MemoryRepository(),
        summary_client=FakeSummaryClient(),
    )

    result = await coordinator.recover_overflow(_request(ContextManagementMode.ENFORCE, segments))

    assert result.outcome == CompactionOutcome.DETERMINISTIC_TRIM
    assert result.budget is not None
    assert result.estimate_after is not None
    assert result.estimate_after.total_tokens > result.budget.target_tokens
    assert result.estimate_after.total_tokens <= result.budget.hard_input_budget
    assert result.messages == ("55:message-0",)


@pytest.mark.asyncio
async def test_provider_overflow_recovery_still_rejects_irreducible_context_above_hard_budget() -> None:
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=MemoryRepository(),
        summary_client=FakeSummaryClient(),
    )

    with pytest.raises(ContextBudgetExceededError, match="pinned/current"):
        await coordinator.recover_overflow(
            _request(ContextManagementMode.ENFORCE, (_segment(0, 95, RetentionClass.PINNED),))
        )


@pytest.mark.asyncio
async def test_same_digest_concurrent_preparations_charge_summary_once(tmp_path: Path) -> None:
    class SlowSummaryClient(FakeSummaryClient):
        async def summarize(self, request: SummaryInvocationRequest) -> SummaryInvocationResult:
            await asyncio.sleep(0.02)
            return await super().summarize(request)

    segments = (
        _segment(0, 15, RetentionClass.PINNED),
        _segment(1, 35, RetentionClass.COMPACTABLE),
        _segment(2, 35, RetentionClass.COMPACTABLE),
    )
    request = _request(ContextManagementMode.ENFORCE, segments)
    client = SlowSummaryClient()
    coordinators = [
        ContextCoordinator(
            estimator=FakeEstimator(),
            assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
            repository=ArtifactCheckpointRepository(tmp_path),
            summary_client=client,
        )
        for _ in range(12)
    ]

    results = await asyncio.gather(*(coordinator.prepare(request) for coordinator in coordinators))

    assert client.calls == 1
    assert {result.messages for result in results} == {results[0].messages}
    assert all(result.checkpoint is not None for result in results)


@pytest.mark.asyncio
async def test_invalid_old_checkpoint_is_replaced_with_monotonic_revision() -> None:
    repository = MemoryRepository()
    client = FakeSummaryClient()
    segments = (
        _segment(0, 15, RetentionClass.PINNED),
        _segment(1, 35, RetentionClass.COMPACTABLE),
        _segment(2, 35, RetentionClass.COMPACTABLE),
    )
    request = _request(ContextManagementMode.ENFORCE, segments)
    old_summary = ContextSummaryDocumentV1(goal="old")
    repository.checkpoint = ContextCheckpoint(
        checkpoint_id="old",
        owner=request.owner,
        policy_version=request.policy.policy_version,
        policy_fingerprint="stale-policy",
        prompt_fingerprint=request.prompt_fingerprint,
        model_fingerprint=request.model_fingerprint,
        revision=7,
        epoch=4,
        covered_segment_ids=("segment-1",),
        source_digest="stale-digest",
        summary=old_summary,
        summary_trace_id="old-trace",
    )
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=repository,
        summary_client=client,
    )

    result = await coordinator.prepare(request)

    assert result.checkpoint is not None
    assert result.checkpoint.revision == 8
    assert result.checkpoint.epoch == 5
    assert result.checkpoint.policy_fingerprint == request.policy_fingerprint


@pytest.mark.asyncio
async def test_checkpoint_is_not_committed_before_post_summary_budget_validation() -> None:
    class OversizedSummaryFactory(FakeMessageFactory):
        def build(self, summary: ContextSummaryDocumentV1, checkpoint_id: str) -> str:
            del summary, checkpoint_id
            return "95:oversized-summary"

    repository = MemoryRepository()
    segments = (
        _segment(0, 15, RetentionClass.PINNED),
        _segment(1, 35, RetentionClass.COMPACTABLE),
        _segment(2, 35, RetentionClass.COMPACTABLE),
    )
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(OversizedSummaryFactory()),
        repository=repository,
        summary_client=FakeSummaryClient(),
    )

    result = await coordinator.prepare(_request(ContextManagementMode.ENFORCE, segments))

    assert repository.checkpoint is None
    assert result.outcome == CompactionOutcome.SUMMARY_FAILED


@pytest.mark.asyncio
async def test_compacted_only_resume_reuses_matching_durable_checkpoint() -> None:
    repository = MemoryRepository()
    policy = _policy(ContextManagementMode.ENFORCE)
    owner = ContextOwnerKey(user_id="user", session_id="session", task_epoch="task", logical_agent_path="agent")
    checkpoint = ContextCheckpoint(
        checkpoint_id="checkpoint-resume",
        owner=owner,
        policy_version=policy.policy_version,
        policy_fingerprint=policy.fingerprint(),
        prompt_fingerprint="prompt",
        model_fingerprint="model",
        revision=1,
        epoch=1,
        covered_segment_ids=("raw-segment-no-longer-present",),
        source_digest="a" * 64,
        summary=ContextSummaryDocumentV1(goal="resume"),
        summary_trace_id="trace",
    )
    repository.checkpoint = checkpoint
    task = _segment(0, 10, RetentionClass.PINNED)
    marker = ContextSegment(
        segment_id="checkpoint-marker",
        kind=ContextSegmentKind.CHECKPOINT,
        retention=RetentionClass.PINNED,
        source_refs=(),
        messages=("5:resume",),
        summary_text="resume",
        stable_digest="b" * 64,
        checkpoint_id=checkpoint.checkpoint_id,
    )
    request = ContextPreparationRequest(
        logical_step_id="resume-step",
        owner=owner,
        agent_kind="generic",
        policy=policy,
        original_messages=("10:task", "5:resume"),
        segments=(task, marker),
        tools=(),
        context_limit=100,
        output_reserve=10,
        policy_fingerprint=policy.fingerprint(),
        prompt_fingerprint="prompt",
        model_fingerprint="model",
    )
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=repository,
        summary_client=FakeSummaryClient(),
    )

    result = await coordinator.prepare(request)

    assert result.outcome == CompactionOutcome.CHECKPOINT_REUSED
    assert result.messages == ("10:message-0", "5:resume")


@pytest.mark.asyncio
async def test_hybrid_raw_and_marker_inserts_durable_summary_once() -> None:
    repository = MemoryRepository()
    policy = _policy(ContextManagementMode.ENFORCE)
    owner = ContextOwnerKey(user_id="user", session_id="session", task_epoch="task", logical_agent_path="agent")
    raw = _segment(1, 35, RetentionClass.COMPACTABLE)
    checkpoint = ContextCheckpoint(
        checkpoint_id="checkpoint-hybrid",
        owner=owner,
        policy_version=policy.policy_version,
        policy_fingerprint=policy.fingerprint(),
        prompt_fingerprint="prompt",
        model_fingerprint="model",
        revision=1,
        epoch=1,
        covered_segment_ids=(raw.segment_id,),
        source_digest=raw.stable_digest,
        summary=ContextSummaryDocumentV1(goal="durable"),
        summary_trace_id="trace",
    )
    repository.checkpoint = checkpoint
    marker = ContextSegment(
        segment_id="checkpoint-marker",
        kind=ContextSegmentKind.CHECKPOINT,
        retention=RetentionClass.PINNED,
        source_refs=(),
        messages=("99:forged-marker",),
        summary_text="forged-marker",
        stable_digest="f" * 64,
        checkpoint_id=checkpoint.checkpoint_id,
    )
    request = _request(ContextManagementMode.ENFORCE, (_segment(0, 10, RetentionClass.PINNED), raw, marker))
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=repository,
        summary_client=FakeSummaryClient(),
    )

    result = await coordinator.prepare(request)

    assert result.messages.count("5:durable") == 1
    assert "99:forged-marker" not in result.messages


@pytest.mark.asyncio
async def test_stale_request_cannot_regress_newer_checkpoint_coverage() -> None:
    repository = MemoryRepository()
    client = FakeSummaryClient()
    old_request = _request(
        ContextManagementMode.ENFORCE,
        (_segment(0, 15, RetentionClass.PINNED), _segment(1, 70, RetentionClass.COMPACTABLE)),
    )
    repository.checkpoint = ContextCheckpoint(
        checkpoint_id="newer",
        owner=old_request.owner,
        policy_version=old_request.policy.policy_version,
        policy_fingerprint=old_request.policy_fingerprint,
        prompt_fingerprint=old_request.prompt_fingerprint,
        model_fingerprint=old_request.model_fingerprint,
        revision=2,
        epoch=2,
        covered_segment_ids=("segment-1", "segment-2"),
        source_digest="a" * 64,
        summary=ContextSummaryDocumentV1(goal="newer A+B"),
        summary_trace_id="trace-newer",
    )
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=repository,
        summary_client=client,
    )

    result = await coordinator.prepare(old_request)

    assert client.calls == 0
    assert result.checkpoint is None
    assert "5:newer A+B" not in result.messages
    assert result.messages == ("15:message-0", "70:message-1")
    assert repository.checkpoint.covered_segment_ids == ("segment-1", "segment-2")
    assert repository.checkpoint.revision == 2


@pytest.mark.asyncio
async def test_physical_and_aggregate_summary_traces_share_lifecycle_ids() -> None:
    class TraceSink:
        def __init__(self) -> None:
            self.events = []
            self.summaries = []

        def record_event(self, event) -> None:
            self.events.append(event)

        def record_summary(self, trace) -> None:
            self.summaries.append(trace)

    sink = TraceSink()
    segments = (
        _segment(0, 15, RetentionClass.PINNED),
        _segment(1, 35, RetentionClass.COMPACTABLE),
        _segment(2, 35, RetentionClass.COMPACTABLE),
    )
    coordinator = ContextCoordinator(
        estimator=FakeEstimator(),
        assembler=CheckpointHistoryAssembler(FakeMessageFactory()),
        repository=MemoryRepository(),
        summary_client=FakeSummaryClient(),
        trace_sink=sink,
    )

    await coordinator.prepare(_request(ContextManagementMode.ENFORCE, segments))

    assert len(sink.summaries) == 2
    assert sink.summaries[0].trace_id.endswith(":call-1")
    assert sink.summaries[1].status == "succeeded"
    assert {trace.compaction_event_id for trace in sink.summaries} == {sink.events[0].event_id}
    assert sink.events[-1].event_id == sink.events[0].event_id
