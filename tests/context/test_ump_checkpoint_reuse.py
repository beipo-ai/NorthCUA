from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock, ToolUseBlock

from northcua.agentinfra.context.adapters import UMPContextProjector, UMPSummaryMessageFactory
from northcua.agentkernel.context.application import CheckpointHistoryAssembler, ContextCoordinator
from northcua.agentkernel.context.domain.config import ContextManagementConfig
from northcua.agentkernel.context.domain.models import (
    ContextCheckpoint,
    ContextOwnerKey,
    ContextSummaryDocumentV1,
    ContextTokenEstimate,
    SummaryInvocationRequest,
    SummaryInvocationResult,
    SummaryItem,
)
from northcua.agentkernel.context.domain.presentation import ContextPresentationNotice
from northcua.agentkernel.context.domain.runtime import ContextPreparationRequest

_OVERLAY_NOTICE = (
    "Action-overlay history image: shows where a previous action was attempted. "
    "It is not current page state, and refs in it are expired."
)


class _MessageCountEstimator:
    def estimate(
        self,
        messages: tuple[Message, ...],
        tools: tuple[object, ...],
    ) -> ContextTokenEstimate:
        del tools
        total = len(messages) * 100
        return ContextTokenEstimate(
            message_tokens=total,
            tool_tokens=0,
            total_tokens=total,
            estimator="message-count",
        )


class _MemoryRepository:
    def __init__(self) -> None:
        self.checkpoint: ContextCheckpoint | None = None

    async def load(self, owner: ContextOwnerKey) -> ContextCheckpoint | None:
        return self.checkpoint if self.checkpoint is not None and self.checkpoint.owner == owner else None

    async def compare_and_swap(
        self,
        owner: ContextOwnerKey,
        *,
        expected_revision: int,
        checkpoint: ContextCheckpoint,
    ) -> bool:
        current_revision = self.checkpoint.revision if self.checkpoint is not None else 0
        if owner != checkpoint.owner or current_revision != expected_revision:
            return False
        self.checkpoint = checkpoint
        return True

    @asynccontextmanager
    async def lease(self, owner: ContextOwnerKey, source_digest: str) -> AsyncIterator[None]:
        del owner, source_digest
        yield


class _CapturingSummaryClient:
    def __init__(self) -> None:
        self.requests: list[SummaryInvocationRequest] = []

    @property
    def source_capacity_bytes(self) -> None:
        return None

    async def summarize(self, request: SummaryInvocationRequest) -> SummaryInvocationResult:
        self.requests.append(request)
        source_ids = tuple(segment.segment_id for segment in request.segments)
        return SummaryInvocationResult(
            invocation_id=request.invocation_id,
            document=ContextSummaryDocumentV1(
                goal=f"checkpoint-{len(self.requests)}",
                verified_facts=(SummaryItem(text="verified", source_segment_ids=source_ids),),
            ),
            model="fake",
            provider="fake",
        )

    async def close(self) -> None:
        return None


def _task() -> Message:
    return Message(role=Role.USER, content=[TextBlock(text="complete the form")])


def _turn(call_id: str, text: str, *, image_notice: str = "") -> tuple[Message, Message]:
    assistant = Message(
        role=Role.ASSISTANT,
        content=[ToolUseBlock(id=call_id, name="click", input={"x": 10, "y": 20})],
    )
    result_content: list[TextBlock | ImageBlock] = [TextBlock(text=text)]
    if image_notice:
        result_content.extend(
            [
                TextBlock(text=image_notice),
                ImageBlock(url=f"data:image/png;base64,{call_id}"),
            ]
        )
    tool = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id=call_id, content=result_content)],
    )
    return assistant, tool


def _policy(agent_kind: str = "web"):
    return ContextManagementConfig.model_validate(
        {
            "mode": "enforce",
            "budget": {
                "soft_trigger_ratio": 0.3,
                "target_ratio": 0.2,
                "safety_margin_ratio": 0,
                "safety_margin_min_tokens": 0,
                "recent_iterations": 2,
            },
        }
    ).resolve(agent_kind)


def _request(
    *,
    step: int,
    messages: list[Message],
    projector: UMPContextProjector,
    agent_kind: str = "web",
) -> ContextPreparationRequest[Message, object]:
    policy = _policy(agent_kind)
    return ContextPreparationRequest(
        logical_step_id=f"step-{step}",
        owner=ContextOwnerKey(
            user_id="user",
            session_id="session",
            task_epoch="task",
            logical_agent_path=agent_kind,
        ),
        agent_kind=agent_kind,
        policy=policy,
        original_messages=tuple(messages),
        segments=projector.project(messages, round_num=step),
        tools=(),
        context_limit=1000,
        output_reserve=0,
        policy_fingerprint=policy.fingerprint(),
        prompt_fingerprint="prompt",
        model_fingerprint="model",
    )


@pytest.mark.asyncio
async def test_overlay_aging_reuses_checkpoint_and_summarizes_only_new_turn() -> None:
    projector = UMPContextProjector(policy_profile="web", recent_iterations=2)
    repository = _MemoryRepository()
    summary_client = _CapturingSummaryClient()
    coordinator = ContextCoordinator(
        estimator=_MessageCountEstimator(),
        assembler=CheckpointHistoryAssembler(UMPSummaryMessageFactory()),
        repository=repository,
        summary_client=summary_client,
    )
    first_messages = [
        _task(),
        *_turn("call-a", "first result", image_notice=_OVERLAY_NOTICE),
        *_turn("call-b", "second result"),
        *_turn("call-c", "current result", image_notice="Visual history image."),
    ]
    await coordinator.prepare(_request(step=1, messages=first_messages, projector=projector))

    second_messages = [
        _task(),
        *_turn("renamed-a", "first result"),
        *_turn("renamed-b", "second result"),
        *_turn("renamed-c", "third result"),
        *_turn("renamed-d", "current result", image_notice="Visual history image."),
    ]
    await coordinator.prepare(_request(step=2, messages=second_messages, projector=projector))

    assert len(summary_client.requests) == 2
    assert summary_client.requests[0].previous_summary is None
    assert summary_client.requests[1].previous_summary is not None
    assert len(summary_client.requests[0].segments) == 1
    assert len(summary_client.requests[1].segments) == 1
    assert repository.checkpoint is not None
    assert repository.checkpoint.epoch == 2
    assert len(repository.checkpoint.covered_sources) == 2


def _desktop_turn(call_id: str, text: str, *, replayed: bool) -> tuple[Message, Message]:
    assistant = Message(
        role=Role.ASSISTANT,
        content=[ToolUseBlock(id=call_id, name="click", input={"x": 10, "y": 20})],
    )
    content: str | list[TextBlock | ImageBlock]
    if replayed:
        content = f"{text}\n\n{ContextPresentationNotice.DESKTOP_HISTORY_IMAGE_OMITTED.value}"
    else:
        content = [TextBlock(text=text), ImageBlock(url=f"data:image/png;base64,{call_id}")]
    return assistant, Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id=call_id, content=content)],
    )


@pytest.mark.asyncio
async def test_desktop_image_window_aging_reuses_checkpoint_and_summarizes_only_new_turn() -> None:
    projector = UMPContextProjector(policy_profile="desktop", recent_iterations=2)
    repository = _MemoryRepository()
    summary_client = _CapturingSummaryClient()
    coordinator = ContextCoordinator(
        estimator=_MessageCountEstimator(),
        assembler=CheckpointHistoryAssembler(UMPSummaryMessageFactory()),
        repository=repository,
        summary_client=summary_client,
    )
    first_messages = [
        _task(),
        *_desktop_turn("call-a", "first result", replayed=False),
        *_turn("call-b", "second result"),
        *_desktop_turn("call-c", "current result", replayed=False),
    ]
    await coordinator.prepare(_request(step=1, messages=first_messages, projector=projector, agent_kind="desktop"))

    second_messages = [
        _task(),
        *_desktop_turn("renamed-a", "first result", replayed=True),
        *_turn("renamed-b", "second result"),
        *_desktop_turn("renamed-c", "third result", replayed=True),
        *_desktop_turn("renamed-d", "current result", replayed=False),
    ]
    await coordinator.prepare(_request(step=2, messages=second_messages, projector=projector, agent_kind="desktop"))

    assert len(summary_client.requests) == 2
    assert summary_client.requests[0].previous_summary is None
    assert summary_client.requests[1].previous_summary is not None
    assert len(summary_client.requests[0].segments) == 1
    assert len(summary_client.requests[1].segments) == 1
    assert repository.checkpoint is not None
    assert repository.checkpoint.epoch == 2
    assert len(repository.checkpoint.covered_sources) == 2
