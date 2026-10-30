from __future__ import annotations

import asyncio

import pytest
from nexau.archs.main_sub.execution.hooks import Middleware, MiddlewareManager, ModelCallParams
from nexau.core.messages import Message, Role, TextBlock

from northcua.agentinfra.context.adapters import (
    LazySummaryClient,
    ProviderContextBudgetGuard,
    install_provider_context_budget_guard,
)
from northcua.agentinfra.context.bootstrap.runtime import _summary_source_capacity_bytes
from northcua.agentkernel.context.domain.config import ContextSummaryConfig
from northcua.agentkernel.context.domain.enums import ContextSegmentKind
from northcua.agentkernel.context.domain.models import (
    ContextOwnerKey,
    ContextSummaryDocumentV1,
    SummaryInvocationRequest,
    SummaryInvocationResult,
    SummarySourceSegment,
)
from northcua.agentkernel.context.strategies import ContextBudgetExceededError


def test_provider_guard_checks_final_model_call_params_without_swallowing() -> None:
    guard = ProviderContextBudgetGuard(model="gpt-4o")
    guard.configure(hard_input_budget=100, tools=[])
    messages = [Message(role=Role.SYSTEM, content=[TextBlock(text="expanded " * 500)])]

    class Caller:
        def __init__(self) -> None:
            self.provider_calls = 0
            self.middleware_manager = None

        def _call_with_retry(self, params: ModelCallParams) -> object:
            del params
            self.provider_calls += 1
            return object()

        async def _call_with_retry_async(self, params: ModelCallParams) -> object:
            del params
            self.provider_calls += 1
            return object()

    caller = Caller()
    install_provider_context_budget_guard(caller, guard)

    params = ModelCallParams(
        messages=messages,
        max_tokens=100,
        force_stop_reason=None,
        agent_state=None,
        tool_call_mode="structured",
        tools=None,
        api_params={"tools": []},
    )
    with pytest.raises(ContextBudgetExceededError, match="provider call boundary"):
        asyncio.run(caller._call_with_retry_async(params))

    assert caller.provider_calls == 0


def test_provider_guard_validates_after_middleware_mutation() -> None:
    guard = ProviderContextBudgetGuard(model="gpt-4o")
    guard.configure(hard_input_budget=100, tools=[])

    class ExpandingMiddleware(Middleware):
        def wrap_model_call(self, params: ModelCallParams, call_next: object) -> object:
            params.messages = [Message(role=Role.SYSTEM, content=[TextBlock(text="expanded " * 500)])]
            return call_next(params)  # type: ignore[operator]

    class Caller:
        def __init__(self) -> None:
            self.provider_calls = 0
            self.middleware_manager = MiddlewareManager([ExpandingMiddleware()])

        def call_llm(self, params: ModelCallParams) -> object:
            return self.middleware_manager.wrap_model_call(params, self._provider)

        async def call_llm_async(self, params: ModelCallParams) -> object:
            return self.middleware_manager.wrap_model_call(params, self._provider)

        def _provider(self, params: ModelCallParams) -> object:
            del params
            self.provider_calls += 1
            return object()

    caller = Caller()
    install_provider_context_budget_guard(caller, guard)
    params = ModelCallParams(
        messages=[Message(role=Role.SYSTEM, content=[TextBlock(text="small")])],
        max_tokens=100,
        force_stop_reason=None,
        agent_state=None,
        tool_call_mode="structured",
        tools=None,
        api_params={"tools": []},
    )

    with pytest.raises(ContextBudgetExceededError, match="provider call boundary"):
        asyncio.run(caller.call_llm_async(params))
    assert caller.provider_calls == 0


@pytest.mark.asyncio
async def test_lazy_summary_client_does_not_construct_delegate_until_needed() -> None:
    constructed = 0

    class Delegate:
        async def summarize(self, request: SummaryInvocationRequest) -> SummaryInvocationResult:
            return SummaryInvocationResult(
                invocation_id=request.invocation_id,
                document=ContextSummaryDocumentV1(goal="done"),
                model="fake",
                provider="fake",
            )

        async def close(self) -> None:
            return None

    def factory() -> Delegate:
        nonlocal constructed
        constructed += 1
        return Delegate()

    unused = LazySummaryClient(factory)
    await unused.close()
    assert constructed == 0

    client = LazySummaryClient(factory)
    request = SummaryInvocationRequest(
        invocation_id="invocation",
        compaction_event_id="event",
        checkpoint_id="checkpoint",
        owner=ContextOwnerKey(
            user_id="user",
            session_id="session",
            task_epoch="task",
            logical_agent_path="root",
        ),
        policy_profile="generic",
        prompt_fingerprint="prompt",
        source_digest="digest",
        segments=(
            SummarySourceSegment(
                segment_id="segment",
                kind=ContextSegmentKind.USER_HISTORY,
                text="history",
            ),
        ),
        max_output_tokens=128,
    )

    await client.summarize(request)
    await client.summarize(request)
    assert constructed == 1


def test_summary_source_capacity_uses_worst_case_json_wire_expansion() -> None:
    capacity = _summary_source_capacity_bytes(
        context_limit_tokens=8192,
        config=ContextSummaryConfig(max_output_tokens=2048, max_chunks=4, max_calls=6),
    )

    assert capacity == 1920
    assert len(("\n" * 20_000).encode("utf-8")) > capacity

    long_capacity = _summary_source_capacity_bytes(
        context_limit_tokens=128_000,
        config=ContextSummaryConfig(max_output_tokens=2048, max_chunks=4, max_calls=6),
    )
    assert long_capacity == 76_680
    assert long_capacity < 125_132
