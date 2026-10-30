from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import cast

import pytest
from nexau.archs.main_sub.history_list import HistoryList
from nexau.archs.session import AgentRunActionKey
from nexau.archs.session.agent_run_action_service import AgentRunActionService
from nexau.archs.session.models import AgentRunActionModel
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock

from northcua.agentinfra.nexau_compat.adapters.history import NexAUHistoryCompatibilityAdapter

_PUBLIC_HISTORY_SURFACE = "raw_output" in ToolResultBlock.model_fields
_requires_public_history_surface = pytest.mark.skipif(
    not _PUBLIC_HISTORY_SURFACE,
    reason="private baseline already has the target history contract",
)


def _tool_message(raw_output: object) -> Message:
    return Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call-1",
                content=[TextBlock(text="safe formatted result")],
                raw_output=raw_output,  # type: ignore[arg-type]
            )
        ],
    )


@_requires_public_history_surface
def test_runtime_policy_disables_raw_output_at_both_executor_construction_paths() -> None:
    import inspect

    import nexau.archs.main_sub.execution.executor as executor_module

    source = inspect.getsource(executor_module)
    assert source.count("raw_output=_coerce_raw_output(output)") == 2

    NexAUHistoryCompatibilityAdapter.install()
    NexAUHistoryCompatibilityAdapter.install()

    assert executor_module._coerce_raw_output({"secret": "must-not-persist"}) is None
    assert executor_module._coerce_raw_output(["must-not-persist"]) is None


@_requires_public_history_surface
def test_generated_tool_result_fingerprint_matches_private_message_contract() -> None:
    import nexau.archs.main_sub.execution.executor as executor_module

    NexAUHistoryCompatibilityAdapter.install()
    generated = _tool_message(executor_module._coerce_raw_output({"secret": "must-not-persist"}))
    generated_block = generated.content[0]
    assert isinstance(generated_block, ToolResultBlock)
    private_shape = generated.model_copy(update={"content": [generated_block.model_copy(update={"raw_output": None})]})

    assert HistoryList._fingerprint_message(generated) == HistoryList._fingerprint_message(private_shape)
    assert generated.model_dump(mode="json", exclude_none=True) == private_shape.model_dump(
        mode="json", exclude_none=True
    )


@_requires_public_history_surface
def test_agent_run_action_roundtrip_never_persists_synthetic_raw_secret() -> None:
    import nexau.archs.main_sub.execution.executor as executor_module

    NexAUHistoryCompatibilityAdapter.install()
    message = _tool_message(executor_module._coerce_raw_output({"secret": "must-not-persist"}))
    row = AgentRunActionModel.create_append(
        user_id="user",
        session_id="session",
        agent_id="agent",
        run_id="run",
        root_run_id="run",
        messages=[message],
    )
    payload = row.model_dump(mode="json", exclude_none=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    restored = AgentRunActionModel.model_validate(payload)

    assert "must-not-persist" not in encoded
    assert restored.append_messages is not None
    block = restored.append_messages[0].content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.raw_output is None


@pytest.mark.asyncio
@_requires_public_history_surface
async def test_public_eager_and_per_iteration_flushes_are_deferred_to_private_batch_boundary() -> None:
    class _ActionService:
        def __init__(self) -> None:
            self.appends: list[list[Message]] = []

        async def persist_append(self, **kwargs: object) -> None:
            self.appends.append(list(cast(list[Message], kwargs["messages"])))

    actions = _ActionService()
    manager = SimpleNamespace(agent_run_action=actions)
    history = HistoryList(
        [],
        session_manager=manager,
        history_key=AgentRunActionKey(user_id="user", session_id="session", agent_id="agent"),
        run_id="run",
        root_run_id="run",
    )
    NexAUHistoryCompatibilityAdapter.install()
    history.append(Message.user("task"))

    await history.flush_async()
    history.flush(iter_index=0)
    assert actions.appends == []
    assert history.has_pending_messages

    history.flush()
    await asyncio.gather(*tuple(history._background_tasks))
    assert [[message.get_text_content() for message in batch] for batch in actions.appends] == [["task"]]


@pytest.mark.asyncio
@_requires_public_history_surface
async def test_public_run_lifecycle_actions_are_private_compatible_noops() -> None:
    NexAUHistoryCompatibilityAdapter.install()
    service = cast(AgentRunActionService, object.__new__(AgentRunActionService))
    common = {
        "key": AgentRunActionKey(user_id="user", session_id="session", agent_id="agent"),
        "run_id": "run",
        "root_run_id": "run",
    }

    assert await service.persist_run_start(**common) is None
    assert await service.persist_run_end(**common, status="ok") is None
