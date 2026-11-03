import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from northcua.agentinfra.providers import response_recovery as recovery


def truncated():
    return NS(content="", tool_calls=[], reasoning_content="fixture", usage=NS(completion_tokens=16000))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["empty", "truncated", "503"])
async def test_three_retries_same_call_bounded_backoff(monkeypatch, kind):
    sleep = AsyncMock()
    monkeypatch.setattr(recovery, "sleep", sleep)
    monkeypatch.setattr(recovery.random, "uniform", lambda a, b: 0)
    failure = {
        "empty": RuntimeError("No response content or tool calls"),
        "truncated": truncated(),
        "503": RuntimeError("internal server error"),
    }[kind]
    final = NS(content="Done", tool_calls=[])
    call = AsyncMock(side_effect=[failure, failure, failure, final])
    assert await recovery.recover_response(call, retries=3, max_tokens=16000) is final
    assert call.await_count == 4
    assert [x.args[0] for x in sleep.await_args_list] == [2, 4, 8]


@pytest.mark.asyncio
async def test_exhaustion_and_no_tool_text(monkeypatch):
    monkeypatch.setattr(recovery, "sleep", AsyncMock())
    call = AsyncMock(return_value=truncated())
    with pytest.raises(RuntimeError, match="Reasoning-only"):
        await recovery.recover_response(call, retries=3, max_tokens=16000)
    assert call.await_count == 4
    call = AsyncMock(return_value=NS(content="[INFEASIBLE] missing file", tool_calls=[]))
    await recovery.recover_response(call, retries=3, max_tokens=16000)
    assert call.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [ValueError("bad request"), TimeoutError("deadline"), asyncio.CancelledError()])
async def test_nonretryable_and_cancel_propagate(exception):
    call = AsyncMock(side_effect=exception)
    with pytest.raises(type(exception)):
        await recovery.recover_response(call, retries=3, max_tokens=16000)
    assert call.await_count == 1


def test_scope_restores_agent_after_failure():
    agent = NS()
    with pytest.raises(ValueError):
        with recovery.response_recovery_scope(agent, enabled=True):
            assert agent._northcua_response_recovery
            raise ValueError()
    assert not hasattr(agent, "_northcua_response_recovery")
    agent._northcua_response_recovery = False
    with recovery.response_recovery_scope(agent, enabled=True):
        pass
    assert agent._northcua_response_recovery is False
