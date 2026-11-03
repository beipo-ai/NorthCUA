"""Regression checks at the harness's real runtime boundaries."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from pydantic import ValidationError

from northcua.agentinfra.providers.no_tool_retry import scoped_no_tool_retry
from northcua.agentinfra.runtime.artifacts import sanitize_session_id
from northcua.agents.desktop.runtime.model import DesktopModelStepRunner
from northcua.core.config_base import ExecutionConfig
from northcua.agentinfra.providers.response_recovery import retryable
from tests.support.local_callback import lookup_context_tool
from northcua.transport.ws.agent_config import RuntimeAgentConfigBuilder
from northcua.transport.ws.transport import NorthCUAWebSocketTransport
from tests.agentinfra.test_no_tool_retry import agent_for, completion


@pytest.mark.asyncio
async def test_lookup_context_survives_real_runtime_preparation():
    tool = lookup_context_tool(lambda question: f"answer: {question}")
    config = AgentConfig(
        name="PreparedUserTool",
        llm_config=LLMConfig(model="fixture", api_key="fixture", base_url="https://fixture.invalid/v1"),
        tools=[tool],
    )
    transport = object.__new__(NorthCUAWebSocketTransport)
    transport._runtime_agent_config_builder = RuntimeAgentConfigBuilder()
    transport._session_manager = None
    transport._recursively_apply_middlewares = lambda config: config
    agent = await transport.prepare_agent(user_id="fixture", session_id="prepared-tool", agent_config=config)
    try:
        prepared = next(t for t in agent.config.tools if t.name == "lookup_context")
        assert prepared.implementation(question="Which file?") == "answer: Which file?"
        with pytest.raises(ValidationError):
            prepared.implementation(question="Which file?", undeclared=True)
    finally:
        await agent.stop()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_http_errors_do_not_retry_on_message_keywords(status):
    error = RuntimeError("rate limit parameter unsupported; timeout value invalid")
    error.status_code = status
    assert not retryable(error)


def test_response_retry_policies_cannot_multiply_each_other():
    with pytest.raises(ValidationError, match="cannot be combined"):
        ExecutionConfig(no_tool_call_attempts=3, model_response_recovery=True)


@pytest.mark.asyncio
async def test_cancelled_sync_bridge_does_not_issue_followup_requests(tmp_path):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(5)
        return completion("text without tools")

    resource = NS(create=create)
    client = NS(chat=NS(completions=resource))
    agent = agent_for(client, None)

    def thread_call():
        try:
            return resource.create(model="test-model")
        finally:
            finished.set()

    async def step():
        with scoped_no_tool_retry(agent, attempts=3, audit_dir=tmp_path, logical_step=1):
            return await asyncio.to_thread(thread_call)

    task = asyncio.create_task(step())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
    assert len(calls) == 1
    assert resource.create is create


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute", [False, True])
async def test_retry_receipts_use_the_existing_safe_session_namespace(tmp_path, absolute):
    session_id = str(tmp_path / "absolute-escape") if absolute else "../../escape"
    resource = NS(create=lambda **kwargs: completion("done"))
    client = NS(chat=NS(completions=resource))
    agent = agent_for(client, None)
    root = tmp_path / "artifacts"

    class Executor:
        async def execute(self, **kwargs):
            resource.create(model="test-model")

    runtime = NS(
        app_config=NS(
            execution=NS(no_tool_call_attempts=3, model_response_recovery=False),
            artifacts=NS(root_dir=str(root)),
        ),
        _prepared_agent_executor=lambda: Executor(),
    )
    await DesktopModelStepRunner._execute_prepared_context(
        runtime,
        messages=[],
        agent_config=None,
        prepared_agent=agent,
        is_remote=False,
        remote_tool_dispatcher=None,
        trace_state=None,
        session_id=session_id,
        round_num=1,
        session_step=1,
        round_step=1,
        trace_context_metadata={},
        candidate=NS(name="primary"),
        side_effect_fence=None,
    )
    receipt_dir = root / sanitize_session_id(session_id) / "no_tool_attempts"
    receipts = list(receipt_dir.glob("*.json"))
    assert len(receipts) == 3
    assert all(json.loads(path.read_text())["logical_step"] == 1 for path in receipts)
    assert not (tmp_path.parent / "escape").exists()
