"""Real NexAU/SDK boundary with an in-process HTTP fixture, never a live API."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.core.messages import Message, Role, TextBlock
from openai import AsyncOpenAI, OpenAI

from northcua.agents.desktop.runtime.model import DesktopModelStepRunner
from northcua.agentinfra.providers import response_recovery as recovery
from northcua.transport.ws.local_step.single_model import SingleModelStepRunner


@pytest.mark.asyncio
async def test_real_provider_retries_same_request_without_polluting_history(tmp_path, monkeypatch):
    wire = []
    monkeypatch.setattr(recovery, "sleep", AsyncMock())
    from northcua.transport.ws.local_step import single_model

    original_call = single_model._NEXAU_PROVIDER_ADAPTER.call_llm_async

    async def mutating_call(caller, request):
        response = await original_call(caller, request)
        # A failed provider/hook may mutate its input. Recovery must start
        # from the frozen baseline, including caller-owned message blocks.
        request.messages[0].content[0].text = "failed-attempt mutation"
        request.tools = [{"type": "function", "function": {"name": "injected", "parameters": {"type": "object"}}}]
        return response

    monkeypatch.setattr(single_model._NEXAU_PROVIDER_ADAPTER, "call_llm_async", mutating_call)

    def handle(request):
        wire.append(json.loads(request.content))
        truncated = len(wire) < 4
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if truncated else "stop",
                        "message": {
                            "role": "assistant",
                            "content": "" if truncated else "Done",
                            "reasoning_content": "fixture reasoning",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 16000 if truncated else 4,
                    "total_tokens": 16010 if truncated else 14,
                },
            },
        )

    sync = OpenAI(
        api_key="fixture",
        base_url="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    client = AsyncOpenAI(
        api_key="fixture",
        base_url="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    config = AgentConfig(
        name="OSWorld1Fixture",
        tool_call_mode="structured",
        max_iterations=1,
        llm_config=LLMConfig(
            model="fixture",
            api_key="fixture",
            base_url="https://fixture.invalid/v1",
            api_type="openai_chat_completion",
            max_retries=3,
            max_tokens=16000,
            temperature=0.7,
            timeout=120,
        ),
    )
    agent = await Agent.create(config=config, openai_client=sync)
    previous = agent.executor.llm_caller.async_openai_client
    if previous is not None:
        await previous.close()
    agent.executor.llm_caller.async_openai_client = client
    messages = [Message(role=Role.USER, content=[TextBlock(text="Original query")])]
    captured = []

    class Executor:
        async def execute(self, **kwargs):
            captured.append(
                await SingleModelStepRunner().run(messages=kwargs["messages"], agent=kwargs["prepared_agent"])
            )

    runtime = SimpleNamespace(
        app_config=SimpleNamespace(
            execution=SimpleNamespace(no_tool_call_attempts=1, model_response_recovery=True),
            artifacts=SimpleNamespace(root_dir=str(tmp_path)),
        ),
        _prepared_agent_executor=lambda: Executor(),
    )
    try:
        async with asyncio.timeout(10):
            await DesktopModelStepRunner._execute_prepared_context(
                runtime,
                messages=messages,
                agent_config=config,
                prepared_agent=agent,
                is_remote=False,
                remote_tool_dispatcher=None,
                trace_state=None,
                session_id="fixture",
                round_num=1,
                session_step=1,
                round_step=1,
                trace_context_metadata={},
                candidate=SimpleNamespace(name="primary"),
                side_effect_fence=None,
            )
        assert len(wire) == 4 and all(item == wire[0] for item in wire)
        assert wire[0]["temperature"] == 0.7
        assert captured[0].response_text == "Done"
        assert len(messages) == 1
        assert messages[0].content[0].text == "Original query"
        assert not hasattr(agent, "_northcua_response_recovery")
        assert not list(tmp_path.rglob("no_tool_attempts/*.json"))
    finally:
        await agent.stop()
        sync.close()
        await client.close()
