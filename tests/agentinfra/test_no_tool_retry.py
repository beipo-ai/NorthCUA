import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from northcua.agentinfra.providers.no_tool_retry import retry_async, retry_sync, scoped_no_tool_retry


def completion(text="done", *, tool=False, reason="stop"):
    message = {"role": "assistant", "content": text}
    if tool:
        message["tool_calls"] = [
            {"id": "call_1", "type": "function", "function": {"name": "click", "arguments": '{"x":500,"y":500}'}}
        ]
    return ChatCompletion.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "message": message, "finish_reason": reason}],
        }
    )


@pytest.mark.parametrize("tool_at", [1, 2, 3, None])
@pytest.mark.parametrize("use_async", [False, True])
def test_selected_response_and_frozen_full_request(tool_at, use_async):
    baseline = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}]}
        ],
        "tools": [{"type": "function", "function": {"name": "click"}}],
        "temperature": 0.8,
        "extra_body": {"presence_penalty": 1},
    }
    original = copy.deepcopy(baseline)
    calls, records = [], []

    def call(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        kwargs["messages"].append({"role": "assistant", "content": "must never enter next request"})
        return completion(str(len(calls)), tool=len(calls) == tool_at)

    async def acall(**kwargs):
        return call(**kwargs)

    def record(n, response, selected):
        records.append((n, response.choices[0].message.content, selected))

    result = asyncio.run(retry_async(acall, baseline, record)) if use_async else retry_sync(call, baseline, record)
    count = tool_at or 3
    assert len(calls) == count
    assert all(request == original for request in calls)
    assert baseline == original
    assert result.choices[0].message.content == str(count)
    assert [n for n, _, selected in records if selected] == [count]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("kind", ["api", "filter", "malformed", "cancel"])
def test_errors_filter_and_cancellation_do_not_consume_no_tool_attempts(use_async, kind):
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        if kind == "api":
            raise TimeoutError("transport timeout")
        if kind == "cancel":
            raise asyncio.CancelledError()
        if kind == "malformed":
            return SimpleNamespace(choices=[])
        return completion("", reason="content_filter")

    async def acall(**kwargs):
        return call(**kwargs)

    def run():
        return (
            asyncio.run(retry_async(acall, {}, lambda *a: None)) if use_async else retry_sync(call, {}, lambda *a: None)
        )

    if kind == "filter":
        assert run().choices[0].finish_reason == "content_filter"
    else:
        with pytest.raises({"api": TimeoutError, "malformed": ValueError, "cancel": asyncio.CancelledError}[kind]):
            run()
    assert len(calls) == 1


def agent_for(sync, asynchronous):
    return SimpleNamespace(
        config=SimpleNamespace(llm_config=SimpleNamespace(api_type="openai_chat_completion", model="test-model")),
        openai_client=sync,
        executor=SimpleNamespace(llm_caller=SimpleNamespace(openai_client=sync, async_openai_client=asynchronous)),
    )


@pytest.mark.parametrize("use_async", [False, True])
def test_actual_sdk_wire_receipts_and_scope_restoration(tmp_path, use_async):
    wire = []

    def handle(request):
        payload = json.loads(request.content)
        wire.append(payload)
        return httpx.Response(200, json=completion(str(len(wire))).model_dump())

    sync = OpenAI(
        api_key="fixture-only",
        base_url="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    asynchronous = AsyncOpenAI(
        api_key="fixture-only",
        base_url="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    agent = agent_for(sync, asynchronous)
    original = sync.chat.completions.create
    kwargs = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "original task"}],
        "temperature": 0.8,
        "extra_body": {"presence_penalty": 1},
        "prompt_cache_key": "a" * 64,
    }

    async def run():
        with scoped_no_tool_retry(agent, attempts=3, audit_dir=tmp_path, logical_step=56):
            assert agent._no_tool_call_attempts == 3
            selected = (
                await asynchronous.chat.completions.create(**kwargs)
                if use_async
                else sync.chat.completions.create(**kwargs)
            )
            assert selected.choices[0].message.content == "3"
            # Auxiliary calls on the same SDK object are unaffected.
            sync.chat.completions.create(model="auxiliary", messages=kwargs["messages"])
        await asynchronous.close()

    asyncio.run(run())
    sync.close()
    assert wire[0] == wire[1] == wire[2]
    assert len(wire) == 4 and wire[3]["model"] == "auxiliary"
    records = [json.loads(f.read_text()) for f in sorted(tmp_path.glob("*.json"))]
    assert len(records) == 3 and [x["selected_for_logical_round"] for x in records] == [False, False, True]
    assert len({x["request_sha256"] for x in records}) == 1
    assert all(x["logical_step"] == 56 for x in records)
    assert sync.chat.completions.create == original
    assert not hasattr(agent, "_no_tool_call_attempts")


@pytest.mark.parametrize(
    "extra", [{"stream": True}, {"n": 2}, {"prompt_cache_key": "汉" * 22}, {"prompt_cache_key": ""}]
)
def test_invalid_contract_fails_before_request_and_restores(tmp_path, extra):
    calls = []
    resource = SimpleNamespace(create=lambda **kw: calls.append(kw))
    client = SimpleNamespace(chat=SimpleNamespace(completions=resource))
    agent = agent_for(client, None)
    original = resource.create
    with pytest.raises(ValueError):
        with scoped_no_tool_retry(agent, attempts=3, audit_dir=tmp_path, logical_step=1):
            resource.create(model="test-model", **extra)
    assert not calls and resource.create is original


def test_nested_or_shared_scope_rejected_and_default_is_noop(tmp_path):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: completion())))
    agent = agent_for(client, None)
    original = client.chat.completions.create
    with scoped_no_tool_retry(agent, attempts=1, audit_dir=tmp_path, logical_step=1):
        assert client.chat.completions.create is original
    with scoped_no_tool_retry(agent, attempts=3, audit_dir=tmp_path, logical_step=1):
        with pytest.raises(RuntimeError, match="active retry scope"):
            with scoped_no_tool_retry(agent, attempts=3, audit_dir=tmp_path, logical_step=1):
                pass
    assert client.chat.completions.create is original


@pytest.mark.asyncio
async def test_real_nexau_step_commits_only_selected_response(tmp_path):
    from nexau.archs.llm.llm_config import LLMConfig
    from nexau.archs.main_sub.agent import Agent
    from nexau.archs.main_sub.config import AgentConfig
    from nexau.core.messages import Message, Role, TextBlock
    from northcua.transport.ws.local_step.single_model import SingleModelStepRunner

    wire = []

    def handle(request):
        wire.append(json.loads(request.content))
        return httpx.Response(200, json=completion(f"response {len(wire)}").model_dump())

    sync = OpenAI(
        api_key="fixture-only",
        base_url="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    asynchronous = AsyncOpenAI(
        api_key="fixture-only",
        base_url="https://fixture.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    config = AgentConfig(
        name="HarnessFixture",
        llm_config=LLMConfig(
            model="test-model",
            api_key="fixture-only",
            base_url="https://fixture.invalid/v1",
            api_type="openai_chat_completion",
            max_retries=0,
            temperature=0.8,
            timeout=1,
        ),
        max_iterations=1,
        tool_call_mode="structured",
    )
    agent = await Agent.create(config=config, openai_client=sync)
    original_async = agent.executor.llm_caller.async_openai_client
    if original_async is not None:
        await original_async.close()
    agent.executor.llm_caller.async_openai_client = asynchronous
    messages = [Message(role=Role.USER, content=[TextBlock(text="Original task")])]
    try:
        from northcua.agents.desktop.runtime.model import DesktopModelStepRunner

        captured = []

        class Executor:
            async def execute(self, **kwargs):
                captured.append(
                    await SingleModelStepRunner().run(messages=kwargs["messages"], agent=kwargs["prepared_agent"])
                )

        runtime = SimpleNamespace(
            app_config=SimpleNamespace(
                execution=SimpleNamespace(no_tool_call_attempts=3), artifacts=SimpleNamespace(root_dir=str(tmp_path))
            ),
            _prepared_agent_executor=lambda: Executor(),
        )
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
        result = captured[0]
        assert result.response_text == "response 3"
        assert result.pending_tool_calls == []
        assert len(wire) == 3 and wire[0] == wire[1] == wire[2]
        assert len(messages) == 1  # no intermediate assistant response reaches caller history
        assert result.assistant_message is not None
        assert len(list((tmp_path / "fixture/no_tool_attempts").glob("*.json"))) == 3
    finally:
        await agent.stop()
        sync.close()
        await asynchronous.close()
