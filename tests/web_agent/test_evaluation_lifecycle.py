import asyncio
import json
from types import SimpleNamespace

import pytest

from northcua.agents.web.domain.results import WebAgentResult, WebRunStatus, WebStepRecord, WebTerminalStatus
from northcua.agents.web.runner.models import WebAgentRunnerCallbacks
from northcua.agents.web.runner.run_loop import _finalize_web_agent_loop
from northcua.agents.web.runner import WebAgentRunnerRunSpec
from northcua.agents.web.evaluation.offline.replay import load_web_offline_evaluation_bundle
from tests.web_agent.test_runner_persistence import _DoneRunner, _build_runner


@pytest.mark.asyncio
async def test_evaluator_uses_registered_actor_session_before_close_and_persists_each_round(tmp_path):
    runner, store, client = _build_runner(tmp_path, _DoneRunner)
    evaluated = []

    async def evaluate(result):
        registered = runner.session_registry.get_registered_session(result.session_id)
        assert registered.client is client
        assert not client.closed
        assert store.result_path.exists()
        result.evaluation_report = {"round": result.round_num}
        evaluated.append(result.round_num)

    first = await runner.run(
        WebAgentRunnerRunSpec(user_query="first"), callbacks=WebAgentRunnerCallbacks(on_finalize=evaluate)
    )
    assert client.closed
    client.closed = False
    second = await runner.run(
        WebAgentRunnerRunSpec(user_query="second"), callbacks=WebAgentRunnerCallbacks(on_finalize=evaluate)
    )
    assert client.closed
    assert evaluated == [1, 2]
    assert first.round_num == 1 and second.round_num == 2
    assert json.loads(store.result_path.read_text())["evaluation_report"] == {"round": 2}
    for round_num, query in ((1, "first"), (2, "second")):
        bundle = load_web_offline_evaluation_bundle(store.root, round_num=round_num)
        assert bundle.result.user_query == query
        assert bundle.result.evaluation_report == {"round": round_num}
    with pytest.raises(ValueError, match="No persisted steps"):
        load_web_offline_evaluation_bundle(store.root, round_num=3)


@pytest.mark.parametrize("failure", ["verify", "persist", "evaluator", "cancel", "close", "none"])
@pytest.mark.asyncio
async def test_finalizer_always_releases_once_and_retains_evaluation(tmp_path, monkeypatch, failure):
    events = []
    saved = []
    result = WebAgentResult(
        status="success",
        terminal_status="completed",
        session_id="s",
        user_query="task",
        success=True,
        steps=[WebStepRecord(step_index=1, session_id="s", status="ok")],
    )

    async def verify(**kwargs):
        events.append("verify")
        if failure == "verify":
            raise RuntimeError("verification failed")

    def persist(result):
        events.append("persist")
        if failure == "persist":
            raise OSError("disk unavailable")
        saved.append(result.model_dump(mode="json"))

    async def evaluate(result):
        events.append("evaluate")
        result.evaluation_report = {"verdict": "inspected"}
        if failure == "evaluator":
            raise RuntimeError("evaluator failed")
        if failure == "cancel":
            raise asyncio.CancelledError()

    async def close():
        events.append("close")
        if failure == "close":
            raise RuntimeError("release failed")

    monkeypatch.setattr("northcua.agents.web.runner.run_loop._attach_terminal_verification", verify)
    operation = _finalize_web_agent_loop(
        runner=SimpleNamespace(artifact_persistence=SimpleNamespace(persist_result=persist)),
        result=result,
        resource=SimpleNamespace(close=close),
        session_id="s",
        callbacks=WebAgentRunnerCallbacks(on_finalize=evaluate),
    )
    if failure in {"verify", "persist", "evaluator", "cancel"}:
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else (RuntimeError, OSError)):
            await operation
    else:
        await operation
    assert events.count("close") == 1
    if failure in {"evaluator", "cancel", "close", "none"}:
        assert saved[-1]["evaluation_report"] == {"verdict": "inspected"}
    if failure == "close":
        assert result.status == WebRunStatus.ERROR and result.cleanup_errors
    if failure == "none":
        assert events == ["verify", "persist", "evaluate", "persist", "close"]


@pytest.mark.asyncio
async def test_interrupted_actor_does_not_start_evaluator(tmp_path, monkeypatch):
    calls = []

    async def verify(**kwargs):
        pass

    async def close():
        calls.append("close")

    async def evaluate(result):
        calls.append("evaluate")

    monkeypatch.setattr("northcua.agents.web.runner.run_loop._attach_terminal_verification", verify)
    result = WebAgentResult(
        status="running",
        terminal_status="running",
        session_id="s",
        user_query="task",
        steps=[WebStepRecord(step_index=1, session_id="s", status="ok")],
    )
    await _finalize_web_agent_loop(
        runner=SimpleNamespace(artifact_persistence=SimpleNamespace(persist_result=lambda _: None)),
        result=result,
        resource=SimpleNamespace(close=close),
        session_id="s",
        callbacks=WebAgentRunnerCallbacks(on_finalize=evaluate),
    )
    assert calls == ["close"]
    assert result.terminal_status == WebTerminalStatus.INTERRUPTED


def test_live_probe_nested_evidence_reaches_evaluator_prompt_without_media_paths():
    from northcua.agentkit.evaluator.runtime.prompt_history import EvaluatorToolResultContentProjector

    projector = EvaluatorToolResultContentProjector()
    output = {
        "status": "success",
        "tool": "client_browser_snapshot",
        "result": {
            "status": "ok",
            "result": {
                "snapshot": "button CONFIRMED",
                "screenshot_path": "/private/session.png",
                "image_url": "data:image/png;base64,AAAA",
            },
        },
    }
    text = projector.text(output)
    assert "CONFIRMED" in text
    assert "/private" not in text
    assert "data:image" not in text
