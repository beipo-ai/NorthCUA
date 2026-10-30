"""Failure-mode coverage for desktop round evaluator persistence policy."""

from __future__ import annotations

import asyncio
from types import MethodType

import pytest

from northcua.agents.desktop.evaluation.service import RoundEvaluatorService
from northcua.core.config import ArtifactConfig, NorthCUAAppConfig
from northcua.core.enums import EvaluationVerdict, TaskStatus
from northcua.runtime.artifacts.store import NorthCUAArtifactStore
from tests.cua.evaluation_helpers import agent_config as make_agent_config
from tests.cua.evaluation_helpers import step


def test_round_evaluator_fail_open_writes_uncertain_record(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-fail-open", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    app_config.evaluation.fail_open = True
    service = RoundEvaluatorService(
        app_config=app_config,
        agent_config=make_agent_config(tmp_path),
        artifact_store=store,
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )

    async def _raise_impl(self, **kwargs):
        del self, kwargs
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(service, "_evaluate_round_impl", MethodType(_raise_impl, service))
    record = asyncio.run(
        service.evaluate_round(
            round_request="Open Calculator",
            round_num=1,
            step_logs=[
                step(
                    session_id="sess-fail-open",
                    session_step=1,
                    round_num=1,
                    round_step=1,
                    request="Open Calculator",
                    result="",
                    status=TaskStatus.CONTINUE,
                    clean_screenshot_path="",
                    tool_image_path="",
                    a11y_path="",
                )
            ],
        )
    )

    assert record is not None
    assert record.overall_verdict == EvaluationVerdict.UNCERTAIN
    assert "judge exploded" in store.eval_round_result_path(1).read_text(encoding="utf-8")
    assert store.round_evaluation_log_path.exists()


def test_round_evaluator_fail_closed_raises_without_record(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-fail-closed", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    app_config.evaluation.fail_open = False
    service = RoundEvaluatorService(
        app_config=app_config,
        agent_config=make_agent_config(tmp_path),
        artifact_store=store,
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )

    async def _raise_impl(self, **kwargs):
        del self, kwargs
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(service, "_evaluate_round_impl", MethodType(_raise_impl, service))
    with pytest.raises(RuntimeError, match="judge exploded"):
        asyncio.run(
            service.evaluate_round(
                round_request="Open Calculator",
                round_num=1,
                step_logs=[],
            )
        )
    assert not store.round_evaluation_log_path.exists()
