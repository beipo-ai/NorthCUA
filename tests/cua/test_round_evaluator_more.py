"""Regression coverage for desktop round evaluator artifacts and judge orchestration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from nexau.core.messages import ImageBlock

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.evaluator.tools import ArtifactInspectionToolFactory as EvaluationToolFactory
from northcua.agents.desktop.evaluation.protocol import (
    EvaluationCriterion,
    EvaluationEvidenceManifest,
    EvaluationEvidenceRef,
)
from northcua.agents.desktop.evaluation.service import RoundEvaluatorService, _PhaseRunResult
from northcua.agents.desktop.evaluation.spec import desktop_round_evaluator_spec
from northcua.agents.desktop.round_execution.evaluation_dispatch import RoundEvaluationScheduler
from northcua.core.config import ArtifactConfig, NorthCUAAppConfig
from northcua.core.enums import (
    EvaluationEvidenceKind,
    EvaluationSchedule,
    EvaluationToolCheckStatus,
    EvaluationVerdict,
    TaskStatus,
)
from northcua.runtime.artifacts.store import NorthCUAArtifactStore
from tests.cua.evaluation_helpers import agent_config as make_agent_config
from tests.cua.evaluation_helpers import image, step


def test_async_round_evaluator_fail_closed_surfaces_completed_failure(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-async-fail", ArtifactConfig())
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
        raise RuntimeError("async judge exploded")

    monkeypatch.setattr(service, "_evaluate_round_impl", MethodType(_raise_impl, service))

    async def _run() -> None:
        service.schedule_round_evaluation(
            round_request="Open Calculator",
            round_num=1,
            step_logs=[],
            terminal=False,
        )
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="async judge exploded"):
            await service.drain()

    asyncio.run(_run())


def test_async_round_evaluator_scheduler_skips_nonterminal_steps(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-skip-nonterminal", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    app_config.evaluation.schedule = EvaluationSchedule.AFTER_ROUND_ASYNC

    class _FakeEvaluatorService:
        enabled = True

        def __init__(self) -> None:
            self.scheduled = 0

        def schedule_round_evaluation(self, **kwargs):
            del kwargs
            self.scheduled += 1
            raise AssertionError("non-terminal async steps must not enqueue evaluator work")

        async def drain(self) -> None:
            return None

    service = _FakeEvaluatorService()
    scheduler = RoundEvaluationScheduler(
        round_evaluator_service=service,  # type: ignore[arg-type]
        app_config=app_config,
        artifact_store=store,
    )
    step_log = step(
        session_id="sess-skip-nonterminal",
        session_step=1,
        round_num=1,
        round_step=1,
        request="Open Calculator",
        result="still working",
        status=TaskStatus.CONTINUE,
        clean_screenshot_path="",
        tool_image_path="",
        a11y_path="",
    )

    assert asyncio.run(scheduler.maybe_run_round_evaluation(step_log, [step_log])) is False

    assert service.scheduled == 0


def test_sync_round_evaluator_scheduler_skips_nonterminal_steps(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-sync-skip-nonterminal", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    app_config.evaluation.schedule = EvaluationSchedule.AFTER_ROUND_SYNC

    class _FakeEvaluatorService:
        enabled = True

        def __init__(self) -> None:
            self.calls = 0

        async def evaluate_round_and_maybe_rollup(self, **kwargs):
            del kwargs
            self.calls += 1
            raise AssertionError("non-terminal sync steps must not run evaluator work")

        async def drain(self) -> None:
            return None

    service = _FakeEvaluatorService()
    scheduler = RoundEvaluationScheduler(
        round_evaluator_service=service,  # type: ignore[arg-type]
        app_config=app_config,
        artifact_store=store,
    )
    step_log = step(
        session_id="sess-sync-skip-nonterminal",
        session_step=1,
        round_num=1,
        round_step=1,
        request="Open Calculator",
        result="still working",
        status=TaskStatus.CONTINUE,
        clean_screenshot_path="",
        tool_image_path="",
        a11y_path="",
    )

    assert asyncio.run(scheduler.maybe_run_round_evaluation(step_log, [step_log])) is False

    assert service.calls == 0


def test_async_round_evaluator_scheduler_reports_not_completed(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-async-return", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    app_config.evaluation.schedule = EvaluationSchedule.AFTER_ROUND_ASYNC

    class _FakeEvaluatorService:
        enabled = True

        def __init__(self) -> None:
            self.scheduled = 0

        def schedule_round_evaluation(self, **kwargs):
            del kwargs
            self.scheduled += 1
            task = asyncio.get_running_loop().create_future()
            task.set_result(None)
            return task

        def effective_schedule(self) -> EvaluationSchedule:
            return EvaluationSchedule.AFTER_ROUND_ASYNC

    service = _FakeEvaluatorService()
    scheduler = RoundEvaluationScheduler(
        round_evaluator_service=service,  # type: ignore[arg-type]
        app_config=app_config,
        artifact_store=store,
    )
    step_log = step(
        session_id="sess-async-return",
        session_step=1,
        round_num=1,
        round_step=1,
        request="Open Calculator",
        result="done",
        status=TaskStatus.FINISH,
        clean_screenshot_path="",
        tool_image_path="",
        a11y_path="",
    )

    assert asyncio.run(scheduler.maybe_run_round_evaluation(step_log, [step_log])) is False
    assert service.scheduled == 1


def test_sync_round_evaluator_scheduler_reports_completed(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-sync-return", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    app_config.evaluation.schedule = EvaluationSchedule.AFTER_ROUND_SYNC

    class _FakeEvaluatorService:
        enabled = True

        def __init__(self) -> None:
            self.calls = 0

        async def evaluate_round_and_maybe_rollup(self, **kwargs):
            del kwargs
            self.calls += 1

        def effective_schedule(self) -> EvaluationSchedule:
            return EvaluationSchedule.AFTER_ROUND_SYNC

    service = _FakeEvaluatorService()
    scheduler = RoundEvaluationScheduler(
        round_evaluator_service=service,  # type: ignore[arg-type]
        app_config=app_config,
        artifact_store=store,
    )
    step_log = step(
        session_id="sess-sync-return",
        session_step=1,
        round_num=1,
        round_step=1,
        request="Open Calculator",
        result="done",
        status=TaskStatus.FINISH,
        clean_screenshot_path="",
        tool_image_path="",
        a11y_path="",
    )

    assert asyncio.run(scheduler.maybe_run_round_evaluation(step_log, [step_log])) is True
    assert service.calls == 1


def test_async_round_evaluator_scheduler_runs_step_limit_terminal_eval(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-step-limit-eval", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    app_config.evaluation.schedule = EvaluationSchedule.AFTER_ROUND_ASYNC

    class _FakeEvaluatorService:
        enabled = True

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def schedule_round_evaluation(self, **kwargs):
            self.calls.append(kwargs)
            task = asyncio.create_task(asyncio.sleep(0))
            return task

        async def drain(self) -> None:
            return None

    service = _FakeEvaluatorService()
    scheduler = RoundEvaluationScheduler(
        round_evaluator_service=service,  # type: ignore[arg-type]
        app_config=app_config,
        artifact_store=store,
    )
    step_log = step(
        session_id="sess-step-limit-eval",
        session_step=2,
        round_num=1,
        round_step=2,
        request="Open Calculator",
        result="still working",
        status=TaskStatus.CONTINUE,
        clean_screenshot_path="",
        tool_image_path="",
        a11y_path="",
    )

    async def _run() -> None:
        await scheduler.run_step_limit_evaluation(step_log, [step_log])
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert service.calls == [
        {
            "round_request": "Open Calculator",
            "round_num": 1,
            "step_logs": [step_log],
            "terminal": True,
            "client_tool_dispatcher": None,
        }
    ]


def test_round_evaluator_timeout_fail_open_writes_uncertain_record(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-timeout", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    app_config.evaluation.fail_open = True
    app_config.evaluation.round_timeout_seconds = 0.01
    service = RoundEvaluatorService(
        app_config=app_config,
        agent_config=make_agent_config(tmp_path),
        artifact_store=store,
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )

    async def _slow_impl(self, **kwargs):
        del self, kwargs
        await asyncio.sleep(1)
        raise AssertionError("unreachable")

    monkeypatch.setattr(service, "_evaluate_round_impl", MethodType(_slow_impl, service))
    record = asyncio.run(
        service.evaluate_round(
            round_request="Open Calculator",
            round_num=1,
            step_logs=[],
        )
    )

    assert record is not None
    assert record.overall_verdict == EvaluationVerdict.UNCERTAIN
    assert "timed out" in record.reason


def test_terminal_evaluator_context_error_becomes_pending(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-eval-context", ArtifactConfig())
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
    attempts: list[str] = []

    async def _raise_context_impl(self, **kwargs):
        del self
        attempts.append(str(kwargs.get("projection_level")))
        raise RuntimeError("context length exceeded: 249426 > 182160")

    terminal_step = step(
        session_id="sess-eval-context",
        session_step=42,
        round_num=1,
        round_step=42,
        request="Finish the task",
        result="Task complete.",
        status=TaskStatus.FINISH,
        clean_screenshot_path="",
        tool_image_path="",
        a11y_path="",
    )
    monkeypatch.setattr(service, "_evaluate_round_impl", MethodType(_raise_context_impl, service))

    record = asyncio.run(
        service.evaluate_round(
            round_request="Finish the task",
            round_num=1,
            step_logs=[terminal_step],
            terminal=True,
        )
    )

    assert record is not None
    assert record.overall_verdict == EvaluationVerdict.UNCERTAIN
    assert attempts == ["standard", "tight"]
    evaluation = terminal_step.terminal_decision["metadata"]["evaluation"]
    assert evaluation["status"] == "pending"
    assert evaluation["attempts"] == 2
    assert evaluation["projection_level"] == "tight"


def test_terminal_evaluator_timeout_does_not_start_tight_retry(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-eval-timeout", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    app_config.evaluation.fail_open = False
    app_config.evaluation.round_timeout_seconds = 0.01
    service = RoundEvaluatorService(
        app_config=app_config,
        agent_config=make_agent_config(tmp_path),
        artifact_store=store,
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )
    attempts: list[str] = []

    async def _slow_impl(self, **kwargs):
        del self
        attempts.append(str(kwargs.get("projection_level")))
        await asyncio.sleep(1)

    terminal_step = step(
        session_id="sess-eval-timeout",
        session_step=42,
        round_num=1,
        round_step=42,
        request="Finish the task",
        result="Task complete.",
        status=TaskStatus.FINISH,
        clean_screenshot_path="",
        tool_image_path="",
        a11y_path="",
    )
    monkeypatch.setattr(service, "_evaluate_round_impl", MethodType(_slow_impl, service))

    record = asyncio.run(
        service.evaluate_round(
            round_request="Finish the task",
            round_num=1,
            step_logs=[terminal_step],
            terminal=True,
        )
    )

    assert record is not None
    assert record.overall_verdict == EvaluationVerdict.UNCERTAIN
    assert attempts == ["standard"]
    evaluation = terminal_step.terminal_decision["metadata"]["evaluation"]
    assert evaluation["status"] == "pending"
    assert evaluation["attempts"] == 1
    assert evaluation["projection_level"] == "standard"


@pytest.mark.parametrize(
    "error_message",
    (
        "context deadline exceeded",
        "request timed out",
        "websocket transport timeout",
    ),
)
def test_terminal_evaluator_wrapped_timeout_does_not_start_tight_retry(
    monkeypatch,
    tmp_path,
    error_message: str,
) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-eval-wrapped-timeout", ArtifactConfig())
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
    attempts: list[str] = []

    async def _raise_wrapped_timeout(self, **kwargs):
        del self
        attempts.append(str(kwargs.get("projection_level")))
        raise RuntimeError(error_message)

    terminal_step = step(
        session_id="sess-eval-wrapped-timeout",
        session_step=42,
        round_num=1,
        round_step=42,
        request="Finish the task",
        result="Task complete.",
        status=TaskStatus.FINISH,
        clean_screenshot_path="",
        tool_image_path="",
        a11y_path="",
    )
    monkeypatch.setattr(service, "_evaluate_round_impl", MethodType(_raise_wrapped_timeout, service))

    record = asyncio.run(
        service.evaluate_round(
            round_request="Finish the task",
            round_num=1,
            step_logs=[terminal_step],
            terminal=True,
        )
    )

    assert record is not None
    assert record.overall_verdict == EvaluationVerdict.UNCERTAIN
    assert attempts == ["standard"]
    evaluation = terminal_step.terminal_decision["metadata"]["evaluation"]
    assert evaluation["status"] == "pending"
    assert evaluation["attempts"] == 1
    assert evaluation["projection_level"] == "standard"


def test_terminal_evaluator_context_retry_then_wrapped_timeout_stops(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-eval-context-timeout", ArtifactConfig())
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
    attempts: list[str] = []

    async def _context_then_timeout(self, **kwargs):
        del self
        projection_level = str(kwargs.get("projection_level"))
        attempts.append(projection_level)
        if projection_level == "standard":
            raise RuntimeError("context length exceeded: 249426 > 182160")
        raise RuntimeError("request timed out")

    terminal_step = step(
        session_id="sess-eval-context-timeout",
        session_step=42,
        round_num=1,
        round_step=42,
        request="Finish the task",
        result="Task complete.",
        status=TaskStatus.FINISH,
        clean_screenshot_path="",
        tool_image_path="",
        a11y_path="",
    )
    monkeypatch.setattr(service, "_evaluate_round_impl", MethodType(_context_then_timeout, service))

    record = asyncio.run(
        service.evaluate_round(
            round_request="Finish the task",
            round_num=1,
            step_logs=[terminal_step],
            terminal=True,
        )
    )

    assert record is not None
    assert record.overall_verdict == EvaluationVerdict.UNCERTAIN
    assert attempts == ["standard", "tight"]
    evaluation = terminal_step.terminal_decision["metadata"]["evaluation"]
    assert evaluation["status"] == "pending"
    assert evaluation["attempts"] == 2
    assert evaluation["projection_level"] == "tight"


def test_terminal_evaluator_code_bug_still_raises(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-eval-bug", ArtifactConfig())
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

    async def _raise_bug_impl(self, **kwargs):
        del self, kwargs
        raise AssertionError("projection invariant violated")

    terminal_step = step(
        session_id="sess-eval-bug",
        session_step=43,
        round_num=1,
        round_step=43,
        request="Finish the task",
        result="Task complete.",
        status=TaskStatus.FINISH,
        clean_screenshot_path="",
        tool_image_path="",
        a11y_path="",
    )
    monkeypatch.setattr(service, "_evaluate_round_impl", MethodType(_raise_bug_impl, service))

    with pytest.raises(AssertionError, match="projection invariant violated"):
        asyncio.run(
            service.evaluate_round(
                round_request="Finish the task",
                round_num=1,
                step_logs=[terminal_step],
                terminal=True,
            )
        )
    assert "metadata" not in terminal_step.terminal_decision


def test_eval_result_redacts_trace_payloads(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-redact", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    service = RoundEvaluatorService(
        app_config=app_config,
        agent_config=make_agent_config(tmp_path),
        artifact_store=store,
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )

    class _FakeRedactionStepExecutor:
        async def run_step(self, **kwargs):
            agent_config = SimpleNamespace(max_iterations=0)
            kwargs["configure_agent"](agent_config)
            payload = {
                "overall_verdict": "success",
                "process_rating": 5,
                "outcome_label": "success",
                "completion_rating": 5,
                "confidence_rating": 5,
                "controllability": "controllable",
                "reason": "Done.",
                "criteria": [],
                "diagnoses": [],
                "side_effects": [],
                "needs_more_evidence": False,
            }
            return SimpleNamespace(
                response_text="calling final_report",
                meta={"pending_tool_calls": [{"id": "call-final", "name": "final_report", "input": payload}]},
                trace_state=TraceCaptureState(
                    request_payload={
                        "Authorization": "Bearer secret",
                        "image": "data:image/png;base64,abc",
                        "safety_identifier": "user-raw",
                    },
                    response_payload={"api_key": "secret", "nested": {"token": "secret"}},
                    assistant_text="calling final_report",
                ),
            )

    async def _phase_must_not_run(self, **kwargs):
        del self, kwargs
        raise AssertionError("legacy JSON phases must not run")

    service.evaluator_runner.model_step_executor = _FakeRedactionStepExecutor()  # type: ignore[assignment]
    monkeypatch.setattr(service, "_run_phase", MethodType(_phase_must_not_run, service))
    asyncio.run(service.evaluate_round(round_request="Open Calculator", round_num=1, step_logs=[]))

    payload_text = store.eval_round_result_path(1).read_text(encoding="utf-8")
    structured_text = (
        store.root / "evaluations" / app_config.evaluation.default_evaluator_kind / "round_0001" / "llm_trace.jsonl"
    ).read_text(encoding="utf-8")
    structured_row = json.loads(structured_text.splitlines()[0])
    assert "Bearer secret" not in payload_text
    assert "data:image/png;base64,abc" not in payload_text
    assert "data:image/png;base64,abc" in structured_text
    assert structured_row["source"] == "evaluator"
    assert structured_row["request"] == "Open Calculator"
    assert structured_row["llm_trace"]["request"]["image"] == "data:image/png;base64,abc"
    assert structured_row["llm_trace"]["request"]["safety_identifier"] == "[redacted sensitive value]"
    assert "[redacted sensitive value]" in structured_text
    assert "secret" not in structured_text
    assert "user-raw" not in structured_text
