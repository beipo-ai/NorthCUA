"""Evaluator terminal-decision regression tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nexau.core.messages import Message, Role, TextBlock

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.evaluator import (
    AgentKitEvaluatorRunner,
    EvaluationEvidenceManifest,
    EvaluationEvidenceRef,
    EvaluationJudgeMetadata,
    EvaluatorArtifactLayout,
    EvaluatorArtifactWriter,
    EvaluatorRunContext,
    EvaluatorSpec,
    RoundEvaluationBundle,
    RoundEvaluationRecord,
    RoundEvaluationReport,
)
from northcua.core.enums import EvaluationControllability, EvaluationEvidenceKind, EvaluationOutcome, EvaluationVerdict
from northcua.core.terminal import AgentTerminalAction, AgentTerminalReason


def _report_payload(evidence_ref: str = "E001") -> dict[str, object]:
    return {
        "overall_verdict": EvaluationVerdict.SUCCESS.value,
        "process_rating": 5,
        "outcome_label": EvaluationOutcome.SUCCESS.value,
        "completion_rating": 5,
        "confidence_rating": 5,
        "controllability": EvaluationControllability.CONTROLLABLE.value,
        "reason": "Evidence supports completion.",
        "criteria": [
            {
                "criterion_id": "criterion_1",
                "score": 5,
                "reason": "The final state is correct.",
                "evidence_refs": [evidence_ref],
            }
        ],
        "diagnoses": [],
        "side_effects": [],
        "safety": {
            "risk_level": "none",
            "findings": [],
            "reason": "No safety issues observed.",
            "evidence_refs": [],
        },
        "efficiency": {
            "rating": 5,
            "unnecessary_step_count": 0,
            "repeated_action_count": 0,
            "detour_count": 0,
            "direct_path_available": False,
            "reason": "The agent used a direct path.",
            "evidence_refs": [],
        },
        "first_divergence_step": None,
        "first_divergence_reason": "",
        "first_divergence_evidence_refs": [],
        "needs_more_evidence": False,
    }


def _bundle_with_images() -> RoundEvaluationBundle:
    return RoundEvaluationBundle(
        session_id="sess",
        round_num=1,
        round_request="Finish the task",
        session_request="Finish the task",
        screenshot_refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Step 1 screenshot",
                summary="early state",
                path="/tmp/1.png",
                session_step=1,
                round_step=1,
            )
        ],
    )


@pytest.mark.parametrize(
    ("verdict", "reported_need", "expected_need"),
    [
        (EvaluationVerdict.SUCCESS, True, True),
        (EvaluationVerdict.UNCERTAIN, False, True),
        (EvaluationVerdict.SUCCESS, False, False),
    ],
)
def test_round_record_preserves_reported_evidence_need(
    verdict: EvaluationVerdict,
    reported_need: bool,
    expected_need: bool,
) -> None:
    payload = _report_payload()
    payload["overall_verdict"] = verdict.value
    payload["needs_more_evidence"] = reported_need
    if verdict == EvaluationVerdict.UNCERTAIN:
        payload["outcome_label"] = EvaluationOutcome.UNCERTAIN.value
    report = RoundEvaluationReport.model_validate(payload)

    record = RoundEvaluationRecord.from_report(
        session_id="session",
        round_num=1,
        round_request="task",
        report=report,
        evidence_manifest=EvaluationEvidenceManifest(),
        tool_checks=[],
        judge_metadata=EvaluationJudgeMetadata(),
    )

    assert record.judge_metadata.needs_more_evidence is expected_need


def test_runner_reprompts_no_tool_call_until_step_budget(tmp_path: Path) -> None:
    """Evaluator no-tool responses get a repair prompt before max-step failure."""

    class _FakeStepExecutor:
        def __init__(self) -> None:
            self.calls = 0
            self.last_messages: list[Message] = []

        async def run_step(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            envelope = kwargs["build_envelope"]([], None)  # type: ignore[index]
            self.last_messages = list(kwargs["envelope_messages"](envelope))  # type: ignore[index]
            return SimpleNamespace(
                response_text="I am done without tools",
                meta={},
                trace_state=TraceCaptureState(assistant_text="I am done without tools"),
            )

    executor = _FakeStepExecutor()
    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator", max_steps=2),
        artifact_writer=EvaluatorArtifactWriter(
            EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="fake_eval")
        ),
        model_step_executor=executor,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="final_report"):
        asyncio.run(
            runner.run_round(
                context=EvaluatorRunContext(
                    bundle=_bundle_with_images(),
                    evidence_manifest=EvaluationEvidenceManifest(),
                    artifact_root=tmp_path,
                    step_trace_by_session_step={},
                ),
                messages=[Message(role=Role.USER, content=[TextBlock(text="evaluate")])],
            )
        )

    runtime_log = tmp_path / "evaluations" / "fake_eval" / "round_0001" / "runtime.jsonl"
    terminal_events = [
        json.loads(line)["terminal_decision"]
        for line in runtime_log.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == "agentkit_evaluator_terminal_decision"
    ]
    assert executor.calls == 2
    assert any("required final_report tool" in block.text for block in executor.last_messages[-1].content)
    assert terminal_events[-1]["reason"] == AgentTerminalReason.MAX_STEPS.value


def test_runner_recovers_from_reasoning_only_response_with_final_report_retry(tmp_path: Path) -> None:
    """A reasoning-only evaluator response can recover on the next model step."""

    class _FakeStepExecutor:
        def __init__(self) -> None:
            self.calls = 0
            self.second_call_messages: list[Message] = []

        async def run_step(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            envelope = kwargs["build_envelope"]([], None)  # type: ignore[index]
            if self.calls == 1:
                return SimpleNamespace(
                    response_text="",
                    meta={},
                    trace_state=TraceCaptureState(reasoning="I know the report but forgot the tool."),
                )
            self.second_call_messages = list(kwargs["envelope_messages"](envelope))  # type: ignore[index]
            return SimpleNamespace(
                response_text="calling final_report",
                meta={"pending_tool_calls": [{"id": "call-final", "name": "final_report", "input": _report_payload()}]},
                trace_state=TraceCaptureState(assistant_text="calling final_report"),
            )

    executor = _FakeStepExecutor()
    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator", max_steps=3),
        artifact_writer=EvaluatorArtifactWriter(
            EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="fake_eval")
        ),
        model_step_executor=executor,  # type: ignore[arg-type]
    )

    result = asyncio.run(
        runner.run_round(
            context=EvaluatorRunContext(
                bundle=_bundle_with_images(),
                evidence_manifest=EvaluationEvidenceManifest(
                    refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final")]
                ),
                artifact_root=tmp_path,
                step_trace_by_session_step={},
            ),
            messages=[Message(role=Role.USER, content=[TextBlock(text="evaluate")])],
        )
    )

    assert executor.calls == 2
    assert result.terminal_decision.action == AgentTerminalAction.FINISH
    assert any("required final_report tool" in block.text for block in executor.second_call_messages[-1].content)


def test_runner_repairs_invalid_final_report_evidence_refs(tmp_path: Path) -> None:
    """Invalid final_report evidence ids get a finalizer-only repair turn with valid refs."""

    class _FakeStepExecutor:
        def __init__(self) -> None:
            self.calls = 0
            self.messages_by_call: list[list[Message]] = []
            self.tool_names_by_call: list[list[str]] = []

        async def run_step(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            agent_config = SimpleNamespace(max_iterations=0, global_storage={})
            kwargs["configure_agent"](agent_config)  # type: ignore[index]
            self.tool_names_by_call.append([tool.name for tool in agent_config.tools])
            envelope = kwargs["build_envelope"]([], None)  # type: ignore[index]
            self.messages_by_call.append(list(kwargs["envelope_messages"](envelope)))  # type: ignore[index]
            if self.calls == 1:
                return SimpleNamespace(
                    response_text="calling invalid final_report",
                    meta={
                        "pending_tool_calls": [
                            {"id": "bad-final", "name": "final_report", "input": _report_payload("E999")}
                        ]
                    },
                    trace_state=TraceCaptureState(assistant_text="calling invalid final_report"),
                )
            return SimpleNamespace(
                response_text="calling repaired final_report",
                meta={"pending_tool_calls": [{"id": "good-final", "name": "final_report", "input": _report_payload()}]},
                trace_state=TraceCaptureState(assistant_text="calling repaired final_report"),
            )

    executor = _FakeStepExecutor()
    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator", max_steps=3),
        artifact_writer=EvaluatorArtifactWriter(
            EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="fake_eval")
        ),
        model_step_executor=executor,  # type: ignore[arg-type]
    )

    result = asyncio.run(
        runner.run_round(
            context=EvaluatorRunContext(
                bundle=_bundle_with_images(),
                evidence_manifest=EvaluationEvidenceManifest(
                    refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final screenshot")]
                ),
                artifact_root=tmp_path,
                step_trace_by_session_step={},
            ),
            messages=[Message(role=Role.SYSTEM, content=[TextBlock(text="Evaluate.")])],
        )
    )

    repair_text = "\n".join(
        block.text
        for message in executor.messages_by_call[1]
        for block in message.content
        if isinstance(block, TextBlock)
    )

    assert result.terminal_decision.action == AgentTerminalAction.FINISH
    assert executor.calls == 2
    assert executor.tool_names_by_call[1] == ["final_report"]
    assert "final_report_tool_error" in repair_text
    assert "Invalid evidence refs: E999" in repair_text
    assert "Available evidence refs" in repair_text
    assert "E001" in repair_text


def test_runner_uses_compact_finalizer_only_repair_after_artifact_history(tmp_path: Path) -> None:
    """Finalizer repair does not replay bulky artifact history after evidence collection."""

    class _FakeStepExecutor:
        def __init__(self) -> None:
            self.calls = 0
            self.messages_by_call: list[list[Message]] = []
            self.tool_names_by_call: list[list[str]] = []

        async def run_step(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            agent_config = SimpleNamespace(max_iterations=0, global_storage={})
            kwargs["configure_agent"](agent_config)  # type: ignore[index]
            self.tool_names_by_call.append([tool.name for tool in agent_config.tools])
            envelope = kwargs["build_envelope"]([], None)  # type: ignore[index]
            self.messages_by_call.append(list(kwargs["envelope_messages"](envelope)))  # type: ignore[index]
            if self.calls == 1:
                return SimpleNamespace(
                    response_text="Need the bundle.",
                    meta={"pending_tool_calls": [{"id": "query-artifacts", "name": "query_artifacts", "input": {}}]},
                    trace_state=TraceCaptureState(assistant_text="Need the bundle."),
                )
            if self.calls == 2:
                return SimpleNamespace(
                    response_text="",
                    meta={},
                    trace_state=TraceCaptureState(
                        reasoning="The evidence supports success; now I should submit the typed report."
                    ),
                )
            return SimpleNamespace(
                response_text="calling final_report",
                meta={"pending_tool_calls": [{"id": "call-final", "name": "final_report", "input": _report_payload()}]},
                trace_state=TraceCaptureState(assistant_text="calling final_report"),
            )

    executor = _FakeStepExecutor()
    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator", max_steps=4),
        artifact_writer=EvaluatorArtifactWriter(
            EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="fake_eval")
        ),
        model_step_executor=executor,  # type: ignore[arg-type]
    )

    result = asyncio.run(
        runner.run_round(
            context=EvaluatorRunContext(
                bundle=_bundle_with_images(),
                evidence_manifest=EvaluationEvidenceManifest(
                    refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final")]
                ),
                artifact_root=tmp_path,
                step_trace_by_session_step={},
            ),
            messages=[Message(role=Role.SYSTEM, content=[TextBlock(text="Always call final_report.")])],
        )
    )

    repair_messages = executor.messages_by_call[2]
    repair_text = "\n".join(
        block.text for message in repair_messages for block in message.content if isinstance(block, TextBlock)
    )

    assert result.terminal_decision.action == AgentTerminalAction.FINISH
    assert executor.calls == 3
    assert "query_artifacts" in executor.tool_names_by_call[0]
    assert executor.tool_names_by_call[2] == ["final_report"]
    assert "Finalizer repair context" in repair_text
    assert "The evidence supports success" in repair_text
    assert "query_artifacts" not in repair_text
    assert "Tool result" not in repair_text


def test_runner_forces_finalizer_after_evidence_step_limit(tmp_path: Path) -> None:
    """The evidence cap preserves full history and reserves a final_report step."""

    long_evidence_note = "evidence-" + ("x" * 13_000) + "-full-history-tail"

    class _FakeStepExecutor:
        def __init__(self) -> None:
            self.calls = 0
            self.messages_by_call: list[list[Message]] = []
            self.tool_names_by_call: list[list[str]] = []
            self.tool_choices_by_call: list[object] = []

        async def run_step(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            agent_config = SimpleNamespace(max_iterations=0, global_storage={})
            kwargs["configure_agent"](agent_config)  # type: ignore[index]
            tool_names = [tool.name for tool in agent_config.tools]
            self.tool_names_by_call.append(tool_names)
            envelope = kwargs["build_envelope"]([], None)  # type: ignore[index]
            self.messages_by_call.append(list(kwargs["envelope_messages"](envelope)))  # type: ignore[index]
            self.tool_choices_by_call.append(envelope.tool_choice)
            if tool_names == ["final_report"]:
                if self.calls == 49:
                    return SimpleNamespace(
                        response_text="calling invalid final_report",
                        meta={
                            "pending_tool_calls": [
                                {"id": "bad-final", "name": "final_report", "input": _report_payload("E999")}
                            ]
                        },
                        trace_state=TraceCaptureState(assistant_text="calling invalid final_report"),
                    )
                return SimpleNamespace(
                    response_text="calling final_report",
                    meta={
                        "pending_tool_calls": [{"id": "call-final", "name": "final_report", "input": _report_payload()}]
                    },
                    trace_state=TraceCaptureState(assistant_text="calling final_report"),
                )
            return SimpleNamespace(
                response_text="Need another artifact.",
                meta={"pending_tool_calls": [{"id": f"query-{self.calls}", "name": "query_artifacts", "input": {}}]},
                trace_state=TraceCaptureState(
                    assistant_text=long_evidence_note if self.calls == 1 else "Need another artifact."
                ),
            )

    executor = _FakeStepExecutor()
    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(
            kind="fake_eval",
            display_name="Fake Evaluator",
            max_steps=50,
            force_finalizer_after_steps=48,
        ),
        artifact_writer=EvaluatorArtifactWriter(
            EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="fake_eval")
        ),
        model_step_executor=executor,  # type: ignore[arg-type]
    )

    result = asyncio.run(
        runner.run_round(
            context=EvaluatorRunContext(
                bundle=_bundle_with_images(),
                evidence_manifest=EvaluationEvidenceManifest(
                    refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final")]
                ),
                artifact_root=tmp_path,
                step_trace_by_session_step={},
            ),
            messages=[Message(role=Role.SYSTEM, content=[TextBlock(text="Evaluate.")])],
        )
    )

    repair_text = "\n".join(
        block.text
        for message in executor.messages_by_call[-1]
        for block in message.content
        if isinstance(block, TextBlock)
    )
    repair_payload = json.dumps(
        [message.model_dump(mode="json") for message in executor.messages_by_call[-1]],
        ensure_ascii=False,
    )

    assert result.terminal_decision.action == AgentTerminalAction.FINISH
    assert executor.calls == 50
    assert all(tool_names != ["final_report"] for tool_names in executor.tool_names_by_call[:48])
    assert executor.tool_names_by_call[48] == ["final_report"]
    assert executor.tool_names_by_call[49] == ["final_report"]
    assert executor.tool_choices_by_call[48:] == ["required", "required"]
    assert "Evidence-gathering step limit reached" in repair_text
    assert "retained without repair-summary compaction" in repair_text
    assert "full-history-tail" in repair_payload
    assert "[truncated" not in repair_payload
    assert "Invalid evidence refs: E999" in repair_payload
    assert "Previous evaluator response before the required final_report tool call" not in repair_text


def test_evidence_limit_reasoning_only_response_keeps_full_history(tmp_path: Path) -> None:
    """A no-tool response at the evidence limit bypasses the 12k repair summary."""

    long_reasoning = "reasoning-" + ("y" * 13_000) + "-reasoning-tail"

    class _FakeStepExecutor:
        def __init__(self) -> None:
            self.calls = 0
            self.finalizer_messages: list[Message] = []
            self.finalizer_tool_choice: object = None

        async def run_step(self, **kwargs: object) -> SimpleNamespace:
            self.calls += 1
            agent_config = SimpleNamespace(max_iterations=0, global_storage={})
            kwargs["configure_agent"](agent_config)  # type: ignore[index]
            envelope = kwargs["build_envelope"]([], None)  # type: ignore[index]
            if self.calls == 1:
                return SimpleNamespace(
                    response_text="",
                    meta={},
                    trace_state=TraceCaptureState(reasoning=long_reasoning),
                )
            self.finalizer_messages = list(kwargs["envelope_messages"](envelope))  # type: ignore[index]
            self.finalizer_tool_choice = envelope.tool_choice
            return SimpleNamespace(
                response_text="calling final_report",
                meta={"pending_tool_calls": [{"id": "call-final", "name": "final_report", "input": _report_payload()}]},
                trace_state=TraceCaptureState(assistant_text="calling final_report"),
            )

    executor = _FakeStepExecutor()
    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(
            kind="fake_eval",
            display_name="Fake Evaluator",
            max_steps=3,
            force_finalizer_after_steps=1,
        ),
        artifact_writer=EvaluatorArtifactWriter(
            EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="fake_eval")
        ),
        model_step_executor=executor,  # type: ignore[arg-type]
    )

    result = asyncio.run(
        runner.run_round(
            context=EvaluatorRunContext(
                bundle=_bundle_with_images(),
                evidence_manifest=EvaluationEvidenceManifest(
                    refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final")]
                ),
                artifact_root=tmp_path,
                step_trace_by_session_step={},
            ),
            messages=[Message(role=Role.SYSTEM, content=[TextBlock(text="Evaluate.")])],
        )
    )

    finalizer_payload = json.dumps(
        [message.model_dump(mode="json") for message in executor.finalizer_messages],
        ensure_ascii=False,
    )
    assert result.terminal_decision.action == AgentTerminalAction.FINISH
    assert executor.calls == 2
    assert executor.finalizer_tool_choice == "required"
    assert "reasoning-tail" in finalizer_payload
    assert "[truncated" not in finalizer_payload


def test_runner_persists_full_evaluator_llm_trace(tmp_path: Path) -> None:
    """Evaluator traces retain provider payloads instead of redacted placeholders."""

    image_data = "data:image/png;base64,abc123"

    class _FakeStepExecutor:
        async def run_step(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                response_text="calling final_report",
                meta={"pending_tool_calls": [{"id": "call-final", "name": "final_report", "input": _report_payload()}]},
                trace_state=TraceCaptureState(
                    request_payload={
                        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": image_data}]}]
                    },
                    response_payload={
                        "output": [
                            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Looks done."}]},
                            {
                                "type": "function_call",
                                "call_id": "call-final",
                                "name": "final_report",
                                "arguments": json.dumps(_report_payload()),
                            },
                        ]
                    },
                    reasoning="Looks done.",
                    assistant_text="calling final_report",
                    usage={"input_tokens": 11, "output_tokens": 7},
                ),
            )

    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator", max_steps=1),
        artifact_writer=EvaluatorArtifactWriter(
            EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="fake_eval")
        ),
        model_step_executor=_FakeStepExecutor(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        runner.run_round(
            context=EvaluatorRunContext(
                bundle=_bundle_with_images(),
                evidence_manifest=EvaluationEvidenceManifest(
                    refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final")]
                ),
                artifact_root=tmp_path,
                step_trace_by_session_step={},
            ),
            messages=[Message(role=Role.USER, content=[TextBlock(text="evaluate")])],
        )
    )

    trace_log = tmp_path / "evaluations" / "fake_eval" / "round_0001" / "llm_trace.jsonl"
    row = json.loads(trace_log.read_text(encoding="utf-8").splitlines()[0])

    assert result.terminal_decision.action == AgentTerminalAction.FINISH
    assert row["source"] == "evaluator"
    assert row["session_id"] == "sess"
    assert row["usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert row["llm_trace"]["request"]["messages"][0]["content"][0]["image_url"] == image_data
    assert row["llm_trace"]["reasoning"] == "Looks done."
    assert row["llm_trace"]["tool_executions"][0]["tool_name"] == "final_report"


def test_runner_records_final_report_tool_error_terminal_decision(tmp_path: Path) -> None:
    """Invalid final_report payloads become shared tool_error decisions."""

    class _FakeStepExecutor:
        async def run_step(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                response_text="invalid final",
                meta={"pending_tool_calls": [{"id": "bad-final", "name": "final_report", "input": {}}]},
                trace_state=TraceCaptureState(assistant_text="invalid final"),
            )

    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator", max_steps=1),
        artifact_writer=EvaluatorArtifactWriter(
            EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="fake_eval")
        ),
        model_step_executor=_FakeStepExecutor(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="final_report"):
        asyncio.run(
            runner.run_round(
                context=EvaluatorRunContext(
                    bundle=_bundle_with_images(),
                    evidence_manifest=EvaluationEvidenceManifest(),
                    artifact_root=tmp_path,
                    step_trace_by_session_step={},
                ),
                messages=[Message(role=Role.USER, content=[TextBlock(text="evaluate")])],
            )
        )

    runtime_log = tmp_path / "evaluations" / "fake_eval" / "round_0001" / "runtime.jsonl"
    assert '"reason": "tool_error"' in runtime_log.read_text(encoding="utf-8")


def test_runner_records_max_step_terminal_decision(tmp_path: Path) -> None:
    """Evaluator step exhaustion is recorded as a shared max_steps decision."""

    class _FakeStepExecutor:
        async def run_step(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                response_text="need more evidence",
                meta={"pending_tool_calls": [{"id": "read-bundle", "name": "read_round_bundle", "input": {}}]},
                trace_state=TraceCaptureState(assistant_text="need more evidence"),
            )

    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator", max_steps=2),
        artifact_writer=EvaluatorArtifactWriter(
            EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind="fake_eval")
        ),
        model_step_executor=_FakeStepExecutor(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="final_report"):
        asyncio.run(
            runner.run_round(
                context=EvaluatorRunContext(
                    bundle=_bundle_with_images(),
                    evidence_manifest=EvaluationEvidenceManifest(),
                    artifact_root=tmp_path,
                    step_trace_by_session_step={},
                ),
                messages=[Message(role=Role.USER, content=[TextBlock(text="evaluate")])],
            )
        )

    runtime_log = tmp_path / "evaluations" / "fake_eval" / "round_0001" / "runtime.jsonl"
    terminal_events = [
        json.loads(line)["terminal_decision"]
        for line in runtime_log.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == "agentkit_evaluator_terminal_decision"
    ]
    assert terminal_events[-1]["reason"] == AgentTerminalReason.MAX_STEPS.value
    assert terminal_events[-1]["metadata"] == {"scope": "round", "limit": 2, "current": 2}
