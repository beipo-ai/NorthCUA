"""AgentKit evaluator runtime contracts and safety gates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock
from pydantic import ValidationError

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.evaluator import (
    AgentKitEvaluatorRunner,
    ArtifactEvaluationRequest,
    ArtifactEvaluationRunnerBase,
    EvaluationCriterion,
    EvaluationEvidenceManifest,
    EvaluationEvidenceRef,
    EvaluatorArtifactLayout,
    EvaluatorArtifactWriter,
    EvaluatorRunContext,
    EvaluatorSpec,
    EvaluatorToolRef,
    EvidencePolicy,
    EvidenceSelectionReason,
    EvidenceSelector,
    FinalReportState,
    FinalReportToolFactory,
    RemoteEvaluatorClientToolDispatcher,
    RoundEvaluationBundle,
    ToolPlacementPolicy,
    standard_client_probe_tools,
)
from northcua.agentkit.evaluator.routing import PlacementResolution, ToolPlacementResolver
from northcua.core.config import NorthCUAAppConfig
from northcua.core.enums import (
    EvaluationControllability,
    EvaluationEvidenceKind,
    EvaluationOutcome,
    EvaluationSchedule,
    EvaluationToolPlacement,
    EvaluationVerdict,
    ResultStatus,
)
from northcua.core.models import CommandExecutionResultRecord
from northcua.core.terminal import AgentTerminalAction, AgentTerminalReason
from northcua.transport.ws.message_seed import SUPPRESS_RAW_AGENT_MCP_SERVERS_KEY
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult


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
            ),
            EvaluationEvidenceRef(
                id="E002",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Terminal verification screenshot",
                summary="verified final state",
                path="/tmp/2.png",
                session_step=2,
                round_step=2,
                metadata={"pinned": True},
            ),
            EvaluationEvidenceRef(
                id="E003",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Step 3 screenshot",
                summary="latest state",
                path="/tmp/3.png",
                session_step=3,
                round_step=3,
            ),
        ],
    )


def test_runner_run_round_uses_final_report_tool(tmp_path) -> None:
    class _FakeStepExecutor:
        async def run_step(self, **kwargs):
            agent_config = SimpleNamespace(max_iterations=0)
            kwargs["configure_agent"](agent_config)
            envelope = kwargs["build_envelope"]([], None)
            assert kwargs["envelope_messages"](envelope)
            return SimpleNamespace(
                response_text="calling final_report",
                meta={
                    "pending_tool_calls": [
                        {
                            "id": "call-final",
                            "name": "final_report",
                            "input": _report_payload(),
                        }
                    ]
                },
                trace_state=TraceCaptureState(assistant_text="calling final_report"),
            )

    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator"),
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

    assert result.record.overall_verdict == EvaluationVerdict.SUCCESS
    assert result.terminal_decision.action == AgentTerminalAction.FINISH
    assert result.terminal_decision.reason == AgentTerminalReason.COMPLETED
    assert (tmp_path / "evaluations" / "fake_eval" / "round_0001" / "final_report.json").exists()


def test_runner_executes_artifact_tool_then_final_report(tmp_path) -> None:
    class _FakeStepExecutor:
        def __init__(self) -> None:
            self.calls = 0
            self.step_indices: list[int] = []

        async def run_step(self, **kwargs):
            self.calls += 1
            self.step_indices.append(kwargs["step_index"])
            agent_config = SimpleNamespace(max_iterations=0, global_storage={})
            kwargs["configure_agent"](agent_config)
            assert {tool.name for tool in agent_config.tools} >= {"read_step", "final_report"}
            assert agent_config.mcp_servers == []
            assert agent_config.global_storage[SUPPRESS_RAW_AGENT_MCP_SERVERS_KEY] is True
            if self.calls == 1:
                return SimpleNamespace(
                    response_text="need trace",
                    meta={
                        "pending_tool_calls": [{"id": "call-read", "name": "read_step", "input": {"session_step": 1}}]
                    },
                    trace_state=TraceCaptureState(assistant_text="need trace"),
                )
            return SimpleNamespace(
                response_text="final",
                meta={"pending_tool_calls": [{"id": "call-final", "name": "final_report", "input": _report_payload()}]},
                trace_state=TraceCaptureState(assistant_text="final"),
            )

    executor = _FakeStepExecutor()
    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator", max_steps=3),
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
                step_trace_by_session_step={1: {"assistant_text": "done"}},
            ),
            messages=[Message(role=Role.USER, content=[TextBlock(text="evaluate")])],
        )
    )

    assert executor.calls == 2
    assert executor.step_indices == [1, 2]
    assert result.record.overall_verdict == EvaluationVerdict.SUCCESS


def test_runner_builds_remote_client_probe_with_explicit_dispatcher(tmp_path) -> None:
    class _Dispatcher:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def dispatch(self, *, tool, parameters, remote):
            self.calls.append({"tool": tool.name, "parameters": parameters, "remote": remote})
            return {"status": "success", "artifact_ref": "client://shot"}

    spec = EvaluatorSpec.with_default_tools(
        kind="fake_eval",
        display_name="Fake Evaluator",
        schedule=EvaluationSchedule.AFTER_ROUND_SYNC,
        placement_policy=ToolPlacementPolicy(
            allowed_placements=frozenset(
                {
                    EvaluationToolPlacement.SERVER_ARTIFACT,
                    EvaluationToolPlacement.FINALIZER,
                    EvaluationToolPlacement.CLIENT_PROBE,
                }
            ),
            allow_client_live_probe=True,
        ),
        tools=[standard_client_probe_tools()[0]],
    )
    runner = AgentKitEvaluatorRunner(spec=spec)
    context = EvaluatorRunContext(
        bundle=_bundle_with_images(),
        evidence_manifest=EvaluationEvidenceManifest(),
        artifact_root=tmp_path,
        step_trace_by_session_step={},
        remote=True,
    )

    with pytest.raises(ValueError, match="client_tool_dispatcher"):
        runner.build_toolset(context)

    dispatcher = _Dispatcher()
    toolset = runner.build_toolset(context, client_tool_dispatcher=dispatcher)
    tool = next(tool for tool in toolset.tools if tool.name == "client_screenshot")
    result = asyncio.run(tool.execute_async(all_screens=False))

    assert result["status"] == "success"
    assert dispatcher.calls == [{"tool": "client_screenshot", "parameters": {"all_screens": False}, "remote": True}]


def test_final_report_rejects_invalid_first_divergence_refs() -> None:
    factory = FinalReportToolFactory(
        evidence_manifest=EvaluationEvidenceManifest(
            refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final")]
        )
    )
    payload = _report_payload()
    payload["first_divergence_step"] = 2
    payload["first_divergence_reason"] = "The agent diverged."
    payload["first_divergence_evidence_refs"] = ["E404"]

    result = factory._submit_final_report(**payload)

    assert result["status"] == "failure"
    assert "E404" in str(result["summary"])
