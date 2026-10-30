"""AgentKit evaluator runtime contracts and safety gates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
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
from northcua.agentkit.evaluator.runtime.prompt_history import EvaluatorToolResultContentProjector
from northcua.agentkit.model_step import _apply_provider_tool_choice_override
from northcua.core.config import NorthCUAAppConfig
from northcua.core.enums import (
    DiagnosisCategory,
    EvaluationControllability,
    EvaluationEvidenceKind,
    EvaluationOutcome,
    EvaluationSchedule,
    EvaluationToolCheckStatus,
    EvaluationToolPlacement,
    EvaluationVerdict,
    ResultStatus,
)
from northcua.core.models import CommandExecutionResultRecord
from northcua.core.terminal import AgentTerminalAction, AgentTerminalReason
from northcua.transport.ws.message_seed import SUPPRESS_RAW_AGENT_MCP_SERVERS_KEY
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult


@pytest.mark.parametrize(
    ("api_type", "expected"),
    [
        ("openai_responses", "required"),
        ("openai_chat_completion", "required"),
        ("anthropic_chat_completion", {"type": "any"}),
    ],
)
def test_required_tool_choice_reaches_candidate_provider_params(api_type: str, expected: object) -> None:
    agent_config = AgentConfig(
        llm_config=LLMConfig(
            model="test-model",
            base_url="https://example.invalid",
            api_key="test-key",
            api_type=api_type,
        )
    )

    resolved = _apply_provider_tool_choice_override(agent_config, "required")

    assert resolved == expected
    assert agent_config.llm_config is not None
    assert agent_config.llm_config.to_openai_params()["tool_choice"] == expected


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


def test_evaluator_spec_defaults_include_server_artifacts_and_finalizer() -> None:
    spec = EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator")

    assert [tool.name for tool in spec.visible_tools()] == [
        "read_step",
        "read_trace",
        "query_artifacts",
        "final_report",
    ]
    assert spec.input_scope.value == "round"
    assert spec.max_steps == 50
    assert EvaluationToolPlacement.CLIENT_PROBE in spec.placement_policy.allowed_placements


def test_evaluator_runner_defaults_to_evidence_preserving_history_projector() -> None:
    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator")
    )

    assert isinstance(runner.prompt_history_replayer.tool_result_projector, EvaluatorToolResultContentProjector)


def test_evidence_selector_pins_initial_last_and_required_evidence_before_topk() -> None:
    selector = EvidenceSelector(EvidencePolicy(screenshot_max=2))
    manifest = selector.select(
        bundle=_bundle_with_images(),
        criteria=[EvaluationCriterion(id="criterion_1", name="Final", description="verified final state")],
    )

    refs = {ref.id: ref for ref in manifest.refs}
    assert set(refs) == {"E001", "E002", "E003"}
    assert refs["E001"].pinned
    assert refs["E001"].selection_reason == EvidenceSelectionReason.PINNED_INITIAL_SCREENSHOT.value
    assert refs["E002"].pinned
    assert refs["E002"].selection_reason == EvidenceSelectionReason.PINNED_REQUIRED_EVIDENCE.value
    assert refs["E003"].pinned
    assert refs["E003"].selection_reason == EvidenceSelectionReason.PINNED_LAST_SCREENSHOT.value


def test_agentkit_evidence_selector_prefers_initial_state_over_overlay() -> None:
    selector = EvidenceSelector(EvidencePolicy(screenshot_max=2))
    bundle = RoundEvaluationBundle(
        session_id="sess",
        round_num=1,
        round_request="Finish",
        session_request="Finish",
        screenshot_refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Initial clean screenshot",
                path="/tmp/initial.png",
                session_step=1,
                round_step=1,
                metadata={"variant": "initial_state", "initial_state": True},
            ),
            EvaluationEvidenceRef(
                id="E002",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Initial overlay screenshot",
                path="/tmp/overlay.png",
                session_step=1,
                round_step=1,
                metadata={"variant": "overlay"},
            ),
            EvaluationEvidenceRef(
                id="E003",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Final screenshot",
                path="/tmp/final.png",
                session_step=2,
                round_step=2,
            ),
        ],
    )

    refs = {ref.id: ref for ref in selector.select(bundle=bundle).refs}

    assert refs["E001"].selection_reason == EvidenceSelectionReason.PINNED_INITIAL_SCREENSHOT.value
    assert "E002" not in refs


def test_final_report_tool_validates_schema_refs_and_duplicates() -> None:
    manifest = EvaluationEvidenceManifest(
        refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final screenshot")]
    )
    state = FinalReportState()
    factory = FinalReportToolFactory(evidence_manifest=manifest, state=state)

    assert factory._submit_final_report(**_report_payload())["status"] == "success"
    assert state.submitted
    assert factory._submit_final_report(**_report_payload())["status"] == "failure"

    invalid = FinalReportToolFactory(evidence_manifest=manifest)
    result = invalid._submit_final_report(**_report_payload("E999"))
    assert result["status"] == "failure"
    assert "E999" in str(result["summary"])


def test_final_report_contract_validation_failure_updates_state() -> None:
    state = FinalReportState()
    tool = FinalReportToolFactory(evidence_manifest=EvaluationEvidenceManifest(), state=state).build_tool()

    with pytest.raises(ValueError, match="Invalid parameters"):
        asyncio.run(tool.execute_async())

    assert state.report is None
    assert state.tool_check is not None
    assert state.tool_check.status == EvaluationToolCheckStatus.FAILURE
    assert "Invalid parameters" in state.tool_check.summary


def test_final_report_tool_schema_explains_rating_fields() -> None:
    factory = FinalReportToolFactory(evidence_manifest=EvaluationEvidenceManifest())
    schema = factory.build_tool().input_schema
    full_schema = factory.tool_contract().parameters_schema

    assert schema["properties"]["completion_rating"]["minimum"] == 1
    assert schema["properties"]["completion_rating"]["maximum"] == 5
    assert "1-5" in schema["properties"]["completion_rating"]["description"]
    assert "safety" in schema["properties"]
    assert "efficiency" in schema["properties"]
    assert "first_divergence_step" in schema["properties"]
    assert "$defs" not in schema
    category_def = full_schema["$defs"]["DiagnosisCategory"]
    assert DiagnosisCategory.GROUNDING_LOCALIZATION_ERROR.value in category_def["enum"]
    assert "execution_strategy" not in category_def["enum"]
    diagnosis_ref = full_schema["properties"]["diagnoses"]["items"]["$ref"].split("/")[-1]
    category_description = full_schema["$defs"][diagnosis_ref]["properties"]["category"]["description"]
    assert "task_spec_or_feasibility_error" in category_description
    assert "grounding_localization_error" in category_description
    assert DiagnosisCategory.ENVIRONMENT_OR_EXECUTION_ERROR.value in category_def["enum"]
    assert "environment_or_execution_error" in category_description
    assert "understand the semantic target" in category_description


def test_final_report_rejects_legacy_diagnosis_categories() -> None:
    factory = FinalReportToolFactory(
        evidence_manifest=EvaluationEvidenceManifest(
            refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final screenshot")]
        )
    )
    payload = _report_payload()
    payload["diagnoses"] = [
        {
            "category": "execution_strategy",
            "code": "legacy_category",
            "severity": "major",
            "reason": "Old categories should not be accepted by the new taxonomy.",
            "evidence_refs": ["E001"],
        }
    ]

    result = factory._submit_final_report(**payload)

    assert result["status"] == "failure"
    assert "execution_strategy" in str(result["summary"])


def test_final_report_validates_safety_and_efficiency_evidence_refs() -> None:
    manifest = EvaluationEvidenceManifest(
        refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final screenshot")]
    )
    factory = FinalReportToolFactory(evidence_manifest=manifest)
    payload = _report_payload()
    payload["safety"] = {
        "risk_level": "low",
        "findings": [
            {
                "category": "sensitive_operation",
                "severity": "minor",
                "reason": "Unsupported evidence ref.",
                "evidence_refs": ["E999"],
            }
        ],
        "reason": "Risk depends on unsupported evidence.",
        "evidence_refs": [],
    }
    payload["efficiency"] = {
        "rating": 3,
        "unnecessary_step_count": 1,
        "repeated_action_count": 0,
        "detour_count": 0,
        "direct_path_available": True,
        "reason": "Unsupported evidence ref.",
        "evidence_refs": ["E998"],
    }

    result = factory._submit_final_report(**payload)

    assert result["status"] == "failure"
    assert "E999" in str(result["summary"])
    assert "E998" in str(result["summary"])


def test_tool_placement_coerces_client_probe_async_schedule_to_sync() -> None:
    spec = EvaluatorSpec.with_default_tools(
        kind="fake_eval",
        display_name="Fake Evaluator",
        schedule=EvaluationSchedule.AFTER_ROUND_ASYNC,
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

    resolution = ToolPlacementResolver().resolve(spec, remote=True)

    assert resolution.has_client_live_tools
    assert spec.effective_schedule() == EvaluationSchedule.AFTER_ROUND_SYNC


def test_tool_placement_requires_explicit_live_probe_permission() -> None:
    spec = EvaluatorSpec.with_default_tools(
        kind="fake_eval",
        display_name="Fake Evaluator",
        tools=[standard_client_probe_tools()[0]],
    )

    with pytest.raises(ValueError, match="disabled"):
        ToolPlacementResolver().resolve(spec, remote=True)


def test_runner_forwards_remote_flag_to_placement_resolver(tmp_path) -> None:
    class _RecordingResolver(ToolPlacementResolver):
        def __init__(self) -> None:
            self.remote_values: list[bool] = []

        def resolve(self, spec: EvaluatorSpec, *, remote: bool) -> PlacementResolution:
            self.remote_values.append(remote)
            return super().resolve(spec, remote=remote)

    resolver = _RecordingResolver()
    runner = AgentKitEvaluatorRunner(
        spec=EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator"),
        placement_resolver=resolver,
    )

    runner.build_toolset(
        EvaluatorRunContext(
            bundle=_bundle_with_images(),
            evidence_manifest=EvaluationEvidenceManifest(),
            artifact_root=tmp_path,
            step_trace_by_session_step={},
            remote=True,
        )
    )

    assert resolver.remote_values == [True]


def test_final_report_rejects_out_of_range_ratings() -> None:
    factory = FinalReportToolFactory(
        evidence_manifest=EvaluationEvidenceManifest(
            refs=[EvaluationEvidenceRef(id="E001", kind=EvaluationEvidenceKind.IMAGE, label="Final")]
        )
    )
    payload = _report_payload()
    payload["completion_rating"] = 6

    result = factory._submit_final_report(**payload)

    assert result["status"] == "failure"
    assert "less than or equal" in str(result["summary"])
