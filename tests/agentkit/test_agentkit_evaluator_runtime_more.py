"""AgentKit evaluator runtime contracts and safety gates."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock
from PIL import Image
from pydantic import ValidationError

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.evaluator import (
    AgentKitEvaluatorRunner,
    ArtifactEvaluationRequest,
    ArtifactEvaluationRunnerBase,
    ArtifactInspectionToolFactory,
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
from northcua_runtime_shared.tool_contracts import ToolPlacement, define_tool


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


def _artifact_tool_factory(tmp_path: Path, *refs: EvaluationEvidenceRef) -> ArtifactInspectionToolFactory:
    bundle = RoundEvaluationBundle(
        session_id="sess",
        round_num=1,
        round_request="Finish the task",
        session_request="Finish the task",
        screenshot_refs=list(refs),
    )
    return ArtifactInspectionToolFactory(
        artifact_root=tmp_path,
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(refs=list(refs)),
        step_trace_by_session_step={},
    )


def test_runner_builds_filtered_toolset_and_structured_artifacts(tmp_path) -> None:
    spec = EvaluatorSpec.with_default_tools(kind="fake_eval", display_name="Fake Evaluator")
    writer = EvaluatorArtifactWriter(EvaluatorArtifactLayout(session_root=tmp_path, evaluator_kind=spec.kind))
    runner = AgentKitEvaluatorRunner(spec=spec, artifact_writer=writer)
    bundle = _bundle_with_images()
    manifest = EvaluationEvidenceManifest(refs=[bundle.screenshot_refs[0]])

    runner.persist_inputs(round_num=1, bundle=bundle, evidence_manifest=manifest)
    toolset = runner.build_toolset(
        EvaluatorRunContext(
            bundle=bundle,
            evidence_manifest=manifest,
            artifact_root=tmp_path,
            step_trace_by_session_step={1: {"assistant_text": "done"}},
        )
    )
    runner.persist_final_report(round_num=1, report=_report_payload())

    assert [tool.name for tool in toolset.tools] == ["read_step", "read_trace", "query_artifacts", "final_report"]
    root = tmp_path / "evaluations" / "fake_eval" / "round_0001"
    assert json.loads((root / "bundle.json").read_text(encoding="utf-8"))["round_num"] == 1
    assert json.loads((root / "evidence_manifest.json").read_text(encoding="utf-8"))["refs"][0]["id"] == "E001"
    assert json.loads((root / "final_report.json").read_text(encoding="utf-8"))["overall_verdict"] == "success"


def test_compare_images_uses_lanczos_when_resizing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    Image.new("RGB", (8, 8), color="white").save(left_path)
    Image.new("RGB", (4, 4), color="black").save(right_path)
    factory = _artifact_tool_factory(
        tmp_path,
        EvaluationEvidenceRef(id="left", kind=EvaluationEvidenceKind.IMAGE, label="left", path=str(left_path)),
        EvaluationEvidenceRef(id="right", kind=EvaluationEvidenceKind.IMAGE, label="right", path=str(right_path)),
    )
    original_resize = Image.Image.resize
    observed: dict[str, object] = {}

    def _resize(self, size, resample=None, *args, **kwargs):
        observed["resample"] = resample
        return original_resize(self, size, resample=resample, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", _resize)

    result = factory._compare_images("left", "right")

    assert result["status"] == "success"
    assert observed["resample"] == Image.Resampling.LANCZOS


def test_extract_image_text_summary_times_out_tesseract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "screen.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    factory = _artifact_tool_factory(
        tmp_path,
        EvaluationEvidenceRef(id="screen", kind=EvaluationEvidenceKind.IMAGE, label="screen", path=str(image_path)),
    )
    monkeypatch.setattr(
        "northcua.agentkit.evaluator.tools.artifact_tools.shutil.which", lambda name: "/usr/bin/tesseract"
    )

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", "tesseract"), timeout=kwargs.get("timeout", 30))

    monkeypatch.setattr("northcua.agentkit.evaluator.tools.artifact_tools.subprocess.run", _timeout)

    result = factory._extract_image_text_summary("screen")

    assert result == {
        "status": "failure",
        "error": "tesseract OCR timed out after 30 seconds.",
        "timeout_seconds": 30,
    }


def test_artifact_evaluation_runner_base_resolves_single_session_and_delegates(tmp_path) -> None:
    class _Runner(ArtifactEvaluationRunnerBase[str, str, str]):
        primary_log_name = "trace.jsonl"

        def load_records(self, session_dir: Path) -> list[str]:
            return (session_dir / self.primary_log_name).read_text(encoding="utf-8").splitlines()

        def select_input(self, *, records: list[str], requested_unit_id: int) -> str:
            assert requested_unit_id == 2
            return records[-1]

        async def evaluate_selected(
            self,
            *,
            request: ArtifactEvaluationRequest,
            session_dir: Path,
            records: list[str],
            selected: str,
        ) -> str:
            assert request.write_rollup is False
            assert session_dir.name == "session-1"
            assert records == ["one", "two"]
            return selected

    root = tmp_path / "root"
    session = root / "session-1"
    session.mkdir(parents=True)
    (session / "trace.jsonl").write_text("one\ntwo\n", encoding="utf-8")

    result = asyncio.run(
        _Runner().run(
            ArtifactEvaluationRequest(
                app_config_path=tmp_path / "app.yaml",
                artifact_dir=root,
                unit_id=2,
                write_rollup=False,
            )
        )
    )

    assert result == "two"


def test_runner_builds_custom_server_tool_from_spec(tmp_path) -> None:
    def _count_refs() -> dict[str, object]:
        return {"count": 1}

    spec = EvaluatorSpec.with_default_tools(
        kind="fake_eval",
        display_name="Fake Evaluator",
        tools=[
            EvaluatorToolRef.from_contract(
                define_tool(
                    tool_id="count_refs",
                    description="Count selected evidence refs.",
                    placement=ToolPlacement.EVALUATOR_ARTIFACT,
                    json_schema={"type": "object", "properties": {}, "additionalProperties": False},
                ),
                placement=EvaluationToolPlacement.SERVER_ARTIFACT,
                implementation=_count_refs,
            )
        ],
    )
    runner = AgentKitEvaluatorRunner(spec=spec)
    toolset = runner.build_toolset(
        EvaluatorRunContext(
            bundle=_bundle_with_images(),
            evidence_manifest=EvaluationEvidenceManifest(),
            artifact_root=tmp_path,
            step_trace_by_session_step={},
        )
    )

    count_refs = next(tool for tool in toolset.tools if tool.name == "count_refs")
    assert asyncio.run(count_refs.execute_async()) == {"count": 1}


def test_runner_requires_visible_final_report_tool(tmp_path) -> None:
    spec = EvaluatorSpec.with_default_tools(
        kind="fake_eval",
        display_name="Fake Evaluator",
        placement_policy=ToolPlacementPolicy(
            allowed_placements=frozenset({EvaluationToolPlacement.SERVER_ARTIFACT}),
        ),
    )
    runner = AgentKitEvaluatorRunner(spec=spec)

    with pytest.raises(ValueError, match="final_report"):
        runner.build_toolset(
            EvaluatorRunContext(
                bundle=_bundle_with_images(),
                evidence_manifest=EvaluationEvidenceManifest(),
                artifact_root=tmp_path,
                step_trace_by_session_step={},
            )
        )
