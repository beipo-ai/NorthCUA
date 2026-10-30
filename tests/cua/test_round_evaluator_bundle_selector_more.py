"""Regression coverage for desktop evaluator bundles and evidence selection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from northcua.agentkit.evaluator import (
    ArtifactEvaluationRequest,
    EvidencePolicy,
    UnavailableEvaluatorClientToolDispatcher,
)
from northcua.agents.desktop.evaluation.bundle import BundleBuilder
from northcua.agents.desktop.evaluation.protocol import (
    EvaluationCriterion,
    EvaluationEvidenceManifest,
    EvaluationEvidenceRef,
    EvaluationJudgeMetadata,
    RoundEvaluationRecord,
)
from northcua.agents.desktop.evaluation.selector import EvidenceSelector
from northcua.core.accessibility import AccessibilityStepRecord
from northcua.core.config import ArtifactConfig, NorthCUAAppConfig
from northcua.core.enums import (
    AccessibilityCaptureStatus,
    AccessibilityScope,
    AccessibilitySourceKind,
    EvaluationControllability,
    EvaluationEvidenceKind,
    EvaluationOutcome,
    EvaluationVerdict,
    TaskStatus,
)
from northcua.runtime.artifacts.store import NorthCUAArtifactStore
from tests.cua.evaluation_helpers import image, step


def test_evidence_selector_keeps_one_default_image_per_step() -> None:
    selector = EvidenceSelector(
        NorthCUAAppConfig(agent_config_path="agent.yaml").evaluation.selector,
        policy=EvidencePolicy(screenshot_max=5),
    )
    criteria = [EvaluationCriterion(id="criterion_1", name="Final digit", description="Click the requested digit")]
    bundle = type(
        "_Bundle",
        (),
        {
            "screenshot_refs": [
                EvaluationEvidenceRef(
                    id="E001",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label="Step 1 raw image",
                    summary="raw screenshot",
                    path="/tmp/step1-raw.png",
                    session_step=1,
                    round_step=1,
                    metadata={"variant": "raw"},
                ),
                EvaluationEvidenceRef(
                    id="E002",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label="Step 1 overlay image",
                    summary="overlay screenshot",
                    path="/tmp/step1-overlay.png",
                    session_step=1,
                    round_step=1,
                    metadata={"variant": "overlay"},
                ),
                EvaluationEvidenceRef(
                    id="E003",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label="Step 2 raw image",
                    summary="raw screenshot",
                    path="/tmp/step2-raw.png",
                    session_step=2,
                    round_step=2,
                    metadata={"variant": "raw"},
                ),
                EvaluationEvidenceRef(
                    id="E004",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label="Step 2 overlay image",
                    summary="overlay screenshot",
                    path="/tmp/step2-overlay.png",
                    session_step=2,
                    round_step=2,
                    metadata={"variant": "overlay"},
                ),
                EvaluationEvidenceRef(
                    id="E005",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label="Step 3 raw image",
                    summary="raw screenshot",
                    path="/tmp/step3-raw.png",
                    session_step=3,
                    round_step=3,
                    metadata={"variant": "raw"},
                ),
                EvaluationEvidenceRef(
                    id="E006",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label="Step 3 current image",
                    summary="current screenshot",
                    path="/tmp/step3-current.png",
                    session_step=3,
                    round_step=3,
                ),
            ],
            "a11y_refs": [],
            "runtime_event_refs": [],
        },
    )()

    manifest = selector.select(bundle=bundle, criteria=criteria)

    image_ids = [ref.id for ref in manifest.refs if ref.kind == EvaluationEvidenceKind.IMAGE]
    assert len(image_ids) == 3
    assert image_ids[:2] == ["E004", "E002"]
    assert image_ids[-1] == "E006"
    assert "E001" not in image_ids
    assert "E003" not in image_ids
    assert "E005" not in image_ids


def test_evidence_selector_pins_initial_state_before_overlay() -> None:
    selector = EvidenceSelector(
        NorthCUAAppConfig(agent_config_path="agent.yaml").evaluation.selector,
        policy=EvidencePolicy(screenshot_max=3),
    )
    bundle = type(
        "_Bundle",
        (),
        {
            "screenshot_refs": [
                EvaluationEvidenceRef(
                    id="E001",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label="Step 1 initial screenshot",
                    path="/tmp/step1-initial.png",
                    session_step=1,
                    round_step=1,
                    metadata={"variant": "initial_state", "initial_state": True},
                ),
                EvaluationEvidenceRef(
                    id="E002",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label="Step 1 overlay image",
                    path="/tmp/step1-overlay.png",
                    session_step=1,
                    round_step=1,
                    metadata={"variant": "overlay"},
                ),
                EvaluationEvidenceRef(
                    id="E003",
                    kind=EvaluationEvidenceKind.IMAGE,
                    label="Step 2 current image",
                    path="/tmp/step2-current.png",
                    session_step=2,
                    round_step=2,
                ),
            ],
            "a11y_refs": [],
            "runtime_event_refs": [],
        },
    )()

    manifest = selector.select(bundle=bundle, criteria=[])

    refs = {ref.id: ref for ref in manifest.refs if ref.kind == EvaluationEvidenceKind.IMAGE}
    assert refs["E001"].pinned
    assert refs["E001"].selection_reason == "pinned_initial_screenshot"
    assert "E002" not in refs


def test_desktop_artifact_evaluation_runner_loads_latest_round(tmp_path) -> None:
    from northcua.agents.desktop.evaluation.offline import DesktopArtifactEvaluationRunner

    session_dir = tmp_path / "sess-replay"
    session_dir.mkdir()
    steps = [
        step(
            session_id="sess-replay",
            session_step=1,
            round_num=1,
            round_step=1,
            request="Round one",
            result="continue",
            status=TaskStatus.CONTINUE,
            clean_screenshot_path="/tmp/one.png",
            tool_image_path="/tmp/one-tool.png",
            a11y_path="/tmp/one.json",
        ),
        step(
            session_id="sess-replay",
            session_step=2,
            round_num=2,
            round_step=1,
            request="Round two",
            result="done",
            status=TaskStatus.FINISH,
            clean_screenshot_path="/tmp/two.png",
            tool_image_path="/tmp/two-tool.png",
            a11y_path="/tmp/two.json",
        ),
    ]
    (session_dir / "steps.jsonl").write_text(
        "\n".join(item.model_dump_json() for item in steps) + "\n",
        encoding="utf-8",
    )

    runner = DesktopArtifactEvaluationRunner()
    records = runner.load_records(session_dir)
    selected = runner.select_input(records=records, requested_unit_id=0)

    assert selected.round_num == 2
    assert selected.round_request == "Round two"
    assert [item.session_step for item in selected.step_logs] == [2]


@pytest.mark.asyncio
async def test_desktop_artifact_replay_injects_unavailable_client_dispatcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from northcua.agents.desktop.application import NorthCUAApplication
    from northcua.agents.desktop.evaluation.offline import DesktopArtifactEvaluationRunner, DesktopRoundArtifactInput

    captured: dict[str, object] = {}
    round_paths = SimpleNamespace(
        final_report_path=tmp_path / "final_report.json",
        root=tmp_path / "round_0001",
    )

    class _Evaluator:
        event_callback = None
        structured_artifact_sink = SimpleNamespace(
            writer=SimpleNamespace(layout=SimpleNamespace(round_paths=lambda round_num: round_paths))
        )
        artifact_store = SimpleNamespace(
            eval_round_result_path=lambda round_num: tmp_path / "result.json",
            round_evaluation_log_path=tmp_path / "round_evaluation.jsonl",
            session_evaluation_rollup_path=tmp_path / "session_evaluation.json",
        )

        async def evaluate_round_and_maybe_rollup(self, **kwargs: object) -> None:
            captured.update(kwargs)

    evaluator = _Evaluator()
    app = SimpleNamespace(
        facade=SimpleNamespace(
            round_execution_service=SimpleNamespace(round_evaluator_service=evaluator),
        ),
        stop=lambda: None,
    )

    async def _stop() -> None:
        return None

    app.stop = _stop
    monkeypatch.setattr(
        NorthCUAApplication,
        "from_app_config",
        classmethod(lambda cls, *args, **kwargs: app),
    )
    source_step = step(
        session_id="offline-session",
        session_step=1,
        round_num=1,
        round_step=1,
        request="Evaluate offline",
        result="done",
        status=TaskStatus.FINISH,
        clean_screenshot_path="/tmp/offline-clean.png",
        tool_image_path="/tmp/offline-tool.png",
        a11y_path="/tmp/offline-a11y.json",
    )

    await DesktopArtifactEvaluationRunner().evaluate_selected(
        request=ArtifactEvaluationRequest(
            app_config_path=tmp_path / "app.yaml",
            artifact_dir=tmp_path,
            unit_id=1,
            write_rollup=False,
            force_enable=False,
        ),
        session_dir=tmp_path,
        records=[source_step],
        selected=DesktopRoundArtifactInput(
            round_num=1,
            round_request="Evaluate offline",
            step_logs=[source_step],
        ),
    )

    assert isinstance(captured["client_tool_dispatcher"], UnavailableEvaluatorClientToolDispatcher)
