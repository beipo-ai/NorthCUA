"""Regression coverage for desktop round evaluator tool helpers."""

from __future__ import annotations

import asyncio
import json

from northcua.agentkit.evaluator.tools import ArtifactInspectionToolFactory as EvaluationToolFactory
from northcua.agents.desktop.evaluation.bundle import BundleBuilder
from northcua.agents.desktop.evaluation.protocol import (
    EvaluationEvidenceManifest,
    EvaluationEvidenceRef,
)
from northcua.core.config import ArtifactConfig, NorthCUAAppConfig
from northcua.core.enums import EvaluationEvidenceKind, TaskStatus
from northcua.runtime.artifacts.store import NorthCUAArtifactStore
from tests.cua.evaluation_helpers import image, step


def test_evaluation_tool_factory_queries_bundle_runtime_tool_results_and_a11y(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-tools", ArtifactConfig())
    image_path = image(store.initial_screenshot_path(1), (255, 0, 0))
    a11y_path = store.a11y_snapshot_path(1)
    store.write_json(
        a11y_path,
        {
            "status": "success",
            "focused_name": "Equals",
            "tree": {
                "role": "desktop",
                "name": "desktop",
                "children": [{"role": "button", "name": f"Button {index}"} for index in range(50)],
            },
        },
    )
    bundle = BundleBuilder(
        artifact_store=store,
        config=NorthCUAAppConfig(agent_config_path="agent.yaml").evaluation.bundle,
    ).build(
        round_num=1,
        round_request="Open Calculator",
        step_logs=[
            step(
                session_id="sess-tools",
                session_step=1,
                round_num=1,
                round_step=1,
                request="Open Calculator",
                result="Opened Calculator.",
                status=TaskStatus.FINISH,
                clean_screenshot_path=image_path,
                tool_image_path=image_path,
                a11y_path=str(a11y_path),
            )
        ],
    )
    manifest = EvaluationEvidenceManifest(
        refs=[
            EvaluationEvidenceRef(
                id="EIMG",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Current screenshot",
                summary="calculator open",
                path=image_path,
                session_step=1,
                round_step=1,
            ),
            EvaluationEvidenceRef(
                id="EA11Y",
                kind=EvaluationEvidenceKind.A11Y,
                label="Accessibility snapshot",
                summary="focused equals button",
                path=str(a11y_path),
                session_step=1,
                round_step=1,
            ),
        ]
    )
    factory = EvaluationToolFactory(
        artifact_root=store.root,
        bundle=bundle,
        evidence_manifest=manifest,
        step_trace_by_session_step={1: {"assistant_text": "Opened Calculator."}},
    )

    read_bundle_tool = next(tool for tool in factory.build_tools() if tool.name == "read_round_bundle")
    read_step_tool = next(tool for tool in factory.build_tools() if tool.name == "read_step")
    a11y_tool = next(tool for tool in factory.build_tools() if tool.name == "query_a11y_snapshot")

    section_schema = read_bundle_tool.input_schema["properties"]["section"]
    assert "trajectory" in section_schema["description"]
    assert "compact" in section_schema["description"]
    assert "Defaults to summary" not in section_schema["description"]
    assert "trace" in section_schema["examples"]
    assert asyncio.run(read_bundle_tool.execute_async(section="session_id"))["value"] == "sess-tools"
    selected_manifest = asyncio.run(read_bundle_tool.execute_async(section="selected_evidence_manifest"))
    assert selected_manifest["status"] == "success"
    assert selected_manifest["value"][0]["id"] == "EIMG"
    assert asyncio.run(read_step_tool.execute_async(session_step=1))["status"] == "success"
    assert factory._read_step_trace(1)["status"] == "success"
    runtime_events = factory._query_runtime_events("step")
    assert runtime_events["status"] == "success"
    assert factory._query_tool_results(tool_name="click", session_step=1)["matches"][0]["name"] == "click"
    assert factory._query_a11y_snapshot(evidence_ref="EA11Y")["snapshot"]["focused_name"] == "Equals"
    a11y_payload = asyncio.run(a11y_tool.execute_async(evidence_ref="EA11Y"))
    assert len(json.dumps(a11y_payload, ensure_ascii=False)) < 50_000
    assert a11y_payload["snapshot"]["tree_preview"]["omitted_children"] == 42
    assert factory._query_evidence_manifest(kind="a11y")["refs"][0]["id"] == "EA11Y"


def test_evaluation_tool_factory_resolves_session_prefixed_remote_artifact_paths(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-remote", ArtifactConfig())
    image_path = image(store.initial_screenshot_path(1), (0, 0, 255))
    a11y_path = store.a11y_snapshot_path(1)
    store.write_json(
        a11y_path,
        {
            "status": "success",
            "focused_name": "Seven",
            "tree": {"role": "desktop", "name": "desktop"},
        },
    )
    bundle = BundleBuilder(
        artifact_store=store,
        config=NorthCUAAppConfig(agent_config_path="agent.yaml").evaluation.bundle,
    ).build(
        round_num=1,
        round_request="Click a digit",
        step_logs=[
            step(
                session_id="sess-remote",
                session_step=1,
                round_num=1,
                round_step=1,
                request="Click a digit",
                result="Clicked 7.",
                status=TaskStatus.FINISH,
                clean_screenshot_path=image_path,
                tool_image_path=image_path,
                a11y_path="logs/sess-remote/query_0001/a11y_step1.json",
            )
        ],
    )
    manifest = EvaluationEvidenceManifest(
        refs=[
            EvaluationEvidenceRef(
                id="EA11Y",
                kind=EvaluationEvidenceKind.A11Y,
                label="Accessibility snapshot",
                path="logs/sess-remote/query_0001/a11y_step1.json",
                session_step=1,
                round_step=1,
            )
        ]
    )
    factory = EvaluationToolFactory(
        artifact_root=store.root,
        bundle=bundle,
        evidence_manifest=manifest,
        step_trace_by_session_step={},
    )

    result = factory._query_a11y_snapshot(evidence_ref="EA11Y")

    assert result["status"] == "success"
    assert result["snapshot"]["focused_name"] == "Seven"


def test_evaluation_tool_factory_compares_images_and_reports_missing_ocr_dependency(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-images", ArtifactConfig())
    left_path = image(store.initial_screenshot_path(1), (255, 0, 0))
    right_path = image(store.tool_screenshot_path(1, 0), (255, 0, 0))
    manifest = EvaluationEvidenceManifest(
        refs=[
            EvaluationEvidenceRef(
                id="ELEFT",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Left image",
                summary="left",
                path=left_path,
            ),
            EvaluationEvidenceRef(
                id="ERIGHT",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Right image",
                summary="right",
                path=right_path,
            ),
        ]
    )
    factory = EvaluationToolFactory(
        artifact_root=store.root,
        bundle=BundleBuilder(
            artifact_store=store,
            config=NorthCUAAppConfig(agent_config_path="agent.yaml").evaluation.bundle,
        ).build(round_num=1, round_request="Compare screenshots", step_logs=[]),
        evidence_manifest=manifest,
        step_trace_by_session_step={},
    )
    monkeypatch.setattr("northcua.agentkit.evaluator.tools.artifact_tools.shutil.which", lambda name: None)

    comparison = factory._compare_images("ELEFT", "ERIGHT")
    ocr = factory._extract_image_text_summary("ELEFT")

    assert comparison["status"] == "success"
    assert comparison["mean_pixel_delta"] == 0.0
    assert ocr["status"] == "failure"
    assert ocr["missing_dependency"] == "tesseract"


def test_evaluation_tool_factory_rejects_external_artifact_paths(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-safe", ArtifactConfig())
    outside = image(tmp_path / "outside.png", (1, 2, 3))
    manifest = EvaluationEvidenceManifest(
        refs=[
            EvaluationEvidenceRef(
                id="EOUT",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Outside image",
                summary="outside",
                path=outside,
            )
        ]
    )
    factory = EvaluationToolFactory(
        artifact_root=store.root,
        bundle=BundleBuilder(
            artifact_store=store,
            config=NorthCUAAppConfig(agent_config_path="agent.yaml").evaluation.bundle,
        ).build(round_num=1, round_request="Compare screenshots", step_logs=[]),
        evidence_manifest=manifest,
        step_trace_by_session_step={},
    )

    assert factory._compare_images("EOUT", "EOUT")["status"] == "failure"
    assert factory._extract_image_text_summary("EOUT")["status"] == "failure"
