from __future__ import annotations

import pytest

from northcua.agentkit.evaluator.protocol import EvaluationEvidenceManifest, RoundEvaluationReport
from northcua.agentkit.evaluator.spec import EvaluatorSpec, EvaluatorToolRef
from northcua.agentkit.evaluator.tools.artifact_tools import ARTIFACT_TOOL_REFS
from northcua.agentkit.evaluator.tools.client_probe import standard_client_gui_tools, standard_client_probe_tools
from northcua.agentkit.evaluator.tools.final_report import FinalReportToolFactory
from northcua.agents.desktop.evaluation.spec import desktop_round_evaluator_spec
from northcua.core.config import NorthCUAAppConfig
from northcua.core.enums import EvaluationToolPlacement
from northcua_runtime_shared.tool_contracts import (
    EvaluatorClientBinding,
    ToolAudience,
    ToolPlacement,
    ToolProfile,
    ToolSurfaceResolver,
    define_tool,
    fingerprint_json,
)


def test_standard_evaluator_client_tools_are_contract_backed() -> None:
    tools = (*standard_client_probe_tools(), *standard_client_gui_tools())

    assert tools
    for tool in tools:
        assert tool.contract is not None
        assert tool.input_schema == tool.contract.parameters_schema
        assert tool.readonly == tool.contract.is_readonly
        assert tool.timeout_seconds == tool.contract.timeout_seconds
        assert tool.contract.audience == ToolAudience.EVALUATOR
        assert tool.contract.server_name == tool.server_name
        assert tool.contract.binding is not None
        assert isinstance(tool.contract.binding, EvaluatorClientBinding)
        assert tool.contract.binding.target_tool_id
        assert tool.contract.binding.target_schema_fingerprint == fingerprint_json(tool.contract.parameters_schema)

    screenshot = next(tool for tool in tools if tool.name == "client_screenshot")
    assert screenshot.contract is not None
    assert screenshot.contract.aliases == ("capture_desktop_screenshot",)


def test_evaluator_tool_ref_rejects_schema_override_constructor() -> None:
    tool = standard_client_probe_tools()[0]
    assert tool.contract is not None

    with pytest.raises(TypeError):
        EvaluatorToolRef(
            name="client_screenshot_copy",
            description=tool.description,
            placement=EvaluationToolPlacement.CLIENT_PROBE,
            input_schema={"type": "object", "properties": {"drifted": {"type": "string"}}},
            contract=tool.contract,
        )


def test_evaluator_client_tools_must_match_contract_dispatch_and_placement() -> None:
    mutating_tool = next(tool for tool in standard_client_gui_tools() if tool.name == "client_gui_click")
    assert mutating_tool.contract is not None

    with pytest.raises(TypeError):
        EvaluatorToolRef(
            name="fake_readonly_click",
            description="Pretend click is readonly.",
            placement=EvaluationToolPlacement.CLIENT_PROBE,
            readonly=True,
            dispatch_name="click",
        )

    with pytest.raises(ValueError, match="placement drifts"):
        EvaluatorToolRef.from_contract(
            mutating_tool.contract,
            placement=EvaluationToolPlacement.CLIENT_PROBE,
        )

    with pytest.raises(TypeError):
        EvaluatorToolRef(
            name="fake_click_alias",
            description=mutating_tool.description,
            placement=EvaluationToolPlacement.CLIENT_MUTATING,
            readonly=False,
            dispatch_name="keypress",
            contract=mutating_tool.contract,
        )


def test_evaluator_finalizer_contract_must_publish_final_report_name() -> None:
    with pytest.raises(ValueError, match="FINALIZER"):
        EvaluatorToolRef.from_contract(
            define_tool(
                tool_id="submit_report",
                public_name="submit_report",
                description="Invalid finalizer alias.",
                placement=ToolPlacement.FINALIZER,
                json_schema=RoundEvaluationReport.model_json_schema(),
                side_effect=False,
                readonly=True,
            ),
            placement=EvaluationToolPlacement.FINALIZER,
        )


def test_evaluator_artifact_specs_project_tool_schema_from_contracts() -> None:
    assert ARTIFACT_TOOL_REFS

    for spec in ARTIFACT_TOOL_REFS:
        assert spec.input_schema == spec.contract.parameters_schema
        assert spec.contract.is_readonly is True
        assert spec.contract.binding is not None

    surface = ToolSurfaceResolver(tuple(spec.contract for spec in ARTIFACT_TOOL_REFS)).resolve(
        ToolProfile(name="evaluator.desktop.round", audiences=frozenset({ToolAudience.EVALUATOR}))
    )
    assert {tool.name for tool in surface.provider_tools} == {spec.name for spec in ARTIFACT_TOOL_REFS}


def test_default_evaluator_spec_tools_are_contract_backed() -> None:
    spec = EvaluatorSpec.with_default_tools(kind="round_default", display_name="Round Default")

    assert {tool.name for tool in spec.tools} == {"read_step", "read_trace", "query_artifacts", "final_report"}
    for tool in spec.tools:
        assert tool.contract is not None
        assert tool.input_schema == tool.contract.parameters_schema


def test_desktop_round_evaluator_artifact_tools_are_contract_backed() -> None:
    spec = desktop_round_evaluator_spec(NorthCUAAppConfig(agent_config_path="agent.yaml"))

    for tool in spec.tools:
        assert tool.contract is not None
        assert tool.input_schema == tool.contract.parameters_schema


def test_final_report_tool_uses_report_model_contract_schema() -> None:
    factory = FinalReportToolFactory(
        report_model=RoundEvaluationReport,
        evidence_manifest=EvaluationEvidenceManifest(refs=[]),
    )

    contract = factory.tool_contract()
    tool = factory.build_tool()

    assert contract.tool_id == "final_report"
    assert contract.parameters_schema == RoundEvaluationReport.model_json_schema()
    assert tool.input_schema == contract.provider_parameters_schema
    assert tool.input_schema != contract.parameters_schema
