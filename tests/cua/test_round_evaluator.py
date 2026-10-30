"""Regression coverage for desktop round evaluator artifacts and judge orchestration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

from nexau.core.messages import ImageBlock, TextBlock

from northcua.agentkernel.tracing.state import TraceCaptureState
from northcua.agentkit.evaluator.tools import ArtifactInspectionToolFactory as EvaluationToolFactory
from northcua.agents.desktop.evaluation.prompting import DesktopEvaluatorPromptBuilder
from northcua.agents.desktop.evaluation.protocol import (
    EvaluationCriterion,
    EvaluationEvidenceManifest,
    EvaluationEvidenceRef,
)
from northcua.agents.desktop.evaluation.rendering import EvaluationTemplateRenderer
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


def test_desktop_round_evaluator_spec_uses_configured_max_steps() -> None:
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    assert app_config.evaluation.max_steps == 12

    app_config.evaluation.max_steps = 12

    spec = desktop_round_evaluator_spec(app_config)

    assert spec.max_steps == 12


def test_round_evaluator_service_writes_round_jsonl_and_rollup(monkeypatch, tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-eval", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    evaluator_agent_config = make_agent_config(tmp_path)

    current = image(store.initial_screenshot_path(1), (255, 0, 0))
    tool = image(store.tool_screenshot_path(1, 0), (0, 255, 0))
    a11y_path = str(store.a11y_snapshot_path(1))
    store.write_text(Path(a11y_path), json.dumps({"status": "success"}))
    step_logs = [
        step(
            session_id="sess-eval",
            session_step=1,
            round_num=1,
            round_step=1,
            request="Open Calculator",
            result="Opened Calculator.",
            status=TaskStatus.FINISH,
            clean_screenshot_path=current,
            tool_image_path=tool,
            a11y_path=a11y_path,
        )
    ]

    service = RoundEvaluatorService(
        app_config=app_config,
        agent_config=evaluator_agent_config,
        artifact_store=store,
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )

    class _FakeFinalReportStepExecutor:
        async def run_step(self, **kwargs):
            agent_config = SimpleNamespace(max_iterations=0)
            kwargs["configure_agent"](agent_config)
            envelope = kwargs["build_envelope"]([], None)
            assert kwargs["envelope_messages"](envelope)
            assert "final_report" in {tool.name for tool in agent_config.tools}
            payload = {
                "overall_verdict": "success",
                "process_rating": 5,
                "outcome_label": "success",
                "completion_rating": 5,
                "confidence_rating": 5,
                "controllability": "controllable",
                "reason": "The round completed the requested action and the screenshots support it.",
                "criteria": [
                    {
                        "criterion_id": "criterion_1",
                        "score": 5,
                        "reason": "The app was opened successfully.",
                        "evidence_refs": [],
                    }
                ],
                "diagnoses": [],
                "side_effects": [],
                "needs_more_evidence": False,
            }
            return SimpleNamespace(
                response_text="calling final_report",
                meta={
                    "pending_tool_calls": [
                        {
                            "id": "call-final",
                            "name": "final_report",
                            "input": payload,
                        }
                    ]
                },
                trace_state=TraceCaptureState(assistant_text="calling final_report"),
            )

    async def _phase_must_not_run(self, **kwargs):
        del self, kwargs
        raise AssertionError("desktop evaluator should use AgentKit final_report runner, not legacy JSON phases")

    service.evaluator_runner.model_step_executor = _FakeFinalReportStepExecutor()  # type: ignore[assignment]
    monkeypatch.setattr(service, "_run_phase", MethodType(_phase_must_not_run, service))

    asyncio.run(
        service.evaluate_round_and_maybe_rollup(
            round_request="Open Calculator",
            round_num=1,
            step_logs=step_logs,
            terminal=True,
        )
    )

    round_lines = store.round_evaluation_log_path.read_text(encoding="utf-8").strip().splitlines()
    persisted_round = json.loads(round_lines[-1])
    assert persisted_round["round_num"] == 1
    assert persisted_round["reason"] != ""
    assert persisted_round["criteria"][0]["reason"] != ""
    assert persisted_round["safety"]["risk_level"] == "none"
    assert persisted_round["efficiency"]["rating"] == 5
    assert store.eval_round_bundle_path(1).exists()
    assert store.eval_round_result_path(1).exists()
    assert store.session_evaluation_rollup_path.exists()
    structured_root = store.root / "evaluations" / app_config.evaluation.default_evaluator_kind / "round_0001"
    assert (structured_root / "bundle.json").exists()
    assert (structured_root / "evidence_manifest.json").exists()
    assert (structured_root / "llm_trace.jsonl").exists()
    assert (structured_root / "runtime.jsonl").exists()
    assert (structured_root / "tool_trace.jsonl").exists()
    final_report = json.loads((structured_root / "final_report.json").read_text(encoding="utf-8"))
    assert final_report["overall_verdict"] == "success"
    assert final_report["safety"]["risk_level"] == "none"
    assert final_report["efficiency"]["rating"] == 5


def test_round_evaluator_service_filters_tools_through_spec(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-filter", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    app_config.evaluation.enabled = True
    app_config.evaluation.tools.allowed_placements = []
    service = RoundEvaluatorService(
        app_config=app_config,
        agent_config=make_agent_config(tmp_path),
        artifact_store=store,
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )
    bundle = service.bundle_builder.build(round_num=1, round_request="Open Calculator", step_logs=[])

    assert service._build_evaluator_tools(bundle, EvaluationEvidenceManifest(), []) == []


def test_round_evaluator_prompt_attaches_selected_images_as_image_blocks(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-prompt", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    service = RoundEvaluatorService(
        app_config=app_config,
        agent_config=make_agent_config(tmp_path),
        artifact_store=store,
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )
    image_path = image(store.initial_screenshot_path(1), (10, 20, 30))
    bundle = service.bundle_builder.build(round_num=1, round_request="Click a digit", step_logs=[])
    manifest = EvaluationEvidenceManifest(
        refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Step 1 overlay image",
                path=image_path,
                session_step=1,
                round_step=1,
            )
        ]
    )

    messages = service._build_agentkit_messages(
        bundle=bundle,
        evidence_manifest=manifest,
        criteria=[EvaluationCriterion(id="criterion_1", name="Task", description="Click a digit")],
    )

    user_text = messages[1].content[0].text
    assert "available_server_tools" not in user_text
    assert "final_report_guidance" not in user_text
    assert any(isinstance(block, ImageBlock) for block in messages[1].content)


def test_round_evaluator_prompt_uses_configured_templates(tmp_path) -> None:
    template_root = tmp_path / "templates"
    template_root.mkdir()
    (template_root / "agentkit_final_report_system.j2").write_text("SYSTEM SENTINEL", encoding="utf-8")
    (template_root / "agentkit_final_report_user.j2").write_text(
        "USER SENTINEL {{ payload_json }}",
        encoding="utf-8",
    )
    builder = DesktopEvaluatorPromptBuilder(artifact_root=tmp_path / "logs", template_root=template_root)
    bundle = RoundEvaluatorService(
        app_config=NorthCUAAppConfig(agent_config_path="agent.yaml"),
        agent_config=make_agent_config(tmp_path),
        artifact_store=NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-template", ArtifactConfig()),
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    ).bundle_builder.build(round_num=1, round_request="点击数字 4", step_logs=[])

    messages = builder.build_messages(
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(),
        criteria=[EvaluationCriterion(id="criterion_1", name="任务", description="点击数字 4")],
    )

    assert messages[0].content[0].text == "SYSTEM SENTINEL"
    assert messages[1].content[0].text.startswith("USER SENTINEL")
    assert "点击数字 4" in messages[1].content[0].text


def test_round_evaluator_prompt_explains_infeasible_judgment(tmp_path) -> None:
    builder = DesktopEvaluatorPromptBuilder(artifact_root=tmp_path / "logs")
    bundle = RoundEvaluatorService(
        app_config=NorthCUAAppConfig(agent_config_path="agent.yaml"),
        agent_config=make_agent_config(tmp_path),
        artifact_store=NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-infeasible", ArtifactConfig()),
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    ).bundle_builder.build(round_num=1, round_request="Open a private portal", step_logs=[])

    messages = builder.build_messages(
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(),
        criteria=[EvaluationCriterion(id="criterion_1", name="Task", description="Open a private portal")],
    )

    system_text = messages[0].content[0].text
    assert "[INFEASIBLE]" in system_text
    assert "A correct infeasibility judgment can be a successful outcome" in system_text
    assert "incorrect or premature infeasibility claim should be penalized" in system_text
    assert "client_bash_readonly is available" in system_text
    assert "treat direct bash-observed state as stronger evidence than screenshots" in system_text


def test_legacy_round_evaluator_prompt_prefers_bash_for_ambiguous_final_state() -> None:
    renderer = EvaluationTemplateRenderer(Path("northcua/agents/desktop/evaluation/templates"))

    system_text = renderer.render("evaluator_judge_system.j2", phase_name="artifact_aware")

    assert "client_bash_readonly is available" in system_text
    assert "visual evidence may be stale or shifted" in system_text
    assert "treat direct bash-observed state as stronger evidence than screenshots" in system_text


def test_round_evaluator_prompt_preserves_unicode_payload_text(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-prompt-unicode", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    service = RoundEvaluatorService(
        app_config=app_config,
        agent_config=make_agent_config(tmp_path),
        artifact_store=store,
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )
    bundle = service.bundle_builder.build(round_num=1, round_request="点击数字 4", step_logs=[])

    messages = service._build_agentkit_messages(
        bundle=bundle,
        evidence_manifest=EvaluationEvidenceManifest(),
        criteria=[EvaluationCriterion(id="criterion_1", name="任务", description="点击数字 4")],
    )

    user_text = messages[1].content[0].text
    assert "Current round request:\n点击数字 4" in user_text
    assert "点击数字 4" in user_text
    assert "\\u70b9\\u51fb" not in user_text


def test_round_evaluator_prompt_explains_terminal_and_overlay_images(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-prompt-variants", ArtifactConfig())
    app_config = NorthCUAAppConfig(agent_config_path="agent.yaml")
    service = RoundEvaluatorService(
        app_config=app_config,
        agent_config=make_agent_config(tmp_path),
        artifact_store=store,
        execution_transport=object(),  # type: ignore[arg-type]
        session_manager=object(),  # type: ignore[arg-type]
        user_id="test-user",
    )
    terminal_path = image(store.verification_screenshot_path(1), (10, 20, 30))
    overlay_path = image(store.tool_screenshot_path(1, 1), (40, 50, 60))
    bundle = service.bundle_builder.build(round_num=1, round_request="Click a digit", step_logs=[])
    manifest = EvaluationEvidenceManifest(
        refs=[
            EvaluationEvidenceRef(
                id="E001",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Step 1 terminal verification",
                path=terminal_path,
                session_step=1,
                round_step=1,
                metadata={"variant": "terminal_verification"},
            ),
            EvaluationEvidenceRef(
                id="E002",
                kind=EvaluationEvidenceKind.IMAGE,
                label="Step 1 click overlay",
                path=overlay_path,
                session_step=1,
                round_step=1,
                metadata={"variant": "overlay"},
            ),
        ]
    )

    messages = service._build_agentkit_messages(
        bundle=bundle,
        evidence_manifest=manifest,
        criteria=[EvaluationCriterion(id="criterion_1", name="Task", description="Click a digit")],
    )

    system_text = messages[0].content[0].text
    user_blocks = messages[1].content
    assert len(user_blocks) == 5
    assert isinstance(user_blocks[0], TextBlock)
    assert isinstance(user_blocks[1], TextBlock)
    assert isinstance(user_blocks[2], ImageBlock)
    assert isinstance(user_blocks[3], TextBlock)
    assert isinstance(user_blocks[4], ImageBlock)
    assert user_blocks[2].detail == "high"
    assert user_blocks[4].detail == "high"
    user_text = "\n".join(block.text for block in user_blocks if isinstance(block, TextBlock))
    assert "variant=terminal_verification" in system_text
    assert "captured after the round stopped" in system_text
    assert "variant=overlay" in system_text
    assert "not part of the application UI" in system_text
    assert "Failure diagnosis taxonomy" in system_text
    assert "task_spec_or_feasibility_error" in system_text
    assert "grounding_localization_error" in system_text
    assert "environment_or_execution_error" in system_text
    assert "execution_strategy" not in system_text
    assert "same language as the current round request" in system_text
    assert "terminal verification screenshot captured after the round stopped" in user_blocks[1].text
    assert "use it as the final-state view" in user_blocks[1].text
    assert "overlay screenshot" in user_blocks[3].text
    assert "not part of the app UI" in user_blocks[3].text
    assert "terminal verification" in user_text
