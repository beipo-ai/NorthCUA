"""Lightweight evaluator prompt template checks."""

from __future__ import annotations

from pathlib import Path

from northcua.agents.desktop.evaluation.rendering import EvaluationTemplateRenderer


def test_desktop_evaluator_prompts_prefer_readonly_bash_for_ambiguous_final_state() -> None:
    renderer = EvaluationTemplateRenderer(Path("northcua/agents/desktop/evaluation/templates"))

    judge_system = renderer.render("evaluator_judge_system.j2", phase_name="artifact_aware")
    final_report_system = renderer.render("agentkit_final_report_system.j2")

    for prompt in (judge_system, final_report_system):
        assert "client_bash_readonly is available" in prompt
        assert "visual evidence may be stale or shifted" in prompt
        assert "treat direct bash-observed state as stronger evidence than screenshots" in prompt
