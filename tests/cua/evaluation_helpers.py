"""Shared helpers for desktop evaluator tests."""

from __future__ import annotations

from pathlib import Path

from nexau.archs.main_sub.config import AgentConfig
from PIL import Image

from northcua.core.accessibility import AccessibilityDiffSummary, AccessibilityStepRecord
from northcua.core.enums import (
    AccessibilityCaptureStatus,
    AccessibilityDiffMode,
    AccessibilityScope,
    AccessibilitySourceKind,
    TaskStatus,
)
from northcua.core.models import LLMTraceRecord, StepLogRecord, ToolResultArtifact, WindowState


def write_agent_yaml(path: Path) -> None:
    """Write a minimal evaluator agent config for tests."""

    path.write_text(
        """
name: NorthCUAAgent
description: Desktop agent
system_prompt: You are NorthCUA.
llm_config:
  model: gpt-5.4
  base_url: http://localhost:3001/v1
  api_key: key
  api_type: openai_responses
""".strip(),
        encoding="utf-8",
    )


def agent_config(tmp_path: Path) -> AgentConfig:
    """Return a parsed test agent config under ``tmp_path``."""

    agent_yaml = tmp_path / "agent.yaml"
    write_agent_yaml(agent_yaml)
    return AgentConfig.from_yaml(agent_yaml)


def image(path: Path, color: tuple[int, int, int]) -> str:
    """Create a tiny RGB image and return its string path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color).save(path)
    return str(path)


def step(
    *,
    session_id: str,
    session_step: int,
    round_num: int,
    round_step: int,
    request: str,
    result: str,
    status: TaskStatus,
    clean_screenshot_path: str,
    tool_image_path: str,
    a11y_path: str,
) -> StepLogRecord:
    """Create a representative desktop step record for evaluator tests."""

    return StepLogRecord(
        session_id=session_id,
        session_step=session_step,
        round_step=round_step,
        round_num=round_num,
        status=status,
        request=request,
        result=result,
        llm_trace=LLMTraceRecord(),
        current_window=WindowState(name="Calculator", type="window"),
        clean_screenshot_path=clean_screenshot_path,
        assistant_text="I made progress.",
        tool_results=[
            ToolResultArtifact(
                tool_call_id=f"call-{session_step}",
                name="click",
                content="Clicked calculator button.",
                status="success",
                raw_image_path=tool_image_path,
                image_path=tool_image_path,
            )
        ],
        accessibility=AccessibilityStepRecord(
            status=AccessibilityCaptureStatus.SUCCESS,
            source=AccessibilitySourceKind.AX,
            scope=AccessibilityScope.DESKTOP,
            artifact_path=a11y_path,
            node_count=12,
            max_depth=3,
            focused_role="button",
            focused_name="Equals",
            capture_ms=12.0,
            diff_mode=AccessibilityDiffMode.BASIC,
            diff=AccessibilityDiffSummary(),
        ),
    )


__all__ = ["agent_config", "image", "step", "write_agent_yaml"]
