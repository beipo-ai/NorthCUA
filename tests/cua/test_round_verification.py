"""Regression coverage for terminal verification capture."""

from __future__ import annotations

import asyncio

from nexau.core.messages import ImageBlock, ToolResultBlock

from northcua.agents.desktop.round_execution.verification import RoundEndVerificationCapture
from northcua.core.config import ArtifactConfig, DesktopPolicyConfig
from northcua.core.enums import RuntimeMode, TaskStatus
from northcua.core.models import StepLogRecord, ToolResultArtifact
from northcua.runtime.artifacts.store import NorthCUAArtifactStore
from northcua.transport.ws.result_blocks import RemoteToolDispatchResult
from tests.cua.evaluation_helpers import image


class _UnusedExecutor:
    async def execute_command_request(self, payload):  # pragma: no cover - local path unused
        raise AssertionError("remote verification should not call local executor")


def test_remote_verification_accepts_remote_tool_dispatch_result(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-verify", ArtifactConfig())
    source = image(store.initial_screenshot_path(1), (10, 20, 30))
    capture = RoundEndVerificationCapture(
        desktop_policy=DesktopPolicyConfig(),
        artifact_store=store,
        command_executor=_UnusedExecutor(),
    )
    step_log = StepLogRecord(
        session_id="sess-verify",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.FINISH,
        request="Open Calculator",
        result="done",
    )

    async def _dispatcher(pending_tool_calls):
        assert pending_tool_calls[0]["name"] == "capture_desktop_screenshot"
        return RemoteToolDispatchResult(
            tool_blocks=[
                ToolResultBlock(
                    tool_use_id="verify-1",
                    content=[ImageBlock(url=source)],
                    is_error=False,
                )
            ]
        )

    asyncio.run(
        capture.capture_for_terminal_step(
            step_log=step_log,
            runtime_mode=RuntimeMode.REMOTE,
            remote_tool_dispatcher=_dispatcher,
        )
    )

    assert step_log.observation_metadata["round_end_verification_status"] == "success"
    assert step_log.clean_screenshot_path.endswith("action_step1_verify.png")


def test_remote_verification_does_not_inherit_cua_target_from_previous_round(tmp_path) -> None:
    store = NorthCUAArtifactStore.for_session(tmp_path / "logs", "sess-verify-cua-round", ArtifactConfig())
    source = image(store.initial_screenshot_path(1), (20, 40, 60))
    capture = RoundEndVerificationCapture(
        desktop_policy=DesktopPolicyConfig(),
        artifact_store=store,
        command_executor=_UnusedExecutor(),
    )
    previous_step = StepLogRecord(
        session_id="sess-verify-cua-round",
        session_step=1,
        round_step=1,
        round_num=1,
        status=TaskStatus.CONTINUE,
        request="Use Calculator",
        result="typed expression",
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-type",
                name="cua_type_text",
                content="Type 4 character(s).",
                status="success",
                metadata={"cua_target": {"pid": 123, "window_id": 456}},
            )
        ],
    )
    terminal_step = StepLogRecord(
        session_id="sess-verify-cua-round",
        session_step=2,
        round_step=1,
        round_num=2,
        status=TaskStatus.FINISH,
        request="Use the normal desktop",
        result="done",
    )

    async def _dispatcher(pending_tool_calls):
        assert pending_tool_calls[0]["name"] == "capture_desktop_screenshot"
        return RemoteToolDispatchResult(
            tool_blocks=[
                ToolResultBlock(
                    tool_use_id="verify-2",
                    content=[ImageBlock(url=source)],
                    is_error=False,
                )
            ]
        )

    asyncio.run(
        capture.capture_for_terminal_step(
            step_log=terminal_step,
            runtime_mode=RuntimeMode.REMOTE,
            remote_tool_dispatcher=_dispatcher,
            previous_steps=[previous_step],
        )
    )

    assert terminal_step.observation_metadata["round_end_verification_source"] == (
        "mcp:DesktopNativeDataExecutor.capture_desktop_screenshot"
    )
    assert "round_end_verification_cua_target" not in terminal_step.observation_metadata
