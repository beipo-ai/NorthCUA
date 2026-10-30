from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

from northcua.agentinfra.runtime.protocol import (
    DesktopRunRequest,
    DesktopRunResult,
)
from northcua.app.task_runner import TaskRunResult
from northcua.app.terminal_renderer import NorthCUATerminalRenderer
from northcua.cli import (
    build_parser,
    default_app_config_path,
    default_session_id,
    evaluate_artifact_command,
    resolve_app_config_path,
    resolve_log_root,
    run_server_command,
    run_task_command,
    submit_task_command,
)
from northcua.cli.controller_commands import _task_row_json
from northcua.core.enums import TaskStatus, WebSocketMessageType
from northcua.core.models import AssistantToolCallRecord, LLMTraceRecord, StepLogRecord, ToolResultArtifact
from northcua.core.protocol import NorthCUAWebSocketEnvelope, TaskFinishedPayload
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef
from northcua_protocol.payloads import TaskAccepted
from northcua_protocol.state import TaskState

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _FakeFacade:
    def __init__(self, step_logs: list[StepLogRecord]) -> None:
        self._step_logs = list(step_logs)
        self.enqueued_queries: list[str] = []
        self._index = 0

    async def enqueue_initial_request(self, query: str, metadata: dict[str, object] | None = None) -> None:
        del metadata
        self.enqueued_queries.append(query)

    async def next_round(self):
        if self._index >= len(self._step_logs):
            return None
        return SimpleNamespace(query=self._step_logs[self._index].request)

    async def run_round(self, round_input):
        del round_input
        step_log = self._step_logs[self._index]
        self._index += 1
        return step_log


class _FakeApplication:
    def __init__(self, step_logs: list[StepLogRecord] | None = None) -> None:
        self.facade = _FakeFacade(step_logs or [])
        self.task_runner = SimpleNamespace(run=self._run)
        self.artifact_store = SimpleNamespace(
            root=Path("/tmp/northcua-test"),
            step_log_path=Path("/tmp/northcua-test/steps.jsonl"),
            runtime_log_path=Path("/tmp/northcua-test/runtime.jsonl"),
            evaluation_log_path=Path("/tmp/northcua-test/evaluation.jsonl"),
        )
        self.app_config = SimpleNamespace(websocket=SimpleNamespace(host="127.0.0.1", port=8765))
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def _run(
        self,
        query: str,
        *,
        max_steps: int,
        max_session_steps: int = 0,
        on_before_step=None,
        on_step=None,
    ) -> TaskRunResult:
        del max_steps, max_session_steps
        await self.facade.enqueue_initial_request(query)
        steps: list[StepLogRecord] = []
        while True:
            round_input = await self.facade.next_round()
            if round_input is None:
                break
            if on_before_step is not None:
                on_before_step(len(steps) + 1)
            step_log = await self.facade.run_round(round_input)
            steps.append(step_log)
            if on_step is not None:
                on_step(step_log)
            if step_log.status in {TaskStatus.FINISH, TaskStatus.FAIL}:
                break
        terminal_status = steps[-1].status if steps else None
        return TaskRunResult(steps=steps, terminal_status=terminal_status, hit_step_limit=False)

    async def run_query(self, request: DesktopRunRequest, **kwargs) -> DesktopRunResult:
        result = await self._run(
            request.query,
            max_steps=request.max_steps,
            on_before_step=kwargs.get("on_before_step"),
            on_step=kwargs.get("on_step"),
        )
        return DesktopRunResult(
            session_id=request.session_id,
            terminal_status=result.terminal_status.value if result.terminal_status is not None else None,
            hit_step_limit=result.hit_step_limit,
            primary_result_path=self.artifact_store.step_log_path,
            step_count=len(result.steps),
            step_log_path=self.artifact_store.step_log_path,
            runtime_log_path=self.artifact_store.runtime_log_path,
            evaluation_log_path=self.artifact_store.evaluation_log_path,
        )


def _step(status: TaskStatus, *, request: str = "Open Calculator", session_step: int = 1) -> StepLogRecord:
    return StepLogRecord(
        session_id="sess-1",
        session_step=session_step,
        round_step=session_step,
        round_num=1,
        status=status,
        request=request,
        result="done" if status != TaskStatus.CONTINUE else "",
        llm_trace=LLMTraceRecord(reasoning="I should verify Calculator is open before typing."),
        assistant_text="I will confirm Calculator is open before entering the expression.",
        assistant_tool_calls=[
            AssistantToolCallRecord(
                id="call-1",
                name="get_desktop_app_target_info",
                arguments='{"refresh_app_windows": true}',
            )
        ],
        tool_results=[
            ToolResultArtifact(
                tool_call_id="call-1",
                name="get_desktop_app_target_info",
                content="Calculator window is now visible.",
                status="success",
            )
        ],
    )


def test_terminal_renderer_wraps_long_assistant_content_within_panel_width() -> None:
    renderer = NorthCUATerminalRenderer()
    long_text = "我会先确认 Calculator 已经打开，然后继续向用户汇报每一步进展，避免这一整段说明把终端右边框直接顶开。"
    step = _step(TaskStatus.CONTINUE)
    step.assistant_text = long_text
    rendered = renderer.render_step(step)

    for line in rendered.splitlines():
        assert renderer._visible_len(line) <= renderer.width


class _FakeWebSocketClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.sent_task = None
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def send_task(self, payload) -> None:
        self.sent_task = payload

    async def iter_messages(self):
        yield NorthCUAWebSocketEnvelope.wrap(
            WebSocketMessageType.TASK_FINISHED,
            TaskFinishedPayload(
                session_id=self.sent_task.session_id if self.sent_task is not None else "unknown",
                status=TaskStatus.FINISH,
                result="/tmp/northcua-test/result.json",
                error=None,
            ),
        )

    async def close(self) -> None:
        self.closed = True
