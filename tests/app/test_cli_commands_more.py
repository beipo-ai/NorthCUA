from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace

from northcua.core.enums import TaskStatus
from northcua_protocol.artifacts import ArtifactKind, ArtifactRef
from northcua_protocol.payloads import TaskAccepted
from northcua_protocol.state import TaskState
from tests.app.test_cli import (
    ANSI_RE,
    _FakeApplication,
    _step,
    _task_row_json,
    build_parser,
    evaluate_artifact_command,
    run_server_command,
    run_task_command,
    submit_task_command,
)


def test_web_evaluate_artifact_command_uses_offline_runner(monkeypatch, tmp_path, capsys) -> None:
    from northcua.agents.web.evaluation.offline import WebOfflineEvaluationRequest, WebOfflineEvaluationResult
    from northcua.cli.web_commands import evaluate_web_artifact_command

    class _FakeRunner:
        async def run(self, request: WebOfflineEvaluationRequest) -> WebOfflineEvaluationResult:
            return WebOfflineEvaluationResult(
                report={"overall_verdict": "pass"},
                artifact_dir=request.artifact_dir,
                round_num=request.round_num,
                final_report_path=tmp_path / "evaluations" / "web_round_evaluator" / "round_0003" / "final_report.json",
                evidence_manifest_path=tmp_path
                / "evaluations"
                / "web_round_evaluator"
                / "round_0003"
                / "evidence_manifest.json",
                llm_trace_path=tmp_path / "evaluations" / "web_round_evaluator" / "round_0003" / "llm_trace.jsonl",
                tool_trace_path=tmp_path / "evaluations" / "web_round_evaluator" / "round_0003" / "tool_trace.jsonl",
                runtime_log_path=tmp_path / "evaluations" / "web_round_evaluator" / "round_0003" / "runtime.jsonl",
                structured_round_dir=tmp_path / "evaluations" / "web_round_evaluator" / "round_0003",
            )

    monkeypatch.setattr("northcua.agents.web.evaluation.offline.WebArtifactEvaluationRunner", _FakeRunner)

    exit_code = asyncio.run(evaluate_web_artifact_command(None, str(tmp_path), 3))

    assert exit_code == 0
    output = ANSI_RE.sub("", capsys.readouterr().out)
    assert "NorthCUA Web Evaluator" in output
    assert "Web Evaluator Summary" in output
    assert "final_report.json" in output


def test_evaluate_artifact_command_uses_runner_and_panels(monkeypatch, tmp_path, capsys) -> None:
    from northcua.agents.desktop.evaluation.events import DesktopEvaluationEvent
    from northcua.agents.desktop.evaluation.offline import OfflineEvaluationRequest, OfflineEvaluationResult

    class _FakeRunner:
        def __init__(self, *, event_callback: Callable[[DesktopEvaluationEvent], None] | None = None) -> None:
            self._event_callback = event_callback

        async def run(self, request: OfflineEvaluationRequest) -> OfflineEvaluationResult:
            if self._event_callback is not None:
                self._event_callback(
                    DesktopEvaluationEvent(
                        name="bundle_built",
                        session_id="sess-1",
                        round_num=request.round_num,
                        payload={
                            "step_count": 2,
                            "screenshot_ref_count": 2,
                            "a11y_ref_count": 1,
                            "runtime_ref_count": 0,
                        },
                    )
                )
            return OfflineEvaluationResult(
                record=None,
                artifact_dir=tmp_path,
                round_num=request.round_num,
                step_count=2,
                final_report_path=tmp_path / "final_report.json",
                legacy_result_path=tmp_path / "eval_round_0001_result.json",
                round_log_path=tmp_path / "round_evaluations.jsonl",
                session_rollup_path=tmp_path / "session_evaluation.json",
                structured_round_dir=tmp_path / "evaluations" / "desktop_round_completion" / "round_0001",
            )

    monkeypatch.setattr(
        "northcua.agents.desktop.evaluation.offline.DesktopArtifactEvaluationRunner",
        _FakeRunner,
    )

    exit_code = asyncio.run(evaluate_artifact_command(None, str(tmp_path), 1, True))

    assert exit_code == 0
    output = ANSI_RE.sub("", capsys.readouterr().out)
    assert "NorthCUA Evaluator" in output
    assert "Evaluator Round 1 [bundle]" in output
    assert "Evaluator Summary" in output
    assert "Final Report" in output


def test_submit_task_command_wait_prints_terminal_summary(monkeypatch, capsys) -> None:
    class _FakeControllerClient:
        def __init__(self, url: str) -> None:
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def submit_task(self, **kwargs):
            return TaskAccepted(task_id="task-1", trace_id="trace-1", session_id="sess-1")

        async def wait_for_finished(self, task_id: str, timeout_seconds: float):
            return SimpleNamespace(
                state=TaskState.FINISHED,
                result="done",
                artifacts=[
                    ArtifactRef.from_inline_bytes(
                        artifact_id="task-1-steps",
                        kind=ArtifactKind.JSON,
                        data=b'{"step":1}\n',
                        mime_type="application/x-ndjson",
                    )
                ],
                error=None,
            )

    monkeypatch.setattr("northcua.cli.desktop_commands.ControllerSubmitterClient", _FakeControllerClient)

    exit_code = asyncio.run(
        submit_task_command(
            None,
            "Open Calculator",
            "sess-1",
            5,
            0,
            controller_url="ws://127.0.0.1:8765",
            controller_token="submitter-token",
            wait_for_controller_result=True,
            controller_wait_timeout=30.0,
            target_device_id="",
            target_capabilities=None,
        )
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "status=finished" in output
    assert "result=done" in output
    assert "artifacts=1" in output


def test_controller_task_row_json_redacts_tokens_and_inline_payloads() -> None:
    row = {
        "task_id": "task-1",
        "trace_id": "trace-1",
        "agent_kind": "desktop",
        "status": "finished",
        "request_json": json.dumps(
            {
                "request": "Open Calculator",
                "token": "submitter-secret",
                "input_artifacts": [
                    {
                        "artifact_id": "artifact-1",
                        "inline_data": "ZmFrZQ==",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        "result_json": json.dumps(
            {
                "token": "result-secret",
                "image_url": "data:image/png;base64,ZmFrZQ==",
            },
            ensure_ascii=False,
        ),
    }

    sanitized = _task_row_json(row)

    assert sanitized["request"]["token"] == "<redacted>"
    assert sanitized["request"]["input_artifacts"][0]["inline_data"].startswith("<base64:")
    assert sanitized["result"]["token"] == "<redacted>"
    assert sanitized["result"]["image_url"].startswith("<data-url:image/png:")


def test_run_server_command_starts_and_stops(monkeypatch) -> None:
    app = _FakeApplication()

    class _FakeEvent:
        def __init__(self) -> None:
            self._set = False

        def set(self) -> None:
            self._set = True

        async def wait(self) -> None:
            self._set = True

    monkeypatch.setattr(
        "northcua.cli.NorthCUAApplication.from_app_config",
        lambda app_config, session_id="", artifact_root_dir_override=None: app,
    )
    monkeypatch.setattr("northcua.cli.desktop_commands.asyncio.Event", _FakeEvent)

    exit_code = asyncio.run(run_server_command("config/app.local.yaml", "sess-1", None))

    assert exit_code == 0
    assert app.started is True
    assert app.stopped is True
