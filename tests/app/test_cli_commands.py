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


def test_parser_supports_grouped_local_submit_controller_and_provider_commands() -> None:
    parser = build_parser()

    task_args = parser.parse_args(["local", "desktop", "run", "--query", "Open Calculator", "--max-session-steps", "9"])
    evaluate_args = parser.parse_args(["local", "desktop", "evaluate", "--artifact-dir", "/tmp/session"])
    web_evaluate_args = parser.parse_args(["local", "web", "evaluate", "--artifact-dir", "/tmp/web-session"])
    serve_args = parser.parse_args(["local", "desktop", "serve"])
    submit_desktop_args = parser.parse_args(
        ["submit", "desktop", "--query", "Open Calculator", "--controller", "ws://controller.example"]
    )
    controller_task_args = parser.parse_args(["controller", "task", "task-123", "--watch"])
    provider_probe_args = parser.parse_args(
        [
            "provider",
            "probe",
            "--model",
            "claude-opus-4-6",
            "--base-url",
            "https://gateway.example.invalid/v1",
            "--api-key",
            "key",
            "--api-type",
            "anthropic_chat_completion",
            "--provider-profile",
            "claude-compatible-gateway",
            "--transcode-images-to-jpeg",
        ]
    )
    assert task_args.command_group == "local"
    assert task_args.agent_kind == "desktop"
    assert task_args.agent_action == "run"
    assert task_args.max_steps == 100
    assert task_args.max_session_steps == 9
    assert evaluate_args.command_group == "local"
    assert evaluate_args.agent_kind == "desktop"
    assert evaluate_args.agent_action == "evaluate"
    assert evaluate_args.artifact_dir == "/tmp/session"
    assert evaluate_args.round_num == 0
    assert web_evaluate_args.command_group == "local"
    assert web_evaluate_args.agent_kind == "web"
    assert web_evaluate_args.agent_action == "evaluate"
    assert web_evaluate_args.artifact_dir == "/tmp/web-session"
    assert web_evaluate_args.round_num == 1
    assert serve_args.command_group == "local"
    assert serve_args.agent_kind == "desktop"
    assert serve_args.agent_action == "serve"
    assert submit_desktop_args.command_group == "submit"
    assert submit_desktop_args.agent_kind == "desktop"
    assert submit_desktop_args.max_steps == 100
    assert controller_task_args.command_group == "controller"
    assert controller_task_args.controller_command == "task"
    assert controller_task_args.task_id == "task-123"
    assert controller_task_args.watch is True
    assert provider_probe_args.command_group == "provider"
    assert provider_probe_args.provider_command == "probe"
    assert provider_probe_args.provider_profile == "claude-compatible-gateway"
    assert provider_probe_args.transcode_images_to_jpeg is True
    assert task_args.app_config is None
    assert serve_args.app_config is None
    assert task_args.log_dir is None
    assert serve_args.log_dir is None


def test_run_task_command_returns_zero_on_finish(monkeypatch, capsys) -> None:
    app = _FakeApplication([_step(TaskStatus.FINISH)])

    monkeypatch.setattr(
        "northcua.cli.NorthCUAApplication.from_app_config",
        lambda app_config, session_id="", artifact_root_dir_override=None: app,
    )

    exit_code = asyncio.run(run_task_command("config/app.local.yaml", "Open Calculator", "sess-1", 5, 0, None))

    assert exit_code == 0
    assert app.facade.enqueued_queries == ["Open Calculator"]
    output = ANSI_RE.sub("", capsys.readouterr().out)
    assert "Trace Dir" in output
    assert "/tmp/northcua-test" in output
    assert "Step Log: /tmp/northcua-test/steps.jsonl" in output
    assert "Runtime Log: /tmp/northcua-test/runtime.jsonl" in output
    assert "Step 1 [finish]" in output
    assert "Step 1" in output
    assert "running..." in output
    assert "Thought" in output
    assert "Assistant" in output
    assert "Tools" in output
    assert "Result" in output
    assert "I should verify Calculator is open before typing." in output
    assert "get_desktop_app_target_info" in output
    assert "Calculator window is now visible." in output
    assert "\n  done\n" not in output
    assert "Artifacts" not in output
    assert "Window   :" not in output
    assert "Clean    :" not in output


def test_run_task_command_returns_one_on_fail(monkeypatch) -> None:
    app = _FakeApplication([_step(TaskStatus.FAIL)])

    monkeypatch.setattr(
        "northcua.cli.NorthCUAApplication.from_app_config",
        lambda app_config, session_id="", artifact_root_dir_override=None: app,
    )

    exit_code = asyncio.run(run_task_command("config/app.local.yaml", "Open Calculator", "sess-1", 5, 0, None))

    assert exit_code == 1


def test_run_task_command_returns_one_on_error(monkeypatch) -> None:
    app = _FakeApplication([_step(TaskStatus.ERROR)])

    monkeypatch.setattr(
        "northcua.cli.NorthCUAApplication.from_app_config",
        lambda app_config, session_id="", artifact_root_dir_override=None: app,
    )

    exit_code = asyncio.run(run_task_command("config/app.local.yaml", "Open Calculator", "sess-1", 5, 0, None))

    assert exit_code == 1


def test_submit_task_command_wait_returns_one_for_failed_state_without_error(monkeypatch) -> None:
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
            return SimpleNamespace(state=TaskState.FAILED, result=None, artifacts=[], error=None)

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

    assert exit_code == 1


def test_desktop_step_limit_is_nonzero_and_always_stops(monkeypatch):
    from unittest.mock import AsyncMock
    from northcua.agentinfra.runtime.protocol import DesktopRunResult

    app = _FakeApplication([])
    app.run_query = AsyncMock(return_value=DesktopRunResult(session_id="limit", hit_step_limit=True))
    monkeypatch.setattr("northcua.cli.NorthCUAApplication.from_app_config", lambda *args, **kwargs: app)
    assert asyncio.run(run_task_command(None, "task", "limit", 1, 0, None)) == 1
    assert app.stopped


def test_desktop_runtime_exception_still_stops(monkeypatch):
    from unittest.mock import AsyncMock
    import pytest

    app = _FakeApplication([])
    app.run_query = AsyncMock(side_effect=RuntimeError("actor broke"))
    monkeypatch.setattr("northcua.cli.NorthCUAApplication.from_app_config", lambda *args, **kwargs: app)
    with pytest.raises(RuntimeError, match="actor broke"):
        asyncio.run(run_task_command(None, "task", "exception", 1, 0, None))
    assert app.stopped
