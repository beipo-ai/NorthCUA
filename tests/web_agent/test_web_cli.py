from __future__ import annotations

import asyncio
from types import SimpleNamespace

from northcua.cli.parser import build_parser, resolve_cli_adapter
from northcua.cli.web_commands import submit_web_command


def test_local_web_run_parser_shape() -> None:
    args = build_parser().parse_args(
        [
            "local",
            "web",
            "run",
            "--query",
            "Find pricing",
            "--url",
            "https://example.com",
            "--headed",
            "--max-steps",
            "3",
            "--max-session-steps",
            "9",
        ]
    )

    assert args.command_group == "local"
    assert args.agent_kind == "web"
    assert args.agent_action == "run"
    assert args.query == "Find pricing"
    assert args.url == "https://example.com"
    assert args.headed is True
    assert args.max_steps == 3
    assert args.max_session_steps == 9
    assert resolve_cli_adapter("web") is not None


def test_submit_web_uses_controller_submitter(monkeypatch, capsys) -> None:
    class _FakeSubmitter:
        def __init__(self, server_url: str) -> None:
            self.server_url = server_url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        async def submit_task(self, **kwargs: object):
            assert self.server_url == "ws://controller"
            assert kwargs["agent_kind"] == "web"
            assert kwargs["request"] == "Find pricing"
            assert kwargs["metadata"] == {
                "session_id": "sess",
                "url": "https://example.com",
                "headed": True,
                "max_steps": 3,
                "max_session_steps": 0,
                "allowed_tool_names": [],
                "tool_choice_policy": "auto",
            }
            return SimpleNamespace(task_id="task", trace_id="trace", session_id="sess")

    monkeypatch.setattr("northcua.cli.web_commands.ControllerSubmitterClient", _FakeSubmitter)
    exit_code = asyncio.run(
        submit_web_command(
            None,
            "Find pricing",
            "https://example.com",
            True,
            3,
            0,
            "sess",
            controller_url="ws://controller",
            controller_token="",
            wait_for_controller_result=False,
            controller_wait_timeout=1.0,
            target_device_id="",
            target_capabilities=None,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status=accepted" in captured.out
