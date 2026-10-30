"""Behavioral contracts for copyable configuration and the scripted fixture."""

import json

import pytest
import yaml

from northcua.cli.config_commands import run_config_command
from northcua.cli.parser import build_parser
from northcua.demo.server import DemoServer, write_configuration
from northcua.demo.server import visible_window_owner
from northcua.core.config_loader import NorthCUAConfigLoader
from northcua.batch.manifest.loader import load_batch_manifest


def test_init_sets_both_models_without_copying_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_MODEL_KEY", "secret-value-not-for-disk")
    args = build_parser().parse_args(
        [
            "config",
            "init",
            "--directory",
            str(tmp_path),
            "--model",
            "vision-model",
            "--base-url",
            "http://127.0.0.1:9988/v1",
            "--api-key-env",
            "MY_MODEL_KEY",
        ]
    )
    assert run_config_command(args) == 0
    for name in ("agent.example.yaml", "agent.web.example.yaml"):
        text = (tmp_path / name).read_text()
        data = yaml.safe_load(text)["llm_config"]
        assert data["model"] == "vision-model"
        assert data["base_url"] == "http://127.0.0.1:9988/v1"
        assert data["api_key"] == "${env.MY_MODEL_KEY}"
        assert "secret-value-not-for-disk" not in text
    for kind in ("web", "desktop"):
        NorthCUAConfigLoader.from_app_config(tmp_path / "app.example.yaml", agent_kind=kind)
    with pytest.raises(SystemExit, match="already exists"):
        run_config_command(args)


def test_demo_configs_load_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NORTHCUA_MODEL", raising=False)
    server = DemoServer(tmp_path, 0)
    try:
        write_configuration(server, 9876)
        for kind in ("web", "desktop"):
            NorthCUAConfigLoader.from_app_config(tmp_path / "app.example.yaml", agent_kind=kind)
        manifest = load_batch_manifest(tmp_path / "batch.yaml")
        assert manifest.defaults.sandbox.provider == "local-process"
        assert len(manifest.tasks) == 3
        assert manifest.defaults.controller_url == "ws://127.0.0.1:9876"
    finally:
        server.server_close()


def test_fixture_concurrent_conversations_do_not_share_turns(tmp_path):
    server = DemoServer(tmp_path, 0)
    request = {"messages": [], "tools": [{"function": {"name": "web_click_point"}}]}
    try:
        first = server.answer(request)
        assert first["tool_calls"][0]["function"]["name"] == "web_note"
        assert server.answer(request)["tool_calls"][0]["function"]["name"] == "web_note"
        next_request = {**request, "messages": [first]}
        assert server.answer(next_request)["tool_calls"][0]["function"]["name"] == "web_click_point"
    finally:
        server.server_close()


def test_evaluator_cannot_pass_without_observed_receipt(tmp_path):
    server = DemoServer(tmp_path, 0)
    request = {"messages": [], "tools": [{"function": {"name": "final_report"}}]}
    try:
        probe = server.answer(request)
        assert probe["tool_calls"][0]["function"]["name"] == "client_browser_snapshot"
        final = server.answer({**request, "messages": [probe, {"role": "tool", "content": "unchanged page"}]})
        assert json.loads(final["tool_calls"][0]["function"]["arguments"])["overall_verdict"] == "failure"
    finally:
        server.server_close()


def test_native_guard_ignores_cursor_but_not_covering_windows():
    point = {"x": 165, "y": 255}
    bounds = {"X": 100, "Y": 150, "Width": 540, "Height": 218}
    form = {"kCGWindowOwnerPID": 123, "kCGWindowBounds": bounds, "kCGWindowLayer": 19}
    cursor = {"kCGWindowOwnerPID": 1, "kCGWindowBounds": bounds, "kCGWindowLayer": 999}
    obstruction = {"kCGWindowOwnerPID": 456, "kCGWindowBounds": bounds, "kCGWindowLayer": 20}
    assert visible_window_owner([cursor, form], point, cursor_layer=999) == 123
    assert visible_window_owner([cursor, obstruction, form], point, cursor_layer=999) == 456


@pytest.mark.parametrize("other_task", ["different", "expected-2", "expected0"])
def test_fixture_evaluation_rejects_another_tasks_live_receipt(tmp_path, other_task):
    server = DemoServer(tmp_path, 0)
    request = {
        "messages": [{"role": "user", "content": "Task ID: expected. Verify the form."}],
        "tools": [{"function": {"name": "final_report"}}],
    }
    try:
        probe = server.answer(request)
        messages = [*request["messages"], probe, {"role": "tool", "content": server.marker + "/" + other_task}]
        final = server.answer({**request, "messages": messages})
        assert json.loads(final["tool_calls"][0]["function"]["arguments"])["overall_verdict"] == "failure"
        messages[-1]["content"] = server.marker + "/expected"
        final = server.answer({**request, "messages": messages})
        assert json.loads(final["tool_calls"][0]["function"]["arguments"])["overall_verdict"] == "success"
    finally:
        server.server_close()
