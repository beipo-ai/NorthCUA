"""Real browser + real NexAU execution against a deterministic local model endpoint.

This validates runtime wiring, not hosted-model quality. The evaluator must inspect
page memory that disappears in a replacement browser or after a reload.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

MARKER = "CONFIRMED_MEMORY_ONLY_712"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("NORTHCUA_RUN_BROWSER_TESTS") != "1",
        reason="Set NORTHCUA_RUN_BROWSER_TESTS=1 for real browser tests",
    ),
]


class ModelFixture:
    def __init__(self, desktop_text: str | None = None):
        self.desktop_text = desktop_text
        self.actor_calls = 0
        self.evaluator_calls = 0
        self.live_marker_seen = False
        self.requests = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = (
                    f'<html><title>Isolated task fixture</title><body><button style="position:absolute;'
                    f'left:16px;top:16px;width:180px;height:60px" onclick="this.textContent=\'{MARKER}\'">'
                    "Activate</button></body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.requests.append(request)
                tools = {tool["function"]["name"] for tool in request.get("tools", [])}
                assert request.get("stream") is not True, "Fixture requires non-streaming config"
                if "final_report" in tools:
                    fixture.evaluator_calls += 1
                    if fixture.evaluator_calls == 1:
                        message = fixture.tool_call("live-snapshot", "client_browser_snapshot", {})
                    else:
                        fixture.live_marker_seen = any(
                            m.get("role") == "tool"
                            and m.get("tool_call_id") == "live-snapshot"
                            and MARKER in str(m.get("content"))
                            for m in request["messages"]
                        )
                        verdict = "success" if fixture.live_marker_seen else "failure"
                        message = fixture.tool_call(
                            "report",
                            "final_report",
                            {
                                "overall_verdict": verdict,
                                "process_rating": 5,
                                "outcome_label": verdict,
                                "completion_rating": 5 if fixture.live_marker_seen else 1,
                                "confidence_rating": 5,
                                "controllability": "controllable",
                                "reason": "Checked the live DOM marker.",
                                "page_state_verified": fixture.live_marker_seen,
                                "dom_state_verified": fixture.live_marker_seen,
                            },
                        )
                else:
                    fixture.actor_calls += 1
                    image_count = sum(
                        isinstance(block, dict) and block.get("type") == "image_url"
                        for message in request["messages"]
                        if isinstance(message.get("content"), list)
                        for block in message["content"]
                    )
                    assert image_count <= (1 if fixture.desktop_text else 2), image_count
                    if fixture.actor_calls > 1:
                        assert sum("Active working memory" in str(m.get("content")) for m in request["messages"]) == 1
                    if fixture.desktop_text is not None:
                        assert image_count == 1, "Desktop must retain its current grounding screenshot"
                        if fixture.actor_calls == 1:
                            message = fixture.tool_call(
                                "memory", "note", {"text": "Fill the test window", "key": "plan"}
                            )
                        elif fixture.actor_calls == 2:
                            message = fixture.tool_call("type", "type", {"text": fixture.desktop_text})
                        elif fixture.actor_calls == 3:
                            message = fixture.tool_call("enter", "keypress", {"key": "enter"})
                        else:
                            message = {"role": "assistant", "content": "The test window is filled."}
                    elif fixture.actor_calls == 1:
                        message = fixture.tool_call(
                            "memory", "web_note", {"text": "Activate the button then verify.", "key": "plan"}
                        )
                    elif fixture.actor_calls == 2:
                        message = fixture.tool_call("click", "web_click_point", {"x": 80, "y": 40})
                    else:
                        message = {"role": "assistant", "content": "Activated the button."}
                response = {
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fixture",
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                }
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @staticmethod
    def tool_call(call_id, name, arguments):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            ],
        }

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def write_config(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        agent = {
            "name": "integration-web",
            "max_iterations": 1,
            "llm_config": {
                "model": "fixture",
                "api_type": "openai_chat_completion",
                "api_key": "test-only",
                "base_url": self.url + "/v1",
                "stream": False,
                "max_tokens": 1024,
                "max_context_tokens": 128000,
                "max_retries": 0,
                "timeout": 30,
            },
            "mcp_servers": [
                {"name": "WebAgentProbeExecutor", "type": "in-memory", "model_visible": False},
                {"name": "WebAgentActionExecutor", "type": "in-memory", "model_visible": True},
            ],
        }
        (directory / "agent.yaml").write_text(yaml.safe_dump(agent))
        app = {
            "agent_config_path": "agent.yaml",
            "web_agent_config_path": "agent.yaml",
            "artifacts": {"root_dir": str(directory / "artifacts")},
            "web": {
                "browser_executable_path": os.getenv("NORTHCUA_TEST_BROWSER_EXECUTABLE") or None,
                "max_steps": 4,
                "post_action_wait_ms": 0,
                "browser_channel": os.getenv("NORTHCUA_TEST_BROWSER_CHANNEL") or None,
            },
            "evaluation": {
                "enabled": True,
                "max_steps": 4,
                "force_finalizer_after_steps": 3,
                "fail_open": False,
                "tools": {"allow_client_live_probe": True},
            },
            "prompt": {"screenshot_window_images": 2},
        }
        path = directory / "app.yaml"
        path.write_text(yaml.safe_dump(app))
        return path


@pytest.mark.asyncio
async def test_local_actor_and_evaluator_share_browser_memory(tmp_path):
    from northcua.agents.web import WebAgentApplication
    from northcua.agentinfra.runtime.protocol import WebRunRequest
    from northcua.agentkernel.working_memory import WorkingMemoryStore

    with ModelFixture() as fixture:
        path = fixture.write_config(tmp_path)
        app = WebAgentApplication.from_app_config(str(path), session_id="browser-integration")
        try:
            result = await app.run_query(
                WebRunRequest(
                    session_id="browser-integration",
                    query="Activate the button.",
                    url=fixture.url + "/fixture",
                    max_steps=4,
                )
            )
        finally:
            await app.stop()
        assert result.status == "success", result
        assert fixture.actor_calls == 3
        assert fixture.evaluator_calls >= 2
        assert fixture.live_marker_seen, "Evaluator did not observe the original browser's in-memory mutation"
        payload = json.loads(app.artifact_store.result_path.read_text())
        assert payload["evaluation_report"]["overall_verdict"] == "success"
        assert not payload["cleanup_errors"]
        assert WorkingMemoryStore(app.artifact_store.root).active_entries(1)[0]["key"] == "plan"


@pytest.mark.asyncio
@pytest.mark.parametrize("executor_mode", ["in_process", "process"])
async def test_controller_worker_actor_and_live_evaluator(tmp_path, executor_mode):
    import asyncio
    import socket
    import subprocess
    import sys
    import time
    from northcua.submitter import ControllerSubmitterClient, WebSubmitSpec

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"ws://127.0.0.1:{port}"
    processes = []
    handles = []

    def start(name, arguments):
        handle = (tmp_path / f"{name}.log").open("w")
        handles.append(handle)
        env = dict(os.environ)
        if env.get("NORTHCUA_TEST_BROWSER_EXECUTABLE"):
            env["NORTHCUA_BROWSER_EXECUTABLE"] = env["NORTHCUA_TEST_BROWSER_EXECUTABLE"]
        process = subprocess.Popen([sys.executable, *arguments], stdout=handle, stderr=subprocess.STDOUT, env=env)
        processes.append(process)
        return process

    with ModelFixture() as fixture:
        config_path = fixture.write_config(tmp_path)
        controller = start(
            "controller",
            [
                "-m",
                "northcua",
                "controller",
                "serve",
                "--app-config",
                str(config_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--db-path",
                str(tmp_path / "controller.sqlite"),
                "--blob-dir",
                str(tmp_path / "blobs"),
                "--artifact-root-dir",
                str(tmp_path / "artifacts"),
                "--executor-mode",
                executor_mode,
                "--runner-workers",
                "1",
                "--worker-queue-max",
                "2",
                "--insecure-dev-no-auth",
            ],
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.close()
                    await writer.wait_closed()
                    break
                except OSError:
                    assert controller.poll() is None, (tmp_path / "controller.log").read_text()[-5000:]
                    assert time.monotonic() < deadline, "Controller did not start"
                    await asyncio.sleep(0.1)
            start(
                "device",
                [
                    "-m",
                    "northcua_device.cli",
                    "connect",
                    "--server",
                    url,
                    "--platform",
                    "browser",
                    "--agentkit-agent-kind",
                    "web",
                    "--device-id",
                    "test-browser",
                    "--log-dir",
                    str(tmp_path / "device"),
                    "--insecure-dev-no-auth",
                ],
            )
            spec = WebSubmitSpec(
                query="Activate the button.", url=fixture.url + "/fixture", max_steps=4, target_device_id="test-browser"
            )
            async with ControllerSubmitterClient(url) as client:
                accepted = await client.submit_task(token="", **spec.submit_payload())
                finished = await client.wait_for_finished(accepted.task_id, timeout_seconds=90)
            assert finished.state.value == "finished", finished.model_dump(mode="json")
            assert fixture.actor_calls == 3 and fixture.evaluator_calls >= 2
            assert fixture.live_marker_seen
            assert finished.result["evaluation_report"]["overall_verdict"] == "success"
            assert not finished.result["evaluation_error"]
            reports = list((tmp_path / "artifacts").rglob("working_memory.sqlite3"))
            assert len(reports) == 1
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                try:
                    await asyncio.to_thread(process.wait, timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait, timeout=5)
            for handle in handles:
                handle.close()


@pytest.mark.asyncio
async def test_websocket_serve_actor_and_live_evaluator(tmp_path):
    import asyncio
    import websockets
    from northcua.agents.web import WebAgentApplication
    from northcua.agents.web.server import WebAgentWebSocketServer
    from northcua.core.enums import WebSocketMessageType
    from northcua.core.protocol import NorthCUAWebSocketEnvelope, WebTaskRequestPayload

    with ModelFixture() as fixture:
        app = WebAgentApplication.from_app_config(str(fixture.write_config(tmp_path)), session_id="serve")
        server = WebAgentWebSocketServer(facade=app.facade, host="127.0.0.1", port=0)
        try:
            await server.start()
            port = server._server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}", proxy=None) as socket:
                await socket.send(
                    NorthCUAWebSocketEnvelope.wrap(
                        WebSocketMessageType.TASK_REQUEST,
                        WebTaskRequestPayload(
                            session_id="serve",
                            request="Activate the button.",
                            url=fixture.url + "/fixture",
                            max_steps=4,
                        ),
                    ).model_dump_json()
                )
                accepted = NorthCUAWebSocketEnvelope.model_validate_json(await asyncio.wait_for(socket.recv(), 30))
                finished = NorthCUAWebSocketEnvelope.model_validate_json(await asyncio.wait_for(socket.recv(), 60))
                assert accepted.type == WebSocketMessageType.TASK_ACCEPTED
                assert finished.type == WebSocketMessageType.TASK_FINISHED
                assert finished.payload.status.value == "finish"
            assert fixture.actor_calls == 3 and fixture.live_marker_seen
            result = json.loads(app.artifact_store.result_path.read_text())
            assert result["evaluation_report"]["overall_verdict"] == "success"
        finally:
            await server.stop()
            await app.stop()


def test_local_cli_actor_and_live_evaluator(tmp_path):
    import subprocess
    import sys

    with ModelFixture() as fixture:
        config = fixture.write_config(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "northcua",
                "local",
                "web",
                "run",
                "--app-config",
                str(config),
                "--session-id",
                "cli",
                "--query",
                "Activate the button.",
                "--url",
                fixture.url + "/fixture",
                "--max-steps",
                "4",
                "--evaluate",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
        assert fixture.actor_calls == 3 and fixture.live_marker_seen
        saved = json.loads((tmp_path / "artifacts/cli/result.json").read_text())
        assert saved["evaluation_report"]["overall_verdict"] == "success"
