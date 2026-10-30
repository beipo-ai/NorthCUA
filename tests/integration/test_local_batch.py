"""Execute public local batch commands with real browser workers and resume."""

import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from northcua.demo.server import DemoServer, write_configuration
from northcua.sandboxes.provider.local_process import LocalProcessSandboxProvider
from northcua.sandboxes.contracts import SandboxHandle

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("NORTHCUA_RUN_BROWSER_TESTS") != "1" or os.name != "posix",
        reason="Set NORTHCUA_RUN_BROWSER_TESTS=1 on macOS/Linux with Chromium installed",
    ),
]


@pytest.mark.parametrize("executor_mode", ["in_process", "process"])
def test_cli_local_batch_and_resume(tmp_path, executor_mode):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = DemoServer(tmp_path, 0)
    write_configuration(server, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    controller = None

    def run(name, *args):
        with (tmp_path / f"{name}.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, "-m", "northcua", *args], stdout=log, stderr=subprocess.STDOUT, timeout=180
            )
        assert result.returncode == 0, (tmp_path / f"{name}.log").read_text()[-10000:]

    try:
        with (tmp_path / "controller.log").open("w") as log:
            controller = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "northcua",
                    "controller",
                    "serve",
                    "--app-config",
                    str(tmp_path / "app.batch.yaml"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--executor-mode",
                    executor_mode,
                    "--runner-workers",
                    "2",
                    "--worker-queue-max",
                    "4",
                    "--db-path",
                    str(tmp_path / "controller.sqlite"),
                    "--blob-dir",
                    str(tmp_path / "blobs"),
                    "--artifact-root-dir",
                    str(tmp_path / "controller-artifacts"),
                    "--insecure-dev-no-auth",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        deadline = time.monotonic() + 30
        while True:
            assert controller.poll() is None, (tmp_path / "controller.log").read_text()[-4000:]
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                assert time.monotonic() < deadline, "Controller did not start"
                time.sleep(0.1)
        run(
            "run", "batch", "run", "--manifest", str(tmp_path / "batch.yaml"), "--output-dir", str(tmp_path / "batches")
        )
        path = tmp_path / "batches/local-web/batch_summary.json"
        summary = json.loads(path.read_text())
        assert (summary["total"], summary["succeeded"], summary["failed"]) == (3, 3, 0)
        assert {s["attempts"] for s in summary["task_states"]} == {1}
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert {e["task"] for e in events if e["kind"] == "web_action"} == {"web-1", "web-2", "web-3"}
        assert sum(e["kind"] == "evaluation" and e["verified"] for e in events) == 3
        provider = LocalProcessSandboxProvider()
        for state in summary["task_states"]:
            handle = SandboxHandle(provider="local-process", template="browser", sandbox_id=state["sandbox_id"])
            receipt = json.loads((provider._directory(handle) / "finished.json").read_text())
            assert receipt["cleanup_confirmed"] is True, receipt
            assert receipt["remaining_pids"] == [], receipt
        run("resume", "batch", "resume", "--batch-id", "local-web", "--output-dir", str(tmp_path / "batches"))
        resumed = json.loads(path.read_text())
        assert (resumed["succeeded"], resumed["failed"]) == (3, 0)
        assert {s["attempts"] for s in resumed["task_states"]} == {1}
        assert [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()] == events
    finally:
        if controller is not None:
            controller.terminate()
            try:
                controller.wait(timeout=15)
            except subprocess.TimeoutExpired:
                controller.kill()
                controller.wait(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
