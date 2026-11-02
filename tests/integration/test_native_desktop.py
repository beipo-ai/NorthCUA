"""Opt-in physical input against a disposable window owned by this test."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        os.getenv("NORTHCUA_RUN_NATIVE_TESTS") != "1",
        reason="Requires an interactive test desktop",
    ),
]


def _window(directory: Path):
    import tkinter as tk

    window = tk.Tk()
    window.title("NorthCUA disposable input test")
    window.geometry("420x150+100+150")
    window.attributes("-topmost", True)
    entry = tk.Entry(window, width=35, font=("Arial", 18))
    entry.pack(padx=15, pady=30)

    entry.bind("<Return>", lambda event: (directory / "result.txt").write_text(entry.get()))
    entry.bind("<Button-1>", lambda event: (directory / "clicked").write_text("clicked"))
    window.update()
    window.lift()
    window.focus_force()
    entry.focus_set()

    def ready():
        if sys.platform == "darwin":
            from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication

            # Tk focus alone does not activate a newly launched macOS process.
            # Activate only this disposable fixture before publishing its target.
            NSRunningApplication.currentApplication().activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            window.lift()
            entry.focus_force()
            window.update_idletasks()
        (directory / "ready.json").write_text(
            json.dumps(
                {
                    "x": entry.winfo_rootx() + 50,
                    "y": entry.winfo_rooty() + entry.winfo_height() // 2,
                }
            )
        )

    # Wait until the native window has finished its initial map/geometry update.
    window.after(500, ready)
    window.after(30000, window.destroy)
    window.mainloop()


def _focus_fixture(directory, backend):
    (directory / "clicked").unlink(missing_ok=True)
    point = json.loads((directory / "ready.json").read_text())
    deadline = time.monotonic() + 5
    while not (directory / "clicked").exists():
        assert time.monotonic() < deadline, "The test window did not receive a native click"
        backend.click(point["x"], point["y"])
        time.sleep(0.1)


def test_native_click_type_key_and_capture(tmp_path):
    from northcua.runtime.mcp.desktop_native.backend import PyAutoGuiDesktopBackend

    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(tmp_path)])
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "ready.json").exists():
            assert process.poll() is None, "Test window exited before opening"
            assert time.monotonic() < deadline, "Test window did not open"
            time.sleep(0.1)
        backend = PyAutoGuiDesktopBackend()
        _focus_fixture(tmp_path, backend)
        backend.type_text("NorthCUA physical input 712")
        backend.keypress(key="enter")
        deadline = time.monotonic() + 5
        while not (tmp_path / "result.txt").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert (tmp_path / "result.txt").read_text() == "NorthCUA physical input 712"
        data_url, frame = backend.capture_desktop_screenshot()
        assert data_url.startswith("data:image/png;base64,") and len(data_url) > 1000
        assert frame.width > 0 and frame.height > 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.asyncio
async def test_local_desktop_actor_through_nexau(tmp_path):
    import asyncio
    import yaml
    from northcua.agents.desktop.application import DesktopApplication
    from northcua.agentinfra.runtime.protocol import DesktopRunRequest
    from northcua.agentkernel.working_memory import WorkingMemoryStore
    from tests.integration.test_web_modes import ModelFixture

    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(tmp_path)])
    app = None
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "ready.json").exists():
            assert process.poll() is None and time.monotonic() < deadline
            await asyncio.sleep(0.1)
        from northcua.runtime.mcp.desktop_native.backend import PyAutoGuiDesktopBackend

        with ModelFixture(desktop_text="NorthCUA actor input 913") as fixture:
            config_path = fixture.write_config(tmp_path / "config")
            config = yaml.safe_load(config_path.read_text())
            config["evaluation"]["enabled"] = False
            config["observation"] = {"a11y": {"enabled": False}}
            config["prompt"]["screenshot_window_images"] = 1
            config_path.write_text(yaml.safe_dump(config))
            agent_path = config_path.parent / "agent.yaml"
            agent = yaml.safe_load(agent_path.read_text())
            agent["mcp_servers"] = [
                {"name": "DesktopNativeDataExecutor", "type": "in-memory", "model_visible": False},
                {"name": "DesktopNativeActionExecutor", "type": "in-memory", "model_visible": True},
            ]
            agent_path.write_text(yaml.safe_dump(agent))
            app = DesktopApplication.from_app_config(str(config_path), session_id="原生/actor")
            # Initialize the app before taking input focus for this task.
            await asyncio.to_thread(_focus_fixture, tmp_path, PyAutoGuiDesktopBackend())
            result = await app.run_query(
                DesktopRunRequest(session_id="原生/actor", query="Fill the test window.", max_steps=5)
            )
            assert result.terminal_status == "finish", result
            assert not result.hit_step_limit
            assert fixture.actor_calls == 4
            deadline = time.monotonic() + 5
            while not (tmp_path / "result.txt").exists() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert (tmp_path / "result.txt").read_text() == "NorthCUA actor input 913"
            entries = WorkingMemoryStore(app.artifact_store.root).active_entries(1)
            assert [entry["key"] for entry in entries] == ["plan"]
    finally:
        if app is not None:
            await app.stop()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    _window(Path(sys.argv[1]))
